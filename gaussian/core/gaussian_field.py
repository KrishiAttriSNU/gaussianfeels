"""
ObjectGaussianMap with full precision storage.
Full FP32 for all parameters: positions/rotations/scales/opacity/SH coefficients.
No artificial Gaussian count limits.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class GaussianConfig:
    """Configuration for Gaussian field parameters (Industry Standards: Maximum Capacity)"""
    max_gaussians: int = None  # Unbounded for maximum capacity (industry standard)
    sh_degree: int = 3  # 48 SH coefficients (Gaussian Splatting standard)
    position_lr: float = 0.00016  # Original Gaussian Splatting learning rate
    rotation_lr: float = 0.001    # Industry standard
    scale_lr: float = 0.005      # Industry standard
    opacity_lr: float = 0.05     # Industry standard
    sh_lr: float = 0.0025       # Industry standard
    use_mixed_precision: bool = False  # Full FP32 for maximum quality
    densify_threshold: float = 0.0002  # Stricter than 0.01 for higher quality
    prune_threshold: float = 0.005     # Industry standard

class ObjectGaussianMap(nn.Module):
    """
    Object-centric Gaussian Splatting representation with full precision.
    
    Memory layout (Full FP32):
    - FP32: positions [N,3], rotations [N,4], scales [N,3], opacity [N,1], sh_coeffs [N,48]
    """
    
    def __init__(self, config: GaussianConfig):
        super().__init__()
        self.config = config
        self.max_gaussians = config.max_gaussians or float('inf')  # Unbounded
        self.sh_degree = config.sh_degree
        self.sh_coeffs_count = (config.sh_degree + 1) ** 2 * 3  # 48 for degree 3
        
        # Initialize empty tensors - will be allocated on first use
        # All tensors FP32
        self._positions = None  # [N, 3] FP32
        self._rotations = None  # [N, 4] FP32 (quaternions)
        self._scales = None     # [N, 3] FP32 (log scale)
        self._opacity = None    # [N, 1] FP32
        self._sh_coeffs = None  # [N, 48] FP32
        
        # Tracking
        self._num_gaussians = 0
        self._active_mask = None
        self._device = None
        
        # Spatial indexing for neighbor queries
        self._spatial_hash = None
        
    def initialize_gaussians(self, positions: torch.Tensor, device: str = 'cuda', colors: Optional[torch.Tensor] = None):
        """Initialize Gaussians from point cloud positions and optional RGB colors.
        If colors provided (Nx3 in [0,1] or [0,255]), initialize SH DC from colors.
        """
        self._device = device
        n_points = positions.shape[0]
        
        # Full precision parameters
        self._positions = positions[:n_points].clone().to(device, dtype=torch.float32)
        self._rotations = torch.zeros(n_points, 4, device=device, dtype=torch.float32)
        self._rotations[:, 0] = 1.0  # Initialize as identity quaternions
        
        # Initialize scales based on local point density with numerical stability
        distances = self._compute_local_density(self._positions)
        # Use a reasonable scale factor that won't cause tiny log values
        # Scale factor 0.5 instead of 0.1 to avoid extremely small scales
        scale_values = distances.unsqueeze(-1).repeat(1, 3) * 0.5
        
        # CRITICAL: Clamp scale values before taking log to prevent NaN/extreme values
        min_scale = 0.001  # 1mm minimum scale -> log(0.001) ≈ -6.9
        max_scale = 0.02   # 2cm maximum scale -> log(0.02) ≈ -3.9  
        scale_values = torch.clamp(scale_values, min=min_scale, max=max_scale)
        
        self._scales = torch.log(scale_values)
        
        # Full precision appearance parameters
        self._opacity = torch.ones(n_points, 1, device=device, dtype=torch.float32) * 0.1
        self._sh_coeffs = torch.zeros(n_points, self.sh_coeffs_count, device=device, dtype=torch.float32)

        # Initialize SH DC from provided colors if available; else small random
        if colors is not None:
            col = colors.to(device=device, dtype=torch.float32)
            if col.max() > 1.0:
                col = col / 255.0
            col = col.clamp(1e-4, 1 - 1e-4)
            dc = torch.logit(col)
            self._sh_coeffs[:dc.shape[0], :3] = dc
        else:
            self._sh_coeffs[:, :3] = torch.randn(n_points, 3, device=device, dtype=torch.float32) * 0.01
        
        self._num_gaussians = n_points
        self._active_mask = torch.ones(n_points, device=device, dtype=torch.bool)
        
        # Update spatial hash for neighbor queries
        self._update_spatial_hash()
        
    def _compute_local_density(self, positions: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Compute local point density for scale initialization with memory-efficient processing"""
        with torch.no_grad():
            n_points = positions.shape[0]
            
            if n_points < k:
                return torch.ones(n_points, device=positions.device) * 0.01
            
            # Memory-efficient processing for large point clouds
            memory_threshold = 5000  # 5k points threshold for memory optimization (reduced for GPU constraints)
            
            if n_points <= memory_threshold:
                # Small point cloud - use direct computation
                dists = torch.cdist(positions, positions)
                dists.fill_diagonal_(float('inf'))
                knn_dists, _ = torch.topk(dists, k, dim=-1, largest=False)
                avg_dists = knn_dists.mean(dim=-1)
            else:
                # Large point cloud - use memory-efficient approach
                print(f"⚙️ Large point cloud detected ({n_points:,} points). Using memory-efficient density estimation...")
                
                # Try sklearn approach first (most memory efficient)
                try:
                    from sklearn.neighbors import NearestNeighbors
                    import numpy as np
                    
                    # Convert to numpy for sklearn
                    positions_np = positions.cpu().numpy()
                    
                    # Use sklearn NearestNeighbors (memory efficient)
                    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
                    nbrs.fit(positions_np)
                    distances_np, indices = nbrs.kneighbors(positions_np)
                    
                    # Exclude self-distance (first column) and compute mean
                    avg_dists = torch.from_numpy(distances_np[:, 1:].mean(axis=1)).to(positions.device, dtype=positions.dtype)
                    print(f"   ✅ Used sklearn NearestNeighbors for efficient k-NN computation")
                    
                except ImportError:
                    raise ImportError("scikit-learn is required for KNN-based density computation - no fallback algorithm allowed. Install with: pip install scikit-learn")
            
            # CRITICAL: Clamp to prevent numerical instability in log(scale) computation
            min_dist = 0.001  # 1mm minimum
            max_dist = 0.1    # 10cm maximum
            avg_dists = torch.clamp(avg_dists, min=min_dist, max=max_dist)
            
            return avg_dists
    
    @property 
    def positions(self) -> torch.Tensor:
        """Get active Gaussian positions [N_active, 3]"""
        if self._positions is None:
            return torch.empty(0, 3)
        return self._positions[self._active_mask]
    
    @property
    def rotations(self) -> torch.Tensor:
        """Get active Gaussian rotations [N_active, 4]"""
        if self._rotations is None:
            return torch.empty(0, 4)
        return self._rotations[self._active_mask]
    
    @property 
    def scales(self) -> torch.Tensor:
        """Get active Gaussian scales [N_active, 3]"""
        if self._scales is None:
            return torch.empty(0, 3)
        return self._scales[self._active_mask]
    
    @property
    def opacity(self) -> torch.Tensor:
        """Get active Gaussian opacity [N_active, 1]"""
        if self._opacity is None:
            return torch.empty(0, 1)
        return self._opacity[self._active_mask]
    
    @property
    def sh_coeffs(self) -> torch.Tensor:
        """Get active Gaussian SH coefficients [N_active, 48]"""
        if self._sh_coeffs is None:
            return torch.empty(0, self.sh_coeffs_count)
        return self._sh_coeffs[self._active_mask]
    
    @property
    def num_active_gaussians(self) -> int:
        """Get number of active Gaussians"""
        if self._active_mask is None:
            raise RuntimeError("Active mask not initialized. Call initialize() first.")
        return self._active_mask.sum().item()
    
    def get_gaussian_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all active Gaussian parameters for rendering"""
        return {
            'positions': self.positions,
            'rotations': self.rotations, 
            'scales': self.scales,
            'opacity': self.opacity,
            'sh_coeffs': self.sh_coeffs
        }
    
    def get_covariance(self) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scaling and rotation parameters.
        
        Returns:
            Covariance matrices [N_active, 3, 3]
        """
        if self._scales is None or self._rotations is None:
            return torch.empty(0, 3, 3, device=self._device)
        
        # Get active scales and rotations
        scales = self.scales  # [N_active, 3] - log scales
        rotations = self.rotations  # [N_active, 4] - quaternions
        
        # Convert log scales to actual scales
        scale_values = torch.exp(scales)  # [N_active, 3]
        
        n_gaussians = scales.shape[0]
        covariance_matrices = torch.zeros(n_gaussians, 3, 3, device=scales.device, dtype=scales.dtype)
        
        for i in range(n_gaussians):
            # Create scaling matrix S
            S = torch.diag(scale_values[i])  # [3, 3]
            
            # Convert quaternion to rotation matrix
            q = rotations[i]  # [4] - quaternion [w, x, y, z]
            R = self._quaternion_to_rotation_matrix(q)  # [3, 3]
            
            # Covariance = R * S * S^T * R^T
            covariance_matrices[i] = R @ S @ S.T @ R.T
        
        return covariance_matrices
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: Quaternion [4] as [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Normalize quaternion
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=0),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=0),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=0)
        ], dim=0).to(device=q.device, dtype=q.dtype)
        
        return R
    
    def add_gaussians(self, positions: torch.Tensor, rotations: Optional[torch.Tensor] = None,
                     scales: Optional[torch.Tensor] = None) -> bool:
        """Add new Gaussians to the field"""
        # No limits
        n_new = positions.shape[0]
            
        # Expand tensors if needed
        if self._positions is None:
            self.initialize_gaussians(positions[:n_new], self._device or 'cuda')
            return True
            
        # Add to existing tensors
        new_positions = positions[:n_new].to(self._device, dtype=torch.float32)
        
        if rotations is None:
            new_rotations = torch.zeros(n_new, 4, device=self._device, dtype=torch.float32)
            new_rotations[:, 0] = 1.0
        else:
            new_rotations = rotations[:n_new].to(self._device, dtype=torch.float32)
            
        if scales is None:
            distances = self._compute_local_density(new_positions)
            # Use a reasonable scale factor that won't cause tiny log values
            # Scale factor 0.5 instead of 0.1 to avoid extremely small scales
            scale_values = distances.unsqueeze(-1).repeat(1, 3) * 0.5
            
            # CRITICAL: Clamp scale values before taking log to prevent NaN/extreme values
            min_scale = 0.001  # 1mm minimum scale -> log(0.001) ≈ -6.9
            max_scale = 0.02   # 2cm maximum scale -> log(0.02) ≈ -3.9  
            scale_values = torch.clamp(scale_values, min=min_scale, max=max_scale)
            new_scales = torch.log(scale_values)
        else:
            new_scales = scales[:n_new].to(self._device, dtype=torch.float32)
            
        # FP32 parameters
        new_opacity = torch.ones(n_new, 1, device=self._device, dtype=torch.float32) * 0.1
        new_sh_coeffs = torch.zeros(n_new, self.sh_coeffs_count, device=self._device, dtype=torch.float32)
        new_sh_coeffs[:, :3] = torch.randn(n_new, 3, device=self._device, dtype=torch.float32) * 0.01
        
        # Concatenate with existing
        self._positions = torch.cat([self._positions, new_positions], dim=0)
        self._rotations = torch.cat([self._rotations, new_rotations], dim=0)
        self._scales = torch.cat([self._scales, new_scales], dim=0)
        self._opacity = torch.cat([self._opacity, new_opacity], dim=0)
        self._sh_coeffs = torch.cat([self._sh_coeffs, new_sh_coeffs], dim=0)
        
        # Update active mask
        new_mask = torch.ones(n_new, device=self._device, dtype=torch.bool)
        self._active_mask = torch.cat([self._active_mask, new_mask], dim=0)
        
        self._num_gaussians += n_new
        return True
    
    def prune_gaussians(self, mask: torch.Tensor):
        """Remove Gaussians based on pruning mask"""
        if self._active_mask is None:
            return
            
        # Update active mask
        active_indices = torch.where(self._active_mask)[0]
        self._active_mask[active_indices[mask]] = False
    
    def compact_gaussians(self):
        """Remove inactive Gaussians to free memory"""
        if self._active_mask is None or self._active_mask.all():
            return
            
        active_indices = self._active_mask
        
        self._positions = self._positions[active_indices].contiguous()
        self._rotations = self._rotations[active_indices].contiguous()
        self._scales = self._scales[active_indices].contiguous()
        self._opacity = self._opacity[active_indices].contiguous()
        self._sh_coeffs = self._sh_coeffs[active_indices].contiguous()
        
        self._num_gaussians = active_indices.sum().item()
        self._active_mask = torch.ones(self._num_gaussians, device=self._device, dtype=torch.bool)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB"""
        if self._positions is None:
            return {'total_mb': 0.0}
            
        # FP32 parameters
        pos_mb = self._positions.numel() * 4 / 1024 / 1024  # FP32
        rot_mb = self._rotations.numel() * 4 / 1024 / 1024  # FP32
        scale_mb = self._scales.numel() * 4 / 1024 / 1024   # FP32
        opacity_mb = self._opacity.numel() * 4 / 1024 / 1024  # FP32 (changed from FP16)
        sh_mb = self._sh_coeffs.numel() * 4 / 1024 / 1024     # FP32 (changed from FP16)
        mask_mb = self._active_mask.numel() / 8 / 1024 / 1024  # bool
        
        return {
            'positions_mb': pos_mb,
            'rotations_mb': rot_mb,
            'scales_mb': scale_mb,
            'opacity_mb': opacity_mb,
            'sh_coeffs_mb': sh_mb,
            'active_mask_mb': mask_mb,
            'total_mb': pos_mb + rot_mb + scale_mb + opacity_mb + sh_mb + mask_mb,
            'num_gaussians': self._num_gaussians,
            'num_active': self.num_active_gaussians
        }
    
    def _update_spatial_hash(self):
        """Update spatial hash for efficient neighbor queries"""
        if self._positions is None or self._positions.numel() == 0:
            return
        
        from ..spatial.spatial_hash import CUDASpatialHash, SpatialHashConfig
        
        try:
            # Initialize spatial hash if needed
            if self._spatial_hash is None:
                config = SpatialHashConfig(cell_size=0.01, device=self._device or 'cuda')
                self._spatial_hash = CUDASpatialHash(config)
            
            # Clear and rebuild with current positions
            self._spatial_hash.clear()
            active_positions = self.positions
            if active_positions.shape[0] > 0:
                # Create explicit point IDs that map directly to active indices [0, num_active-1]
                active_point_ids = torch.arange(
                    active_positions.shape[0], 
                    dtype=torch.int32, 
                    device=self._device or 'cuda'
                )
                self._spatial_hash.insert_points(active_positions, active_point_ids)
        except Exception as e:
            print(f"⚠️ Spatial hash update failed: {e}")
            self._spatial_hash = None
    
    def query_neighbors(self, query_position: torch.Tensor, radius: float = 0.01) -> torch.Tensor:
        """Query neighboring Gaussians within given radius"""
        # Note: spatial hash should be updated by caller, not on every query
        
        if self._spatial_hash is None:
            # Fallback: return empty tensor if spatial hash unavailable
            return torch.empty(0, dtype=torch.long, device=self._device or 'cuda')
        
        try:
            # Query spatial hash - indices returned are in the current active space
            raw_neighbors = self._spatial_hash.query_neighbors(query_position, radius)
            
            # Safety check: ensure indices are within valid range for active Gaussians
            num_active = self.num_active_gaussians
            if len(raw_neighbors) > 0:
                # Filter out any invalid indices (should not happen, but safety first)
                valid_mask = (raw_neighbors >= 0) & (raw_neighbors < num_active)
                raw_neighbors = raw_neighbors[valid_mask]
            
            return raw_neighbors
        except Exception as e:
            print(f"⚠️ Neighbor query failed: {e}")
            return torch.empty(0, dtype=torch.long, device=self._device or 'cuda')