# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ObjectGaussianMap backend for GaussianFeels with mixed precision and object-centric coordinates.
Implements efficient Gaussian splatting with tactile-visual fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import math

from ..spatial.spatial_hash import get_global_spatial_hash, SpatialHashConfig
from ..memory.memory_planner import MemoryPlanner, MemoryMode


@dataclass
class GaussianParams:
    """Mixed precision Gaussian parameters"""
    # High precision (FP32) for geometry
    positions: torch.Tensor    # [N, 3] FP32 - object-centric coordinates
    rotations: torch.Tensor    # [N, 4] FP32 - quaternions
    scales: torch.Tensor       # [N, 3] FP32 - anisotropic scaling
    
    # Lower precision (FP16) for appearance
    colors: torch.Tensor       # [N, 3] FP16 - RGB values
    opacities: torch.Tensor    # [N, 1] FP16 - alpha values
    
    # Object transformation
    T_obj_world: torch.Tensor  # [4, 4] FP32 - object to world transform
    
    def to(self, device: torch.device):
        """Move to device"""
        return GaussianParams(
            positions=self.positions.to(device),
            rotations=self.rotations.to(device),
            scales=self.scales.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
            T_obj_world=self.T_obj_world.to(device)
        )
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]
    
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB"""
        total_bytes = (
            self.positions.element_size() * self.positions.numel() +
            self.rotations.element_size() * self.rotations.numel() +
            self.scales.element_size() * self.scales.numel() +
            self.colors.element_size() * self.colors.numel() +
            self.opacities.element_size() * self.opacities.numel() +
            self.T_obj_world.element_size() * self.T_obj_world.numel()
        )
        return total_bytes / (1024 ** 2)


class ObjectGaussianMap(nn.Module):
    """Object-centric Gaussian field with mixed precision and spatial indexing"""
    
    def __init__(self,
                 max_gaussians: int = 300000,
                 memory_mode: MemoryMode = MemoryMode.AGGRESSIVE,
                 device: str = "cuda"):
        super().__init__()
        
        self.max_gaussians = max_gaussians
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Memory management
        self.memory_planner = MemoryPlanner(target_gaussians=max_gaussians, mode=memory_mode)
        self.memory_budget = self.memory_planner.calculate_budget()
        
        # Spatial indexing
        hash_config = SpatialHashConfig(
            cell_size=0.005,  # 5mm cells
            max_neighbors=64,
            device=device
        )
        self.spatial_hash = get_global_spatial_hash()
        
        # Initialize empty Gaussian field
        self.gaussians = self._initialize_gaussians()
        self.active_count = 0
        
        # Optimization state
        self.optimizer_state = {}
        self.densification_grad_threshold = 0.005
        self.pruning_opacity_threshold = 0.05
        self.densification_interval = 100  # steps
        self.step_count = 0
        
        # Performance tracking
        self.last_optimization_time = 0.0
        self.last_render_time = 0.0
        
    def _initialize_gaussians(self) -> GaussianParams:
        """Initialize empty Gaussian field with proper precision"""
        # High precision geometry (FP32)
        positions = torch.zeros(self.max_gaussians, 3, dtype=torch.float32, device=self.device)
        rotations = torch.zeros(self.max_gaussians, 4, dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.0  # Initialize as identity quaternions
        scales = torch.ones(self.max_gaussians, 3, dtype=torch.float32, device=self.device) * 0.01
        
        # Lower precision appearance (FP16)
        colors = torch.zeros(self.max_gaussians, 3, dtype=torch.float16, device=self.device)
        opacities = torch.zeros(self.max_gaussians, 1, dtype=torch.float16, device=self.device)
        
        # Object-centric coordinate system (identity initially)
        T_obj_world = torch.eye(4, dtype=torch.float32, device=self.device)
        
        return GaussianParams(positions, rotations, scales, colors, opacities, T_obj_world)
    
    def add_gaussians(self, 
                     positions: torch.Tensor,
                     colors: Optional[torch.Tensor] = None,
                     scales: Optional[torch.Tensor] = None,
                     opacities: Optional[torch.Tensor] = None) -> int:
        """Add new Gaussians to the field"""
        n_new = positions.shape[0]
        
        if self.active_count + n_new > self.max_gaussians:
            # Need to prune or reject
            available = self.max_gaussians - self.active_count
            if available > 0:
                n_new = available
                positions = positions[:n_new]
                if colors is not None:
                    colors = colors[:n_new]
            else:
                raise RuntimeError(f"No space available for {n_new} new Gaussians. Maximum capacity: {self.max_gaussians}")
        
        # Set default values
        if colors is None:
            colors = torch.ones(n_new, 3, dtype=torch.float16, device=self.device) * 0.5
        if scales is None:
            scales = torch.ones(n_new, 3, dtype=torch.float32, device=self.device) * 0.01
        if opacities is None:
            opacities = torch.ones(n_new, 1, dtype=torch.float16, device=self.device) * 0.1
        
        # Add to field
        start_idx = self.active_count
        end_idx = start_idx + n_new
        
        self.gaussians.positions[start_idx:end_idx] = positions.to(torch.float32)
        self.gaussians.colors[start_idx:end_idx] = colors.to(torch.float16)
        self.gaussians.scales[start_idx:end_idx] = scales.to(torch.float32)
        self.gaussians.opacities[start_idx:end_idx] = opacities.to(torch.float16)
        
        # Initialize rotations as identity
        self.gaussians.rotations[start_idx:end_idx] = 0
        self.gaussians.rotations[start_idx:end_idx, 0] = 1.0
        
        # Update spatial hash
        point_ids = torch.arange(start_idx, end_idx, device=self.device)
        self.spatial_hash.insert_points(positions, point_ids)
        
        self.active_count += n_new
        return n_new
    
    def get_active_gaussians(self) -> GaussianParams:
        """Get currently active Gaussians"""
        if self.active_count == 0:
            return self._initialize_gaussians()
            
        return GaussianParams(
            positions=self.gaussians.positions[:self.active_count],
            rotations=self.gaussians.rotations[:self.active_count],
            scales=self.gaussians.scales[:self.active_count],
            colors=self.gaussians.colors[:self.active_count],
            opacities=self.gaussians.opacities[:self.active_count],
            T_obj_world=self.gaussians.T_obj_world.clone()
        )
    
    def transform_to_world(self, positions_obj: torch.Tensor) -> torch.Tensor:
        """Transform object coordinates to world coordinates"""
        # Convert to homogeneous coordinates
        ones = torch.ones(positions_obj.shape[0], 1, device=positions_obj.device, dtype=positions_obj.dtype)
        pos_homo = torch.cat([positions_obj, ones], dim=1)  # [N, 4]
        
        # Apply transformation
        pos_world_homo = (self.gaussians.T_obj_world @ pos_homo.T).T  # [N, 4]
        
        return pos_world_homo[:, :3]  # Return [N, 3]
    
    def transform_to_object(self, positions_world: torch.Tensor) -> torch.Tensor:
        """Transform world coordinates to object coordinates"""
        # Invert transformation
        T_world_obj = torch.inverse(self.gaussians.T_obj_world)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(positions_world.shape[0], 1, device=positions_world.device, dtype=positions_world.dtype)
        pos_homo = torch.cat([positions_world, ones], dim=1)  # [N, 4]
        
        # Apply inverse transformation
        pos_obj_homo = (T_world_obj @ pos_homo.T).T  # [N, 4]
        
        return pos_obj_homo[:, :3]  # Return [N, 3]
    
    def update_object_pose(self, T_obj_world: torch.Tensor):
        """Update object pose in world coordinates"""
        self.gaussians.T_obj_world = T_obj_world.to(self.device)
        
        # Update spatial hash with world coordinates
        if self.active_count > 0:
            world_positions = self.transform_to_world(self.gaussians.positions[:self.active_count])
            self.spatial_hash.clear()
            point_ids = torch.arange(self.active_count, device=self.device)
            self.spatial_hash.insert_points(world_positions, point_ids)
    
    def query_neighbors(self, position_world: torch.Tensor, radius: float) -> torch.Tensor:
        """Query Gaussian neighbors around world position"""
        return self.spatial_hash.query_neighbors(position_world, radius)
    
    def densification_step(self, gradients: torch.Tensor):
        """Perform adaptive densification based on gradients"""
        if self.step_count % self.densification_interval != 0:
            return
        
        # Calculate gradient magnitudes
        grad_norms = torch.norm(gradients[:self.active_count], dim=1)
        
        # Find Gaussians with high gradients
        high_grad_mask = grad_norms > self.densification_grad_threshold
        high_grad_indices = torch.where(high_grad_mask)[0]
        
        if len(high_grad_indices) == 0:
            return
        
        # Split large Gaussians with high gradients
        large_scale_mask = torch.max(self.gaussians.scales[:self.active_count], dim=1)[0] > 0.02
        split_candidates = high_grad_indices[large_scale_mask[high_grad_indices]]
        
        if len(split_candidates) > 0:
            self._split_gaussians(split_candidates[:min(len(split_candidates), 1000)])  # Limit splits
        
        # Clone small Gaussians with high gradients
        small_scale_mask = torch.max(self.gaussians.scales[:self.active_count], dim=1)[0] <= 0.02
        clone_candidates = high_grad_indices[small_scale_mask[high_grad_indices]]
        
        if len(clone_candidates) > 0:
            self._clone_gaussians(clone_candidates[:min(len(clone_candidates), 1000)])  # Limit clones
    
    def _split_gaussians(self, indices: torch.Tensor):
        """Split Gaussians by reducing scale and adding offset copies"""
        n_splits = len(indices)
        space_available = self.max_gaussians - self.active_count
        n_splits = min(n_splits, space_available)
        
        if n_splits == 0:
            return
        
        indices = indices[:n_splits]
        
        # Reduce scale of original Gaussians
        self.gaussians.scales[indices] *= 0.8
        
        # Create offset copies
        start_idx = self.active_count
        end_idx = start_idx + n_splits
        
        # Copy original parameters
        self.gaussians.positions[start_idx:end_idx] = self.gaussians.positions[indices]
        self.gaussians.rotations[start_idx:end_idx] = self.gaussians.rotations[indices]
        self.gaussians.scales[start_idx:end_idx] = self.gaussians.scales[indices]
        self.gaussians.colors[start_idx:end_idx] = self.gaussians.colors[indices]
        self.gaussians.opacities[start_idx:end_idx] = self.gaussians.opacities[indices]
        
        # Add random offset
        offset = torch.randn(n_splits, 3, device=self.device) * 0.01
        self.gaussians.positions[start_idx:end_idx] += offset
        
        self.active_count += n_splits
        
        # Update spatial hash
        world_positions = self.transform_to_world(self.gaussians.positions[start_idx:end_idx])
        point_ids = torch.arange(start_idx, end_idx, device=self.device)
        self.spatial_hash.insert_points(world_positions, point_ids)
    
    def _clone_gaussians(self, indices: torch.Tensor):
        """Clone Gaussians with small random offset"""
        n_clones = len(indices)
        space_available = self.max_gaussians - self.active_count
        n_clones = min(n_clones, space_available)
        
        if n_clones == 0:
            return
        
        indices = indices[:n_clones]
        
        start_idx = self.active_count
        end_idx = start_idx + n_clones
        
        # Copy parameters
        self.gaussians.positions[start_idx:end_idx] = self.gaussians.positions[indices]
        self.gaussians.rotations[start_idx:end_idx] = self.gaussians.rotations[indices]
        self.gaussians.scales[start_idx:end_idx] = self.gaussians.scales[indices]
        self.gaussians.colors[start_idx:end_idx] = self.gaussians.colors[indices]
        self.gaussians.opacities[start_idx:end_idx] = self.gaussians.opacities[indices]
        
        # Add small random offset
        offset = torch.randn(n_clones, 3, device=self.device) * 0.005
        self.gaussians.positions[start_idx:end_idx] += offset
        
        self.active_count += n_clones
        
        # Update spatial hash
        world_positions = self.transform_to_world(self.gaussians.positions[start_idx:end_idx])
        point_ids = torch.arange(start_idx, end_idx, device=self.device)
        self.spatial_hash.insert_points(world_positions, point_ids)
    
    def pruning_step(self):
        """Remove Gaussians with low opacity"""
        if self.active_count == 0:
            return
        
        # Find Gaussians to keep
        opacities_fp32 = self.gaussians.opacities[:self.active_count].float()
        keep_mask = opacities_fp32.squeeze() > self.pruning_opacity_threshold
        keep_indices = torch.where(keep_mask)[0]
        
        n_keep = len(keep_indices)
        if n_keep == self.active_count:
            return  # Nothing to prune
        
        # Compact the arrays
        if n_keep > 0:
            self.gaussians.positions[:n_keep] = self.gaussians.positions[keep_indices]
            self.gaussians.rotations[:n_keep] = self.gaussians.rotations[keep_indices]
            self.gaussians.scales[:n_keep] = self.gaussians.scales[keep_indices]
            self.gaussians.colors[:n_keep] = self.gaussians.colors[keep_indices]
            self.gaussians.opacities[:n_keep] = self.gaussians.opacities[keep_indices]
        
        self.active_count = n_keep
        
        # Rebuild spatial hash
        if self.active_count > 0:
            world_positions = self.transform_to_world(self.gaussians.positions[:self.active_count])
            self.spatial_hash.clear()
            point_ids = torch.arange(self.active_count, device=self.device)
            self.spatial_hash.insert_points(world_positions, point_ids)
    
    def forward(self, camera_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Render Gaussians using the production rasterizer (gsplat/DGR if available)."""
        import time
        from ..render.rasterizer import GaussianRasterizer, RenderConfig, CameraParams
        start_time = time.time()

        H = int(camera_params.get('height', 480))
        W = int(camera_params.get('width', 640))
        if self.active_count == 0:
            return {
                'image': torch.zeros(H, W, 3, device=self.device),
                'depth': torch.zeros(H, W, 1, device=self.device),
                'alpha': torch.zeros(H, W, 1, device=self.device),
                'num_gaussians': 0,
            }

        # Build rasterizer and camera
        rconfig = RenderConfig(image_height=H, image_width=W, sh_degree=3, backend='auto')
        rasterizer = GaussianRasterizer(rconfig).to(self.device)

        K = camera_params['K']
        fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
        cam = CameraParams(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
                           near=float(camera_params.get('near', 0.01)),
                           far=float(camera_params.get('far', 10.0)))
        T_WC = camera_params['T_WC']  # 4x4 world->camera

        # Prepare gaussian parameters in rasterizer format
        act = self.get_active_gaussians()
        params: Dict[str, torch.Tensor] = {
            'positions': act.positions,                   # [N,3] fp32
            'rotations': act.rotations,                   # [N,4] fp32
            'scales': act.scales,                         # [N,3] fp32 (log-scales expected by our renderer)
            'opacity': act.opacities.float(),             # [N,1] fp16->fp32
        }
        # Map colors to SH DC with contiguous per-channel layout
        # For degree d, DC indices are [0, (d+1)^2, 2*(d+1)^2]
        colors01 = act.colors.float().clamp(0, 1)
        dc = torch.logit(colors01.clamp(1e-4, 1 - 1e-4))  # [N,3]
        num = (rconfig.sh_degree + 1) ** 2
        # Interleaved layout for gsplat compatibility: DC at [:3] -> [R0,G0,B0]
        sh = torch.zeros((dc.shape[0], num * 3), device=self.device, dtype=dc.dtype)
        sh[:, 0:3] = dc  # [R_dc, G_dc, B_dc]
        params['sh_coeffs'] = sh

        out = rasterizer(params, cam, T_WC)

        self.last_render_time = time.time() - start_time
        return {
            'image': out['rgb'],
            'depth': out['depth'],
            'alpha': None,
            'num_gaussians': self.active_count,
        }

    def get_gaussian_parameters(self) -> Dict[str, torch.Tensor]:
        """Return active gaussian parameters as dict for training/fusion."""
        act = self.get_active_gaussians()
        # Build SH DC from colors by default
        colors01 = act.colors.float().clamp(0, 1)
        dc = torch.logit(colors01.clamp(1e-4, 1 - 1e-4))
        deg = 3
        sh = torch.zeros((dc.shape[0], (deg + 1) ** 2 * 3), device=self.device, dtype=dc.dtype)
        sh[:, :3] = dc
        return {
            'positions': act.positions,
            'rotations': act.rotations,
            'scales': act.scales,
            'opacity': act.opacities.float(),
            'sh_coeffs': sh,
        }
    
    def optimize_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Perform optimization step with densification and pruning"""
        import time
        start_time = time.time()
        
        # Backward pass
        loss.backward()
        
        # Get gradients for densification
        pos_grad = self.gaussians.positions.grad
        if pos_grad is not None and self.active_count > 0:
            self.densification_step(pos_grad[:self.active_count])
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Periodic pruning
        if self.step_count % (self.densification_interval * 2) == 0:
            self.pruning_step()
        
        self.step_count += 1
        self.last_optimization_time = time.time() - start_time
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        active_gaussians = self.get_active_gaussians()
        
        return {
            'active_gaussians': self.active_count,
            'max_gaussians': self.max_gaussians,
            'memory_usage_mb': active_gaussians.memory_usage_mb(),
            'memory_budget_mb': self.memory_budget.total / (1024**2),
            'utilization': self.active_count / self.max_gaussians,
            'spatial_hash_stats': self.spatial_hash.get_statistics()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'last_render_time_ms': self.last_render_time * 1000,
            'last_optimization_time_ms': self.last_optimization_time * 1000,
            'densification_threshold': self.densification_grad_threshold,
            'pruning_threshold': self.pruning_opacity_threshold,
            'step_count': self.step_count
        }


if __name__ == "__main__":
    # Test Gaussian field
    print("Testing ObjectGaussianMap...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussian_field = ObjectGaussianMap(max_gaussians=10000, device=device)
    
    # Add some test Gaussians
    positions = torch.randn(100, 3) * 0.1  # 10cm spread
    colors = torch.rand(100, 3)
    n_added = gaussian_field.add_gaussians(positions, colors)
    print(f"Added {n_added} Gaussians")
    
    # Test rendering
    camera_params = {
        'T_WC': torch.eye(4),
        'K': torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32),
        'height': 480,
        'width': 640
    }
    
    render_output = gaussian_field(camera_params)
    print(f"Rendered image shape: {render_output['image'].shape}")
    
    # Test neighbor queries
    query_pos = torch.zeros(3)
    neighbors = gaussian_field.query_neighbors(query_pos, radius=0.05)
    print(f"Found {len(neighbors)} neighbors within 5cm")
    
    # Memory stats
    mem_stats = gaussian_field.get_memory_stats()
    print(f"Memory stats: {mem_stats}")