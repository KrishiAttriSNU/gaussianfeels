"""
Gaussian Splatting Surface Constraints
Implements efficient surface constraint methods that are native to Gaussian representations
Replaces SDF-based approaches with Mahalanobis distance and Gaussian-native constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .gaussian_field import ObjectGaussianMap


@dataclass
class SurfaceConstraintConfig:
    """Configuration for Gaussian surface constraints"""
    # Mahalanobis distance parameters
    mahalanobis_k_neighbors: int = 8  # Number of nearest Gaussians to consider
    mahalanobis_weight_decay: float = 0.1  # Weight decay with distance
    mahalanobis_max_distance: float = 0.05  # Maximum distance to consider (5cm)
    
    # Surface fitting parameters
    surface_flatness_weight: float = 1.0  # Encourage flat Gaussians
    surface_alignment_weight: float = 0.5  # Encourage aligned Gaussians
    surface_density_weight: float = 0.2  # Encourage appropriate density
    
    # Contact constraint parameters
    contact_penetration_penalty: float = 10.0  # Strong penalty for penetration
    contact_surface_tolerance: float = 0.002  # 2mm surface tolerance
    
    # Performance parameters
    enable_spatial_hashing: bool = False  # Use O(n) queries (winner from comparison test)
    batch_processing: bool = True  # Process constraints in batches
    device: str = "cuda"


class MahalanobisDistanceConstraint(nn.Module):
    """
    Mahalanobis distance-based surface constraints for tactile contact points
    This is the recommended SDF alternative for Gaussian Splatting
    """
    
    def __init__(self, gaussian_map: ObjectGaussianMap, config: SurfaceConstraintConfig):
        super().__init__()
        self.gaussian_map = gaussian_map
        self.config = config
        from shared.utils.device_utils import setup_device
        self.device = setup_device(config)
        
        # Spatial hash for fast neighbor queries (if available)
        self.spatial_hash = None
        if config.enable_spatial_hashing:
            from ..utils.spatial_hash import get_global_spatial_hash
            self.spatial_hash = get_global_spatial_hash()
    
    def _find_k_nearest_gaussians(self, query_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest Gaussians for each query point
        
        Args:
            query_points: [N, 3] query points
            
        Returns:
            nearest_indices: [N, k] indices of nearest Gaussians
            distances: [N, k] distances to nearest Gaussians
        """
        gaussian_positions = self.gaussian_map.positions  # [M, 3]
        
        if len(gaussian_positions) == 0:
            raise RuntimeError("CRITICAL: No Gaussians available for neighbor search")
        
        # Compute pairwise distances [N, M]
        distances = torch.cdist(query_points, gaussian_positions)
        
        # Find k nearest
        k = min(self.config.mahalanobis_k_neighbors, len(gaussian_positions))
        nearest_distances, nearest_indices = torch.topk(distances, k, dim=1, largest=False)
        
        return nearest_indices, nearest_distances
    
    def _compute_mahalanobis_distance(self, points: torch.Tensor, gaussian_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from points to specified Gaussians
        
        Args:
            points: [N, 3] points to query
            gaussian_indices: [N, k] indices of Gaussians to compute distance to
            
        Returns:
            mahal_distances: [N, k] Mahalanobis distances
        """
        batch_size, k = gaussian_indices.shape
        
        # Ensure consistent dtype (Float32) throughout computation
        points = points.float()  # Convert to float32 if needed
        
        # Validate input points for NaN/Inf
        if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
            # This should not happen - input points should be valid
            raise RuntimeError("Invalid input points contain NaN/Inf values. This indicates corrupted point cloud.")
        
        # Get Gaussian parameters and ensure float32 dtype
        positions = self.gaussian_map.positions[gaussian_indices].float()  # [N, k, 3]
        rotations = self.gaussian_map.rotations[gaussian_indices].float()  # [N, k, 4]
        scales = self.gaussian_map.scales[gaussian_indices].float()  # [N, k, 3]
        
        # Validate Gaussian parameters
        if torch.any(torch.isnan(positions)) or torch.any(torch.isnan(rotations)) or torch.any(torch.isnan(scales)):
            # This should not happen - Gaussian field should be valid
            raise RuntimeError("Invalid Gaussian parameters contain NaN values. This indicates corrupted Gaussian field.")
        
        # Clamp scales to reasonable range to prevent numerical issues
        scales = torch.clamp(scales, min=-5.0, max=5.0)  # exp(-5) = 0.007mm, exp(5) = 148mm
        
        # Convert quaternions to rotation matrices with robust normalization
        q_norm = torch.norm(rotations, dim=-1, keepdim=True)
        q_norm = torch.clamp(q_norm, min=1e-8)  # Prevent division by zero
        q = rotations / q_norm  # Normalize quaternions
        
        # Ensure quaternion components are valid
        if torch.any(torch.isnan(q)):
            raise RuntimeError("CRITICAL: Quaternion normalization produced NaN values")
        
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Rotation matrix elements with numerical stability
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrices = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
        ], dim=-2)  # [N, k, 3, 3]
        
        # Validate rotation matrices
        if torch.any(torch.isnan(rotation_matrices)):
            raise RuntimeError("CRITICAL: Rotation matrix computation produced NaN values")
        
        # Create covariance matrices from scales and rotations  
        # scales are log-σ -> σ
        log_sigma = scales  # [N, k, 3] log-σ
        sigma = torch.exp(log_sigma).clamp_(min=1e-4, max=1.0)
        covariances = torch.matmul(torch.matmul(rotation_matrices, torch.diag_embed(sigma * sigma)), 
                                 rotation_matrices.transpose(-1, -2))  # [N, k, 3, 3]
        
        # Compute Mahalanobis distance
        points_expanded = points.unsqueeze(1).expand(-1, k, -1)  # [N, k, 3]
        diff = points_expanded - positions  # [N, k, 3]
        
        # jittered Cholesky for stability
        jitter = 1e-6 * torch.eye(3, device=covariances.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        regularized_cov = covariances + jitter
        
        # Try CUDA Cholesky, fallback to CPU if CUDA fails
        try:
            L = torch.linalg.cholesky(regularized_cov)
        except RuntimeError as e:
            if "cusolver" in str(e):
                # Move to CPU, compute, then back to GPU
                regularized_cov_cpu = regularized_cov.cpu()
                L_cpu = torch.linalg.cholesky(regularized_cov_cpu)
                L = L_cpu.to(regularized_cov.device)
            else:
                raise
        inv_cov = torch.cholesky_inverse(L)
        
        # Mahalanobis distance computation
        quad = torch.matmul(torch.matmul(diff.unsqueeze(-2), inv_cov), diff.unsqueeze(-1))  # [N, k, 1, 1]
        quad = quad.squeeze(-1).squeeze(-1)  # [N, k]
        
        # Handle numerical issues
        quad = torch.nan_to_num(quad, nan=0.0, posinf=1e6, neginf=0.0)
        quad = quad.clamp_max(1e6)
        mahal_distances = torch.sqrt(quad.clamp_min(0.0))
        
        return mahal_distances
    
    def compute_surface_distance(self, contact_points: torch.Tensor) -> torch.Tensor:
        """
        Compute surface distance for contact points using Mahalanobis distance
        
        Args:
            contact_points: [N, 3] contact points in object frame
            
        Returns:
            surface_distances: [N] distance to surface (should be ~0 for surface contact)
        """
        if len(contact_points) == 0:
            return torch.empty(0, device=self.device)
        
        # Find nearest Gaussians
        nearest_indices, euclidean_distances = self._find_k_nearest_gaussians(contact_points)
        
        if nearest_indices.shape[1] == 0:
            raise RuntimeError("CRITICAL: No Gaussians available for surface distance computation")
        
        # Filter by maximum distance
        valid_mask = euclidean_distances < self.config.mahalanobis_max_distance
        
        # Compute Mahalanobis distances
        mahal_distances = self._compute_mahalanobis_distance(contact_points, nearest_indices)
        
        # Replace the old weighted+inf code with a finite masked-min:
        large = torch.full_like(mahal_distances, 1e3)
        masked_mahal = torch.where(valid_mask, mahal_distances, large)
        surface_distances = masked_mahal.min(dim=1).values
        surface_distances = torch.nan_to_num(surface_distances, nan=1e3, posinf=1e3, neginf=1e3)
        
        return surface_distances
    
    def compute_contact_loss(self, contact_points: torch.Tensor, 
                           contact_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contact loss for tactile points (they should be on the surface)
        
        Args:
            contact_points: [N, 3] contact points in object frame
            contact_weights: [N] optional weights for each contact point
            
        Returns:
            contact_loss: scalar loss value
        """
        if len(contact_points) == 0:
            return torch.tensor(0.0, device=self.device)
        
        surface_distances = self.compute_surface_distance(contact_points)
        surface_distances = torch.nan_to_num(surface_distances, nan=1e3, posinf=1e3, neginf=1e3)
        
        # Contact points should be close to surface
        contact_residuals = torch.clamp(surface_distances - self.config.contact_surface_tolerance, min=0.0)
        
        # Apply weights if provided
        if contact_weights is not None:
            contact_residuals = contact_residuals * contact_weights
        
        # Penalize penetration more strongly
        penetration_mask = surface_distances < 0
        contact_residuals = torch.where(
            penetration_mask,
            contact_residuals * self.config.contact_penetration_penalty,
            contact_residuals
        )
        
        return torch.mean(contact_residuals ** 2)


class TactileDepthResidual(nn.Module):
    """Per-contact tactile residual that penalizes deviation from Gaussian surface.

    Approximates per-pixel tactile depth by encouraging contact points to lie
    on the nearest Gaussian ellipsoid surface using Mahalanobis-like distance.
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device

    def forward(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        contact_points: torch.Tensor,   # [K, 3]
        contact_normals: torch.Tensor,  # [K, 3]
        contact_confidence: torch.Tensor  # [K]
    ) -> torch.Tensor:
        if contact_points.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        mu = gaussian_params['positions']  # [N,3]
        s = torch.exp(gaussian_params['scales'])  # [N,3]
        # Nearest Gaussian by Euclidean distance
        dists = torch.cdist(contact_points, mu)
        nn_idx = torch.argmin(dists, dim=1)
        mu_k = mu[nn_idx]
        s_k = s[nn_idx]
        # Scaled-space norm should be ≈ 1 on surface
        delta = (contact_points - mu_k) / (s_k + 1e-6)
        m_norm = torch.norm(delta, dim=-1)
        residual = (m_norm - 1.0) ** 2
        w = torch.clamp(contact_confidence, 0.0, 1.0)
        return (residual * w).mean()


class GaussianSurfaceRegularizer(nn.Module):
    """
    Regularize Gaussians to form better surfaces
    Encourages anisotropic, flat Gaussians that align with implicit surfaces
    """
    
    def __init__(self, gaussian_map: ObjectGaussianMap, config: SurfaceConstraintConfig):
        super().__init__()
        self.gaussian_map = gaussian_map
        self.config = config
        from shared.utils.device_utils import setup_device
        self.device = setup_device(config)
    
    def compute_flatness_loss(self) -> torch.Tensor:
        """
        Encourage Gaussians to be flat (anisotropic)
        Penalize spherical Gaussians, encourage disc-like shapes
        """
        scales = self.gaussian_map.scales  # [N, 3] log scales
        
        if len(scales) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Convert log scales to actual scales
        actual_scales = torch.exp(scales)  # [N, 3]
        
        # Sort scales to get largest and smallest
        sorted_scales, _ = torch.sort(actual_scales, dim=-1)  # [N, 3]
        smallest = sorted_scales[:, 0]   # Smallest scale
        middle = sorted_scales[:, 1]     # Middle scale  
        largest = sorted_scales[:, 2]    # Largest scale
        
        # Encourage flat shapes: large aspect ratios
        # Good surface Gaussians should have two large scales and one small scale
        aspect_ratio = largest / (smallest + 1e-8)
        flatness_reward = torch.log(aspect_ratio + 1e-8)  # Log to prevent explosion
        
        # Also encourage the middle scale to be similar to the largest (disc-like)
        disc_similarity = 1.0 / (torch.abs(largest - middle) + 1e-3)
        
        # Combine rewards (negative loss)
        flatness_loss = -torch.mean(flatness_reward + 0.1 * disc_similarity)
        
        return flatness_loss
    
    def compute_alignment_loss(self) -> torch.Tensor:
        """
        Encourage nearby Gaussians to be aligned (similar orientations)
        """
        positions = self.gaussian_map.positions  # [N, 3]
        rotations = self.gaussian_map.rotations  # [N, 4]
        
        if len(positions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Find pairs of nearby Gaussians
        distances = torch.cdist(positions, positions)  # [N, N]
        
        # Only consider pairs within a certain distance
        max_align_distance = 0.02  # 2cm
        nearby_mask = (distances < max_align_distance) & (distances > 0)
        
        if not nearby_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # Get pairs of nearby Gaussians
        i_indices, j_indices = torch.where(nearby_mask)
        
        if len(i_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Compute orientation differences
        qi = rotations[i_indices]  # [M, 4]
        qj = rotations[j_indices]  # [M, 4]
        
        # Quaternion dot product for similarity
        # |q1 · q2| should be close to 1 for similar orientations
        dot_products = torch.sum(qi * qj, dim=-1)  # [M]
        orientation_similarity = torch.abs(dot_products)
        
        # Loss is 1 - similarity (want similarity close to 1)
        alignment_loss = torch.mean(1.0 - orientation_similarity)
        
        return alignment_loss
    
    def compute_density_loss(self) -> torch.Tensor:
        """
        Regularize density of Gaussians for good surface coverage
        """
        positions = self.gaussian_map.positions  # [N, 3]
        opacity = self.gaussian_map.opacity  # [N, 1]
        
        if len(positions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute local density using k-nearest neighbors
        k = min(8, len(positions) - 1)
        distances = torch.cdist(positions, positions)  # [N, N]
        
        # Get k+1 nearest (including self)
        knn_distances, _ = torch.topk(distances, k+1, dim=1, largest=False)
        
        # Average distance to k nearest neighbors (excluding self)
        avg_neighbor_distance = torch.mean(knn_distances[:, 1:], dim=1)  # [N]
        
        # Target density: neither too sparse nor too dense
        target_distance = 0.01  # 1cm average spacing
        density_error = torch.abs(avg_neighbor_distance - target_distance)
        
        # Weight by opacity (more important for visible Gaussians)
        weighted_density_error = density_error * opacity.squeeze(-1)
        
        return torch.mean(weighted_density_error)
    
    def compute_total_regularization_loss(self) -> torch.Tensor:
        """Compute total surface regularization loss"""
        flatness_loss = self.compute_flatness_loss()
        alignment_loss = self.compute_alignment_loss()  
        density_loss = self.compute_density_loss()
        
        total_loss = (self.config.surface_flatness_weight * flatness_loss + 
                     self.config.surface_alignment_weight * alignment_loss +
                     self.config.surface_density_weight * density_loss)
        
        return total_loss


class GaussianSurfaceConstraints(nn.Module):
    """
    Combined surface constraint system for Gaussian Splatting
    This is the main interface replacing SDF-based approaches
    """
    
    def __init__(self, gaussian_map: ObjectGaussianMap, config: Optional[SurfaceConstraintConfig] = None):
        super().__init__()
        
        self.config = config or SurfaceConstraintConfig()
        self.gaussian_map = gaussian_map
        
        # Initialize constraint modules
        self.mahalanobis_constraint = MahalanobisDistanceConstraint(gaussian_map, self.config)
        self.surface_regularizer = GaussianSurfaceRegularizer(gaussian_map, self.config)
    
    def compute_tactile_constraint_loss(self, contact_points: torch.Tensor, 
                                      contact_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss for tactile contact constraints
        This is the main SDF replacement for pose optimization
        """
        return self.mahalanobis_constraint.compute_contact_loss(contact_points, contact_weights)
    
    def compute_surface_regularization_loss(self) -> torch.Tensor:
        """Compute surface quality regularization loss"""
        return self.surface_regularizer.compute_total_regularization_loss()
    
    def query_surface_distance(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Query distance to surface (SDF-like interface)
        
        Args:
            query_points: [N, 3] points to query
            
        Returns:
            distances: [N] distances to surface
        """
        return self.mahalanobis_constraint.compute_surface_distance(query_points)
    
    def update_gaussian_map(self, gaussian_map: ObjectGaussianMap):
        """Update reference to Gaussian map"""
        self.gaussian_map = gaussian_map
        self.mahalanobis_constraint.gaussian_map = gaussian_map
        self.surface_regularizer.gaussian_map = gaussian_map


# Factory functions
def create_surface_constraints(gaussian_map: ObjectGaussianMap, 
                             config: Optional[SurfaceConstraintConfig] = None) -> GaussianSurfaceConstraints:
    """Create surface constraints for Gaussian Splatting"""
    return GaussianSurfaceConstraints(gaussian_map, config)


def create_tactile_constraint_config(precision_level: str = "high") -> SurfaceConstraintConfig:
    """
    Create configuration optimized for tactile sensing
    
    Args:
        precision_level: "high", "medium", or "low" precision requirements
    """
    if precision_level == "high":
        return SurfaceConstraintConfig(
            mahalanobis_k_neighbors=12,
            contact_surface_tolerance=0.001,  # 1mm
            contact_penetration_penalty=20.0,
            surface_flatness_weight=2.0
        )
    elif precision_level == "medium":
        return SurfaceConstraintConfig(
            mahalanobis_k_neighbors=8,
            contact_surface_tolerance=0.002,  # 2mm
            contact_penetration_penalty=10.0,
            surface_flatness_weight=1.0
        )
    else:  # low
        return SurfaceConstraintConfig(
            mahalanobis_k_neighbors=4,
            contact_surface_tolerance=0.005,  # 5mm
            contact_penetration_penalty=5.0,
            surface_flatness_weight=0.5
        )
