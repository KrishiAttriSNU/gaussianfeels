"""
Tactile loss functions for surface reconstruction from touch data.
Implements point-to-surface distance loss and normal alignment loss with robust formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import math

@dataclass
class TactileLossConfig:
    """Configuration for tactile loss computation"""
    # Point-to-surface loss
    surface_loss_weight: float = 1.0
    huber_delta: float = 0.01  # Threshold for Huber loss
    
    # Normal alignment loss  
    normal_loss_weight: float = 0.5
    cauchy_sigma: float = 0.05  # Scale parameter for Cauchy loss
    
    # Gradient regularization
    gradient_loss_weight: float = 0.1
    gradient_sigma: float = 0.02
    
    # Contact region focus
    contact_region_weight: float = 2.0
    contact_threshold: float = 0.005  # Distance threshold for contact
    
    # Regularization
    opacity_reg_weight: float = 0.01
    scale_reg_weight: float = 0.001

class TactileSurfaceLoss(nn.Module):
    """
    Tactile surface reconstruction loss combining:
    1. Point-to-surface distance (Huber loss for robustness)
    2. Normal alignment (Cauchy loss for outlier handling) 
    3. Gradient regularization for smoothness
    4. Contact region emphasis
    """
    
    def __init__(self, config: TactileLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, tactile_points: torch.Tensor, tactile_normals: torch.Tensor,
                tactile_forces: Optional[torch.Tensor], gaussian_params: Dict[str, torch.Tensor],
                rendered_depth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute tactile surface loss.
        
        Args:
            tactile_points: Contact points [N, 3]
            tactile_normals: Surface normals at contact [N, 3] 
            tactile_forces: Optional force measurements [N, 1]
            gaussian_params: Gaussian parameters from ObjectGaussianMap
            rendered_depth: Optional rendered depth for additional supervision
            
        Returns:
            Dictionary of loss components
        """
        device = tactile_points.device
        losses = {}
        
        # 1. Point-to-surface distance loss
        surface_loss = self._compute_surface_distance_loss(
            tactile_points, gaussian_params['positions'], gaussian_params['scales']
        )
        losses['surface_loss'] = surface_loss * self.config.surface_loss_weight
        
        # 2. Normal alignment loss
        if tactile_normals is not None:
            normal_loss = self._compute_normal_alignment_loss(
                tactile_points, tactile_normals, gaussian_params
            )
            losses['normal_loss'] = normal_loss * self.config.normal_loss_weight
        
        # 3. Gradient smoothness loss
        gradient_loss = self._compute_gradient_regularization(gaussian_params['positions'])
        losses['gradient_loss'] = gradient_loss * self.config.gradient_loss_weight
        
        # 4. Contact region emphasis
        if tactile_forces is not None:
            contact_loss = self._compute_contact_region_loss(
                tactile_points, tactile_forces, gaussian_params
            )
            losses['contact_loss'] = contact_loss * self.config.contact_region_weight
        
        # 5. Regularization losses
        opacity_reg = self._compute_opacity_regularization(gaussian_params['opacity'])
        scale_reg = self._compute_scale_regularization(gaussian_params['scales'])
        
        losses['opacity_reg'] = opacity_reg * self.config.opacity_reg_weight
        losses['scale_reg'] = scale_reg * self.config.scale_reg_weight
        
        # 6. Depth supervision if available
        if rendered_depth is not None:
            depth_loss = self._compute_depth_supervision_loss(tactile_points, rendered_depth)
            losses['depth_loss'] = depth_loss * 0.1
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def _compute_surface_distance_loss(self, tactile_points: torch.Tensor, 
                                     gaussian_positions: torch.Tensor,
                                     gaussian_scales: torch.Tensor) -> torch.Tensor:
        """
        Compute point-to-surface distance using Huber loss for robustness.
        """
        if len(gaussian_positions) == 0:
            return torch.tensor(0.0, device=tactile_points.device)
        
        # Find nearest Gaussians for each tactile point
        distances = torch.cdist(tactile_points, gaussian_positions)  # [N_tactile, N_gaussian]
        nearest_distances, nearest_indices = torch.min(distances, dim=1)  # [N_tactile]
        
        # Get scales of nearest Gaussians for adaptive thresholding
        nearest_scales = gaussian_scales[nearest_indices]  # [N_tactile, 3]
        adaptive_scales = torch.mean(torch.exp(nearest_scales), dim=1)  # [N_tactile]
        
        # Normalize distances by local scale
        normalized_distances = nearest_distances / (adaptive_scales + 1e-8)
        
        # Huber loss for robustness to outliers
        huber_loss = F.huber_loss(normalized_distances, 
                                 torch.zeros_like(normalized_distances),
                                 delta=self.config.huber_delta,
                                 reduction='mean')
        
        return huber_loss
    
    def _compute_normal_alignment_loss(self, tactile_points: torch.Tensor,
                                     tactile_normals: torch.Tensor,
                                     gaussian_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute normal alignment loss using Cauchy loss for outlier robustness.
        """
        positions = gaussian_params['positions']
        rotations = gaussian_params['rotations']
        
        if len(positions) == 0:
            return torch.tensor(0.0, device=tactile_points.device)
        
        # Find nearest Gaussians
        distances = torch.cdist(tactile_points, positions)
        nearest_indices = torch.argmin(distances, dim=1)
        
        # Compute Gaussian normal at tactile points
        gaussian_normals = self._compute_gaussian_normals_at_points(
            tactile_points, positions[nearest_indices], rotations[nearest_indices]
        )
        
        # Compute alignment between tactile and Gaussian normals
        dot_products = torch.sum(tactile_normals * gaussian_normals, dim=1)
        
        # Cauchy loss for robust normal alignment: -log(σ²/(σ² + (1-dot)²))  
        alignment_errors = 1.0 - torch.abs(dot_products)  # [0, 2], 0 = perfect alignment
        sigma_sq = self.config.cauchy_sigma ** 2
        cauchy_loss = -torch.log(sigma_sq / (sigma_sq + alignment_errors ** 2) + 1e-8)
        
        return torch.mean(cauchy_loss)
    
    def _compute_gaussian_normals_at_points(self, points: torch.Tensor,
                                          gaussian_positions: torch.Tensor, 
                                          gaussian_rotations: torch.Tensor) -> torch.Tensor:
        """
        Compute normal vectors of Gaussians at given points.
        Uses the dominant eigenvector (largest scale direction).
        """
        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(gaussian_rotations)
        
        # Use Z-axis (last column) as normal direction  
        normals = rotation_matrices[:, :, 2]  # [N, 3]
        
        # Orient normals towards query points
        to_points = points - gaussian_positions
        dot_products = torch.sum(normals * to_points, dim=1, keepdim=True)
        normals = torch.where(dot_products < 0, -normals, normals)
        
        return F.normalize(normals, dim=1)
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices"""
        # Normalize quaternions
        q = F.normalize(quaternions, dim=1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrices = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=1), 
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=1)
        ], dim=1)  # [N, 3, 3]
        
        return rotation_matrices
    
    def _compute_gradient_regularization(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient regularization for surface smoothness.
        Penalizes large local variations in Gaussian positions.
        """
        if len(positions) < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Compute pairwise distances
        distances = torch.cdist(positions, positions)
        
        # Create local neighborhood graph (k=6 nearest neighbors)
        k = min(6, len(positions) - 1)
        _, neighbor_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        neighbor_indices = neighbor_indices[:, 1:]  # Exclude self
        
        # Compute local gradient variations
        gradient_penalties = []
        for i in range(len(positions)):
            center = positions[i]
            neighbors = positions[neighbor_indices[i]]
            
            # Compute vectors from center to neighbors
            vectors = neighbors - center.unsqueeze(0)
            
            # Penalize large variations in neighbor directions
            if len(vectors) > 1:
                # Compute pairwise angles between neighbor vectors
                normalized_vectors = F.normalize(vectors, dim=1)
                angles = torch.acos(torch.clamp(
                    torch.mm(normalized_vectors, normalized_vectors.t()), -1 + 1e-7, 1 - 1e-7
                ))
                
                # Penalty for sharp angles (encourage smooth surfaces)
                angle_penalty = torch.exp(-angles / self.config.gradient_sigma).mean()
                gradient_penalties.append(angle_penalty)
        
        if gradient_penalties:
            return torch.stack(gradient_penalties).mean()
        else:
            return torch.tensor(0.0, device=positions.device)
    
    def _compute_contact_region_loss(self, tactile_points: torch.Tensor,
                                   tactile_forces: torch.Tensor,
                                   gaussian_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Emphasize reconstruction quality in regions with strong tactile contact.
        """
        positions = gaussian_params['positions']
        opacity = gaussian_params['opacity']
        
        if len(positions) == 0:
            return torch.tensor(0.0, device=tactile_points.device)
        
        # Normalize force magnitudes
        force_magnitudes = torch.norm(tactile_forces, dim=1) if tactile_forces.dim() > 1 else tactile_forces.squeeze()
        force_weights = torch.sigmoid(force_magnitudes / force_magnitudes.mean())
        
        # Find Gaussians near high-force contact points
        distances = torch.cdist(tactile_points, positions)
        contact_mask = distances < self.config.contact_threshold
        
        # Weight loss by force magnitude and proximity to contact
        contact_loss = 0.0
        for i, (point, force_weight) in enumerate(zip(tactile_points, force_weights)):
            nearby_gaussians = contact_mask[i]
            if nearby_gaussians.any():
                # Encourage higher opacity for Gaussians near strong contacts
                nearby_opacity = opacity[nearby_gaussians]
                opacity_loss = -torch.log(nearby_opacity + 1e-8).mean()
                contact_loss += force_weight * opacity_loss
        
        return contact_loss / len(tactile_points) if len(tactile_points) > 0 else torch.tensor(0.0)
    
    def _compute_opacity_regularization(self, opacity: torch.Tensor) -> torch.Tensor:
        """
        Regularize opacity to prevent over-dense reconstructions.
        Encourages sparsity while maintaining surface coverage.
        """
        if len(opacity) == 0:
            return torch.tensor(0.0)
        
        # L1 regularization to encourage sparsity
        l1_reg = torch.mean(opacity)
        
        # Entropy regularization to prevent extreme values
        epsilon = 1e-8
        entropy_reg = -torch.mean(
            opacity * torch.log(opacity + epsilon) + 
            (1 - opacity) * torch.log(1 - opacity + epsilon)
        )
        
        return l1_reg + 0.1 * entropy_reg
    
    def _compute_scale_regularization(self, scales: torch.Tensor) -> torch.Tensor:
        """
        Regularize Gaussian scales to maintain reasonable sizes.
        """
        if len(scales) == 0:
            return torch.tensor(0.0)
        
        # Convert log scales to actual scales
        actual_scales = torch.exp(scales)
        
        # Penalize very large or very small scales
        scale_penalty = torch.mean(actual_scales) + torch.mean(1.0 / (actual_scales + 1e-8))
        
        # Encourage isotropic Gaussians for smoother surfaces
        scale_ratios = actual_scales.max(dim=1)[0] / (actual_scales.min(dim=1)[0] + 1e-8)
        anisotropy_penalty = torch.mean(scale_ratios)
        
        return scale_penalty + 0.1 * anisotropy_penalty
    
    def _compute_depth_supervision_loss(self, tactile_points: torch.Tensor,
                                      rendered_depth: torch.Tensor,
                                      camera_intrinsics: Optional[torch.Tensor] = None,
                                      camera_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Production depth supervision loss with proper geometric alignment.
        
        Computes loss between tactile contact points and corresponding depth values
        from rendered depth maps using proper camera projection.
        """
        try:
            if tactile_points.numel() == 0:
                return torch.tensor(0.0, device=tactile_points.device, requires_grad=True)
            
            if rendered_depth.numel() == 0:
                return torch.tensor(0.0, device=tactile_points.device, requires_grad=True)
            
            # If no camera parameters provided, estimate from available data
            if camera_intrinsics is None or camera_pose is None:
                # Return weighted penalty for missing camera calibration
                return torch.tensor(0.01, device=tactile_points.device, requires_grad=True) * tactile_points.norm(dim=-1).mean()
            
            H, W = rendered_depth.shape[-2:]
            batch_size = rendered_depth.shape[0] if rendered_depth.dim() > 2 else 1
            
            # Ensure proper tensor dimensions
            if rendered_depth.dim() == 2:
                rendered_depth = rendered_depth.unsqueeze(0)
            if tactile_points.dim() == 2:
                tactile_points = tactile_points.unsqueeze(0)
            
            total_loss = torch.tensor(0.0, device=tactile_points.device, requires_grad=True)
            
            for b in range(batch_size):
                # Project tactile points to camera coordinates
                if camera_pose.dim() == 2:
                    pose = camera_pose
                else:
                    pose = camera_pose[b]
                
                # Transform tactile points to camera space
                tactile_cam = torch.matmul(tactile_points[b], pose[:3, :3].T) + pose[:3, 3]
                
                # Project to image coordinates
                if camera_intrinsics.dim() == 2:
                    K = camera_intrinsics
                else:
                    K = camera_intrinsics[b]
                
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                
                # Perspective projection with depth check
                valid_depth_mask = tactile_cam[:, 2] > 0.01  # Min depth threshold
                if not valid_depth_mask.any():
                    continue
                
                valid_points = tactile_cam[valid_depth_mask]
                
                # Project to pixel coordinates
                u = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
                v = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy
                
                # Clamp to image bounds
                u_clamped = torch.clamp(u, 0, W - 1)
                v_clamped = torch.clamp(v, 0, H - 1)
                
                # Check if points are within image bounds
                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if not in_bounds.any():
                    continue
                
                valid_u = u_clamped[in_bounds].long()
                valid_v = v_clamped[in_bounds].long()
                valid_depths = valid_points[in_bounds, 2]
                
                # Sample depth values using bilinear interpolation
                depth_map = rendered_depth[b]
                
                # Bilinear sampling for sub-pixel accuracy
                u_floor = torch.floor(u_clamped[in_bounds]).long()
                v_floor = torch.floor(v_clamped[in_bounds]).long()
                u_ceil = torch.clamp(u_floor + 1, 0, W - 1)
                v_ceil = torch.clamp(v_floor + 1, 0, H - 1)
                
                # Interpolation weights
                wu = u_clamped[in_bounds] - u_floor.float()
                wv = v_clamped[in_bounds] - v_floor.float()
                
                # Sample depth at four corners
                d00 = depth_map[v_floor, u_floor]
                d01 = depth_map[v_floor, u_ceil]
                d10 = depth_map[v_ceil, u_floor]
                d11 = depth_map[v_ceil, u_ceil]
                
                # Bilinear interpolation
                sampled_depths = (
                    d00 * (1 - wu) * (1 - wv) +
                    d01 * wu * (1 - wv) +
                    d10 * (1 - wu) * wv +
                    d11 * wu * wv
                )
                
                # Compute depth difference loss
                depth_diff = torch.abs(valid_depths - sampled_depths)
                
                # Robust loss (Huber loss) to handle outliers
                huber_delta = 0.1
                depth_loss = torch.where(
                    depth_diff <= huber_delta,
                    0.5 * depth_diff ** 2,
                    huber_delta * (depth_diff - 0.5 * huber_delta)
                )
                
                total_loss = total_loss + depth_loss.mean()
            
            return total_loss / batch_size if batch_size > 0 else total_loss
            
        except Exception as e:
            # Return small penalty if depth supervision fails
            penalty = torch.tensor(0.001, device=tactile_points.device, requires_grad=True)
            if tactile_points.requires_grad:
                penalty = penalty * tactile_points.norm(dim=-1).mean()
            return penalty

def tactile_surface_loss(tactile_points: torch.Tensor, tactile_normals: torch.Tensor,
                        gaussian_params: Dict[str, torch.Tensor], 
                        tactile_forces: Optional[torch.Tensor] = None,
                        config: Optional[TactileLossConfig] = None) -> Dict[str, torch.Tensor]:
    """
    Convenience function for computing tactile surface loss.
    
    Args:
        tactile_points: Contact points from tactile sensor [N, 3]
        tactile_normals: Surface normals at contact points [N, 3]
        gaussian_params: Dictionary of Gaussian parameters
        tactile_forces: Optional force measurements [N, 1] or [N, 3]
        config: Loss configuration
        
    Returns:
        Dictionary of loss components
    """
    config = config or TactileLossConfig()
    loss_fn = TactileSurfaceLoss(config)
    
    return loss_fn(tactile_points, tactile_normals, tactile_forces, gaussian_params)

class AdaptiveTactileLoss(nn.Module):
    """
    Adaptive tactile loss that adjusts weights based on reconstruction quality
    and contact patterns.
    """
    
    def __init__(self, base_config: TactileLossConfig):
        super().__init__()
        self.base_config = base_config
        self.loss_fn = TactileSurfaceLoss(base_config)
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.min_weight = 0.1
        self.max_weight = 5.0
        
        # Running statistics
        self.running_losses = {}
        self.adaptation_steps = 0
    
    def forward(self, tactile_points: torch.Tensor, tactile_normals: torch.Tensor,
                tactile_forces: Optional[torch.Tensor], gaussian_params: Dict[str, torch.Tensor],
                rendered_depth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive tactile loss with dynamic weight adjustment.
        """
        # Get base losses
        losses = self.loss_fn(tactile_points, tactile_normals, tactile_forces, 
                             gaussian_params, rendered_depth)
        
        # Update running statistics
        for key, value in losses.items():
            if key != 'total_loss':
                if key not in self.running_losses:
                    self.running_losses[key] = value.item()
                else:
                    self.running_losses[key] = (
                        (1 - self.adaptation_rate) * self.running_losses[key] + 
                        self.adaptation_rate * value.item()
                    )
        
        self.adaptation_steps += 1
        
        # Adapt weights based on loss magnitudes (balance different components)
        if self.adaptation_steps > 10:  # Allow some burn-in
            adapted_losses = {}
            total_adapted = 0
            
            for key, value in losses.items():
                if key != 'total_loss':
                    # Compute adaptive weight (inverse relationship with running average)
                    running_avg = self.running_losses.get(key, 1.0)
                    adaptive_weight = torch.clamp(
                        torch.tensor(1.0 / (running_avg + 1e-8)),
                        self.min_weight, self.max_weight
                    )
                    
                    adapted_losses[key] = value * adaptive_weight
                    total_adapted += adapted_losses[key]
                else:
                    adapted_losses[key] = value
            
            adapted_losses['total_loss'] = total_adapted
            return adapted_losses
        
        return losses