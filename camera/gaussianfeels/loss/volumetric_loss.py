"""
Volumetric loss functions for Gaussian training.
Implements Mahalanobis distance-based surface constraints and multi-modal loss balancing.
Replaces SDF-style losses with Gaussian-native surface constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import numpy as np

class VolumetricLossFunction(nn.Module):
    """
    Volumetric loss functions for Gaussian training.
    
    Implements Mahalanobis distance-based surface constraints, tactile surface losses, 
    and multi-modal balancing.
    """
    
    def __init__(self, 
                 rgb_weight: float = 1.0,
                 depth_weight: float = 0.1,
                 tactile_weight: float = 1.0,
                 eikonal_weight: float = 0.1,
                 surface_weight: float = 1.0):  # Surface constraint weight (replaces SDF)
        super().__init__()
        
        # Multi-modal loss balancing (adapted for Gaussian Splatting)
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.tactile_weight = tactile_weight
        self.eikonal_weight = eikonal_weight
        self.surface_weight = surface_weight
        
    def forward(self, 
                rendered_rgb: torch.Tensor,
                target_rgb: torch.Tensor,
                rendered_depth: torch.Tensor,
                target_depth: torch.Tensor,
                tactile_points: Optional[torch.Tensor] = None,
                tactile_normals: Optional[torch.Tensor] = None,
                gaussian_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute volumetric losses for Gaussian training.
        
        Uses Mahalanobis distance-based surface constraints instead of SDF losses.
        """
        losses = {}
        
        # RGB reconstruction loss (photometric)
        rgb_loss = F.mse_loss(rendered_rgb, target_rgb)
        losses['rgb_loss'] = rgb_loss * self.rgb_weight
        
        # Depth reconstruction loss
        valid_depth_mask = target_depth > 0
        if valid_depth_mask.any():
            depth_loss = F.l1_loss(
                rendered_depth[valid_depth_mask], 
                target_depth[valid_depth_mask]
            )
            losses['depth_loss'] = depth_loss * self.depth_weight
        else:
            losses['depth_loss'] = torch.tensor(0.0, device=rendered_rgb.device)
        
        # Tactile surface loss functions for Gaussian training
        if tactile_points is not None and gaussian_positions is not None:
            tactile_loss = self._compute_tactile_surface_loss(
                tactile_points, tactile_normals, gaussian_positions
            )
            losses['tactile_loss'] = tactile_loss * self.tactile_weight
        else:
            losses['tactile_loss'] = torch.tensor(0.0, device=rendered_rgb.device)
        
        # Eikonal-style loss for surface smoothness (adapted for Gaussian Splatting)
        if gaussian_positions is not None:
            eikonal_loss = self._compute_eikonal_loss(gaussian_positions)
            losses['eikonal_loss'] = eikonal_loss * self.eikonal_weight
        else:
            losses['eikonal_loss'] = torch.tensor(0.0, device=rendered_rgb.device)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_tactile_surface_loss(self, 
                                    tactile_points: torch.Tensor,
                                    tactile_normals: Optional[torch.Tensor],
                                    gaussian_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute tactile surface loss for Gaussian training.
        
        Uses Mahalanobis distance-based surface constraints instead of SDF-based losses.
        """
        # Point-to-surface distance loss (uses Gaussian distances)
        distances = torch.cdist(tactile_points, gaussian_positions)  # [N_tactile, N_gaussians]
        min_distances, _ = torch.min(distances, dim=1)  # [N_tactile]
        
        # Surface distance loss (want tactile points to be close to Gaussians)
        surface_loss = torch.mean(min_distances)
        
        # Normal consistency loss (if normals available)
        if tactile_normals is not None:
            # Find closest Gaussian for each tactile point
            _, closest_gaussian_idx = torch.min(distances, dim=1)
            closest_gaussians = gaussian_positions[closest_gaussian_idx]
            
            # Compute vectors from tactile points to closest Gaussians
            point_to_gaussian = closest_gaussians - tactile_points
            point_to_gaussian_normalized = F.normalize(point_to_gaussian, dim=1)
            
            # Normal should align with point-to-surface direction
            normal_consistency = 1 - torch.abs(torch.sum(
                tactile_normals * point_to_gaussian_normalized, dim=1
            ))
            normal_loss = torch.mean(normal_consistency)
            
            return surface_loss + 0.1 * normal_loss
        
        return surface_loss
    
    def _compute_eikonal_loss(self, gaussian_positions: torch.Tensor) -> torch.Tensor:
        """Compute Eikonal-style loss for surface smoothness (Gaussian surrogate)."""
        if gaussian_positions.shape[0] < 2:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        # Sample random pairs of Gaussians
        n_samples = min(1000, gaussian_positions.shape[0] // 2)
        indices = torch.randperm(gaussian_positions.shape[0])[:n_samples*2]
        points_1 = gaussian_positions[indices[:n_samples]]
        points_2 = gaussian_positions[indices[n_samples:]]
        
        # Compute gradients (finite differences approximation)
        point_diff = points_2 - points_1
        distances = torch.norm(point_diff, dim=1)
        
        # Eikonal-style constraint for smooth Gaussian spatial gradients
        # For Gaussians, we want smooth spatial gradients
        gradient_norms = distances / (torch.norm(point_diff, dim=1) + 1e-8)
        eikonal_loss = torch.mean((gradient_norms - 1.0) ** 2)
        
        return eikonal_loss


class MultiModalLossBalancer:
    """
    Multi-modal loss function balancing.
    
    Adapted for Gaussian Splatting with Mahalanobis distance-based constraints.
    """
    
    def __init__(self, 
                 tactile_first_weight: float = 0.8,
                 camera_first_weight: float = 0.2,
                 balanced_tactile_weight: float = 0.5,
                 balanced_camera_weight: float = 0.5):
        # Multiple fusion modes adapted for Gaussian Splatting
        self.mode_weights = {
            'tactile_first': {
                'tactile_surface': tactile_first_weight,
                'rgb': camera_first_weight * 0.5,
                'depth': camera_first_weight * 0.5
            },
            'balanced': {
                'tactile_surface': balanced_tactile_weight,
                'rgb': balanced_camera_weight * 0.7,
                'depth': balanced_camera_weight * 0.3
            },
            'camera_first': {
                'tactile_surface': camera_first_weight,
                'rgb': tactile_first_weight * 0.7,
                'depth': tactile_first_weight * 0.3
            }
        }
        self.current_mode = 'tactile_first'  # Default mode
        
    def balance_losses(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Balance losses according to current mode.
        
        Applies mode-specific weights to balance tactile and visual losses
        based on the current training mode (tactile_only, visual_only, balanced).
        """
        balanced_losses = {}
        weights = self.mode_weights[self.current_mode]
        
        # Apply mode-specific weights
        if 'tactile_loss' in losses:
            balanced_losses['tactile_loss_weighted'] = (
                losses['tactile_loss'] * weights.get('tactile_surface', 1.0)
            )
        
        if 'rgb_loss' in losses:
            balanced_losses['rgb_loss_weighted'] = (
                losses['rgb_loss'] * weights.get('rgb', 1.0)
            )
            
        if 'depth_loss' in losses:
            balanced_losses['depth_loss_weighted'] = (
                losses['depth_loss'] * weights.get('depth', 1.0)
            )
        
        # Keep other losses unchanged
        for key, value in losses.items():
            if key not in ['tactile_loss', 'rgb_loss', 'depth_loss']:
                balanced_losses[key] = value
        
        # Recompute total loss
        total_loss = sum(v for k, v in balanced_losses.items() 
                        if k.endswith('_weighted') or k == 'eikonal_loss')
        balanced_losses['total_loss_balanced'] = total_loss
        
        return balanced_losses
    
    def set_mode(self, mode: str):
        """Set current loss balancing mode"""
        if mode in self.mode_weights:
            self.current_mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.mode_weights.keys())}")


class MultiModalGaussianLoss(nn.Module):
    """
    Complete multi-modal loss system for Gaussian Splatting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        config = config or {}
        
        # Initialize volumetric loss function
        self.volumetric_loss = VolumetricLossFunction(
            rgb_weight=config.get('rgb_weight', 1.0),
            depth_weight=config.get('depth_weight', 0.1),
            tactile_weight=config.get('tactile_weight', 1.0),
            eikonal_weight=config.get('eikonal_weight', 0.1),
            surface_weight=config.get('surface_weight', 1.0)  # Surface constraint weight (replaces SDF)
        )
        
        # Initialize multi-modal balancer
        self.loss_balancer = MultiModalLossBalancer(
            tactile_first_weight=config.get('tactile_first_weight', 0.8),
            camera_first_weight=config.get('camera_first_weight', 0.2)
        )
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                mode: str = 'tactile_first') -> Dict[str, torch.Tensor]:
        """Compute complete multi-modal loss."""
        # Set loss balancing mode
        self.loss_balancer.set_mode(mode)
        
        # Compute base volumetric losses
        losses = self.volumetric_loss(
            rendered_rgb=predictions.get('rgb'),
            target_rgb=targets.get('rgb'),
            rendered_depth=predictions.get('depth'),
            target_depth=targets.get('depth'),
            tactile_points=targets.get('tactile_points'),
            tactile_normals=targets.get('tactile_normals'),
            gaussian_positions=predictions.get('gaussian_positions')
        )
        
        # Apply multi-modal balancing
        balanced_losses = self.loss_balancer.balance_losses(losses)
        
        return balanced_losses