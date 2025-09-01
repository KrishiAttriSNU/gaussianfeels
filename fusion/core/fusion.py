"""
Object-centric multi-modal fusion pipeline following Consensus Final specifications.

Critical requirements:
- Transform inputs to object frame before loss/updates
- Tactile primary, camera secondary loss weighting  
- Maintain T_WO(i) per frame for object-centric updates
- Configurable integration modes (concurrent|tactile_first|vision_first)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from gaussian.core.gaussian_field import ObjectGaussianMap, GaussianConfig
from fusion.loss.tactile_loss import tactile_surface_loss, TactileLossConfig
from gaussian.render.rasterizer import render_rgbd, RenderConfig, CameraParams

logger = logging.getLogger(__name__)

class IntegrationMode(Enum):
    CONCURRENT = "concurrent"
    TACTILE_FIRST = "tactile_first" 
    VISION_FIRST = "vision_first"

@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion pipeline"""
    
    # Integration mode
    integration_mode: IntegrationMode = IntegrationMode.CONCURRENT
    
    # Loss weighting (tactile primary, vision secondary)
    tactile_loss_weight: float = 0.8  # Primary modality
    vision_loss_weight: float = 0.2   # Secondary modality
    
    # Learning rates per modality
    tactile_lr_multiplier: float = 1.0
    vision_lr_multiplier: float = 0.5  # Slower learning for secondary
    
    # Fusion parameters
    temporal_consistency_weight: float = 0.1
    spatial_consistency_weight: float = 0.05
    
    # Object-centric parameters
    pose_prediction_enabled: bool = True
    pose_update_threshold: float = 0.01  # meters
    max_pose_delta: float = 1.0  # meters per frame
    
    # Memory and performance
    max_points_per_modality: int = 50_000
    fusion_batch_size: int = 4096
    
    # Integration timing (for tactile_first/vision_first modes)
    tactile_first_delay_ms: float = 50.0  # Wait for tactile before vision
    vision_first_delay_ms: float = 16.7   # ~60 FPS vision lead

class ObjectCentricFusion(nn.Module):
    """
    Object-centric multi-modal fusion pipeline.
    
    Key features:
    - Transforms all inputs to object coordinate frame before processing
    - Tactile-primary fusion with configurable loss weighting
    - Maintains object pose T_WO(i) per frame
    - Supports multiple integration modes
    """
    
    def __init__(self, 
                 gaussian_config: GaussianConfig,
                 fusion_config: FusionConfig,
                 tactile_loss_config: TactileLossConfig,
                 render_config: RenderConfig):
        super().__init__()
        
        self.fusion_config = fusion_config
        self.tactile_loss_config = tactile_loss_config
        self.render_config = render_config
        
        # Core Gaussian field
        self.gaussian_field = ObjectGaussianMap(gaussian_config)
        
        # Object pose tracking
        self.current_pose = torch.eye(4, dtype=torch.float32)  # T_WO current
        self.pose_history = []  # Store recent poses for temporal consistency
        self.max_pose_history = 10
        
        # Fusion state
        self.last_tactile_timestamp = 0.0
        self.last_vision_timestamp = 0.0
        self.pending_tactile_data = None
        self.pending_vision_data = None
        
        # Performance tracking
        self.fusion_metrics = {
            'tactile_loss_history': [],
            'vision_loss_history': [],
            'pose_updates': 0,
            'fusion_cycles': 0
        }
        
        logger.info(f"ObjectCentricFusion initialized with mode: {fusion_config.integration_mode}")
    
    def update_object_pose(self, new_pose: torch.Tensor) -> bool:
        """
        Update object pose T_WO with validation.
        Critical requirement: maintain T_WO(i) per frame
        """
        if new_pose.shape != (4, 4):
            logger.error(f"Invalid pose shape: {new_pose.shape}, expected (4, 4)")
            return False
            
        # Validate pose change magnitude
        if len(self.pose_history) > 0:
            last_pose = self.pose_history[-1]
            pose_delta = torch.norm(new_pose[:3, 3] - last_pose[:3, 3]).item()
            
            if pose_delta > self.fusion_config.max_pose_delta:
                logger.warning(f"Large pose delta {pose_delta:.4f}m, clamping to {self.fusion_config.max_pose_delta}m")
                # Clamp the pose change
                direction = (new_pose[:3, 3] - last_pose[:3, 3]) / pose_delta
                new_pose[:3, 3] = last_pose[:3, 3] + direction * self.fusion_config.max_pose_delta
        
        # Update current pose
        self.current_pose = new_pose.clone()
        
        # Maintain pose history
        self.pose_history.append(new_pose.clone())
        if len(self.pose_history) > self.max_pose_history:
            self.pose_history.pop(0)
            
        self.fusion_metrics['pose_updates'] += 1
        return True
    
    def transform_to_object_frame(self, 
                                 points_world: torch.Tensor,
                                 normals_world: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transform points from world to object coordinate frame: pc_world → pc_obj
        Critical requirement from Consensus Final
        """
        if points_world.numel() == 0:
            return points_world, normals_world
            
        # Get inverse transform (world → object)
        T_OW = torch.inverse(self.current_pose)
        
        # Transform points
        points_homo = torch.cat([points_world, torch.ones(points_world.shape[0], 1, device=points_world.device)], dim=1)
        points_obj_homo = (T_OW @ points_homo.T).T
        points_obj = points_obj_homo[:, :3]
        
        # Transform normals if provided
        normals_obj = None
        if normals_world is not None:
            # Normals transform with rotation only (not translation)
            R_OW = T_OW[:3, :3]
            normals_obj = (R_OW @ normals_world.T).T
            
        return points_obj, normals_obj
    
    def process_tactile_data(self, tactile_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process tactile data in object frame with depth×mask rule enforcement.
        Critical: Apply depth×mask rule before back-projection
        """
        processed_data = {}
        
        for sensor_name, sensor_data in tactile_data.items():
            if 'points' not in sensor_data or 'depths' not in sensor_data:
                continue
                
            points_world = sensor_data['points']
            depths = sensor_data['depths'] 
            
            # Apply depth×mask rule (critical requirement)
            if 'mask' in sensor_data:
                valid_mask = sensor_data['mask'] & (depths > 0)  # depth×mask
                points_world = points_world[valid_mask]
                depths = depths[valid_mask]
            
            # Handle negative depths (indentation)
            if 'flip_negative_depth' in sensor_data and sensor_data['flip_negative_depth']:
                negative_mask = depths < 0
                depths[negative_mask] = -depths[negative_mask]
            
            # Transform to object frame
            points_obj, normals_obj = self.transform_to_object_frame(
                points_world, 
                sensor_data.get('normals')
            )
            
            processed_data[sensor_name] = {
                'points_obj': points_obj,
                'normals_obj': normals_obj,
                'depths': depths,
                'original_points': points_world
            }
            
        return processed_data
    
    def process_vision_data(self, vision_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process vision data in object frame"""
        processed_data = {}
        
        if 'points' in vision_data:
            points_world = vision_data['points']
            
            # Transform to object frame
            points_obj, normals_obj = self.transform_to_object_frame(
                points_world,
                vision_data.get('normals')
            )
            
            processed_data = {
                'points_obj': points_obj,
                'normals_obj': normals_obj,
                'colors': vision_data.get('colors'),
                'original_points': points_world
            }
            
        return processed_data
    
    def compute_tactile_loss(self, 
                           tactile_data: Dict[str, torch.Tensor],
                           gaussian_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute tactile surface loss in object frame"""
        total_loss = torch.tensor(0.0, device=gaussian_params['positions'].device)
        
        for sensor_name, sensor_data in tactile_data.items():
            if 'points_obj' not in sensor_data:
                continue
                
            points_obj = sensor_data['points_obj']
            depths = sensor_data['depths']
            
            # Compute surface loss
            loss = tactile_surface_loss(
                tactile_points=points_obj,
                tactile_depths=depths,
                gaussian_positions=gaussian_params['positions'],
                gaussian_scales=gaussian_params['scales'],
                gaussian_rotations=gaussian_params['rotations'],
                config=self.tactile_loss_config
            )
            
            total_loss += loss
            
        return total_loss
    
    def compute_vision_loss(self,
                           vision_data: Dict[str, torch.Tensor],
                           gaussian_params: Dict[str, torch.Tensor],
                           camera_params: CameraParams) -> torch.Tensor:
        """Compute vision loss with differentiable rendering (L1+SSIM on RGB)."""
        device = gaussian_params['positions'].device
        if 'rgb' not in vision_data:
            raise RuntimeError("RGB data required for vision loss computation but not found in vision_data")

        # Render from current viewpoint using rasterizer convenience
        from gaussian.render.rasterizer import GaussianRasterizer, RenderConfig
        H = camera_params.height
        W = camera_params.width
        rconfig = RenderConfig(image_height=H, image_width=W, sh_degree=3, backend='auto')
        rasterizer = GaussianRasterizer(rconfig).to(device)

        T_WC = vision_data['T_WC'] if 'T_WC' in vision_data else torch.eye(4, device=device)
        out = rasterizer(gaussian_params, camera_params, T_WC)

        pred = out['rgb'].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        tgt = vision_data['rgb'].to(device).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

        l1 = torch.nn.functional.l1_loss(pred, tgt)
        # Simple SSIM proxy if real SSIM not available here
        cos = 1.0 - torch.nn.functional.cosine_similarity(pred.view(1, -1), tgt.view(1, -1))
        ssim_like = cos.mean()
        return 0.8 * l1 + 0.2 * ssim_like
    
    def fuse_modalities(self,
                       tactile_data: Optional[Dict[str, torch.Tensor]] = None,
                       vision_data: Optional[Dict[str, torch.Tensor]] = None,
                       camera_params: Optional[CameraParams] = None,
                       timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Main fusion pipeline with configurable integration modes.
        
        Returns fusion result with losses and updated Gaussian parameters.
        """
        self.fusion_metrics['fusion_cycles'] += 1
        
        # Handle integration modes
        if self.fusion_config.integration_mode == IntegrationMode.TACTILE_FIRST:
            return self._fuse_tactile_first(tactile_data, vision_data, camera_params, timestamp)
        elif self.fusion_config.integration_mode == IntegrationMode.VISION_FIRST:
            return self._fuse_vision_first(tactile_data, vision_data, camera_params, timestamp)
        else:  # CONCURRENT
            return self._fuse_concurrent(tactile_data, vision_data, camera_params, timestamp)
    
    def _fuse_concurrent(self,
                        tactile_data: Optional[Dict[str, torch.Tensor]],
                        vision_data: Optional[Dict[str, torch.Tensor]], 
                        camera_params: Optional[CameraParams],
                        timestamp: float) -> Dict[str, Any]:
        """Concurrent fusion: process both modalities simultaneously"""
        
        total_loss = torch.tensor(0.0)
        losses = {}
        
        # Get current Gaussian parameters
        gaussian_params = self.gaussian_field.get_gaussian_parameters()
        if gaussian_params['positions'].numel() == 0:
            return {'total_loss': total_loss, 'losses': losses, 'updated': False}
        
        # Process tactile data (primary modality)
        if tactile_data is not None:
            processed_tactile = self.process_tactile_data(tactile_data)
            tactile_loss = self.compute_tactile_loss(processed_tactile, gaussian_params)
            weighted_tactile_loss = tactile_loss * self.fusion_config.tactile_loss_weight
            total_loss += weighted_tactile_loss
            losses['tactile'] = tactile_loss.item()
            self.fusion_metrics['tactile_loss_history'].append(tactile_loss.item())
            
        # Process vision data (secondary modality)
        if vision_data is not None and camera_params is not None:
            processed_vision = self.process_vision_data(vision_data)
            vision_loss = self.compute_vision_loss(processed_vision, gaussian_params, camera_params)
            weighted_vision_loss = vision_loss * self.fusion_config.vision_loss_weight
            total_loss += weighted_vision_loss
            losses['vision'] = vision_loss.item()
            self.fusion_metrics['vision_loss_history'].append(vision_loss.item())
        
        # Add temporal consistency loss
        if len(self.pose_history) > 1:
            temporal_loss = self._compute_temporal_consistency_loss()
            total_loss += temporal_loss * self.fusion_config.temporal_consistency_weight
            losses['temporal'] = temporal_loss.item()
        
        return {
            'total_loss': total_loss,
            'losses': losses,
            'gaussian_params': gaussian_params,
            'updated': True,
            'timestamp': timestamp
        }
    
    def _fuse_tactile_first(self,
                           tactile_data: Optional[Dict[str, torch.Tensor]],
                           vision_data: Optional[Dict[str, torch.Tensor]],
                           camera_params: Optional[CameraParams], 
                           timestamp: float) -> Dict[str, Any]:
        """Tactile-first fusion: process tactile data first, then vision"""
        
        # Store vision data if tactile not ready
        if tactile_data is None and vision_data is not None:
            self.pending_vision_data = vision_data
            self.last_vision_timestamp = timestamp
            return {'total_loss': torch.tensor(0.0), 'losses': {}, 'updated': False}
        
        # Process tactile when available
        if tactile_data is not None:
            self.last_tactile_timestamp = timestamp
            
            # Check if we should wait for more recent vision data
            vision_to_use = vision_data
            if (self.pending_vision_data is not None and 
                timestamp - self.last_vision_timestamp < self.fusion_config.tactile_first_delay_ms / 1000.0):
                vision_to_use = self.pending_vision_data
                self.pending_vision_data = None
            
            return self._fuse_concurrent(tactile_data, vision_to_use, camera_params, timestamp)
        
        return {'total_loss': torch.tensor(0.0), 'losses': {}, 'updated': False}
    
    def _fuse_vision_first(self,
                          tactile_data: Optional[Dict[str, torch.Tensor]],
                          vision_data: Optional[Dict[str, torch.Tensor]],
                          camera_params: Optional[CameraParams],
                          timestamp: float) -> Dict[str, Any]:
        """Vision-first fusion: process vision data first, then tactile"""
        
        # Store tactile data if vision not ready
        if vision_data is None and tactile_data is not None:
            self.pending_tactile_data = tactile_data
            self.last_tactile_timestamp = timestamp
            return {'total_loss': torch.tensor(0.0), 'losses': {}, 'updated': False}
        
        # Process vision when available
        if vision_data is not None:
            self.last_vision_timestamp = timestamp
            
            # Check if we should wait for more recent tactile data
            tactile_to_use = tactile_data
            if (self.pending_tactile_data is not None and
                timestamp - self.last_tactile_timestamp < self.fusion_config.vision_first_delay_ms / 1000.0):
                tactile_to_use = self.pending_tactile_data
                self.pending_tactile_data = None
            
            return self._fuse_concurrent(tactile_to_use, vision_data, camera_params, timestamp)
        
        return {'total_loss': torch.tensor(0.0), 'losses': {}, 'updated': False}
    
    def _compute_temporal_consistency_loss(self) -> torch.Tensor:
        """Compute temporal consistency loss between consecutive poses"""
        if len(self.pose_history) < 2:
            return torch.tensor(0.0)
            
        current_pose = self.pose_history[-1]
        previous_pose = self.pose_history[-2]
        
        # Simple L2 loss on pose difference
        pose_diff = current_pose - previous_pose
        temporal_loss = torch.norm(pose_diff[:3, 3])  # Position difference
        
        return temporal_loss
    
    def add_points_to_field(self, 
                           points_obj: torch.Tensor,
                           normals_obj: Optional[torch.Tensor] = None) -> bool:
        """Add new points to the Gaussian field in object coordinates"""
        if points_obj.numel() == 0:
            return False
            
        # Limit points per addition to manage memory
        max_points = self.fusion_config.max_points_per_modality
        if points_obj.shape[0] > max_points:
            # Sample points to stay within limit
            indices = torch.randperm(points_obj.shape[0])[:max_points]
            points_obj = points_obj[indices]
            if normals_obj is not None:
                normals_obj = normals_obj[indices]
        
        return self.gaussian_field.add_gaussians(points_obj)
    
    def get_fusion_metrics(self) -> Dict[str, Any]:
        """Get fusion performance metrics"""
        metrics = self.fusion_metrics.copy()
        
        # Add current state
        metrics.update({
            'current_pose': self.current_pose.numpy(),
            'pose_history_length': len(self.pose_history),
            'active_gaussians': self.gaussian_field.num_active_gaussians,
            'memory_usage': self.gaussian_field.get_memory_usage(),
            'integration_mode': self.fusion_config.integration_mode.value
        })
        
        return metrics
    
    def reset_fusion_state(self):
        """Reset fusion state for new sequence"""
        self.current_pose = torch.eye(4, dtype=torch.float32)
        self.pose_history.clear()
        self.pending_tactile_data = None
        self.pending_vision_data = None
        self.last_tactile_timestamp = 0.0
        self.last_vision_timestamp = 0.0
        
        # Reset metrics
        self.fusion_metrics = {
            'tactile_loss_history': [],
            'vision_loss_history': [],
            'pose_updates': 0,
            'fusion_cycles': 0
        }
        
        logger.info("Fusion state reset")
    
    def forward(self, 
               tactile_data: Optional[Dict[str, torch.Tensor]] = None,
               vision_data: Optional[Dict[str, torch.Tensor]] = None,
               camera_params: Optional[CameraParams] = None,
               timestamp: float = 0.0) -> Dict[str, Any]:
        """Forward pass through fusion pipeline"""
        return self.fuse_modalities(tactile_data, vision_data, camera_params, timestamp)