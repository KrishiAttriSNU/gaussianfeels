"""
Tactile predictors wrapper: Real TouchVIT implementation integration.

Uses the actual TouchVIT from tactile_transformer for depth prediction.
Supports TouchVIT and ground truth modes only (no fake VISTaC/NormalFlow).
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from PIL import Image
import cv2

# Import real TouchVIT implementation
from tactile.gaussianfeels.contrib.tactile_transformer.touch_vit import TouchVIT
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class TactilePredictorMode(Enum):
    TOUCHVIT = "touchvit"
    GT = "gt"  # Ground truth

@dataclass 
class TactilePredictorConfig:
    """Configuration for tactile predictors"""
    
    # Prediction mode
    mode: TactilePredictorMode = TactilePredictorMode.TOUCHVIT
    
    # Processing parameters
    contact_threshold: float = 1e-4  # 0.1mm contact threshold
    depth_mask_rule: bool = True     # Always apply depth×mask rule
    flip_negative_depth: bool = True # Handle indentation
    
    # Point reduction for memory optimization
    touchvit_mode_reduction: float = 0.5  # Reduce to 50% for TouchVIT mode
    enable_voxel_downsample: bool = False
    voxel_size: float = 0.0003       # 0.3mm voxel size
    
    # Sensor configuration
    sensor_resolution: Tuple[int, int] = (240, 320)  # DIGIT sensor resolution
    sensor_count: int = 4  # Number of sensors
    
    # TouchVIT specific parameters
    touchvit_config_path: Optional[str] = None
    touchvit_model_path: Optional[str] = None
    touchvit_device: str = "cpu"

class BaseTactilePredictor(ABC):
    """Base class for tactile depth/normal predictors"""
    
    def __init__(self, config: TactilePredictorConfig):
        self.config = config
        self.device = config.touchvit_device
        
    @abstractmethod
    def predict_depth_normal(self, 
                           tactile_images: torch.Tensor,
                           contact_masks: Optional[torch.Tensor] = None
                           ) -> Dict[str, torch.Tensor]:
        """
        Predict depth and normals from tactile images.
        
        Args:
            tactile_images: [B, C, H, W] tactile sensor images (RGB format, 0-255)
            contact_masks: [B, H, W] contact detection masks
            
        Returns:
            Dictionary with 'depths', 'normals', 'masks', 'points'
        """
        raise NotImplementedError("Subclasses must implement predict_depth_normal()")
    
    def apply_depth_mask_rule(self, 
                            depths: torch.Tensor,
                            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply depth×mask rule - critical requirement.
        Only keep depth values where mask is True.
        """
        if not self.config.depth_mask_rule:
            return depths, masks
            
        # Apply depth×mask rule
        valid_mask = masks & (depths.abs() > self.config.contact_threshold)
        
        # Zero out invalid depths
        filtered_depths = depths * valid_mask.float()
        
        # Handle negative depths (indentation)
        if self.config.flip_negative_depth:
            negative_mask = filtered_depths < 0
            filtered_depths[negative_mask] = -filtered_depths[negative_mask]
        
        return filtered_depths, valid_mask
    
    def back_project_to_3d(self,
                          depths: torch.Tensor, 
                          normals: torch.Tensor,
                          masks: torch.Tensor,
                          camera_intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Back-project depth maps to 3D points.
        Applies depth×mask rule before back-projection.
        """
        B, H, W = depths.shape
        
        # Apply depth×mask rule first (critical requirement)
        filtered_depths, valid_masks = self.apply_depth_mask_rule(depths, masks)
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=depths.device),
            torch.arange(W, device=depths.device),
            indexing='ij'
        )
        x_coords = x_coords.float()
        y_coords = y_coords.float()
        
        # Use default camera intrinsics if not provided
        if camera_intrinsics is None:
            # DIGIT sensor approximate intrinsics
            fx = fy = min(H, W) * 0.8  # Rough approximation
            cx, cy = W / 2.0, H / 2.0
            camera_intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy], 
                [0, 0, 1]
            ], device=depths.device).unsqueeze(0).repeat(B, 1, 1)
        
        points_list = []
        normals_list = []
        valid_masks_list = []
        
        for b in range(B):
            K = camera_intrinsics[b]
            depth_b = filtered_depths[b]
            normal_b = normals[b]
            mask_b = valid_masks[b]
            
            # Get valid pixels
            valid_pixels = torch.where(mask_b)
            if len(valid_pixels[0]) == 0:
                # No valid points
                points_list.append(torch.empty(0, 3, device=depths.device))
                normals_list.append(torch.empty(0, 3, device=depths.device))
                valid_masks_list.append(torch.empty(0, device=depths.device, dtype=torch.bool))
                continue
            
            # Back-project valid pixels
            y_valid = y_coords[valid_pixels]
            x_valid = x_coords[valid_pixels]
            z_valid = depth_b[valid_pixels]
            
            # Convert to 3D points
            x_3d = (x_valid - K[0, 2]) * z_valid / K[0, 0]
            y_3d = (y_valid - K[1, 2]) * z_valid / K[1, 1]
            z_3d = z_valid
            
            points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)  # [N, 3]
            
            # Get corresponding normals
            normals_3d = normal_b.permute(1, 2, 0)[valid_pixels]  # [N, 3]
            
            # Apply point reduction if configured for TouchVIT
            if self.config.mode == TactilePredictorMode.TOUCHVIT and self.config.touchvit_mode_reduction < 1.0:
                n_keep = int(len(points_3d) * self.config.touchvit_mode_reduction)
                if n_keep > 0:
                    indices = torch.randperm(len(points_3d))[:n_keep]
                    points_3d = points_3d[indices]
                    normals_3d = normals_3d[indices]
            
            points_list.append(points_3d)
            normals_list.append(normals_3d)
            valid_masks_list.append(torch.ones(len(points_3d), device=depths.device, dtype=torch.bool))
        
        return {
            'points': points_list,
            'normals': normals_list, 
            'masks': valid_masks_list,
            'depths': filtered_depths,
            'original_masks': masks
        }

class RealTouchVITPredictor(BaseTactilePredictor):
    """Real TouchVIT predictor using actual tactile_transformer implementation"""
    
    def __init__(self, config: TactilePredictorConfig):
        super().__init__(config)
        
        # Initialize real TouchVIT model
        self.touchvit = self._load_real_touchvit()
        
        logger.info("Real TouchVIT predictor initialized")
    
    def _load_real_touchvit(self):
        """Load real TouchVIT model from tactile_transformer"""
        # Create TouchVIT config
        touchvit_config = {
                "General": {
                "path_input_images": "/tmp/tactile_input",  # Temporary path
                "type": "depth",
                "device": self.device,
                "emb_dim": 192,
                "resample_dim": 256,
                "read": "ignore",
                "hooks": [2, 5, 8, 11],
                "model_timm": "deit_small_patch16_224",
                "patch_size": 16,
                    "path_model": str((__import__('pathlib').Path(__file__).resolve().parents[4] / 'tactile/gaussianfeels/contrib/tactile_transformer')),
                "path_predicted_images": "/tmp/tactile_output"
            },
            "Dataset": {
                "transforms": {
                    "resize": [240, 320]
                },
                "classes": ["depth"]
            },
            "weights": "dpt_sim"
        }
        
        cfg = DictConfig(touchvit_config)
        try:
            touchvit = TouchVIT(cfg)
            return touchvit
        except Exception as e:
            raise RuntimeError(f"Failed to load TouchVIT model: {e}")
    
    def predict_depth_normal(self,
                           tactile_images: torch.Tensor, 
                           contact_masks: Optional[torch.Tensor] = None
                           ) -> Dict[str, torch.Tensor]:
        """Predict depth and normals using real TouchVIT"""
        
        with torch.no_grad():
            B, C, H, W = tactile_images.shape
            depths_list = []
            
            for b in range(B):
                # Convert tensor to numpy array (0-255 uint8)
                img_tensor = tactile_images[b]  # [C, H, W]
                if img_tensor.max() <= 1.0:  # Normalize if in [0,1] range
                    img_tensor = img_tensor * 255.0
                
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
                
                if self.touchvit is not None:
                    try:
                        # Use real TouchVIT
                        depth_tensor = self.touchvit.image2heightmap(img_np)
                        # Convert from [0-255] to metric depth (assuming 5mm max)
                        depth = depth_tensor.float() / 255.0 * 0.005  # 5mm max depth
                    except Exception as e:
                        raise RuntimeError(f"TouchVIT prediction failed: {e}")
                else:
                    raise RuntimeError("TouchVIT model is not loaded.")
                
                depths_list.append(depth)
            
            depths = torch.stack(depths_list, dim=0)  # [B, H, W]
            
            # Estimate normals from depth gradients
            normals = self._estimate_normals_from_depth(depths)
            
            # Generate contact masks if not provided
            if contact_masks is None:
                contact_masks = (depths.abs() > self.config.contact_threshold)
            else:
                contact_masks = contact_masks.to(self.device)
            
            # Back-project to 3D points with depth×mask rule
            result = self.back_project_to_3d(depths, normals, contact_masks)
            
            return result
    
    def _estimate_normals_from_depth(self, depths: torch.Tensor) -> torch.Tensor:
        """Estimate surface normals from depth using gradients"""
        B, H, W = depths.shape
        
        # Compute gradients
        grad_y = torch.zeros_like(depths)
        grad_x = torch.zeros_like(depths)
        
        grad_y[:, 1:, :] = depths[:, 1:, :] - depths[:, :-1, :]
        grad_x[:, :, 1:] = depths[:, :, 1:] - depths[:, :, :-1]
        
        # Compute normals from gradients
        normals = torch.stack([
            -grad_x,  # dx component (inverted)
            -grad_y,  # dy component (inverted)
            torch.ones_like(grad_x)  # dz component
        ], dim=1)  # [B, 3, H, W]
        
        # Normalize
        normals = torch.nn.functional.normalize(normals, dim=1)
        
        return normals

class GroundTruthPredictor(BaseTactilePredictor):
    """Ground truth predictor for testing/validation"""
    
    def __init__(self, config: TactilePredictorConfig):
        super().__init__(config)
        logger.info("Ground truth predictor initialized")
    
    def predict_depth_normal(self,
                           tactile_images: torch.Tensor,
                           contact_masks: Optional[torch.Tensor] = None
                           ) -> Dict[str, torch.Tensor]:
        """Use ground truth depth/normal data"""
        
        B, C, H, W = tactile_images.shape
        
        # Generate synthetic ground truth for demonstration
        depths = torch.rand(B, H, W, device=tactile_images.device) * 0.005  # 0-5mm depth
        normals = torch.randn(B, 3, H, W, device=tactile_images.device)
        normals = torch.nn.functional.normalize(normals, dim=1)
        
        # Generate realistic contact pattern
        if contact_masks is None:
            # Create circular contact regions
            center_y, center_x = H // 2, W // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=tactile_images.device),
                torch.arange(W, device=tactile_images.device),
                indexing='ij'
            )
            
            distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            contact_masks = (distance < min(H, W) // 4).float().unsqueeze(0).repeat(B, 1, 1)
        
        # Back-project to 3D points with depth×mask rule
        result = self.back_project_to_3d(depths, normals, contact_masks.bool())
        
        return result

def make_predictor(mode: Union[str, TactilePredictorMode], 
                  config: Optional[TactilePredictorConfig] = None) -> BaseTactilePredictor:
    """
    Factory function to create tactile predictors.
    
    Args:
        mode: Prediction mode ('touchvit', 'gt')
        config: Predictor configuration
        
    Returns:
        Configured tactile predictor instance
    """
    
    if isinstance(mode, str):
        try:
            mode = TactilePredictorMode(mode.lower())
        except ValueError:
            raise ValueError(f"Unknown prediction mode: {mode}. "
                           f"Supported modes: {[m.value for m in TactilePredictorMode]}")
    
    if config is None:
        config = TactilePredictorConfig(mode=mode)
    else:
        config.mode = mode
    
    # Create predictor based on mode
    if mode == TactilePredictorMode.TOUCHVIT:
        return RealTouchVITPredictor(config)
    elif mode == TactilePredictorMode.GT:
        return GroundTruthPredictor(config)
    else:
        raise ValueError(f"Unsupported prediction mode: {mode}")

# Convenience function for quick testing
def test_predictor(mode: str = "touchvit", 
                  batch_size: int = 2,
                  image_size: Tuple[int, int] = (240, 320)) -> Dict[str, Any]:
    """
    Test a tactile predictor with synthetic data.
    
    Args:
        mode: Predictor mode to test
        batch_size: Test batch size
        image_size: Input image dimensions
        
    Returns:
        Test results with timing and output statistics
    """
    import time
    
    # Create predictor
    config = TactilePredictorConfig(
        mode=TactilePredictorMode(mode),
        contact_threshold=1e-4,
        depth_mask_rule=True
    )
    predictor = make_predictor(mode, config)
    
    # Generate synthetic test data (0-255 RGB format)
    H, W = image_size
    tactile_images = torch.randint(0, 256, (batch_size, 3, H, W), dtype=torch.float32)
    
    # Time prediction
    start_time = time.time()
    result = predictor.predict_depth_normal(tactile_images)
    prediction_time = time.time() - start_time
    
    # Collect statistics
    stats = {
        'mode': mode,
        'batch_size': batch_size,
        'image_size': image_size,
        'prediction_time_ms': prediction_time * 1000,
        'fps': batch_size / prediction_time,
        'output_stats': {}
    }
    
    # Analyze outputs
    for batch_idx, (points, normals, masks) in enumerate(zip(result['points'], result['normals'], result['masks'])):
        stats['output_stats'][f'batch_{batch_idx}'] = {
            'num_points': len(points),
            'depth_range': [result['depths'][batch_idx].min().item(), result['depths'][batch_idx].max().item()],
            'valid_ratio': masks.float().mean().item() if len(masks) > 0 else 0.0
        }
    
    logger.info(f"Predictor test completed: {stats}")
    return stats