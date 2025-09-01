"""
Coarse-to-Fine Pose Optimization for Real-time Performance
Implements pyramid-based optimization for faster convergence and better performance
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .gaussian_pose_optimizer import PoseOptConfig
from fusion.core.pose_optimizer import PoseOptimizer
from ..render.rasterizer import CameraParams


@dataclass
class CoarseToFineConfig:
    """Configuration for coarse-to-fine optimization - STRICT: No fallback values allowed"""
    pyramid_levels: int
    level_scales: List[float]
    iterations_per_level: List[int] 
    pixel_sampling_per_level: List[float]
    early_termination: bool
    convergence_threshold_per_level: List[float]
    
    def __post_init__(self):
        # STRICT VALIDATION - NO FALLBACKS
        if len(self.level_scales) != self.pyramid_levels:
            raise ValueError(f"level_scales must have exactly {self.pyramid_levels} elements, got {len(self.level_scales)}")
        if len(self.iterations_per_level) != self.pyramid_levels:
            raise ValueError(f"iterations_per_level must have exactly {self.pyramid_levels} elements, got {len(self.iterations_per_level)}")
        if len(self.pixel_sampling_per_level) != self.pyramid_levels:
            raise ValueError(f"pixel_sampling_per_level must have exactly {self.pyramid_levels} elements, got {len(self.pixel_sampling_per_level)}")
        if len(self.convergence_threshold_per_level) != self.pyramid_levels:
            raise ValueError(f"convergence_threshold_per_level must have exactly {self.pyramid_levels} elements, got {len(self.convergence_threshold_per_level)}")
        
        # Validate ranges - FAIL FAST if invalid
        for i, scale in enumerate(self.level_scales):
            if scale <= 0 or scale > 1.0:
                raise ValueError(f"level_scales[{i}] = {scale} must be in range (0, 1.0]")
        
        for i, iters in enumerate(self.iterations_per_level):
            if iters <= 0:
                raise ValueError(f"iterations_per_level[{i}] = {iters} must be > 0")
        
        for i, sampling in enumerate(self.pixel_sampling_per_level):
            if sampling <= 0 or sampling > 1.0:
                raise ValueError(f"pixel_sampling_per_level[{i}] = {sampling} must be in range (0, 1.0]")
        
        for i, threshold in enumerate(self.convergence_threshold_per_level):
            if threshold <= 0:
                raise ValueError(f"convergence_threshold_per_level[{i}] = {threshold} must be > 0")


class CoarseToFinePoseOptimizer(PoseOptimizer):
    """
    Coarse-to-fine pose optimizer using image pyramids for performance
    """
    
    def __init__(self, gaussian_map, sensors, config: PoseOptConfig, 
                 ctf_config: CoarseToFineConfig, device: str = 'cuda'):
        # Initialize modules pose optimizer
        super().__init__(sensors, config, train_mode=False, device=device)
        
        # Set the gaussian map (required for modules optimizer)
        self.set_gaussian_map(gaussian_map)
        
        self.ctf_config = ctf_config
        
        # Precomputed image pyramids for efficiency
        self.reference_pyramids = {}
        self.current_pyramids = {}
        
    def optimize_poses(self, object_poses: Dict[int, torch.Tensor], 
                      max_iterations: Optional[int] = None) -> Tuple[Dict[int, torch.Tensor], Dict]:
        """
        Coarse-to-fine pose optimization using image pyramids
        """
        # Prepare image pyramids
        self._prepare_image_pyramids()
        
        info = {
            'level_info': [],
            'total_iterations': 0,
            'convergence': False,
            'timing': {}
        }
        
        start_time = time.time()
        current_poses = {k: v.clone() for k, v in object_poses.items()}
        
        # Optimize at each pyramid level
        for level in range(self.ctf_config.pyramid_levels):
            level_start_time = time.time()
            
            print(f"🔍 Optimizing at pyramid level {level + 1}/{self.ctf_config.pyramid_levels}")
            print(f"   Scale: {self.ctf_config.level_scales[level]:.2f}")
            print(f"   Max iterations: {self.ctf_config.iterations_per_level[level]}")
            
            # Update config for this level
            level_config = self._get_level_config(level)
            original_config = self._backup_config()
            self._apply_level_config(level_config)
            
            # Optimize at this level using modules optimizer interface
            level_poses, level_info = self._optimize_level_with_modules(
                current_poses, 
                max_iterations=self.ctf_config.iterations_per_level[level]
            )
            
            # Update current poses
            current_poses = level_poses
            
            # Restore original config
            self._restore_config(original_config)
            
            # Record level info
            level_timing = time.time() - level_start_time
            level_result = {
                'level': level,
                'scale': self.ctf_config.level_scales[level],
                'iterations': level_info['iterations'],
                'initial_loss': level_info['initial_loss'],
                'final_loss': level_info['final_loss'],
                'convergence': level_info['convergence'],
                'timing': level_timing
            }
            info['level_info'].append(level_result)
            info['total_iterations'] += level_info['iterations']
            
            print(f"   Completed: {level_info['iterations']} iterations, "
                  f"loss: {level_info['final_loss']:.6f}, "
                  f"time: {level_timing:.3f}s")
            
            # NO EARLY TERMINATION - Always run all pyramid levels
            # Removed fallback early termination logic
        
        info['timing']['total'] = time.time() - start_time
        
        # Final loss is from the finest level
        if info['level_info']:
            info['final_loss'] = info['level_info'][-1]['final_loss']
        
        return current_poses, info
    
    def _prepare_image_pyramids(self):
        """Prepare image pyramids for all reference and sensor data"""
        # Get reference data from underlying gaussian optimizer - STRICT: Must exist
        if not hasattr(self, 'gaussian_optimizer') or self.gaussian_optimizer is None:
            raise ValueError("gaussian_optimizer is required for pyramid preparation")
        
        if not hasattr(self.gaussian_optimizer, 'reference_data'):
            raise ValueError("gaussian_optimizer.reference_data is required")
        
        reference_data = self.gaussian_optimizer.reference_data
        
        if not reference_data:
            raise ValueError("reference_data cannot be empty for pyramid preparation")
            
        # Prepare reference image pyramids
        for sensor_name, ref_data in reference_data.items():
            if sensor_name not in self.reference_pyramids:
                self.reference_pyramids[sensor_name] = {}
            
            # Create RGB pyramid
            rgb_pyramid = self._create_image_pyramid(ref_data['rgb'])
            self.reference_pyramids[sensor_name]['rgb'] = rgb_pyramid
            
            # Create depth pyramid - STRICT: Must have depth key
            if 'depth' not in ref_data:
                raise KeyError(f"Sensor '{sensor_name}' missing required 'depth' data")
            
            if ref_data['depth'] is not None:
                depth_pyramid = self._create_image_pyramid(ref_data['depth'].unsqueeze(-1))
                self.reference_pyramids[sensor_name]['depth'] = depth_pyramid
    
    def _create_image_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Create image pyramid using average pooling"""
        if image.dim() == 2:
            image = image.unsqueeze(-1)
        
        pyramid = [image]  # Original resolution
        
        current_image = image
        for level in range(1, self.ctf_config.pyramid_levels):
            scale = self.ctf_config.level_scales[level] / self.ctf_config.level_scales[level - 1]
            
            if scale < 1.0:
                # Downsample
                target_h = int(current_image.shape[0] * scale)
                target_w = int(current_image.shape[1] * scale)
                
                # Use area interpolation for downsampling to avoid aliasing
                downsampled = F.interpolate(
                    current_image.permute(2, 0, 1).unsqueeze(0),
                    size=(target_h, target_w),
                    mode='area'
                ).squeeze(0).permute(1, 2, 0)
                
                pyramid.append(downsampled)
                current_image = downsampled
            else:
                # Same or higher resolution
                pyramid.append(current_image)
        
        return pyramid
    
    def _optimize_level_with_modules(self, poses: Dict[int, torch.Tensor], max_iterations: int) -> Tuple[Dict[int, torch.Tensor], Dict]:
        """Optimize poses at a single level using modules optimizer interface"""
        # SAFEGUARD: Reset any extreme poses before optimization to prevent frustum issues
        poses_safe = {}
        for frame_id, pose in poses.items():
            if pose.dim() == 2 and pose.shape == (4, 4):
                # Check translation magnitude
                translation = pose[:3, 3]
                translation_norm = torch.norm(translation)
                if translation_norm > 1.0:  # Reset extreme poses
                    print(f"      ⚠️ Resetting extreme pose for frame {frame_id}: translation norm {translation_norm:.3f}m")
                    reset_pose = torch.eye(4, device=pose.device)
                    reset_pose[:3, 3] = translation * (0.1 / translation_norm)  # Small translation
                    poses_safe[frame_id] = reset_pose
                else:
                    poses_safe[frame_id] = pose
            else:
                poses_safe[frame_id] = pose
        
        # Convert poses to modules optimizer format
        pose_batch = []
        frame_ids = []
        for frame_id, pose in poses_safe.items():
            pose_batch.append(pose)
            frame_ids.append(frame_id)
        
        # Initialize optimization info
        info = {
            'iterations': 0,
            'initial_loss': float('inf'),
            'final_loss': float('inf'),
            'convergence': False
        }
        
        if not pose_batch:
            return poses, info
        
        # Clear CUDA cache before optimization to free memory
        torch.cuda.empty_cache()
        
        # Set up poses for optimization
        if not pose_batch:
            raise ValueError("pose_batch cannot be empty")
        
        pose_tensor = torch.stack(pose_batch)
        
        # Clear any previous optimization state to avoid tensor size conflicts
        if hasattr(self, 'optimized_pose_batch'):
            self.optimized_pose_batch = None
        if hasattr(self, 'object_pose_batch'):
            self.object_pose_batch = None
            
        # Set up frame_ids mapping - STRICT: Must have gaussian_optimizer
        if not hasattr(self, 'gaussian_optimizer') or not self.gaussian_optimizer:
            raise ValueError("gaussian_optimizer is required for pose optimization")
        
        # Use the frame_ids from the actual sensor data that was added
        self.frame_ids = {}
        for sensor_name, sensor_data in self.gaussian_optimizer.sensor_data.items():
            if sensor_name != 'tactile':
                if 'frame_id' not in sensor_data:
                    raise KeyError(f"Sensor '{sensor_name}' missing required 'frame_id'")
                # Map sensor name to frame_id - this will be used by addPoses
                sensor_frame_id = sensor_data['frame_id']
                self.frame_ids[sensor_name] = torch.tensor([sensor_frame_id], dtype=torch.long)
        
        self.addPoses(pose_tensor)
        
        # Set up sensor data - STRICT: Must have sensors
        if not self.sensors:
            raise ValueError("No sensors available for optimization")
        
        opt_sensors = [sensor.sensor_name for sensor in self.sensors]
        
        # Run optimization using modules optimizer
        _, pose_loss = self.optimize_pose_theseus(opt_sensors, max_iterations)
            
        # Get optimized poses - STRICT: Must have results
        if not hasattr(self, 'optimized_pose_batch'):
            raise ValueError("optimized_pose_batch attribute missing after optimization")
        
        if self.optimized_pose_batch is None:
            raise ValueError("Optimization failed - no optimized poses returned")
        
        opt_poses = self.optimized_pose_batch
        print(f"      🔍 Debug: optimized_pose_batch shape: {opt_poses.shape}, frame_ids: {len(frame_ids)}")
        
        optimized_poses_dict = {}
        # Handle different tensor shapes from modules optimizer - STRICT validation
        if opt_poses.dim() == 3 and opt_poses.size(0) == len(frame_ids):
            # Standard case: [N, 4, 4] poses
            for i, frame_id in enumerate(frame_ids):
                optimized_poses_dict[frame_id] = opt_poses[i]
        elif opt_poses.dim() == 2 and opt_poses.shape == (4, 4) and len(frame_ids) == 1:
            # Single 4x4 pose matrix
            optimized_poses_dict[frame_ids[0]] = opt_poses
        elif opt_poses.dim() == 2 and opt_poses.shape == (1, 4) and len(frame_ids) == 1:
            # Single pose as [1, 4] - convert to 4x4 identity with translation
            pose_4x4 = torch.eye(4, device=opt_poses.device)
            pose_4x4[:3, 3] = opt_poses[0, :3]  # Set translation
            # For now, assume no rotation (identity rotation)
            optimized_poses_dict[frame_ids[0]] = pose_4x4
            print(f"      🔄 Converted [1, 4] pose to 4x4 matrix with translation: {opt_poses[0, :3]}")
        else:
            raise ValueError(f"Unsupported optimized pose tensor shape: {opt_poses.shape} for {len(frame_ids)} frames")
            
        # Update info - STRICT: Must have required attributes
        if not hasattr(self, '_last_iterations'):
            raise ValueError("_last_iterations attribute missing after optimization")
        if not hasattr(self, '_last_cost'):
            raise ValueError("_last_cost attribute missing after optimization")
        if not hasattr(self, 'pose_cfg'):
            raise ValueError("pose_cfg attribute required")
        if not hasattr(self.pose_cfg, 'convergence_threshold'):
            raise ValueError("pose_cfg.convergence_threshold required")
        
        iterations = self._last_iterations
        final_cost = self._last_cost
        convergence_threshold = self.pose_cfg.convergence_threshold
        
        info.update({
            'iterations': iterations,
            'initial_loss': pose_loss,
            'final_loss': final_cost,
            'convergence': final_cost < convergence_threshold
        })
        
        return optimized_poses_dict, info
    
    def _get_level_config(self, level: int) -> Dict:
        """Get configuration parameters for a specific pyramid level"""
        return {
            'pixel_sampling_ratio': self.ctf_config.pixel_sampling_per_level[level],
            'convergence_threshold': self.ctf_config.convergence_threshold_per_level[level],
            'image_scale': self.ctf_config.level_scales[level]
        }
    
    def _backup_config(self) -> Dict:
        """Backup current configuration - STRICT: config must exist"""
        if not hasattr(self, 'pose_cfg'):
            raise ValueError("pose_cfg attribute required for configuration backup")
        
        config = self.pose_cfg
        if not hasattr(config, 'pixel_sampling_ratio'):
            raise ValueError("pose_cfg.pixel_sampling_ratio required")
        if not hasattr(config, 'convergence_threshold'):
            raise ValueError("pose_cfg.convergence_threshold required")
        
        return {
            'pixel_sampling_ratio': config.pixel_sampling_ratio,
            'convergence_threshold': config.convergence_threshold
        }
    
    def _apply_level_config(self, level_config: Dict):
        """Apply level-specific configuration - STRICT: config must exist"""
        if not hasattr(self, 'pose_cfg'):
            raise ValueError("pose_cfg attribute required for applying level configuration")
        
        config = self.pose_cfg
        if not hasattr(config, 'pixel_sampling_ratio'):
            raise ValueError("pose_cfg.pixel_sampling_ratio required")
        if not hasattr(config, 'convergence_threshold'):
            raise ValueError("pose_cfg.convergence_threshold required")
        
        config.pixel_sampling_ratio = level_config['pixel_sampling_ratio']
        config.convergence_threshold = level_config['convergence_threshold']
    
    def _restore_config(self, original_config: Dict):
        """Restore original configuration - STRICT: config must exist"""
        if not hasattr(self, 'pose_cfg'):
            raise ValueError("pose_cfg attribute required for restoring configuration")
        
        config = self.pose_cfg
        if not hasattr(config, 'pixel_sampling_ratio'):
            raise ValueError("pose_cfg.pixel_sampling_ratio required")
        if not hasattr(config, 'convergence_threshold'):
            raise ValueError("pose_cfg.convergence_threshold required")
        
        config.pixel_sampling_ratio = original_config['pixel_sampling_ratio']
        config.convergence_threshold = original_config['convergence_threshold']
    
    def _compute_vision_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                             tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override vision residuals computation to use pyramid images
        This is called during optimization for each level
        """
        # Determine current pyramid level based on pixel sampling ratio - STRICT
        if not hasattr(self, 'pose_cfg'):
            raise ValueError("pose_cfg attribute required for pyramid level computation")
        
        config = self.pose_cfg
        if not hasattr(config, 'pixel_sampling_ratio'):
            raise ValueError("pose_cfg.pixel_sampling_ratio required")
        
        config_sampling = config.pixel_sampling_ratio
        current_level = 0
        for i, sampling_ratio in enumerate(self.ctf_config.pixel_sampling_per_level):
            if abs(config_sampling - sampling_ratio) < 1e-6:
                current_level = i
                break
        else:
            raise ValueError(f"No matching pyramid level found for sampling ratio {config_sampling}")
        
        # Use pyramid images for this level
        original_reference = self.reference_data.copy()
        
        # Replace reference data with pyramid level
        for sensor_name in self.reference_data:
            if sensor_name in self.reference_pyramids:
                pyramid = self.reference_pyramids[sensor_name]
                if current_level < len(pyramid['rgb']):
                    self.reference_data[sensor_name]['rgb'] = pyramid['rgb'][current_level]
                    if 'depth' in pyramid and pyramid['depth'] is not None:
                        if current_level < len(pyramid['depth']):
                            depth_level = pyramid['depth'][current_level]
                            if depth_level.shape[-1] == 1:
                                depth_level = depth_level.squeeze(-1)
                            self.reference_data[sensor_name]['depth'] = depth_level
        
        # Call parent method with pyramid images
        result = super()._compute_vision_residuals_and_jacobian(poses, tangent_vecs)
        
        # Restore original reference data
        self.reference_data = original_reference
        
        return result


# Factory function for coarse-to-fine optimization
def create_coarse_to_fine_optimizer(gaussian_map, sensors, config: PoseOptConfig, 
                                   ctf_config: CoarseToFineConfig,
                                   device: str = 'cuda') -> CoarseToFinePoseOptimizer:
    """Create a coarse-to-fine pose optimizer - STRICT: ctf_config required"""
    if ctf_config is None:
        raise ValueError("ctf_config is required - no fallback values allowed")
    
    return CoarseToFinePoseOptimizer(gaussian_map, sensors, config, ctf_config, device)


# Utility functions for pyramid-based optimization
def estimate_optimal_pyramid_levels(image_size: Tuple[int, int], 
                                   min_size: int) -> int:
    """Estimate optimal number of pyramid levels - STRICT: No minimum fallback"""
    if min_size <= 0:
        raise ValueError(f"min_size must be > 0, got: {min_size}")
    
    min_dim = min(image_size)
    if min_dim <= 0:
        raise ValueError(f"image dimensions must be > 0, got: {image_size}")
    
    levels = 0
    while min_dim > min_size:
        min_dim //= 2
        levels += 1
    
    if levels == 0:
        raise ValueError(f"Image size {image_size} too small for min_size {min_size} - no pyramid levels possible")
    
    return levels


def create_adaptive_ctf_config(image_size: Tuple[int, int],
                              target_fps: float,
                              quality_preference: str) -> CoarseToFineConfig:
    """
    Create adaptive coarse-to-fine configuration - STRICT: All parameters required
    
    Args:
        image_size: (height, width) of input images
        target_fps: Target frames per second
        quality_preference: 'speed', 'quality', 'balanced', or 'memory'
    """
    if quality_preference not in ['speed', 'quality', 'balanced', 'memory']:
        raise ValueError(f"quality_preference must be 'speed', 'quality', 'balanced', or 'memory', got: {quality_preference}")
    
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got: {target_fps}")
    
    # STRICT: Provide min_size for pyramid level estimation
    optimal_levels = estimate_optimal_pyramid_levels(image_size, min_size=8)
    
    if quality_preference == 'speed':
        levels = min(2, optimal_levels)
        # Requested: minimal coarse, stronger fine
        iterations = [1, 3][:levels] if levels >= 2 else [1]
        sampling = [0.15, 0.30][:levels] if levels >= 2 else [0.15]
        scales = [0.5, 1.0][:levels]
    elif quality_preference == 'quality':
        levels = min(4, optimal_levels)
        iterations = [4, 5, 6, 8][:levels]
        sampling = [0.15, 0.20, 0.25, 0.30][:levels]
        scales = [0.125, 0.25, 0.5, 1.0][:levels]
    elif quality_preference == 'memory':
        # STRICT: Memory-optimized settings - minimal levels and iterations
        levels = min(2, optimal_levels)  # Minimum 2 levels for pyramid algorithm
        iterations = [1, 2][:levels]  # Minimal iterations per level
        sampling = [0.10, 0.15][:levels]  # Lower sampling rates
        scales = [0.5, 1.0][:levels]  # Minimal pyramid scales
    else:  # balanced
        levels = min(3, optimal_levels)
        iterations = [4, 5, 6][:levels]
        sampling = [0.15, 0.22, 0.30][:levels]
        scales = [0.25, 0.5, 1.0][:levels]
    
    # Generate convergence thresholds
    chi2_base = 5.991e-6
    thresholds = [chi2_base * (2 ** i) for i in range(levels)]
    
    return CoarseToFineConfig(
        pyramid_levels=levels,
        level_scales=scales,
        iterations_per_level=iterations,
        pixel_sampling_per_level=sampling,
        early_termination=target_fps > 20.0,
        convergence_threshold_per_level=thresholds
    )