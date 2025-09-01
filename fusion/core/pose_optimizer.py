"""
Pose Optimizer using Gaussian Splatting.
Provides a drop-in replacement for Theseus-based pose optimization with a compatible interface.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time

from gaussian.core.gaussian_pose_optimizer import GaussianPoseOptimizer, PoseOptConfig
from gaussian.core.gaussian_field import ObjectGaussianMap
from gaussian.render.rasterizer import CameraParams


class PoseOptimizer(nn.Module):
    """
    Drop-in replacement PoseOptimizer using Gaussian Splatting.
    Maintains the same interface but uses rasterizer gradients instead of Theseus.
    """
    
    def __init__(self, sensors, cfg, train_mode, device):
        super(PoseOptimizer, self).__init__()
        
        # Store configuration compatible with existing interface
        self.pose_cfg = cfg
        self.sensors = sensors
        self.train_mode = train_mode
        self.device = device
        
        # Convert external config to GaussianSplatting config
        gs_config = self._convert_config(cfg)
        
        # Initialize the core Gaussian Splatting pose optimizer
        # Will be set when Gaussian map is available
        self.gaussian_optimizer = None
        self.gs_config = gs_config
        
        # Maintain interface compatibility
        self.pose_history = {}
        self.sensor_pose_batch = {}
        self.depth_batch = {}
        self.frame_ids = {}
        self.all_frame_ids = None
        self.optimized_pose_batch = None
        
        # Extract weights from config - STRICT: No fallbacks
        if not hasattr(cfg, 'w_vision'):
            raise ValueError("cfg.w_vision is required")
        if not hasattr(cfg, 'w_tactile'):
            raise ValueError("cfg.w_tactile is required")
        if not hasattr(cfg, 'second_order'):
            raise ValueError("cfg.second_order is required")
        if not hasattr(cfg.second_order, 'lm_iters'):
            raise ValueError("cfg.second_order.lm_iters is required")
        if not hasattr(cfg.second_order, 'num_iters'):
            raise ValueError("cfg.second_order.num_iters is required")
        
        self.w_vision = cfg.w_vision
        self.w_tactile = cfg.w_tactile
        self.lm_iters = cfg.second_order.lm_iters
        self.num_iters = cfg.second_order.num_iters
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Initialized Gaussian Splatting Pose Optimizer")
        logger.info(f"Training mode: {train_mode}")
        logger.info(f"Vision weight: {self.w_vision}, Tactile weight: {self.w_tactile}")
        logger.info(f"Max iterations: {self.num_iters}")
    
    def _convert_config(self, external_cfg) -> PoseOptConfig:
        """Convert external config to GaussianSplatting PoseOptConfig - STRICT: No fallbacks"""
        if external_cfg is None:
            raise ValueError("external_cfg cannot be None")
        
        config = PoseOptConfig()  # Uses industry-standard defaults
        
        # Extract optimization parameters - STRICT: Required attributes
        if not hasattr(external_cfg, 'second_order'):
            raise ValueError("external_cfg.second_order is required")
        
        second_order = external_cfg.second_order
        if not hasattr(second_order, 'num_iters'):
            raise ValueError("external_cfg.second_order.num_iters is required")
        if not hasattr(second_order, 'lm_damping'):
            raise ValueError("external_cfg.second_order.lm_damping is required")
        if not hasattr(second_order, 'icp_fitness'):
            raise ValueError("external_cfg.second_order.icp_fitness is required")
        if not hasattr(second_order, 'icp_inlier_rmse'):
            raise ValueError("external_cfg.second_order.icp_inlier_rmse is required")
        
        # Apply industry standards with strict validation
        config.max_iterations = max(20, second_order.num_iters)
        config.lm_damping_init = second_order.lm_damping
        config.icp_fitness_threshold = min(0.4, second_order.icp_fitness)
        config.icp_inlier_rmse_threshold = min(0.008, second_order.icp_inlier_rmse)
        
        # Extract cost weights - STRICT: Required attributes
        if not hasattr(external_cfg, 'w_vision'):
            raise ValueError("external_cfg.w_vision is required")
        if not hasattr(external_cfg, 'w_tactile'):
            raise ValueError("external_cfg.w_tactile is required")
        
        config.w_vision = external_cfg.w_vision
        config.w_tactile = external_cfg.w_tactile
        
        # Regularization
        if hasattr(external_cfg, 'second_order') and hasattr(external_cfg.second_order, 'reg_w'):
            config.w_regularization = external_cfg.second_order.reg_w
        
        return config
    
    def addVariables(self, depth_batch, sensor_pose_batch, frame_ids, sensor_name):
        """Add sensor variables (maintaining neuralfeels interface)"""
        # Convert to SE(3) format if needed
        if sensor_pose_batch.shape[-1] == 4 and sensor_pose_batch.shape[-2] == 3:
            # Convert [N, 3, 4] to [N, 4, 4]
            se3_poses = torch.zeros(sensor_pose_batch.shape[0], 4, 4, device=sensor_pose_batch.device)
            se3_poses[:, :3, :] = sensor_pose_batch
            se3_poses[:, 3, 3] = 1.0
            sensor_pose_batch = se3_poses
        
        self.sensor_pose_batch[sensor_name] = sensor_pose_batch
        self.depth_batch[sensor_name] = depth_batch
        self.frame_ids[sensor_name] = torch.tensor(frame_ids, device=depth_batch.device)
        
        # Reset optimization state
        self.optimized_pose_batch = None
        self.object_pcd, self.frame_pcd = None, None
    
    def addPointCloud(self, object_pcd, frame_pcd):
        """Add point clouds for ICP constraint"""
        self.object_pcd = object_pcd
        self.frame_pcd = frame_pcd
        
        if self.gaussian_optimizer is not None:
            self.gaussian_optimizer.add_point_clouds(object_pcd, frame_pcd)
    
    def addPoses(self, object_pose_batch):
        """Add object poses for optimization"""
        # Convert to SE(3) format if needed
        if object_pose_batch.shape[-1] == 4 and object_pose_batch.shape[-2] == 3:
            # Convert [N, 3, 4] to [N, 4, 4]
            se3_poses = torch.zeros(object_pose_batch.shape[0], 4, 4, device=object_pose_batch.device)
            se3_poses[:, :3, :] = object_pose_batch
            se3_poses[:, 3, 3] = 1.0
            object_pose_batch = se3_poses
        
        self.object_pose_batch = object_pose_batch
        
        # Get all frame IDs (handle empty frame_ids case)
        if self.frame_ids:
            self.all_frame_ids, _ = torch.sort(
                torch.unique(torch.cat(list(self.frame_ids.values())))
            )
        else:
            # Default frame IDs if none provided
            self.all_frame_ids = torch.arange(object_pose_batch.shape[0], dtype=torch.long)
    
    def addSDF(self, frozen_sdf):
        """Add SDF for constraint (compatibility only - uses MDS instead)"""
        # Note: This is for neuralfeels compatibility only
        # We don't actually use SDF - our Gaussian optimizer uses Mahalanobis distance
        self.frozen_sdf_map = frozen_sdf
        import logging
        logging.getLogger(__name__).warning(
            "SDF interface used for compatibility, but Gaussian optimizer uses Mahalanobis distance-based surface constraints instead"
        )
    
    def set_gaussian_map(self, gaussian_map: ObjectGaussianMap):
        """Set the Gaussian map for optimization"""
        self.gaussian_map = gaussian_map
        
        # Initialize the core optimizer
        self.gaussian_optimizer = GaussianPoseOptimizer(
            gaussian_map=gaussian_map,
            sensors=self.sensors,
            config=self.gs_config,
            device=self.device
        )
        
        # Add point clouds if available
        if hasattr(self, 'object_pcd') and hasattr(self, 'frame_pcd'):
            if self.object_pcd is not None and self.frame_pcd is not None:
                self.gaussian_optimizer.add_point_clouds(self.object_pcd, self.frame_pcd)
    
    def solve(self):
        """
        Solve pose optimization (compatibility method)
        Returns optimized poses in batch format
        """
        if self.gaussian_optimizer is None:
            raise RuntimeError("Gaussian map not set! Call set_gaussian_map() first")
        
        if not hasattr(self, 'object_pose_batch') or self.object_pose_batch is None:
            raise RuntimeError("No poses to optimize! Call addPoses() first")
        
        # Use all available sensor names for optimization  
        opt_sensors = [sensor.sensor_name for sensor in self.sensors] if self.sensors else []
        
        # Strict: max_iterations must be specified
        if not hasattr(self.pose_cfg, 'max_iterations'):
            raise ValueError("pose_cfg.max_iterations is required")
        
        num_iters = self.pose_cfg.max_iterations
        
        # Run optimization using the existing method
        _, pose_loss = self.optimize_pose_theseus(opt_sensors, num_iters)
        
        # Store results for later access
        self._last_cost = pose_loss
        self._last_iterations = num_iters
        
        return self.optimized_pose_batch
    
    def optimize_pose_theseus(self, opt_sensors, num_iters):
        """
        Main optimization function (maintains neuralfeels interface name)
        Uses Gaussian Splatting optimization instead of Theseus
        """
        if self.gaussian_optimizer is None:
            raise RuntimeError("Gaussian map not set! Call set_gaussian_map() first")
        
        start_time = time.time()
        
        # Prepare sensor data for optimization
        self._prepare_sensor_data(opt_sensors)
        
        # Convert object poses to frame-indexed dictionary
        poses_dict = {}
        for i, frame_id in enumerate(self.all_frame_ids):
            poses_dict[frame_id.item()] = self.object_pose_batch[i]
        
        # Run optimization
        optimized_poses, info = self.gaussian_optimizer.optimize_poses(
            object_poses=poses_dict,
            max_iterations=num_iters
        )
        
        # Convert back to batch format
        optimized_batch = []
        for frame_id in self.all_frame_ids:
            optimized_batch.append(optimized_poses[frame_id.item()])
        
        self.optimized_pose_batch = torch.stack(optimized_batch, dim=0)
        
        # Extract pose loss - STRICT: Must have required keys
        if 'final_loss' not in info:
            raise KeyError("info must contain 'final_loss' key")
        if 'iterations' not in info:
            raise KeyError("info must contain 'iterations' key")
        if 'convergence' not in info:
            raise KeyError("info must contain 'convergence' key")
        
        pose_loss = info['final_loss']
        
        end_time = time.time()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Gaussian Splatting pose optimization completed:")
        logger.info(f"Iterations: {info['iterations']}")
        logger.info(f"Time: {end_time - start_time:.3f}s")
        logger.info(f"Final loss: {pose_loss:.6f}")
        logger.info(f"Converged: {info['convergence']}")
        
        # Expose iteration and cost info for upstream coarse-to-fine wrapper
        # Strict: require iterations in info; otherwise fall back to requested iterations
        try:
            self._last_iterations = int(info.get('iterations', num_iters))
        except Exception:
            self._last_iterations = num_iters
        self._last_cost = float(pose_loss)
        
        # Return sample points (for visualization compatibility)
        sample_pts = self._get_sample_points() if hasattr(self.pose_cfg, 'show_samples') and self.pose_cfg.show_samples else None
        
        return sample_pts, pose_loss
    
    def _prepare_sensor_data(self, opt_sensors: List[str]):
        """Prepare sensor data for Gaussian Splatting optimization"""
        for sensor in self.sensors:
            sensor_name = sensor.sensor_name
            if sensor_name not in opt_sensors:
                continue
            
            if sensor_name not in self.sensor_pose_batch or sensor_name not in self.depth_batch:
                continue
            
            # Get sensor data
            depth_batch = self.depth_batch[sensor_name]
            pose_batch = self.sensor_pose_batch[sensor_name]
            frame_ids_tensor = self.frame_ids[sensor_name]
            
            # Skip if no valid depth data
            if torch.nansum(depth_batch) == 0:
                continue
            
            # Create camera parameters - STRICT: Must have intrinsics
            if not hasattr(sensor, 'fx'):
                raise ValueError(f"Sensor '{sensor.sensor_name}' missing required 'fx' parameter")
            if not hasattr(sensor, 'fy'):
                raise ValueError(f"Sensor '{sensor.sensor_name}' missing required 'fy' parameter")
            if not hasattr(sensor, 'cx'):
                raise ValueError(f"Sensor '{sensor.sensor_name}' missing required 'cx' parameter")
            if not hasattr(sensor, 'cy'):
                raise ValueError(f"Sensor '{sensor.sensor_name}' missing required 'cy' parameter")
            
            camera_params = CameraParams(
                fx=sensor.fx,
                fy=sensor.fy,
                cx=sensor.cx,
                cy=sensor.cy,
                width=640,
                height=480
            )

            # Add depth-only sensor data for each frame when RGB is unavailable
            for i, frame_id in enumerate(frame_ids_tensor):
                if i >= len(depth_batch) or i >= len(pose_batch):
                    continue

                # If vision terms are enabled but no RGB available, skip adding this sensor - STRICT
                if not hasattr(self.pose_cfg, 'w_vision'):
                    raise ValueError("pose_cfg.w_vision is required")
                
                if self.pose_cfg.w_vision > 0.0:
                    continue

                self.gaussian_optimizer.add_sensor_data(
                    sensor_name=sensor_name,
                    rgb=torch.zeros(1, 1, 3, device=depth_batch.device),  # unused when w_vision == 0
                    depth=depth_batch[i],
                    camera_params=camera_params,
                    T_world_cam=pose_batch[i],
                    frame_id=frame_id.item()
                )
                # Reference depth for depth-only optimization
                if i == 0:
                    self.gaussian_optimizer.set_reference_data(
                        sensor_name=sensor_name,
                        ref_rgb=torch.zeros(1, 1, 3, device=depth_batch.device),
                        ref_depth=depth_batch[i]
                    )
            
            # Tactile data integration occurs in dedicated tactile pipeline
    
    def _get_sample_points(self):
        """Get sample points for visualization from the Gaussian map"""
        # Return some sample points from the Gaussian map
        if hasattr(self, 'gaussian_map') and self.gaussian_map is not None:
            positions = self.gaussian_map.positions
            if len(positions) > 0:
                return positions.cpu().numpy()
        return None
    
    def getOptimizedPoses(self, matrix=False):
        """Get optimized poses (maintaining neuralfeels interface)"""
        assert self.optimized_pose_batch is not None, "Must optimize first"
        
        if matrix:
            return self.optimized_pose_batch, self.all_frame_ids
        else:
            # Convert to [N, 3, 4] format for compatibility
            pose_34 = self.optimized_pose_batch[:, :3, :]
            return pose_34, self.all_frame_ids
    
    # Timer functions for compatibility
    def startTimer(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        self.start_mem = torch.cuda.max_memory_allocated() / 1048576
        self.start_event.record()

    def stopTimer(self):
        self.end_event.record()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.end_mem = torch.cuda.max_memory_allocated() / 1048576
        forward_mem = self.end_mem - self.start_mem
        forward_time = self.start_event.elapsed_time(self.end_event)
        return forward_time, forward_mem


# Additional utility functions for pose optimization
def create_pose_optimizer(sensors, config, train_mode='object', device='cuda'):
    """Factory function to create pose optimizer"""
    return PoseOptimizer(sensors, config, train_mode, device)


def optimize_poses_with_gaussian_splatting(gaussian_map: ObjectGaussianMap, 
                                         sensors: List[Any],
                                         sensor_data: Dict[str, Any],
                                         initial_poses: Dict[int, torch.Tensor],
                                         config: Optional[PoseOptConfig] = None) -> Tuple[Dict[int, torch.Tensor], Dict]:
    """
    High-level function for pose optimization with Gaussian Splatting
    
    Args:
        gaussian_map: The Gaussian representation of the scene
        sensors: List of sensor objects
        sensor_data: Dictionary with sensor measurements
        initial_poses: Initial pose estimates
        config: Optimization configuration
        
    Returns:
        optimized_poses: Dictionary of optimized poses
        optimization_info: Statistics and debug information
    """
    if config is None:
        raise ValueError("config is required - no fallback configuration allowed")
    
    # Create optimizer
    optimizer = GaussianPoseOptimizer(gaussian_map, sensors, config)
    
    # Add sensor data
    for sensor_name, data in sensor_data.items():
        if 'rgb' in data and 'depth' in data:
            optimizer.add_sensor_data(
                sensor_name=sensor_name,
                rgb=data['rgb'],
                depth=data['depth'],
                camera_params=data['camera_params'],
                T_world_cam=data['T_world_cam'],
                frame_id=data['frame_id']
            )
            
            if 'ref_rgb' in data:
                if 'ref_depth' not in data:
                    raise KeyError(f"Sensor '{sensor_name}' has ref_rgb but missing required ref_depth")
                
                optimizer.set_reference_data(
                    sensor_name=sensor_name,
                    ref_rgb=data['ref_rgb'],
                    ref_depth=data['ref_depth']
                )
        
        if 'tactile_points' in data:
            optimizer.add_tactile_data(
                sensor_name=sensor_name,
                contact_points=data['tactile_points'],
                T_world_sensor=data['T_world_sensor'],
                frame_id=data['frame_id']
            )
    
    # Run optimization
    optimized_poses, info = optimizer.optimize_poses(initial_poses)
    
    return optimized_poses, info