"""
World coordinate system transforms.
Implements standard camera extrinsics and world coordinate processing.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class WorldCoordinateTransforms:
    """
    World coordinate transform system.
    """
    
    def __init__(self):
        """Initialize world coordinate transform system"""
        self.world_origin = np.array([0.0, 0.0, 0.0])
        self.world_up = np.array([0.0, 0.0, 1.0])  # Z-up world coordinate system
        
        logger.info("Initialized world coordinate transforms")
    
    def get_standard_camera_extrinsics(self, 
                                     camera_position: np.ndarray,
                                     camera_target: np.ndarray,
                                     up_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get standard camera extrinsics transform.
        
        Args:
            camera_position: Camera position in world coordinates [3]
            camera_target: Target point to look at [3]
            up_vector: Up vector (default: world up)
            
        Returns:
            4x4 camera extrinsics matrix (world to camera transform)
        """
        if up_vector is None:
            up_vector = self.world_up
        
        # Standard camera coordinate frame construction
        forward = camera_target - camera_position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, up_vector)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Camera extrinsics matrix (world to camera)
        extrinsics = np.eye(4)
        extrinsics[:3, 0] = right
        extrinsics[:3, 1] = up
        extrinsics[:3, 2] = -forward  # Camera looks down negative Z
        extrinsics[:3, 3] = camera_position
        
        # Invert to get world-to-camera transform
        extrinsics = np.linalg.inv(extrinsics)
        
        logger.debug(f"Standard camera extrinsics computed for position {camera_position}")
        return extrinsics
    
    def hora_to_neural_transform(self) -> np.ndarray:
        """
        Get HORA to neural coordinate transform.
        
        Returns:
            4x4 transformation matrix
        """
        # Fixed HORA to neural transform
        hora_to_neural = np.array([
            [0.000000, -1.000000, 0.000000, 0.000021],
            [0.000000, 0.000000, 1.000000, -0.017545],
            [-1.000000, 0.000000, 0.000000, -0.002132],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ])
        
        return hora_to_neural
    
    def transform_points_to_world(self, points: np.ndarray, 
                                 transform_matrix: np.ndarray) -> np.ndarray:
        """
        Transform points to world coordinates.
        
        Args:
            points: Points in local coordinates [N, 3]
            transform_matrix: 4x4 transformation matrix to world
            
        Returns:
            Points in world coordinates [N, 3]
        """
        if points.shape[0] == 0:
            return points
        
        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])
        
        # Transform to world coordinates
        world_points_homo = (transform_matrix @ points_homo.T).T
        world_points = world_points_homo[:, :3]
        
        logger.debug(f"Transformed {len(points)} points to world coordinates")
        return world_points
    
    def transform_poses_to_world(self, poses: Dict[str, np.ndarray],
                               base_transform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform poses to world coordinates.
        
        Args:
            poses: Dictionary of poses in local coordinates
            base_transform: Base transformation to world coordinates
            
        Returns:
            Dictionary of poses in world coordinates
        """
        world_poses = {}
        
        for pose_name, local_pose in poses.items():
            # Transform pose to world coordinates
            world_pose = base_transform @ local_pose
            world_poses[pose_name] = world_pose
            
            logger.debug(f"Transformed {pose_name} pose to world coordinates")
        
        return world_poses
    
    def get_world_to_camera_transform(self, camera_pose: np.ndarray) -> np.ndarray:
        """
        Get world-to-camera transformation matrix.
        
        Args:
            camera_pose: Camera pose in world coordinates [4, 4]
            
        Returns:
            World-to-camera transformation matrix [4, 4]
        """
        # Camera pose is camera-to-world, invert for world-to-camera
        world_to_camera = np.linalg.inv(camera_pose)
        
        logger.debug("Computed world-to-camera transformation")
        return world_to_camera
    
    def project_world_points_to_image(self, world_points: np.ndarray,
                                    camera_intrinsics: np.ndarray,
                                    camera_extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project world points to image coordinates.
        
        Args:
            world_points: Points in world coordinates [N, 3]
            camera_intrinsics: Camera intrinsics matrix [3, 3]
            camera_extrinsics: Camera extrinsics matrix [4, 4]
            
        Returns:
            Tuple of (image_coordinates [N, 2], depths [N])
        """
        if world_points.shape[0] == 0:
            return np.array([]).reshape(0, 2), np.array([])
        
        # Transform to camera coordinates
        ones = np.ones((world_points.shape[0], 1))
        world_points_homo = np.hstack([world_points, ones])
        camera_points_homo = (camera_extrinsics @ world_points_homo.T).T
        camera_points = camera_points_homo[:, :3]
        
        # Project to image
        depths = camera_points[:, 2]
        valid_mask = depths > 1e-6  # Avoid division by zero
        
        image_coords = np.zeros((len(world_points), 2))
        if valid_mask.any():
            normalized_coords = camera_points[valid_mask, :2] / camera_points[valid_mask, 2:3]
            image_coords[valid_mask] = (camera_intrinsics[:2, :2] @ normalized_coords.T).T + camera_intrinsics[:2, 2]
        
        logger.debug(f"Projected {len(world_points)} world points to image")
        return image_coords, depths


class WorldCompatibleTransforms:
    """
    Coordinate transform utilities.
    """
    
    def __init__(self):
        """Initialize transforms"""
        self.world_transforms = WorldCoordinateTransforms()
        
        # Coordinate system parameters
        self.coordinate_system = "world"  # Not object-centric
        
        logger.info("Initialized world-compatible transforms")
    
    def convert_object_centric_to_world(self, 
                                      object_centric_data: Dict[str, np.ndarray],
                                      object_pose: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert object-centric data to world coordinates.
        
        Args:
            object_centric_data: Data in object-centric coordinates
            object_pose: Object pose in world coordinates [4, 4]
            
        Returns:
            Data converted to world coordinates
        """
        world_data = {}
        
        for key, data in object_centric_data.items():
            if isinstance(data, np.ndarray) and data.shape[-1] >= 3:
                # Transform point data to world coordinates
                if data.ndim == 2:  # Point array [N, 3+]
                    points = data[:, :3]
                    world_points = self.world_transforms.transform_points_to_world(
                        points, object_pose
                    )
                    if data.shape[1] > 3:
                        # Preserve additional dimensions
                        world_data[key] = np.hstack([world_points, data[:, 3:]])
                    else:
                        world_data[key] = world_points
                else:
                    world_data[key] = data  # Keep as-is for non-point data
            else:
                world_data[key] = data  # Keep non-spatial data as-is
        
        logger.info(f"Converted {len(object_centric_data)} data items to world coordinates")
        return world_data
    
    def get_world_coordinate_camera_setup(self, 
                                        camera_positions: np.ndarray,
                                        scene_center: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Set up cameras in world coordinate system.
        
        Args:
            camera_positions: Camera positions in world coordinates [N, 3]
            scene_center: Scene center point (default: origin)
            
        Returns:
            Dictionary with camera extrinsics matrices
        """
        if scene_center is None:
            scene_center = np.array([0.0, 0.0, 0.0])
        
        camera_extrinsics = {}
        
        for i, camera_pos in enumerate(camera_positions):
            extrinsics = self.world_transforms.get_standard_camera_extrinsics(
                camera_pos, scene_center
            )
            camera_extrinsics[f'camera_{i}'] = extrinsics
        
        logger.info(f"Set up {len(camera_positions)} cameras in world coordinate system")
        return camera_extrinsics