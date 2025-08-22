#!/usr/bin/env python3
"""
Frame Data Utilities for GaussianFeels Camera
Simple data structures for RGB-D frame handling
"""

import numpy as np
from typing import Optional, Dict, Any, Union


class FrameData:
    """
    Simple frame data container for RGB-D sensor data
    """
    
    def __init__(
        self,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        mask: Optional[Union[np.ndarray, Dict]] = None,
        T_WC: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        intrinsics: Optional[Dict[str, float]] = None,
        tactile_data: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize frame data
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W)
            mask: Segmentation mask or mask dictionary
            T_WC: Camera pose transform (4, 4)
            frame_idx: Frame index
            intrinsics: Camera intrinsic parameters
            tactile_data: Tactile sensor data if available
            **kwargs: Additional frame metadata
        """
        self.rgb = rgb
        self.depth = depth
        self.mask = mask
        self.T_WC = T_WC if T_WC is not None else np.eye(4)
        self.frame_idx = frame_idx
        self.intrinsics = intrinsics
        self.tactile_data = tactile_data
        
        # Store additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def has_rgb(self) -> bool:
        """Check if RGB data is available"""
        return self.rgb is not None
    
    @property
    def has_depth(self) -> bool:
        """Check if depth data is available"""
        return self.depth is not None
    
    @property
    def has_mask(self) -> bool:
        """Check if mask data is available"""
        return self.mask is not None
    
    @property
    def image_shape(self) -> Optional[tuple]:
        """Get image shape from RGB or depth"""
        if self.has_rgb:
            return self.rgb.shape[:2]  # (H, W)
        elif self.has_depth:
            return self.depth.shape
        return None
    
    def get_object_mask(self) -> Optional[np.ndarray]:
        """Get object segmentation mask"""
        if not self.has_mask:
            return None
        
        if isinstance(self.mask, dict):
            return self.mask.get('object_mask', None)
        else:
            return self.mask
    
    def get_tactile_masks(self) -> Dict[str, np.ndarray]:
        """Get tactile contact masks"""
        if not self.has_mask or not isinstance(self.mask, dict):
            return {}
        
        return self.mask.get('tactile_masks', {})
    
    def copy(self):
        """Create a copy of the frame data"""
        return FrameData(
            rgb=self.rgb.copy() if self.rgb is not None else None,
            depth=self.depth.copy() if self.depth is not None else None,
            mask=self.mask,  # Shallow copy for mask data
            T_WC=self.T_WC.copy() if self.T_WC is not None else None,
            frame_idx=self.frame_idx,
            intrinsics=self.intrinsics,
            tactile_data=self.tactile_data
        )


def create_identity_pose() -> np.ndarray:
    """Create identity pose matrix"""
    return np.eye(4, dtype=np.float32)


def pose_from_translation_rotation(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Create pose matrix from translation and rotation
    
    Args:
        translation: Translation vector (3,)
        rotation: Rotation matrix (3, 3) or quaternion (4,)
        
    Returns:
        Pose matrix (4, 4)
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = translation
    
    if rotation.shape == (3, 3):
        pose[:3, :3] = rotation
    elif rotation.shape == (4,):
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(rotation)
        pose[:3, :3] = r.as_matrix()
    else:
        raise ValueError(f"Invalid rotation shape: {rotation.shape}")
    
    return pose


def pose_from_config(pose_config: list) -> np.ndarray:
    """
    Create pose matrix from configuration list
    
    Args:
        pose_config: [x, y, z, qx, qy, qz, qw] or similar format
        
    Returns:
        Pose matrix (4, 4)
    """
    if len(pose_config) == 7:
        # [x, y, z, qx, qy, qz, qw]
        translation = np.array(pose_config[:3])
        quaternion = np.array(pose_config[3:])
        return pose_from_translation_rotation(translation, quaternion)
    elif len(pose_config) == 6:
        # [x, y, z, rx, ry, rz] (Euler angles)
        translation = np.array(pose_config[:3])
        euler_angles = np.array(pose_config[3:])
        from scipy.spatial.transform import Rotation as R
        rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
        return pose_from_translation_rotation(translation, rotation_matrix)
    else:
        # Default to identity
        return create_identity_pose()


# Export main classes and functions
__all__ = [
    'FrameData',
    'create_identity_pose',
    'pose_from_translation_rotation',
    'pose_from_config'
]