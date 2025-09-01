"""
Dataset initialization utilities for GaussianFeels

Consolidated dataset initialization patterns to eliminate duplication.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path


def validate_dataset_structure(root_dir: str, required_subdirs: List[str]) -> bool:
    """
    Validate dataset directory structure.
    
    Args:
        root_dir: Root dataset directory
        required_subdirs: List of required subdirectories
    
    Returns:
        bool: True if valid structure
    
    Raises:
        RuntimeError: If structure is invalid
    """
    if not os.path.exists(root_dir):
        raise RuntimeError(f"Dataset root directory does not exist: {root_dir}")
    
    for subdir in required_subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            raise RuntimeError(f"Required dataset subdirectory missing: {subdir_path}")
    
    return True


def resolve_camera_name(root_dir: str, 
                       camera_subdir: str = "realsense", 
                       camera_name: Optional[str] = None,
                       enforce_single_camera: bool = True,
                       preferred_camera: str = "front-left") -> str:
    """
    Resolve camera name from dataset structure with fallback logic.
    
    Args:
        root_dir: Dataset root directory
        camera_subdir: Name of camera subdirectory (default: "realsense")
        camera_name: Explicit camera name (if provided)
        enforce_single_camera: Whether to enforce single camera usage
        preferred_camera: Preferred camera if multiple found
    
    Returns:
        str: Resolved camera name
    
    Raises:
        RuntimeError: If camera resolution fails
    """
    if camera_name is not None:
        return camera_name
    
    camera_root = os.path.join(root_dir, camera_subdir)
    if not os.path.isdir(camera_root):
        raise RuntimeError(f"Camera directory not found: {camera_root}")
    
    cameras = [d for d in os.listdir(camera_root) 
               if os.path.isdir(os.path.join(camera_root, d))]
    
    if len(cameras) == 0:
        raise RuntimeError(f"No cameras found in {camera_root}")
    
    if len(cameras) == 1:
        return cameras[0]
    
    if len(cameras) > 1:
        if enforce_single_camera:
            if preferred_camera in cameras:
                print(f"Multiple cameras found {cameras}. Using {preferred_camera} camera.")
                return preferred_camera
            else:
                raise RuntimeError(
                    f"Multiple cameras found {cameras}. {preferred_camera} camera "
                    f"required but not available. Specify camera_name explicitly."
                )
        else:
            raise RuntimeError(
                f"Multiple cameras found {cameras}. Specify camera_name explicitly "
                f"for single-camera dataset, or use multi-camera pipeline."
            )
    
    return cameras[0]


def get_file_list(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get sorted list of files with specific extensions.
    
    Args:
        directory: Directory to scan
        extensions: List of file extensions (default: ['.jpg', '.png'])
    
    Returns:
        List[str]: Sorted list of filenames
    """
    if extensions is None:
        extensions = ['.jpg', '.png']
    
    if not os.path.isdir(directory):
        return []
    
    files = []
    for f in os.listdir(directory):
        if any(f.lower().endswith(ext.lower()) for ext in extensions):
            files.append(f)
    
    return sorted(files)


def create_dataset_config_validator(required_keys: List[str]) -> callable:
    """
    Create a configuration validator function.
    
    Args:
        required_keys: List of required configuration keys
    
    Returns:
        callable: Validator function
    """
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate dataset configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            bool: True if valid
        
        Raises:
            ValueError: If validation fails
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        return True
    
    return validate_config