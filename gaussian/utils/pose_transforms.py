#!/usr/bin/env python3
"""
Pose Transformation Utilities for GaussianFeels Camera Pipeline
Implements proper coordinate transformations.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R


def transform_points_to_object_frame(points_cam: np.ndarray, T_world_obj: np.ndarray, T_world_cam: np.ndarray, 
                                   p_WO_W: np.ndarray = None) -> np.ndarray:
    """
    Transform points from camera frame to object frame using temporal tracking approach
    
    Args:
        points_cam: Points in camera coordinate frame (N, 3)
        T_world_obj: Object pose in world frame at frame time (4, 4) - transforms object→world
        T_world_cam: Camera pose in world frame (4, 4) - transforms camera→world
        p_WO_W: Reference object pose (4, 4) - canonical object frame. If None, uses T_world_obj
    
    Returns:
        points_obj: Points in canonical object coordinate frame (N, 3)
    """
    # Validate input
    if len(points_cam) == 0:
        return np.array([]).reshape(0, 3)
    
    # Ensure points_cam is 2D array with shape (N, 3)
    points_cam = np.asarray(points_cam)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must be (N, 3) array, got shape {points_cam.shape}")
    
    # Use reference pose if provided, otherwise use current frame object pose (backward compatibility)
    if p_WO_W is None:
        # Old approach for backward compatibility
        T_obj_world = np.linalg.inv(T_world_obj)
        T_obj_cam = T_obj_world @ T_world_cam
    else:
        # Temporal tracking approach
        # Step 1: Transform to object frame at frame time
        frame_obj_state = T_world_obj  # Object pose at frame time
        tf_pose = np.linalg.inv(frame_obj_state) @ T_world_cam
        
        # Step 2: Transform to canonical object frame
        T_obj_cam = p_WO_W @ tf_pose
    
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    
    # Apply transformation
    # First transpose to get (4, N), then multiply to get (4, N), then transpose back to (N, 4)
    points_transposed = points_homogeneous.T  # (4, N)
    transformed_transposed = T_obj_cam @ points_transposed  # (4, 4) @ (4, N) = (4, N)
    points_transformed = transformed_transposed.T  # (N, 4)
    
    # Validate transformation result
    if points_transformed.ndim != 2 or points_transformed.shape[1] != 4:
        raise ValueError(f"Transformation resulted in unexpected shape: {points_transformed.shape}")
    
    # Convert back to 3D coordinates with proper normalization
    w_coords = points_transformed[:, -1]
    # Avoid division by zero
    w_coords = np.where(np.abs(w_coords) < 1e-8, 1e-8, w_coords)
    points_obj = points_transformed[:, :3] / w_coords[:, None]
    
    return points_obj.astype(np.float32)


def select_optimal_keyframes(data: dict, max_frames: int = 8, rot_threshold_deg: float = 15, 
                           trans_threshold_m: float = 0.03) -> List[int]:
    """
    Select keyframes based on pose diversity for maximum surface coverage
    
    Args:
        data: Loaded data.pkl containing pose information
        max_frames: Maximum frames to select
        rot_threshold_deg: Minimum rotation change in degrees
        trans_threshold_m: Minimum translation change in meters
    
    Returns:
        List of frame indices providing optimal coverage
    """
    total_frames = len(data['object']['pose'])
    if total_frames == 0:
        return []
    
    keyframes = [0]  # Always include first frame
    
    if total_frames == 1 or max_frames == 1:
        return keyframes
    
    last_obj_pose = data['object']['pose'][0]
    # Choose first available camera if multi-camera present
    cam_keys = list(data['realsense'].keys())
    cam_key = cam_keys[0]
    last_cam_pose = data['realsense'][cam_key]['pose'][0]
    
    for i in range(1, total_frames):
        curr_obj_pose = data['object']['pose'][i]
        curr_cam_pose = data['realsense'][cam_key]['pose'][i]
        
        # Calculate relative motion in object-camera relationship
        last_rel_pose = np.linalg.inv(last_obj_pose) @ last_cam_pose
        curr_rel_pose = np.linalg.inv(curr_obj_pose) @ curr_cam_pose
        
        delta_pose = np.linalg.inv(last_rel_pose) @ curr_rel_pose
        
        # Extract rotation and translation changes
        delta_trans = np.linalg.norm(delta_pose[:3, 3])
        
        # Calculate rotation change using trace method (more stable)
        trace = np.trace(delta_pose[:3, :3])
        # Clamp trace to valid range to avoid numerical errors
        trace_clamped = np.clip((trace - 1) / 2, -1, 1)
        delta_rot_rad = np.arccos(trace_clamped)
        delta_rot_deg = np.degrees(delta_rot_rad)
        
        # Add frame if significant motion detected
        if delta_rot_deg > rot_threshold_deg or delta_trans > trans_threshold_m:
            keyframes.append(i)
            last_obj_pose = curr_obj_pose
            last_cam_pose = curr_cam_pose
            
            if len(keyframes) >= max_frames:
                break
    
    print(f"Selected {len(keyframes)} keyframes from {total_frames} total: {keyframes}")
    print(f"   Rotation threshold: {rot_threshold_deg}°, Translation threshold: {trans_threshold_m}m")
    
    return keyframes


def validate_pose_matrices(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> bool:
    """
    Validate that pose matrices are proper 4x4 transformation matrices
    
    Args:
        T_world_obj: Object pose matrix (4, 4)
        T_world_cam: Camera pose matrix (4, 4)
    
    Returns:
        True if matrices are valid, False otherwise
    """
    def is_valid_transform(T):
        if T.shape != (4, 4):
            return False
        
        # Check if bottom row is [0, 0, 0, 1]
        if not np.allclose(T[3, :], [0, 0, 0, 1]):
            return False
        
        # Check if rotation part is orthogonal
        R_part = T[:3, :3]
        should_be_identity = R_part @ R_part.T
        if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
            return False
        
        # Check if determinant is 1 (proper rotation, not reflection)
        if not np.isclose(np.linalg.det(R_part), 1.0, atol=1e-3):
            return False
        
        return True
    
    return is_valid_transform(T_world_obj) and is_valid_transform(T_world_cam)


def project_points_to_image(points_3d: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    Project 3D points back to 2D image coordinates for validation
    
    Args:
        points_3d: 3D points in camera frame (N, 3)
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
    
    Returns:
        2D pixel coordinates (N, 2)
    """
    if len(points_3d) == 0:
        return np.array([]).reshape(0, 2)
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Extract X, Y, Z coordinates
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    # Project to 2D using pinhole camera model
    # NOTE: Use negative X for RealSense optical frame consistency
    u = -X * fx / Z + cx  # Negative X for RealSense optical frame
    v = Y * fy / Z + cy
    
    return np.column_stack([u, v])


def compute_pose_diversity_score(keyframes: List[int], data: dict) -> float:
    """
    Compute a diversity score for the selected keyframes based on pose variation
    
    Args:
        keyframes: List of selected frame indices
        data: Dataset containing pose information
    
    Returns:
        Diversity score (higher = more diverse viewpoints)
    """
    if len(keyframes) < 2:
        return 0.0
    
    total_rotation = 0.0
    total_translation = 0.0
    
    for i in range(1, len(keyframes)):
        prev_frame = keyframes[i-1]
        curr_frame = keyframes[i]
        
        prev_obj_pose = data['object']['pose'][prev_frame]
        curr_obj_pose = data['object']['pose'][curr_frame]
        cam_keys = list(data['realsense'].keys())
        cam_key = cam_keys[0]
        prev_cam_pose = data['realsense'][cam_key]['pose'][prev_frame]
        curr_cam_pose = data['realsense'][cam_key]['pose'][curr_frame]
        
        # Calculate relative camera-object poses
        prev_rel = np.linalg.inv(prev_obj_pose) @ prev_cam_pose
        curr_rel = np.linalg.inv(curr_obj_pose) @ curr_cam_pose
        
        delta_pose = np.linalg.inv(prev_rel) @ curr_rel
        
        # Accumulate rotation and translation changes
        delta_trans = np.linalg.norm(delta_pose[:3, 3])
        trace_clamped = np.clip((np.trace(delta_pose[:3, :3]) - 1) / 2, -1, 1)
        delta_rot = np.arccos(trace_clamped)
        
        total_rotation += delta_rot
        total_translation += delta_trans
    
    # Combine rotation (radians) and translation (meters) into single score
    diversity_score = total_rotation + total_translation * 10  # Weight translation more
    
    return diversity_score


def analyze_pose_sequence(data: dict) -> dict:
    """
    Analyze the pose sequence to understand object and camera motion patterns
    
    Args:
        data: Dataset containing pose information
    
    Returns:
        Dictionary with analysis results
    """
    total_frames = len(data['object']['pose'])
    
    if total_frames < 2:
        return {'total_frames': total_frames, 'motion_analysis': 'insufficient_data'}
    
    obj_translations = []
    obj_rotations = []
    cam_translations = []
    cam_rotations = []
    
    for i in range(1, total_frames):
        # Object motion
        prev_obj = data['object']['pose'][i-1]
        curr_obj = data['object']['pose'][i]
        obj_delta = np.linalg.inv(prev_obj) @ curr_obj
        
        obj_trans = np.linalg.norm(obj_delta[:3, 3])
        trace_clamped = np.clip((np.trace(obj_delta[:3, :3]) - 1) / 2, -1, 1)
        obj_rot = np.degrees(np.arccos(trace_clamped))
        
        obj_translations.append(obj_trans)
        obj_rotations.append(obj_rot)
        
        # Camera motion  
        cam_keys = list(data['realsense'].keys())
        cam_key = cam_keys[0]
        prev_cam = data['realsense'][cam_key]['pose'][i-1]
        curr_cam = data['realsense'][cam_key]['pose'][i]
        cam_delta = np.linalg.inv(prev_cam) @ curr_cam
        
        cam_trans = np.linalg.norm(cam_delta[:3, 3])
        trace_clamped = np.clip((np.trace(cam_delta[:3, :3]) - 1) / 2, -1, 1)
        cam_rot = np.degrees(np.arccos(trace_clamped))
        
        cam_translations.append(cam_trans)
        cam_rotations.append(cam_rot)
    
    analysis = {
        'total_frames': total_frames,
        'object_motion': {
            'total_translation': np.sum(obj_translations),
            'max_translation_per_frame': np.max(obj_translations),
            'avg_translation_per_frame': np.mean(obj_translations),
            'total_rotation_deg': np.sum(obj_rotations),
            'max_rotation_per_frame': np.max(obj_rotations),
            'avg_rotation_per_frame': np.mean(obj_rotations)
        },
        'camera_motion': {
            'total_translation': np.sum(cam_translations),
            'max_translation_per_frame': np.max(cam_translations),
            'avg_translation_per_frame': np.mean(cam_translations),
            'total_rotation_deg': np.sum(cam_rotations),
            'max_rotation_per_frame': np.max(cam_rotations),
            'avg_rotation_per_frame': np.mean(cam_rotations)
        }
    }
    
    return analysis