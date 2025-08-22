#!/usr/bin/env python3
"""
Constants for gaussianfeels pipeline
====================================

Centralized constants and configuration values used across modules.
This replaces scattered magic numbers with named constants.

Note: Camera intrinsics and depth ranges should now be read from data.pkl files
instead of using hardcoded defaults.
"""

import pickle
import os
import numpy as np
from typing import Dict, Tuple, Optional, List

# Tactile processing constants
CONTACT_THRESH = 1e-4  # meters - threshold for contact detection
DEFAULT_TACTILE_DEPTH_RANGE = (-0.01, 0.01)  # meters - valid depth range for tactile sensors

# Tactile depth clamping ranges
TACTILE_FLIP_MODE_CLAMP_RANGE = (0.0, 0.03)   # 0mm to 30mm indentation (flip mode) - no floor clipping
TACTILE_DIRECT_MODE_CLAMP_RANGE = (0.0, 0.01)  # 0mm to 10mm distance (direct mode) - no floor clipping

# Gaussian fusion constants  
DEFAULT_K_SIGMA = 3.0  # Standard deviations for influence zone
DEFAULT_LAMBDA_POS_BASE = 1.0  # Base position loss weight
DEFAULT_LAMBDA_COV_BASE = 0.5  # Base covariance loss weight
DEFAULT_LAMBDA_REG = 1e-6  # Regularization weight
DEFAULT_EPSILON = 1e-8  # Numerical stability floor

# Camera processing constants - now read from data.pkl
# DEFAULT_DEPTH_RANGE = (0.1, 2.0)  # Removed: should be read from data.pkl
# DEFAULT_FX = 640.0  # Removed: should be read from data.pkl intrinsics
# DEFAULT_FY = 480.0  # Removed: should be read from data.pkl intrinsics
# DEFAULT_CX = 320.0  # Removed: should be read from data.pkl intrinsics
# DEFAULT_CY = 240.0  # Removed: should be read from data.pkl intrinsics

# Processing limits
# DEFAULT_MAX_FRAMES = 50  # Removed: frame limit should be dynamic based on available data
DEFAULT_POINT_REDUCTION_RATIO = 0.5  # For memory optimization
DEFAULT_VOXEL_SIZE = 0.0003  # 0.3mm voxel size for downsampling

# File extensions and patterns
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']
SUPPORTED_DEPTH_FORMATS = ['.png', '.npy', '.npz']
PLY_EXTENSION = '.ply'

# Rendering and visualization
DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480
DEFAULT_FPS = 30

# Threading and performance
DEFAULT_NUM_THREADS = 4
DEFAULT_BATCH_SIZE = 1000

# Error thresholds
MAX_POSE_DELTA_THRESHOLD = 1.0  # Maximum acceptable pose delta
MAX_POINT_SURFACE_RMSE = 2.0  # mm - maximum point-to-surface RMSE

# Memory limits removed for unbounded processing approach
# These limits were added for initial development but removed for research quality
# MAX_MEMORY_USAGE_MB = 158.0  # MB for active set - REMOVED: Unbounded memory approach
# MAX_GAUSSIAN_COUNT = 300000  # Maximum Gaussians - REMOVED: Unbounded complexity approach

# Color coding for visualization
CAMERA_POINT_COLOR = [0.0, 0.0, 1.0]  # Blue for camera points
TACTILE_POINT_COLOR = [1.0, 0.0, 0.0]  # Red for tactile points
FUSED_POINT_COLOR = [0.0, 1.0, 0.0]  # Green for fused points


def list_realsense_cameras(data_pkl_path: str) -> List[str]:
    """
    Return a sorted list of available RealSense camera names from data.pkl.
    """
    if not os.path.exists(data_pkl_path):
        raise FileNotFoundError(f"Data file not found: {data_pkl_path}")
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
    if 'realsense' not in data or not isinstance(data['realsense'], dict):
        raise KeyError("No realsense data found in data.pkl")
    return sorted(list(data['realsense'].keys()))


def load_camera_intrinsics_from_data(data_pkl_path: str, camera_type: str = "realsense", camera_name: str = None) -> Optional[Dict]:
    """
    Load camera intrinsics from data.pkl file.
    
    Args:
        data_pkl_path: Path to the data.pkl file
        camera_type: Type of camera ("realsense" or "digit_info")
        camera_name: Name of the camera ("front-left" for realsense)
        
    Returns:
        Dictionary containing intrinsics: {"fx": float, "fy": float, "cx": float, "cy": float, "w": int, "h": int}
        Raises exception if file not found or data not available
    """
    if not os.path.exists(data_pkl_path):
        raise FileNotFoundError(f"Data file not found: {data_pkl_path}")
        
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
            
        if camera_type == "realsense" and camera_type in data:
            # Auto-pick first camera if none provided
            if camera_name is None:
                cams = list_realsense_cameras(data_pkl_path)
                camera_name = cams[0] if cams else None
            if camera_name and camera_name in data[camera_type] and "intrinsics" in data[camera_type][camera_name]:
                intrinsics = data[camera_type][camera_name]["intrinsics"]
                # Convert tensor values to float if needed
                result = {}
                for key, value in intrinsics.items():
                    if hasattr(value, 'item'):  # Handle tensor values
                        result[key] = float(value.item())
                    else:
                        result[key] = float(value) if key in ['fx', 'fy', 'cx', 'cy'] else int(value)
                return result
                
        elif camera_type == "digit_info" and camera_type in data:
            if "intrinsics" in data[camera_type]:
                intrinsics = data[camera_type]["intrinsics"]
                result = {}
                for key, value in intrinsics.items():
                    if hasattr(value, 'item'):  # Handle tensor/numpy values
                        result[key] = float(value.item())
                    else:
                        result[key] = float(value) if key in ['fx', 'fy', 'cx', 'cy'] else int(value)
                return result
                
    # If data not found, raise exception
    raise KeyError(f"No intrinsics found for camera_type={camera_type}, camera_name={camera_name}")


def get_camera_intrinsics(data_pkl_path: str, camera_type: str = "realsense", camera_name: str = None) -> Tuple[float, float, float, float]:
    """
    Get camera intrinsics (fx, fy, cx, cy) from data.pkl - requires valid data.
    
    Args:
        data_pkl_path: Path to the data.pkl file
        camera_type: Type of camera ("realsense" or "digit_info")
        camera_name: Name of the camera ("front-left" for realsense)
        
    Returns:
        Tuple of (fx, fy, cx, cy)
        
    Raises:
        RuntimeError: If intrinsics cannot be loaded from data file
    """
    intrinsics = load_camera_intrinsics_from_data(data_pkl_path, camera_type, camera_name)
    
    if intrinsics:
        return (intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'])
    else:
        raise RuntimeError(f"Could not load camera intrinsics from {data_pkl_path}. Valid data.pkl required.")


def get_depth_range_from_data(data_pkl_path: str, camera_type: str = "realsense", camera_name: str = None) -> Tuple[float, float]:
    """
    Get depth range from data.pkl or return sensible defaults.
    
    Args:
        data_pkl_path: Path to the data.pkl file
        camera_type: Type of camera ("realsense" or "digit_info")
        camera_name: Name of the camera ("front-left" for realsense)
        
    Returns:
        Tuple of (min_depth, max_depth) in meters
    """
    if not os.path.exists(data_pkl_path):
        raise FileNotFoundError(f"Data file not found: {data_pkl_path}")
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # STRICT: Read actual depth ranges from data.pkl - no hardcoded fallbacks
    if camera_type not in data:
        raise KeyError(f"No {camera_type} data found in data.pkl")
    
    camera_data = data[camera_type]
    if 'depth_range' not in camera_data:
        raise KeyError(f"Camera type '{camera_type}' missing required 'depth_range' field in data.pkl")
    
    depth_range = camera_data['depth_range']
    if not isinstance(depth_range, (list, tuple)) or len(depth_range) != 2:
        raise ValueError(f"depth_range for {camera_type} must be a 2-element list/tuple, got: {depth_range}")
    
    return tuple(depth_range)


class TemporalAnalyzer:
    """
    Comprehensive temporal analysis using time field from data.pkl
    """
    
    def __init__(self, data_pkl_path: str):
        self.data_pkl_path = data_pkl_path
        self.time_data = None
        self.frame_count = 0
        self.duration = 0.0
        self.frame_rate = 0.0
        self.load_temporal_data()
    
    def load_temporal_data(self):
        """Load temporal data from data.pkl"""
        try:
            with open(self.data_pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'time' not in data:
                raise KeyError("No 'time' field found in data.pkl")
            
            self.time_data = data['time']
            self.frame_count = len(self.time_data)
            self.duration = float(self.time_data[-1] - self.time_data[0])
            self.frame_rate = self.frame_count / self.duration if self.duration > 0 else 0.0
            
        except (FileNotFoundError, KeyError, pickle.PickleError) as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load temporal data from {self.data_pkl_path}: {e}")
            raise
    
    def get_frame_rate(self) -> float:
        """Get average frame rate in Hz"""
        return self.frame_rate
    
    def get_duration(self) -> float:
        """Get total sequence duration in seconds"""
        return self.duration
    
    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return self.frame_count
    
    def get_timestamp(self, frame_idx: int) -> float:
        """Get timestamp for specific frame"""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.frame_count-1}]")
        return float(self.time_data[frame_idx])
    
    def get_frame_interval(self, frame_idx: int) -> float:
        """Get time interval to next frame"""
        if frame_idx >= self.frame_count - 1:
            return 0.0
        return float(self.time_data[frame_idx + 1] - self.time_data[frame_idx])
    
    def validate_temporal_consistency(self, tolerance: float = 0.1) -> Dict[str, any]:
        """Validate temporal consistency and detect anomalies"""
        if self.frame_count < 2:
            return {"valid": False, "error": "Insufficient frames for validation"}
        
        import numpy as np
        intervals = np.diff(self.time_data)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Detect frame drops (intervals significantly larger than mean)
        anomalies = np.where(intervals > mean_interval + 3 * std_interval)[0]
        
        # Check for consistent frame rate
        expected_interval = 1.0 / self.frame_rate
        consistent = np.abs(mean_interval - expected_interval) < tolerance
        
        return {
            "valid": len(anomalies) == 0 and consistent,
            "frame_rate": self.frame_rate,
            "mean_interval": float(mean_interval),
            "std_interval": float(std_interval),
            "anomalies": anomalies.tolist(),
            "frame_drops": len(anomalies),
            "consistent_rate": consistent
        }
    
    def get_frame_at_time(self, target_time: float) -> int:
        """Get frame index closest to target time"""
        if target_time < self.time_data[0] or target_time > self.time_data[-1]:
            raise ValueError(f"Target time {target_time} outside sequence range [{self.time_data[0]}, {self.time_data[-1]}]")
        
        import numpy as np
        return int(np.argmin(np.abs(self.time_data - target_time)))
    
    def interpolate_poses_at_time(self, poses, target_time: float):
        """Interpolate pose at specific time using temporal data"""
        import numpy as np
        
        if poses.shape[0] != self.frame_count:
            raise ValueError(f"Pose array length {poses.shape[0]} doesn't match frame count {self.frame_count}")
        
        if target_time <= self.time_data[0]:
            return poses[0]
        if target_time >= self.time_data[-1]:
            return poses[-1]
        
        # Find surrounding frames
        idx_after = np.searchsorted(self.time_data, target_time)
        idx_before = idx_after - 1
        
        # Linear interpolation factor
        t_before = self.time_data[idx_before]
        t_after = self.time_data[idx_after]
        alpha = (target_time - t_before) / (t_after - t_before)
        
        # Interpolate pose (simple linear interpolation)
        return (1 - alpha) * poses[idx_before] + alpha * poses[idx_after]


def get_object_name_from_data(data_pkl_path: str) -> Optional[str]:
    """
    Extract object name from data.pkl
    
    Args:
        data_pkl_path: Path to the data.pkl file
        
    Returns:
        Object name string
    """
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'object' not in data or 'name' not in data['object']:
        raise KeyError("No object name found in data.pkl")
    return data['object']['name']


def get_object_mesh_path_from_data(data_pkl_path: str) -> Optional[str]:
    """
    Extract object mesh path from data.pkl
    
    Args:
        data_pkl_path: Path to the data.pkl file
        
    Returns:
        Object mesh path string
    """
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'object' not in data or 'mesh' not in data['object']:
        raise KeyError("No object mesh path found in data.pkl")
    mesh_path = data['object']['mesh']
    # Convert to relative path if possible
    from pathlib import Path
    if mesh_path.startswith('/home/'):
        # Try to make relative to project root
        try:
            mesh_path = str(Path(mesh_path).relative_to(Path.cwd()))
        except ValueError:
            pass  # Keep absolute path if can't make relative
    return mesh_path


def get_depth_scale_from_data(data_pkl_path: str, camera_type: str = "realsense", camera_name: str = None) -> float:
    """
    Get depth scale from data.pkl - STRICT: No fallback values
    
    Args:
        data_pkl_path: Path to the data.pkl file
        camera_type: Type of camera ("realsense" or "digit_info")
        camera_name: Name of the camera ("front-left" for realsense)
        
    Returns:
        Depth scale value
    """
    try:
        with open(data_pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if camera_type == "digit_info" and camera_type in data:
            if "depth_scale" in data[camera_type]:
                return float(data[camera_type]["depth_scale"])
        
        elif camera_type == "realsense" and camera_type in data:
            if camera_name is None:
                cams = sorted(list(data[camera_type].keys()))
                camera_name = cams[0] if cams else None
            if camera_name and camera_name in data[camera_type] and "depth_scale" in data[camera_type][camera_name]:
                return float(data[camera_type][camera_name]["depth_scale"])
                
    except (FileNotFoundError, KeyError, pickle.PickleError) as e:
        raise RuntimeError(f"Could not load depth scale from {data_pkl_path}: {e}. Valid data.pkl required.")


class DataPklManager:
    """
    Comprehensive manager for data.pkl operations
    Replaces scattered hardcoded values with centralized data loading
    """
    
    def __init__(self, data_pkl_path: str):
        from pathlib import Path
        self.data_pkl_path = Path(data_pkl_path)
        self.data = None
        self.temporal_analyzer = None
        self._load_data()
    
    def _load_data(self):
        """Load and validate data.pkl"""
        if not os.path.exists(self.data_pkl_path):
            raise FileNotFoundError(f"Data file not found: {self.data_pkl_path}")
        
        try:
            with open(self.data_pkl_path, 'rb') as f:
                self.data = pickle.load(f)
            
            # Validate required structure
            required_keys = ['object', 'allegro', 'digit_info', 'realsense', 'time']
            missing_keys = [key for key in required_keys if key not in self.data]
            
            if missing_keys:
                raise KeyError(f"Required keys missing in data.pkl: {missing_keys}")
            
            self.temporal_analyzer = TemporalAnalyzer(str(self.data_pkl_path))
            
        except pickle.PickleError as e:
            raise ValueError(f"Invalid pickle file {self.data_pkl_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {self.data_pkl_path}: {e}")
    
    @property
    def object_name(self) -> str:
        """Get object name from data"""
        return self.data['object']['name']
    
    @property
    def object_mesh_path(self) -> str:
        """Get object mesh path from data"""
        return self.data['object']['mesh']
    
    @property
    def frame_count(self) -> int:
        """Get total frame count"""
        return self.temporal_analyzer.get_frame_count()
    
    def get_digit_calibration(self) -> Dict:
        """
        Get DIGIT sensor calibration data for tactile processing
        
        Returns:
            Dictionary containing:
            - depth_scale: Scaling factor for TouchVIT heightmaps 
            - cam_dist: Camera distance from gel surface (meters)
            - intrinsics: Camera intrinsic parameters
        """
        if 'digit_info' not in self.data:
            raise KeyError("digit_info not found in data.pkl")
            
        return self.data['digit_info']
    
    @property
    def frame_rate(self) -> float:
        """Get sequence frame rate"""
        return self.temporal_analyzer.get_frame_rate()
    
    @property
    def duration(self) -> float:
        """Get sequence duration"""
        return self.temporal_analyzer.get_duration()
    
    def get_realsense_intrinsics(self, camera_name: str = None) -> Dict:
        """Get RealSense camera intrinsics"""
        if camera_name is None:
            cams = sorted(list(self.data['realsense'].keys()))
            camera_name = cams[0] if cams else None
        return self.data['realsense'][camera_name]['intrinsics']
    
    def get_digit_intrinsics(self) -> Dict:
        """Get digit sensor intrinsics"""
        return self.data['digit_info']['intrinsics']
    
    def get_object_poses(self):
        """Get object poses over time"""
        return self.data['object']['pose']
    
    def get_allegro_joint_states(self):
        """Get Allegro joint states over time"""
        return self.data['allegro']['joint_state']
    
    def get_allegro_base_pose(self):
        """Get Allegro base pose"""
        return self.data['allegro']['base_pose']
    
    def get_finger_poses(self, frame_idx: int):
        """Get finger poses for specific frame"""
        finger_poses_list = self.data['allegro']['finger_poses']
        if frame_idx >= len(finger_poses_list):
            raise IndexError(f"Frame {frame_idx} not available, max frame: {len(finger_poses_list)-1}")
        return finger_poses_list[frame_idx]  # Shape: (4, 4, 4)
    
    def get_realsense_poses(self, camera_name: str = None):
        """Get camera poses over time"""
        if camera_name is None:
            cams = sorted(list(self.data['realsense'].keys()))
            camera_name = cams[0] if cams else None
        return self.data['realsense'][camera_name]['pose']
    
    def get_depth_scales(self) -> Dict[str, float]:
        """Get all available depth scales"""
        result = {}
        if 'digit_info' in self.data and 'depth_scale' in self.data['digit_info']:
            result['digit_info'] = float(self.data['digit_info']['depth_scale'])
        if 'realsense' in self.data:
            for cam, val in self.data['realsense'].items():
                if isinstance(val, dict) and 'depth_scale' in val:
                    result[f'realsense/{cam}'] = float(val['depth_scale'])
        return result
    
    def get_tactile_rgb_paths(self, frame_idx: int) -> Dict[str, str]:
        """Get RGB tactile image paths for all fingers at a specific frame"""
        base_path = self.data_pkl_path.parent / "allegro"
        rgb_paths = {}
        
        fingers = ['thumb', 'index', 'middle', 'ring']
        for finger in fingers:
            rgb_path = base_path / finger / "image" / f"{frame_idx}.jpg"
            if rgb_path.exists():
                rgb_paths[finger] = str(rgb_path)
        
        return rgb_paths
    
    def get_finger_poses(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Get finger poses for all fingers at a specific frame"""
        finger_poses = self.data['allegro']['finger_poses'][frame_idx]
        return {
            'thumb': finger_poses[0],
            'index': finger_poses[1], 
            'middle': finger_poses[2],
            'ring': finger_poses[3]
        }
    
    def get_object_poses(self, frame_idx: int) -> Dict:
        """Get all poses (object, allegro) for given frame"""
        try:
            poses = {}
            
            # Get object pose
            if 'object' in self.data and 'pose' in self.data['object']:
                object_poses = np.array(self.data['object']['pose'])
                if frame_idx < len(object_poses):
                    poses['object'] = object_poses[frame_idx]
                else:
                    import logging
                    logging.getLogger(__name__).warning(f"Frame {frame_idx} beyond object pose data ({len(object_poses)} frames)")
            
            # Get allegro hand poses using forward kinematics
            if 'allegro' in self.data:
                allegro_data = self.data['allegro']
                if 'joint_state' in allegro_data:
                    joint_states = allegro_data['joint_state']
                    if frame_idx < len(joint_states):
                        # Use forward kinematics to get finger poses
                        finger_poses = self._compute_finger_poses_from_joints(joint_states[frame_idx])
                        if finger_poses:
                            poses['allegro'] = finger_poses
                elif isinstance(allegro_data, np.ndarray):
                    # Handle array data structures
                    if frame_idx < len(allegro_data):
                        poses['allegro'] = allegro_data[frame_idx]
            
            return poses if poses else None
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to get poses for frame {frame_idx}: {e}")
            return None
    
    def _compute_finger_poses_from_joints(self, joint_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute finger poses from joint states using forward kinematics"""
        try:
            # Use proper package imports (require pip install -e .)
            # Import Allegro class for forward kinematics
            from tactile.gaussianfeels.modules.allegro import Allegro
            
            # Create temporary Allegro instance with base pose from data
            if not hasattr(self, '_allegro_fk'):
                # Use base pose from data.pkl - required for proper FK
                if 'allegro' in self.data and 'base_pose' in self.data['allegro']:
                    base_pose = self.data['allegro']['base_pose']
                else:
                    raise RuntimeError("No allegro base_pose found in data.pkl. Valid calibration data required.")
                
                # Create minimal Allegro instance for FK computation
                try:
                    # Convert 4x4 matrix to pose config format expected by Allegro
                    from scipy.spatial.transform import Rotation as R
                    
                    translation = base_pose[:3, 3]
                    rotation_matrix = base_pose[:3, :3]
                    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
                    
                    base_pose_config = {
                        'translation': {'x': float(translation[0]), 'y': float(translation[1]), 'z': float(translation[2])},
                        'rotation': {'x': float(quat[0]), 'y': float(quat[1]), 'z': float(quat[2]), 'w': float(quat[3])}
                    }
                    
                    self._allegro_fk = Allegro(base_pose=base_pose_config, device='cpu')
                    import logging
                    logging.getLogger(__name__).info(f"Created Allegro FK instance with base pose shape: {base_pose.shape}")
                except Exception as allegro_error:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to create Allegro instance: {allegro_error}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Compute forward kinematics (with caching to avoid repeated computation)
            joint_state_key = tuple(joint_state.flatten()) if hasattr(joint_state, 'flatten') else tuple(joint_state)
            if not hasattr(self, '_fk_cache'):
                self._fk_cache = {}
            
            if joint_state_key not in self._fk_cache:
                finger_poses = self._allegro_fk.get_fk(joint_state=joint_state)
                self._fk_cache[joint_state_key] = finger_poses
            else:
                finger_poses = self._fk_cache[joint_state_key]
            
            # Removed verbose logging for performance
            
            return finger_poses
            
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Import failed for Allegro forward kinematics: {e}")
            return None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to compute finger poses from joints: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_digit_intrinsics(self) -> Dict[str, float]:
        """Get DIGIT sensor intrinsics for tactile processing - STRICT: No fallback values"""
        # STRICT: digit_info and intrinsics must exist in data.pkl
        if 'digit_info' not in self.data:
            raise KeyError("digit_info missing from data.pkl - required for DIGIT intrinsics")
        
        if 'intrinsics' not in self.data['digit_info']:
            raise KeyError("digit_info.intrinsics missing from data.pkl - required for DIGIT intrinsics")
        
        digit_intrinsics = self.data['digit_info']['intrinsics']
        
        # STRICT: All intrinsic parameters must be explicitly provided
        required_keys = ['fx', 'fy', 'cx', 'cy']
        for key in required_keys:
            if key not in digit_intrinsics:
                raise KeyError(f"DIGIT intrinsics missing required '{key}' parameter in data.pkl")
        
        return {
            'fx': float(digit_intrinsics['fx']),
            'fy': float(digit_intrinsics['fy']),  
            'cx': float(digit_intrinsics['cx']),
            'cy': float(digit_intrinsics['cy'])
        }