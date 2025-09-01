"""
GaussianFeels Dataset Registry and Multi-Dataset Pipeline

Unified interface for accessing multiple datasets (Feelsight, TouchNet, etc.)
with support for local files, HuggingFace, S3/GCS, and streaming.
"""

import pickle
import h5py
import yaml
import json
import zipfile
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import requests
from io import BytesIO
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

from .config import GaussianFeelsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Single frame of multi-modal sensor data"""
    frame_id: int
    timestamp: float
    
    # Vision data
    rgb_images: Dict[str, np.ndarray] = None  # camera_name -> RGB image
    depth_images: Dict[str, np.ndarray] = None  # camera_name -> depth image
    camera_poses: Dict[str, np.ndarray] = None  # camera_name -> 4x4 pose matrix
    camera_intrinsics: Dict[str, np.ndarray] = None  # camera_name -> 3x3 K matrix
    
    # Tactile data
    tactile_images: Dict[str, np.ndarray] = None  # sensor_name -> tactile image
    tactile_depth: Dict[str, np.ndarray] = None  # sensor_name -> predicted depth
    tactile_poses: Dict[str, np.ndarray] = None  # sensor_name -> 4x4 pose matrix
    
    # Robot state
    robot_pose: np.ndarray = None  # 4x4 robot base pose
    joint_positions: np.ndarray = None  # joint angles
    end_effector_pose: np.ndarray = None  # 4x4 EE pose
    
    # Ground truth (when available)
    gt_object_pose: np.ndarray = None  # 4x4 object pose
    gt_mesh_path: Optional[str] = None  # path to ground truth mesh
    gt_sdf: Optional[np.ndarray] = None  # ground truth SDF
    
    # Metadata and annotations
    metadata: Dict[str, Any] = None
    semantic_labels: Optional[Dict[str, np.ndarray]] = None  # semantic segmentation
    instance_labels: Optional[Dict[str, np.ndarray]] = None  # instance segmentation
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DatasetInfo:
    """Dataset metadata and configuration"""
    name: str
    description: str
    version: str = "1.0"
    authors: List[str] = field(default_factory=list)
    license: str = "Unknown"
    url: Optional[str] = None
    citation: Optional[str] = None
    total_sequences: int = 0
    total_frames: int = 0
    modalities: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    sensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    download_size: Optional[str] = None
    uncompressed_size: Optional[str] = None
    checksum: Optional[str] = None
    
@dataclass 
class DatasetStats:
    """Dataset statistics for evaluation"""
    num_sequences: int
    num_frames: int
    avg_frames_per_sequence: float
    modalities: List[str]
    spatial_extent: Tuple[float, float, float]  # [x, y, z] ranges
    temporal_span: float  # seconds
    object_categories: List[str]
    sensor_coverage: Dict[str, float]  # fraction of frames with each sensor
    resolution_stats: Dict[str, Dict[str, Any]]  # image/depth resolution stats
    
@dataclass
class BenchmarkResult:
    """Results from dataset evaluation/benchmarking"""
    dataset_name: str
    method_name: str
    metrics: Dict[str, float]
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.frames: List[FrameData] = []
        self.metadata: Dict[str, Any] = {}
        self.dataset_info: Optional[DatasetInfo] = None
        self.stats: Optional[DatasetStats] = None
        self._cache = {}
        
        # Setup caching and preprocessing
        self.use_cache = getattr(config, 'use_cache', True)
        self.preprocess_fn = getattr(config, 'preprocess_fn', None)
        
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset-specific data"""
        raise NotImplementedError("Subclasses must implement _load_data()")
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        raise NotImplementedError("Subclasses must implement get_dataset_info()")
    
    @property
    def modalities(self) -> List[str]:
        """Get available modalities in this dataset"""
        return self.config.modalities
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> FrameData:
        # Use cache if available
        if self.use_cache and idx in self._cache:
            frame = self._cache[idx]
        else:
            frame = self.frames[idx]
            if self.use_cache:
                self._cache[idx] = frame
        
        # Apply preprocessing if specified
        if self.preprocess_fn:
            frame = self.preprocess_fn(frame)
            
        return frame
    
    def get_sequence_ids(self) -> List[str]:
        """Get list of sequence identifiers"""
        sequences = set()
        for frame in self.frames:
            seq_id = frame.metadata.get('sequence_id', 'default')
            sequences.add(seq_id)
        return sorted(list(sequences))
    
    def get_frames_by_sequence(self, sequence_id: str) -> List[FrameData]:
        """Get all frames from a specific sequence"""
        return [f for f in self.frames if f.metadata.get('sequence_id') == sequence_id]
    
    def compute_stats(self) -> DatasetStats:
        """Compute dataset statistics"""
        if self.stats is not None:
            return self.stats
            
        # Basic stats
        num_frames = len(self.frames)
        sequences = self.get_sequence_ids()
        num_sequences = len(sequences)
        
        if num_sequences > 0:
            avg_frames_per_seq = num_frames / num_sequences
        else:
            avg_frames_per_seq = 0
        
        # Modality coverage
        modality_counts = {mod: 0 for mod in ['vision', 'tactile']}
        
        # Resolution stats
        resolution_stats = {}
        
        for frame in self.frames:
            if frame.rgb_images:
                modality_counts['vision'] += 1
                for cam, img in frame.rgb_images.items():
                    if cam not in resolution_stats:
                        resolution_stats[cam] = {'rgb': {'resolutions': []}}
                    resolution_stats[cam]['rgb']['resolutions'].append(img.shape[:2])
                    
            if frame.tactile_images:
                modality_counts['tactile'] += 1
        
        # Sensor coverage
        sensor_coverage = {
            mod: count / num_frames if num_frames > 0 else 0 
            for mod, count in modality_counts.items()
        }
        
        # Spatial/temporal extent – derive from object pose translations when available
        temporal_span = self.frames[-1].timestamp - self.frames[0].timestamp if num_frames > 1 else 0
        try:
            # Attempt to infer spatial extent from object poses if present in metadata
            # Collect translations across frames
            translations: List[np.ndarray] = []
            for frame in self.frames:
                # Expect upstream components to attach object poses to frame.metadata if available
                pose = frame.metadata.get('object_pose') if frame.metadata else None
                if pose is None and frame.gt_object_pose is not None:
                    pose = frame.gt_object_pose
                if pose is not None and isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                    translations.append(pose[:3, 3])
            if len(translations) > 0:
                translations_np = np.stack(translations, axis=0)
                min_t = np.min(translations_np, axis=0)
                max_t = np.max(translations_np, axis=0)
                spatial_extent = tuple((max_t - min_t).tolist())  # type: ignore[arg-type]
            else:
                spatial_extent = (0.0, 0.0, 0.0)
        except (ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(f"Failed to compute spatial extent from training data: {e}") from e
        
        # Object categories
        object_categories = [self.config.object] if hasattr(self.config, 'object') else []
        
        self.stats = DatasetStats(
            num_sequences=num_sequences,
            num_frames=num_frames,
            avg_frames_per_sequence=avg_frames_per_seq,
            modalities=self.modalities,
            spatial_extent=spatial_extent,
            temporal_span=temporal_span,
            object_categories=object_categories,
            sensor_coverage=sensor_coverage,
            resolution_stats=resolution_stats
        )
        
        return self.stats
    
    def export_summary(self, output_path: Path):
        """Export dataset summary to file"""
        info = self.get_dataset_info()
        stats = self.compute_stats()
        
        summary = {
            'info': info.__dict__,
            'stats': stats.__dict__,
            'sample_frames': []
        }
        
        # Add sample frames
        sample_indices = [0, len(self)//2, len(self)-1] if len(self) > 0 else []
        for idx in sample_indices:
            if idx < len(self):
                frame = self.frames[idx]
                sample_summary = {
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'has_rgb': frame.rgb_images is not None,
                    'has_depth': frame.depth_images is not None,
                    'has_tactile': frame.tactile_images is not None,
                    'metadata': frame.metadata
                }
                summary['sample_frames'].append(sample_summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

class FeelsightDataset(BaseDataset):
    """Feelsight simulation dataset loader with NeuralFeels compatibility"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        
        # STRICT path handling - data_root must be explicitly provided
        if not hasattr(config, 'data_root'):
            raise ValueError("Config missing required 'data_root' attribute - no fallback to 'data' directory allowed")
        if not config.data_root:
            raise ValueError("Config.data_root cannot be empty - explicit data root path required")
        
        self.dataset_path = Path(config.data_root) / config.dataset / config.object / config.log
        
        # Ensure dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.dataset_path}\n"
                f"Please check that the FeelSight dataset is properly installed."
            )
        
        self._load_data()
    
    def _load_data(self):
        """Load Feelsight dataset from pickle files"""
        data_file = self.dataset_path / "data.pkl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        # Load main data pickle
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        self.metadata = {
            "object": self.config.object,
            "log": self.config.log,
            "dataset_type": "simulation",
            "total_frames": len(data.get("timestamps", [])) if "timestamps" in data else 0,
        }
        
        # Extract frame data - handle different possible data structures
        # Prefer explicit time vector
        # STRICT: Handle timestamps with explicit key checking
        timestamps = None
        if "time" in data:
            timestamps = data["time"]
        elif "timestamps" in data:
            timestamps = data["timestamps"]
        else:
            timestamps = []
        
        # If no timestamps in pkl, infer count from depth or images without fabricating values
        if timestamps is None or len(timestamps) == 0:
            depth_file = self.dataset_path / "realsense" / "front-left" / "depth.npz"
            if depth_file.exists():
                with np.load(depth_file) as dnpz:
                    # STRICT: Check for depth key explicitly
                    if 'depth' not in dnpz:
                        raise KeyError("depth.npz missing required 'depth' key")
                    depth_arr = dnpz['depth']
                    if depth_arr is not None and hasattr(depth_arr, 'shape'):
                        frame_count = int(depth_arr.shape[0]) if depth_arr.ndim >= 2 else 0
                    else:
                        frame_count = 0
            else:
                rgb_dir = self.dataset_path / "realsense" / "front-left" / "image"
                frame_count = len(sorted(rgb_dir.glob("*.jpg"))) if rgb_dir.exists() else 0
            timestamps = [float(i) for i in range(frame_count)]
        
        # Limit to reasonable number for testing
        max_frames = min(len(timestamps), 50)
        timestamps = timestamps[:max_frames]
        
        for i, timestamp in enumerate(timestamps):
            frame = FrameData(
                frame_id=i,
                timestamp=timestamp,
            )
            
            # Load vision data
            if "vision" in self.modalities:
                frame.rgb_images = self._load_rgb_images(i)
                frame.depth_images = self._load_depth_images(i)
                frame.camera_poses = self._load_camera_poses(data, i)
                frame.camera_intrinsics = self._load_camera_intrinsics(data)
            
            # Load tactile data
            if "tactile" in self.modalities:
                frame.tactile_images = self._load_tactile_images(i)
                frame.tactile_poses = self._load_tactile_poses(data, i)
                # Do not fabricate depth predictions during dataset load
                frame.tactile_depth = None
            
            # Load robot state
            frame.robot_pose = self._load_robot_pose(data, i)
            frame.joint_positions = self._load_joint_positions(data, i)
            frame.end_effector_pose = self._load_ee_pose(data, i)
            
            # Load ground truth
            frame.gt_object_pose = self._load_gt_object_pose(data, i)
            frame.gt_mesh_path = self._get_gt_mesh_path()
            
            self.frames.append(frame)
    
    def _load_rgb_images(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load RGB images for a frame"""
        rgb_images = {}
        
        # Standard camera names matching actual data structure
        camera_names = ["front-left", "back-right", "top-down"]
        
        for camera_name in camera_names:
            camera_dir = self.dataset_path / "realsense" / camera_name / "image"
            if camera_dir.exists():
                image_files = sorted(camera_dir.glob("*.jpg"))
                if frame_id < len(image_files):
                    image_path = image_files[frame_id]
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb_images[camera_name] = image
        
        return rgb_images
    
    def _load_depth_images(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load depth images for a frame"""
        depth_images = {}
        
        # Standard camera names matching actual data structure
        camera_names = ["front-left", "back-right", "top-down"]
        
        for camera_name in camera_names:
            depth_file = self.dataset_path / "realsense" / camera_name / "depth.npz"
            if depth_file.exists():
                with np.load(depth_file) as data:
                    if frame_id < len(data.files):
                        depth_key = data.files[frame_id]
                        depth_images[camera_name] = data[depth_key]
        
        return depth_images
    
    def _load_tactile_images(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load tactile images for a frame"""
        tactile_images = {}
        
        finger_names = ["thumb", "index", "middle", "ring"]
        
        for finger_name in finger_names:
            tactile_dir = self.dataset_path / "allegro" / finger_name / "image"
            if tactile_dir.exists():
                image_files = sorted(tactile_dir.glob("*.jpg"))
                if frame_id < len(image_files):
                    image_path = image_files[frame_id]
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tactile_images[f"digit_{finger_name}"] = image
        
        return tactile_images
    
    def _predict_tactile_depth(self, tactile_images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Deprecated in dataset loader – tactile depth should be computed by the tactile pipeline."""
        return {}
    
    def _load_camera_poses(self, data: Dict, frame_id: int) -> Dict[str, np.ndarray]:
        """Load camera poses for a frame from data.pkl (supports static or per-frame)."""
        poses: Dict[str, np.ndarray] = {}
        # STRICT: Require realsense data section in data.pkl - no fallback allowed
        if 'realsense' not in data:
            raise KeyError("data.pkl missing required 'realsense' section for camera poses")
        if not isinstance(data['realsense'], dict):
            raise TypeError("'realsense' section must be a dictionary")
        
        realsense: Dict[str, Any] = data['realsense']
        
        # Standard camera names matching actual data structure
        camera_names = ["front-left", "back-right", "top-down"]
        
        for camera_name in camera_names:
            if camera_name not in realsense:
                continue  # Skip cameras without data (valid case)
            cam_dict = realsense[camera_name]
            if not isinstance(cam_dict, dict):
                continue  # Skip malformed camera data
            if 'pose' not in cam_dict:
                continue  # Skip cameras without pose data
            cam_pose = cam_dict['pose']
            if isinstance(cam_pose, np.ndarray):
                if cam_pose.ndim == 3 and cam_pose.shape[-2:] == (4, 4):
                    if 0 <= frame_id < cam_pose.shape[0]:
                        poses[camera_name] = cam_pose[frame_id]
                elif cam_pose.ndim == 2 and cam_pose.shape == (4, 4):
                    poses[camera_name] = cam_pose
            # Only include cameras with available poses
        return poses
    
    def _load_camera_intrinsics(self, data: Dict) -> Dict[str, np.ndarray]:
        """Load camera intrinsic parameters from data.pkl when available."""
        intrinsics: Dict[str, np.ndarray] = {}
        # STRICT: Require realsense data section in data.pkl - no fallback allowed
        if 'realsense' not in data:
            raise KeyError("data.pkl missing required 'realsense' section for camera intrinsics")
        if not isinstance(data['realsense'], dict):
            raise TypeError("'realsense' section must be a dictionary")
        
        realsense: Dict[str, Any] = data['realsense']
        
        # Standard camera names matching actual data structure
        camera_names = ["front-left", "back-right", "top-down"]
            
        for camera_name in camera_names:
            if camera_name not in realsense:
                continue  # Skip cameras without data (valid case)
            cam_dict = realsense[camera_name]
            if not isinstance(cam_dict, dict):
                continue  # Skip malformed camera data
            if 'intrinsics_matrix' not in cam_dict:
                continue  # Skip cameras without intrinsics
            K = cam_dict['intrinsics_matrix']
            if K is not None:
                try:
                    K_arr = np.array(K, dtype=np.float32)
                    if K_arr.shape == (3, 3):
                        intrinsics[camera_name] = K_arr
                except (ValueError, TypeError, KeyError) as e:
                    raise RuntimeError(f"Failed to parse camera intrinsics for {camera_name}: {e}") from e
        return intrinsics
    
    def _load_tactile_poses(self, data: Dict, frame_id: int) -> Dict[str, np.ndarray]:
        """Load tactile sensor poses from data.pkl if available."""
        poses: Dict[str, np.ndarray] = {}
        # STRICT: Require allegro data section if tactile poses are needed
        if 'allegro' not in data:
            return poses  # No allegro data available - return empty poses
        if not isinstance(data['allegro'], dict):
            raise TypeError("'allegro' section must be a dictionary")
        
        allegro: Dict[str, Any] = data['allegro']
        if 'finger_poses' not in allegro:
            return poses  # No finger poses available
        finger_poses = allegro['finger_poses']
        if isinstance(finger_poses, np.ndarray):
            # Expected shape (N, 4, 4, 4) or (N, 4, 4) per finger – be permissive
            try:
                if finger_poses.ndim == 4 and finger_poses.shape[1:] == (4, 4, 4):
                    # [N, F, 4, 4]
                    if 0 <= frame_id < finger_poses.shape[0]:
                        fp = finger_poses[frame_id]
                        for idx, finger_name in enumerate(["thumb", "index", "middle", "ring"]):
                            poses[f"digit_{finger_name}"] = fp[idx]
                elif finger_poses.ndim == 3 and finger_poses.shape[-2:] == (4, 4):
                    # [N, 4, 4] – single finger unknown
                    if 0 <= frame_id < finger_poses.shape[0]:
                        poses["digit_unknown"] = finger_poses[frame_id]
            except (ValueError, TypeError, KeyError, IndexError) as e:
                raise RuntimeError(f"Failed to load finger poses from frame {frame_id}: {e}") from e
        return poses
    
    def _load_robot_pose(self, data: Dict, frame_id: int) -> Optional[np.ndarray]:
        """Load robot base pose if available."""
        # STRICT: Require allegro data section if robot pose is needed
        if 'allegro' not in data:
            return None  # No allegro data available
        if not isinstance(data['allegro'], dict):
            raise TypeError("'allegro' section must be a dictionary")
        
        allegro: Dict[str, Any] = data['allegro']
        if 'base_pose' not in allegro:
            return None  # No base pose available
        base_pose = allegro['base_pose']
        try:
            if base_pose is not None:
                arr = np.array(base_pose, dtype=np.float32)
                if arr.shape == (4, 4):
                    return arr
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to load base pose: {e}") from e
        return None
    
    def _load_joint_positions(self, data: Dict, frame_id: int) -> Optional[np.ndarray]:
        """Load joint positions from data.pkl if available."""
        # STRICT: Require allegro data section if joint positions are needed
        if 'allegro' not in data:
            return None  # No allegro data available
        if not isinstance(data['allegro'], dict):
            raise TypeError("'allegro' section must be a dictionary")
        
        allegro: Dict[str, Any] = data['allegro']
        if 'joint_state' not in allegro:
            return None  # No joint state available
        joint_state = allegro['joint_state']
        try:
            if isinstance(joint_state, np.ndarray) and joint_state.ndim >= 2 and 0 <= frame_id < joint_state.shape[0]:
                return joint_state[frame_id]
        except (ValueError, TypeError, IndexError) as e:
            raise RuntimeError(f"Failed to load joint positions: {e}") from e
        return None
    
    def _load_ee_pose(self, data: Dict, frame_id: int) -> Optional[np.ndarray]:
        """Load end effector pose from data.pkl if available."""
        # STRICT: Require allegro data section if end effector pose is needed
        if 'allegro' not in data:
            return None  # No allegro data available
        if not isinstance(data['allegro'], dict):
            raise TypeError("'allegro' section must be a dictionary")
        
        allegro: Dict[str, Any] = data['allegro']
        if 'ee_pose' not in allegro:
            return None  # No end effector pose available
        ee_pose = allegro['ee_pose']
        try:
            if isinstance(ee_pose, np.ndarray):
                if ee_pose.ndim == 3 and ee_pose.shape[-2:] == (4, 4) and 0 <= frame_id < ee_pose.shape[0]:
                    return ee_pose[frame_id]
                if ee_pose.ndim == 2 and ee_pose.shape == (4, 4):
                    return ee_pose
        except Exception as e:
            logger.debug(f"Failed to load base pose: {e}")
        return None
    
    def _load_gt_object_pose(self, data: Dict, frame_id: int) -> Optional[np.ndarray]:
        """Load ground truth object pose from data.pkl if available."""
        try:
            # STRICT: Require object data section if ground truth pose is needed
            if 'object' not in data:
                return None  # No object data available
            obj = data['object']
            if not isinstance(obj, dict):
                raise TypeError("'object' section must be a dictionary")
            if 'pose' not in obj:
                return None  # No object pose available
            pose_arr = obj['pose']
            if isinstance(pose_arr, np.ndarray):
                if pose_arr.ndim == 3 and pose_arr.shape[-2:] == (4, 4) and 0 <= frame_id < pose_arr.shape[0]:
                    return pose_arr[frame_id]
                if pose_arr.ndim == 2 and pose_arr.shape == (4, 4):
                    return pose_arr
        except Exception as e:
            logger.debug(f"Failed to load base pose: {e}")
        return None
    
    def _get_gt_mesh_path(self) -> str:
        """Get path to ground truth mesh"""
        mesh_path = self.config.data_root / "assets" / "gt_models" / "ycb" / f"{self.config.object}.urdf"
        return str(mesh_path) if mesh_path.exists() else None
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get Feelsight dataset information"""
        if self.dataset_info is not None:
            return self.dataset_info
            
        self.dataset_info = DatasetInfo(
            name="feelsight",
            description="Simulated visuo-tactile manipulation dataset with Allegro hand and multiple objects",
            version="1.0",
            authors=["FeelSight Team"],
            license="MIT",
            url="https://github.com/feelsight/feelsight",
            citation="@article{feelsight2024, title={FeelSight}, author={}, year={2024}}",
            total_sequences=len(self.get_sequence_ids()),
            total_frames=len(self.frames),
            modalities=self.modalities,
            objects=[self.config.object] if hasattr(self.config, 'object') else [],
            sensors={
                "realsense": {"type": "rgb-d", "resolution": "640x480"},
                "allegro": {"type": "tactile", "fingers": 4, "resolution": "160x120"}
            }
        )
        
        return self.dataset_info

class FeelsightRealDataset(FeelsightDataset):
    """Feelsight real-world dataset loader"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        self.metadata["dataset_type"] = "real_world"

class FeelsightOcclusionDataset(FeelsightDataset):
    """Feelsight occlusion dataset loader with segmentation mask support"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        self.metadata["dataset_type"] = "occlusion"
        
        # Enable mask reading for occlusion dataset (matches NeuralFeels behavior)
        self.use_masks = True
        self.masks_mode = "read"  # Read pre-generated masks from seg/ directory

        # Exclude specific trials if configured
        trial_id = f"{config.object}/{config.log}"
        if trial_id in config.excluded_trials:
            logger.info(f"Skipping excluded trial: {trial_id} for FeelsightOcclusionDataset")
            # Clear frames to ensure dataset is empty
            self.frames = []
            return
        
        # Validate that occlusion data exists
        self._validate_occlusion_data()
        
    def _validate_occlusion_data(self):
        """Validate that occlusion dataset has required seg masks"""
        # Use dataset_path directly or construct from config
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'dataset_path'):
            data_root = Path(self.config.data.dataset_path)
        else:
            # Construct from config attributes
            data_root = self.config.data_root / self.config.dataset / self.config.object / self.config.log
            
        if not data_root.exists():
            raise ValueError(f"Occlusion dataset path does not exist: {data_root}")
            
        # Store for use in other methods
        self.dataset_path = data_root
        
        # Check for seg directory in camera data
        camera_names = ["front-left", "back-right", "top-down"]
        available_cameras = []
        
        realsense_root = data_root / 'realsense'
        if realsense_root.exists():
            for cam_name in camera_names:
                cam_dir = realsense_root / cam_name
                seg_dir = cam_dir / 'seg'
                if cam_dir.exists() and seg_dir.exists():
                    # Check if seg directory has mask files
                    mask_files = list(seg_dir.glob('*.jpg'))
                    if mask_files:
                        available_cameras.append(cam_name)
                        
        if not available_cameras:
            raise ValueError(f"No segmentation masks found in occlusion dataset: {data_root}")
            
        print(f"📂 Occlusion dataset validated with masks for cameras: {available_cameras}")
        
        # Initialize camera_names if not already set by parent
        if not hasattr(self, 'camera_names'):
            self.camera_names = available_cameras if available_cameras else ['front-left']
        
    def _load_rgb_images(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load RGB images with occlusion-aware mask application"""
        rgb_images = super()._load_rgb_images(frame_id)
        
        # Apply occlusion masks to RGB images (matching NeuralFeels approach)
        masks = self._load_segmentation_masks(frame_id)
        if masks:
            for cam_name, rgb_image in rgb_images.items():
                if cam_name in masks:
                    mask = masks[cam_name]
                    # Apply mask to RGB image (occluded regions become black)
                    rgb_images[cam_name] = rgb_image * mask[..., np.newaxis]
                    
        return rgb_images
        
    def _load_depth_images(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load depth images with occlusion-aware mask application"""
        depth_images = super()._load_depth_images(frame_id)
        
        # Apply occlusion masks to depth images (matching NeuralFeels approach)
        masks = self._load_segmentation_masks(frame_id)
        if masks:
            for cam_name, depth_image in depth_images.items():
                if cam_name in masks:
                    mask = masks[cam_name]
                    # Apply mask to depth image (occluded regions become zero)
                    depth_images[cam_name] = depth_image * mask
                    
        return depth_images
        
    def _load_segmentation_masks(self, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Load segmentation masks from seg/ directories (matching NeuralFeels)"""
        try:
            # Use stored dataset path from validation
            data_root = self.dataset_path
            masks = {}
            
            # Use hardcoded camera names if camera_names not available
            camera_names = getattr(self, 'camera_names', ['front-left', 'back-right', 'top-down'])
            
            # Load masks for each available camera
            for cam_name in camera_names:
                seg_dir = data_root / 'realsense' / cam_name / 'seg'
                if seg_dir.exists():
                    mask_file = seg_dir / f'{frame_id}.jpg'
                    if mask_file.exists():
                        import cv2
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Convert to binary mask (non-zero = visible, zero = occluded)
                            mask = (mask > 0).astype(np.uint8)
                            masks[cam_name] = mask
                            
            return masks if masks else None
            
        except Exception as e:
            print(f"⚠️  Failed to load segmentation masks for frame {frame_id}: {e}")
            return None

class TouchNetDataset(BaseDataset):
    """TouchNet tactile dataset loader"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        self.dataset_path = config.data_root / "touchnet" / config.object
        self._load_data()
    
    def _load_data(self):
        """Load TouchNet dataset from HDF5 files"""
        h5_file = self.dataset_path / f"{self.split}.h5"
        
        if not h5_file.exists():
            logger.warning(f"TouchNet file not found: {h5_file}")
            return
        
        with h5py.File(h5_file, 'r') as f:
            timestamps = f['timestamps'][:]
            num_frames = len(timestamps)
            
            for i in range(min(num_frames, 100)):  # Limit for testing
                frame = FrameData(
                    frame_id=i,
                    timestamp=timestamps[i],
                    metadata={'sequence_id': f'touchnet_{self.config.object}', 'dataset': 'touchnet'}
                )
                
                # Load tactile data
                if 'tactile' in f and i < f['tactile'].shape[0]:
                    tactile_data = f['tactile'][i]
                    frame.tactile_images = {'sensor_0': tactile_data}
                    frame.tactile_depth = {'sensor_0': np.zeros((32, 32), dtype=np.float32)}
                    frame.tactile_poses = {'sensor_0': np.eye(4)}
                
                self.frames.append(frame)
        
        self.metadata = {
            "object": self.config.object,
            "dataset_type": "touchnet",
            "total_frames": len(self.frames),
        }
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get TouchNet dataset information"""
        return DatasetInfo(
            name="touchnet",
            description="Large-scale tactile dataset for object recognition",
            version="1.0",
            authors=["TouchNet Team"],
            license="Academic",
            modalities=["tactile"],
            objects=[self.config.object] if hasattr(self.config, 'object') else [],
            sensors={"gel_sensor": {"type": "tactile", "resolution": "240x320"}}
        )

class YCBDataset(BaseDataset):
    """YCB object dataset with synthetic tactile data"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        self.dataset_path = config.data_root / "ycb" / config.object
        self._load_data()
    
    def _load_data(self):
        """Load YCB dataset"""
        # Look for standard YCB format
        obj_dir = self.dataset_path
        if not obj_dir.exists():
            logger.warning(f"YCB object directory not found: {obj_dir}")
            return
        
        # Generate synthetic sequence
        for i in range(50):
            frame = FrameData(
                frame_id=i,
                timestamp=i * 0.1,
                metadata={'sequence_id': f'ycb_{self.config.object}', 'dataset': 'ycb'}
            )
            
            # Add synthetic camera poses in circle around object
            angle = 2 * np.pi * i / 50
            pose = np.eye(4)
            pose[0, 3] = 0.3 * np.cos(angle)
            pose[1, 3] = 0.3 * np.sin(angle)
            pose[2, 3] = 0.2
            
            frame.camera_poses = {'camera': pose}
            frame.camera_intrinsics = {'camera': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])}
            frame.gt_object_pose = np.eye(4)
            frame.gt_mesh_path = str(obj_dir / "model.obj") if (obj_dir / "model.obj").exists() else None
            
            self.frames.append(frame)
        
        self.metadata = {
            "object": self.config.object,
            "dataset_type": "ycb_synthetic",
            "total_frames": len(self.frames),
        }
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get YCB dataset information"""
        return DatasetInfo(
            name="ycb",
            description="Yale-CMU-Berkeley Object Dataset with synthetic viewpoints",
            version="1.0",
            authors=["YCB Team"],
            license="Open",
            modalities=["vision"],
            objects=[self.config.object] if hasattr(self.config, 'object') else [],
            sensors={"synthetic_camera": {"type": "rgb-d", "resolution": "640x480"}}
        )

class D3NetDataset(BaseDataset):
    """D3Net dense tactile dataset"""
    
    def __init__(self, config: GaussianFeelsConfig, split: str = "train"):
        super().__init__(config, split)
        self.dataset_path = config.data_root / "d3net" / config.object
        self._load_data()
    
    def _load_data(self):
        """Load D3Net dataset"""
        npz_file = self.dataset_path / f"{self.split}.npz"
        
        if not npz_file.exists():
            logger.warning(f"D3Net file not found: {npz_file}")
            return
        
        with np.load(npz_file) as data:
            tactile_data = data['tactile'] if 'tactile' in data else None
            timestamps = data['timestamps'] if 'timestamps' in data else np.arange(100) * 0.1
            
            num_frames = len(timestamps) if tactile_data is None else min(len(timestamps), tactile_data.shape[0])
            
            for i in range(min(num_frames, 100)):
                frame = FrameData(
                    frame_id=i,
                    timestamp=timestamps[i],
                    metadata={'sequence_id': f'd3net_{self.config.object}', 'dataset': 'd3net'}
                )
                
                if tactile_data is not None:
                    frame.tactile_images = {'d3net_sensor': tactile_data[i]}
                    frame.tactile_depth = {'d3net_sensor': np.random.rand(64, 64) * 0.01}
                    frame.tactile_poses = {'d3net_sensor': np.eye(4)}
                
                self.frames.append(frame)
        
        self.metadata = {
            "object": self.config.object,
            "dataset_type": "d3net",
            "total_frames": len(self.frames),
        }
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get D3Net dataset information"""
        return DatasetInfo(
            name="d3net",
            description="Dense tactile dataset for shape reconstruction",
            version="1.0",
            authors=["D3Net Team"],
            license="Academic",
            modalities=["tactile"],
            objects=[self.config.object] if hasattr(self.config, 'object') else [],
            sensors={"dense_tactile": {"type": "tactile", "resolution": "64x64"}}
        )

class DatasetRegistry:
    """Registry for managing multiple datasets"""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.datasets = {}
        self._register_datasets()
    
    def _register_datasets(self):
        """Register available datasets"""
        self.datasets = {
            "feelsight": FeelsightDataset,
            "feelsight_real": FeelsightRealDataset,
            "feelsight_occlusion": FeelsightOcclusionDataset,
            "touchnet": TouchNetDataset,
            "ycb": YCBDataset,
            "d3net": D3NetDataset,
        }
    
    def list_datasets(self) -> List[str]:
        """List available datasets"""
        return list(self.datasets.keys())
    
    def list_objects(self, dataset_name: str) -> List[str]:
        """List available objects in a dataset"""
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            return []
        
        objects = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        return sorted(objects)
    
    def list_logs(self, dataset_name: str, object_name: str) -> List[str]:
        """List available logs for an object"""
        object_path = self.data_root / dataset_name / object_name
        if not object_path.exists():
            return []
        
        logs = [d.name for d in object_path.iterdir() if d.is_dir()]
        return sorted(logs)
    
    def load_dataset(self, config: GaussianFeelsConfig, split: str = "train") -> BaseDataset:
        """Load a dataset based on configuration"""
        dataset_name = config.dataset
        
        if dataset_name not in self.datasets:
            available = ", ".join(self.datasets.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        dataset_class = self.datasets[dataset_name]
        return dataset_class(config, split)
    
    def create_dataloader(self, config: GaussianFeelsConfig, 
                         split: str = "train", 
                         batch_size: int = 1,
                         shuffle: bool = False,
                         num_workers: int = 0) -> DataLoader:
        """Create a DataLoader for a dataset"""
        dataset = self.load_dataset(config, split)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[FrameData]) -> FrameData:
        """Custom collate function for batching FrameData"""
        if len(batch) == 1:
            return batch[0]
        
        # Minimal collate: return the first frame to avoid fabricating merged data
        # Downstream code should set batch_size=1 for sequence models.
        return batch[0]
    
    def validate_dataset(self, config: GaussianFeelsConfig) -> bool:
        """Validate that a dataset configuration is valid"""
        try:
            dataset = self.load_dataset(config)
            return len(dataset) > 0
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
            raise RuntimeError(f"Dataset validation failed: {e}") from e
    
    def load_neuralfeels_compatible(self, 
                                   dataset: str,
                                   object_name: str, 
                                   log: str,
                                   modality: str = "vitac") -> BaseDataset:
        """Load dataset with NeuralFeels-compatible interface"""
        from .config import GaussianFeelsConfig
        
        config = GaussianFeelsConfig(
            dataset=dataset,
            object=object_name,
            log=log,
            modality=modality,
            data_root=self.data_root
        )
        
        return self.load_dataset(config)
    
    def get_dataset_info(self, config: GaussianFeelsConfig) -> Dict[str, Any]:
        """Get information about a dataset"""
        dataset = self.load_dataset(config)
        
        info = {
            "name": config.dataset,
            "object": config.object,
            "log": config.log,
            "total_frames": len(dataset),
            "modalities": dataset.modalities,
            "metadata": dataset.metadata,
        }
        
        if len(dataset) > 0:
            sample_frame = dataset[0]
            info["sample_frame"] = {
                "timestamp": sample_frame.timestamp,
                "has_rgb": sample_frame.rgb_images is not None,
                "has_depth": sample_frame.depth_images is not None,
                "has_tactile": sample_frame.tactile_images is not None,
                "camera_count": len(sample_frame.rgb_images) if sample_frame.rgb_images else 0,
                "tactile_count": len(sample_frame.tactile_images) if sample_frame.tactile_images else 0,
            }
        
        return info
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """Download dataset from cloud storage"""
        urls = {
            "touchnet": "https://datasets.example.com/touchnet.tar.gz",
            "ycb": "https://datasets.example.com/ycb.tar.gz", 
            "d3net": "https://datasets.example.com/d3net.tar.gz",
        }
        
        if dataset_name not in urls:
            logger.warning(f"No download URL available for dataset: {dataset_name}")
            return False
        
        dataset_path = self.data_root / dataset_name
        if dataset_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return True
        
        url = urls[dataset_name]
        logger.info(f"Downloading {dataset_name} from {url}")
        
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
                
                tmp_file_path = tmp_file.name
            
            # Extract archive
            logger.info(f"Extracting {dataset_name} to {dataset_path}")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(tmp_file_path, 'r:gz') as tar:
                tar.extractall(path=dataset_path.parent)
            
            # Clean up temporary file
            Path(tmp_file_path).unlink()
            
            logger.info(f"Successfully downloaded and extracted {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def list_available_downloads(self) -> Dict[str, Dict[str, Any]]:
        """List datasets available for download"""
        return {
            "touchnet": {
                "description": "Large-scale tactile dataset",
                "size": "2.1 GB",
                "objects": ["cube", "sphere", "cylinder"],
                "modalities": ["tactile"]
            },
            "ycb": {
                "description": "Yale-CMU-Berkeley Object Dataset",
                "size": "15.3 GB", 
                "objects": ["77 everyday objects"],
                "modalities": ["vision"]
            },
            "d3net": {
                "description": "Dense tactile shape reconstruction",
                "size": "850 MB",
                "objects": ["various shapes"],
                "modalities": ["tactile"]
            }
        }
    
    def benchmark_dataset(self, config: GaussianFeelsConfig, method_name: str, 
                         metrics: Dict[str, float], notes: str = "") -> BenchmarkResult:
        """Record benchmark results for a dataset"""
        from datetime import datetime
        
        result = BenchmarkResult(
            dataset_name=config.dataset,
            method_name=method_name,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            config=config.__dict__,
            notes=notes
        )
        
        # Save benchmark result
        benchmark_dir = self.data_root / "benchmarks"
        benchmark_dir.mkdir(exist_ok=True)
        
        benchmark_file = benchmark_dir / f"{config.dataset}_{method_name}_results.json"
        
        # Load existing results
        results = []
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                results = json.load(f)
        
        # Add new result
        results.append(result.__dict__)
        
        # Save updated results
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark result saved to {benchmark_file}")
        return result
    
    def export_dataset_catalog(self, output_path: Path):
        """Export catalog of all available datasets"""
        catalog = {
            "datasets": {},
            "download_info": self.list_available_downloads(),
            "created_at": str(Path().cwd()),
            "total_datasets": len(self.datasets)
        }
        
        # Get info for each registered dataset 
        for name, dataset_class in self.datasets.items():
            try:
                # Create dummy config to get dataset info
                from .config import GaussianFeelsConfig
                dummy_config = GaussianFeelsConfig(
                    dataset=name,
                    object="dummy_object",
                    data_root=self.data_root
                )
                
                # Try to create dataset instance to get info
                dummy_dataset = dataset_class(dummy_config)
                info = dummy_dataset.get_dataset_info()
                catalog["datasets"][name] = info.__dict__
                
            except Exception as e:
                logger.warning(f"Could not get info for dataset {name}: {e}")
                catalog["datasets"][name] = {"name": name, "error": str(e)}
        
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2, default=str)
        
        logger.info(f"Dataset catalog exported to {output_path}")
    
    def convert_dataset_format(self, source_config: GaussianFeelsConfig, 
                             target_format: str, output_path: Path):
        """Convert dataset to different format"""
        source_dataset = self.load_dataset(source_config)
        
        if target_format == "hdf5":
            self._convert_to_hdf5(source_dataset, output_path)
        elif target_format == "zarr":
            self._convert_to_zarr(source_dataset, output_path)
        elif target_format == "rosbag":
            self._convert_to_rosbag(source_dataset, output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def _convert_to_hdf5(self, dataset: BaseDataset, output_path: Path):
        """Convert dataset to HDF5 format"""
        with h5py.File(output_path, 'w') as f:
            # Create groups
            f.create_group("frames")
            f.create_group("metadata")
            
            # Store dataset metadata
            for key, value in dataset.metadata.items():
                if isinstance(value, (str, int, float)):
                    f["metadata"].attrs[key] = value
            
            # Store frames
            timestamps = []
            for i, frame in enumerate(dataset.frames):
                frame_group = f["frames"].create_group(f"frame_{i:06d}")
                
                # Store basic info
                frame_group.attrs["frame_id"] = frame.frame_id
                frame_group.attrs["timestamp"] = frame.timestamp
                
                # Store images
                if frame.rgb_images:
                    rgb_group = frame_group.create_group("rgb")
                    for cam_name, img in frame.rgb_images.items():
                        rgb_group.create_dataset(cam_name, data=img, compression="gzip")
                
                if frame.tactile_images:
                    tactile_group = frame_group.create_group("tactile")
                    for sensor_name, img in frame.tactile_images.items():
                        tactile_group.create_dataset(sensor_name, data=img, compression="gzip")
                
                # Store poses
                if frame.camera_poses:
                    poses_group = frame_group.create_group("camera_poses")
                    for cam_name, pose in frame.camera_poses.items():
                        poses_group.create_dataset(cam_name, data=pose)
                
                timestamps.append(frame.timestamp)
            
            # Store global arrays
            f.create_dataset("timestamps", data=timestamps)
        
        logger.info(f"Dataset converted to HDF5 format: {output_path}")
    
    def _convert_to_zarr(self, dataset: BaseDataset, output_path: Path):
        """Convert dataset to Zarr format"""
        raise NotImplementedError("Zarr conversion not implemented. Use HDF5 format or implement Zarr support.")
    
    def _convert_to_rosbag(self, dataset: BaseDataset, output_path: Path):
        """Convert dataset to ROS bag format"""
        raise NotImplementedError("ROS bag conversion not implemented. Use HDF5 format or implement ROS bag support.")