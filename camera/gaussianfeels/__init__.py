# GaussianFeels - Phase 1 Implementation
# Object-centric Gaussian splatting with tactile-visual fusion

__version__ = "1.0.0"
__author__ = "GaussianFeels Team"

# Core exports
from .backend.gaussian_field import ObjectGaussianMap, GaussianParams
from .spatial.spatial_hash import CUDASpatialHash, SpatialHashConfig, create_spatial_hash
from .memory.memory_planner import MemoryPlanner, MemoryMode, MemoryBudget
from .memory.memory_monitor import MemoryMonitor
from .modes.mode_manager import ModeManager, OperationMode
from .instrumentation.live_counters import LiveCounters
from .config.config_system import ConfigManager, ConfigProfile, GaussianFeelsConfig

# Segmentation and reconstruction exports
from .io.segmentation import (
    SegmentationProcessor,
    FeelsightRealSegmentationLoader,
    create_segmentation_processor,
    create_feelsight_real_segmenter,
)
from .io.frame_data import FrameData, pose_from_config, create_identity_pose
from .utils.object_reconstruction import ObjectReconstructor, reconstruct_object_from_feelsight

__all__ = [
    # Core classes
    'ObjectGaussianMap',
    'GaussianParams', 
    'CUDASpatialHash',
    'SpatialHashConfig',
    'MemoryPlanner',
    'MemoryMonitor',
    'ModeManager',
    'LiveCounters',
    'ConfigManager',
    
    # Segmentation and reconstruction classes
    'SegmentationProcessor',
    'FeelsightRealSegmentationLoader',
    'ObjectReconstructor',
    'FrameData',
    
    # Enums
    'MemoryMode',
    'OperationMode', 
    'ConfigProfile',
    
    # Data classes
    'MemoryBudget',
    'GaussianFeelsConfig',
    
    # Factory functions
    'create_spatial_hash',
    'create_segmentation_processor',
    'create_feelsight_real_segmenter',
    'reconstruct_object_from_feelsight',
    'pose_from_config',
    'create_identity_pose',
]