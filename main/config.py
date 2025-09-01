"""
GaussianFeels Configuration System

Hierarchical configuration supporting multiple datasets, modes, and modalities.
Inspired by Hydra but simpler and more focused on Gaussian splatting workflows.
"""

import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import json

@dataclass
class LearningRates:
    """Learning rates for different Gaussian parameters and poses"""
    position: float = 2e-4
    rotation: float = 1e-3
    scale: float = 5e-3
    opacity: float = 5e-2
    color: float = 1e-2
    pose: float = 1e-4  # Camera and tactile sensor pose optimization

@dataclass
class GaussianParams:
    """Gaussian splatting specific parameters"""
    max_gaussians: Optional[int] = None
    initial_gaussians: int = 10000
    densify_threshold: float = 0.0002
    prune_threshold: float = 0.005
    densify_interval: int = 100
    prune_interval: int = 100
    split_threshold: float = 0.02
    clone_threshold: float = 0.01
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000

@dataclass
class SensorConfig:
    """Configuration for individual sensors"""
    name: str
    type: str  # "realsense", "digit", etc.
    enabled: bool = True
    resolution: tuple = (640, 480)
    fps: int = 30
    calibration_file: Optional[str] = None
    
@dataclass
class ModalityConfig:
    """Configuration for different modality combinations"""
    name: str  # "vitac", "vi", "tac"
    sensors: List[SensorConfig] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)  # fusion weights
    
    def __post_init__(self):
        """Set default weights if not provided"""
        if not self.weights:
            if self.name == "vitac":
                self.weights = {"vision": 0.2, "tactile": 0.8}
            elif self.name == "vi":
                self.weights = {"vision": 1.0}
            elif self.name == "tac":
                self.weights = {"tactile": 1.0}

@dataclass
class DatasetConfig:
    """Configuration for dataset access"""
    name: str
    path: str
    type: str = "local"  # "local", "huggingface", "s3", "gcs"
    streaming: bool = False
    cache_dir: Optional[str] = None
    download: bool = True
    
@dataclass
class ViewerConfig:
    """Configuration for visualization"""
    type: str = "open3d"  # "open3d", "web", "none"
    port: int = 8080  # for web viewer
    host: str = "localhost"
    fps: int = 30
    resolution: tuple = (1920, 1080)
    background_color: tuple = (0.1, 0.1, 0.1)
    show_cameras: bool = True
    show_sensors: bool = True
    show_trajectory: bool = True
    point_size: float = 1.0

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    max_steps: int = 5000
    fps: int = 1  # optimization steps per second
    batch_size: int = 1
    gradient_clip: float = 1.0
    mixed_precision: bool = False
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
@dataclass
class LossConfig:
    """Loss function configuration"""
    rgb_weight: float = 1.0
    depth_weight: float = 0.1
    silhouette_weight: float = 0.05
    tactile_weight: float = 0.8
    pose_weight: float = 0.1
    smoothness_weight: float = 0.01

@dataclass
class GaussianFeelsConfig:
    """Main configuration class for GaussianFeels"""
    
    # Core experiment settings
    dataset: str = "feelsight"
    mode: str = "slam"  # "slam", "pose", "map"
    modality: str = "vitac"  # "vitac", "vi", "tac"
    object: str = "contactdb_rubber_duck"
    log: str = "00"
    
    # Optional direct modalities specification
    _modalities: Optional[List[str]] = None
    
    # Device and performance
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    
    # Paths
    output_dir: Path = Path("outputs")
    data_root: Path = Path("data")
    
    # Sub-configurations
    learning_rates: LearningRates = field(default_factory=LearningRates)
    gaussian_params: GaussianParams = field(default_factory=GaussianParams)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Advanced options
    three_camera_mode: bool = False
    record: bool = False
    debug: bool = False
    profile: bool = False
    excluded_trials: List[str] = field(default_factory=list)
    
    # Convenience properties that update sub-configs
    fps: int = field(default=1)
    max_steps: int = field(default=5000)
    densify_threshold: float = field(default=0.0002)
    prune_threshold: float = field(default=0.005)
    densify_interval: int = field(default=100)
    
    # Internal storage for modalities override
    _modalities_override: Optional[List[str]] = field(default=None, init=False, repr=False)
    
    def __init__(self, modalities: Optional[List[str]] = None, **kwargs):
        """Initialize config with optional modalities parameter"""
        # Initialize all dataclass fields with their defaults or provided values
        for field_name, field_info in self.__dataclass_fields__.items():
            if field_info.init and field_name != 'modalities':
                if field_name in kwargs:
                    value = kwargs.pop(field_name)
                elif field_info.default != dataclasses.MISSING:
                    value = field_info.default
                elif field_info.default_factory != dataclasses.MISSING:  # type: ignore
                    value = field_info.default_factory()
                else:
                    raise TypeError(f"Missing required argument: {field_name}")
                setattr(self, field_name, value)
        
        # Store modalities override
        self._modalities_override = modalities
        
        # Handle any remaining kwargs
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown arguments: {unknown}")
        
        # Call post_init
        self.__post_init__()
    
    def __post_init__(self):
        """Update sub-configurations from convenience properties"""
        # Ensure expected attributes exist even if defaults were omitted somehow
        if not hasattr(self, 'max_gaussians'):
            self.max_gaussians = None
        if not hasattr(self, 'gaussian_params') or self.gaussian_params is None:
            self.gaussian_params = GaussianParams()
        if not hasattr(self, 'training') or self.training is None:
            self.training = TrainingConfig()

        self.training.fps = self.fps
        self.training.max_steps = self.max_steps
        self.gaussian_params.max_gaussians = self.max_gaussians
        self.gaussian_params.densify_threshold = self.densify_threshold
        self.gaussian_params.prune_threshold = self.prune_threshold
        self.gaussian_params.densify_interval = self.densify_interval
        
        # Ensure paths are Path objects
        self.output_dir = Path(self.output_dir)
        self.data_root = Path(self.data_root)
    
    @property
    def dataset_path(self) -> Path:
        """Get the full dataset path"""
        return self.data_root / self.dataset / self.object / self.log
    
    @property
    def experiment_name(self) -> str:
        """Generate experiment name for outputs"""
        return f"{self.dataset}_{self.object}_{self.log}_{self.mode}_{self.modality}"
    
    @property
    def modalities(self) -> List[str]:
        """Get list of active modalities"""
        # If modalities is directly specified, use that
        if self._modalities_override is not None:
            return self._modalities_override
        
        # Otherwise derive from modality
        if self.modality == "vitac":
            return ["vision", "tactile"]
        elif self.modality == "vi":
            return ["vision"]
        elif self.modality == "tac":
            return ["tactile"]
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
    
    @modalities.setter
    def modalities(self, value: Optional[List[str]]):
        """Set list of active modalities"""
        self._modalities_override = value
    
    def get_sensor_config(self, modality: str) -> ModalityConfig:
        """Get sensor configuration for a modality"""
        if modality == "vitac":
            sensors = [
                SensorConfig("front-left", "realsense"),
                SensorConfig("digit_thumb", "digit"),
                SensorConfig("digit_index", "digit"),
                SensorConfig("digit_middle", "digit"),
                SensorConfig("digit_ring", "digit", enabled=False),  # often problematic
            ]
            if self.three_camera_mode:
                sensors.extend([
                    SensorConfig("back-right", "realsense"),
                    SensorConfig("top-down", "realsense"),
                ])
        elif modality == "vi":
            sensors = [SensorConfig("front-left", "realsense")]
            if self.three_camera_mode:
                sensors.extend([
                    SensorConfig("back-right", "realsense"),
                    SensorConfig("top-down", "realsense"),
                ])
        elif modality == "tac":
            sensors = [
                SensorConfig("digit_thumb", "digit"),
                SensorConfig("digit_index", "digit"),
                SensorConfig("digit_middle", "digit"),
                SensorConfig("digit_ring", "digit", enabled=False),
            ]
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        return ModalityConfig(modality, sensors)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            self.save_yaml(path)
        elif path.suffix == ".json":
            self.save_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def save_yaml(self, path: Union[str, Path]):
        """Save configuration as YAML"""
        import yaml
        from dataclasses import asdict
        
        data = asdict(self)
        # Convert Path objects to strings for serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        data = convert_paths(data)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def save_json(self, path: Union[str, Path]):
        """Save configuration as JSON"""
        from dataclasses import asdict
        import json
        
        data = asdict(self)
        # Convert Path objects to strings for serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        data = convert_paths(data)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianFeelsConfig":
        """Load configuration from file"""
        path = Path(path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            return cls.load_yaml(path)
        elif path.suffix == ".json":
            return cls.load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> "GaussianFeelsConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "GaussianFeelsConfig":
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianFeelsConfig":
        """Create configuration from dictionary"""
        # Handle modalities parameter - extract it to pass to constructor
        modalities = data.pop("modalities", None)
        
        # Convert nested dictionaries to dataclass instances
        if "learning_rates" in data:
            data["learning_rates"] = LearningRates(**data["learning_rates"])
        if "gaussian_params" in data:
            data["gaussian_params"] = GaussianParams(**data["gaussian_params"])
        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])
        if "viewer" in data:
            data["viewer"] = ViewerConfig(**data["viewer"])
        if "loss" in data:
            data["loss"] = LossConfig(**data["loss"])
        
        # Convert string paths back to Path objects
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "data_root" in data:
            data["data_root"] = Path(data["data_root"])
        
        return cls(modalities=modalities, **data)

# Preset configurations
PRESET_CONFIGS = {
    "slam-sim": {
        "dataset": "feelsight",
        "mode": "slam", 
        "modality": "vitac",
        "object": "contactdb_rubber_duck",
        "log": "00",
        "fps": 1,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "pose-sim": {
        "dataset": "feelsight",
        "mode": "pose",
        "modality": "vitac", 
        "object": "077_rubiks_cube",
        "log": "00",
        "fps": 1,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "slam-real": {
        "dataset": "feelsight_real",
        "mode": "slam",
        "modality": "vitac",
        "object": "bell_pepper", 
        "log": "00",
        "fps": 1,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "pose-real": {
        "dataset": "feelsight_real",
        "mode": "pose",
        "modality": "vitac",
        "object": "large_dice",
        "log": "00", 
        "fps": 1,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "three-cam": {
        "dataset": "feelsight_real",
        "mode": "pose",
        "modality": "vi",
        "object": "large_dice",
        "log": "00",
        "fps": 1,
        "three_camera_mode": True,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "occlusion-sim": {
        "dataset": "feelsight_occlusion",
        "mode": "pose",
        "modality": "vitac",
        "object": "077_rubiks_cube",
        "log": "00",
        "fps": 1,
        "viewer": {"type": "open3d"},
        "record": True,
    },
    
    "web-demo": {
        "dataset": "feelsight",
        "mode": "pose",
        "modality": "vitac",
        "object": "contactdb_rubber_duck",
        "log": "00",
        "fps": 2,
        "viewer": {"type": "web", "port": 8080},
        "record": True,
    },
}

def get_preset_config(preset_name: str) -> GaussianFeelsConfig:
    """Get a preset configuration by name"""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    preset_data = PRESET_CONFIGS[preset_name].copy()
    return GaussianFeelsConfig.from_dict(preset_data)