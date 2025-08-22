# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
JSON-based configuration system for GaussianFeels with runtime tuning and profile management.
Supports memory limits, performance targets, and operational modes.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
import time
from enum import Enum

from ..memory.memory_planner import MemoryMode
from ..modes.mode_manager import OperationMode


class ConfigProfile(Enum):
    """Predefined configuration profiles"""
    PERFORMANCE = "performance"     # Maximum performance, high memory usage
    BALANCED = "balanced"          # Balance between performance and efficiency  
    EFFICIENCY = "efficiency"     # Low memory usage, moderate performance
    DEVELOPMENT = "development"    # Development settings with debugging
    PRODUCTION = "production"      # Production settings optimized for stability


@dataclass
class MemoryConfig:
    """Memory management configuration - ALL FIELDS REQUIRED"""
    target_gaussians: int
    memory_mode: str  # REQUIRED: aggressive (only mode allowed)
    vram_limit_gb: float
    enable_lod: bool
    lod_threshold_mb: float
    max_densification_rate: float
    pruning_interval: int
    
    def to_memory_mode(self) -> MemoryMode:
        """Convert string to MemoryMode enum - AGGRESSIVE ONLY"""
        mode_map = {
            "aggressive": MemoryMode.AGGRESSIVE  # Only mode allowed
        }
        if self.memory_mode not in mode_map:
            raise ValueError(f"Invalid memory mode: {self.memory_mode}. Must be one of: {list(mode_map.keys())}")
        return mode_map[self.memory_mode]


@dataclass
class PerformanceConfig:
    """Performance target configuration - ALL FIELDS REQUIRED"""
    render_fps_target: float
    render_fps_min: float
    optimize_hz_target: float
    optimize_hz_min: float
    interpolate_hz_target: float
    interpolate_hz_min: float
    max_render_resolution: tuple
    adaptive_quality: bool
    quality_threshold_fps: float


@dataclass
class ModeConfig:
    """Operation mode configuration - ALL FIELDS REQUIRED"""
    default_mode: str  # REQUIRED - NO DEFAULTS
    auto_transitions: bool
    stability_threshold: float
    transition_timeout_s: float
    emergency_mode_threshold: float
    mode_timeouts: Dict[str, float]  # REQUIRED - must specify all timeouts
    
    def to_operation_mode(self) -> OperationMode:
        """Convert string to OperationMode enum"""
        mode_map = {
            "tactile_only": OperationMode.TACTILE_ONLY,
            "tactile_first": OperationMode.TACTILE_FIRST,
            "balanced": OperationMode.BALANCED,
            "camera_first": OperationMode.CAMERA_FIRST,
            "camera_only": OperationMode.CAMERA_ONLY
        }
        if self.default_mode not in mode_map:
            raise ValueError(f"Invalid operation mode: {self.default_mode}. Must be one of: {list(mode_map.keys())}")
        return mode_map[self.default_mode]


@dataclass
class SpatialConfig:
    """Spatial hash configuration - ALL FIELDS REQUIRED"""
    cell_size: float
    max_neighbors: int
    hash_table_size: int
    adaptive_cell_size: bool
    rebuild_threshold: float


@dataclass
class RenderConfig:
    """Rendering configuration - ALL FIELDS REQUIRED"""
    default_resolution: tuple
    max_resolution: tuple
    fov_degrees: float
    near_plane: float
    far_plane: float
    densification_threshold: float
    opacity_threshold: float
    enable_sh_spherical_harmonics: bool


@dataclass
class DebugConfig:
    """Debug and development configuration - ALL FIELDS REQUIRED"""
    enable_debug_output: bool
    log_level: str
    performance_profiling: bool
    memory_profiling: bool
    save_debug_frames: bool
    debug_frame_interval: int


@dataclass
class GaussianFeelsConfig:
    """Main configuration structure"""
    memory: MemoryConfig
    performance: PerformanceConfig
    modes: ModeConfig
    spatial: SpatialConfig
    render: RenderConfig
    debug: DebugConfig
    
    # Metadata
    profile: str  # REQUIRED - NO DEFAULTS
    version: str
    created_at: str
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Memory validation
        if self.memory.target_gaussians <= 0:
            errors.append("target_gaussians must be positive")
        if self.memory.vram_limit_gb <= 0:
            errors.append("vram_limit_gb must be positive")
            
        # Performance validation
        if self.performance.render_fps_target <= 0:
            errors.append("render_fps_target must be positive")
        if self.performance.optimize_hz_target <= 0:
            errors.append("optimize_hz_target must be positive")
        if self.performance.interpolate_hz_target <= 0:
            errors.append("interpolate_hz_target must be positive")
            
        # Spatial validation
        if self.spatial.cell_size <= 0:
            errors.append("spatial cell_size must be positive")
        if self.spatial.max_neighbors <= 0:
            errors.append("max_neighbors must be positive")
            
        # Mode validation
        valid_modes = ["tactile_only", "tactile_first", "balanced", "camera_first", "camera_only"]
        if self.modes.default_mode not in valid_modes:
            errors.append(f"default_mode must be one of {valid_modes}")
            
        return errors


class ConfigManager:
    """Configuration manager with file I/O and runtime updates"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            raise ValueError("config_dir is required - no default directory allowed")
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config: Optional[GaussianFeelsConfig] = None
        self.config_file: Optional[Path] = None
        
        # Runtime update callbacks
        self.update_callbacks: List[Callable[[GaussianFeelsConfig], None]] = []
        
        # File watching for hot reload
        self.watch_enabled = False
        self.watch_thread: Optional[threading.Thread] = None
        self.last_modified: float = 0.0
        
    def create_profile_config(self, profile: ConfigProfile) -> GaussianFeelsConfig:
        """REMOVED: No default profile creation allowed. Configs must be explicit."""
        raise NotImplementedError("Profile creation removed. All configs must be explicitly provided.")
    
    def load_config(self, config_path: Union[str, Path]) -> GaussianFeelsConfig:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}. NO FALLBACK CREATION ALLOWED.")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # STRICT: All sections must be present - no fallbacks
        if 'memory' not in config_dict:
            raise KeyError("'memory' section required in config")
        if 'performance' not in config_dict:
            raise KeyError("'performance' section required in config")
        if 'modes' not in config_dict:
            raise KeyError("'modes' section required in config")
        if 'spatial' not in config_dict:
            raise KeyError("'spatial' section required in config")
        if 'render' not in config_dict:
            raise KeyError("'render' section required in config")
        if 'debug' not in config_dict:
            raise KeyError("'debug' section required in config")
            
        memory_config = MemoryConfig(**config_dict['memory'])
        performance_config = PerformanceConfig(**config_dict['performance'])
        modes_config = ModeConfig(**config_dict['modes'])
        spatial_config = SpatialConfig(**config_dict['spatial'])
        render_config = RenderConfig(**config_dict['render'])
        debug_config = DebugConfig(**config_dict['debug'])
        
        config = GaussianFeelsConfig(
            memory=memory_config,
            performance=performance_config,
            modes=modes_config,
            spatial=spatial_config,
            render=render_config,
            debug=debug_config,
            profile=config_dict['profile'],  # REQUIRED
            version=config_dict['version']   # REQUIRED
        )
        
        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        self.config = config
        self.config_file = config_path
        self.last_modified = config_path.stat().st_mtime
        
        return config
    
    def save_config(self, config: GaussianFeelsConfig, config_path: Union[str, Path]):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save with pretty formatting
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        
        self.config_file = config_path
        self.last_modified = config_path.stat().st_mtime
    
    def create_default_configs(self):
        """REMOVED: No default config creation allowed."""
        raise NotImplementedError("Default config creation removed. All configs must be explicitly provided.")
    
    def get_current_config(self) -> Optional[GaussianFeelsConfig]:
        """Get currently loaded configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any], save: bool = True):
        """Update configuration with partial changes"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        # STRICT: No partial updates allowed
        raise NotImplementedError("Partial config updates removed. Replace entire config explicitly.")
        
        # REMOVED
    
    def register_callback(self, callback: Callable[[GaussianFeelsConfig], None]):
        """Register callback for configuration updates"""
        self.update_callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks of config change"""
        if self.config:
            for callback in self.update_callbacks:
                callback(self.config)  # FAIL FAST - no exception handling
    
    def start_file_watching(self):
        """Start watching config file for changes"""
        if self.watch_enabled or not self.config_file:
            return
        
        self.watch_enabled = True
        self.watch_thread = threading.Thread(target=self._watch_file, daemon=True)
        self.watch_thread.start()
    
    def stop_file_watching(self):
        """Stop watching config file"""
        self.watch_enabled = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)
    
    def _watch_file(self):
        """File watching thread"""
        while self.watch_enabled and self.config_file:
            try:
                if self.config_file.exists():
                    current_mtime = self.config_file.stat().st_mtime
                    if current_mtime > self.last_modified:
                        print("Config file changed, reloading...")
                        self.load_config(self.config_file)
                        self._notify_callbacks()
                        print("Config reloaded successfully")
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                raise RuntimeError(f"File watching failed: {e}") from e


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance - STRICT"""
    global _config_manager
    if _config_manager is None:
        raise RuntimeError("Config manager not initialized. Must call set_config_manager() first.")
    return _config_manager


def set_config_manager(config_dir: str) -> ConfigManager:
    """Initialize global configuration manager - REQUIRED"""
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return _config_manager


def load_profile(profile: Union[ConfigProfile, str]) -> GaussianFeelsConfig:
    """Load configuration profile"""
    manager = get_config_manager()
    
    if isinstance(profile, str):
        profile = ConfigProfile(profile)
    
    config_path = manager.config_dir / f"{profile.value}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Profile config not found: {config_path}. NO FALLBACK CREATION ALLOWED.")
    
    return manager.load_config(config_path)


if __name__ == "__main__":
    # Test configuration system
    print("Testing GaussianFeels configuration system...")
    
    # REMOVED: No default config creation in ultrathink mode
    raise RuntimeError("Default config creation removed. Provide explicit config files.")
    # REMOVED