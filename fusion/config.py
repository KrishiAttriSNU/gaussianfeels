#!/usr/bin/env python3.11
"""
Configuration for camera-tactile fusion processing
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TactileFusionConfig:
    """Configuration for camera-tactile fusion processing"""
    
    # Data paths (defaults resolved relative to repo root when not provided)
    trial_path: str = ""
    output_dir: str = ""
    
    # Test parameters
    max_frames: int = -1  # Process all available frames
    tactile_sensors: List[str] = None
    batch_size: int = 20  # Larger batches for efficiency
    
    # Contact detection parameters
    contact_ratio: float = 1.2        # Contact detection threshold multiplier
    contact_clip: float = 10.0        # Minimum depth difference for contact
    contact_border: int = 0           # Border pixels to ignore
    background_frames: int = 1        # Number of frames for background template
    subsample_factor: int = 4         # Point cloud subsampling
    outlier_thresh: float = 5.0       # Outlier filtering threshold
    
    # Device
    device: str = "cuda"
    
    # Finger control
    all_fingers: bool = False  # Enable all fingers including problematic ones
    allow_approx_tip_poses: bool = False  # Allow approximate fingertip poses when finger_poses missing (feelsight_real)
    use_forward_kinematics: bool = True   # Use proper forward kinematics for feelsight_real data (requires torchkin)
    
    def __post_init__(self):
        if self.tactile_sensors is None:
            self.tactile_sensors = ['thumb', 'index', 'middle', 'ring']