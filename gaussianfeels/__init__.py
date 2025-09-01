"""
GaussianFeels: Tactile-guided Multi-modal 3D Reconstruction

A unified framework for real-time 3D reconstruction using
RGB-D cameras and tactile sensors with Gaussian splatting.
"""

__version__ = "0.1.0"
__author__ = "Krishi Attri"

# Main components
from main.main import main
from main.server import main as server_main
from main.evaluation import cli_main as eval_main

# Core modules
from gaussian.core.gaussian_field import ObjectGaussianMap
from fusion.core.fusion import ObjectCentricFusion
from camera.camera_pipeline import CameraPipeline

__all__ = [
    # Entry points
    'main', 'server_main', 'eval_main',
    # Core components
    'ObjectGaussianMap', 'ObjectCentricFusion', 'CameraPipeline',
    # Metadata
    '__version__', '__author__'
]