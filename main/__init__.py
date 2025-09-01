"""
GaussianFeels: Complete Pipeline for Visuo-Tactile Gaussian Splatting

This package provides a unified interface for multi-modal Gaussian splatting
with vision and tactile sensing, replacing neural fields with explicit 3D Gaussians.
"""

__version__ = "0.1.0"
__author__ = "GaussianFeels Team"

# Core exports
from .config import GaussianFeelsConfig
from .trainer import GaussianTrainer
from .datasets import DatasetRegistry

# Compatibility alias for legacy code
GaussianSplattingTrainer = GaussianTrainer

__all__ = [
    "GaussianFeelsConfig",
    "GaussianTrainer",
    "GaussianSplattingTrainer", 
    "DatasetRegistry",
]