#!/usr/bin/env python3.11
"""
Camera-Tactile Fusion Module

This module implements tactile processing pipeline for camera-tactile fusion
with TouchVIT depth prediction, background subtraction and contact detection.
"""

from .config import TactileFusionConfig
from .tactile_processor import TactileProcessor
from .fusion_test import TactileFusionTest

__all__ = [
    'TactileFusionConfig',
    'TactileProcessor', 
    'TactileFusionTest'
]

__version__ = '1.0.0'
__author__ = 'Fusion Team'
__description__ = 'Camera-Tactile Fusion with TouchVIT Implementation'