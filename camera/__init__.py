"""
Camera module for GaussianFeels

Multi-camera RGB-D data acquisition and processing pipeline.
"""

from .camera_pipeline import CameraPipeline
from .io.rgbd_dataset import RGBDDataset
from .io.rgbd_sensor import Sensor as RGBDSensor
from .io.depth_filters import filter_depth_percentile, undistort_depth
from .io.image_transforms import DepthTransform, BGRtoRGB, DepthScale
from .io.sam_segmenter import SAMSegmenter

__all__ = [
    'CameraPipeline',
    'RGBDDataset', 
    'RGBDSensor',
    'filter_depth_percentile',
    'undistort_depth',
    'DepthTransform',
    'BGRtoRGB', 
    'DepthScale',
    'SAMSegmenter'
]