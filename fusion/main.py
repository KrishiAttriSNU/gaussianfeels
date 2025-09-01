#!/usr/bin/env python3.11
"""
Main entry point for Camera-Tactile Fusion
"""

import argparse
import os
import sys
import logging
import warnings

# Suppress OpenGL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="OpenGL")
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Use normal package imports; avoid absolute sys.path hacks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress OpenGL accelerate warnings
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

from .config import TactileFusionConfig
from .fusion_test import TactileFusionTest


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Camera-Tactile Fusion Test")
    from pathlib import Path as _P
    default_data = str(_P(__file__).resolve().parents[1] / 'data/feelsight/contactdb_rubber_duck/02')
    default_output = str(_P.cwd() / 'fusion_test_camera_tactile_results')
    parser.add_argument("--data-path", 
                       default=default_data,
                       help="Path to feelsight trial data")
    parser.add_argument("--max-frames", type=int, default=-1,
                       help="Maximum frames to process (-1 for all frames)")
    parser.add_argument("--output", 
                       default=default_output,
                       help="Output directory for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device for computation")
    
    args = parser.parse_args()
    
    # Create config for camera-tactile fusion
    config = TactileFusionConfig(
        trial_path=args.data_path,
        output_dir=args.output,
        max_frames=args.max_frames,
        device=args.device
    )
    
    # Run camera-tactile fusion test
    test = TactileFusionTest(config)
    results = test.run_complete_tactile_fusion_test()
    
    if results['overall_success']:
        logger.info("✅ CAMERA-TACTILE FUSION TEST COMPLETED SUCCESSFULLY!")
        return 0
    else:
        logger.error("❌ CAMERA-TACTILE FUSION TEST FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())