#!/usr/bin/env python3.11
"""
Main entry point for camera-tactile fusion processing
"""

import argparse
import sys
from pathlib import Path

# Avoid sys.path manipulation; assume package installation or repo root execution

from .config import TactileFusionConfig
from .fusion_test import TactileFusionTest


def main():
    parser = argparse.ArgumentParser(description='Camera-Tactile Fusion with TouchVIT')
    
    parser.add_argument('--data-path', type=str, 
                       default=str(Path(__file__).resolve().parents[1] / 'data/feelsight/contactdb_rubber_duck/02'),
                       help='Path to trial data directory')
    
    parser.add_argument('--max-frames', type=int, default=-1,
                       help='Maximum frames to process (-1 for all)')
    
    parser.add_argument('--output-dir', type=str, 
                       default=str(Path.cwd() / 'fusion_test_tactile_results'),
                       help='Output directory for results')
    
    parser.add_argument('--all-fingers', action='store_true',
                       help='Enable all fingers including problematic ones (default: only reliable fingers)')
    
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for processing')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = TactileFusionConfig(
        trial_path=args.data_path,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        all_fingers=args.all_fingers,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Run fusion test
    test = TactileFusionTest(config)
    results = test.run_complete_tactile_fusion_test()
    
    print(f"\nFUSION TEST COMPLETE!")
    if 'success' not in results:
        raise KeyError("results dictionary missing required 'success' key")
    print(f"Success: {results['success']}")
    if results['success']:
        if 'total_points' not in results:
            raise KeyError("Successful results missing required 'total_points' key")
        if 'total_time' not in results:
            raise KeyError("Successful results missing required 'total_time' key")
        print(f"Total points: {results['total_points']:,}")
        print(f"Processing time: {results['total_time']:.1f}s")


if __name__ == '__main__':
    main()