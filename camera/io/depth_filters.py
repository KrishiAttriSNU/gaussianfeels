#!/usr/bin/env python3
"""
Depth preprocessing: undistortion and percentile-based filtering for robust vision processing.
"""

from typing import Tuple
import numpy as np
import cv2


def undistort_depth(depth: np.ndarray, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        np.array(distortion_coeffs),
        np.eye(3),
        camera_matrix,
        (W, H),
        cv2.CV_32FC1,
    )
    depth_ud = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)
    return depth_ud


def filter_depth_percentile(depth: np.ndarray, outlier_max_perc: float = 99.0,
                             outlier_min_perc: float = 1.0, cutoff: float = 2.0) -> np.ndarray:
    """Filter depth by removing extreme percentiles and enforcing a cutoff.
    Returns a boolean mask of valid depth.
    """
    d = np.abs(depth[depth != 0.0])
    if d.size == 0:
        return np.zeros_like(depth, dtype=bool)
    hi = np.percentile(d, outlier_max_perc)
    lo = np.percentile(d, outlier_min_perc)
    mask = (np.abs(depth) >= lo) & (np.abs(depth) <= min(hi, cutoff))
    return mask


