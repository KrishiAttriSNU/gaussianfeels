# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Camera geometry and frustum calculations

import numpy as np
import torch


def get_frustum_normals(R_WC, H, W, fx, fy, cx, cy):
    """Calculate the four frustum normal vectors for a camera"""
    c = np.array([0, W, W, 0])
    r = np.array([0, 0, H, H])
    x = (c - cx) / fx
    y = (r - cy) / fy
    corner_dirs_C = np.vstack((x, y, np.ones(4))).T
    corner_dirs_W = (R_WC * corner_dirs_C[..., None, :]).sum(axis=-1)

    frustum_normals = np.empty((4, 3))
    frustum_normals[0] = np.cross(corner_dirs_W[0], corner_dirs_W[1])
    frustum_normals[1] = np.cross(corner_dirs_W[1], corner_dirs_W[2])
    frustum_normals[2] = np.cross(corner_dirs_W[2], corner_dirs_W[3])
    frustum_normals[3] = np.cross(corner_dirs_W[3], corner_dirs_W[0])
    frustum_normals = frustum_normals / np.linalg.norm(frustum_normals, axis=1)[:, None]

    return frustum_normals


def check_inside_frustum(points, cam_center, frustum_normals):
    """Check if points are inside the camera frustum.
    
    For a point to be within the frustum, the projection on each normal
    vector must be positive.
    
    Args:
        points: 3D points to check [N, 3]
        cam_center: Camera center position [3]
        frustum_normals: Frustum normal vectors [4, 3]
        
    Returns:
        Boolean mask indicating which points are inside frustum [N]
    """
    pts = points - cam_center
    dots = np.dot(pts, frustum_normals.T)
    return (dots >= 0).all(axis=1)


def is_visible(
    points, T_WC, depth, H, W, fx, fy, cx, cy, trunc=0.2, use_projection=True
):
    """Check if 3D points are visible in the given camera frame.
    
    Points up to 'trunc' metres behind the surface count as visible.
    
    Args:
        points: 3D points in world coordinates [N, 3]
        T_WC: World-to-camera transformation matrix [4, 4]
        depth: Depth image [H, W]
        H, W: Image height and width
        fx, fy, cx, cy: Camera intrinsic parameters
        trunc: Truncation distance behind surface (meters)
        use_projection: Use image projection (True) or frustum check (False)
        
    Returns:
        Boolean mask indicating visible points [N]
    """
    # Forward project points to camera coordinates
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    ones = np.ones([len(points), 1])
    homog_points = np.concatenate((points, ones), axis=-1)
    points_C = (np.linalg.inv(T_WC) @ homog_points.T)[:3]
    uv = K @ points_C
    z = uv[2]
    uv = uv[:2] / z
    uv = uv.T

    if use_projection:
        # Check if projected points are within image bounds
        x_valid = np.logical_and(uv[:, 0] > 0, uv[:, 0] < W)
        y_valid = np.logical_and(uv[:, 1] > 0, uv[:, 1] < H)
        xy_valid = np.logical_and(x_valid, y_valid)
    else:
        # Use frustum geometry for validity check
        R_WC = T_WC[:3, :3]
        cam_center = T_WC[:3, 3]
        frustum_normals = get_frustum_normals(R_WC, H, W, fx, fy, cx, cy)
        xy_valid = check_inside_frustum(points, cam_center, frustum_normals)

    # Check depth validity
    uv = uv.astype(int)
    depth_vals = depth[uv[xy_valid, 1], uv[xy_valid, 0]]
    max_depths = np.full(len(uv), -np.inf)
    max_depths[xy_valid] = depth_vals + trunc
    z_valid = np.logical_and(z > 0, z < max_depths)

    inside = np.logical_and(xy_valid, z_valid)

    return inside