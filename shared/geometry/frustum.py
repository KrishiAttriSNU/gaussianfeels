# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared camera frustum operations.

Consolidated frustum and visibility checking functions used across
camera and tactile processing modules.
"""

import numpy as np
import torch

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def get_frustum_normals(R_WC, H, W, fx, fy, cx, cy):
    """Compute camera frustum normal vectors in world coordinates
    
    Args:
        R_WC: (3, 3) camera-to-world rotation matrix
        H, W: Image height and width
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        frustum_normals: (4, 3) normal vectors for frustum planes
    """
    # Corner pixel coordinates (top-left, top-right, bottom-right, bottom-left)
    c = np.array([0, W, W, 0])
    r = np.array([0, 0, H, H])
    
    # Convert to camera ray directions
    x = (c - cx) / fx
    y = (r - cy) / fy
    corner_dirs_C = np.vstack((x, y, np.ones(4))).T
    
    # Transform to world coordinates
    corner_dirs_W = (R_WC * corner_dirs_C[..., None, :]).sum(axis=-1)

    # Compute frustum plane normals
    frustum_normals = np.empty((4, 3))
    frustum_normals[0] = np.cross(corner_dirs_W[0], corner_dirs_W[1])  # Top plane
    frustum_normals[1] = np.cross(corner_dirs_W[1], corner_dirs_W[2])  # Right plane  
    frustum_normals[2] = np.cross(corner_dirs_W[2], corner_dirs_W[3])  # Bottom plane
    frustum_normals[3] = np.cross(corner_dirs_W[3], corner_dirs_W[0])  # Left plane
    
    # Normalize
    frustum_normals = frustum_normals / np.linalg.norm(frustum_normals, axis=1)[:, None]

    return frustum_normals


def check_inside_frustum(points, cam_center, frustum_normals):
    """Check if points are inside camera frustum
    
    Args:
        points: (N, 3) 3D points to test
        cam_center: (3,) camera center position
        frustum_normals: (4, 3) frustum plane normals
    
    Returns:
        inside_mask: (N,) boolean mask indicating which points are inside
    """
    # Vector from camera center to points
    pts = points - cam_center
    
    # Dot product with each frustum normal
    dots = np.dot(pts, frustum_normals.T)
    
    # Point is inside if all dot products are positive
    return (dots >= 0).all(axis=1)


def is_visible(
    points, T_WC, depth, H, W, fx, fy, cx, cy, trunc=0.2, use_projection=True
):
    """Check if points are visible in camera frame
    
    Args:
        points: (N, 3) 3D points in world coordinates
        T_WC: (4, 4) world-to-camera transformation matrix
        depth: (H, W) depth image
        H, W: Image dimensions  
        fx, fy, cx, cy: Camera intrinsics
        trunc: Truncation distance behind surface (meters)
        use_projection: Whether to use depth buffer occlusion check
    
    Returns:
        visible_mask: (N,) boolean mask indicating visible points
    """
    # Extract camera pose components
    R_WC = T_WC[:3, :3]
    t_WC = T_WC[:3, 3]
    
    # Check frustum visibility
    frustum_normals = get_frustum_normals(R_WC, H, W, fx, fy, cx, cy)
    frustum_mask = check_inside_frustum(points, t_WC, frustum_normals)
    
    if not use_projection:
        return frustum_mask
    
    # Transform points to camera coordinates
    points_homo = np.column_stack([points, np.ones(len(points))])
    points_cam = (T_WC @ points_homo.T).T[:, :3]
    
    # Project to image coordinates
    x_img = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    y_img = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    
    # Check image bounds
    bounds_mask = (
        (x_img >= 0) & (x_img < W) &
        (y_img >= 0) & (y_img < H) &
        (points_cam[:, 2] > 0)  # Positive depth
    )
    
    # Check depth buffer occlusion
    occlusion_mask = np.ones(len(points), dtype=bool)
    
    if depth is not None:
        valid_indices = np.where(bounds_mask)[0]
        for i in valid_indices:
            # Get depth at projected pixel
            u = int(np.round(x_img[i]))
            v = int(np.round(y_img[i]))
            
            # Clamp to valid range
            u = max(0, min(W-1, u))
            v = max(0, min(H-1, v))
            
            surface_depth = depth[v, u]
            point_depth = points_cam[i, 2]
            
            # Point is visible if within truncation distance of surface
            if surface_depth > 0:
                occlusion_mask[i] = point_depth <= surface_depth + trunc
    
    return frustum_mask & bounds_mask & occlusion_mask


def test_inside_frustum(T_WC, depth):
    """Test function for frustum visibility (kept for compatibility)
    
    Args:
        T_WC: (4, 4) world-to-camera transform
        depth: (H, W) depth image
    """
    # Production frustum culling test with comprehensive validation
    try:
        H, W = depth.shape
        
        # Validate inputs
        if T_WC.shape != (4, 4):
            raise ValueError(f"Invalid T_WC shape: {T_WC.shape}. Expected (4, 4)")
        if depth.size == 0:
            raise ValueError("Empty depth image provided")
        
        # Extract camera parameters from depth image statistics
        valid_depth = depth[depth > 0]
        if len(valid_depth) == 0:
            raise ValueError("No valid depth values in image")
        
        # Estimate camera parameters based on depth statistics
        depth_range = valid_depth.max() - valid_depth.min()
        fx = fy = W * 0.8  # Realistic focal length estimate
        cx, cy = W/2, H/2
        
        # Generate comprehensive test points in camera field of view
        n_test_points = 500
        
        # Sample points at different depth layers
        depth_layers = np.linspace(valid_depth.min(), valid_depth.max(), 5)
        test_points = []
        
        for depth_val in depth_layers:
            # Generate points in a grid pattern at this depth
            n_per_layer = n_test_points // len(depth_layers)
            
            # Convert image coordinates to world coordinates
            u_coords = np.linspace(0, W-1, int(np.sqrt(n_per_layer)))
            v_coords = np.linspace(0, H-1, int(np.sqrt(n_per_layer)))
            u_grid, v_grid = np.meshgrid(u_coords, v_coords)
            
            # Convert to camera coordinates
            x_cam = (u_grid.flatten() - cx) * depth_val / fx
            y_cam = (v_grid.flatten() - cy) * depth_val / fy
            z_cam = np.full_like(x_cam, depth_val)
            
            # Transform to world coordinates
            points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)
            points_world = (T_WC @ points_cam.T)[:3, :].T
            test_points.append(points_world)
        
        test_points = np.vstack(test_points)
        
        # Perform comprehensive visibility test
        visible_mask = is_visible(
            test_points, T_WC, depth, H, W, fx, fy, cx, cy, trunc=0.1
        )
        
        # Additional validation: check frustum consistency
        if visible_mask.sum() == 0:
            raise RuntimeError("No points visible - possible frustum culling error")
        
        # Return visibility statistics for debugging
        visibility_stats = {
            'total_points': len(test_points),
            'visible_points': visible_mask.sum(),
            'visibility_ratio': visible_mask.mean(),
            'depth_range': (valid_depth.min(), valid_depth.max()),
            'camera_params': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        }
        
        return visible_mask, visibility_stats
        
    except Exception as e:
        raise RuntimeError(f"Frustum culling test failed: {e}") from e


__all__ = [
    'get_frustum_normals',
    'check_inside_frustum',
    'is_visible', 
    'test_inside_frustum'
]