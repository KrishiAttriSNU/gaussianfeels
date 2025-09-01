"""
Shared geometry utilities for GaussianFeels.

This module consolidates geometric operations from across the codebase
into a single, well-tested location.
"""

from .transforms import (
    viz_boolean_img,
    euler2matrix,
    normalize,
    transform_points,
    transform_points_batch,
    transform_points_np,
    ray_dirs_C,
    origin_dirs_W,
    rays_box_frame,
    ray_box_intersection
)

from .pointcloud import (
    depth_image_to_point_cloud_GPU,
    backproject_pointclouds,
    project_pointclouds,
    point_cloud_to_image_plane
)

from .frustum import (
    get_frustum_normals,
    check_inside_frustum,
    is_visible,
    test_inside_frustum
)

__all__ = [
    # Transform functions
    'viz_boolean_img',
    'euler2matrix',
    'normalize',
    'transform_points',
    'transform_points_batch', 
    'transform_points_np',
    'ray_dirs_C',
    'origin_dirs_W',
    'rays_box_frame',
    'ray_box_intersection',
    
    # Point cloud functions
    'depth_image_to_point_cloud_GPU',
    'backproject_pointclouds',
    'project_pointclouds',
    'point_cloud_to_image_plane',
    
    # Frustum functions
    'get_frustum_normals',
    'check_inside_frustum', 
    'is_visible',
    'test_inside_frustum'
]