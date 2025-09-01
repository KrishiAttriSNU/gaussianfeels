# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared 3D transformations and geometry functions.

Consolidated from multiple modules to provide a single source of truth
for all geometric transformation operations.
"""

import typing
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchlie.functional as lieF
from scipy import interpolate
from scipy.spatial.transform import Rotation as R

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    
try:
    import PIL
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from skimage import morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def viz_boolean_img(x):
    """Visualize boolean image as RGB"""
    x = x.to(torch.uint8)[..., None].repeat(1, 1, 3)
    return x.cpu().numpy() * 255


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
    """Convert Euler angles to 4x4 transformation matrix"""
    r = R.from_euler(xyz, angles, degrees=degrees)
    pose = np.eye(4)
    pose[:3, :3] = r.as_matrix()
    pose[:3, 3] = translation
    return pose


def normalize(x):
    """Normalize vector to unit length"""
    return x / torch.norm(x, dim=-1, keepdim=True)


def transform_points(points, T):
    """Transform points using 4x4 transformation matrix"""
    if points.numel() == 0:
        return points
    # Add homogeneous coordinate
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    # Transform
    points_transformed = (T @ points_homo.T).T
    return points_transformed[:, :3]


def transform_points_batch(points, T):
    """Transform batch of points using batch of transformation matrices
    
    Args:
        points: (B, N, 3) or (N, 3)
        T: (B, 4, 4) or (4, 4)
    """
    if points.dim() == 2 and T.dim() == 2:
        return transform_points(points, T)
    
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if T.dim() == 2:
        T = T.unsqueeze(0)
    
    B, N, _ = points.shape
    # Add homogeneous coordinate
    points_homo = torch.cat([points, torch.ones(B, N, 1, device=points.device)], dim=-1)
    # Transform: (B, 4, 4) @ (B, N, 4).transpose(-2, -1) -> (B, 4, N)
    points_transformed = torch.bmm(T, points_homo.transpose(-2, -1))
    # Back to (B, N, 3)
    return points_transformed.transpose(-2, -1)[..., :3]


def transform_points_np(points, T):
    """Transform points using NumPy (for compatibility)"""
    if len(points) == 0:
        return points
    # Add homogeneous coordinate
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Transform
    points_transformed = (T @ points_homo.T).T
    return points_transformed[:, :3]


def ray_dirs_C(
    B: int, H: int, W: int, fx: float, fy: float, cx: float, cy: float, device: str = "cuda"
):
    """Generate camera ray directions in camera coordinate system.
    
    Args:
        B: Batch size
        H: Image height
        W: Image width
        fx, fy: Focal lengths
        cx, cy: Principal point
        device: Torch device
    """
    u, v = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy"
    )
    
    # Convert to camera coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    
    # Stack to (H, W, 3) and expand to batch
    dirs_C = torch.stack([x, y, z], dim=-1)
    dirs_C = dirs_C.unsqueeze(0).expand(B, -1, -1, -1)
    
    return dirs_C


def origin_dirs_W(T_WC: torch.Tensor, dirs_C: torch.Tensor):
    """Transform camera rays to world coordinates"""
    # Extract rotation and translation
    R_WC = T_WC[..., :3, :3]  # (..., 3, 3)
    t_WC = T_WC[..., :3, 3]   # (..., 3)
    
    # Transform directions (rotation only)
    dirs_W = torch.einsum('...ij,...hwj->...hwi', R_WC, dirs_C)
    
    # Origin is translation (same for all pixels)
    origin_shape = dirs_W.shape[:-1] + (3,)
    origins_W = t_WC.view(t_WC.shape + (1, 1)).expand(origin_shape)
    
    return origins_W, dirs_W


def rays_box_frame(origins, dirs_W, box_transform):
    """Transform ray origins and directions to box coordinate frame
    
    Args:
        origins: Ray origin points
        dirs_W: Ray directions in world coordinates
        box_transform: Box transformation matrix
        
    Returns:
        origins_g: Ray origins in box frame
        dirs_g: Ray directions in box frame
    """
    T_GW = torch.inverse(box_transform)
    origins_g = transform_points(origins, T_GW)
    dirs_g = torch.einsum("ij,mj-> mi", T_GW[:3, :3], dirs_W)
    
    return origins_g, dirs_g


def ray_box_intersection(
    origins,
    dirs_W,
    box_extents,
    box_transform,
    visualize=False,
):
    """Compute ray-box intersection points
    
    Calculates the depth values at which rays enter and exit a 3D bounding box.
    
    Args:
        origins: Ray origin points
        dirs_W: Ray directions in world coordinates
        box_extents: Box dimensions (x, y, z)
        box_transform: Box transformation matrix
        visualize: Whether to show debug visualization
        
    Returns:
        z_near: Depth values where rays enter the box (NaN for no intersection)
        z_far: Depth values where rays exit the box (NaN for no intersection)
    """
    # Normal direction for each face of axis aligned box
    normals = torch.eye(3).repeat_interleave(2, 0).to(origins.device)
    normals[1::2] *= -1
    normals = normals[None, ...]

    # Point at the centre of each face of the box
    plane_pts = torch.diag(box_extents / 2).repeat_interleave(2, 0)
    plane_pts[1::2] *= -1
    plane_pts = plane_pts[None, ...]

    # Get origins and ray directions in the box frame
    origins_g, dirs_g = rays_box_frame(origins, dirs_W, box_transform)

    # Intersection depth along ray with plane
    numerator = torch.hstack(
        [box_extents[None, :] / 2 - origins_g, box_extents[None, :] / 2 + origins_g]
    )
    denom = torch.hstack([dirs_g, -dirs_g])
    intsct_depth = numerator / denom

    # Intersection points
    intsct_pts = origins_g[:, None, :] + intsct_depth[:, :, None] * dirs_g[:, None, :]

    # Check intersection points within the face bounds
    check = torch.abs(intsct_pts) <= (box_extents / 2)
    check[:, torch.arange(6), [0, 1, 2, 0, 1, 2]] = True
    intersects_face = (check).all(dim=-1)
    intersects_box = intersects_face.sum(dim=-1) == 2
    # Remove face intersections if there are not exactly 2 for the ray
    intersects_face = intersects_face * intersects_box[:, None]

    z_range = intsct_depth[intersects_face].reshape(-1, 2)
    z_range = z_range.sort(dim=-1)[0]

    n_rays = dirs_W.shape[0]
    z_near = torch.full([n_rays], torch.nan, device=origins.device)
    z_far = torch.full([n_rays], torch.nan, device=origins.device)
    z_near[intersects_box] = z_range[:, 0]
    z_far[intersects_box] = z_range[:, 1]

    if visualize:
        try:
            import trimesh
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.primitives.Box(extents=box_extents.cpu()))
            scene.add_geometry(
                trimesh.PointCloud(plane_pts.view(-1, 3).cpu(), colors=[255, 0, 0])
            )
            scene.add_geometry(
                trimesh.PointCloud(origins_g.detach().cpu(), colors=[0, 0, 255])
            )
            line_starts_g = origins_g - dirs_g * 1e-6
            line_ends_g = origins_g + dirs_g * 0.2
            path_g = trimesh.load_path(
                torch.cat((line_starts_g[:, None], line_ends_g[:, None]), dim=1).cpu()
            )
            scene.add_geometry(path_g)
            scene.add_geometry(
                trimesh.PointCloud(
                    intsct_pts[intersects_face].cpu().view(-1, 3), colors=[0, 255, 0]
                )
            )
            scene.show()
        except ImportError:
            print("Warning: trimesh not available for visualization")

    return z_near, z_far


__all__ = [
    'viz_boolean_img',
    'euler2matrix', 
    'normalize',
    'transform_points',
    'transform_points_batch',
    'transform_points_np',
    'ray_dirs_C',
    'origin_dirs_W',
    'rays_box_frame',
    'ray_box_intersection'
]