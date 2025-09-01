"""
Shared point cloud processing functions.

Consolidated from multiple modules to provide a single source of truth
for point cloud operations including projection, backprojection, and processing.
"""

import numpy as np
import torch
import torch.nn.functional as F


def depth_image_to_point_cloud_GPU(
    depth,
    fx, fy, cx, cy,
    T_WC=None,
    mask=None,
    color=None,
    device="cuda"
):
    """Convert depth image to 3D point cloud on GPU
    
    Args:
        depth: (H, W) depth image
        fx, fy, cx, cy: Camera intrinsics
        T_WC: Optional (4, 4) world-to-camera transform
        mask: Optional (H, W) boolean mask
        color: Optional (H, W, 3) color image
        device: Target device
    
    Returns:
        points: (N, 3) 3D points
        colors: (N, 3) colors if color provided, else None
    """
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).to(device)
    
    H, W = depth.shape
    
    # Create coordinate grids
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )
    
    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device)
        valid_mask = mask & (depth > 0)
    else:
        valid_mask = depth > 0
    
    # Get valid coordinates
    valid_u = u[valid_mask]
    valid_v = v[valid_mask] 
    valid_depth = depth[valid_mask]
    
    if len(valid_depth) == 0:
        return torch.empty(0, 3, device=device), None
    
    # Backproject to 3D
    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth
    
    points = torch.stack([x, y, z], dim=1)
    
    # Transform to world coordinates if transform provided
    if T_WC is not None:
        if isinstance(T_WC, np.ndarray):
            T_WC = torch.from_numpy(T_WC).to(device)
        # Add homogeneous coordinate
        points_homo = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)
        points_world = (T_WC @ points_homo.T).T
        points = points_world[:, :3]
    
    # Extract colors if provided
    colors = None
    if color is not None:
        if isinstance(color, np.ndarray):
            color = torch.from_numpy(color).to(device)
        colors = color[valid_mask] / 255.0 if color.max() > 1.0 else color[valid_mask]
    
    return points, colors


def backproject_pointclouds(depths, fx, fy, cx, cy, device="cuda"):
    """Backproject depth images to point clouds
    
    Args:
        depths: (B, H, W) depth images
        fx, fy, cx, cy: Camera intrinsics
        device: Target device
    
    Returns:
        points: (B, H, W, 3) 3D points
    """
    B, H, W = depths.shape
    
    # Create coordinate grids
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )
    
    # Expand to batch
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)
    
    # Backproject
    x = (u - cx) * depths / fx
    y = (v - cy) * depths / fy
    z = depths
    
    points = torch.stack([x, y, z], dim=-1)
    return points


def project_pointclouds(pcs, fx, fy, cx, cy, w, h, device="cuda"):
    """Project 3D point clouds to image coordinates
    
    Args:
        pcs: (B, N, 3) or (N, 3) point clouds
        fx, fy, cx, cy: Camera intrinsics  
        w, h: Image dimensions
        device: Target device
    
    Returns:
        coords: (B, N, 2) or (N, 2) image coordinates
        depths: (B, N) or (N,) depths
        valid_mask: (B, N) or (N,) mask for valid projections
    """
    if pcs.dim() == 2:
        pcs = pcs.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, N, _ = pcs.shape
    
    # Project to image plane
    x_img = fx * pcs[..., 0] / pcs[..., 2] + cx
    y_img = fy * pcs[..., 1] / pcs[..., 2] + cy
    
    coords = torch.stack([x_img, y_img], dim=-1)
    depths = pcs[..., 2]
    
    # Check bounds and positive depth
    valid_mask = (
        (depths > 0) & 
        (x_img >= 0) & (x_img < w) & 
        (y_img >= 0) & (y_img < h)
    )
    
    if squeeze_output:
        coords = coords.squeeze(0)
        depths = depths.squeeze(0) 
        valid_mask = valid_mask.squeeze(0)
    
    return coords, depths, valid_mask


def point_cloud_to_image_plane(
    pc, T_WC, fx, fy, cx, cy, H, W, device="cuda"
):
    """Project point cloud to image plane with camera transform
    
    Args:
        pc: (N, 3) point cloud in world coordinates
        T_WC: (4, 4) world-to-camera transform
        fx, fy, cx, cy: Camera intrinsics
        H, W: Image dimensions
        device: Target device
    
    Returns:
        coords: (N, 2) image coordinates
        depths: (N,) depths in camera frame
        valid_mask: (N,) mask for valid projections
    """
    if isinstance(pc, np.ndarray):
        pc = torch.from_numpy(pc).to(device)
    if isinstance(T_WC, np.ndarray):
        T_WC = torch.from_numpy(T_WC).to(device)
    
    # Transform to camera coordinates
    pc_homo = torch.cat([pc, torch.ones(len(pc), 1, device=device)], dim=1)
    pc_cam_homo = (T_WC @ pc_homo.T).T
    pc_cam = pc_cam_homo[:, :3]
    
    # Project to image
    return project_pointclouds(pc_cam, fx, fy, cx, cy, W, H, device)


__all__ = [
    'depth_image_to_point_cloud_GPU',
    'backproject_pointclouds',
    'project_pointclouds', 
    'point_cloud_to_image_plane'
]