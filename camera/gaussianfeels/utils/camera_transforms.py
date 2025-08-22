# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 3D transformations and geometry functions for camera operations

import typing
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchlie.functional as lieF
from scipy import interpolate
from scipy.spatial.transform import Rotation as R


def viz_boolean_img(x):
    x = x.to(torch.uint8)[..., None].repeat(1, 1, 3)
    return x.cpu().numpy() * 255


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
    r = R.from_euler(xyz, angles, degrees=degrees)
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r.as_matrix()
    return pose


@torch.jit.script
def ray_dirs_C(
    B: int,
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    depth_type: str = "z",
):
    """Generate camera ray directions in camera coordinate system"""
    c, r = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(H, device=device), indexing="ij"
    )
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = -(C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == "euclidean":
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1.0 / norm)[:, :, :, None]

    return dirs


def transform_points(points, T):
    """Transform 3D points using homogeneous transformation matrix"""
    points_T = (
        T
        @ torch.hstack(
            (points, torch.ones((points.shape[0], 1), device=points.device))
        ).T
    ).T
    points_T = points_T[:, :3] / points_T[:, -1][:, None]
    return points_T


def transform_points_batch(points, T):
    """Transform batched 3D points using homogeneous transformation matrices"""
    points_one = torch.hstack(
        (points, torch.ones((points.shape[0], 1), device=points.device, dtype=T.dtype))
    )  # [B * S, 4]
    points_T = torch.bmm(T, points_one.unsqueeze(2)).squeeze()
    points_T = points_T[:, :3] / points_T[:, -1][:, None]
    return points_T


def transform_points_np(points, T):
    """NumPy version of point transformation"""
    points_T = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    points_T = points_T[:, 0:3] / points_T[:, -1][:, None]
    points_T = points_T.reshape(-1, 3).astype(np.float32)
    return points_T


@torch.jit.script
def depth_image_to_point_cloud_GPU(
    depth,
    width: float,
    height: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
):
    """
    Convert depth image to 3D point cloud on GPU
    Based on https://github.com/PKU-MARL/DexterousHands/blob/main/docs/point-cloud.md
    """
    u = torch.arange(0, width, device=device)
    v = torch.arange(0, height, device=device)
    v2, u2 = torch.meshgrid(v, u, indexing="ij")

    Z = depth
    X = -(u2 - cx) / fx * Z
    Y = (v2 - cy) / fy * Z

    points = torch.dstack((X, Y, Z))
    return points


def point_cloud_to_image_plane(
    points,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
):
    """
    Project a set of 3D points to a camera image plane.
    """
    (X, Y, Z) = torch.unbind(points, dim=1)
    u = -(X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    u = torch.clamp(u, 0, width - 1)
    v = torch.clamp(v, 0, height - 1)
    depth = torch.stack((u, v), dim=1).to(torch.int32)
    return depth


@torch.jit.script
def origin_dirs_W(T_WC: torch.Tensor, dirs_C: torch.Tensor):
    """Transform camera ray directions to world coordinates"""
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)  # rotation
    origins = T_WC[:, :3, -1]
    return origins, dirs_W


def normalize(x):
    assert x.ndim == 1, "x must be a vector (ndim: 1)"
    return x / np.linalg.norm(x)


def look_at(
    eye,
    target: typing.Optional[typing.Any] = None,
    up: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.
    
    Parameters
    ----------
    eye: (3,) float
        Camera position.
    target: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).

    Returns
    -------
    R, t: rotation matrix and translation vector
        Camera to world transformation components.
    """
    eye = np.asarray(eye, dtype=float)

    if target is None:
        target = np.array([0, 0, 0], dtype=float)
    else:
        target = np.asarray(target, dtype=float)

    if up is None:
        up = np.array([0, 0, -1], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), "eye must be (3,) float"
    assert target.shape == (3,), "target must be (3,) float"
    assert up.shape == (3,), "up must be (3,) float"

    # create new axes
    z_axis: np.ndarray = normalize(target - eye)
    x_axis: np.ndarray = normalize(np.cross(up, z_axis))
    y_axis: np.ndarray = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R: np.ndarray = np.vstack((x_axis, y_axis, z_axis))
    t: np.ndarray = eye

    return R.T, t


def positionquat2tf(position_quat):
    """Convert position+quaternion to transformation matrix"""
    try:
        position_quat = np.atleast_2d(position_quat)
        # position_quat : N x 7
        N = position_quat.shape[0]
        T = np.zeros((4, 4, N))
        T[0:3, 0:3, :] = np.moveaxis(
            R.from_quat(position_quat[:, 3:]).as_matrix(), 0, -1
        )
        T[0:3, 3, :] = position_quat[:, :3].T
        T[3, 3, :] = 1
    except ValueError:
        print("Zero quat error!")
    return T.squeeze() if N == 1 else T


def interpolation(keypoints, n_points):
    """Interpolate keypoints to generate smooth trajectory"""
    tick, _ = interpolate.splprep(keypoints.T, s=0)
    points = interpolate.splev(np.linspace(0, 1, n_points), tick)
    points = np.array(points, dtype=np.float64).T
    return points


def backproject_pointclouds(depths, fx, fy, cx, cy, device="cuda"):
    """Backproject batch of depth images to point clouds"""
    pcs = []
    batch_size = depths.shape[0]
    for batch_i in range(batch_size):
        batch_depth = depths[batch_i]
        h, w = batch_depth.shape
        batch_depth = torch.tensor(batch_depth).to(device)
        pcd = depth_image_to_point_cloud_GPU(batch_depth, w, h, fx, fy, cx, cy, device)
        if torch.is_tensor(pcd):
            pcd = pcd.cpu().numpy()
        pc_flat = pcd.reshape(-1, 3)
        pcs.append(pc_flat)

    pcs = np.stack(pcs, axis=0)
    return pcs


def project_pointclouds(pcs, fx, fy, cx, cy, w, h, device="cuda"):
    """Project batch of point clouds to image coordinates"""
    depths = []
    batch_size = pcs.shape[0]
    for batch_i in range(batch_size):
        batch_points = pcs[batch_i]
        batch_points = torch.tensor(batch_points).to(device)
        depth = point_cloud_to_image_plane(batch_points, fx, fy, cx, cy, w, h)
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        depths.append(depth)
    depths = np.stack(depths, axis=0)
    return depths


def pc_bounds(pc):
    """Compute bounding box of point cloud"""
    min_x = np.min(pc[:, 0])
    max_x = np.max(pc[:, 0])
    min_y = np.min(pc[:, 1])
    max_y = np.max(pc[:, 1])
    min_z = np.min(pc[:, 2])
    max_z = np.max(pc[:, 2])
    extents = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    centroid = np.array(
        [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2]
    )

    return extents, centroid