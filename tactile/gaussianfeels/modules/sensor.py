# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import dill as pickle
import git
import numpy as np
import torch
from omegaconf import DictConfig
from termcolor import cprint
from torch import nn
from torchvision import transforms

from shared import geometry
# TactileDepth import - optional due to compatibility issues
try:
    from tactile.gaussianfeels.contrib.tactile_transformer import TactileDepth
    TACTILE_DEPTH_AVAILABLE = True
except ImportError:
    TactileDepth = None
    TACTILE_DEPTH_AVAILABLE = False
from tactile.gaussianfeels.datasets import dataset
from camera.io import image_transforms
from shared.io.frame_data import FrameData
from tactile.gaussianfeels.modules import sample

# quicklink to the root and folder directories
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Sensor(nn.Module):
    def __init__(
        self,
        cfg_sensor: DictConfig,
        device: str = "cuda",
    ):
        super(Sensor, self).__init__()
        self.device = device
        self.sensor_name = cfg_sensor.name
        cprint(f"Adding Sensor: {self.sensor_name}", color="yellow")

        self.end = False

        self.kf_min_loss = cfg_sensor.kf_min_loss  # threshold for adding to kf set

        # sampling parameters
        self.loss_ratio = cfg_sensor.sampling.loss_ratio
        self.free_space_ratio = cfg_sensor.sampling.free_space_ratio
        self.max_depth = -cfg_sensor.sampling.depth_range[1]  # -z towards the object
        self.min_depth = -cfg_sensor.sampling.depth_range[0]
        self.dist_behind_surf = cfg_sensor.sampling.dist_behind_surf
        self.n_rays = cfg_sensor.sampling.n_rays
        self.n_strat_samples = cfg_sensor.sampling.n_strat_samples
        self.n_surf_samples = cfg_sensor.sampling.n_surf_samples
        self.surface_samples_offset = cfg_sensor.sampling.surface_samples_offset

        # vizualisation/rendering parameters
        self.reduce_factor = cfg_sensor.viz.reduce_factor
        self.reduce_factor_up = cfg_sensor.viz.reduce_factor_up

    def set_intrinsics(self, intrinsics: dict):
        self.W = intrinsics["w"]
        self.H = intrinsics["h"]
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        print(
            f"{self.sensor_name} intrinsics: {self.W}x{self.H}, fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}"
        )
        self.camera_matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.distortion_coeffs = []
        if "k1" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k1"])
        if "k2" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k2"])
        if "p1" in intrinsics:
            self.distortion_coeffs.append(intrinsics["p1"])
        if "p2" in intrinsics:
            self.distortion_coeffs.append(intrinsics["p2"])
        if "k3" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k3"])

        self.set_viz_cams()
        self.set_directions()
        self.set_active_sampling_params()

    def set_viz_cams(self):
        reduce_factor = self.reduce_factor
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        reduce_factor_up = self.reduce_factor_up
        self.H_vis_up = self.H // reduce_factor_up
        self.W_vis_up = self.W // reduce_factor_up
        self.fx_vis_up = self.fx / reduce_factor_up
        self.fy_vis_up = self.fy / reduce_factor_up
        self.cx_vis_up = self.cx / reduce_factor_up
        self.cy_vis_up = self.cy / reduce_factor_up

    def set_directions(self):
        self.dirs_C = geometry.transform.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        self.dirs_C_vis = geometry.transform.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

        self.dirs_C_vis_up = geometry.transform.ray_dirs_C(
            1,
            self.H_vis_up,
            self.W_vis_up,
            self.fx_vis_up,
            self.fy_vis_up,
            self.cx_vis_up,
            self.cy_vis_up,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

    def set_active_sampling_params(self):
        # for active_sampling
        self.loss_approx_factor = 8
        w_block = self.W // self.loss_approx_factor
        h_block = self.H // self.loss_approx_factor
        increments_w = (
            torch.arange(self.loss_approx_factor, device=self.device) * w_block
        )
        increments_h = (
            torch.arange(self.loss_approx_factor, device=self.device) * h_block
        )
        c, r = torch.meshgrid(increments_w, increments_h)
        c, r = c.t(), r.t()
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
        surface_samples_offset=None,
        box_extents=None,
        box_transform=None,
        free_space_ratio=None,
        grad=False,
        viz_masks=False,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        if n_rays is None:
            n_rays = self.n_rays
        if dist_behind_surf is None:
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            n_surf_samples = self.n_surf_samples
        if surface_samples_offset is None:
            surface_samples_offset = self.surface_samples_offset

        n_frames = depth_batch.shape[0]

        free_space_ratio = (
            self.free_space_ratio if free_space_ratio is None else free_space_ratio
        )
        free_space_rays = int(n_rays * free_space_ratio)
        object_rays = n_rays - free_space_rays

        # box extents not needed for tactile sensors
        box_extents, box_transform = None, None

        # sample on the object surface (useful for tracking)
        obj_ray_sample = None
        obj_mask = ~torch.isnan(depth_batch)
        if object_rays:
            indices_b, indices_h, indices_w = sample.sample_pixels(
                object_rays,
                n_frames,
                self.H,
                self.W,
                device=self.device,
                mask=obj_mask,
            )
            (
                dirs_C_sample,
                T_WC_sample,
                depth_sample,
                norm_sample,
            ) = sample.get_batch_data(
                T_WC_batch,
                self.dirs_C,
                indices_b,
                indices_h,
                indices_w,
                depth_batch=depth_batch,
                norm_batch=norm_batch,
            )
            # use min and max depth for object rays
            max_depth = (
                depth_sample + torch.sign(depth_sample + 1e-8) * dist_behind_surf
            )
            pc, z_vals, valid_ray = sample.sample_along_rays(
                T_WC_sample,
                dirs_C_sample,
                n_strat_samples,
                n_surf_samples=n_surf_samples,
                surf_samples_offset=surface_samples_offset,
                min_depth=self.min_depth,
                max_depth=max_depth,
                box_extents=box_extents,
                box_transform=box_transform,
                gt_depth=depth_sample,
                grad=grad,
            )  # pc: (num_samples, N + M + 1, 3)

            if valid_ray is not None:
                if not valid_ray.all():
                    warnings.warn("Some object rays miss the box")
                    # filter out invalid rays to match the dimensions of the other tensors
                    indices_b, indices_h, indices_w, dirs_C_sample, depth_sample = (
                        indices_b[valid_ray],
                        indices_h[valid_ray],
                        indices_w[valid_ray],
                        dirs_C_sample[valid_ray],
                        depth_sample[valid_ray],
                    )

            # all rays should be valid
            obj_ray_sample = {
                "pc": pc,
                "z_vals": z_vals,
                "indices_b": indices_b,
                "indices_h": indices_h,
                "indices_w": indices_w,
                "dirs_C_sample": dirs_C_sample,
                "depth_sample": depth_sample,
                "T_WC_sample": T_WC_sample,
                "norm_sample": norm_sample,
            }

        # free-space sampling removed - not needed for tactile sensors
        free_space_sample = None

        # use only object ray samples for tactile sensors (no free-space sampling)
        samples = obj_ray_sample if obj_ray_sample is not None else {}

        free_space_mask = torch.isnan(depth_batch)
        binary_masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        if obj_ray_sample is not None:
            binary_masks[obj_ray_sample["indices_b"], obj_ray_sample["indices_h"], obj_ray_sample["indices_w"]] = 1

        sample_pts = {
            "depth_batch": depth_batch,
            "free_space_mask": free_space_mask,
            "binary_masks": binary_masks,
            "format": self.sensor_name,
            "object_rays": object_rays * n_frames,
        }
        return {**samples, **sample_pts}

    def batch_indices(self):
        indices = np.arange(len(self.scene_dataset), dtype=int)  # use all frames
        return indices

    def project(self, pc):
        return geometry.transform.project_pointclouds(
            pc,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.W,
            self.H,
            device=self.device,
        )

    def backproject(self, depth):
        pc = geometry.backproject_pointclouds(
            depth,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            device=self.device,
        ).squeeze()
        return pc, ~np.isnan(pc).any(axis=1)


class DigitSensor(Sensor):
    tac_depth = None

    def __init__(
        self,
        cfg_sensor: DictConfig,
        dataset_path: str = None,
        calibration: dict = None,
        device: str = "cuda",
    ):
        super(DigitSensor, self).__init__(cfg_sensor, device)
        assert (dataset_path is None) != (
            calibration is None
        ), "Pass only one of dataset path or frame from ROS"

        self.gt_depth = cfg_sensor.tactile_depth.mode == "gt"
        if dataset_path is not None:
            sensor_location = cfg_sensor.name.replace("digit_", "")
            seq_dir = os.path.join(root, dataset_path, "allegro", sensor_location)

            pkl_path = os.path.join(root, dataset_path, "data.pkl")
            with open(pkl_path, "rb") as p:
                digit_info = pickle.load(p)["digit_info"]

            self.scene_dataset = dataset.TactileDataset(
                root_dir=seq_dir, gt_depth=self.gt_depth
            )

            cfg_sensor.tactile_depth.use_real_data = "real" in dataset_path
            # More conservative loss ratio for real data
            if cfg_sensor.tactile_depth.use_real_data:
                self.loss_ratio *= 0.1
        else:
            digit_info = calibration

        self.set_intrinsics(digit_info["intrinsics"])

        # Load one model for all sensor classes
        if not self.gt_depth and DigitSensor.tac_depth is None:
            DigitSensor.tac_depth = TactileDepth(
                cfg_sensor.tactile_depth.mode,
                real=cfg_sensor.tactile_depth.use_real_data,
                device=device,
            )

        # tactile transforms
        # Aligning cam_dist with standard direct usage.
        # Standard DepthTransform adds this value directly to depth.
        # If digit_info["cam_dist"] is a positive distance (e.g., 0.022m),
        # then depth will become raw_depth + 0.022.
        # The 'use_flip' logic in tactile_pipeline.py will then handle
        # the sign convention for backprojection.
        self.cam_dist = digit_info["cam_dist"]
        self.inv_depth_scale = 1.0 / digit_info["depth_scale"]
        # RGB transform removed - tactile sensors process depth only
        self.depth_transform = transforms.Compose(
            [
                image_transforms.DepthScale(self.inv_depth_scale),
                image_transforms.DepthTransform(self.cam_dist),
            ]
        )

        self.outlier_thresh = 5.0

        # gel depth for thresholding renderer
        self.gel_depth = self.get_gel_depth(cfg_sensor)

    def get_frame_data(self, idx, poses, msg_data=None):
        if msg_data is not None:
            image = msg_data["color"]
        else:
            image, depth = self.scene_dataset[idx]  # extract rgb, d, transform

        if not self.gt_depth:
            depth = DigitSensor.tac_depth.image2heightmap(
                image[:, :, ::-1], sensor_name=self.sensor_name
            )  # RGB -> BGR

            mask = DigitSensor.tac_depth.heightmap2mask(
                depth, sensor_name=self.sensor_name
            )
            depth = (
                depth.cpu().numpy().astype(np.int64)
                if torch.is_tensor(depth)
                else depth.astype(np.int64)
            )

            mask = (
                mask.cpu().numpy().astype(np.int64)
                if torch.is_tensor(mask)
                else mask.astype(np.int64)
            )

            depth = depth * mask  # apply contact mask

        # RGB transform removed - only process depth for tactile sensors
        # scale from px to m and transform to gel frame
        depth = self.depth_transform(depth)

        # gt_depth = self.depth_transform(gt_depth.cpu().numpy())
        im_np = image[None, ...]  # (1, H, W, C)
        depth_np = depth[None, ...]  # (1, H, W)

        T_np = poses[None, ...]  # (1, 4, 4)

        im = torch.from_numpy(im_np).float().to(self.device) / 255.0
        depth = torch.from_numpy(depth_np).float().to(self.device)
        T = torch.from_numpy(T_np).float().to(self.device)

        data = FrameData(
            frame_id=np.array([idx]),
            im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
            format=[self.sensor_name],
            frame_avg_losses=torch.zeros([1], device=self.device),
        )
        return data

    def outlier_rejection_depth(self, depth):
        # outlier thresholding
        abs_depth = np.abs(depth[depth != 0.0])
        if len(abs_depth) > 0:
            outlier_thresh_max = np.percentile(abs_depth, 100 - self.outlier_thresh)
            outlier_thresh_min = np.percentile(abs_depth, 0.1)
            outlier_mask = (outlier_thresh_min < np.abs(np.nan_to_num(depth))) & (
                np.abs(np.nan_to_num(depth)) < outlier_thresh_max
            )
            reject_fraction = 1 - np.sum(depth * outlier_mask) / np.sum(depth)
            # Flat surfaces can cause outlier_thresh_min and outlier_thresh_max to be almost equal
            if reject_fraction > 0.2:
                outlier_mask = np.ones_like(depth, dtype=bool)
            depth = depth * outlier_mask

        return depth

    def get_gel_depth(self, cfg: DictConfig):
        g = cfg.gel
        origin = g.origin

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        # Curved gel surface
        N = g.countW
        W, H = g.width, g.height
        M = int(N * H / W)
        R = g.R
        zrange = g.curvatureMax

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)
        h = R - np.maximum(0, R**2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
        xx = X0 - zrange * h / h.max()

        gel_depth = torch.tensor(-xx).to(
            self.device
        )  # negative for our sign convention
        return gel_depth


def get_center_of_grasp(digit_poses, offset=0.0):
    sensor_points = np.dstack(
        [digit_pose[:3, -1] for digit_pose in digit_poses.values()]
    )
    grasp_center = sensor_points.mean(axis=2)
    # add small offset in the Z axis
    grasp_center[:, 2] += offset

    sensor_points = sensor_points.squeeze().T
    return sensor_points, grasp_center


def filter_depth(
    depth,
    outlier_thresh_max_perc,
    outlier_thresh_min_perc,
    cutoff_depth,
    do_reject_frac=True,
):
    abs_depth = np.abs(depth[depth != 0.0])
    if len(abs_depth) > 0:
        outlier_thresh_max = np.percentile(abs_depth, 100 - outlier_thresh_max_perc)
        outlier_thresh_min = np.percentile(abs_depth, outlier_thresh_min_perc)
        outlier_mask = (
            (outlier_thresh_min < np.abs(np.nan_to_num(depth)))
            & (np.abs(np.nan_to_num(depth)) < outlier_thresh_max)
            & (np.abs(np.nan_to_num(depth)) < cutoff_depth)
        )

        if do_reject_frac:
            # Flat surfaces can cause outlier_thresh_min and outlier_thresh_max to be almost equal
            reject_fraction = 1 - np.sum(depth * outlier_mask) / np.sum(depth)
            if reject_fraction > 0.1:
                outlier_mask = np.abs(np.nan_to_num(depth)) < cutoff_depth
                # print(
                #     f"% reject_fraction: {reject_fraction},"
                #     f"outlier_min: {outlier_thresh_min:.2f},"
                #     f"outlier_max: {outlier_thresh_max:.2f}"
                # )

        depth = depth * outlier_mask

    depth[depth == 0.0] = torch.nan
    return depth