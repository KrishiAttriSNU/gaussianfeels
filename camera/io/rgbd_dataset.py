# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RGBDDataset(torch.utils.data.Dataset):
    """RGBD data loader for general datasets"""

    def __init__(
        self,
        root_dir: str,
        gt_seg: bool,
        col_ext: str = ".jpg",
        camera_name: str | None = None,
        enforce_single_camera: bool = True,
    ):
        # pre-load depth data
        # Resolve camera name using shared utility
        from shared.utils.dataset_utils import resolve_camera_name
        camera_name = resolve_camera_name(
            root_dir=root_dir,
            camera_name=camera_name,
            enforce_single_camera=enforce_single_camera,
            preferred_camera="front-left"
        )

        # Store the selected camera name
        self.camera_name = camera_name
        self.enforce_single_camera = enforce_single_camera
        
        depth_file = os.path.join(root_dir, "realsense", camera_name, "depth.npz")
        depth_loaded = np.load(depth_file, fix_imports=True, encoding="latin1")
        self.depth_data = depth_loaded["depth"]
        self.depth_scale = depth_loaded["depth_scale"]
        self.depth_data = self.depth_data.astype(np.float32)
        self.depth_data = self.depth_data * self.depth_scale

        self.rgb_dir = os.path.join(root_dir, "realsense", camera_name, "image")
        self.seg_dir = os.path.join(root_dir, "realsense", camera_name, "seg")
        self.col_ext = col_ext
        self.gt_seg = gt_seg

    def __len__(self):
        return len(os.listdir(self.rgb_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_file = os.path.join(self.rgb_dir, f"{idx}" + self.col_ext)
        image = cv2.imread(rgb_file)
        depth = self.depth_data[idx]

        if self.gt_seg:
            mask = self.get_gt_seg(idx)
            depth = depth * mask  # mask depth with gt segmentation

        return image, depth

    def get_avg_seg_area(self):
        """
        Returns the average segmentation area of the dataset
        """
        seg_area = 0.0
        for i in range(len(self)):
            mask = self.get_gt_seg(i)
            seg_area += mask.sum() / mask.size
        seg_area /= len(self)
        return seg_area

    def get_gt_seg(self, idx: int):
        """
        Returns a binary mask of the segmentation ground truth
        """
        seg_file = os.path.join(self.seg_dir, f"{idx}" + self.col_ext)
        mask = cv2.imread(seg_file, 0).astype(np.int64)
        # round every pixel to either 0, 255/2, 255
        mask = np.round(mask / 127.5) * 127.5
        # check if there exists three classes, if not fail fast
        if np.unique(mask).size != 3:
            raise RuntimeError(f"Invalid segmentation mask at index {idx}: expected 3 classes, found {np.unique(mask).size}. Segmentation data corrupted or incomplete.")
        else:
            mask = mask == 255
        return mask