#!/usr/bin/env python3.11
"""
Thin adapter to mirror NeuraFeels TactileDepth interface while using our TouchVIT model.
Provides:
- image2heightmap(image)
- process(image, sensor_name)  # with temporal blending
- heightmap2mask(heightmap, sensor_name)
- set_background(sensor_name, image) / set_background_from_heightmap(sensor_name, hm)
"""

from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from tactile.gaussianfeels.contrib.tactile_transformer.touch_vit import TouchVIT


class TactileDepthAdapter:
    def __init__(self, *, real: bool, device: str = "cuda", blend_sz: int = 5,
                 contact_border: int = 0, contact_ratio: float = 1.2, contact_clip: float = 0.0,
                 contact_quantile: float = 0.9, model_dir: Optional[str] = "/home/krishi/gaussianfeels/data/tactile_transformer"):
        self.device = device
        self.real = real
        self.blend_sz = blend_sz
        self.b = contact_border
        self.r = contact_ratio
        self.clip = contact_clip
        self.q = contact_quantile

        weights = "dpt_real" if real else "dpt_sim"
        cfg = DictConfig({
            "General": {
                "path_input_images": "/tmp/tactile_input",
                "type": "depth",
                "device": device,
                "emb_dim": 384,
                "resample_dim": 128,
                "read": "projection",
                "hooks": [2, 5, 8, 11],
                "model_timm": "vit_small_patch16_224.dino",
                "patch_size": 16,
                "path_model": model_dir,
                "path_predicted_images": "/tmp/tactile_output",
            },
            "Dataset": {"transforms": {"resize": [224, 224]}, "classes": ["depth"]},
            "weights": weights,
        })
        self.model = TouchVIT(cfg)
        self.bg_template: Dict[str, torch.Tensor] = {}
        self._windows: Dict[str, deque] = {}

    def image2heightmap(self, image: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            hm = self.model.image2heightmap(image)
            if not torch.is_tensor(hm):
                hm = torch.from_numpy(hm)
            return hm.to(self.device).float()

    def _blend(self, heightmap: torch.Tensor, sensor_name: str) -> torch.Tensor:
        if not self.blend_sz or self.blend_sz <= 1:
            return heightmap
        if sensor_name not in self._windows:
            self._windows[sensor_name] = deque([])
        win = self._windows[sensor_name]
        if len(win) >= self.blend_sz:
            win.popleft()
        win.append(heightmap.detach())
        n = len(win)
        if n == 1:
            return win[-1]
        weights = torch.tensor([x / n for x in range(1, n + 1)], device=heightmap.device, dtype=heightmap.dtype)
        weights = torch.exp(weights)
        weights = weights / torch.sum(weights)
        stacked = torch.stack(list(win), dim=0)
        return torch.sum((stacked * weights[:, None, None]) / weights.sum(), dim=0)

    def process(self, image: np.ndarray, sensor_name: str) -> torch.Tensor:
        hm = self.image2heightmap(image)
        return self._blend(hm, sensor_name)

    def heightmap2mask(self, heightmap: torch.Tensor, sensor_name: str) -> torch.Tensor:
        if sensor_name not in self.bg_template:
            raise RuntimeError(f"No background template for {sensor_name}")
        heightmap = heightmap.squeeze().to(self.device).float()
        bg_template = self.bg_template[sensor_name]
        if bg_template.shape != heightmap.shape:
            bg_template = torch.nn.functional.interpolate(bg_template[None, None, :, :], heightmap.shape[-2:], mode="bilinear").squeeze()
        init_height = bg_template
        if self.b:
            heightmap = heightmap[self.b:-self.b, self.b:-self.b]
            init_height = init_height[self.b:-self.b, self.b:-self.b]
        diff_heights = heightmap - init_height
        diff_heights[diff_heights < self.clip] = 0
        threshold = torch.quantile(diff_heights, self.q) * self.r
        contact_mask = diff_heights > threshold
        if self.b:
            padded = torch.zeros_like(bg_template, dtype=torch.bool)
            padded[self.b:-self.b, self.b:-self.b] = contact_mask
            return padded
        return contact_mask

    def set_background(self, sensor_name: str, bg_rgb: np.ndarray):
        hm = self.image2heightmap(bg_rgb)
        self.bg_template[sensor_name] = hm.to(self.device).float()

    def set_background_from_heightmap(self, sensor_name: str, bg_hm: torch.Tensor):
        self.bg_template[sensor_name] = bg_hm.to(self.device).float()

