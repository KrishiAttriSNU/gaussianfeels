#!/usr/bin/env python3
"""
SAM-based segmentation utility
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class SAMConfig:
    weights_path: str
    model_type: str = "vit_b"  # one of: vit_b, vit_l, vit_h
    device: str = "cpu"
    optimal_mask_size: float = 15000.0
    is_real: bool = True  # Whether this is real data (no blur for real data)


class SAMUnavailableError(RuntimeError):
    pass


class SAMSegmenter:
    """
    SAM-based segmentation
    """

    def __init__(self, config: SAMConfig):
        self.config = config
        self._sam = None
        self._predictor = None
        self._available = False
        self.logits = None

        # Try import lazily
        try:
            from segment_anything import sam_model_registry, SamPredictor

            weights_path = Path(self.config.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"SAM weights not found: {weights_path}")

            self._sam = sam_model_registry[self.config.model_type](
                checkpoint=str(weights_path)
            )
            self._sam.to(device=self.config.device)
            self._predictor = SamPredictor(self._sam)
            self._available = True
            logger.info(f"SAM loaded ({self.config.model_type}) from {weights_path}")
        except Exception as e:
            logger.warning(f"SAM unavailable: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get_optimal_mask(self, mask, logits):
        """
        Compute area of each mask and choose the one closest to the optimal area
        Disregard masks that are too small or too large
        """
        min_mask_size = self.config.optimal_mask_size // 5
        max_mask_size = self.config.optimal_mask_size * 1.5
        valid_masks_min = [i for i, m in enumerate(mask) if np.sum(m) > min_mask_size]
        valid_masks_max = [i for i, m in enumerate(mask) if np.sum(m) < max_mask_size]
        valid_masks = list(set(valid_masks_min) & set(valid_masks_max))
        if len(valid_masks) > 0:
            mask, logits = mask[valid_masks], logits[valid_masks]
            mask = [
                mask[i]
                for i in np.argsort(
                    np.abs(np.sum(mask, axis=(1, 2)) - self.config.optimal_mask_size)
                )
            ]
            logits = [
                logits[i]
                for i in np.argsort(
                    np.abs(np.sum(mask, axis=(1, 2)) - self.config.optimal_mask_size)
                )
            ]
        mask, logits = mask[0], logits[0]
        return mask, logits

    def segment_object(self, image, mask_pixels, sensor_pixels, visible_sensors):
        """
        Segment object with prompts
        - Resize to 480x640
        - Blur if not real
        - Positive prompt from mask_pixels
        - Negative prompts from sensor_pixels, zeroed when not visible
        - Reuse logits, multimask output, area-based selection
        - Return boolean mask resized back to original resolution
        """
        if not self._available:
            raise SAMUnavailableError("SAM not available")

        image_shape = image.shape[:-1]
        target_shape = (480, 640)
        resized_image = cv2.resize(
            image,
            dsize=(target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        if not self.config.is_real:
            # Light blur to handle un-photorealistic scenes - reduced from (15,15) to (5,5)
            # Heavy blur was degrading segmentation quality for sim data
            resized_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
        
        self._predictor.set_image(resized_image)
        
        # object points: positive labels for SAM
        object_points = (mask_pixels / image_shape[::-1]) * target_shape[::-1]
        object_points = object_points.astype(np.int32)
        object_labels = np.ones(object_points.shape[0])
        
        # DIGIT points: negative labels for SAM
        all_sensor_pixels = sensor_pixels.copy()
        visible_sensor_pixels = all_sensor_pixels.copy()
        # Zero-out occluded sensors
        visible_sensor_pixels[~visible_sensors] = 0
        sensor_points_all = (visible_sensor_pixels / image_shape[::-1]) * target_shape[::-1]
        sensor_points_all = sensor_points_all.astype(np.int32)
        # Exclude placeholder (0,0) negatives from SAM input; keep for debug only
        nonzero_mask = ~((sensor_points_all[:, 0] == 0) & (sensor_points_all[:, 1] == 0))
        sensor_points = sensor_points_all[nonzero_mask]
        sensor_labels = np.zeros(sensor_points.shape[0])
        
        # without negative prompts
        # input_points = object_points
        # input_labels = object_labels
        # with negative prompts
        input_points = np.concatenate((object_points, sensor_points), axis=0)
        input_labels = np.concatenate((object_labels, sensor_labels), axis=0)
        
        # multi-mask prediction usually segments out: (object part, object, foreground) len(mask) = 3
        mask, scores, logits = self._predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=self.logits,
            multimask_output=True,
        )
        
        # Area-based filtering first
        min_mask_size = self.config.optimal_mask_size // 5
        max_mask_size = self.config.optimal_mask_size * 1.5
        areas = np.array([np.sum(m) for m in mask], dtype=float)
        valid_idx = [i for i, a in enumerate(areas) if (a >= min_mask_size and a <= max_mask_size)]
        if len(valid_idx) == 0:
            valid_idx = list(range(len(mask)))

        # Build a negative-region mask to penalize finger coverage
        neg_mask = np.zeros(target_shape, dtype=np.uint8)
        try:
            # Larger radius for front-left where fingers commonly intrude
            r_sel = 12
            for (nu, nv) in sensor_points:
                if (nu, nv) != (0, 0):
                    cv2.circle(neg_mask, (int(nu), int(nv)), r_sel, 1, -1)
        except Exception as e:
            logger.debug(f"Failed to add sensor points to negative mask: {e}")

        # Candidate scoring: prefer masks containing the positive point with minimal overlap with negatives
        pos_u, pos_v = int(object_points[0][0]), int(object_points[0][1])
        candidates = []
        for i in valid_idx:
            m_i = mask[i].astype(np.uint8)
            # Must include positive pixel
            try:
                contains_pos = bool(m_i[pos_v, pos_u])
            except Exception:
                contains_pos = True
            overlap = int((m_i & neg_mask).sum())
            area_diff = float(abs(areas[i] - self.config.optimal_mask_size))
            # Sort by (overlap, area_diff)
            candidates.append((overlap, area_diff, i, contains_pos))
        # Split into those containing pos and those not; prefer containing pos
        contain = [c for c in candidates if c[3]]
        pool = contain if len(contain) > 0 else candidates
        pool.sort(key=lambda t: (t[0], t[1]))
        best_idx = pool[0][2]

        # Select and store logits accordingly
        selected_mask = mask[best_idx]
        selected_logits = logits[best_idx]
        mask = selected_mask
        logits = selected_logits
        
        self.logits = logits[None, :]
        mask = mask.squeeze().astype(np.uint8)
        mask = cv2.resize(
            mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC
        )
        
        # scale input_points back to original image dimensions
        input_points = (input_points / target_shape[::-1]) * image_shape[::-1]
        return mask.astype(bool), input_points