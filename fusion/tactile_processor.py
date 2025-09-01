#!/usr/bin/env python3.11
"""
Tactile processor for camera-tactile fusion with TouchVIT depth prediction
"""

import logging
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time
import matplotlib.pyplot as plt
from collections import deque

from .config import TactileFusionConfig

logger = logging.getLogger(__name__)

# Avoid sys.path hacks by using package-relative imports where possible

import numba

from tactile.gaussianfeels.datasets.redwood_depth_noise_model import _simulate
from .tactile_depth_adapter import TactileDepthAdapter

from omegaconf import DictConfig


class TactileProcessor:
    """Tactile processor for camera-tactile fusion with TouchVIT depth prediction"""
    
    def __init__(self, config: TactileFusionConfig, digit_info: Dict):
        self.config = config
        self.digit_info = digit_info
        self.data_dir = config.trial_path  # Add missing data_dir attribute
        
        # Extract DIGIT parameters from data.pkl
        self.depth_scale = digit_info['depth_scale']
        self.cam_dist = digit_info['cam_dist'] 
        self.intrinsics = digit_info['intrinsics']
        
        logger.info(f"🔧 DIGIT Parameters from data.pkl:")
        logger.info(f"   📏 Depth scale: {self.depth_scale}")
        logger.info(f"   📏 Camera distance: {self.cam_dist}m")
        
        # Detect dataset type BEFORE initializing predictor (needed for weight selection)
        trial_path_str = str(config.trial_path).lower()
        self.is_feelsight_real = ('feelsight_real' in trial_path_str)
        self.is_feelsight_occlusion = ('feelsight_occlusion' in trial_path_str)
        
        # Initialize tactile depth wrapper (NeuraFeels-style interface)
        # Set NeuraFeels thresholds based on dataset type
        if self.is_feelsight_real:
            r, c = 1.2, 10.0   # real settings from vit.yaml
            blend = 0
        else:
            r, c = 0.0, 5.0    # sim settings from vit.yaml (occlusion treated as sim)
            blend = 0          # sim uses no temporal blending to avoid ghosting
        self.contact_ratio = r
        self.contact_clip = c
        self.tactile_depth = TactileDepthAdapter(
            real=self.is_feelsight_real,  # occlusion counts as sim (real=False)
            device=self.config.device,
            blend_sz=blend,
            contact_border=0,
            contact_ratio=r,
            contact_clip=c,
            contact_quantile=0.9,
        )
        # Mirror key adapter params for logging
        self.blend_sz = self.tactile_depth.blend_sz
        
        # Background templates (stored in adapter)
        self.bg_selected_paths = {}
        self.bg_selected_images = {}
        
        # Use multiple low-variance frames to build a robust background template
        self.bg_blend_k = 5
        
        # Contact detection parameters (NeuraFeels-style)
        self.contact_border = 0         # Border crop (b)
        
        # Additional parameters for fuller contact visualization
        self.use_better_background = True  # Use lowest-variance frame instead of frame 0
        
        # NeuraFeels-style detection: q=0.9, r=1.2, clip≈0
        self.contact_quantile = 0.9
        self.contact_ratio = 1.2
        self.contact_clip = 0.0
        logger.info("NeuraFeels-style contact detection active")
        
        # No 'full contact' visualization mode in NeuraFeels parity
        self.show_full_contact = False
        
        # No per-finger multipliers in NeuraFeels
        self.finger_contact_multipliers = None
        
        # Different finger configuration for sim vs real data
        if self.is_feelsight_real:
            # Real data: disable all fingers (uses vision-only approach)
            self.enabled_fingers = {
                'thumb': False,   # Real data - disabled
                'index': False,   # Real data - disabled
                'middle': False,  # Real data - disabled
                'ring': False     # Real data - disabled
            }
        else:
            # Sim data: enable 2 working fingers (middle + ring)
            self.enabled_fingers = {
                'thumb': False,   # Sim - disabled (problematic)
                'index': False,   # Sim - disabled (not working properly)  
                'middle': True,   # Sim - enabled (good performer)
                'ring': True      # Sim - enabled (best performer)
            }
        logger.info("🖐️ Enabled fingers: " + ", ".join([f for f, enabled in self.enabled_fingers.items() if enabled]))
        
        # Redwood: not used as a tactile post-filter in NeuraFeels runtime
        self.use_redwood_filtering = False
        logger.info("🚫 Redwood-based tactile filtering disabled (NeuraFeels parity)")
        
        # No additional rule-based coherence checks in NeuraFeels tactile masking
        self.min_contact_pixels = 0
        self.noise_threshold_ratio = 0.0
        
        # Extract timestamps
        self.timestamps = []  # Will be populated during processing
        
        logger.info(f"🚀 Tactile Processor")
        logger.info(f"   Device: {config.device}")
        logger.info(f"   Tactile heightmap blending window: {self.tactile_depth.blend_sz}")
        logger.info(f"   NeuraFeels thresholds: q=0.9, r={self.contact_ratio}, clip={self.contact_clip}, border=0")
    
    # Removed legacy predictor/blender methods (adapter handles both)
    
    def create_background_template(self, finger: str, first_frame_image: np.ndarray):
        """Create background templates for contact detection (adapter-based)"""
        logger.info(f"🔄 Creating background template for {finger}...")
        try:
            with torch.no_grad():
                bg_heightmap = self.tactile_depth.image2heightmap(first_frame_image)
                self.tactile_depth.set_background_from_heightmap(finger, bg_heightmap)
                logger.info(f"   ✅ Background template created for {finger} from provided frame image")
        except Exception as e:
            raise RuntimeError(f"Background template creation failed for {finger}: {e}") from e
    
    def enable_all_fingers(self, enable: bool = True):
        """Enable or disable all fingers"""
        for finger in self.enabled_fingers:
            self.enabled_fingers[finger] = enable
        # Only log when actually disabling (not during initialization)
        if not enable and hasattr(self, 'enabled_fingers') and any(self.enabled_fingers.values()):
            logger.info(f"❌ Disabled all fingers")
        # Don't log when enabling all - it's misleading since we disable specific ones after
    
    def enable_finger(self, finger: str, enable: bool = True):
        """Enable or disable a specific finger"""
        if finger in self.enabled_fingers:
            self.enabled_fingers[finger] = enable
            status = '✅ Enabled' if enable else '❌ Disabled'
            logger.info(f"{status} {finger} finger")
        else:
            logger.warning(f"⚠️ Unknown finger: {finger}")
    
    def get_enabled_fingers(self):
        """Get list of currently enabled fingers"""
        return [finger for finger, enabled in self.enabled_fingers.items() if enabled]
    
    def create_tactile_background_template_with_variance_selection(self, finger: str):
        """Create background template using variance-based frame selection"""
        
        finger_path = Path(self.config.trial_path) / "allegro" / finger / "image"
        if not finger_path.exists():
            logger.warning(f"   ⚠️ Finger path does not exist: {finger_path}")
            return
            
        print(f"🔍 Searching for best background frame in {finger_path}")
        
        # Find best background frame (lowest variance approach)
        if self.use_better_background and not self.is_feelsight_real:
            print(f"   🔧 Using variance-based background selection (top-{self.bg_blend_k})...")
            bg_paths = self._find_best_background_frames(finger_path, top_k=self.bg_blend_k)
        else:
            # STRICT: For feelsight_real, frame 0 must exist - no fallback
            bg_path = finger_path / "0.jpg"
            if not bg_path.exists():
                raise FileNotFoundError(f"Required background frame 0.jpg not found at {bg_path}")
            print(f"   📁 Using frame 0 as background (feelsight_real)")
        
        if self.use_better_background and not self.is_feelsight_real:
            # Blend top-k backgrounds to a robust template
            if not bg_paths:
                raise FileNotFoundError(f"No valid background frames found in {finger_path}")
            imgs_rgb = []
            for p in bg_paths:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                imgs_rgb.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not imgs_rgb:
                raise RuntimeError(f"Failed to read any background images for {finger}")
            try:
                with torch.no_grad():
                    hms = []
                    for rgb in imgs_rgb:
                        hm = self.tactile_depth.image2heightmap(rgb)
                        hms.append(hm.to(dtype=torch.float, device=self.config.device))
                    # Robust aggregation: median across top-k heightmaps
                    bg_heightmap = torch.median(torch.stack(hms, dim=0), dim=0).values
                    # store into adapter's bg_template map
                    self.tactile_depth.set_background_from_heightmap(finger, bg_heightmap)
                    self.bg_selected_paths[finger] = ",".join(str(p) for p in bg_paths)
                    self.bg_selected_images[finger] = imgs_rgb[0]
                    logger.info(f"   ✅ Background template created for {finger} from top-{len(bg_paths)} low-variance frames")
            except (RuntimeError, ValueError, TypeError) as e:
                raise RuntimeError(f"Background template creation failed for {finger}: {e}") from e
        else:
            if bg_path and bg_path.exists():
                input_img = cv2.imread(str(bg_path))
                if input_img is not None:
                    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    
                    # Generate background heightmap
                    try:
                        with torch.no_grad():
                            bg_heightmap = self.tactile_depth.image2heightmap(input_img_rgb)
                            self.tactile_depth.set_background_from_heightmap(finger, bg_heightmap)
                            self.bg_selected_paths[finger] = str(bg_path)
                            self.bg_selected_images[finger] = input_img_rgb
                            logger.info(f"   ✅ Background template created for {finger} from {bg_path.name}")
                    except (RuntimeError, ValueError, TypeError) as e:
                        raise RuntimeError(f"Background template creation failed for {finger}: {e}") from e
    
    # Removed legacy single-frame background selector (using top-k median instead)

    def _find_best_background_frames(self, finger_path: Path, top_k: int = 5) -> List[Path]:
        """Return top-k lowest variance frames for robust background construction."""
        all_images = sorted(list(finger_path.glob("*.jpg")))
        if not all_images:
            raise FileNotFoundError(f"No images found in {finger_path}")
        sample_images = all_images[::10]
        variances = []
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                hm = self.tactile_depth.image2heightmap(img_rgb)
                if hm is None:
                    continue
                var = float(hm.float().var().item())
                variances.append((var, img_path))
        if not variances:
            raise FileNotFoundError(f"No valid background frames found in {finger_path}")
        variances.sort(key=lambda x: x[0])
        selected = [p for _, p in variances[:max(1, top_k)]]
        print(f"   📊 Background selection: {len(variances)} frames analyzed")
        print(f"   🎯 Selected top-{len(selected)}: {[p.name for p in selected]}")
        if variances:
            min_v, max_v = variances[0][0], variances[-1][0]
            print(f"   📈 Range: {min_v:.4f} to {max_v:.4f}")
        return selected
    
    def _generate_contact_mask(self, heightmap: torch.Tensor, finger: str) -> torch.Tensor:
        """NeuraFeels-style contact mask: bg subtraction + adaptive thresholding (no per-finger, no Redwood)."""
        
        if finger not in self.tactile_depth.bg_template:
            raise RuntimeError(f"No background template available for {finger}. Create background template first.")
        
        # Background subtraction algorithm
        bg_template = self.tactile_depth.bg_template[finger]
        
        # Ensure same shape
        if bg_template.shape != heightmap.shape:
            bg_template = torch.nn.functional.interpolate(
                bg_template[None, None, :, :], heightmap.shape[-2:], mode="bilinear"
            ).squeeze()
            
        # Ensure both tensors are float for quantile operations
        heightmap = heightmap.float()
        bg_template = bg_template.float()
        
        # Apply border cropping if specified
        if self.contact_border > 0:
            b = self.contact_border
            heightmap_crop = heightmap[b:-b, b:-b]
            init_height = bg_template[b:-b, b:-b]
        else:
            heightmap_crop = heightmap
            init_height = bg_template
        
        # Differential heightmap and quantile thresholding
        diff_heights = heightmap_crop - init_height
        # NeuraFeels: always clamp values below clip to 0. If clip==0, negatives are zeroed.
        diff_heights[diff_heights < self.contact_clip] = 0
        if diff_heights.numel() == 0:
            raise RuntimeError(f"Invalid heightmap differences for {finger}: empty data")
        base_threshold = torch.quantile(diff_heights, self.contact_quantile) * self.contact_ratio
        contact_mask = diff_heights > base_threshold
        
        # Restore full size if border was applied
        if self.contact_border > 0:
            padded_mask = torch.full_like(bg_template, False, dtype=torch.bool)
            padded_mask[b:-b, b:-b] = contact_mask
            final_mask = padded_mask
        else:
            final_mask = contact_mask
        
        # NeuraFeels tactile mask has no further Redwood filtering
        return final_mask
    
    def _filter_contact_noise_redwood(self, contact_mask: torch.Tensor, finger: str) -> torch.Tensor:
        """Filter contact noise using Redwood depth noise model characteristics"""
        
        if contact_mask.numel() == 0:
            return contact_mask
            
        if not self.use_redwood_filtering:
            # Redwood filtering disabled (real data or occlusion); return mask unchanged
            return contact_mask
        
        # Convert to numpy for analysis
        if torch.is_tensor(contact_mask):
            mask_np = contact_mask.cpu().numpy()
        else:
            mask_np = contact_mask
            
        contact_pixels = np.sum(mask_np)
        total_pixels = mask_np.size
        
        # Redwood noise characteristics (STRICT – same as Feelsight sim):
        # 1. Tends to create sparse, scattered patterns
        # 2. Real contact should have more coherent regions
        # 3. Depth discontinuities create characteristic noise patterns

        # Too few pixels for meaningful contact
        if contact_pixels < 30:
            raise RuntimeError(f"Insufficient contact pixels ({contact_pixels}) for meaningful analysis. Redwood filtering failed.")
            
        # Calculate contact density - Redwood noise typically creates very sparse patterns
        contact_ratio = contact_pixels / total_pixels
        if contact_ratio < 0.008:  # Less than 0.8% - likely Redwood-type noise
            raise RuntimeError(f"Contact ratio too sparse ({contact_ratio:.4f}): likely Redwood noise, not real contact.")
        
        # Analyze spatial coherence using connected components
        mask_uint8 = mask_np.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:  # 0 is background
            component_sizes = stats[1:, cv2.CC_STAT_AREA]
            
            # Redwood noise creates many small scattered components
            # Real contact tends to have fewer, larger coherent areas
            small_components = np.sum(component_sizes < 25)   # Very small components
            large_components = np.sum(component_sizes >= 100) # Substantial components
            
            # If we have too many tiny components and no large ones, it's likely noise
            if small_components > 8 and large_components == 0:
                raise RuntimeError(f"Contact pattern too fragmented: {small_components} small components, no large coherent areas. Redwood noise detected.")
            
            # If largest component is too small relative to total contact, it's scattered noise
            if len(component_sizes) > 0:
                largest_component = np.max(component_sizes)
                if largest_component < contact_pixels * 0.4 and small_components > 5:
                    raise RuntimeError(f"Contact pattern scattered: largest component {largest_component} < 40% of total, {small_components} small components. Redwood noise.")
                
                # Keep only components that are reasonably sized (filter out noise specks)
                valid_labels = []
                for i, size in enumerate(component_sizes):
                    if size >= 25:  # Minimum component size to avoid noise specks
                        valid_labels.append(i + 1)  # +1 because label 0 is background
                
                if len(valid_labels) == 0:
                    raise RuntimeError("No valid contact components found after Redwood filtering. All components too small.")
                elif len(valid_labels) < len(component_sizes):
                    # Some filtering happened - create cleaned mask
                    filtered_mask = np.full_like(mask_np, False, dtype=bool)
                    for label in valid_labels:
                        filtered_mask[labels == label] = True
                    return torch.from_numpy(filtered_mask).to(contact_mask.device)
        
        # If we got here, the contact pattern looks legitimate
        return contact_mask
        
    
    def process_tactile_image(self, tactile_image: np.ndarray, finger: str, frame_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process tactile image using TouchVIT depth prediction and contact detection"""
        
        # Check if finger is enabled
        # STRICT: All finger enabled states must be explicitly configured
        if finger not in self.enabled_fingers:
            raise KeyError(f"Finger '{finger}' missing required enabled state configuration")
        if not self.enabled_fingers[finger]:
            raise RuntimeError(f"Finger {finger} is disabled. Enable finger before processing tactile images.")
        
        # Image is already RGB from fusion_test.py
        input_img_rgb = tactile_image
        
        
        # Get prediction via adapter
        try:
            with torch.no_grad():
                heightmap = self.tactile_depth.process(input_img_rgb, finger)
                contact_mask_tensor = self.tactile_depth.heightmap2mask(heightmap, finger)
                depth_tensor = heightmap
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Tactile prediction failed for {finger}: {e}") from e
        
        # Ensure tensors are on the correct device
        depth_tensor = depth_tensor.to(self.config.device)
        contact_mask_tensor = contact_mask_tensor.to(self.config.device)
        
        # Apply DIGIT transforms and processing (simplified)
        inv_depth_scale = 1.0 / self.depth_scale
        depth_scaled = depth_tensor * inv_depth_scale
        depth_corrected = depth_scaled + self.cam_dist
        
        # Apply contact mask
        z = depth_corrected
        lo = self.cam_dist + 0.0005
        hi = self.cam_dist + 0.0060
        depth_masked = torch.where(contact_mask_tensor & (z > lo) & (z < hi), z, torch.tensor(float('nan'), device=z.device))
        
        # Replace background plane with NaN
        depth_filtered = depth_masked.clone()
        depth_filtered[depth_filtered == self.cam_dist] = float('nan')

        return depth_filtered, contact_mask_tensor

    def debug_save_composite(self, finger: str, frame_idx: int, out_path: str) -> Dict:
        """Save a 2x2 composite: input, background, TouchVIT depth, contact mask.

        - Uses the same background template and Redwood thresholds.
        - If Redwood rejects the mask, saves the pre-Redwood mask and stats and re-raises.
        """
        from matplotlib import pyplot as _plt
        import math as _math

        # Ensure background exists
        if finger not in self.tactile_depth.bg_template:
            # Attempt to create background with variance selection
            self.create_tactile_background_template_with_variance_selection(finger)
            if finger not in self.tactile_depth.bg_template:
                raise RuntimeError(f"No background template for {finger}; cannot debug composite.")

        # Load tactile image for frame
        img_path = Path(self.config.trial_path) / "allegro" / finger / "image" / f"{frame_idx}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Tactile image not found: {img_path}")
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            raise RuntimeError(f"Failed to load tactile image: {img_path}")
        input_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # Predict depth via adapter (with temporal blending)
        heightmap = self.tactile_depth.process(input_rgb, finger)
        depth_map = heightmap

        # Compose mask using NeuraFeels adapter
        pre_mask = self.tactile_depth.heightmap2mask(depth_map, finger)
        redwood_ok = True
        post_mask = pre_mask
        redwood_error = None

        # Prepare visualization panels
        bg_rgb = self.bg_selected_images.get(finger, None)
        if bg_rgb is None and finger in self.bg_selected_paths:
            tmp_bgr = cv2.imread(self.bg_selected_paths[finger])
            if tmp_bgr is not None:
                bg_rgb = cv2.cvtColor(tmp_bgr, cv2.COLOR_BGR2RGB)
        if bg_rgb is None:
            # fallback to grayscale rendering of bg heightmap
            bg_map = self.tactile_depth.bg_template[finger].detach().float().cpu().numpy()
            bg_rgb = cv2.cvtColor((255 * (bg_map - bg_map.min()) / (np.ptp(bg_map) + 1e-6)).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Normalize depth to 0-255 for display
        dm = depth_map.detach().float().cpu().numpy()
        dm_viz = (255 * (dm - np.nanmin(dm)) / (np.nanmax(dm) - np.nanmin(dm) + 1e-6)).astype(np.uint8)
        dm_viz = cv2.applyColorMap(dm_viz, cv2.COLORMAP_TURBO)

        # Masks to uint8
        pre = pre_mask.detach().bool().cpu().numpy().astype(np.uint8) * 255
        post = post_mask.detach().bool().cpu().numpy().astype(np.uint8) * 255

        # Stats
        def _mask_stats(m):
            m8 = (m > 0).astype(np.uint8)
            total = m8.size
            count = int(m8.sum())
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
            comp_sizes = [] if num_labels <= 1 else stats[1:, cv2.CC_STAT_AREA].tolist()
            small = int(np.sum(np.array(comp_sizes) < 25)) if comp_sizes else 0
            large = int(np.sum(np.array(comp_sizes) >= 100)) if comp_sizes else 0
            largest = int(np.max(comp_sizes)) if comp_sizes else 0
            return {
                'pixels': count,
                'density': float(count / max(total, 1)),
                'small_components': small,
                'large_components': large,
                'largest_component': largest,
                'components': len(comp_sizes)
            }

        pre_stats = _mask_stats(pre)
        post_stats = _mask_stats(post)

        # Resolve destination path and ensure directory exists
        from pathlib import Path as _Path
        out_p = _Path(out_path)
        if out_p.exists() and out_p.is_dir():
            dest_dir = out_p
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_p = dest_dir / f"{finger}_f{frame_idx:04d}.png"
        else:
            # If provided path doesn't have a suffix, treat as directory
            if out_p.suffix == "":
                dest_dir = out_p
                dest_dir.mkdir(parents=True, exist_ok=True)
                out_p = dest_dir / f"{finger}_f{frame_idx:04d}.png"
            else:
                out_p.parent.mkdir(parents=True, exist_ok=True)

        # Compose 2x2 figure
        _plt.figure(figsize=(10, 10))
        _plt.subplot(2, 2, 1); _plt.imshow(input_rgb); _plt.axis('off'); _plt.title(f"Input {finger} f{frame_idx}")
        _plt.subplot(2, 2, 2); _plt.imshow(bg_rgb); _plt.axis('off'); _plt.title(f"Background ({Path(self.bg_selected_paths.get(finger,'?')).name})")
        _plt.subplot(2, 2, 3); _plt.imshow(dm_viz[..., ::-1]); _plt.axis('off'); _plt.title("TouchVIT Depth")
        # Show NeuraFeels-style mask
        _plt.subplot(2, 2, 4); _plt.imshow(post, cmap='gray'); _plt.axis('off');
        title = f"Mask (NeuraFeels): px={pre_stats['pixels']}, small={pre_stats['small_components']}, large={pre_stats['large_components']}"
        _plt.title(title)
        _plt.tight_layout()
        _plt.savefig(str(out_p), dpi=120)
        _plt.close()

        return {
            'pre_stats': pre_stats,
            'post_stats': post_stats,
            'redwood_ok': redwood_ok,
            'redwood_error': redwood_error,
            'bg_frame': self.bg_selected_paths.get(finger, None),
            'out_path': str(out_p)
        }
    
    def get_raw_depth_prediction(self, tactile_image: np.ndarray, finger: str) -> torch.Tensor:
        """Get raw TouchVIT depth prediction without contact filtering"""
        try:
            with torch.no_grad():
                hm = self.tactile_depth.image2heightmap(tactile_image)
                return hm.to(self.config.device).float()
        except (RuntimeError, ValueError, TypeError) as e:
            raise RuntimeError(f"Raw depth prediction failed for {finger}: {e}") from e
    
    def _create_debug_visualization(self, finger: str, frame_idx: int, 
                                   input_img: np.ndarray, depth_img: np.ndarray, contact_mask: np.ndarray):
        """Debug visualization disabled for production"""
        # Intentionally empty - debug visualizations are disabled in production
        # To enable, set config.debug_visualizations = True and implement visualization logic
        return
    
    def back_project_masked_depths_only(self, depth_map: torch.Tensor, contact_mask: torch.Tensor, finger: str = "unknown") -> torch.Tensor:
        """Back-project only the masked contact depths to 3D points"""
        
        H, W = depth_map.shape
        
        # Only process pixels where depth is non-zero (already contact-filtered)
        valid_pixels = torch.where((depth_map != 0) & (~torch.isnan(depth_map)))
        
        if len(valid_pixels[0]) == 0:
            return torch.empty(0, 3, device=depth_map.device)
        
        logger.debug(f"   Contact regions: {len(valid_pixels[0])} pixels from {H*W} total")
        
        # Extract coordinates and depths from CONTACT REGIONS ONLY
        v_coords = valid_pixels[0].float()  # y coordinates
        u_coords = valid_pixels[1].float()  # x coordinates  
        z_values = depth_map[valid_pixels]  # depth values (already contact-filtered)
        
        # Back-projection using DIGIT intrinsics
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy'] 
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']
        
        # Convert to 3D points in sensor coordinate frame
        x_sensor = -(u_coords - cx) * z_values / fx  # CRITICAL: NEGATIVE SIGN for proper coordinate system
        y_sensor = (v_coords - cy) * z_values / fy
        z_sensor = z_values
        
        contact_points = torch.stack([x_sensor, y_sensor, z_sensor], dim=1)
        
        # Apply subsampling to contact points
        if self.config.subsample_factor > 1 and len(contact_points) > 0:
            n_points = len(contact_points)
            indices = torch.arange(0, n_points, self.config.subsample_factor, device=contact_points.device)
            contact_points = contact_points[indices]
        
        # Apply surface filtering to keep only points likely on object surface
        if len(contact_points) > 0:
            contact_points = self._filter_surface_points(contact_points, finger)
        
        return contact_points
    
    def _filter_surface_points(self, contact_points: torch.Tensor, finger: str) -> torch.Tensor:
        """Filter tactile points to keep only those likely on object surface"""
        
        if len(contact_points) < 10:
            return contact_points
            
        # Method 1: Remove outliers using statistical filtering
        # Calculate distances from each point to its k-nearest neighbors
        k = min(8, len(contact_points) - 1)
        if k < 3:
            return contact_points
            
        # Compute pairwise distances
        distances = torch.cdist(contact_points, contact_points)  # [N, N]
        
        # Find k nearest neighbors for each point (excluding self)
        distances.fill_diagonal_(float('inf'))  # Exclude self-distance
        knn_distances, _ = torch.topk(distances, k, dim=1, largest=False)  # [N, k]
        
        # Calculate mean distance to k nearest neighbors
        mean_nn_distances = knn_distances.mean(dim=1)  # [N]
        
        # Filter out statistical outliers (points far from local neighborhood)
        distance_threshold = torch.quantile(mean_nn_distances, 0.8)  # Keep 80% of points
        inlier_mask = mean_nn_distances <= distance_threshold
        
        filtered_points = contact_points[inlier_mask]
        
        # Method 2: Remove points that are too far from the contact centroid
        if len(filtered_points) > 5:
            centroid = filtered_points.mean(dim=0)
            distances_to_centroid = torch.norm(filtered_points - centroid, dim=1)
            
            # Keep points within reasonable distance from centroid (object size dependent)
            max_distance = torch.quantile(distances_to_centroid, 0.9)  # Keep 90% closest to centroid
            surface_mask = distances_to_centroid <= max_distance
            filtered_points = filtered_points[surface_mask]
        
        # Surface filtering applied
        
        return filtered_points
    
    def get_T_optical_to_tip(self, finger: str) -> np.ndarray:
        """Build T_optical_to_tip transform matrix for given finger.

        Returns identity transform for standard DIGIT sensor configuration.
        """
        # Standard DIGIT sensor extrinsics (optical frame to fingertip frame)
        # Using identity transform for standard setup
        identity_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        identity_t = [0.0, 0.0, 0.0]
        
        E = {'R': identity_R, 't': identity_t}
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.asarray(E['R'], dtype=np.float32)
        T[:3, 3] = np.asarray(E['t'], dtype=np.float32)
        return T
