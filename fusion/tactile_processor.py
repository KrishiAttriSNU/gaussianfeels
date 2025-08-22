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

from .config import TactileFusionConfig

logger = logging.getLogger(__name__)

# Avoid sys.path hacks by using package-relative imports where possible

import numba

from tactile.gaussianfeels.datasets.redwood_depth_noise_model import _simulate
from tactile.gaussianfeels.contrib.tactile_transformer.touch_vit import TouchVIT

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
        
        # Detect feelsight_real data BEFORE initializing predictor (needed for weight selection)
        self.is_feelsight_real = 'feelsight_real' in str(config.trial_path).lower()
        
        # Initialize predictor
        self.predictor = self._initialize_predictor()
        
        # Background templates for contact detection
        self.bg_templates = {}
        
        # Contact detection parameters (will be overridden for feelsight_real)
        self.contact_border = 0         # Contact detection border size
        
        # Additional parameters for fuller contact visualization
        self.use_better_background = True  # Use lowest-variance frame instead of frame 0
        
        # Use same contact detection settings for both feelsight and feelsight_real
        self.contact_quantile = 0.9       # Conservative for normal data
        self.contact_ratio = 1.2          # Standard setting
        self.contact_clip = 10.0          # Standard noise floor
        logger.info("Standard contact detection for all sensor data")
        
        self.show_full_contact = True     # Show fuller contact area for debugging
        
        # Use same per-finger contact sensitivity for both feelsight and feelsight_real
        self.finger_contact_multipliers = {
            'thumb': 2.0,     # Red - conservative
            'index': 2.0,     # Green - conservative  
            'middle': 1.0,    # Blue - normal
            'ring': 1.0       # Yellow - normal
        }
        
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
        
        # Redwood noise model for filtering - only use for feelsight sim, not feelsight_real
        if self.is_feelsight_real:
            # Disable Redwood filtering for feelsight_real data
            self.use_redwood_filtering = False
            logger.info("🚫 Redwood filtering disabled for feelsight_real (only for feelsight sim)")
        else:
            # Redwood filtering required for feelsight sim data
            self.use_redwood_filtering = True
            # Load Redwood distortion model - REQUIRED
            import os
            from pathlib import Path as _P
            project_root = _P(__file__).resolve().parents[1]
            redwood_model_path = str(project_root / "data/feelsight/redwood-depth-dist-model.npy")
            if not os.path.exists(redwood_model_path):
                raise FileNotFoundError(f"REQUIRED Redwood model file not found: {redwood_model_path}")
            self.redwood_dist_model = np.load(redwood_model_path)
            self.redwood_dist_model = self.redwood_dist_model.reshape(80, 80, 5)
            logger.info("✅ Redwood distortion model loaded for feelsight sim")
        
        # Strict filtering only
        self.min_contact_pixels = 50      # Minimum pixels required for valid contact
        self.noise_threshold_ratio = 0.05 # Max 5% of pixels can be "contact" before it's considered noise
        
        # Extract timestamps
        self.timestamps = []  # Will be populated during processing
        
        logger.info(f"🚀 Tactile Processor")
        logger.info(f"   Device: {config.device}")
    
    def _initialize_predictor(self):
        """Initialize TouchVIT directly"""
            
        try:
            # Create TouchVIT config to match the saved model dimensions
            # Use correct weights based on dataset type
            weights = "dpt_real" if self.is_feelsight_real else "dpt_sim"
            logger.info(f"🎯 Using TouchVIT weights: {weights} (feelsight_real: {self.is_feelsight_real})")
            
            touchvit_config = {
                "General": {
                    "path_input_images": "/tmp/tactile_input",
                    "type": "depth", 
                    "device": self.config.device,
                    "emb_dim": 384,  # Match saved model
                    "resample_dim": 128,  # Match saved model
                    "read": "projection",  # Match saved model
                    "hooks": [2, 5, 8, 11],
                    "model_timm": "vit_small_patch16_224.dino",  # Match saved model
                    "patch_size": 16,
                    "path_model": "/home/krishi/gaussianfeels/data/tactile_transformer",  # Correct model path
                    "path_predicted_images": "/tmp/tactile_output"
                },
                "Dataset": {
                    "transforms": {
                        "resize": [224, 224]  # Match saved model input size
                    },
                    "classes": ["depth"]
                },
                "weights": weights
            }
            
            cfg = DictConfig(touchvit_config)
            touchvit = TouchVIT(cfg)
            logger.info("✅ TouchVIT initialized directly")
            return touchvit
            
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(f"TouchVIT dependencies not available: {e}") from e
        except (FileNotFoundError, IOError) as e:
            raise FileNotFoundError(f"TouchVIT model files not found: {e}") from e
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"TouchVIT initialization failed: {e}") from e
    
    def create_background_template(self, finger: str, first_frame_image: np.ndarray):
        """Create background templates for contact detection"""
        logger.info(f"🔄 Creating background template for {finger}...")
        
        finger_path = Path(self.config.trial_path) / "allegro" / finger / "image"
        if not finger_path.exists():
            logger.warning(f"  ❌ {finger} data not found")
            return
            
        # Find best background frame (lowest variance approach)
        if not self.use_better_background:
            raise ValueError("Basic background selection disabled - use_better_background must be True")
        bg_path = self._find_best_background_frame(finger_path)
        
        if bg_path and bg_path.exists():
            input_img = cv2.imread(str(bg_path))
            if input_img is not None:
                input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                
                # Prepare for prediction
                img_tensor = torch.from_numpy(input_img_rgb).permute(2, 0, 1).float()
                batch_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                
                # Generate background heightmap
                if self.predictor is not None:
                    try:
                        with torch.no_grad():
                            bg_heightmap = self.predictor.image2heightmap(input_img_rgb)
                            if bg_heightmap is not None:
                                self.bg_templates[finger] = bg_heightmap.to(dtype=torch.float, device=self.config.device)
                                logger.info(f"   ✅ Background template created for {finger} from {bg_path.name}")
                    except (RuntimeError, ValueError, TypeError) as e:
                        raise RuntimeError(f"Background template creation failed for {finger}: {e}") from e
    
    def enable_all_fingers(self, enable: bool = True):
        """Enable or disable all fingers"""
        for finger in self.enabled_fingers:
            self.enabled_fingers[finger] = enable
        logger.info(f"{'✅ Enabled' if enable else '❌ Disabled'} all fingers")
    
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
            print(f"   🔧 Using variance-based background selection...")
            bg_path = self._find_best_background_frame(finger_path)
        else:
            # STRICT: For feelsight_real, frame 0 must exist - no fallback
            bg_path = finger_path / "0.jpg"
            if not bg_path.exists():
                raise FileNotFoundError(f"Required background frame 0.jpg not found at {bg_path}")
            print(f"   📁 Using frame 0 as background (feelsight_real)")
        
        if bg_path and bg_path.exists():
            input_img = cv2.imread(str(bg_path))
            if input_img is not None:
                input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                
                # Generate background heightmap
                if self.predictor is not None:
                    try:
                        with torch.no_grad():
                            bg_heightmap = self.predictor.image2heightmap(input_img_rgb)
                            if bg_heightmap is not None:
                                self.bg_templates[finger] = bg_heightmap.to(dtype=torch.float, device=self.config.device)
                                logger.info(f"   ✅ Background template created for {finger} from {bg_path.name}")
                    except (RuntimeError, ValueError, TypeError) as e:
                        raise RuntimeError(f"Background template creation failed for {finger}: {e}") from e
    
    def _find_best_background_frame(self, finger_path: Path) -> Path:
        """Find frame with lowest variance (flattest/least contact) as background"""
        
        all_images = sorted(list(finger_path.glob("*.jpg")))
        if not all_images:
            raise FileNotFoundError(f"No images found in {finger_path} - cannot select background frame without images")
            
        # Sample every 10th frame for efficiency (following your suggestion)
        sample_images = all_images[::10]
        
        # Scanning frames for best background
        
        variances = []
        failed_count = 0
        
        for img_path in sample_images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise RuntimeError(f"Failed to load tactile image {img_path.name}. Ensure image path is valid and readable.")
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Prepare for prediction
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
                batch_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                
                if self.predictor is None:
                    raise RuntimeError("No tactile predictor available for variance calculation. Initialize predictor first.")
                    
                with torch.no_grad():
                    heightmap = self.predictor.image2heightmap(img_rgb)
                    
                    if heightmap is None:
                        raise RuntimeError(f"TouchVIT prediction failed for {img_path.name}. Ensure valid tactile image input.")
                        failed_count += 1
                        continue
                        
                    # Ensure heightmap is a tensor and convert to float
                    if not torch.is_tensor(heightmap):
                        heightmap = torch.from_numpy(heightmap)
                    
                    # Convert to float for variance calculation
                    heightmap = heightmap.float()
                    
                    # Calculate variance more robustly
                    if heightmap.numel() > 0:
                        variance = float(heightmap.var().item())
                        variances.append((variance, img_path))
                        # Variance calculated
                    else:
                        # Empty heightmap
                        failed_count += 1
                        
            except (RuntimeError, ValueError, TypeError, IOError) as e:
                raise RuntimeError(f"Background frame processing failed: {e}") from e
        
        # Background analysis complete
        
        if variances:
            # Find frame with minimum variance (flattest surface)
            min_variance, best_path = min(variances, key=lambda x: x[0])
            max_variance = max(variances, key=lambda x: x[0])[0]
            print(f"   📊 Background selection: {len(variances)} frames analyzed")
            print(f"   🎯 Selected: {best_path.name} (variance: {min_variance:.4f})")
            print(f"   📈 Range: {min_variance:.4f} to {max_variance:.4f}")
            return best_path
        else:
            raise FileNotFoundError(f"No valid background frames found in {finger_path}")
    
    def _generate_contact_mask(self, heightmap: torch.Tensor, finger: str) -> torch.Tensor:
        """Generate contact mask using background subtraction"""
        
        if finger not in self.bg_templates:
            raise RuntimeError(f"No background template available for {finger}. Create background template first.")
        
        # Background subtraction algorithm
        bg_template = self.bg_templates[finger]
        
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
        
        # Differential heightmap computation for contact detection
        diff_heights = heightmap_crop - init_height
        
        if self.show_full_contact:
            # Option A: Use ReLU (no hard clip) and relative threshold for fuller contact area
            diff_heights = torch.relu(diff_heights)  # Keep all positive differences
            if diff_heights.numel() > 0 and diff_heights.max() > 0:
                threshold = 0.2 * diff_heights.max()  # Keep everything above 20% of max
                contact_mask = diff_heights > threshold
            else:
                raise RuntimeError(f"Invalid heightmap differences for {finger}: empty or zero-valued data")
        else:
            # Contact detection with quantile thresholding
            diff_heights[diff_heights < self.contact_clip] = 0
            if diff_heights.numel() > 0 and diff_heights.max() > 0:
                # Use configurable quantile for contact detection
                base_threshold = torch.quantile(diff_heights, self.contact_quantile) * self.contact_ratio
                
                # Apply per-finger sensitivity multiplier
                # STRICT: All finger multipliers must be explicitly configured
                if finger not in self.finger_contact_multipliers:
                    raise KeyError(f"Finger '{finger}' missing required contact multiplier configuration")
                finger_multiplier = self.finger_contact_multipliers[finger]
                threshold = base_threshold * finger_multiplier
                
                contact_mask = diff_heights > threshold
            else:
                raise RuntimeError(f"Invalid heightmap differences for {finger}: empty or zero-valued data after quantile thresholding")
        
        # Restore full size if border was applied
        if self.contact_border > 0:
            padded_mask = torch.full_like(bg_template, False, dtype=torch.bool)
            padded_mask[b:-b, b:-b] = contact_mask
            final_mask = padded_mask
        else:
            final_mask = contact_mask
        
        # Apply Redwood-based noise filtering
        final_mask = self._filter_contact_noise_redwood(final_mask, finger)
        
        return final_mask
    
    def _filter_contact_noise_redwood(self, contact_mask: torch.Tensor, finger: str) -> torch.Tensor:
        """Filter contact noise using Redwood depth noise model characteristics"""
        
        if contact_mask.numel() == 0:
            return contact_mask
            
        if not self.use_redwood_filtering:
            raise RuntimeError("Redwood model required for filtering")
        
        # Convert to numpy for analysis
        if torch.is_tensor(contact_mask):
            mask_np = contact_mask.cpu().numpy()
        else:
            mask_np = contact_mask
            
        contact_pixels = np.sum(mask_np)
        total_pixels = mask_np.size
        
        # Redwood noise characteristics:
        # 1. Tends to create sparse, scattered patterns
        # 2. Real contact should have more coherent regions
        # 3. Depth discontinuities create characteristic noise patterns
        
        if contact_pixels < 30:  # Too few pixels for meaningful contact
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
            small_components = np.sum(component_sizes < 25)  # Very small components
            large_components = np.sum(component_sizes >= 100)  # Substantial components
            
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
        
        
        # Get prediction
        try:
            if self.predictor is None:
                raise RuntimeError(f"No predictor available for {finger}. Initialize tactile predictor first.")
            else:
                with torch.no_grad():
                    # Use TouchVIT image2heightmap method - pass uint8 like reference
                    depth_pred = self.predictor.image2heightmap(input_img_rgb)
                    
                    if depth_pred is not None:
                        depth_img = depth_pred
                        if torch.is_tensor(depth_img):
                            depth_tensor = depth_pred
                            depth_img = depth_pred.cpu().numpy()
                        else:
                            depth_tensor = torch.from_numpy(depth_pred)
                        
                        # Generate contact mask (ensure tensors are on same device)
                        depth_tensor = depth_tensor.to(self.config.device)
                        contact_mask_tensor = self._generate_contact_mask(depth_tensor, finger)
                        contact_mask = contact_mask_tensor.cpu().numpy() if torch.is_tensor(contact_mask_tensor) else contact_mask_tensor
                    else:
                        raise RuntimeError(f"Invalid depth prediction type for {finger}. Expected torch.Tensor or numpy.ndarray.")
            
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Tactile prediction failed for {finger}: {e}") from e
        
        # Debug visualization disabled for production
        
        # Convert back to tensors for processing
        if not torch.is_tensor(depth_img):
            depth_tensor = torch.from_numpy(depth_img).to(self.config.device)
        else:
            depth_tensor = depth_img.to(self.config.device)
        if not torch.is_tensor(contact_mask):
            contact_mask_tensor = torch.from_numpy(contact_mask).to(self.config.device)
        else:
            contact_mask_tensor = contact_mask.to(self.config.device)
        
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
    
    def get_raw_depth_prediction(self, tactile_image: np.ndarray, finger: str) -> torch.Tensor:
        """Get raw TouchVIT depth prediction without contact filtering"""
        if self.predictor is None:
            raise RuntimeError(f"No tactile predictor available for {finger}. Initialize predictor first.")
        
        try:
            with torch.no_grad():
                depth_pred = self.predictor.image2heightmap(tactile_image)
                if depth_pred is not None:
                    if torch.is_tensor(depth_pred):
                        return depth_pred.to(self.config.device)
                    else:
                        return torch.from_numpy(depth_pred).to(self.config.device)
                else:
                    raise RuntimeError(f"TouchVIT depth prediction returned None for {finger}.")
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