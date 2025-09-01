# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import List

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
from segment_anything import SamPredictor, sam_model_registry
from termcolor import cprint
from torch import nn
from torchvision import transforms

# GaussianFeels imports
from shared.io.frame_data import FrameData, pose_from_config, create_identity_pose
from .segmentation import SegmentationProcessor, FeelsightSegmentationLoader, create_segmentation_processor

# Get package root directory
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Sensor(nn.Module):
    def __init__(
        self,
        cfg_sensor: DictConfig,
        device: str = "cuda",
        data_root: str = None,
        camera_name: str = None,
    ):
        super(Sensor, self).__init__()
        self.device = device
        self.sensor_name = cfg_sensor.name
        self.data_root = data_root
        self.camera_name = camera_name
        cprint(f"Adding Sensor: {self.sensor_name}", color="yellow")

        self.end = False

        # Initialize segmentation processor
        self.segmentation_processor = create_segmentation_processor(device="cpu")  # CPU for image processing
        
        # Initialize appropriate segmentation loader based on data type
        if data_root:
            try:
                # Auto-detect camera if not provided
                if self.camera_name is None:
                    cams_root = os.path.join(data_root, 'realsense')
                    detected = []
                    if os.path.isdir(cams_root):
                        for sub in sorted(os.listdir(cams_root)):
                            sub_path = os.path.join(cams_root, sub)
                            if os.path.isdir(sub_path) and (
                                os.path.exists(os.path.join(sub_path, 'image')) or
                                os.path.exists(os.path.join(sub_path, 'depth.npz'))
                            ):
                                detected.append(sub)
                    if not detected:
                        raise RuntimeError(f"No cameras found under {cams_root}")
                    self.camera_name = detected[0]
                
                # Detect data type and use appropriate loader
                is_feelsight_real = 'feelsight_real' in str(data_root).lower()
                
                if is_feelsight_real:
                    # Real data: use SAM-based segmentation loader
                    from .segmentation import FeelsightRealSegmentationLoader
                    sam_weights_dir = os.path.join(root, "data", "segment-anything")
                    self.feelsight_loader = FeelsightRealSegmentationLoader(
                        data_path=data_root,
                        sam_weights_dir=sam_weights_dir,
                        model_type="vit_l",  # Use high-quality model for real data
                        device="cpu",
                        camera_name=self.camera_name
                    )
                    cprint(f"FeelsightReal SAM-based segmentation loader initialized: {data_root}", color="green")
                else:
                    # Sim data: use precomputed ground truth segmentation (no SAM)
                    self.feelsight_loader = FeelsightSegmentationLoader(data_root, camera_name=self.camera_name)
                    cprint(f"Feelsight precomputed segmentation loader initialized: {data_root} (no SAM)", color="green")
                    
            except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
                raise RuntimeError(f"Failed to initialize segmentation loader for {self.sensor_name}: {e}. Check data path and dependencies.") from e
        else:
            self.feelsight_loader = None

        # Removed kf_min_loss as it is typically tied to SDF-based reconstruction loss.

        if "realsense" in self.sensor_name:
            self.optimal_mask_size = cfg_sensor.optimal_mask_size[self.sensor_name]
            # Live Realsense support is not initialized here; dataset-backed access is used instead.

        # Legacy SAM initialization - now handled by appropriate segmentation loader
        # SAM is initialized in FeelsightRealSegmentationLoader for real data
        # Sim data uses precomputed masks via FeelsightSegmentationLoader (no SAM needed)
        self.sam_predictor = None


    def get_frame_data(self, frame_idx: int = 0) -> FrameData:
        """
        Captures and processes a single frame from the sensor,
        returning RGB, depth, masks, and camera pose (T_WC).
        This method is backend-agnostic and does not involve SDF or sampling.
        """
        rgb_image = None
        depth_image = None
        masks = None
        T_WC = None  # Transform from World to Camera

        if "realsense" in self.sensor_name:
            # Use dataset-backed frames; live acquisition is not mocked here
            if self.data_root and self.camera_name:
                rgb_path = os.path.join(self.data_root, 'realsense', self.camera_name, 'image', f'{frame_idx}.jpg')
                depth_path = os.path.join(self.data_root, 'realsense', self.camera_name, 'depth.npz')
                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    rgb_bgr = cv2.imread(rgb_path)
                    if rgb_bgr is not None:
                        rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    d = np.load(depth_path)
                    depth = d['depth']
                    depth_scale = d['depth_scale'] if 'depth_scale' in d else 1.0
                    if frame_idx < len(depth):
                        depth_image = np.abs(depth[frame_idx].astype(np.float32) * float(depth_scale))
                    # STRICT: Camera pose must be loaded from data.pkl - no default identity
                    data_pkl_path = Path(self.data_dir) / 'data.pkl'
                    if not data_pkl_path.exists():
                        raise FileNotFoundError(f"data.pkl required for camera poses not found at {data_pkl_path}")
                    
                    # Load pose data from data.pkl
                    import pickle
                    try:
                        with open(data_pkl_path, 'rb') as f:
                            pose_data = pickle.load(f)
                        
                        # Extract camera pose for current frame
                        if 'camera_poses' in pose_data and frame_id < len(pose_data['camera_poses']):
                            camera_pose = pose_data['camera_poses'][frame_id]
                        elif 'poses' in pose_data and frame_id < len(pose_data['poses']):
                            camera_pose = pose_data['poses'][frame_id]
                        else:
                            # Use identity if frame not found but data.pkl exists
                            camera_pose = np.eye(4)
                            print(f"Warning: Frame {frame_id} not found in pose data, using identity")
                    except Exception as e:
                        raise RuntimeError(f"Failed to load camera poses from data.pkl: {e}")

        # Process masks using SAM if initialized and RGB image is available
        if self.sam_predictor and rgb_image is not None:
            self.sam_predictor.set_image(rgb_image)
            # Example: Predict masks for the whole image or specific points/boxes.
            # For simplicity, we'll use a single positive point at the image center.
            # In a real application, you would define more sophisticated prompts for SAM.
            input_point = np.array([[rgb_image.shape[1] // 2, rgb_image.shape[0] // 2]])
            input_label = np.array([1])  # Positive point label
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,  # Request a single best mask
            )
            # SAM returns a boolean mask; convert to uint8 (0 or 255) for consistency
            masks = masks[0].astype(np.uint8) * 255
            cprint("SAM masks generated.", color="blue")

        if rgb_image is None or depth_image is None or T_WC is None:
            warnings.warn("Failed to acquire complete sensor data. Returning None.")
            return None

        # Create and return a FrameData object with the acquired sensor data
        frame_data = FrameData(
            rgb=rgb_image,
            depth=depth_image,
            mask=masks,  # Mask can be None if SAM is not used or fails
            T_WC=T_WC,
            frame_idx=frame_idx,
            # Add other relevant data if available, e.g., camera intrinsics, tactile data
            # intrinsics=self.camera_intrinsics,
            # tactile_data=self.tactile_sensor.get_data() if hasattr(self, 'tactile_sensor') else None,
        )
        return frame_data

    def get_segmented_frame_data(self, frame_idx: int = 0, apply_mask_to_depth: bool = True) -> FrameData:
        """
        Get frame data with proper segmentation applied
        
        Args:
            frame_idx: Frame index to load
            apply_mask_to_depth: Whether to apply segmentation mask to depth
            
        Returns:
            FrameData with segmented RGB, depth, and masks
        """
        if not self.feelsight_loader:
            warnings.warn("Feelsight loader not initialized. Cannot load segmentation data.")
            return self.get_frame_data(frame_idx)
        
        try:
            # Load segmentation data
            seg_data = self.feelsight_loader.get_frame_segmentation_data(frame_idx)
            realsense_mask = seg_data['realsense_mask']
            tactile_masks = seg_data['tactile_masks']
            
            # Load RGB and depth from feelsight dataset using detected/provided camera
            cam = self.camera_name
            rgb_path = self.data_root + f"/realsense/{cam}/image/{frame_idx}.jpg"
            depth_path = self.data_root + f"/realsense/{cam}/depth.npz"
            
            # Load RGB
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                raise ValueError(f"Could not load RGB image: {rgb_path}")
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Load depth
            depth_data = np.load(depth_path)
            depth_image = depth_data['depth'][frame_idx]
            depth_scale = depth_data['depth_scale']
            depth_image = np.abs(depth_image * depth_scale)  # Convert to positive meters
            
            # Apply segmentation mask to depth
            if apply_mask_to_depth:
                depth_image = self.segmentation_processor.apply_segmentation_to_depth(
                    depth_image, realsense_mask
                )
            
            # Create combined mask data
            mask_data = {
                'object_mask': realsense_mask,
                'tactile_masks': tactile_masks
            }
            
            # STRICT: Camera pose must be loaded from data.pkl - no default identity
            data_pkl_path = Path(self.data_dir) / 'data.pkl'
            if not data_pkl_path.exists():
                raise FileNotFoundError(f"data.pkl required for camera poses not found at {data_pkl_path}")
            
            # Load pose data from data.pkl
            import pickle
            try:
                with open(data_pkl_path, 'rb') as f:
                    pose_data = pickle.load(f)
                
                # Extract camera pose for current frame
                if 'camera_poses' in pose_data and frame_idx < len(pose_data['camera_poses']):
                    T_WC = pose_data['camera_poses'][frame_idx]
                elif 'poses' in pose_data and frame_idx < len(pose_data['poses']):
                    T_WC = pose_data['poses'][frame_idx]
                else:
                    # Use identity if frame not found but data.pkl exists
                    T_WC = np.eye(4)
                    print(f"Warning: Frame {frame_idx} not found in pose data, using identity")
            except Exception as e:
                raise RuntimeError(f"Failed to load camera poses from data.pkl: {e}")
            
            # Create FrameData with segmentation
            frame_data = FrameData(
                rgb=rgb_image,
                depth=depth_image,
                mask=mask_data,
                T_WC=T_WC,
                frame_idx=frame_idx,
            )
            
            cprint(f"Loaded segmented frame {frame_idx} with {np.sum(realsense_mask)} object pixels", 
                   color="blue")
            
            return frame_data
            
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load segmented frame {frame_idx}: {e}. Check segmentation data and frame index.") from e
    
    def get_object_reconstruction_data(self, frame_indices: List[int] = None, 
                                     min_object_ratio: float = 0.01) -> List[FrameData]:
        """
        Get frame data for object reconstruction across multiple frames
        
        Args:
            frame_indices: Specific frame indices to load (if None, auto-select valid frames)
            min_object_ratio: Minimum object ratio for valid frames
            
        Returns:
            List of FrameData objects with segmented object data
        """
        if not self.feelsight_loader:
            warnings.warn("Feelsight loader not initialized.")
            return []
        
        if frame_indices is None:
            # Auto-select valid frames
            frame_indices = self.feelsight_loader.get_valid_frames(min_object_ratio)
            cprint(f"Auto-selected {len(frame_indices)} valid frames for reconstruction", color="green")
        
        reconstruction_frames = []
        
        for frame_idx in frame_indices:
            frame_data = self.get_segmented_frame_data(frame_idx, apply_mask_to_depth=True)
            if frame_data is not None:
                reconstruction_frames.append(frame_data)
        
        cprint(f"Loaded {len(reconstruction_frames)} frames for object reconstruction", color="green")
        return reconstruction_frames

    def forward(self, *args, **kwargs):
        """
        The forward pass for the sensor, typically used to get a frame.
        Delegates to get_frame_data.
        """
        return self.get_frame_data(*args, **kwargs)

    def __del__(self):
        """Cleanup resources when the Sensor object is destroyed."""
        return