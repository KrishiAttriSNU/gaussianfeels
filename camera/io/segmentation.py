#!/usr/bin/env python3
"""
Segmentation Module
"""

import os
import cv2
import numpy as np
import torch
import warnings
from typing import Tuple, Optional, Union, List
from pathlib import Path
import importlib.util
import sys


class SegmentationProcessor:
    """
    Segmentation logic
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def load_gt_segmentation(self, seg_path: Union[str, Path]) -> np.ndarray:
        """
        Load GT segmentation from a JPEG file
        """
        mask = cv2.imread(str(seg_path), 0).astype(np.int64)
        mask = np.round(mask / 127.5) * 127.5
        if np.unique(mask).size != 3:
            raise RuntimeError(f"Invalid segmentation mask format at {seg_path}")
        mask = mask == 255
        return mask.astype(bool)
    
    def apply_segmentation_to_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        depth = depth * mask
        """
        return depth * mask
    
    def apply_segmentation_to_rgb(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply segmentation mask to RGB image
        """
        if len(rgb.shape) == 3:
            # Expand mask to match RGB channels
            mask_3d = np.stack([mask, mask, mask], axis=2)
            return rgb * mask_3d
        else:
            return rgb * mask
        
    def get_object_pixels(self, mask: np.ndarray) -> Tuple[int, float]:
        object_pixels = np.sum(mask)
        total_pixels = mask.size
        object_ratio = object_pixels / total_pixels if total_pixels > 0 else 0.0
        return object_pixels, object_ratio
    
    def is_valid_segmentation(self, mask: np.ndarray, min_object_ratio: float = 0.01) -> bool:
        _, object_ratio = self.get_object_pixels(mask)
        return object_ratio >= min_object_ratio


class FeelsightRealSegmentationLoader:
    """
    Segmentation logic for FeelSight Real data
    """
    
    def __init__(self, data_path: str, sam_weights_dir: str, model_type: str = "vit_b", 
                 device: str = "cpu", camera_name: str = "front-left"):
        self.data_path = Path(data_path)
        self.camera_name = camera_name
        self.device = device
        
        # Initialize SAM
        try:
            from .sam_segmenter import SAMConfig, SAMSegmenter
        except ImportError:
            from sam_segmenter import SAMConfig, SAMSegmenter
        
        sam_weights = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        weights_path = os.path.join(sam_weights_dir, sam_weights[model_type])
        optimal_mask_sizes = {
            'front-left': 15000.0,
            'back-right': 5000.0, 
            'top-down': 4000.0
        }
        
        config = SAMConfig(
            weights_path=weights_path,
            model_type=model_type,
            device=device,
            optimal_mask_size=optimal_mask_sizes.get(camera_name, 15000.0),
            is_real=True
        )
        
        self.segmenter = SAMSegmenter(config)
        self.seg_processor = SegmentationProcessor()
        
        # Sensor parameters
        self.sam_offset = {'front-left': 0.0, 'back-right': 0.01, 'top-down': 0.0}.get(camera_name, 0.0)
        
    def load_realsense_segmentation(self, frame_idx: int) -> np.ndarray:
        """
        Reads RGB frames from realsense/<camera>/image/{idx}.jpg
        """
        rgb_path = self.data_path / "realsense" / self.camera_name / "image" / f"{frame_idx}.jpg"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        # Load RGB image (BGR) and convert to RGB
        image_bgr = cv2.imread(str(rgb_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Get kinematic prompts
        mask_pixels, sensor_pixels, visible_sensors = self._kinematics_pixel_prompt(frame_idx)
        
        # Segment using SAM
        mask, seg_pixels = self.segmenter.segment_object(image, mask_pixels, sensor_pixels, visible_sensors)

        # Post-process to suppress finger leakage and keep object component around positive prompt
        try:
            mask_u8 = (mask.astype(np.uint8) * 255)
            # Suppress regions around visible finger tips
            r_sup = 14 if (self.camera_name == 'front-left') else 10
            # Suppress ONLY around visible negatives; ignore occluded ((0,0) or invisible)
            for (u, v), vis in zip(sensor_pixels if hasattr(sensor_pixels, '__iter__') else [], 
                                   visible_sensors if hasattr(visible_sensors, '__iter__') else []):
                if vis and (u, v) != (0, 0):
                    if 0 <= int(v) < mask_u8.shape[0] and 0 <= int(u) < mask_u8.shape[1]:
                        cv2.circle(mask_u8, (int(u), int(v)), r_sup, 0, -1)
            # Small closing to heal object mask after suppression
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Keep connected component that contains the positive prompt (focus pixel)
            mask_bin = (mask_u8 > 127).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask_bin, connectivity=8)
            if num_labels > 1 and mask_pixels is not None and len(mask_pixels) > 0:
                u0, v0 = int(mask_pixels[0][0]), int(mask_pixels[0][1])
                if 0 <= v0 < labels.shape[0] and 0 <= u0 < labels.shape[1]:
                    lbl = labels[v0, u0]
                    if lbl > 0:
                        mask = (labels == lbl)
                    else:
                        raise RuntimeError("Unable to identify component containing positive prompt")
                else:
                    mask = mask_bin.astype(bool)
            else:
                mask = mask_bin.astype(bool)

            # Fill internal holes in the selected component to avoid finger-induced cavities
            try:
                comp_u8 = (mask.astype(np.uint8)) * 255
                h, w = comp_u8.shape
                ff = comp_u8.copy()
                ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(ff, ff_mask, (0, 0), 255)
                holes = cv2.bitwise_not(ff)
                comp_filled = cv2.bitwise_or(comp_u8, holes)
                mask = comp_filled > 127
            except (cv2.error, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Flood fill operation failed for segmentation: {e}") from e

            # Depth gating to remove fingers at different depth than object
            try:
                depth_npz = self.data_path / 'realsense' / self.camera_name / 'depth.npz'
                if depth_npz.exists() and mask.sum() > 0:
                    d = np.load(str(depth_npz))
                    depth_arr = d['depth']
                    depth_scale = d['depth_scale'] if 'depth_scale' in d else 1.0
                    depth_img = np.abs(depth_arr[frame_idx].astype(np.float32) * float(depth_scale))
                    # Compute median depth inside mask
                    obj_depth_vals = depth_img[mask]
                    if obj_depth_vals.size > 0:
                        z_med = float(np.median(obj_depth_vals))
                        # INCREASED DEPTH BAND: Previous 8cm band was truncating objects for shape completion
                        sigma = 0.15  # 15cm band (was 4cm)
                        zmin, zmax = z_med - 3 * sigma, z_med + 3 * sigma  # ±45cm range (was ±8cm)
                        gated = (depth_img > zmin) & (depth_img < zmax) & mask
                        # Morphological cleanup and keep CC containing positive prompt
                        gated_u8 = gated.astype(np.uint8)
                        gated_u8 = cv2.morphologyEx(gated_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                        gated_u8 = cv2.morphologyEx(gated_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                        num_labels, labels = cv2.connectedComponents(gated_u8, connectivity=8)
                        if num_labels > 1 and mask_pixels is not None and len(mask_pixels) > 0:
                            u0, v0 = int(mask_pixels[0][0]), int(mask_pixels[0][1])
                            if 0 <= v0 < labels.shape[0] and 0 <= u0 < labels.shape[1]:
                                lbl = labels[v0, u0]
                                if lbl > 0:
                                    final_mask = (labels == lbl)
                                else:
                                    areas = [(labels == i).sum() for i in range(1, num_labels)]
                                    final_mask = (labels == (1 + int(np.argmax(areas)))) if len(areas) > 0 else gated
                            else:
                                final_mask = gated
                        else:
                            final_mask = gated
                        # Strict: require sufficient mask coverage; no fallbacks
                        min_abs = 300
                        min_rel = 0.05
                        if final_mask.sum() < max(min_abs, int(min_rel * mask.sum())):
                            raise RuntimeError("Depth gating produced insufficient mask coverage")
                        mask = final_mask
            except (ValueError, IndexError, KeyError, FileNotFoundError) as e:
                raise RuntimeError(f"Depth gating failed: {e}") from e
        except (RuntimeError, ValueError, IndexError, KeyError, FileNotFoundError) as e:
            raise

        # Expose debug prompts for tests/visualization using the actual pixels we passed
        # Object prompts: focus point(s)
        try:
            self._last_object_prompts = [
                (int(p[0]), int(p[1])) for p in (mask_pixels if hasattr(mask_pixels, "__iter__") else [])
            ]
            # Negative prompts: sensor pixels, zeroed if not visible
            negs = []
            for p, vis in zip(sensor_pixels if hasattr(sensor_pixels, "__iter__") else [], visible_sensors if hasattr(visible_sensors, "__iter__") else []):
                if vis:
                    negs.append((int(p[0]), int(p[1])))
                else:
                    negs.append((0, 0))
            self._last_negative_prompts = negs
        except (ValueError, IndexError, TypeError) as e:
            raise RuntimeError(f"Debug prompt creation failed: {e}") from e
        
        return mask.astype(bool)
        
    def _kinematics_pixel_prompt(self, frame_idx: int):
        """
        Kinematics-based pixel prompting for SAM
        """
        # Load kinematic data
        data_pkl_path = self.data_path / "data.pkl"
        
        try:
            import pickle
            with open(data_pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Get finger poses from data.pkl
            allegro = data.get('allegro', {})
            finger_poses = allegro.get('finger_poses', [])
            
            if frame_idx >= len(finger_poses):
                # Attempt REAL FK if direct finger_poses unavailable
                digit_poses = None
                try:
                    base_pose = allegro.get('base_pose', None)
                    joint_states = allegro.get('joint_state', None)
                    if base_pose is not None and joint_states is not None and frame_idx < len(joint_states):
                        # Dynamically import Allegro FK implementation
                        project_root = Path(__file__).resolve().parents[2]
                        allegro_path = str(project_root / 'tactile' / 'gaussianfeels' / 'modules' / 'allegro.py')
                        spec = importlib.util.spec_from_file_location('allegro_module', allegro_path)
                        allegro_module = importlib.util.module_from_spec(spec)
                        # Use proper module import without sys.path manipulation
                        spec.loader.exec_module(allegro_module)  # type: ignore
                        Allegro = allegro_module.Allegro  # type: ignore
                        # Convert base_pose 4x4 to config
                        from scipy.spatial.transform import Rotation as R  # lazy import
                        rotation_matrix = base_pose[:3, :3]
                        translation = base_pose[:3, 3]
                        quat = R.from_matrix(rotation_matrix).as_quat()  # x,y,z,w
                        base_pose_config = {
                            'translation': {'x': float(translation[0]), 'y': float(translation[1]), 'z': float(translation[2])},
                            'rotation': {'x': float(quat[0]), 'y': float(quat[1]), 'z': float(quat[2]), 'w': float(quat[3])}
                        }
                        allegro_fk = Allegro(base_pose=base_pose_config, device='cpu')
                        js = np.array(joint_states[frame_idx]).astype(np.float32)
                        digit_poses = allegro_fk.get_fk(joint_state=js)
                except (ValueError, IndexError, RuntimeError, AttributeError) as e:
                    raise RuntimeError(f"Forward kinematics computation failed: {e}") from e
            else:
                # Build dict with standard names
                fp = finger_poses[frame_idx]
                finger_names = ['digit_thumb', 'digit_index', 'digit_middle', 'digit_ring']
                digit_poses = {}
                for i, name in enumerate(finger_names):
                    if i < len(fp):
                        digit_poses[name] = np.array(fp[i])
                
            # Get camera pose
            realsense_data = data['realsense'][self.camera_name]
            realsense_pose = realsense_data['pose']
            intrinsics = realsense_data['intrinsics']
            
            # offset the kinematic tracking point to the tip of the DIGIT  
            # Center of grasp computation
            digit_adjust = np.eye(4)
            digit_adjust[1, 3] = 8e-3
            digit_poses = {k: v @ digit_adjust for k, v in digit_poses.items()}
            
            # Get center of grasp
            sensor_points, focus_point = self._get_center_of_grasp(digit_poses)
            
            # Project to pixels
            sensor_pixels = self._point_to_pixel(sensor_points, realsense_pose, intrinsics)
            focus_pixels = self._point_to_pixel(focus_point, realsense_pose, intrinsics)
            
            # Depth-based occlusion visibility when depth available
            visible_sensors = np.ones(len(sensor_pixels), dtype=bool)
            try:
                depth_path = self.data_path / 'realsense' / self.camera_name / 'depth.npz'
                if depth_path.exists():
                    d = np.load(str(depth_path))
                    depth = d['depth']
                    depth_scale = d['depth_scale'] if 'depth_scale' in d else 1.0
                    img = np.abs(depth[frame_idx].astype(np.float32) * float(depth_scale))
                    far_thresh = 0.04
                    vis_list = []
                    # Maintain identical ordering between sensor_pixels and digit_poses
                    keys_in_order = list(digit_poses.keys())
                    for i, key in enumerate(keys_in_order):
                        pose = digit_poses[key]
                        tip_w = np.array([pose[0, 3], pose[1, 3], pose[2, 3], 1.0])
                        Pc = np.linalg.inv(realsense_pose) @ tip_w
                        zc = float(abs(Pc[2]))
                        u, v = int(sensor_pixels[i][0]), int(sensor_pixels[i][1])
                        if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                            zimg = float(img[v, u])
                            is_occluded = (zimg > 0) and (zc > zimg + far_thresh)
                            vis_list.append(not is_occluded)
                        else:
                            vis_list.append(False)
                    if len(vis_list) == len(sensor_pixels):
                        visible_sensors = np.array(vis_list, dtype=bool)
            except (ValueError, IndexError, KeyError) as e:
                raise RuntimeError(f"Depth-based visibility computation failed: {e}") from e
            
            return focus_pixels, sensor_pixels, visible_sensors
            
        except (ValueError, IndexError, KeyError, FileNotFoundError) as e:
            raise RuntimeError(f"Kinematic prompting failed completely: {e}") from e
    
    def _get_center_of_grasp(self, digit_poses):
        """Compute center of grasp"""
        sensor_points = np.dstack(
            [digit_pose[:3, -1] for digit_pose in digit_poses.values()]
        )
        grasp_center = sensor_points.mean(axis=2)
        # add small offset in the Z axis
        grasp_center[:, 2] += self.sam_offset
        
        sensor_points = sensor_points.squeeze().T
        return sensor_points, grasp_center
    
    def _point_to_pixel(self, point, T_WC, intrinsics):
        """Project a 3D point to pixel coordinates"""
        if point.ndim == 1:
            point = point[None, :]
        focus_point = self._transform_points_np(point, np.linalg.inv(T_WC))
        mask_pixels = self._project(focus_point[None, :, :], intrinsics)
        mask_pixels = mask_pixels.reshape(-1, 2)
        return mask_pixels
        
    def _transform_points_np(self, points, transform):
        """Transform points using 4x4 transformation matrix"""
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ points_h.T).T
        return transformed[:, :3]
        
    def _project(self, points_cam, intrinsics):
        """Projection utility"""
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        points_cam = points_cam.squeeze()
        if points_cam.ndim == 1:
            points_cam = points_cam[None, :]
            
        # Projection (negative X)
        u = -(fx * points_cam[:, 0] / points_cam[:, 2]) + cx
        v = (fy * points_cam[:, 1] / points_cam[:, 2]) + cy
        
        return np.column_stack([u, v]).astype(np.int32)
    
    def _get_visible_sensors(self, digit_poses, realsense_pose):
        """Compute sensor visibility using depth-based occlusion testing"""
        if len(digit_poses) == 0:
            return np.array([], dtype=bool)
        
        visible_sensors = []
        
        # Convert poses to numpy if needed
        if isinstance(realsense_pose, torch.Tensor):
            realsense_pose = realsense_pose.cpu().numpy()
        
        for i, digit_pose in enumerate(digit_poses):
            if isinstance(digit_pose, torch.Tensor):
                digit_pose = digit_pose.cpu().numpy()
            
            # Simple visibility check: sensor is visible if within camera frustum
            # and not occluded by other objects
            digit_position = digit_pose[:3, 3]  # Extract position from pose matrix
            camera_position = realsense_pose[:3, 3]
            
            # Check if sensor is in front of camera (simple z-test)
            camera_to_sensor = digit_position - camera_position
            camera_forward = realsense_pose[:3, 2]  # Z-axis is forward in camera frame
            
            # Dot product > 0 means sensor is in front of camera
            is_in_front = np.dot(camera_to_sensor, camera_forward) > 0
            
            # Distance-based visibility (closer sensors are more likely to be visible)
            distance = np.linalg.norm(camera_to_sensor)
            is_close_enough = distance < 2.0  # 2 meter visibility range
            
            # Simple occlusion test: assume sensors are visible if in front and close
            is_visible = is_in_front and is_close_enough
            visible_sensors.append(is_visible)
        
        return np.array(visible_sensors, dtype=bool)

    def get_frame_segmentation_data(self, frame_idx: int) -> dict:
        """
        Get complete segmentation data for a frame
        """
        return {
            'realsense_mask': self.load_realsense_segmentation(frame_idx),
            'tactile_masks': {},  # not used for feelsight_real
            'frame_idx': frame_idx
        }


class FeelsightSegmentationLoader:
    """
    Segmentation loader for feelsight dataset (ground truth segmentation masks)
    """
    
    def __init__(self, data_root: Union[str, Path], camera_name: str = "front-left", **kwargs):
        self.data_root = Path(data_root)
        self.camera_name = camera_name
        self.realsense_seg_dir = self.data_root / "realsense" / camera_name / "seg"
        self.allegro_mask_dirs = {
            'index': self.data_root / "allegro" / "index" / "mask",
            'middle': self.data_root / "allegro" / "middle" / "mask", 
            'ring': self.data_root / "allegro" / "ring" / "mask",
            'thumb': self.data_root / "allegro" / "thumb" / "mask"
        }
        self.seg_processor = SegmentationProcessor()
    
    def load_realsense_segmentation(self, frame_idx: int) -> np.ndarray:
        """Load RealSense camera segmentation mask"""
        seg_path = self.realsense_seg_dir / f"{frame_idx}.jpg"
        return self.seg_processor.load_gt_segmentation(seg_path)
    
    def load_tactile_masks(self, frame_idx: int, enabled_fingers: List[str] = None) -> dict:
        """Load tactile contact masks for enabled fingers only"""
        tactile_masks = {}
        
        # If no enabled_fingers specified, process all fingers (backward compatibility)
        fingers_to_process = enabled_fingers if enabled_fingers is not None else list(self.allegro_mask_dirs.keys())
        
        for finger in fingers_to_process:
            if finger not in self.allegro_mask_dirs:
                raise ValueError(f"Invalid finger name: {finger}. Valid options: {list(self.allegro_mask_dirs.keys())}")
                
            mask_dir = self.allegro_mask_dirs[finger]
            mask_path = mask_dir / f"{frame_idx}.jpg"
            if mask_path.exists():
                tactile_masks[finger] = self.seg_processor.create_contact_mask(mask_path)
            else:
                raise FileNotFoundError(f"Required tactile mask not found: {mask_path}")
                
        return tactile_masks
    
    def get_frame_segmentation_data(self, frame_idx: int, enabled_fingers: List[str] = None) -> dict:
        """Get complete segmentation data for a frame"""
        # Default to empty list if no enabled fingers specified (camera-only mode)
        enabled_fingers = enabled_fingers if enabled_fingers is not None else []
        
        return {
            'realsense_mask': self.load_realsense_segmentation(frame_idx),
            'tactile_masks': self.load_tactile_masks(frame_idx, enabled_fingers),
            'frame_idx': frame_idx
        }


def create_contact_mask(self, tactile_mask_path: Union[str, Path], threshold: float = 127.5) -> np.ndarray:
    """Create contact mask from tactile segmentation data"""
    try:
        mask = cv2.imread(str(tactile_mask_path), 0).astype(np.int64)
        mask = mask > threshold
        
        # Filter tiny masks
        contact_ratio = mask.sum() / mask.size
        if contact_ratio < 0.01:
            raise RuntimeError(f"Tactile mask coverage too low ({contact_ratio:.4f}): insufficient contact detected")
            
        return mask
        
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Failed to load tactile mask from {tactile_mask_path}: {e}") from e


# Add contact mask method to SegmentationProcessor
SegmentationProcessor.create_contact_mask = create_contact_mask


def create_segmentation_processor(device: str = "cpu") -> SegmentationProcessor:
    return SegmentationProcessor(device=device)


def create_feelsight_real_segmenter(
    data_root: Union[str, Path],
    camera_name: str = 'front-left',
    sam_weights_dir: Optional[Union[str, Path]] = None,
    model_type: str = 'vit_l',
    device: str = 'cuda',
) -> FeelsightRealSegmentationLoader:
    """
    Convenience factory for the real segmentation loader.
    """
    return FeelsightRealSegmentationLoader(
        str(data_root), sam_weights_dir=str(sam_weights_dir) if sam_weights_dir else None,
        model_type=model_type, device=device, camera_name=camera_name
    )
