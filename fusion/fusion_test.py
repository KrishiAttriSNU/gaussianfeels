#!/usr/bin/env python3.11
"""
Camera-tactile fusion test with TouchVIT depth prediction
"""

import logging
import pickle
import numpy as np
import torch
import cv2
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import gc

from .config import TactileFusionConfig
from .tactile_processor import TactileProcessor

logger = logging.getLogger(__name__)


class TactileFusionTest:
    """Camera-tactile fusion test with TouchVIT depth prediction"""
    
    def __init__(self, config: TactileFusionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trial data
        self.trial_data = self._load_trial_data()
        
        # Initialize tactile processor
        self.tactile_processor = TactileProcessor(
            config, self.trial_data['digit_info']
        )
        
        # Configure finger enable/disable based on config
        if config.enabled_fingers:
            # Explicit set: disable all then enable selected
            self.tactile_processor.enable_all_fingers(False)
            for f in config.enabled_fingers:
                self.tactile_processor.enable_finger(f, True)
            # Also restrict background creation to these
            self.config.tactile_sensors = config.enabled_fingers
            logger.info(f"🖐️ Enabled fingers (explicit): {config.enabled_fingers}")
        elif config.all_fingers:
            self.tactile_processor.enable_all_fingers(True)
            logger.info("🖐️ All fingers enabled (including problematic ones)")
        else:
            # Use defaults based on dataset (middle+ring for sim/occlusion, none for real)
            enabled = self.tactile_processor.get_enabled_fingers()
            logger.info(f"🖐️ Only reliable fingers enabled by default: {enabled}")
        
        # Extract data
        self.timestamps = self.trial_data['time']
        self.object_poses = self.trial_data['object']['pose']
        
        # Determine actual frame count
        total_available_frames = len(self.timestamps)
        if self.config.max_frames == -1:
            self.actual_max_frames = total_available_frames
        else:
            self.actual_max_frames = min(self.config.max_frames, total_available_frames)
        
        logger.info("🎯 Camera-Tactile Fusion Test initialized")
        logger.info(f"   📂 Trial: {config.trial_path}")
        logger.info(f"   📁 Output: {config.output_dir}")
        logger.info(f"   🎞️ Available frames: {total_available_frames}")
        logger.info(f"   🎯 Processing frames: {self.actual_max_frames}")
    
    def _load_trial_data(self) -> Dict:
        """Load trial data"""
        data_path = Path(self.config.trial_path) / "data.pkl"
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    
    def create_tactile_backgrounds(self):
        """Create tactile background templates using variance-based selection"""
        
        logger.info(f"🔄 Creating tactile background templates...")
        
        # Only create backgrounds for enabled fingers
        enabled_fingers = self.tactile_processor.get_enabled_fingers()
        for finger in self.config.tactile_sensors:
            if finger in enabled_fingers:
                # Use variance-based background selection
                self.tactile_processor.create_tactile_background_template_with_variance_selection(finger)

    def save_tactile_debug_composite(self, finger: str, frame_idx: int, out_path: str) -> Dict:
        """Save 2x2 composite (input, background, TouchVIT depth, contact mask) for a finger/frame.

        Does not change Redwood thresholds; catches Redwood failure only to ensure the image is saved,
        and returns stats including the Redwood error string when applicable.
        """
        # Ensure background is ready for this finger (adapter-based)
        if finger not in getattr(self.tactile_processor.tactile_depth, 'bg_template', {}):
            self.tactile_processor.create_tactile_background_template_with_variance_selection(finger)
        return self.tactile_processor.debug_save_composite(finger, frame_idx, out_path)
    
    def load_tactile_images_batch(self, finger: str, frame_indices: List[int]) -> List[np.ndarray]:
        """Load batch of tactile images"""
        batch_images = []
        
        for frame_idx in frame_indices:
            # Include all frames (including frame 0) to match camera pipeline
                
            tactile_image_path = Path(self.config.trial_path) / "allegro" / finger / "image" / f"{frame_idx}.jpg"
            
            if tactile_image_path.exists():
                try:
                    image = cv2.imread(str(tactile_image_path))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        batch_images.append(image)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {tactile_image_path}: {e}")
        
        return batch_images
    
    def process_finger_tactile(self, finger: str, start_frame: int, end_frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process finger batch using tactile processing pipeline"""
        
        # Check if finger_poses exists in the data (feelsight_real compatibility)
        has_fp = 'finger_poses' in self.trial_data.get('allegro', {})
        if not has_fp and not self.config.allow_approx_tip_poses:
            logger.warning("finger_poses not found in data - feelsight_real data doesn't contain finger poses")
            return np.array([]), np.array([])
        
        # Use same frame indices as camera pipeline (including frame 0)
        frame_indices = list(range(start_frame, end_frame))
        batch_images = self.load_tactile_images_batch(finger, frame_indices)
        
        if not batch_images:
            return np.array([]), np.array([])
        
        finger_mapping = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3}
        finger_idx = finger_mapping.get(finger, -1)
        
        if finger_idx == -1:
            return np.array([]), np.array([])
        
        all_object_points = []
        all_frame_indices = []
        contact_pixels_total = 0
        total_pixels = 0
        
        # Prepare finger poses computation if needed
        computed_finger_poses = None
        if not has_fp and (self.config.allow_approx_tip_poses or self.config.use_forward_kinematics):
            try:
                if self.config.use_forward_kinematics:
                    # Use proper forward kinematics
                    from .allegro_fk import AllegroForwardKinematics
                    fk = AllegroForwardKinematics(device='cpu')  # Use CPU for compatibility
                    computed_finger_poses = {}
                    
                    # Pre-compute finger poses for all frames in this batch
                    for batch_frame_idx in frame_indices:
                        if batch_frame_idx < self.trial_data['allegro']['joint_state'].shape[0]:
                            joint_state = self.trial_data['allegro']['joint_state'][batch_frame_idx]
                            base_pose = self.trial_data['allegro']['base_pose']
                            fp = fk.compute_finger_poses(joint_state, base_pose)
                            computed_finger_poses[batch_frame_idx] = fp
                    
                    logger.debug(f"✅ Computed finger poses using FK for {len(computed_finger_poses)} frames")
                
                else:
                    raise RuntimeError("CRITICAL: No finger poses available and FK computation failed")
                    
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Finger pose computation failed: {e}") from e

        for img_idx, tactile_image in enumerate(batch_images):
            frame_idx = frame_indices[img_idx]
            # Note: frame_idx is already filtered (no frame 0) from load_tactile_images_batch
            
            # Get poses for this frame
            object_pose = self.object_poses[frame_idx]
            if has_fp:
                finger_poses = self.trial_data['allegro']['finger_poses'][frame_idx]
                finger_pose = finger_poses[finger_idx]
            elif computed_finger_poses is not None and frame_idx in computed_finger_poses:
                # Use computed finger poses from FK
                finger_poses = computed_finger_poses[frame_idx]
                finger_pose = finger_poses[finger_idx]
            else:
                raise RuntimeError(f"CRITICAL: No finger pose available for frame {frame_idx}, finger {finger_idx}")
            
            # Tactile processing pipeline
            depth_map, contact_mask = self.tactile_processor.process_tactile_image(
                tactile_image, finger, frame_idx
            )
            
            # Skip frames with insufficient contact to avoid false positives
            try:
                min_px = getattr(self.config, 'tactile_min_contact_pixels', 0) or 0
            except Exception:
                min_px = 0
            if min_px > 0 and contact_mask.numel() > 0:
                if int(contact_mask.sum().item()) < min_px:
                    continue
            
            # Statistics
            if contact_mask.numel() > 0:
                contact_pixel_count = contact_mask.sum().item()
                contact_pixels_total += contact_pixel_count
                total_pixels += contact_mask.numel()
            
            # Back-project only masked contact depths
            sensor_points = self.tactile_processor.back_project_masked_depths_only(
                depth_map, contact_mask, finger
            )
            
            if len(sensor_points) > 0:
                # Transform chain: sensor/optical → fingertip → world → object
                sensor_points_np = sensor_points.cpu().numpy()
                points_h = np.hstack([sensor_points_np, np.ones((len(sensor_points_np), 1), dtype=np.float32)])

                # Transform matrix: T_WC = world_T_tip · tip_T_optical
                T_opt2tip = self.tactile_processor.get_T_optical_to_tip(finger)  # optical → tip
                T_tip2opt = np.linalg.inv(T_opt2tip)  # tip → optical (what we need)
                T_WC = finger_pose @ T_tip2opt  # world_T_optical (the critical T_WC matrix)
                
                # Apply T_WC to transform optical points directly to world
                world_points_h = (T_WC @ points_h.T).T
                world_points = world_points_h[:, :3]
                
                # World → Object
                world_to_object = np.linalg.inv(object_pose)
                object_points_h = (world_to_object @ np.hstack([world_points, np.ones((len(world_points), 1))]).T).T
                object_points = object_points_h[:, :3]
                
                all_object_points.append(object_points)
                frame_labels = np.full(len(object_points), frame_idx)
                all_frame_indices.append(frame_labels)
        
        # Log statistics
        if total_pixels > 0:
            contact_ratio = contact_pixels_total / total_pixels * 100
            logger.debug(f"   {finger}: {contact_pixels_total:,}/{total_pixels:,} pixels contact ({contact_ratio:.1f}%)")
        
        if all_object_points:
            combined_points = np.vstack(all_object_points)
            combined_frames = np.concatenate(all_frame_indices)
            return combined_points, combined_frames
        else:
            return np.array([]), np.array([])
    
    def run_camera_reconstruction_test(self) -> Dict[str, Any]:
        """Run camera reconstruction test"""
        
        logger.info("📷 Starting camera reconstruction test...")
        start_time = time.time()
        
        try:
            # Import camera pipeline
            from camera.camera_pipeline import CameraPipeline
            
            class SimpleConfig:
                def __init__(self, dataset_path):
                    self.data = SimpleData(dataset_path)
            
            class SimpleData:
                def __init__(self, dataset_path):
                    self.dataset_path = dataset_path
            
            config = SimpleConfig(self.config.trial_path)
            pipeline = CameraPipeline(config, output_dir=str(self.output_dir))
            
            # Handle all frames case
            if self.config.max_frames == -1:
                # Let camera pipeline determine total available frames
                results = pipeline.process_frames(max_frames=None, verbose=True)
            else:
                results = pipeline.process_frames(max_frames=self.config.max_frames, verbose=True)
            
            if not results.get("success", False):
                logger.error(f"❌ Camera pipeline failed: {results.get('error')}")
                return {'success': False, 'error': 'Camera pipeline failed'}
            
            ply_name = f"camera_{self.actual_max_frames}f.ply"
            ply_path = pipeline.save_results(results, filename=ply_name)
            
            camera_results = {
                'success': True,
                'method': 'Camera 3DGS (World Frame)',
                'ply_path': ply_path,
                'points_count': results.get('total_points', 0),
                'processing_time': time.time() - start_time,
                'frames_processed': self.actual_max_frames,
                'coordinate_frame': 'world'
            }
            
            logger.info(f"✅ Camera reconstruction: {camera_results['points_count']:,} points")
            return camera_results
            
        except Exception as e:
            logger.error(f"❌ Camera reconstruction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_tactile_reconstruction(self) -> Dict[str, Any]:
        """Run tactile reconstruction using TouchVIT depth prediction"""
        
        logger.info("🤏 Starting tactile reconstruction...")
        start_time = time.time()
        
        # Create tactile background templates first
        self.create_tactile_backgrounds()
        
        all_object_points = []
        all_object_colors = []
        contact_stats = {}
        
        finger_colors = {
            'thumb': [255, 0, 0],    # Red
            'index': [0, 255, 0],    # Green  
            'middle': [0, 0, 255],   # Blue
            'ring': [255, 255, 0]    # Yellow
        }
        
        total_batches = (self.actual_max_frames + self.config.batch_size - 1) // self.config.batch_size
        
        try:
            for batch_idx in range(total_batches):
                start_frame = batch_idx * self.config.batch_size
                end_frame = min(start_frame + self.config.batch_size, self.actual_max_frames)
                
                logger.info(f"📦 Processing batch {batch_idx + 1}/{total_batches} (frames {start_frame}-{end_frame-1})")
                
                # Only process enabled fingers
                enabled_fingers = self.tactile_processor.get_enabled_fingers()
                for finger in enabled_fingers:
                    # Tactile processing
                    object_points, frame_indices = self.process_finger_tactile(
                        finger, start_frame, end_frame
                    )
                    
                    if len(object_points) == 0:
                        contact_stats[finger] = contact_stats.get(finger, 0)
                        continue
                    
                    all_object_points.append(object_points)
                    
                    # Add colors
                    finger_color = np.array(finger_colors[finger])
                    colors = np.tile(finger_color, (len(object_points), 1))
                    all_object_colors.append(colors)
                    
                    contact_stats[finger] = contact_stats.get(finger, 0) + len(object_points)
                    logger.info(f"  {finger}: +{len(object_points):,} tactile contact points")
                
                # Memory management
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                current_total = sum(len(pts) for pts in all_object_points)
                logger.info(f"📊 Batch {batch_idx + 1} complete. Total tactile points: {current_total:,}")
            
            # Combine results
            if all_object_points:
                combined_points = np.vstack(all_object_points)
                combined_colors = np.vstack(all_object_colors)
                
                # Save tactile reconstruction PLY
                tactile_ply_path = self.output_dir / f"tactile_reconstruction_{self.actual_max_frames}f.ply"
                self._save_ply(combined_points, combined_colors, str(tactile_ply_path))
                
                tactile_results = {
                    'success': True,
                    'method': 'TouchVIT Tactile Reconstruction',
                    'ply_path': str(tactile_ply_path),
                    'points_count': len(combined_points),
                    'processing_time': time.time() - start_time,
                    'frames_processed': self.actual_max_frames,
                    'coordinate_frame': 'object',
                    'touchvit_based': True,
                    'contact_stats': contact_stats
                }
                
                logger.info(f"✅ TouchVIT tactile reconstruction: {len(combined_points):,} contact points")
                logger.info(f"📊 Tactile contact statistics:")
                for finger, count in contact_stats.items():
                    logger.info(f"   {finger}: {count:,} contact points")
                
                return tactile_results
            else:
                logger.warning("⚠️ No tactile contact points detected!")
                return {
                    'success': True,
                    'method': 'TouchVIT Tactile - No contacts',
                    'ply_path': None,
                    'points_count': 0,
                    'processing_time': time.time() - start_time,
                    'frames_processed': self.actual_max_frames,
                    'coordinate_frame': 'object',
                    'touchvit_based': True,
                    'contact_stats': contact_stats
                }
                
        except Exception as e:
            logger.error(f"❌ TouchVIT tactile reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def transform_camera_to_object_frame(self, camera_ply_path: str, frame_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Transform camera points to object frame"""
        
        logger.info("🔄 Transforming camera points to object frame...")
        
        try:
            camera_pcd = o3d.io.read_point_cloud(camera_ply_path)
            camera_points_world = np.asarray(camera_pcd.points)
            camera_colors = np.asarray(camera_pcd.colors) * 255.0
            
            # Transform to object frame
            object_pose = self.object_poses[frame_idx]
            world_to_object = np.linalg.inv(object_pose)
            
            camera_points_homogeneous = np.hstack([camera_points_world, np.ones((len(camera_points_world), 1))])
            camera_object_homogeneous = (world_to_object @ camera_points_homogeneous.T).T
            camera_points_object = camera_object_homogeneous[:, :3]
            
            logger.info(f"   ✅ Transformed {len(camera_points_object):,} camera points to object frame")
            
            return camera_points_object, camera_colors
            
        except Exception as e:
            logger.error(f"❌ Camera transformation failed: {e}")
            return np.array([]), np.array([])
    
    def run_camera_tactile_fusion(self, camera_results: Dict, tactile_results: Dict) -> Dict[str, Any]:
        """Run camera-tactile fusion"""
        
        logger.info("🔀 Starting camera-tactile fusion...")
        start_time = time.time()
        
        try:
            # Transform camera points to object frame
            camera_points_object, camera_colors = self.transform_camera_to_object_frame(
                camera_results['ply_path'], frame_idx=0
            )
            
            if len(camera_points_object) == 0:
                logger.error("❌ Failed to transform camera points")
                return {'success': False, 'error': 'Camera transformation failed'}
            
            # Load tactile reconstruction points
            if tactile_results.get('ply_path') and Path(tactile_results['ply_path']).exists():
                tactile_pcd = o3d.io.read_point_cloud(tactile_results['ply_path'])
                tactile_points_object = np.asarray(tactile_pcd.points)
                tactile_colors = np.asarray(tactile_pcd.colors) * 255.0
                logger.info(f"   🤏 Tactile points: {len(tactile_points_object):,}")
            else:
                tactile_points_object = np.array([]).reshape(0, 3)
                tactile_colors = np.array([]).reshape(0, 3)
                logger.info("   🤏 No tactile points available")
            
            # Check alignment
            if len(tactile_points_object) > 0:
                camera_center = np.mean(camera_points_object, axis=0)
                tactile_center = np.mean(tactile_points_object, axis=0)
                
                displacement = np.linalg.norm(camera_center - tactile_center)
                logger.info(f"   📏 Camera-tactile alignment displacement: {displacement:.3f} m")
                
                # Combine points
                fused_points = np.vstack([camera_points_object, tactile_points_object])
                fused_colors = np.vstack([camera_colors, tactile_colors])
            else:
                fused_points = camera_points_object
                fused_colors = camera_colors
                displacement = 0
            
            # Save camera-tactile fusion
            fused_ply_path = self.output_dir / f"fused_camera_tactile_{self.actual_max_frames}f.ply"
            self._save_ply(fused_points, fused_colors, str(fused_ply_path))
            
            
            # TouchVIT neural network was used successfully (even if no contacts detected)
            
            fusion_results = {
                'success': True,
                'method': 'Camera-Tactile Fusion',
                'ply_path': str(fused_ply_path),
                'points_count': len(fused_points),
                'camera_points_transformed': len(camera_points_object),
                'tactile_points': len(tactile_points_object),
                'colors_preserved': True,
                'alignment_displacement': displacement,
                'coordinate_frame': 'object',
                'method_exact': True,
                'touchvit_based': tactile_results.get('touchvit_based', False),  # Forward the flag
                'contact_stats': tactile_results.get('contact_stats', {}),
                'processing_time': time.time() - start_time
            }
            
            logger.info(f"✅ Camera-tactile fusion: {len(fused_points):,} points")
            logger.info(f"   📷 Camera: {len(camera_points_object):,}")
            logger.info(f"   🤏 Tactile: {len(tactile_points_object):,}")
            
            return fusion_results
            
        except Exception as e:
            logger.error(f"❌ Camera-tactile fusion failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _save_ply(self, points: np.ndarray, colors: np.ndarray, filepath: str):
        """Save points and colors as PLY file"""
        if len(points) == 0:
            logger.warning(f"⚠️ No points to save to {filepath}")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) == len(points):
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filepath, pcd)
        logger.info(f"💾 Saved {len(points):,} points to {filepath}")
    
    def run_complete_tactile_fusion_test(self) -> Dict[str, Any]:
        """Run complete camera-tactile fusion test"""
        
        logger.info("🎯 STARTING CAMERA-TACTILE FUSION TEST")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # Step 1: Camera reconstruction
        logger.info("📷 STEP 1: Camera Reconstruction Test")
        logger.info("-" * 50)
        camera_results = self.run_camera_reconstruction_test()
        
        if not camera_results['success']:
            return {'success': False, 'error': 'Camera reconstruction failed'}
        
        # Step 2: TouchVIT tactile reconstruction
        logger.info("\n🤏 STEP 2: TouchVIT Tactile Reconstruction")
        logger.info("-" * 50)
        tactile_results = self.run_tactile_reconstruction()
        
        if not tactile_results['success']:
            return {'success': False, 'error': 'TouchVIT tactile reconstruction failed'}
        
        # Step 3: Camera-tactile fusion
        logger.info("\n🔀 STEP 3: Camera-Tactile Fusion")
        logger.info("-" * 50)
        fusion_results = self.run_camera_tactile_fusion(camera_results, tactile_results)
        
        if not fusion_results['success']:
            return {'success': False, 'error': 'Camera-tactile fusion failed'}
        
        total_time = time.time() - total_start_time
        
        # Final results
        final_results = {
            'camera_results': camera_results,
            'tactile_results': tactile_results,
            'fused_results': fusion_results
        }
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 CAMERA-TACTILE FUSION RESULTS")
        logger.info("=" * 80)
        logger.info(f"🕒 Total time: {total_time:.2f}s")
        logger.info(f"🎞️ Frames processed: {self.actual_max_frames}")
        logger.info(f"📷 Camera points: {camera_results.get('points_count', 0):,}")
        logger.info(f"🤏 TouchVIT tactile points: {tactile_results.get('points_count', 0):,}")
        logger.info(f"🔀 Camera-tactile fusion points: {fusion_results.get('points_count', 0):,}")
        logger.info(f"✅ TouchVIT pipeline used: {fusion_results.get('touchvit_based', False)}")
        logger.info(f"🌈 Colors preserved: {fusion_results.get('colors_preserved', False)}")
        logger.info(f"📏 Final alignment displacement: {fusion_results.get('alignment_displacement', 0):.3f} m")
        
        if 'contact_stats' in tactile_results and tactile_results['contact_stats']:
            logger.info("\n📊 TouchVIT Contact Statistics:")
            for finger, count in tactile_results['contact_stats'].items():
                logger.info(f"   🤏 {finger}: {count:,} contact points")
        
        logger.info(f"\n📁 Results:")
        for key, result in final_results.items():
            if result['success'] and result.get('ply_path'):
                logger.info(f"   {key}: {result['ply_path']}")
        
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info("=" * 80)
        
        # Save results
        results_path = self.output_dir / "camera_tactile_fusion_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(final_results, f)
        
        final_results['total_time'] = total_time
        final_results['overall_success'] = True
        
        return final_results
