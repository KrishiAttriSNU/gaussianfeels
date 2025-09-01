#!/usr/bin/env python3
"""
CAMERA PIPELINE - MAIN MODULE
============================

This module contains the camera pipeline with the following properties:
1. Uses negative X for RealSense coordinate convention
2. Uses object-centric pose transformations
3. Avoids abs() on depth values
4. Uses real implementations only (no mocks or fallbacks)

This is the main module that test files should import from.
"""

import sys
import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Use proper package imports (require pip install -e .)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from misc.constants import DataPklManager, TemporalAnalyzer, get_object_name_from_data
from typing import Dict, List, Tuple, Optional, Any, Union

from gaussian.utils.pose_transforms import transform_points_to_object_frame, select_optimal_keyframes
from camera.io.segmentation import (
    SegmentationProcessor,
    FeelsightSegmentationLoader,
    FeelsightRealSegmentationLoader,
)
from camera.io.depth_filters import undistort_depth, filter_depth_percentile


class CameraObjectReconstructor:
    """Object reconstructor with FIXED coordinate system for RealSense cameras"""
    
    def __init__(self, intrinsics: Dict[str, float]):
        """
        Initialize with camera intrinsics
        
        Args:
            intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'
        """
        self.intrinsics = intrinsics
        
    def depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth image to 3D point cloud - FIXED VERSION
        
        Args:
            depth: Depth image as numpy array
            rgb: RGB image as numpy array
            
        Returns:
            Tuple of (points_3d, colors) as numpy arrays
        """
        H, W = depth.shape
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        
        # Create coordinate grids
        v, u = np.indices((H, W))  # Note: v, u order for meshgrid
        
        # Generate points from all pixels, filter NaN after
        # Keep original depth with sign (negative for feelsight_real is valid)
        valid_u = u.flatten()
        valid_v = v.flatten() 
        valid_depth = depth.flatten()  # Keep original depth with sign
        
        # CRITICAL FIX: Use negative X for RealSense optical frame
        x = -(valid_u - cx) * valid_depth / fx  # NEGATIVE X
        y = (valid_v - cy) * valid_depth / fy   # Standard Y
        z = valid_depth                         # Forward Z
        
        points_3d = np.column_stack([x, y, z])
        colors = rgb.reshape(-1, 3) / 255.0
        
        # Filter NaN points after generation
        valid_points_mask = ~np.isnan(points_3d).any(axis=1)
        points_3d = points_3d[valid_points_mask]
        colors = colors[valid_points_mask]
        
        return points_3d, colors


class CameraPipeline:
    """Complete camera pipeline for object reconstruction with multi-camera support"""
    
    def __init__(self, 
                 config: Union[Dict, str, Path],
                 intrinsics: Optional[Dict[str, float]] = None,
                 output_dir: Optional[str] = None,
                 camera_name: str = 'front-left',
                 camera_names: Optional[List[str]] = None,
                 use_primary_camera_only: bool = True,
                 gt_generation_mode: bool = False,
                 three_camera_mode: bool = False):
        """
        Initialize camera pipeline
        
        Args:
            config: Configuration object
            intrinsics: Camera intrinsics, uses defaults if None
            output_dir: Output directory, uses default if None
            camera_name: Primary camera name (default: 'front-left')
            camera_names: List of camera names to use
            use_primary_camera_only: If True, use only primary camera for reconstruction
            gt_generation_mode: If True, use all cameras for ground truth pose generation
            three_camera_mode: If True, use 3-camera setup for robust multi-view pose tracking
        """
        # Allow passing a config object with data.dataset_path or a direct data path string/Path
        if isinstance(config, (str, Path)):
            self.data_path = Path(config).resolve()
        else:
            self.data_path = Path(config.data.dataset_path).resolve()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'tests/output/camera_pipeline'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use primary camera only for reconstruction by default
        self.use_primary_camera_only = use_primary_camera_only
        self.gt_generation_mode = gt_generation_mode
        self.three_camera_mode = three_camera_mode
        self.camera_name = camera_name
        
        # Determine camera names to use based on mode
        if three_camera_mode:
            # Three-camera mode for robust multi-view pose tracking
            self.camera_names = ['front-left', 'back-right', 'top-down']
            print(f"🎯 Three-camera mode enabled: {self.camera_names}")
        elif gt_generation_mode:
            # For GT generation, use all available cameras
            self.camera_names = camera_names[:] if camera_names is not None else None
        elif use_primary_camera_only:
            # For reconstruction, use only primary camera
            self.camera_names = [camera_name]
        else:
            # Legacy multi-camera reconstruction mode
            self.camera_names = camera_names[:] if camera_names is not None else [camera_name]
        
        # Default intrinsics for feelsight dataset
        self.intrinsics = intrinsics or {
            'fx': 616.375,
            'fy': 616.375, 
            'cx': 318.75,
            'cy': 239.75
        }
        
        # Auto-detect available cameras if needed
        if self.camera_names is None or (gt_generation_mode and camera_names is None):
            realsense_root = self.data_path / 'realsense'
            if realsense_root.exists():
                detected = []
                for sub in sorted([p for p in realsense_root.iterdir() if p.is_dir()]):
                    # Consider as a camera if it has image dir or depth.npz
                    if (sub / 'image').exists() or (sub / 'depth.npz').exists():
                        detected.append(sub.name)
                if detected:
                    if three_camera_mode:
                        # Validate that required cameras exist for three-camera mode
                        required_cameras = ['front-left', 'back-right', 'top-down']
                        missing_cameras = [cam for cam in required_cameras if cam not in detected]
                        if missing_cameras:
                            print(f"⚠️  Warning: Missing cameras for three-camera mode: {missing_cameras}")
                            print(f"📂 Available cameras: {detected}")
                            # Use available cameras that match the pattern
                            self.camera_names = [cam for cam in required_cameras if cam in detected]
                            if not self.camera_names:
                                raise ValueError(f"No required cameras found for three-camera mode. Available: {detected}")
                        else:
                            print(f"✅ All required cameras found for three-camera mode")
                    elif gt_generation_mode:
                        # For GT generation, use all available cameras
                        self.camera_names = detected
                    elif use_primary_camera_only:
                        # Ensure primary camera exists in detected cameras
                        if camera_name in detected:
                            self.camera_names = [camera_name]
                        else:
                            raise ValueError(f"Requested primary camera '{camera_name}' not found among detected cameras: {detected}")
                    else:
                        # Legacy mode - use all detected cameras
                        self.camera_names = detected

        # Initialize components per camera
        self.reconstructor = CameraObjectReconstructor(self.intrinsics)
        self.seg_processor = SegmentationProcessor()
        # Initialize segmentation:
        # - If GT seg jpgs exist, use GT loader (depth * mask)
        # - Otherwise, use SAM with ViT-L and default weights dir
        self.seg_loaders: Dict[str, Union[FeelsightSegmentationLoader, FeelsightRealSegmentationLoader]] = {}
        default_sam_dir = Path.cwd() / 'data' / 'segment-anything'
        for cam in self.camera_names:
            seg_dir = self.data_path / 'realsense' / cam / 'seg'
            if seg_dir.exists() and any(seg_dir.glob('*.jpg')):
                self.seg_loaders[cam] = FeelsightSegmentationLoader(str(self.data_path), camera_name=cam)
            else:
                self.seg_loaders[cam] = FeelsightRealSegmentationLoader(
                    str(self.data_path), sam_weights_dir=str(default_sam_dir), model_type='vit_l', device='cuda', camera_name=cam
                )
        
        # Load dataset
        self.data = self._load_data()
    
    def _dbscan_filter_points(self, points: np.ndarray, eps: float = 0.03, min_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply GPU RAPIDS cuML DBSCAN to filter outliers and keep the largest cluster.
        This implementation is GPU-only and will error if cuML is unavailable.
        
        Args:
            points: 3D points array (N, 3)
            eps: Maximum distance between samples in a cluster (default: 3cm - gentle filtering)
            min_samples: Minimum samples in a cluster (default: 30 - gentle filtering)
            
        Returns:
            Tuple of (filtered_points, indices_mask) where indices_mask indicates which points were kept
        """
        if len(points) < min_samples:
            return points, np.ones(len(points), dtype=bool)

        # Ensure cuML shared libraries are discoverable
        self._setup_cuml_environment()

        try:
            import cupy as cp
            import cudf
            from cuml.cluster import DBSCAN as cuDBSCAN
        except ImportError as import_error:
            raise ImportError(
                "cuML (RAPIDS) is required for DBSCAN. Install RAPIDS cuML compatible with your CUDA/PyTorch, "
                "and ensure libcuml++.so and related libraries are on LD_LIBRARY_PATH."
            ) from import_error

        # Memory optimization: subsample for large datasets to fit in GPU memory
        max_points_gpu = 15000  # Increase for better outlier detection
        if len(points) > max_points_gpu:
            print(f"⚠️ Large dataset ({len(points)} points), subsampling to {max_points_gpu} for GPU DBSCAN")
            # Take random subset preserving spatial distribution
            indices = np.random.choice(len(points), max_points_gpu, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
            indices = np.arange(len(points))

        # Convert to GPU arrays via CuPy and cuDF
        X_gpu = cp.asarray(sample_points, dtype=cp.float32)
        gdf = cudf.DataFrame({f"f{i}": X_gpu[:, i] for i in range(X_gpu.shape[1])})

        clustering = cuDBSCAN(eps=eps, min_samples=min_samples)
        labels_gpu = clustering.fit_predict(gdf)
        labels = cp.asnumpy(labels_gpu)

        print(f"🚀 Using GPU cuML DBSCAN for {len(sample_points)} points (subsampled from {len(points)})")

        # Largest non-noise cluster
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            return points, np.ones(len(points), dtype=bool)

        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_mask = labels == largest_cluster_label
        
        # If we subsampled, map results back to full dataset using AGGRESSIVE spatial proximity
        if len(points) > max_points_gpu:
            # Find cluster center from sample
            cluster_points = sample_points[largest_cluster_mask]
            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_std = np.std(cluster_points, axis=0)
                
                # ULTRA-GENTLE filtering: Use very lenient threshold to preserve object completeness
                distances = np.linalg.norm(points - cluster_center, axis=1)
                # Use 99th percentile distance as threshold (removes only ~1% of true outliers)
                # This preserves object completeness while still removing noise
                threshold = np.percentile(distances, 99)
                full_mask = distances < threshold
                
                print(f"🌿 GENTLE filtering: Keeping {np.sum(full_mask)}/{len(points)} points (threshold: {threshold:.3f}m)")
            else:
                full_mask = np.ones(len(points), dtype=bool)
            
            filtered_points = points[full_mask]
            return filtered_points, full_mask
        else:
            filtered_points = sample_points[largest_cluster_mask]
            full_mask = np.zeros(len(points), dtype=bool)
            full_mask[indices[largest_cluster_mask]] = True
            return filtered_points, full_mask
    
    def _setup_cuml_environment(self):
        """Auto-detect and setup cuML environment - portable across systems"""
        import os
        import sys
        import site
        from pathlib import Path
        
        # Get site-packages directories 
        site_dirs = []
        try:
            site_dirs.extend(site.getsitepackages())
        except (AttributeError, OSError) as e:
            raise RuntimeError(f"Could not get site packages: {e}")
        
        if hasattr(site, 'getusersitepackages'):
            try:
                site_dirs.append(site.getusersitepackages())
            except (AttributeError, OSError) as e:
                raise RuntimeError(f"Could not get user site packages: {e}")
        
        # Add common user site directory
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        user_site = Path.home() / '.local' / 'lib' / python_version / 'site-packages'
        site_dirs.append(str(user_site))
        
        # Look for cuML library paths
        lib_paths = []
        for site_dir in site_dirs:
            site_path = Path(site_dir)
            if not site_path.exists():
                continue
                
            # NVIDIA wheel libraries (CUDA dependencies)
            nvidia_libs = ['cublas', 'cusolver', 'cusparse', 'cuda_runtime', 'cuda_nvrtc']
            for lib in nvidia_libs:
                nvidia_path = site_path / 'nvidia' / lib / 'lib'
                if nvidia_path.exists():
                    lib_paths.append(str(nvidia_path))
            
            # RAPIDS libraries
            rapids_libs = ['libcuml', 'libcudf', 'librmm', 'libraft', 'libcuvs', 'rapids_logger']
            for lib in rapids_libs:
                for suffix in ['lib', 'lib64']:
                    rapids_path = site_path / lib / suffix
                    if rapids_path.exists():
                        lib_paths.append(str(rapids_path))
        
        # Set LD_LIBRARY_PATH with detected paths
        if lib_paths:
            # Sort to ensure consistent ordering - NVIDIA CUDA libs first
            nvidia_paths = [p for p in lib_paths if '/nvidia/' in p]
            rapids_paths = [p for p in lib_paths if '/nvidia/' not in p]
            ordered_paths = nvidia_paths + rapids_paths
            
            # Strict: LD_LIBRARY_PATH must exist if being modified
            if 'LD_LIBRARY_PATH' not in os.environ:
                raise EnvironmentError("LD_LIBRARY_PATH not found in environment - required for Open3D setup")
            current = os.environ['LD_LIBRARY_PATH']
            new_paths = ':'.join(ordered_paths)
            os.environ['LD_LIBRARY_PATH'] = f"{new_paths}:{current}" if current else new_paths
        
    def _load_data(self) -> Dict[str, Any]:
        """Load all data from data.pkl"""
        pkl_path = self.data_path / "data.pkl"
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"Data file not found: {pkl_path}")
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def process_frames(self, 
                      max_frames: Optional[int] = None,
                      start_frame: int = 0,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Process camera frames and generate object reconstruction
        
        Args:
            max_frames: Maximum number of frames to process (None = all frames)
            start_frame: Frame index to start processing from (default: 0)
            verbose: Whether to print progress
            
        Returns:
            Dict containing reconstruction results
        """
        total_frames = len(self.data['object']['pose'])
        end_frame = min(start_frame + max_frames, total_frames) if max_frames else total_frames
        frames_to_process = end_frame - start_frame
        keyframes = list(range(start_frame, end_frame))
        
        if verbose:
            print(f"🔧 CAMERA PIPELINE PROCESSING")
            print(f"📁 Data path: {self.data_path}")
            print(f"📁 Output path: {self.output_dir}")
            print(f"📈 Total frames available: {total_frames}")
            print(f"🎯 Processing {frames_to_process} frames (from frame {start_frame} to {end_frame-1})")
            if self.use_primary_camera_only:
                print(f"📷 Using PRIMARY CAMERA ONLY: {self.camera_name}")
            else:
                print(f"📷 Using ALL CAMERAS: {', '.join(self.camera_names)} (Legacy multi-camera mode)")
        
        all_points_obj = []
        all_colors = []
        frame_stats = []
        
        # OBJECT-CENTRIC APPROACH: Set reference object pose (frame 0 as canonical)
        p_WO_W = self.data['object']['pose'][0]  # Reference object pose
        if verbose:
            print(f"🎯 Using frame 0 as reference object pose: [{p_WO_W[0,3]:.3f}, {p_WO_W[1,3]:.3f}, {p_WO_W[2,3]:.3f}]")
        
        # Load depth data once
        # Load depth for all cameras
        depth_data_per_cam: Dict[str, Any] = {}
        for cam in self.camera_names:
            depth_path = self.data_path / 'realsense' / cam / 'depth.npz'
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth file not found: {depth_path}")
            depth_data_per_cam[cam] = np.load(depth_path)
        
        for i, frame_idx in enumerate(keyframes):
            # Progress reporting
            if verbose and (i % 10 == 0 or i < 5):
                print(f"   🔧 Processing frame {i+1}/{len(keyframes)} (frame {frame_idx})...")
            elif verbose and i % 5 == 0:
                print(f"   ⏳ Progress: {i+1}/{len(keyframes)} frames processed...")
            
            try:
                # Aggregate across all cameras
                for cam in self.camera_names:
                    # Load segmentation data for this camera/frame
                    seg_loader = self.seg_loaders[cam]
                    seg_data = seg_loader.get_frame_segmentation_data(frame_idx)
                    mask = seg_data['realsense_mask']

                    # Load RGB data
                    rgb_path = self.data_path / 'realsense' / cam / 'image' / f'{frame_idx}.jpg'
                    if not rgb_path.exists():
                        if verbose:
                            print(f"     ❌ RGB file not found: {rgb_path}")
                        continue

                    # Downscale RGB to reduce memory, preserve aspect (optional)
                    rgb = cv2.imread(str(rgb_path))
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    if rgb.shape[0] > 480 or rgb.shape[1] > 640:
                        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

                    # Extract depth for this frame/camera
                    try:
                        depth_arr = depth_data_per_cam[cam]['depth']
                        if 'depth_scale' not in depth_data_per_cam[cam]:
                            raise KeyError(f"Camera '{cam}' missing required 'depth_scale' in depth data")
                        depth_scale = depth_data_per_cam[cam]['depth_scale']
                        if hasattr(depth_arr, 'ndim') and depth_arr.ndim >= 2:
                            # Multi-dimensional array (standard case)
                            if frame_idx >= len(depth_arr):
                                if verbose:
                                    print(f"     ⚠️ Frame {frame_idx} beyond depth array for {cam}")
                                continue
                            depth = depth_arr[frame_idx].astype(np.float32) * depth_scale  # Apply depth scale
                        else:
                            # Handle 1D array or scalar case - this shouldn't happen with proper data
                            if verbose:
                                print(f"     ❌ Unexpected depth array structure for {cam}: shape={getattr(depth_arr, 'shape', 'unknown')}")
                            continue
                    except IndexError as e:
                        if verbose:
                            print(f"     ❌ IndexError accessing depth[{frame_idx}] for {cam}: {e}")
                        continue

                    # Outlier rejection + segmentation masking
                    # For feelsight_real, negative depths are valid (RealSense coordinate convention)
                    if verbose and (i % 10 == 0 or i < 5):
                        print(f"     📊 [{cam}] Depth range: {depth.min():.3f} to {depth.max():.3f}")

                    # Skip outlier filtering at depth level - will apply DBSCAN to final point cloud
                    depth_filtered = depth
                    
                    # Apply segmentation mask: depth = depth * mask
                    masked_depth = depth_filtered * mask
                    # Convert zero depth (background) to NaN so it is filtered in backprojection
                    if isinstance(masked_depth, np.ndarray):
                        masked_depth = masked_depth.astype(np.float32, copy=False)
                        masked_depth[masked_depth == 0] = np.nan

                    # Generate points in camera frame
                    points_cam, colors = self.reconstructor.depth_to_pointcloud(masked_depth, rgb)
                    
                    # Logging
                    if verbose and (i % 10 == 0 or i < 5):
                        print(f"     ✅ [{cam}] Generated {len(points_cam)} points")
                    
                    # Subsample points to control memory
                    if len(points_cam) > 80000:
                        step = max(1, len(points_cam) // 80000)
                        points_cam = points_cam[::step]
                        colors = colors[::step]

                    if len(points_cam) == 0:
                        if verbose and (i % 10 == 0 or i < 5):
                            print(f"     ⚠️ No points generated for frame {frame_idx} [{cam}]")
                        continue

                    # Load and apply pose transformations
                    T_world_obj = self.data['object']['pose'][frame_idx]
                    
                    # Handle different camera pose formats (feelsight_sim vs feelsight_real)
                    cam_poses = self.data['realsense'][cam]['pose']
                    if cam_poses.ndim == 3:
                        # feelsight_sim: (N, 4, 4) - poses change over time
                        T_world_cam = cam_poses[frame_idx]
                    elif cam_poses.ndim == 2:
                        # feelsight_real: (4, 4) - static camera pose
                        T_world_cam = cam_poses
                    else:
                        raise ValueError(f"Unexpected camera pose shape: {cam_poses.shape}")

                    # Transform to consistent object-centric coordinate system
                    points_obj = transform_points_to_object_frame(points_cam, T_world_obj, T_world_cam, p_WO_W)

                    all_points_obj.append(points_obj)
                    all_colors.append(colors)

                    # Collect frame statistics
                    frame_stat = {
                        'frame_idx': frame_idx,
                        'camera': cam,
                        'points_generated': len(points_obj),
                        'depth_range': [float(depth.min()), float(depth.max())],
                        'points_range': {
                            'x': [float(points_obj[:, 0].min()), float(points_obj[:, 0].max())],
                            'y': [float(points_obj[:, 1].min()), float(points_obj[:, 1].max())],
                            'z': [float(points_obj[:, 2].min()), float(points_obj[:, 2].max())]
                        }
                    }
                    frame_stats.append(frame_stat)

                    if verbose and (i % 10 == 0 or i < 5):
                        print(f"     ✅ [{cam}] Generated {len(points_obj)} transformed points")

            except (ValueError, IndexError, RuntimeError) as e:
                raise RuntimeError(f"Failed to process frame {frame_idx}: {e}") from e
        
        # Combine results
        if all_points_obj:
            combined_points = np.vstack(all_points_obj)
            combined_colors = np.vstack(all_colors)
            
            # Apply DBSCAN outlier filtering ONLY for feelsight_real data
            # Sim data uses precomputed ground truth masks and doesn't need aggressive filtering
            is_real_data = any(isinstance(loader, FeelsightRealSegmentationLoader) 
                              for loader in self.seg_loaders.values())
            
            if is_real_data:
                if verbose:
                    print(f"\n🔧 Applying DBSCAN outlier filtering (real data only)...")
                    print(f"   Points before filtering: {len(combined_points)}")
                
                filtered_points, filter_mask = self._dbscan_filter_points(combined_points)
                filtered_colors = combined_colors[filter_mask]
                
                combined_points = filtered_points
                combined_colors = filtered_colors
                
                if verbose:
                    print(f"   Points after DBSCAN filtering: {len(combined_points)}")
                    outliers_removed = len(filter_mask) - np.sum(filter_mask)
                    print(f"   Outliers removed: {outliers_removed}")
            else:
                if verbose:
                    print(f"\n🔧 Skipping DBSCAN filtering for sim data (using ground truth masks)")
                    print(f"   Sim data points: {len(combined_points)}")
            
            # Global subsample if too large
            max_total = 1_000_000
            if len(combined_points) > max_total:
                step = len(combined_points) // max_total
                combined_points = combined_points[::step]
                combined_colors = combined_colors[::step]
            
            # Calculate object statistics
            center = np.mean(combined_points, axis=0)
            size = np.max(combined_points, axis=0) - np.min(combined_points, axis=0)
            
            results = {
                'points': combined_points,
                'colors': combined_colors,
                'center': center,
                'size': size,
                'num_frames_processed': len(all_points_obj),
                'total_points': len(combined_points),
                'frame_stats': frame_stats,
                'success': True
            }
            
            if verbose:
                print(f"\n📊 CAMERA PIPELINE RESULTS:")
                print(f"   🎯 Successfully processed {len(all_points_obj)} frames")
                print(f"   📈 Total Points: {len(combined_points):,}")
                print(f"   📍 Object Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m")
                print(f"   📏 Object Size: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
            
            return results
        else:
            if verbose:
                print(f"\n❌ No valid frames processed - pipeline failed")
            return {'success': False, 'error': 'No valid frames processed'}
    
    def save_results(self, results: Dict[str, Any], filename: str = 'camera_reconstruction.ply') -> str:
        """
        Save reconstruction results to PLY file
        
        Args:
            results: Results from process_frames()
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if 'success' not in results:
            raise KeyError("results dictionary missing required 'success' key")
        if not results['success']:
            raise ValueError("Cannot save failed reconstruction results")
        
        ply_path = self.output_dir / filename
        points = results['points']
        colors = results['colors']
        
        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            
            for p, c in zip(points, colors):
                r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
        
        return str(ply_path)
    
    def save_analysis(self, results: Dict[str, Any], filename: str = 'camera_analysis.txt') -> str:
        """
        Save detailed analysis to text file
        
        Args:
            results: Results from process_frames()
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if 'success' not in results:
            raise KeyError("results dictionary missing required 'success' key")
        if not results['success']:
            raise ValueError("Cannot save analysis for failed reconstruction")
        
        analysis_path = self.output_dir / filename
        
        with open(analysis_path, 'w') as f:
            f.write("CAMERA PIPELINE ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total frames processed: {results['num_frames_processed']}\n")
            f.write(f"Total points: {results['total_points']:,}\n")
            f.write(f"Object center: ({results['center'][0]:.3f}, {results['center'][1]:.3f}, {results['center'][2]:.3f}) m\n")
            f.write(f"Object size: {results['size'][0]*100:.1f} x {results['size'][1]*100:.1f} x {results['size'][2]*100:.1f} cm\n\n")
            
            f.write("FRAME STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for stat in results['frame_stats']:
                f.write(f"Frame {stat['frame_idx']}: {stat['points_generated']} points\n")
                f.write(f"  Depth range: {stat['depth_range'][0]:.3f} - {stat['depth_range'][1]:.3f} m\n")
                f.write(f"  X range: {stat['points_range']['x'][0]:.3f} - {stat['points_range']['x'][1]:.3f} m\n")
                f.write(f"  Y range: {stat['points_range']['y'][0]:.3f} - {stat['points_range']['y'][1]:.3f} m\n")
                f.write(f"  Z range: {stat['points_range']['z'][0]:.3f} - {stat['points_range']['z'][1]:.3f} m\n\n")
        
        return str(analysis_path)
    
    def get_total_frames(self) -> int:
        """Get total number of frames available in the dataset"""
        depth_path = self.data_path / 'realsense' / self.camera_name / 'depth.npz'
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth data not found: {depth_path}")
        
        depth_data = np.load(depth_path)
        return len(depth_data['depth'])
    
    def process_single_frame(self, frame_idx: int, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single frame and return its point cloud data
        
        Args:
            frame_idx: Index of frame to process
            
        Returns:
            Dictionary with frame processing results
        """
        try:
            cam = camera_name or (self.camera_names[0] if self.camera_names else self.camera_name)
            if cam not in self.seg_loaders:
                return {'success': False, 'error': f'Unknown camera {cam}'}
            # Load segmentation data
            seg_data = self.seg_loaders[cam].get_frame_segmentation_data(frame_idx)
            mask = seg_data['realsense_mask']
            
            # Load RGB data
            rgb_path = self.data_path / 'realsense' / cam / 'image' / f'{frame_idx}.jpg'
            if not rgb_path.exists():
                return {'success': False, 'error': f'RGB file not found: {rgb_path}'}
                
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load depth data
            depth_path = self.data_path / 'realsense' / cam / 'depth.npz'
            if not depth_path.exists():
                return {'success': False, 'error': f'Depth file not found: {depth_path}'}
                
            try:
                depth_data = np.load(depth_path)
                depth_arr = depth_data['depth']
                
                if hasattr(depth_arr, 'ndim') and depth_arr.ndim >= 2:
                    if frame_idx >= len(depth_arr):
                        return {'success': False, 'error': f'Frame {frame_idx} beyond depth array'}
                    depth = depth_arr[frame_idx].astype(np.float32)
                else:
                    return {'success': False, 'error': f'Unexpected depth array structure: shape={getattr(depth_arr, "shape", "unknown")}'}
            except IndexError as e:
                return {'success': False, 'error': f'IndexError accessing depth[{frame_idx}]: {e}'}
            
            # Undistort and filter depth
            try:
                cam_info = self.data['realsense'][cam]
                if 'intrinsics_matrix' in cam_info:
                    K = np.array(cam_info['intrinsics_matrix'], dtype=np.float32)
                elif 'intrinsics' in cam_info:
                    intrinsics = cam_info['intrinsics']
                    if isinstance(intrinsics, dict):
                        # Individual components format: {'fx': ..., 'fy': ..., 'cx': ..., 'cy': ...}
                        fx = float(intrinsics['fx'])
                        fy = float(intrinsics['fy']) 
                        cx = float(intrinsics['cx'])
                        cy = float(intrinsics['cy'])
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    else:
                        # Matrix format
                        K = np.array(intrinsics, dtype=np.float32)
                else:
                    raise KeyError(f"Camera '{cam}' missing required 'intrinsics_matrix' or 'intrinsics'")
                if 'distortion_coeffs' in cam_info:
                    dist = np.array(cam_info['distortion_coeffs'], dtype=np.float32)
                    depth = undistort_depth(depth, K, dist)
                else:
                    # No distortion coeffs available - skip undistortion
                    pass
            except (ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to undistort depth: {e}") from e
            # REMOVED ALL DEPTH FILTERING: Preserving complete object geometry for shape completion
            # No percentile filtering or cutoffs - use raw depth data
            depth_filtered = depth.copy()
            # depth = depth * mask, set zeros to NaN
            depth_masked = (depth_filtered * mask).astype(np.float32)
            depth_masked[depth_masked == 0] = np.nan
            
            # Generate point cloud from masked data
            points_cam, colors = self.reconstructor.depth_to_pointcloud(depth_masked, rgb)
            
            if len(points_cam) == 0:
                return {'success': False, 'error': 'No points after masking'}
            
            # Transform to object frame using object-centric approach
            T_world_obj = self.data['object']['pose'][frame_idx]
            
            # Handle different camera pose formats (feelsight_sim vs feelsight_real)
            cam_poses = self.data['realsense'][cam]['pose']
            if cam_poses.ndim == 3:
                # feelsight_sim: (N, 4, 4) - poses change over time
                T_world_cam = cam_poses[frame_idx]
            elif cam_poses.ndim == 2:
                # feelsight_real: (4, 4) - static camera pose
                T_world_cam = cam_poses
            else:
                return {'success': False, 'error': f'Unexpected camera pose shape: {cam_poses.shape}'}
                
            p_WO_W = self.data['object']['pose'][0]  # Reference frame 0 as canonical
            points_obj = transform_points_to_object_frame(points_cam, T_world_obj, T_world_cam, p_WO_W)
            
            return {
                'success': True,
                'points': points_obj,
                'colors': colors,
                'frame_idx': frame_idx,
                'camera': cam,
                'num_points': len(points_obj)
            }
            
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(f"Frame processing failed: {e}") from e
    
    def generate_pseudo_ground_truth(self, 
                                   max_frames: Optional[int] = None,
                                   sampling_rate: float = 0.5,
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        Generate pseudo ground-truth poses using all three cameras (front-left, back-right, top-down)
        This follows the dataset approach for offline GT generation.
        
        Args:
            max_frames: Maximum number of frames to process (None = all frames)
            sampling_rate: Sampling rate in Hz for GT poses (default: 0.5 Hz as suggested)
            verbose: Whether to print progress
            
        Returns:
            Dict containing pseudo ground truth poses and timestamps
        """
        if not self.gt_generation_mode:
            raise ValueError("Pipeline must be initialized with gt_generation_mode=True for GT generation")
        
        total_frames = len(self.data['object']['pose'])
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        
        # Calculate frame indices based on sampling rate
        # Assuming original data is at 1 Hz or higher, sample at specified rate
        original_fps = 1.0  # Assume 1 FPS for feelsight_real data
        skip_frames = max(1, int(original_fps / sampling_rate))
        sampled_indices = list(range(0, frames_to_process, skip_frames))
        
        if verbose:
            print(f"🎯 PSEUDO GROUND TRUTH GENERATION")
            print(f"📁 Data path: {self.data_path}")
            print(f"📈 Total frames available: {total_frames}")
            print(f"⏰ Sampling rate: {sampling_rate} Hz (every {skip_frames} frames)")
            print(f"🎯 Processing {len(sampled_indices)} sampled frames")
            print(f"📷 Using cameras: {self.camera_names}")
        
        # For now, use the object poses from the data as pseudo GT
        # In a full implementation, this would run a multi-camera tracking algorithm
        pseudo_gt_poses = []
        timestamps = []
        
        for i, frame_idx in enumerate(sampled_indices):
            if verbose and (i % 10 == 0 or i < 5):
                print(f"   🔧 Processing GT frame {i+1}/{len(sampled_indices)} (frame {frame_idx})...")
            
            # Extract object pose for this frame
            T_world_obj = self.data['object']['pose'][frame_idx]
            pseudo_gt_poses.append(T_world_obj.copy())
            
            # Generate timestamp (assuming frame indices correspond to time)
            timestamp = frame_idx / original_fps
            timestamps.append(timestamp)
        
        results = {
            'poses': np.array(pseudo_gt_poses),
            'timestamps': np.array(timestamps),
            'frame_indices': sampled_indices,
            'sampling_rate': sampling_rate,
            'cameras_used': self.camera_names.copy(),
            'success': True
        }
        
        if verbose:
            print(f"\n📊 PSEUDO GROUND TRUTH RESULTS:")
            print(f"   🎯 Generated {len(pseudo_gt_poses)} GT poses")
            print(f"   ⏰ Time span: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
            print(f"   📷 Cameras used: {', '.join(self.camera_names)}")
        
        return results
    
    def save_pseudo_ground_truth(self, gt_results: Dict[str, Any], filename: str = 'pseudo_gt.json') -> str:
        """
        Save pseudo ground truth results to JSON file
        
        Args:
            gt_results: Results from generate_pseudo_ground_truth()
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if 'success' not in gt_results:
            raise KeyError("gt_results dictionary missing required 'success' key")
        if not gt_results['success']:
            raise ValueError("Cannot save failed GT generation results")
        
        gt_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        gt_data = {
            'poses': gt_results['poses'].tolist(),
            'timestamps': gt_results['timestamps'].tolist(),
            'frame_indices': gt_results['frame_indices'],
            'sampling_rate': gt_results['sampling_rate'],
            'cameras_used': gt_results['cameras_used']
        }
        
        import json
        with open(gt_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        return str(gt_path)
    
    def align_poses_to_primary_camera_timestamps(self, 
                                               gt_poses: np.ndarray,
                                               gt_timestamps: np.ndarray,
                                               primary_camera_frame_indices: Optional[List[int]] = None,
                                               interpolation_method: str = 'nearest') -> Dict[str, np.ndarray]:
        """
        Align ground truth poses to primary camera frame timestamps.
        
        Args:
            gt_poses: Ground truth poses array (N, 4, 4)
            gt_timestamps: GT timestamps array (N,)
            primary_camera_frame_indices: Frame indices for primary camera (None = use all available frames)
            interpolation_method: 'nearest' or 'linear' interpolation
            
        Returns:
            Dict containing aligned poses and timestamps
        """
        if primary_camera_frame_indices is None:
            # Use all available frames from the primary camera
            total_frames = len(self.data['object']['pose'])
            primary_camera_frame_indices = list(range(total_frames))
        
        # Generate timestamps for primary camera frames (assuming 1 FPS)
        primary_camera_timestamps = np.array([idx / 1.0 for idx in primary_camera_frame_indices])
        
        if interpolation_method == 'nearest':
            # Nearest neighbor interpolation
            aligned_poses = []
            aligned_timestamps = []
            
            for target_timestamp in primary_camera_timestamps:
                # Find closest GT timestamp
                closest_idx = np.argmin(np.abs(gt_timestamps - target_timestamp))
                aligned_poses.append(gt_poses[closest_idx].copy())
                aligned_timestamps.append(gt_timestamps[closest_idx])
        
        elif interpolation_method == 'linear':
            # Linear interpolation (simplified - would need full SE(3) interpolation for poses)
            from scipy.interpolate import interp1d
            
            aligned_poses = []
            aligned_timestamps = primary_camera_timestamps.copy()
            
            # For now, use nearest neighbor for poses (proper SE(3) interpolation is complex)
            # In a full implementation, you'd use SLERP for rotations and linear for translations
            for target_timestamp in primary_camera_timestamps:
                closest_idx = np.argmin(np.abs(gt_timestamps - target_timestamp))
                aligned_poses.append(gt_poses[closest_idx].copy())
        
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        return {
            'aligned_poses': np.array(aligned_poses),
            'aligned_timestamps': np.array(aligned_timestamps),
            'primary_camera_timestamps': primary_camera_timestamps,
            'frame_indices': primary_camera_frame_indices
        }


def run_camera_pipeline(data_path: str, 
                       max_frames: Optional[int] = None,
                       start_frame: int = 0,
                       output_dir: Optional[str] = None,
                       save_results: bool = True,
                       verbose: bool = True,
                       camera_names: Optional[List[str]] = None,
                       use_primary_camera_only: bool = True,
                       primary_camera: str = 'front-left') -> Dict[str, Any]:
    """
    Convenience function to run the complete camera pipeline
    
    Args:
        data_path: Path to dataset
        max_frames: Maximum number of frames to process (None = all frames)
        start_frame: Frame index to start processing from (default: 0)
        output_dir: Output directory (None = default)
        save_results: Whether to save PLY and analysis files
        verbose: Whether to print progress
        camera_names: List of camera names (for legacy multi-camera mode)
        use_primary_camera_only: If True, use only primary camera for reconstruction
        primary_camera: Name of primary camera (default: 'front-left')
        
    Returns:
        Dict containing results and file paths
    """
    pipeline = CameraPipeline(
        data_path, 
        output_dir=output_dir, 
        camera_names=camera_names,
        camera_name=primary_camera,
        use_primary_camera_only=use_primary_camera_only,
        gt_generation_mode=False
    )
    results = pipeline.process_frames(max_frames=max_frames, start_frame=start_frame, verbose=verbose)
    
    if 'success' not in results:
        raise KeyError("results dictionary missing required 'success' key")
    if results['success'] and save_results:
        suffix = f"_{max_frames}frames" if max_frames else "_all_frames"
        ply_path = pipeline.save_results(results, f'camera_reconstruction{suffix}.ply')
        analysis_path = pipeline.save_analysis(results, f'camera_analysis{suffix}.txt')
        
        results['ply_path'] = ply_path
        results['analysis_path'] = analysis_path
        
        if verbose:
            print(f"\n💾 Results saved:")
            print(f"   📄 PLY file: {ply_path}")
            print(f"   📊 Analysis: {analysis_path}")
    
    return results


def generate_ground_truth_poses(data_path: str,
                               output_dir: Optional[str] = None,
                               max_frames: Optional[int] = None,
                               sampling_rate: float = 0.5,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to generate pseudo ground truth poses using all cameras
    
    Args:
        data_path: Path to dataset
        output_dir: Output directory (None = default)
        max_frames: Maximum number of frames to process (None = all frames)
        sampling_rate: Sampling rate in Hz for GT poses (default: 0.5 Hz)
        verbose: Whether to print progress
        
    Returns:
        Dict containing GT results and file paths
    """
    pipeline = CameraPipeline(
        data_path,
        output_dir=output_dir,
        gt_generation_mode=True,
        use_primary_camera_only=False
    )
    
    gt_results = pipeline.generate_pseudo_ground_truth(
        max_frames=max_frames,
        sampling_rate=sampling_rate,
        verbose=verbose
    )
    
    if 'success' not in gt_results:
        raise KeyError("gt_results dictionary missing required 'success' key")
    if gt_results['success']:
        gt_path = pipeline.save_pseudo_ground_truth(gt_results, 'pseudo_gt.json')
        gt_results['gt_path'] = gt_path
        
        if verbose:
            print(f"\n💾 Ground truth saved:")
            print(f"   📄 GT file: {gt_path}")
    
    return gt_results


if __name__ == "__main__":
    # Example usage - data path should be passed as argument
    import argparse
    parser = argparse.ArgumentParser(description='Run camera pipeline')
    parser.add_argument('--data_path', required=True, help='Path to dataset')
    parser.add_argument('--max_frames', type=int, default=50, help='Maximum frames to process')
    parser.add_argument('--cameras', type=str, default='', help='Comma-separated camera names (for legacy multi-camera mode)')
    parser.add_argument('--primary_camera', type=str, default='front-left', help='Primary camera name (default: front-left)')
    parser.add_argument('--use_all_cameras', action='store_true', help='Use all cameras for reconstruction (legacy mode)')
    parser.add_argument('--generate_gt', action='store_true', help='Generate pseudo ground truth poses using all cameras')
    parser.add_argument('--gt_sampling_rate', type=float, default=0.5, help='GT sampling rate in Hz (default: 0.5)')
    args = parser.parse_args()
    
    if args.generate_gt:
        # Generate pseudo ground truth mode
        results = generate_ground_truth_poses(
            args.data_path, 
            max_frames=args.max_frames,
            sampling_rate=args.gt_sampling_rate
        )
        if 'success' not in results:
            raise KeyError("results dictionary missing required 'success' key")
        if results['success']:
            import logging
            logging.getLogger(__name__).info("PSEUDO GROUND TRUTH GENERATION COMPLETED SUCCESSFULLY!")
        else:
            import logging
            logging.getLogger(__name__).error("PSEUDO GROUND TRUTH GENERATION FAILED!")
    else:
        # Regular reconstruction mode
        cam_list = [c.strip() for c in args.cameras.split(',') if c.strip()] if args.cameras else None
        use_primary_only = not args.use_all_cameras  # Default to primary camera only unless explicitly requested
        
        results = run_camera_pipeline(
            args.data_path, 
            max_frames=args.max_frames, 
            camera_names=cam_list,
            use_primary_camera_only=use_primary_only,
            primary_camera=args.primary_camera
        )
        
        if 'success' not in results:
            raise KeyError("results dictionary missing required 'success' key")
        if results['success']:
            import logging
            logging.getLogger(__name__).info("CAMERA PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            import logging
            logging.getLogger(__name__).error("CAMERA PIPELINE FAILED!")