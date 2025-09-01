#!/usr/bin/env python3
"""
Evaluation Module
=================

This module provides evaluation metrics and utilities for reconstruction:
- ADD-S (Average Distance of Model Points - Symmetric)
- F-score evaluation
- Pose error computation
- Ground truth alignment and comparison
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class Evaluator:
    """Evaluator for reconstruction results"""
    
    def __init__(self, gt_mesh_path: Optional[str] = None, gt_poses_path: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            gt_mesh_path: Path to ground truth mesh (.obj file)
            gt_poses_path: Path to ground truth poses (.json file)
        """
        self.gt_mesh_path = Path(gt_mesh_path) if gt_mesh_path else None
        self.gt_poses_path = Path(gt_poses_path) if gt_poses_path else None
        
        # Load ground truth data if provided
        self.gt_mesh = None
        self.gt_poses = None
        
        if self.gt_mesh_path and self.gt_mesh_path.exists():
            self.gt_mesh = self._load_mesh(self.gt_mesh_path)
            
        if self.gt_poses_path and self.gt_poses_path.exists():
            self.gt_poses = self._load_poses(self.gt_poses_path)
    
    def _load_mesh(self, mesh_path: Path) -> Dict[str, np.ndarray]:
        """Load mesh from OBJ file"""
        try:
            import trimesh
            mesh = trimesh.load_mesh(str(mesh_path))
            return {
                'vertices': np.array(mesh.vertices),
                'faces': np.array(mesh.faces),
                'mesh_obj': mesh
            }
        except ImportError as e:
            raise RuntimeError("trimesh is required to load meshes") from e
    
    def _load_poses(self, poses_path: Path) -> Dict[str, Any]:
        """Load poses from JSON file"""
        with open(poses_path, 'r') as f:
            pose_data = json.load(f)
        
        # Convert poses back to numpy arrays
        poses = np.array(pose_data['poses'])
        timestamps = np.array(pose_data['timestamps'])
        
        return {
            'poses': poses,
            'timestamps': timestamps,
            'frame_indices': pose_data.get('frame_indices', []),
            'sampling_rate': pose_data.get('sampling_rate', 1.0),
            'cameras_used': pose_data.get('cameras_used', [])
        }
    
    def compute_add_s_error(self, 
                           predicted_mesh_vertices: np.ndarray,
                           gt_pose: np.ndarray,
                           predicted_pose: np.ndarray,
                           num_samples: int = 1000) -> float:
        """
        Compute ADD-S (Average Distance of Model Points - Symmetric)
        Pose error metric from https://arxiv.org/pdf/1711.00199.pdf
        
        Args:
            predicted_mesh_vertices: [N, 3] vertices of predicted mesh
            gt_pose: [4, 4] ground truth pose matrix
            predicted_pose: [4, 4] predicted pose matrix
            num_samples: number of points to sample for evaluation
        
        Returns:
            ADD-S error in meters
        """
        if self.gt_mesh is None:
            raise ValueError("Ground truth mesh not loaded")
            
        # Get ground truth vertices
        gt_vertices = self.gt_mesh['vertices']
        
        # Sample points if we have too many
        if len(gt_vertices) > num_samples:
            indices = np.random.choice(len(gt_vertices), num_samples, replace=False)
            gt_vertices = gt_vertices[indices]
            
        # Transform ground truth points to predicted pose
        gt_points_transformed = self._transform_points(gt_vertices, gt_pose)
        predicted_points_transformed = self._transform_points(predicted_mesh_vertices, predicted_pose)
        
        # Compute symmetric distance (both directions)
        from scipy.spatial.distance import cdist
        
        # Distance from GT to predicted
        dist_gt_to_pred = cdist(gt_points_transformed, predicted_points_transformed)
        min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)
        
        # Distance from predicted to GT
        dist_pred_to_gt = cdist(predicted_points_transformed, gt_points_transformed)
        min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)
        
        # Symmetric ADD-S: average of both directions
        add_s_error = (np.mean(min_dist_gt_to_pred) + np.mean(min_dist_pred_to_gt)) / 2.0
        
        return float(add_s_error)
    
    def compute_f_score(self,
                       predicted_mesh_vertices: np.ndarray,
                       gt_vertices: Optional[np.ndarray] = None,
                       num_mesh_samples: int = 30000,
                       thresholds: List[float] = None) -> Dict[str, List[float]]:
        """
        Compute F-score between ground truth and predicted mesh
        Implementation matching NeuralFeels evaluation
        
        Args:
            predicted_mesh_vertices: [N, 3] vertices of predicted mesh
            gt_vertices: [M, 3] ground truth vertices (uses loaded GT if None)
            num_mesh_samples: number of points to sample from mesh surface
            thresholds: distance thresholds in meters [default: [2e-2, 1e-2, 5e-3, 1e-3]]
            
        Returns:
            Dictionary with 'f_scores', 'precisions', 'recalls', 'distances'
        """
        if thresholds is None:
            thresholds = [2e-2, 1e-2, 5e-3, 1e-3]  # 20mm, 10mm, 5mm, 1mm
            
        # Use provided GT vertices or loaded GT mesh
        if gt_vertices is None:
            if self.gt_mesh is None:
                raise ValueError("No ground truth mesh loaded")
            gt_vertices = self.gt_mesh['vertices']
            
        # Sample points from predicted mesh surface if it's a mesh
        if hasattr(predicted_mesh_vertices, 'sample_points_uniformly'):
            # Open3D mesh
            pred_points = np.asarray(predicted_mesh_vertices.sample_points_uniformly(num_mesh_samples).points)
        elif len(predicted_mesh_vertices.shape) == 2 and predicted_mesh_vertices.shape[0] > num_mesh_samples:
            # Too many vertices, sample
            indices = np.random.choice(len(predicted_mesh_vertices), num_mesh_samples, replace=False)
            pred_points = predicted_mesh_vertices[indices]
        else:
            pred_points = predicted_mesh_vertices
            
        # Use KDTree for efficient nearest neighbor search
        from scipy.spatial import cKDTree as KDTree
        
        # Build KD trees
        gt_tree = KDTree(gt_vertices)
        pred_tree = KDTree(pred_points)
        
        # Distance from each predicted point to nearest GT point
        pred_to_gt_distances, _ = gt_tree.query(pred_points, p=2)
        
        # Distance from each GT point to nearest predicted point  
        gt_to_pred_distances, _ = pred_tree.query(gt_vertices, p=2)
        
        # Compute precision, recall, and F-score for each threshold
        f_scores, precisions, recalls = [], [], []
        
        for threshold in thresholds:
            # Precision: percentage of predicted points within threshold of GT
            precision = np.sum(pred_to_gt_distances < threshold) / len(pred_points)
            
            # Recall: percentage of GT points within threshold of prediction
            recall = np.sum(gt_to_pred_distances < threshold) / len(gt_vertices)
            
            # F-score: harmonic mean of precision and recall
            if precision + recall > 0:
                f_score = 2 * (precision * recall) / (precision + recall)
            else:
                f_score = 0.0
                
            f_scores.append(f_score)
            precisions.append(precision)
            recalls.append(recall)
            
        return {
            'f_scores': f_scores,
            'precisions': precisions, 
            'recalls': recalls,
            'thresholds': thresholds,
            'pred_to_gt_distances': pred_to_gt_distances,
            'gt_to_pred_distances': gt_to_pred_distances
        }
    
    def _transform_points(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Transform points by pose matrix
        
        Args:
            points: [N, 3] array of points
            pose: [4, 4] transformation matrix
            
        Returns:
            [N, 3] transformed points
        """
        # Convert to homogeneous coordinates
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        
        # Apply transformation
        points_transformed = (pose @ points_hom.T).T
        
        # Return to 3D coordinates
        return points_transformed[:, :3]
    
    def compute_pose_error(self,
                          predicted_poses: np.ndarray,
                          gt_poses: Optional[np.ndarray] = None,
                          frame_indices: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute pose estimation errors
        
        Args:
            predicted_poses: [N, 4, 4] predicted pose matrices
            gt_poses: [N, 4, 4] ground truth poses (uses loaded GT if None)
            frame_indices: Frame indices to evaluate (uses all if None)
            
        Returns:
            Dictionary with pose error metrics
        """
        # Use loaded GT poses if not provided
        if gt_poses is None:
            if self.gt_poses is None:
                raise ValueError("No ground truth poses loaded")
            gt_poses = self.gt_poses['poses']
            
        # Use specified frame indices or all frames
        if frame_indices is None:
            frame_indices = list(range(min(len(predicted_poses), len(gt_poses))))
            
        # Extract poses for specified frames
        pred_subset = predicted_poses[frame_indices]
        gt_subset = gt_poses[frame_indices]
        
        # Compute translation and rotation errors
        translation_errors = []
        rotation_errors = []
        
        for i in range(len(frame_indices)):
            pred_pose = pred_subset[i]
            gt_pose = gt_subset[i]
            
            # Translation error (Euclidean distance)
            trans_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
            translation_errors.append(trans_error)
            
            # Rotation error (angular distance)
            pred_rot = pred_pose[:3, :3]
            gt_rot = gt_pose[:3, :3]
            
            # Compute rotation error using trace
            cos_angle = (np.trace(pred_rot.T @ gt_rot) - 1) / 2
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp for numerical stability
            rot_error = np.arccos(cos_angle)
            rotation_errors.append(rot_error)
            
        return {
            'mean_translation_error': float(np.mean(translation_errors)),
            'mean_rotation_error': float(np.mean(rotation_errors)),
            'median_translation_error': float(np.median(translation_errors)),
            'median_rotation_error': float(np.median(rotation_errors)),
            'max_translation_error': float(np.max(translation_errors)),
            'max_rotation_error': float(np.max(rotation_errors)),
            'translation_errors': translation_errors,
            'rotation_errors': rotation_errors
        }
    
    def compute_timing_metrics(self, timing_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute comprehensive timing analysis metrics
        
        Args:
            timing_data: Dictionary with timing measurements
                       e.g., {'pose_opt': [t1, t2, ...], 'map_opt': [t1, t2, ...], 'render': [...]}
                       
        Returns:
            Dictionary with timing statistics
        """
        results = {}
        
        for stage_name, times in timing_data.items():
            if len(times) == 0:
                continue
                
            times_array = np.array(times)
            
            results[f'{stage_name}_mean_time'] = float(np.mean(times_array))
            results[f'{stage_name}_median_time'] = float(np.median(times_array))
            results[f'{stage_name}_std_time'] = float(np.std(times_array))
            results[f'{stage_name}_min_time'] = float(np.min(times_array))
            results[f'{stage_name}_max_time'] = float(np.max(times_array))
            results[f'{stage_name}_total_time'] = float(np.sum(times_array))
            results[f'{stage_name}_fps'] = float(1.0 / np.mean(times_array)) if np.mean(times_array) > 0 else 0.0
            
        # Overall statistics
        all_times = []
        for times in timing_data.values():
            all_times.extend(times)
            
        if len(all_times) > 0:
            results['total_pipeline_time'] = float(np.sum(all_times))
            results['mean_step_time'] = float(np.mean(all_times))
            results['overall_fps'] = float(1.0 / np.mean(all_times)) if np.mean(all_times) > 0 else 0.0
            
        return results
    
    def compute_memory_metrics(self, memory_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute memory usage statistics
        
        Args:
            memory_data: Dictionary with memory measurements in MB
                        e.g., {'gpu_memory': [m1, m2, ...], 'system_memory': [...]}
                        
        Returns:
            Dictionary with memory statistics
        """
        results = {}
        
        for memory_type, usage in memory_data.items():
            if len(usage) == 0:
                continue
                
            usage_array = np.array(usage)
            
            results[f'{memory_type}_mean'] = float(np.mean(usage_array))
            results[f'{memory_type}_median'] = float(np.median(usage_array))
            results[f'{memory_type}_std'] = float(np.std(usage_array))
            results[f'{memory_type}_min'] = float(np.min(usage_array))
            results[f'{memory_type}_max'] = float(np.max(usage_array))
            results[f'{memory_type}_peak'] = float(np.max(usage_array))
            
        return results
    
    def run_comprehensive_evaluation(self,
                                   predicted_mesh: Optional[np.ndarray] = None,
                                   predicted_poses: Optional[np.ndarray] = None,
                                   timing_data: Optional[Dict[str, List[float]]] = None,
                                   memory_data: Optional[Dict[str, List[float]]] = None,
                                   output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation matching NeuralFeels capabilities
        
        Args:
            predicted_mesh: Predicted mesh vertices or mesh object
            predicted_poses: [N, 4, 4] predicted poses
            timing_data: Dictionary with timing measurements
            memory_data: Dictionary with memory measurements
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results dictionary
        """
        results = {
            'timestamp': time.time(),
            'evaluation_config': {
                'gt_mesh_path': str(self.gt_mesh_path) if self.gt_mesh_path else None,
                'gt_poses_path': str(self.gt_poses_path) if self.gt_poses_path else None
            }
        }
        
        # Geometric evaluation
        if predicted_mesh is not None and self.gt_mesh is not None:
            print("Computing F-score metrics...")
            f_score_results = self.compute_f_score(predicted_mesh)
            results['f_score'] = f_score_results
            
            # ADD-S evaluation if we have poses
            if predicted_poses is not None and self.gt_poses is not None:
                print("Computing ADD-S error...")
                # Use first pose for ADD-S (can be extended)
                add_s_error = self.compute_add_s_error(
                    predicted_mesh,
                    self.gt_poses['poses'][0],
                    predicted_poses[0]
                )
                results['add_s_error'] = add_s_error
                
        # Pose evaluation
        if predicted_poses is not None and self.gt_poses is not None:
            print("Computing pose errors...")
            pose_results = self.compute_pose_error(predicted_poses)
            results['pose_errors'] = pose_results
            
        # Timing evaluation
        if timing_data is not None:
            print("Computing timing metrics...")
            timing_results = self.compute_timing_metrics(timing_data)
            results['timing'] = timing_results
            
        # Memory evaluation
        if memory_data is not None:
            print("Computing memory metrics...")
            memory_results = self.compute_memory_metrics(memory_data)
            results['memory'] = memory_results
            
        # Save results if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            results_file = output_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy types to native Python for JSON serialization
                json_results = self._convert_for_json(results)
                json.dump(json_results, f, indent=2)
                
            print(f"Evaluation results saved to {results_file}")
            
            # Generate evaluation plots
            self._generate_evaluation_plots(results, output_dir)
            
        return results
    
    def _convert_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def _generate_evaluation_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate evaluation plots and visualizations"""
        try:
            import matplotlib.pyplot as plt
            
            # F-score plot
            if 'f_score' in results:
                self._plot_f_score(results['f_score'], output_dir)
                
            # Timing plots
            if 'timing' in results:
                self._plot_timing_analysis(results['timing'], output_dir)
                
            # Memory plots
            if 'memory' in results:
                self._plot_memory_analysis(results['memory'], output_dir)
                
            # Pose error plots
            if 'pose_errors' in results:
                self._plot_pose_errors(results['pose_errors'], output_dir)
                
        except ImportError as e:
            raise RuntimeError("matplotlib is required for evaluation plots") from e
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def _plot_f_score(self, f_score_data: Dict, output_dir: Path):
        """Plot F-score results"""
        import matplotlib.pyplot as plt
        
        thresholds = f_score_data['thresholds']
        f_scores = f_score_data['f_scores']
        precisions = f_score_data['precisions']
        recalls = f_score_data['recalls']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # F-score vs threshold
        ax1.plot([t*1000 for t in thresholds], f_scores, 'b-o', label='F-score')
        ax1.set_xlabel('Threshold (mm)')
        ax1.set_ylabel('F-score')
        ax1.set_title('F-score vs Distance Threshold')
        ax1.grid(True)
        ax1.legend()
        
        # Precision-Recall
        ax2.plot([t*1000 for t in thresholds], precisions, 'g-o', label='Precision')
        ax2.plot([t*1000 for t in thresholds], recalls, 'r-o', label='Recall')
        ax2.set_xlabel('Threshold (mm)')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall vs Threshold')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'f_score_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_timing_analysis(self, timing_data: Dict, output_dir: Path):
        """Plot timing analysis"""
        import matplotlib.pyplot as plt
        
        # Extract stage names and mean times
        stages = []
        mean_times = []
        
        for key, value in timing_data.items():
            if key.endswith('_mean_time'):
                stage = key.replace('_mean_time', '')
                stages.append(stage)
                mean_times.append(value * 1000)  # Convert to ms
                
        if len(stages) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(stages, mean_times)
            ax.set_ylabel('Mean Time (ms)')
            ax.set_title('Performance Analysis by Stage')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, mean_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{time_val:.1f}ms', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(output_dir / 'timing_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_memory_analysis(self, memory_data: Dict, output_dir: Path):
        """Plot memory analysis"""
        import matplotlib.pyplot as plt
        
        memory_types = []
        peak_usage = []
        
        for key, value in memory_data.items():
            if key.endswith('_peak'):
                mem_type = key.replace('_peak', '')
                memory_types.append(mem_type)
                peak_usage.append(value)
                
        if len(memory_types) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(memory_types, peak_usage)
            ax.set_ylabel('Peak Memory Usage (MB)')
            ax.set_title('Memory Usage Analysis')
            
            # Add value labels
            for bar, usage in zip(bars, peak_usage):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{usage:.0f}MB', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(output_dir / 'memory_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_pose_errors(self, pose_data: Dict, output_dir: Path):
        """Plot pose error analysis"""
        import matplotlib.pyplot as plt
        
        if 'translation_errors' in pose_data and 'rotation_errors' in pose_data:
            trans_errors = np.array(pose_data['translation_errors']) * 1000  # Convert to mm
            rot_errors = np.array(pose_data['rotation_errors']) * 180 / np.pi  # Convert to degrees
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Translation errors over time
            ax1.plot(trans_errors, 'b-')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Translation Error (mm)')
            ax1.set_title('Translation Error Over Time')
            ax1.grid(True)
            
            # Rotation errors over time  
            ax2.plot(rot_errors, 'r-')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Rotation Error (degrees)')
            ax2.set_title('Rotation Error Over Time')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pose_errors.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def compute_add_s_error(self,
                           predicted_mesh_vertices: np.ndarray,
                           gt_pose: np.ndarray,
                           predicted_pose: np.ndarray,
                           threshold: float = 0.02) -> Dict[str, float]:
        """
        Compute ADD-S (Average Distance of Model Points - Symmetric) error
        
        Args:
            predicted_mesh_vertices: Predicted mesh vertices (N, 3)
            gt_pose: Ground truth pose (4, 4)
            predicted_pose: Predicted pose (4, 4)
            threshold: Threshold for success rate (default: 2cm)
            
        Returns:
            Dictionary with ADD-S metrics
        """
        if self.gt_mesh is None:
            raise ValueError("Ground truth mesh not loaded")
        
        gt_vertices = self.gt_mesh['vertices']
        
        # Transform ground truth vertices by GT pose
        gt_vertices_homo = np.hstack([gt_vertices, np.ones((len(gt_vertices), 1))])
        gt_transformed = (gt_pose @ gt_vertices_homo.T).T[:, :3]
        
        # Transform predicted vertices by predicted pose  
        pred_vertices_homo = np.hstack([predicted_mesh_vertices, np.ones((len(predicted_mesh_vertices), 1))])
        pred_transformed = (predicted_pose @ pred_vertices_homo.T).T[:, :3]
        
        # Compute ADD-S: for each predicted point, find closest GT point
        from scipy.spatial.distance import cdist
        distances = cdist(pred_transformed, gt_transformed)
        min_distances = np.min(distances, axis=1)
        
        add_s_error = np.mean(min_distances)
        success_rate = np.mean(min_distances < threshold)
        
        return {
            'add_s_error': float(add_s_error),
            'success_rate': float(success_rate),
            'threshold': threshold,
            'num_points': len(predicted_mesh_vertices)
        }
    
    def compute_f_score(self,
                       predicted_points: np.ndarray,
                       gt_points: np.ndarray,
                       threshold: float = 0.01) -> Dict[str, float]:
        """
        Compute F-score between predicted and ground truth point clouds
        
        Args:
            predicted_points: Predicted point cloud (N, 3)
            gt_points: Ground truth point cloud (M, 3)
            threshold: Distance threshold for precision/recall (default: 1cm)
            
        Returns:
            Dictionary with F-score metrics
        """
        from scipy.spatial.distance import cdist
        
        # Precision: fraction of predicted points within threshold of GT points
        pred_to_gt_dist = cdist(predicted_points, gt_points)
        pred_min_dist = np.min(pred_to_gt_dist, axis=1)
        precision = np.mean(pred_min_dist < threshold)
        
        # Recall: fraction of GT points within threshold of predicted points
        gt_to_pred_dist = cdist(gt_points, predicted_points)
        gt_min_dist = np.min(gt_to_pred_dist, axis=1)
        recall = np.mean(gt_min_dist < threshold)
        
        # F-score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
        
        return {
            'f_score': float(f_score),
            'precision': float(precision),
            'recall': float(recall),
            'threshold': threshold,
            'num_predicted': len(predicted_points),
            'num_gt': len(gt_points)
        }
    
    def compute_pose_error(self, 
                          gt_pose: np.ndarray, 
                          predicted_pose: np.ndarray) -> Dict[str, float]:
        """
        Compute pose error (translation and rotation)
        
        Args:
            gt_pose: Ground truth pose (4, 4)
            predicted_pose: Predicted pose (4, 4)
            
        Returns:
            Dictionary with pose errors
        """
        # Translation error
        gt_translation = gt_pose[:3, 3]
        pred_translation = predicted_pose[:3, 3]
        translation_error = np.linalg.norm(gt_translation - pred_translation)
        
        # Rotation error (angle between rotation matrices)
        gt_rotation = gt_pose[:3, :3]
        pred_rotation = predicted_pose[:3, :3]
        
        # Compute rotation error using trace of R_gt^T @ R_pred
        rotation_diff = gt_rotation.T @ pred_rotation
        trace_val = np.trace(rotation_diff)
        # Clamp to valid range for arccos
        trace_val = np.clip(trace_val, -1, 3)  # trace can be up to 3 for 3x3 rotation matrix
        angle_error = np.arccos((trace_val - 1) / 2)
        angle_error_deg = np.degrees(angle_error)
        
        return {
            'translation_error': float(translation_error),
            'rotation_error_rad': float(angle_error),
            'rotation_error_deg': float(angle_error_deg)
        }
    
    def evaluate_reconstruction(self,
                              predicted_points: np.ndarray,
                              predicted_poses: np.ndarray,
                              frame_indices: Optional[List[int]] = None,
                              f_score_threshold: float = 0.01,
                              add_s_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Evaluate complete reconstruction against ground truth
        
        Args:
            predicted_points: Predicted point cloud (N, 3)
            predicted_poses: Predicted poses per frame (T, 4, 4)
            frame_indices: Frame indices corresponding to poses
            f_score_threshold: Threshold for F-score computation
            add_s_threshold: Threshold for ADD-S computation
            
        Returns:
            Complete evaluation results
        """
        if self.gt_poses is None:
            raise ValueError("Ground truth poses not loaded")
        
        results = {
            'f_score_results': {},
            'pose_errors': [],
            'add_s_results': {},
            'summary': {}
        }
        
        # F-score evaluation (if GT mesh available)
        if self.gt_mesh is not None:
            gt_vertices = self.gt_mesh['vertices']
            f_score_results = self.compute_f_score(
                predicted_points, gt_vertices, f_score_threshold
            )
            results['f_score_results'] = f_score_results
        
        # Pose evaluation
        gt_poses = self.gt_poses['poses']
        if frame_indices is not None:
            # Use only poses for specified frames
            eval_gt_poses = gt_poses[frame_indices] if len(gt_poses) > max(frame_indices) else gt_poses
        else:
            eval_gt_poses = gt_poses[:len(predicted_poses)]
        
        pose_errors = []
        for i in range(min(len(predicted_poses), len(eval_gt_poses))):
            pose_error = self.compute_pose_error(eval_gt_poses[i], predicted_poses[i])
            pose_error['frame_index'] = frame_indices[i] if frame_indices else i
            pose_errors.append(pose_error)
        
        results['pose_errors'] = pose_errors
        
        # Summary statistics
        if pose_errors:
            translation_errors = [pe['translation_error'] for pe in pose_errors]
            rotation_errors = [pe['rotation_error_deg'] for pe in pose_errors]
            
            results['summary'] = {
                'mean_translation_error': float(np.mean(translation_errors)),
                'std_translation_error': float(np.std(translation_errors)),
                'mean_rotation_error_deg': float(np.mean(rotation_errors)),
                'std_rotation_error_deg': float(np.std(rotation_errors)),
                'num_poses_evaluated': len(pose_errors)
            }
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Evaluation results saved to: {output_path}")


def load_ground_truth_mesh(dataset_name: str, object_name: str, data_root: str = "data") -> str:
    """
    Load ground truth mesh path for a given object
    
    Args:
        dataset_name: Dataset name (e.g., 'feelsight_real')
        object_name: Object name (e.g., 'bell_pepper')
        data_root: Data root directory
        
    Returns:
        Path to ground truth mesh file
    """
    # Look for mesh in assets/gt_models directory
    potential_paths = [
        f"{data_root}/assets/gt_models/dextouch/{object_name}.obj",
        f"{data_root}/assets/gt_models/dextouch/{object_name}.urdf",
        f"{data_root}/assets/gt_models/ycb/{object_name}.urdf"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Ground truth mesh not found for {object_name}. Searched: {potential_paths}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate reconstruction')
    parser.add_argument('--gt_mesh', required=True, help='Path to ground truth mesh')
    parser.add_argument('--gt_poses', required=True, help='Path to ground truth poses JSON')
    parser.add_argument('--predicted_points', required=True, help='Path to predicted point cloud (PLY)')
    parser.add_argument('--predicted_poses', help='Path to predicted poses (optional)')
    parser.add_argument('--output', default='evaluation_results.json', help='Output path for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.gt_mesh, args.gt_poses)
    
    # Load predicted points (would need to implement PLY loading)
    print("Evaluator initialized. Implement PLY loading and pose loading as needed.")