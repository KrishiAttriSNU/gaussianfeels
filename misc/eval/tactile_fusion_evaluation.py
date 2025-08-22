"""
Comprehensive evaluation suite following NeuralFeels approach.
Implements F-score, chamfer distance, ADD-S pose metrics, and more.
"""

import numpy as np
import torch
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import trimesh
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class NeuralFeelsEvaluationSuite:
    """
    Comprehensive evaluation suite following NeuralFeels approach.
    Implements established computer vision metrics like NeuralFeels.
    """
    
    def __init__(self, voxel_size: float = 0.01):
        """Initialize NeuralFeels-style evaluation suite"""
        self.voxel_size = voxel_size
        self.metrics_computed = {}
        
        logger.info("Initialized NeuralFeels-style comprehensive evaluation suite")
    
    def compute_f_score(self, 
                       predicted_points: np.ndarray,
                       ground_truth_points: np.ndarray,
                       threshold: float = 0.01) -> Dict[str, float]:
        """
        Compute F-score metrics (NeuralFeels approach).
        
        Args:
            predicted_points: Predicted 3D points [N, 3]
            ground_truth_points: Ground truth 3D points [M, 3]
            threshold: Distance threshold for matching
            
        Returns:
            Dictionary with F-score metrics
        """
        if len(predicted_points) == 0 or len(ground_truth_points) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0}
        
        # Compute distances from predicted to ground truth
        pred_to_gt_distances = []
        for pred_point in predicted_points:
            distances = np.linalg.norm(ground_truth_points - pred_point, axis=1)
            pred_to_gt_distances.append(np.min(distances))
        pred_to_gt_distances = np.array(pred_to_gt_distances)
        
        # Compute distances from ground truth to predicted
        gt_to_pred_distances = []
        for gt_point in ground_truth_points:
            distances = np.linalg.norm(predicted_points - gt_point, axis=1)
            gt_to_pred_distances.append(np.min(distances))
        gt_to_pred_distances = np.array(gt_to_pred_distances)
        
        # Compute precision and recall
        precision = np.sum(pred_to_gt_distances < threshold) / len(pred_to_gt_distances)
        recall = np.sum(gt_to_pred_distances < threshold) / len(gt_to_pred_distances)
        
        # Compute F-score
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'threshold': threshold
        }
        
        logger.info(f"F-score computed: P={precision:.3f}, R={recall:.3f}, F={f_score:.3f}")
        return results
    
    def compute_chamfer_distance(self,
                               predicted_points: np.ndarray,
                               ground_truth_points: np.ndarray) -> Dict[str, float]:
        """
        Compute Chamfer distance (NeuralFeels approach).
        
        Args:
            predicted_points: Predicted 3D points [N, 3]
            ground_truth_points: Ground truth 3D points [M, 3]
            
        Returns:
            Dictionary with Chamfer distance metrics
        """
        if len(predicted_points) == 0 or len(ground_truth_points) == 0:
            return {'chamfer_distance': float('inf'), 'forward_distance': float('inf'), 'backward_distance': float('inf')}
        
        # Forward distance: predicted to ground truth
        forward_distances = []
        for pred_point in predicted_points:
            distances = np.linalg.norm(ground_truth_points - pred_point, axis=1)
            forward_distances.append(np.min(distances))
        forward_distance = np.mean(forward_distances)
        
        # Backward distance: ground truth to predicted
        backward_distances = []
        for gt_point in ground_truth_points:
            distances = np.linalg.norm(predicted_points - gt_point, axis=1)
            backward_distances.append(np.min(distances))
        backward_distance = np.mean(backward_distances)
        
        # Chamfer distance is the sum of both directions
        chamfer_distance = forward_distance + backward_distance
        
        results = {
            'chamfer_distance': chamfer_distance,
            'forward_distance': forward_distance,
            'backward_distance': backward_distance
        }
        
        logger.info(f"Chamfer distance computed: {chamfer_distance:.6f}")
        return results
    
    def compute_add_s_pose_metric(self,
                                predicted_pose: np.ndarray,
                                ground_truth_pose: np.ndarray,
                                object_points: np.ndarray) -> Dict[str, float]:
        """
        Compute ADD-S pose metric (NeuralFeels approach).
        
        Args:
            predicted_pose: Predicted pose matrix [4, 4]
            ground_truth_pose: Ground truth pose matrix [4, 4]
            object_points: Object model points [N, 3]
            
        Returns:
            Dictionary with ADD-S metrics
        """
        if len(object_points) == 0:
            return {'add_s_distance': float('inf'), 'add_s_score': 0.0}
        
        # Transform object points with both poses
        ones = np.ones((len(object_points), 1))
        object_points_homo = np.hstack([object_points, ones])
        
        pred_transformed = (predicted_pose @ object_points_homo.T).T[:, :3]
        gt_transformed = (ground_truth_pose @ object_points_homo.T).T[:, :3]
        
        # Compute ADD-S distance (symmetric)
        distances = []
        for pred_point in pred_transformed:
            dists_to_gt = np.linalg.norm(gt_transformed - pred_point, axis=1)
            distances.append(np.min(dists_to_gt))
        
        add_s_distance = np.mean(distances)
        
        # ADD-S score (percentage of points within threshold)
        threshold = 0.1 * np.max(np.linalg.norm(object_points, axis=1))  # 10% of object diameter
        add_s_score = np.sum(np.array(distances) < threshold) / len(distances)
        
        results = {
            'add_s_distance': add_s_distance,
            'add_s_score': add_s_score,
            'threshold': threshold
        }
        
        logger.info(f"ADD-S metric computed: distance={add_s_distance:.6f}, score={add_s_score:.3f}")
        return results
    
    def compute_accuracy_precision_metrics(self,
                                         predicted_labels: np.ndarray,
                                         ground_truth_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute accuracy and precision metrics (NeuralFeels approach).
        
        Args:
            predicted_labels: Predicted binary labels
            ground_truth_labels: Ground truth binary labels
            
        Returns:
            Dictionary with accuracy and precision metrics
        """
        if len(predicted_labels) == 0 or len(ground_truth_labels) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Ensure same length
        min_len = min(len(predicted_labels), len(ground_truth_labels))
        pred_labels = predicted_labels[:min_len]
        gt_labels = ground_truth_labels[:min_len]
        
        # Compute metrics
        accuracy = np.mean(pred_labels == gt_labels)
        
        # Handle case where all predictions are the same class
        if len(np.unique(pred_labels)) == 1 or len(np.unique(gt_labels)) == 1:
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        else:
            precision = precision_score(gt_labels, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(gt_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(gt_labels, pred_labels, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Accuracy/Precision computed: A={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}")
        return results
    
    def compute_reconstruction_quality_metrics(self,
                                             predicted_mesh: trimesh.Trimesh,
                                             ground_truth_mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics (NeuralFeels approach).
        
        Args:
            predicted_mesh: Predicted mesh
            ground_truth_mesh: Ground truth mesh
            
        Returns:
            Dictionary with reconstruction quality metrics
        """
        try:
            # Sample points from meshes
            pred_points = predicted_mesh.sample(10000)
            gt_points = ground_truth_mesh.sample(10000)
            
            # Compute basic metrics
            f_score_results = self.compute_f_score(pred_points, gt_points)
            chamfer_results = self.compute_chamfer_distance(pred_points, gt_points)
            
            # Mesh-specific metrics
            volume_error = abs(predicted_mesh.volume - ground_truth_mesh.volume) / ground_truth_mesh.volume
            surface_area_error = abs(predicted_mesh.area - ground_truth_mesh.area) / ground_truth_mesh.area
            
            results = {
                **f_score_results,
                **chamfer_results,
                'volume_error': volume_error,
                'surface_area_error': surface_area_error,
                'mesh_quality_score': 1.0 / (1.0 + chamfer_results['chamfer_distance'])
            }
            
            logger.info(f"Reconstruction quality computed: F={f_score_results['f_score']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Reconstruction quality computation failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_evaluation(self,
                                   prediction_data: Dict[str, Any],
                                   ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete comprehensive evaluation (NeuralFeels approach).
        
        Args:
            prediction_data: Dictionary with prediction results
            ground_truth_data: Dictionary with ground truth data
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'evaluation_type': 'neuralfeels_comprehensive',
            'metrics': {},
            'summary': {}
        }
        
        # Reconstruction quality metrics
        if 'points' in prediction_data and 'points' in ground_truth_data:
            f_score = self.compute_f_score(
                prediction_data['points'], ground_truth_data['points']
            )
            chamfer = self.compute_chamfer_distance(
                prediction_data['points'], ground_truth_data['points']
            )
            results['metrics']['f_score'] = f_score
            results['metrics']['chamfer_distance'] = chamfer
        
        # Pose accuracy metrics
        if 'pose' in prediction_data and 'pose' in ground_truth_data:
            if 'object_points' in ground_truth_data:
                add_s = self.compute_add_s_pose_metric(
                    prediction_data['pose'],
                    ground_truth_data['pose'],
                    ground_truth_data['object_points']
                )
                results['metrics']['add_s_pose'] = add_s
        
        # Classification metrics
        if 'labels' in prediction_data and 'labels' in ground_truth_data:
            accuracy_precision = self.compute_accuracy_precision_metrics(
                prediction_data['labels'], ground_truth_data['labels']
            )
            results['metrics']['accuracy_precision'] = accuracy_precision
        
        # Mesh quality metrics
        if 'mesh' in prediction_data and 'mesh' in ground_truth_data:
            mesh_quality = self.compute_reconstruction_quality_metrics(
                prediction_data['mesh'], ground_truth_data['mesh']
            )
            results['metrics']['reconstruction_quality'] = mesh_quality
        
        # Summary statistics
        all_scores = []
        for metric_group in results['metrics'].values():
            if isinstance(metric_group, dict):
                for key, value in metric_group.items():
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        if 'score' in key or 'f_score' in key or 'accuracy' in key or 'precision' in key:
                            all_scores.append(value)
        
        if all_scores:
            results['summary'] = {
                'overall_score': np.mean(all_scores),
                'num_metrics_computed': len(results['metrics']),
                'evaluation_complete': True
            }
        else:
            results['summary'] = {
                'overall_score': 0.0,
                'num_metrics_computed': len(results['metrics']),
                'evaluation_complete': False
            }
        
        logger.info(f"Comprehensive evaluation completed: {len(results['metrics'])} metric groups")
        return results


class OfflineEvaluationRunner:
    """
    Offline evaluation runner following NeuralFeels approach.
    """
    
    def __init__(self, results_dir: str = "evaluation_results"):
        """Initialize offline evaluation runner"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.evaluation_suite = NeuralFeelsEvaluationSuite()
        
        logger.info(f"Initialized offline evaluation runner (results: {self.results_dir})")
    
    def run_batch_evaluation(self, 
                           experiment_results: List[Dict[str, Any]],
                           ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run batch evaluation after training completion (NeuralFeels approach).
        
        Args:
            experiment_results: List of experiment results
            ground_truth_data: Ground truth data for comparison
            
        Returns:
            Batch evaluation results
        """
        batch_results = {
            'batch_evaluation': True,
            'num_experiments': len(experiment_results),
            'individual_results': [],
            'aggregate_results': {}
        }
        
        # Evaluate each experiment
        for i, experiment_result in enumerate(experiment_results):
            eval_result = self.evaluation_suite.run_comprehensive_evaluation(
                experiment_result, ground_truth_data
            )
            eval_result['experiment_id'] = i
            batch_results['individual_results'].append(eval_result)
        
        # Aggregate results
        if batch_results['individual_results']:
            all_scores = [r['summary']['overall_score'] for r in batch_results['individual_results']]
            batch_results['aggregate_results'] = {
                'mean_score': np.mean(all_scores),
                'std_score': np.std(all_scores),
                'min_score': np.min(all_scores),
                'max_score': np.max(all_scores),
                'median_score': np.median(all_scores)
            }
        
        logger.info(f"Batch evaluation completed for {len(experiment_results)} experiments")
        return batch_results
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        import json
        
        output_path = self.results_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Test the evaluation suite
    print("🧪 Testing NeuralFeels-style Evaluation Suite")
    print("=" * 50)
    
    # Create test data
    pred_points = np.random.randn(1000, 3) * 0.1
    gt_points = np.random.randn(1200, 3) * 0.1
    
    # Initialize evaluation suite
    eval_suite = NeuralFeelsEvaluationSuite()
    
    # Test F-score computation
    f_score_results = eval_suite.compute_f_score(pred_points, gt_points)
    print(f"F-score: {f_score_results}")
    
    # Test Chamfer distance
    chamfer_results = eval_suite.compute_chamfer_distance(pred_points, gt_points)
    print(f"Chamfer distance: {chamfer_results}")
    
    print("\n✅ NeuralFeels evaluation suite test completed!")