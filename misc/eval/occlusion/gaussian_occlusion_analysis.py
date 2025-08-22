"""
Occlusion analysis for Gaussian splatting evaluation

This module provides comprehensive occlusion analysis capabilities adapted from
neuralfeels for gaussianfeels, focusing on how vision occlusion affects pose
estimation and reconstruction quality in Gaussian splatting systems.
"""

import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.transform import Rotation as R
import logging

from .feelsight_init import get_init_pose, get_available_objects, get_available_viewpoints

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OcclusionMetrics:
    """Metrics for occlusion analysis"""
    occlusion_level: float  # Percentage of object occluded (0.0 to 1.0)
    pose_error_translation: float  # Translation error in meters
    pose_error_rotation: float  # Rotation error in degrees
    reconstruction_fscore: float  # F-score for surface reconstruction
    reconstruction_chamfer: float  # Chamfer distance for reconstruction quality
    gaussian_count: int  # Number of Gaussians in the reconstruction
    convergence_iterations: int  # Iterations to convergence
    processing_time: float  # Processing time in seconds
    timestamp: float  # Unix timestamp
    confidence_score: float  # Reconstruction confidence


@dataclass
class ViewpointAnalysis:
    """Analysis results for a specific viewpoint"""
    viewpoint: str
    object_name: str
    occlusion_metrics: List[OcclusionMetrics]
    mean_pose_error: float
    std_pose_error: float
    correlation_occlusion_error: float  # Pearson correlation coefficient
    best_fscore: float
    worst_fscore: float


@dataclass
class OcclusionAnalysisResults:
    """Complete occlusion analysis results"""
    object_name: str
    viewpoint_analyses: List[ViewpointAnalysis]
    overall_correlation: float
    statistical_significance: float  # p-value
    recommendations: List[str]
    analysis_timestamp: float


class GaussianOcclusionAnalyzer:
    """
    Analyzer for studying occlusion effects on Gaussian splatting reconstruction
    """
    
    def __init__(self, 
                 results_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 fscore_thresholds: List[float] = [0.05, 0.02, 0.01, 0.005],
                 pose_error_threshold: float = 0.1):
        """
        Initialize occlusion analyzer.
        
        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory for saving analysis outputs
            fscore_thresholds: F-score distance thresholds in meters
            pose_error_threshold: Pose error threshold for success classification
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.fscore_thresholds = fscore_thresholds
        self.pose_error_threshold = pose_error_threshold
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def compute_pose_error(self, 
                          pred_pose: np.ndarray, 
                          gt_pose: np.ndarray) -> Tuple[float, float]:
        """
        Compute pose error between predicted and ground truth poses.
        
        Args:
            pred_pose: Predicted 4x4 transformation matrix
            gt_pose: Ground truth 4x4 transformation matrix
            
        Returns:
            Tuple of (translation_error, rotation_error_degrees)
        """
        # Translation error (Euclidean distance)
        trans_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
        
        # Rotation error (angle between rotations)
        R_pred = pred_pose[:3, :3]
        R_gt = gt_pose[:3, :3]
        
        # Compute relative rotation
        R_rel = R_pred @ R_gt.T
        
        # Convert to angle-axis representation
        r_rel = R.from_matrix(R_rel)
        angle_error = np.abs(r_rel.as_rotvec())
        rot_error_rad = np.linalg.norm(angle_error)
        
        # Convert to degrees
        rot_error_deg = np.degrees(rot_error_rad)
        
        return trans_error, rot_error_deg
    
    def estimate_occlusion_level(self, 
                                depth_map: np.ndarray,
                                tactile_contact_mask: Optional[np.ndarray] = None) -> float:
        """
        Estimate occlusion level from depth map and tactile contact information.
        
        Args:
            depth_map: Depth map from camera (H x W)
            tactile_contact_mask: Binary mask indicating tactile contact regions
            
        Returns:
            Estimated occlusion level (0.0 to 1.0)
        """
        if depth_map is None or depth_map.size == 0:
            return 1.0  # Full occlusion if no depth data
        
        # Valid depth pixels (non-zero, non-inf, non-nan)
        valid_depth = np.isfinite(depth_map) & (depth_map > 0)
        total_pixels = depth_map.size
        valid_pixels = np.sum(valid_depth)
        
        # Base occlusion from depth availability
        base_occlusion = 1.0 - (valid_pixels / total_pixels)
        
        # Adjust based on tactile contact if available
        if tactile_contact_mask is not None:
            # Tactile contact provides additional information where vision fails
            tactile_compensation = np.sum(tactile_contact_mask) / tactile_contact_mask.size
            # Reduce apparent occlusion based on tactile coverage
            adjusted_occlusion = base_occlusion * (1.0 - 0.3 * tactile_compensation)
            return np.clip(adjusted_occlusion, 0.0, 1.0)
        
        return np.clip(base_occlusion, 0.0, 1.0)
    
    def load_experiment_results(self, 
                               experiment_path: Path) -> Dict[str, Any]:
        """
        Load experiment results from pickle file.
        
        Args:
            experiment_path: Path to experiment results file
            
        Returns:
            Dictionary containing experiment data
        """
        try:
            with open(experiment_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except Exception as e:
            logger.error(f"Failed to load experiment results from {experiment_path}: {e}")
            return {}
    
    def analyze_single_experiment(self, 
                                 experiment_data: Dict[str, Any],
                                 object_name: str,
                                 viewpoint: str) -> Optional[OcclusionMetrics]:
        """
        Analyze a single experiment for occlusion metrics.
        
        Args:
            experiment_data: Loaded experiment results
            object_name: Name of the object
            viewpoint: Camera viewpoint identifier
            
        Returns:
            OcclusionMetrics object or None if analysis fails
        """
        try:
            # Extract required data
            if 'pose_trajectory' not in experiment_data:
                logger.warning("No pose trajectory found in experiment data")
                return None
            
            # Get ground truth initialization pose
            gt_pose = get_init_pose(object_name, viewpoint)
            
            # Get final predicted pose (last in trajectory)
            pose_traj = experiment_data['pose_trajectory']
            if len(pose_traj) == 0:
                logger.warning("Empty pose trajectory")
                return None
                
            pred_pose = pose_traj[-1]['pose']  # Final pose
            
            # Compute pose errors
            trans_error, rot_error = self.compute_pose_error(pred_pose, gt_pose)
            
            # Estimate occlusion level
            depth_map = experiment_data.get('depth_maps', [])
            tactile_mask = experiment_data.get('tactile_contact_masks', None)
            
            if len(depth_map) > 0:
                # Use the first depth map for occlusion estimation
                occlusion_level = self.estimate_occlusion_level(depth_map[0], tactile_mask)
            else:
                occlusion_level = 0.5  # Default moderate occlusion
            
            # Extract reconstruction metrics
            reconstruction_data = experiment_data.get('reconstruction_metrics', {})
            fscore = reconstruction_data.get('fscore', 0.0)
            chamfer = reconstruction_data.get('chamfer_distance', float('inf'))
            
            # Extract Gaussian-specific metrics
            gaussian_count = experiment_data.get('gaussian_count', 0)
            convergence_iter = experiment_data.get('convergence_iterations', 0)
            processing_time = experiment_data.get('processing_time', 0.0)
            confidence = experiment_data.get('confidence_score', 0.0)
            
            return OcclusionMetrics(
                occlusion_level=occlusion_level,
                pose_error_translation=trans_error,
                pose_error_rotation=rot_error,
                reconstruction_fscore=fscore,
                reconstruction_chamfer=chamfer,
                gaussian_count=gaussian_count,
                convergence_iterations=convergence_iter,
                processing_time=processing_time,
                timestamp=experiment_data.get('timestamp', 0.0),
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing experiment for {object_name}:{viewpoint}: {e}")
            return None
    
    def analyze_viewpoint(self, 
                         object_name: str, 
                         viewpoint: str) -> Optional[ViewpointAnalysis]:
        """
        Analyze all experiments for a specific object and viewpoint.
        
        Args:
            object_name: Name of the object
            viewpoint: Camera viewpoint identifier
            
        Returns:
            ViewpointAnalysis object or None if analysis fails
        """
        # Find all experiment files for this object/viewpoint combination
        experiment_pattern = f"{object_name}_{viewpoint}_*.pkl"
        experiment_files = list(self.results_dir.glob(f"**/{experiment_pattern}"))
        
        if not experiment_files:
            logger.warning(f"No experiment files found for {object_name}:{viewpoint}")
            return None
        
        logger.info(f"Found {len(experiment_files)} experiments for {object_name}:{viewpoint}")
        
        metrics_list = []
        for exp_file in experiment_files:
            experiment_data = self.load_experiment_results(exp_file)
            if experiment_data:
                metrics = self.analyze_single_experiment(experiment_data, object_name, viewpoint)
                if metrics:
                    metrics_list.append(metrics)
        
        if not metrics_list:
            logger.warning(f"No valid metrics found for {object_name}:{viewpoint}")
            return None
        
        # Compute viewpoint-level statistics
        pose_errors = [m.pose_error_translation for m in metrics_list]
        occlusion_levels = [m.occlusion_level for m in metrics_list]
        fscores = [m.reconstruction_fscore for m in metrics_list]
        
        mean_pose_error = np.mean(pose_errors)
        std_pose_error = np.std(pose_errors)
        
        # Compute correlation between occlusion and pose error
        if len(pose_errors) > 2:
            correlation, p_value = stats.pearsonr(occlusion_levels, pose_errors)
        else:
            correlation, p_value = 0.0, 1.0
        
        return ViewpointAnalysis(
            viewpoint=viewpoint,
            object_name=object_name,
            occlusion_metrics=metrics_list,
            mean_pose_error=mean_pose_error,
            std_pose_error=std_pose_error,
            correlation_occlusion_error=correlation,
            best_fscore=max(fscores) if fscores else 0.0,
            worst_fscore=min(fscores) if fscores else 0.0
        )
    
    def analyze_object(self, object_name: str) -> OcclusionAnalysisResults:
        """
        Perform complete occlusion analysis for an object across all viewpoints.
        
        Args:
            object_name: Name of the object to analyze
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting occlusion analysis for object: {object_name}")
        
        viewpoints = get_available_viewpoints(object_name)
        viewpoint_analyses = []
        
        for viewpoint in viewpoints:
            analysis = self.analyze_viewpoint(object_name, viewpoint)
            if analysis:
                viewpoint_analyses.append(analysis)
        
        if not viewpoint_analyses:
            logger.error(f"No valid viewpoint analyses for {object_name}")
            # Return empty results
            return OcclusionAnalysisResults(
                object_name=object_name,
                viewpoint_analyses=[],
                overall_correlation=0.0,
                statistical_significance=1.0,
                recommendations=["No valid data found for analysis"],
                analysis_timestamp=np.time.time()
            )
        
        # Compute overall statistics
        all_pose_errors = []
        all_occlusion_levels = []
        
        for vp_analysis in viewpoint_analyses:
            for metrics in vp_analysis.occlusion_metrics:
                all_pose_errors.append(metrics.pose_error_translation)
                all_occlusion_levels.append(metrics.occlusion_level)
        
        # Overall correlation
        if len(all_pose_errors) > 2:
            overall_corr, overall_p = stats.pearsonr(all_occlusion_levels, all_pose_errors)
        else:
            overall_corr, overall_p = 0.0, 1.0
        
        # Generate recommendations
        recommendations = self.generate_recommendations(viewpoint_analyses, overall_corr, overall_p)
        
        return OcclusionAnalysisResults(
            object_name=object_name,
            viewpoint_analyses=viewpoint_analyses,
            overall_correlation=overall_corr,
            statistical_significance=overall_p,
            recommendations=recommendations,
            analysis_timestamp=np.time.time()
        )
    
    def generate_recommendations(self, 
                               viewpoint_analyses: List[ViewpointAnalysis],
                               overall_correlation: float,
                               p_value: float) -> List[str]:
        """
        Generate analysis recommendations based on results.
        
        Args:
            viewpoint_analyses: List of viewpoint analysis results
            overall_correlation: Overall correlation coefficient
            p_value: Statistical significance
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Correlation analysis
        if p_value < 0.05:
            if overall_correlation > 0.5:
                recommendations.append(
                    "Strong positive correlation between occlusion and pose error detected. "
                    "Consider improving tactile sensing or multi-view fusion."
                )
            elif overall_correlation < -0.5:
                recommendations.append(
                    "Interesting negative correlation found - system may be more robust to occlusion than expected."
                )
        else:
            recommendations.append(
                "No statistically significant correlation between occlusion and pose error found. "
                "Consider increasing sample size or checking data quality."
            )
        
        # Viewpoint-specific recommendations
        if viewpoint_analyses:
            best_viewpoint = min(viewpoint_analyses, key=lambda x: x.mean_pose_error)
            worst_viewpoint = max(viewpoint_analyses, key=lambda x: x.mean_pose_error)
            
            recommendations.append(
                f"Best performing viewpoint: {best_viewpoint.viewpoint} "
                f"(mean error: {best_viewpoint.mean_pose_error:.3f}m)"
            )
            recommendations.append(
                f"Worst performing viewpoint: {worst_viewpoint.viewpoint} "
                f"(mean error: {worst_viewpoint.mean_pose_error:.3f}m)"
            )
        
        # F-score analysis
        all_fscores = []
        for vp in viewpoint_analyses:
            for metrics in vp.occlusion_metrics:
                all_fscores.append(metrics.reconstruction_fscore)
        
        if all_fscores:
            mean_fscore = np.mean(all_fscores)
            if mean_fscore < 0.5:
                recommendations.append(
                    "Low reconstruction F-scores detected. Consider tuning Gaussian splatting parameters "
                    "or improving surface reconstruction pipeline."
                )
            elif mean_fscore > 0.8:
                recommendations.append(
                    "Excellent reconstruction quality achieved. Current parameters are well-tuned."
                )
        
        return recommendations
    
    def save_results(self, results: OcclusionAnalysisResults, 
                    filename_prefix: str = "occlusion_analysis") -> Path:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results to save
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to saved file
        """
        timestamp = int(results.analysis_timestamp)
        filename = f"{filename_prefix}_{results.object_name}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        results_dict = asdict(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
        return output_path


def main():
    """Main function for standalone occlusion analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gaussian Splatting Occlusion Analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory for saving analysis outputs")
    parser.add_argument("--object_name", type=str, default="010_potted_meat_can",
                       help="Object name to analyze")
    parser.add_argument("--fscore_thresholds", nargs="+", type=float,
                       default=[0.05, 0.02, 0.01, 0.005],
                       help="F-score distance thresholds")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GaussianOcclusionAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        fscore_thresholds=args.fscore_thresholds
    )
    
    # Run analysis
    results = analyzer.analyze_object(args.object_name)
    
    # Save results
    output_path = analyzer.save_results(results)
    
    print(f"\nOcclusion Analysis Complete!")
    print(f"Object: {results.object_name}")
    print(f"Viewpoints analyzed: {len(results.viewpoint_analyses)}")
    print(f"Overall correlation: {results.overall_correlation:.3f}")
    print(f"Statistical significance: {results.statistical_significance:.3f}")
    print(f"Results saved to: {output_path}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results.recommendations, 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    main()