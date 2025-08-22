"""
Weights & Biases (W&B) integration for gaussianfeels

This module provides comprehensive experiment tracking and visualization
capabilities for Gaussian splatting experiments, adapted from neuralfeels
patterns for gaussianfeels architecture.
"""

import os
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
import json
import pickle
from datetime import datetime
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WandBConfig:
    """Configuration for W&B integration"""
    project_name: str = "gaussianfeels"
    entity: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None
    mode: str = "online"  # online, offline, disabled
    log_frequency: int = 50  # Log every N iterations
    image_log_frequency: int = 100  # Log images every N iterations
    max_images_per_log: int = 5
    save_code: bool = True
    log_system_metrics: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.experiment_name is None:
            self.experiment_name = f"gaussian_splatting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class GaussianFeelsWandBLogger:
    """
    Comprehensive W&B logging integration for Gaussian splatting experiments
    """
    
    def __init__(self, 
                 config: WandBConfig,
                 auto_init: bool = True):
        """
        Initialize W&B logger.
        
        Args:
            config: W&B configuration
            auto_init: Whether to automatically initialize wandb run
        """
        self.config = config
        self.run = None
        self.step = 0
        self.experiment_data = {}
        
        if auto_init:
            self.init_run()
    
    def init_run(self, additional_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize W&B run.
        
        Args:
            additional_config: Additional configuration to log
            
        Returns:
            True if initialization successful
        """
        try:
            # Setup wandb configuration
            wandb_config = {
                'framework': 'gaussianfeels',
                'architecture': 'gaussian_splatting',
                'experiment_name': self.config.experiment_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if additional_config:
                wandb_config.update(additional_config)
            
            # Initialize run
            self.run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=self.config.experiment_name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=wandb_config,
                mode=self.config.mode,
                save_code=self.config.save_code
            )
            
            # Enable system metrics monitoring if requested
            if self.config.log_system_metrics:
                # This will log CPU, GPU, memory usage etc.
                pass  # wandb.watch is for model parameters, system metrics are automatic
            
            logger.info(f"Initialized W&B run: {self.run.name} ({self.run.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            self.run = None
            return False
    
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        """
        Log system information and configuration.
        
        Args:
            system_info: Dictionary containing system configuration
        """
        if not self.run:
            return
        
        try:
            # Log basic system info
            wandb.config.update({
                'system_info': system_info,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'python_version': system_info.get('python_version', 'unknown'),
                'torch_version': torch.__version__
            })
            
            if torch.cuda.is_available():
                wandb.config.update({
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory
                })
            
        except Exception as e:
            logger.warning(f"Failed to log system info: {e}")
    
    def log_experiment_config(self, 
                            gaussian_config: Dict[str, Any],
                            training_config: Dict[str, Any],
                            dataset_config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            gaussian_config: Gaussian splatting configuration
            training_config: Training configuration  
            dataset_config: Dataset configuration
        """
        if not self.run:
            return
        
        try:
            wandb.config.update({
                'gaussian_splatting': gaussian_config,
                'training': training_config,
                'dataset': dataset_config
            })
            
        except Exception as e:
            logger.warning(f"Failed to log experiment config: {e}")
    
    def log_training_step(self, 
                         step: int,
                         losses: Dict[str, float],
                         metrics: Optional[Dict[str, float]] = None,
                         learning_rates: Optional[Dict[str, float]] = None) -> None:
        """
        Log training step metrics.
        
        Args:
            step: Training step/iteration number
            losses: Dictionary of loss values
            metrics: Additional metrics to log
            learning_rates: Learning rates for different parameter groups
        """
        if not self.run:
            return
        
        try:
            log_dict = {}
            
            # Log losses with prefix
            for loss_name, loss_value in losses.items():
                log_dict[f'train/loss_{loss_name}'] = loss_value
            
            # Log metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    log_dict[f'train/{metric_name}'] = metric_value
            
            # Log learning rates
            if learning_rates:
                for lr_name, lr_value in learning_rates.items():
                    log_dict[f'train/lr_{lr_name}'] = lr_value
            
            # Log step number
            log_dict['step'] = step
            self.step = step
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log training step {step}: {e}")
    
    def log_gaussian_metrics(self, 
                           step: int,
                           gaussian_count: int,
                           opacity_stats: Dict[str, float],
                           scale_stats: Dict[str, float],
                           position_stats: Dict[str, float]) -> None:
        """
        Log Gaussian-specific metrics.
        
        Args:
            step: Current step
            gaussian_count: Number of Gaussians
            opacity_stats: Opacity statistics (mean, std, min, max)
            scale_stats: Scale statistics
            position_stats: Position statistics
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'gaussians/count': gaussian_count,
                'gaussians/opacity_mean': opacity_stats.get('mean', 0.0),
                'gaussians/opacity_std': opacity_stats.get('std', 0.0),
                'gaussians/scale_mean': scale_stats.get('mean', 0.0),
                'gaussians/scale_std': scale_stats.get('std', 0.0),
                'gaussians/position_spread': position_stats.get('std', 0.0),
                'step': step
            }
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log Gaussian metrics at step {step}: {e}")
    
    def log_tactile_metrics(self, 
                          step: int,
                          tactile_loss: float,
                          contact_accuracy: float,
                          depth_mae: float,
                          normal_angular_error: float) -> None:
        """
        Log tactile-specific metrics.
        
        Args:
            step: Current step
            tactile_loss: Tactile loss value
            contact_accuracy: Contact detection accuracy
            depth_mae: Depth prediction mean absolute error
            normal_angular_error: Surface normal angular error
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'tactile/loss': tactile_loss,
                'tactile/contact_accuracy': contact_accuracy,
                'tactile/depth_mae': depth_mae,
                'tactile/normal_angular_error': normal_angular_error,
                'step': step
            }
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log tactile metrics at step {step}: {e}")
    
    def log_pose_tracking(self, 
                         step: int,
                         pose_error_translation: float,
                         pose_error_rotation: float,
                         pose_trajectory: Optional[List[np.ndarray]] = None) -> None:
        """
        Log pose tracking metrics.
        
        Args:
            step: Current step
            pose_error_translation: Translation error in meters
            pose_error_rotation: Rotation error in degrees
            pose_trajectory: Optional full pose trajectory
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'pose/error_translation_m': pose_error_translation,
                'pose/error_rotation_deg': pose_error_rotation,
                'step': step
            }
            
            # Log pose trajectory as a plot if provided
            if pose_trajectory and len(pose_trajectory) > 1:
                # Create pose trajectory plot
                positions = np.array([pose[:3, 3] for pose in pose_trajectory])
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
                ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                          color='green', s=100, label='Start')
                ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                          color='red', s=100, label='End')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title('Pose Trajectory')
                ax.legend()
                
                log_dict['pose/trajectory_3d'] = wandb.Image(fig)
                plt.close(fig)
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log pose tracking at step {step}: {e}")
    
    def log_reconstruction_quality(self, 
                                 step: int,
                                 fscore: float,
                                 chamfer_distance: float,
                                 hausdorff_distance: float,
                                 point_cloud: Optional[np.ndarray] = None) -> None:
        """
        Log reconstruction quality metrics.
        
        Args:
            step: Current step
            fscore: F-score for surface reconstruction
            chamfer_distance: Chamfer distance
            hausdorff_distance: Hausdorff distance
            point_cloud: Optional point cloud for visualization
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'reconstruction/fscore': fscore,
                'reconstruction/chamfer_distance': chamfer_distance,
                'reconstruction/hausdorff_distance': hausdorff_distance,
                'step': step
            }
            
            # Create 3D point cloud visualization if provided
            if point_cloud is not None and len(point_cloud) > 0:
                # Subsample for visualization if too large
                if len(point_cloud) > 5000:
                    indices = np.random.choice(len(point_cloud), 5000, replace=False)
                    viz_points = point_cloud[indices]
                else:
                    viz_points = point_cloud
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=viz_points[:, 0],
                    y=viz_points[:, 1], 
                    z=viz_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=viz_points[:, 2],  # Color by Z coordinate
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )])
                
                fig.update_layout(
                    title='Reconstructed Point Cloud',
                    scene=dict(
                        xaxis_title='X (m)',
                        yaxis_title='Y (m)',
                        zaxis_title='Z (m)'
                    )
                )
                
                # Save as temporary HTML and log
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                    fig.write_html(tmp_file.name)
                    log_dict['reconstruction/point_cloud_3d'] = wandb.Html(tmp_file.name)
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log reconstruction quality at step {step}: {e}")
    
    def log_images(self, 
                  step: int,
                  images_dict: Dict[str, np.ndarray],
                  captions: Optional[Dict[str, str]] = None) -> None:
        """
        Log images to W&B.
        
        Args:
            step: Current step
            images_dict: Dictionary mapping image names to image arrays
            captions: Optional captions for images
        """
        if not self.run or step % self.config.image_log_frequency != 0:
            return
        
        try:
            log_dict = {'step': step}
            
            # Limit number of images per log
            image_names = list(images_dict.keys())[:self.config.max_images_per_log]
            
            for img_name in image_names:
                img_array = images_dict[img_name]
                
                # Handle different image formats
                if img_array.dtype != np.uint8:
                    # Normalize to 0-255 range
                    img_array = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                
                caption = captions.get(img_name, img_name) if captions else img_name
                
                log_dict[f'images/{img_name}'] = wandb.Image(img_array, caption=caption)
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log images at step {step}: {e}")
    
    def log_occlusion_analysis(self, 
                             step: int,
                             occlusion_level: float,
                             occlusion_impact_metrics: Dict[str, float]) -> None:
        """
        Log occlusion analysis metrics.
        
        Args:
            step: Current step
            occlusion_level: Estimated occlusion level (0-1)
            occlusion_impact_metrics: Metrics showing occlusion impact
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'occlusion/level': occlusion_level,
                'step': step
            }
            
            for metric_name, metric_value in occlusion_impact_metrics.items():
                log_dict[f'occlusion/{metric_name}'] = metric_value
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log occlusion analysis at step {step}: {e}")
    
    def log_performance_metrics(self, 
                              step: int,
                              processing_time: float,
                              memory_usage: Dict[str, float],
                              fps: float,
                              convergence_iterations: int) -> None:
        """
        Log performance metrics.
        
        Args:
            step: Current step
            processing_time: Processing time in seconds
            memory_usage: Memory usage statistics
            fps: Frames per second
            convergence_iterations: Iterations to convergence
        """
        if not self.run:
            return
        
        try:
            log_dict = {
                'performance/processing_time_s': processing_time,
                'performance/fps': fps,
                'performance/convergence_iterations': convergence_iterations,
                'step': step
            }
            
            # Add memory usage metrics
            for mem_type, mem_value in memory_usage.items():
                log_dict[f'performance/memory_{mem_type}'] = mem_value
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log performance metrics at step {step}: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.run:
            return
        
        try:
            wandb.config.update({'hyperparameters': hyperparams})
            
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_experiment_results(self, 
                             results: Dict[str, Any],
                             save_artifacts: bool = True) -> None:
        """
        Log final experiment results.
        
        Args:
            results: Dictionary containing final results
            save_artifacts: Whether to save results as W&B artifacts
        """
        if not self.run:
            return
        
        try:
            # Log summary metrics
            for key, value in results.items():
                if isinstance(value, (int, float, str, bool)):
                    wandb.summary[key] = value
            
            # Save as artifacts if requested
            if save_artifacts:
                # Save results as JSON artifact
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(results, tmp_file, indent=2, default=str)
                    tmp_file.flush()
                    
                    artifact = wandb.Artifact('experiment_results', type='results')
                    artifact.add_file(tmp_file.name, name='results.json')
                    wandb.log_artifact(artifact)
            
        except Exception as e:
            logger.warning(f"Failed to log experiment results: {e}")
    
    def create_custom_charts(self) -> None:
        """Create custom W&B charts for gaussianfeels-specific metrics."""
        if not self.run:
            return
        
        try:
            # Define custom charts for specific metric combinations
            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("tactile/*", step_metric="step")
            wandb.define_metric("pose/*", step_metric="step")
            wandb.define_metric("reconstruction/*", step_metric="step")
            wandb.define_metric("gaussians/*", step_metric="step")
            wandb.define_metric("occlusion/*", step_metric="step")
            wandb.define_metric("performance/*", step_metric="step")
            
        except Exception as e:
            logger.warning(f"Failed to create custom charts: {e}")
    
    def finish_run(self) -> None:
        """Finish the current W&B run."""
        if self.run:
            try:
                wandb.finish()
                logger.info("Finished W&B run")
                
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
            
            finally:
                self.run = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish_run()


# Convenience function for quick setup
def create_wandb_logger(project_name: str = "gaussianfeels",
                       experiment_name: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       mode: str = "online") -> GaussianFeelsWandBLogger:
    """
    Create a W&B logger with default configuration.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name
        tags: List of tags
        mode: W&B mode (online, offline, disabled)
        
    Returns:
        Configured W&B logger
    """
    config = WandBConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        tags=tags or [],
        mode=mode
    )
    
    return GaussianFeelsWandBLogger(config=config)


def main():
    """Test W&B integration"""
    # Create logger
    logger = create_wandb_logger(
        project_name="gaussianfeels_test",
        experiment_name="integration_test",
        tags=["test", "integration"],
        mode="offline"  # Use offline mode for testing
    )
    
    # Log system info
    system_info = {
        'python_version': '3.11',
        'os': 'linux',
        'gpu_available': torch.cuda.is_available()
    }
    logger.log_system_info(system_info)
    
    # Simulate training loop
    for step in range(100):
        # Simulate training metrics
        losses = {
            'total': np.random.exponential(0.5) + 0.1,
            'rgb': np.random.exponential(0.2) + 0.05,
            'tactile': np.random.exponential(0.3) + 0.02
        }
        
        metrics = {
            'accuracy': min(0.5 + step * 0.005 + np.random.normal(0, 0.1), 1.0)
        }
        
        logger.log_training_step(step, losses, metrics)
        
        # Occasionally log additional metrics
        if step % 25 == 0:
            logger.log_gaussian_metrics(
                step=step,
                gaussian_count=1000 + step * 10,
                opacity_stats={'mean': 0.8, 'std': 0.1},
                scale_stats={'mean': 0.01, 'std': 0.005},
                position_stats={'std': 0.5}
            )
            
            logger.log_pose_tracking(
                step=step,
                pose_error_translation=max(0.1 - step * 0.001, 0.01),
                pose_error_rotation=max(10.0 - step * 0.05, 1.0)
            )
    
    # Log final results
    results = {
        'final_loss': 0.05,
        'final_accuracy': 0.95,
        'total_steps': 100,
        'success': True
    }
    logger.log_experiment_results(results)
    
    print("W&B integration test completed!")


if __name__ == "__main__":
    main()