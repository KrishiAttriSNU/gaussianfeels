"""
Pipeline integration for W&B logging with gaussianfeels

This module demonstrates how to integrate W&B logging with existing 
gaussianfeels training and evaluation pipelines, providing drop-in
compatibility with minimal code changes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import logging
from contextlib import contextmanager

from .wandb_integration import GaussianFeelsWandBLogger, WandBConfig, create_wandb_logger

# Import existing gaussianfeels modules
try:
    from camera.gaussianfeels.utils.performance_diagnostics import PerformanceDiagnostics
    from camera.gaussianfeels.loss.tactile_loss import TactileSurfaceLoss
    from camera.gaussianfeels.loss.volumetric_loss import VolumetricLossFunction
    from misc.eval.tactile_fusion_evaluation import TactileFusionEvaluator
except ImportError as e:
    logging.warning(f"Could not import gaussianfeels modules: {e}")

logger = logging.getLogger(__name__)


class WandBIntegratedTrainer:
    """
    Enhanced trainer with W&B logging integration for gaussianfeels
    """
    
    def __init__(self, 
                 wandb_config: Optional[WandBConfig] = None,
                 enable_logging: bool = True,
                 log_frequency: int = 50):
        """
        Initialize integrated trainer.
        
        Args:
            wandb_config: W&B configuration
            enable_logging: Whether to enable W&B logging
            log_frequency: Logging frequency
        """
        self.enable_logging = enable_logging
        self.log_frequency = log_frequency
        
        # Initialize W&B logger if enabled
        self.wandb_logger = None
        if enable_logging:
            if wandb_config is None:
                wandb_config = WandBConfig()
            
            self.wandb_logger = GaussianFeelsWandBLogger(wandb_config)
            
            # Setup custom charts
            self.wandb_logger.create_custom_charts()
        
        # Performance diagnostics integration
        self.performance_diagnostics = None
        try:
            self.performance_diagnostics = PerformanceDiagnostics()
        except Exception:
            logger.warning("Could not initialize performance diagnostics")
    
    @contextmanager
    def training_context(self, config: Dict[str, Any]):
        """
        Context manager for training with automatic logging setup and teardown.
        
        Args:
            config: Training configuration to log
        """
        try:
            # Log experiment configuration
            if self.wandb_logger:
                self.wandb_logger.log_experiment_config(
                    gaussian_config=config.get('gaussian', {}),
                    training_config=config.get('training', {}),
                    dataset_config=config.get('dataset', {})
                )
                
                # Log system info
                system_info = self._get_system_info()
                self.wandb_logger.log_system_info(system_info)
                
                # Log hyperparameters
                if 'hyperparameters' in config:
                    self.wandb_logger.log_hyperparameters(config['hyperparameters'])
            
            yield self
            
        finally:
            # Cleanup
            if self.wandb_logger:
                self.wandb_logger.finish_run()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for logging."""
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return system_info
    
    def log_training_iteration(self, 
                             step: int,
                             losses: Dict[str, torch.Tensor],
                             metrics: Optional[Dict[str, float]] = None,
                             learning_rates: Optional[Dict[str, float]] = None,
                             gaussian_params: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Log training iteration with comprehensive metrics.
        
        Args:
            step: Training step
            losses: Loss tensors
            metrics: Additional metrics
            learning_rates: Learning rates
            gaussian_params: Gaussian parameters for analysis
        """
        if not self.enable_logging or not self.wandb_logger or step % self.log_frequency != 0:
            return
        
        try:
            # Convert losses to float values
            loss_values = {name: loss.item() if torch.is_tensor(loss) else loss 
                          for name, loss in losses.items()}
            
            # Log basic training metrics
            self.wandb_logger.log_training_step(
                step=step,
                losses=loss_values,
                metrics=metrics,
                learning_rates=learning_rates
            )
            
            # Log Gaussian-specific metrics if available
            if gaussian_params:
                self._log_gaussian_analysis(step, gaussian_params)
            
            # Log performance metrics from diagnostics
            if self.performance_diagnostics:
                perf_snapshot = self.performance_diagnostics.capture_performance_snapshot()
                self._log_performance_snapshot(step, perf_snapshot)
        
        except Exception as e:
            logger.warning(f"Failed to log training iteration {step}: {e}")
    
    def _log_gaussian_analysis(self, 
                             step: int, 
                             gaussian_params: Dict[str, torch.Tensor]) -> None:
        """Log Gaussian parameter analysis."""
        try:
            # Analyze Gaussian parameters
            analysis = {}
            
            # Opacity analysis
            if 'opacity' in gaussian_params:
                opacity = gaussian_params['opacity']
                analysis['opacity'] = {
                    'mean': opacity.mean().item(),
                    'std': opacity.std().item(),
                    'min': opacity.min().item(),
                    'max': opacity.max().item()
                }
            
            # Scale analysis
            if 'scaling' in gaussian_params:
                scaling = gaussian_params['scaling']
                analysis['scale'] = {
                    'mean': scaling.mean().item(),
                    'std': scaling.std().item(),
                    'min': scaling.min().item(),
                    'max': scaling.max().item()
                }
            
            # Position analysis
            if 'xyz' in gaussian_params:
                positions = gaussian_params['xyz']
                analysis['position'] = {
                    'mean': positions.mean().item(),
                    'std': positions.std().item(),
                    'spread_x': positions[:, 0].std().item(),
                    'spread_y': positions[:, 1].std().item(),
                    'spread_z': positions[:, 2].std().item()
                }
            
            # Count
            gaussian_count = len(gaussian_params.get('xyz', []))
            
            self.wandb_logger.log_gaussian_metrics(
                step=step,
                gaussian_count=gaussian_count,
                opacity_stats=analysis.get('opacity', {}),
                scale_stats=analysis.get('scale', {}),
                position_stats=analysis.get('position', {})
            )
            
        except Exception as e:
            logger.warning(f"Failed to log Gaussian analysis: {e}")
    
    def _log_performance_snapshot(self, 
                                step: int, 
                                snapshot: Dict[str, Any]) -> None:
        """Log performance snapshot from diagnostics."""
        try:
            memory_info = snapshot.get('memory_info', {})
            timing_info = snapshot.get('timing_info', {})
            system_info = snapshot.get('system_info', {})
            
            # Extract relevant metrics
            processing_time = timing_info.get('last_iteration_time', 0.0)
            fps = timing_info.get('fps', 0.0)
            
            memory_usage = {
                'system_used_gb': memory_info.get('system_memory_used', 0) / (1024**3),
                'gpu_used_mb': memory_info.get('gpu_memory_used', 0) / (1024**2)
            }
            
            convergence_iterations = system_info.get('convergence_iterations', 0)
            
            self.wandb_logger.log_performance_metrics(
                step=step,
                processing_time=processing_time,
                memory_usage=memory_usage,
                fps=fps,
                convergence_iterations=convergence_iterations
            )
            
        except Exception as e:
            logger.warning(f"Failed to log performance snapshot: {e}")
    
    def log_tactile_metrics(self, 
                          step: int,
                          tactile_results: Dict[str, float]) -> None:
        """
        Log tactile-specific metrics.
        
        Args:
            step: Training step
            tactile_results: Dictionary of tactile metrics
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            self.wandb_logger.log_tactile_metrics(
                step=step,
                tactile_loss=tactile_results.get('loss', 0.0),
                contact_accuracy=tactile_results.get('contact_accuracy', 0.0),
                depth_mae=tactile_results.get('depth_mae', 0.0),
                normal_angular_error=tactile_results.get('normal_angular_error', 0.0)
            )
            
        except Exception as e:
            logger.warning(f"Failed to log tactile metrics: {e}")
    
    def log_pose_estimation(self, 
                          step: int,
                          pose_results: Dict[str, Any]) -> None:
        """
        Log pose estimation results.
        
        Args:
            step: Training step
            pose_results: Dictionary containing pose estimation results
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            self.wandb_logger.log_pose_tracking(
                step=step,
                pose_error_translation=pose_results.get('translation_error', 0.0),
                pose_error_rotation=pose_results.get('rotation_error', 0.0),
                pose_trajectory=pose_results.get('trajectory', None)
            )
            
        except Exception as e:
            logger.warning(f"Failed to log pose estimation: {e}")
    
    def log_reconstruction_evaluation(self, 
                                   step: int,
                                   evaluation_results: Dict[str, Any]) -> None:
        """
        Log reconstruction evaluation results.
        
        Args:
            step: Training step
            evaluation_results: Results from reconstruction evaluation
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            self.wandb_logger.log_reconstruction_quality(
                step=step,
                fscore=evaluation_results.get('fscore', 0.0),
                chamfer_distance=evaluation_results.get('chamfer_distance', float('inf')),
                hausdorff_distance=evaluation_results.get('hausdorff_distance', float('inf')),
                point_cloud=evaluation_results.get('point_cloud', None)
            )
            
        except Exception as e:
            logger.warning(f"Failed to log reconstruction evaluation: {e}")
    
    def log_images_batch(self, 
                       step: int,
                       image_batch: Dict[str, np.ndarray],
                       max_images: int = 5) -> None:
        """
        Log batch of images.
        
        Args:
            step: Training step
            image_batch: Dictionary of image arrays
            max_images: Maximum number of images to log
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            # Limit number of images
            limited_batch = {k: v for i, (k, v) in enumerate(image_batch.items()) 
                           if i < max_images}
            
            self.wandb_logger.log_images(
                step=step,
                images_dict=limited_batch
            )
            
        except Exception as e:
            logger.warning(f"Failed to log images batch: {e}")
    
    def log_occlusion_experiment(self, 
                               step: int,
                               occlusion_results: Dict[str, Any]) -> None:
        """
        Log occlusion experiment results.
        
        Args:
            step: Training step
            occlusion_results: Results from occlusion analysis
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            occlusion_level = occlusion_results.get('occlusion_level', 0.0)
            impact_metrics = {k: v for k, v in occlusion_results.items() 
                            if k != 'occlusion_level'}
            
            self.wandb_logger.log_occlusion_analysis(
                step=step,
                occlusion_level=occlusion_level,
                occlusion_impact_metrics=impact_metrics
            )
            
        except Exception as e:
            logger.warning(f"Failed to log occlusion experiment: {e}")
    
    def finalize_experiment(self, 
                          final_results: Dict[str, Any],
                          save_artifacts: bool = True) -> None:
        """
        Finalize experiment with final results logging.
        
        Args:
            final_results: Final experiment results
            save_artifacts: Whether to save results as artifacts
        """
        if not self.enable_logging or not self.wandb_logger:
            return
        
        try:
            self.wandb_logger.log_experiment_results(
                results=final_results,
                save_artifacts=save_artifacts
            )
            
        except Exception as e:
            logger.warning(f"Failed to finalize experiment: {e}")


# Convenience decorator for automatic W&B logging
def with_wandb_logging(project_name: str = "gaussianfeels",
                      experiment_name: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      mode: str = "online"):
    """
    Decorator to automatically add W&B logging to training functions.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name
        tags: List of tags
        mode: W&B mode
    """
    def decorator(training_func: Callable):
        def wrapper(*args, **kwargs):
            # Create W&B logger
            config = WandBConfig(
                project_name=project_name,
                experiment_name=experiment_name,
                tags=tags or [],
                mode=mode
            )
            
            logger = GaussianFeelsWandBLogger(config)
            
            try:
                # Pass logger to training function
                if 'wandb_logger' not in kwargs:
                    kwargs['wandb_logger'] = logger
                
                result = training_func(*args, **kwargs)
                
                # Log final results if returned as dict
                if isinstance(result, dict):
                    logger.log_experiment_results(result)
                
                return result
                
            finally:
                logger.finish_run()
        
        return wrapper
    return decorator


# Integration examples
def example_training_loop_integration():
    """Example showing how to integrate W&B with existing training loop."""
    
    # Create integrated trainer
    wandb_config = WandBConfig(
        project_name="gaussianfeels_example",
        experiment_name="tactile_fusion_experiment",
        tags=["example", "tactile", "fusion"]
    )
    
    trainer = WandBIntegratedTrainer(wandb_config=wandb_config)
    
    # Training configuration
    config = {
        'gaussian': {
            'max_gaussians': 10000,
            'opacity_threshold': 0.01,
            'scale_threshold': 0.001
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 8,
            'max_iterations': 1000
        },
        'dataset': {
            'name': 'feelsight',
            'object': '010_potted_meat_can',
            'viewpoints': ['00', '01', '02', '03', '04']
        },
        'hyperparameters': {
            'tactile_weight': 0.5,
            'vision_weight': 0.5,
            'regularization_weight': 0.01
        }
    }
    
    # Training loop with context manager
    with trainer.training_context(config) as logged_trainer:
        
        for step in range(1000):
            # Simulate training step
            losses = {
                'total': torch.tensor(np.random.exponential(0.5) + 0.1),
                'rgb': torch.tensor(np.random.exponential(0.2) + 0.05),
                'tactile': torch.tensor(np.random.exponential(0.3) + 0.02),
                'regularization': torch.tensor(np.random.exponential(0.01) + 0.001)
            }
            
            metrics = {
                'accuracy': min(0.5 + step * 0.0005 + np.random.normal(0, 0.05), 1.0),
                'convergence_rate': max(1.0 - step * 0.001, 0.1)
            }
            
            learning_rates = {
                'gaussian_params': 0.001 * (0.9 ** (step // 100)),
                'camera_params': 0.0001
            }
            
            # Simulate Gaussian parameters
            gaussian_params = {
                'xyz': torch.randn(1000 + step, 3) * 0.1,
                'opacity': torch.sigmoid(torch.randn(1000 + step, 1)),
                'scaling': torch.exp(torch.randn(1000 + step, 3) * 0.1),
            }
            
            # Log training iteration
            logged_trainer.log_training_iteration(
                step=step,
                losses=losses,
                metrics=metrics,
                learning_rates=learning_rates,
                gaussian_params=gaussian_params
            )
            
            # Periodically log additional metrics
            if step % 100 == 0:
                # Simulate tactile metrics
                tactile_results = {
                    'loss': losses['tactile'].item(),
                    'contact_accuracy': 0.8 + np.random.normal(0, 0.1),
                    'depth_mae': max(0.01 - step * 0.00005, 0.001),
                    'normal_angular_error': max(5.0 - step * 0.02, 0.5)
                }
                logged_trainer.log_tactile_metrics(step, tactile_results)
                
                # Simulate pose estimation
                pose_results = {
                    'translation_error': max(0.1 - step * 0.0001, 0.001),
                    'rotation_error': max(10.0 - step * 0.01, 0.1)
                }
                logged_trainer.log_pose_estimation(step, pose_results)
                
                # Simulate reconstruction evaluation
                evaluation_results = {
                    'fscore': min(0.3 + step * 0.0007, 0.95),
                    'chamfer_distance': max(0.05 - step * 0.00005, 0.001),
                    'hausdorff_distance': max(0.1 - step * 0.0001, 0.01)
                }
                logged_trainer.log_reconstruction_evaluation(step, evaluation_results)
        
        # Finalize with results
        final_results = {
            'final_loss': losses['total'].item(),
            'final_accuracy': metrics['accuracy'],
            'total_gaussians': len(gaussian_params['xyz']),
            'convergence_achieved': True,
            'experiment_duration_minutes': 30.5
        }
        
        logged_trainer.finalize_experiment(final_results)


@with_wandb_logging(project_name="gaussianfeels_decorated", tags=["decorator", "test"])
def example_decorated_training(wandb_logger: GaussianFeelsWandBLogger):
    """Example training function with decorator-based W&B logging."""
    
    # Training loop
    for step in range(50):
        losses = {
            'total': np.random.exponential(0.5) + 0.1
        }
        
        wandb_logger.log_training_step(step, losses)
    
    # Return results (will be automatically logged)
    return {
        'success': True,
        'final_loss': 0.05,
        'steps_completed': 50
    }


if __name__ == "__main__":
    # Run examples
    print("Running training loop integration example...")
    example_training_loop_integration()
    
    print("Running decorated training example...")
    example_decorated_training()
    
    print("Integration examples completed!")