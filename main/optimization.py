"""
GaussianFeels Performance and Memory Optimization

Advanced optimization techniques for Gaussian splatting training including
memory management, GPU utilization, batch processing, and computational efficiency.
"""

import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import psutil
from torch.profiler import profile, record_function, ProfilerActivity

from .config import GaussianFeelsConfig
from .trainer import GaussianTrainer, GaussianField
from .datasets import BaseDataset, FrameData
from shared.utils.device_utils import get_optimal_memory_config, get_gpu_memory_gb

@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations"""
    # Memory management - automatically configured based on GPU
    enable_gradient_checkpointing: bool = False
    max_memory_gb: float = None  # Auto-detected based on GPU
    memory_cleanup_interval: int = 100
    gaussian_batch_size: int = 1000  # Process Gaussians in batches
    
    def __post_init__(self):
        """Auto-configure based on GPU memory if not explicitly set"""
        if self.max_memory_gb is None:
            try:
                optimal_config = get_optimal_memory_config()
                self.max_memory_gb = optimal_config['max_memory_gb']
                # Only auto-adjust if using defaults
                if self.gaussian_batch_size == 1000:
                    self.gaussian_batch_size = optimal_config['gaussian_batch_size']
                if self.memory_cleanup_interval == 100:
                    self.memory_cleanup_interval = optimal_config['memory_cleanup_interval']
                print(f"🚀 Auto-detected {get_gpu_memory_gb():.1f}GB GPU - configured for {optimal_config['optimization_level']} mode")
            except Exception as e:
                print(f"⚠️ Could not auto-detect GPU memory, using defaults: {e}")
                self.max_memory_gb = 8.0
    
    # GPU optimization
    enable_mixed_precision: bool = True
    enable_compile: bool = False  # torch.compile (PyTorch 2.0+)
    enable_memory_efficient_attention: bool = True
    enable_channels_last: bool = False
    
    # Computation optimization
    enable_fast_math: bool = True
    enable_tf32: bool = True
    optimize_for_inference: bool = False
    cache_gradients: bool = True
    
    # Profiling and monitoring
    enable_profiling: bool = False
    profile_steps: int = 10
    monitor_memory: bool = True
    monitor_gpu_utilization: bool = True
    
    # Advanced optimizations
    enable_fusion: bool = True
    enable_async_operations: bool = True
    prefetch_factor: int = 2
    num_workers: int = 4

class MemoryManager:
    """Advanced memory management for Gaussian splatting"""
    
    def __init__(self, config: OptimizationConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.memory_stats = []
        self.peak_memory = 0.0
        self.last_cleanup = 0
        
        # Memory pools
        self.gaussian_cache = {}
        self.gradient_cache = {}
        self.tensor_pool = []
        
    @contextmanager
    def memory_context(self, operation_name: str = ""):
        """Context manager for memory monitoring"""
        if self.config.monitor_memory:
            start_memory = self.get_memory_usage()
            start_time = time.time()
            
        try:
            yield
        finally:
            if self.config.monitor_memory:
                end_memory = self.get_memory_usage()
                end_time = time.time()
                
                self.memory_stats.append({
                    'operation': operation_name,
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'memory_delta': end_memory - start_memory,
                    'duration': end_time - start_time,
                    'timestamp': time.time()
                })
                
                if end_memory > self.peak_memory:
                    self.peak_memory = end_memory
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required but not available")
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            return allocated
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1e9
    
    def cleanup_memory(self, force: bool = False):
        """Clean up memory"""
        current_memory = self.get_memory_usage()
        
        if force or current_memory > self.config.max_memory_gb:
            # Clear caches
            self.gaussian_cache.clear()
            self.gradient_cache.clear()
            self.tensor_pool.clear()
            
            # PyTorch cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Python cleanup
            gc.collect()
            
            new_memory = self.get_memory_usage()
            freed = current_memory - new_memory
            
            if freed > 0.1:  # Only log if significant cleanup
                print(f"🧹 Memory cleanup: {freed:.2f}GB freed ({current_memory:.2f}GB → {new_memory:.2f}GB)")
    
    def allocate_tensor_pool(self, size: int, dtype: torch.dtype = torch.float32):
        """Pre-allocate tensor pool for efficient reuse"""
        for _ in range(10):  # Pre-allocate 10 tensors
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            self.tensor_pool.append(tensor)
    
    def get_pooled_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or allocate new one"""
        required_size = np.prod(shape)
        
        # Try to find suitable tensor in pool
        for i, tensor in enumerate(self.tensor_pool):
            if tensor.numel() >= required_size and tensor.dtype == dtype:
                # Remove from pool and reshape
                pooled_tensor = self.tensor_pool.pop(i)
                return pooled_tensor.view(shape)
        
        # Allocate new tensor if none found
        return torch.empty(shape, dtype=dtype, device=self.device)
    
    def return_to_pool(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        if len(self.tensor_pool) < 50:  # Limit pool size
            self.tensor_pool.append(tensor.detach())

class GaussianBatchProcessor:
    """Efficient batch processing for large Gaussian fields"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.batch_size = config.gaussian_batch_size
        
    def process_gaussians_batched(self, gaussians: List[Dict], 
                                  operation: callable) -> List[Any]:
        """Process Gaussians in batches to save memory"""
        results = []
        
        for i in range(0, len(gaussians), self.batch_size):
            batch = gaussians[i:i + self.batch_size]
            
            with torch.cuda.amp.autocast(enabled=self.config.enable_mixed_precision):
                batch_result = operation(batch)
            
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Cleanup intermediate results
            if i % (self.batch_size * 5) == 0:  # Every 5 batches
                torch.cuda.empty_cache()
        
        return results
    
    def batched_render(self, gaussians: List[Dict], camera_params: Dict) -> torch.Tensor:
        """Batch render Gaussians for memory efficiency (simple point splat)."""
        if len(gaussians) == 0:
            raise RuntimeError("No Gaussians available for batched rendering. Initialize Gaussian field first.")
        batch_results = []
        for i in range(0, len(gaussians), self.batch_size):
            batch = gaussians[i:i + self.batch_size]
            batch_render = self._render_gaussian_batch(batch, camera_params)
            batch_results.append(batch_render)
        if not batch_results:
            raise RuntimeError("No Gaussian batches rendered. Check batch size and Gaussian count.")
        return torch.sum(torch.stack(batch_results), dim=0)
    
    def _render_gaussian_batch(self, batch: List[Dict], camera_params: Dict) -> torch.Tensor:
        """Render a batch of Gaussians via pinhole projection and nearest-pixel splat."""
        positions = torch.stack([g["position"] for g in batch])  # [B,3]
        colors = torch.stack([g["color"] for g in batch])        # [B,3]
        device = positions.device
        # STRICT: Camera parameters must be explicitly provided - no defaults allowed
        if 'H' not in camera_params:
            raise KeyError("camera_params missing required 'H' key")
        if 'W' not in camera_params:
            raise KeyError("camera_params missing required 'W' key")
        if 'fx' not in camera_params:
            raise KeyError("camera_params missing required 'fx' key")
        if 'fy' not in camera_params:
            raise KeyError("camera_params missing required 'fy' key")
        if 'cx' not in camera_params:
            raise KeyError("camera_params missing required 'cx' key")
        if 'cy' not in camera_params:
            raise KeyError("camera_params missing required 'cy' key")
        
        H = int(camera_params['H'])
        W = int(camera_params['W'])
        fx = float(camera_params['fx'])
        fy = float(camera_params['fy'])
        cx = float(camera_params['cx'])
        cy = float(camera_params['cy'])
        # Project to pixels (optical frame convention: negative X)
        zs = positions[:, 2].clamp(min=1e-6)
        us = (-(fx * positions[:, 0] / zs) + cx).round().long()
        vs = ((fy * positions[:, 1] / zs) + cy).round().long()
        rendered = torch.full((H, W, 3), 1e-8, device=device)
        valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us = us[valid]
        vs = vs[valid]
        cols = colors[valid]
        if us.numel() > 0:
            rendered[vs, us] = cols
        return rendered

class OptimizedGaussianField(GaussianField):
    """Memory and computation optimized Gaussian field"""
    
    def __init__(self, initial_positions: torch.Tensor, device: str = "cuda",
                 optimization_config: OptimizationConfig = None):
        super().__init__(initial_positions, device)
        
        self.opt_config = optimization_config or OptimizationConfig()
        self.memory_manager = MemoryManager(self.opt_config, device)
        self.batch_processor = GaussianBatchProcessor(self.opt_config)
        
        # Optimization state
        self.compiled_forward = None
        self.gradient_cache = {}
        self.last_optimization_step = 0
        
        # Apply optimizations
        self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply various optimizations to the field"""
        if self.opt_config.enable_tf32:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for TF32 optimization")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if self.opt_config.enable_channels_last:
            # Convert tensors to channels_last format where applicable
            # Note: Channels-last memory format optimization is not yet implemented
            # This would require converting all relevant tensors to channels_last format
            # using tensor.to(memory_format=torch.channels_last)
            logger.debug("Channels-last optimization requested but not yet implemented")
        
        if self.opt_config.enable_compile:
            try:
                self.compiled_forward = torch.compile(self.forward)
                print("✅ Enabled torch.compile optimization")
            except (RuntimeError, AttributeError, ImportError) as e:
                if self.opt_config.require_compile:
                    raise RuntimeError(f"torch.compile required but failed: {e}") from e
                else:
                    print(f"⚠️ torch.compile not available (optional): {e}")
    
    def get_gaussians_optimized(self, max_gaussians: Optional[int] = None) -> List[Dict]:
        """Get Gaussians with memory optimization"""
        with self.memory_manager.memory_context("get_gaussians"):
            gaussians = self.get_gaussians()
            
            # Limit number of Gaussians for memory
            if max_gaussians and len(gaussians) > max_gaussians:
                # Sort by opacity and take top ones
                opacities = [torch.sigmoid(g["opacity"]).item() for g in gaussians]
                sorted_indices = np.argsort(opacities)[::-1][:max_gaussians]
                gaussians = [gaussians[i] for i in sorted_indices]
            
            return gaussians
    
    def densify_optimized(self, positions: torch.Tensor):
        """Memory-optimized densification"""
        with self.memory_manager.memory_context("densify"):
            # Check memory before densification
            if self.memory_manager.get_memory_usage() > self.opt_config.max_memory_gb * 0.8:
                print("⚠️ Memory limit approaching, skipping densification")
                return
            
            # Batch densification for large position sets
            if len(positions) > 1000:
                for i in range(0, len(positions), 1000):
                    batch_positions = positions[i:i + 1000]
                    super().densify(batch_positions)
                    
                    # Cleanup between batches
                    if i % 5000 == 0:
                        self.memory_manager.cleanup_memory()
            else:
                super().densify(positions)
    
    def forward_optimized(self, *args, **kwargs):
        """Optimized forward pass"""
        if self.compiled_forward is not None:
            return self.compiled_forward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)

class OptimizedTrainer(GaussianTrainer):
    """Performance-optimized Gaussian trainer"""
    
    def __init__(self, config: GaussianFeelsConfig, dataset: BaseDataset,
                 optimization_config: OptimizationConfig = None):
        self.opt_config = optimization_config or OptimizationConfig()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.opt_config, config.device)
        
        # Initialize base trainer
        super().__init__(config, dataset)
        
        # Replace Gaussian field with optimized version
        self.gaussian_field = OptimizedGaussianField(
            self.gaussian_field.positions.data.clone(),
            config.device,
            self.opt_config
        )
        
        # Setup optimized components
        self._setup_optimizations()
        
        # Performance monitoring
        self.optimization_stats = {
            'memory_usage': [],
            'gpu_utilization': [],
            'throughput': [],
            'efficiency_score': []
        }
        
        # Profiler
        self.profiler = None
        if self.opt_config.enable_profiling:
            self._setup_profiler()
    
    def _setup_optimizations(self):
        """Setup various optimizations"""
        # Mixed precision scaler
        if self.opt_config.enable_mixed_precision:
            self.scaler = GradScaler()
        
        # Pre-allocate tensor pools
        self.memory_manager.allocate_tensor_pool(1000000)  # 1M elements
        
        # Setup optimized optimizers
        self._setup_optimized_optimizers()
        
        # Enable gradient checkpointing if needed
        if self.opt_config.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _setup_optimized_optimizers(self):
        """Setup optimizers with performance optimizations"""
        # Use fused optimizers for better performance
        optimizer_kwargs = {}
        if self.opt_config.enable_fast_math:
            optimizer_kwargs['fused'] = True
        
        # Recreate optimizers with optimizations
        for name, optimizer in self.optimizers.items():
            param_groups = optimizer.param_groups
            
            try:
                # Require fused optimizer if enabled
                new_optimizer = torch.optim.AdamW(
                    param_groups[0]['params'],
                    lr=param_groups[0]['lr'],
                    **optimizer_kwargs
                )
                self.optimizers[name] = new_optimizer
            except (RuntimeError, ValueError, TypeError) as e:
                if self.opt_config.enable_fast_math:
                    raise RuntimeError(f"Fused optimizer required but not available for {name}: {e}") from e
                else:
                    # Create regular optimizer without fused option
                    new_optimizer = torch.optim.AdamW(
                        param_groups[0]['params'],
                        lr=param_groups[0]['lr']
                    )
                    self.optimizers[name] = new_optimizer
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Wrap forward pass with checkpointing
        original_forward = self.gaussian_field.forward
        
        def checkpointed_forward(*args, **kwargs):
            return torch_checkpoint(original_forward, *args, **kwargs)
        
        self.gaussian_field.forward = checkpointed_forward
    
    def _setup_profiler(self):
        """Setup PyTorch profiler"""
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    
    def step_map_optimized(self) -> float:
        """Optimized map step with performance monitoring"""
        step_start = time.time()
        
        with self.memory_manager.memory_context("map_step"):
            # Regular training step
            if self.profiler and self.step < self.opt_config.profile_steps:
                with self.profiler:
                    loss = super().step_map()
            else:
                loss = super().step_map()
            
            # Performance monitoring
            self._update_performance_stats()
            
            # Periodic memory cleanup
            if self.step % self.opt_config.memory_cleanup_interval == 0:
                self.memory_manager.cleanup_memory()
            
            # Adaptive batch sizing based on memory usage
            self._adjust_batch_size()
        
        step_time = time.time() - step_start
        
        return loss
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        # Memory usage
        memory_usage = self.memory_manager.get_memory_usage()
        self.optimization_stats['memory_usage'].append(memory_usage)
        
        # GPU utilization
        if self.opt_config.monitor_gpu_utilization:
            gpu_util = self._get_gpu_utilization()
            self.optimization_stats['gpu_utilization'].append(gpu_util)
        
        # Throughput (Gaussians processed per second)
        if len(self.timings["map_step"]) > 0:
            recent_time = np.mean(self.timings["map_step"][-10:])
            throughput = self.num_gaussians / recent_time if recent_time > 0 else 0.0
            self.optimization_stats['throughput'].append(throughput)
        
        # Efficiency score (combination of speed and memory efficiency)
        efficiency = self._calculate_efficiency_score()
        self.optimization_stats['efficiency_score'].append(efficiency)
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on memory usage"""
        current_memory = self.memory_manager.get_memory_usage()
        memory_ratio = current_memory / self.opt_config.max_memory_gb
        
        if memory_ratio > 0.9:
            # Reduce batch size
            new_batch_size = max(100, self.opt_config.gaussian_batch_size // 2)
            self.opt_config.gaussian_batch_size = new_batch_size
        elif memory_ratio < 0.5:
            # Increase batch size
            new_batch_size = min(5000, self.opt_config.gaussian_batch_size * 2)
            self.opt_config.gaussian_batch_size = new_batch_size
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        if not self.optimization_stats['throughput']:
            raise RuntimeError("Cannot calculate efficiency score: no throughput data available")
        
        # Normalize throughput (higher is better)
        throughput = self.optimization_stats['throughput'][-1]
        throughput_score = min(1.0, throughput / 1000.0)  # Normalize to 1000 Gaussians/sec
        
        # Normalize memory usage (lower is better)
        memory_usage = self.optimization_stats['memory_usage'][-1]
        memory_score = max(0.0, 1.0 - memory_usage / self.opt_config.max_memory_gb)
        
        # Combined efficiency score
        efficiency = 0.6 * throughput_score + 0.4 * memory_score
        return efficiency
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        stats = self.optimization_stats
        
        report = {
            'memory_stats': {
                'current_usage_gb': self.memory_manager.get_memory_usage(),
                'peak_usage_gb': self.memory_manager.peak_memory,
                'average_usage_gb': np.mean(stats['memory_usage']) if stats['memory_usage'] else 0.0,
                'memory_limit_gb': self.opt_config.max_memory_gb,
                'utilization_ratio': self.memory_manager.get_memory_usage() / self.opt_config.max_memory_gb
            },
            
            'performance_stats': {
                'current_throughput': stats['throughput'][-1] if stats['throughput'] else 0.0,
                'average_throughput': np.mean(stats['throughput']) if stats['throughput'] else 0.0,
                'peak_throughput': np.max(stats['throughput']) if stats['throughput'] else 0.0,
                'current_efficiency': stats['efficiency_score'][-1] if stats['efficiency_score'] else 0.0,
                'average_efficiency': np.mean(stats['efficiency_score']) if stats['efficiency_score'] else 0.0,
            },
            
            'optimization_config': {
                'mixed_precision': self.opt_config.enable_mixed_precision,
                'gradient_checkpointing': self.opt_config.enable_gradient_checkpointing,
                'batch_size': self.opt_config.gaussian_batch_size,
                'memory_cleanup_interval': self.opt_config.memory_cleanup_interval,
                'tf32_enabled': self.opt_config.enable_tf32,
            },
            
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        memory_usage = self.memory_manager.get_memory_usage()
        if memory_usage > self.opt_config.max_memory_gb * 0.9:
            recommendations.append("Consider enabling gradient checkpointing or reducing batch size")
        
        # Throughput recommendations
        if self.optimization_stats['throughput']:
            avg_throughput = np.mean(self.optimization_stats['throughput'])
            if avg_throughput < 100:
                recommendations.append("Low throughput detected. Consider enabling mixed precision or optimizing Gaussian count")
        
        # GPU utilization recommendations
        if self.optimization_stats['gpu_utilization']:
            avg_gpu_util = np.mean(self.optimization_stats['gpu_utilization'])
            if avg_gpu_util < 50:
                recommendations.append("Low GPU utilization. Consider increasing batch size or enabling more parallelization")
        
        if not recommendations:
            recommendations.append("Performance looks good! Consider fine-tuning for your specific use case")
        
        return recommendations
    
    def export_profiling_data(self, output_path: Path):
        """Export profiling data for analysis"""
        if self.profiler:
            # Export Chrome trace
            self.profiler.export_chrome_trace(str(output_path / "trace.json"))
            
            # Export memory timeline
            self.profiler.export_memory_timeline(str(output_path / "memory_timeline.html"))
            
            # Save optimization statistics
            import json
            with open(output_path / "optimization_stats.json", "w") as f:
                json.dump(self.optimization_stats, f, indent=2, default=str)
            
            print(f"📊 Profiling data exported to {output_path}")

def create_optimized_trainer(config: GaussianFeelsConfig, dataset: BaseDataset,
                           optimization_level: str = "auto") -> OptimizedTrainer:
    """Create optimized trainer with automatic GPU-based optimization or preset levels"""
    
    if optimization_level == "auto":
        # Auto-detect optimal configuration based on GPU
        try:
            optimal_config = get_optimal_memory_config()
            optimization_level = optimal_config['optimization_level']
            print(f"🚀 Auto-detected {get_gpu_memory_gb():.1f}GB GPU - using '{optimization_level}' configuration")
        except Exception as e:
            print(f"⚠️ Could not auto-detect GPU, using balanced configuration: {e}")
            optimization_level = "balanced"
    
    if optimization_level == "memory_efficient":
        opt_config = OptimizationConfig(
            enable_gradient_checkpointing=True,
            max_memory_gb=None,  # Will auto-detect
            gaussian_batch_size=500,
            enable_mixed_precision=True,
            memory_cleanup_interval=50
        )
    elif optimization_level == "speed_optimized":
        opt_config = OptimizationConfig(
            enable_mixed_precision=True,
            enable_compile=True,
            enable_tf32=True,
            max_memory_gb=None,  # Will auto-detect
            gaussian_batch_size=2000,
            enable_fast_math=True,
            memory_cleanup_interval=200
        )
    elif optimization_level == "balanced":
        opt_config = OptimizationConfig(
            enable_mixed_precision=True,
            enable_tf32=True,
            max_memory_gb=None,  # Will auto-detect
            gaussian_batch_size=1000,
            memory_cleanup_interval=100,
            enable_profiling=False
        )
    elif optimization_level == "debug":
        opt_config = OptimizationConfig(
            enable_profiling=True,
            monitor_memory=True,
            monitor_gpu_utilization=True,
            profile_steps=100,
            memory_cleanup_interval=10,
            max_memory_gb=None  # Will auto-detect
        )
    else:
        opt_config = OptimizationConfig()
    
    print(f"🚀 Creating optimized trainer with '{optimization_level}' configuration")
    return OptimizedTrainer(config, dataset, opt_config)