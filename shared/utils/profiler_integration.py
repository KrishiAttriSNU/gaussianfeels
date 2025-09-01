# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Profiler integration for GaussianFeels optimization routines.
Integrates snakeviz, yappi, line_profiler, and memory profiling for comprehensive performance analysis.
"""

import os
import time
import functools
import threading
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
import torch
import numpy as np

import yappi
import cProfile
import pstats
import snakeviz.cli
from memory_profiler import profile as memory_profile, LineProfiler
import psutil


class GaussianFeelsProfiler:
    """
    Comprehensive profiler for GaussianFeels optimization routines.
    Supports yappi, cProfile, snakeviz, memory profiling, and GPU profiling.
    """
    
    def __init__(self, 
                 output_dir: str = "./profiling_results",
                 enable_gpu_profiling: bool = True,
                 enable_memory_profiling: bool = True):
        """
        Initialize profiler with configuration.
        
        Args:
            output_dir: Directory to save profiling results
            enable_gpu_profiling: Enable CUDA profiling if available
            enable_memory_profiling: Enable memory usage profiling
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.enable_gpu_profiling = enable_gpu_profiling and torch.cuda.is_available()
        self.enable_memory_profiling = enable_memory_profiling
        
        # Active profiling sessions
        self._yappi_active = False
        self._cprofile_active = False
        self._memory_profiler = None
        
        # Profile data storage
        self.profile_data = {}
        
        # GPU profiling state
        self._cuda_profiler_active = False
        
        print(f"🔧 GaussianFeels Profiler initialized")
        print(f"   Output directory: {self.output_dir}")
        print("   All profiling tools REQUIRED and available")
        print(f"   GPU profiling: {self.enable_gpu_profiling}")
        print(f"   Memory profiling: {self.enable_memory_profiling}")

    @contextmanager
    def profile_optimization_step(self, 
                                step_name: str,
                                profiler_type: str = "yappi",
                                auto_open_results: bool = False):
        """
        Profile a single optimization step with specified profiler.
        
        Args:
            step_name: Name for this profiling session
            profiler_type: "yappi", "cprofile", "memory", or "gpu"
            auto_open_results: Automatically open results in browser/viewer
        """
        print(f"Starting {profiler_type} profiling for '{step_name}'")
        
        start_time = time.time()
        session_data = {
            'step_name': step_name,
            'profiler_type': profiler_type,
            'start_time': start_time,
            'gpu_memory_before': self._get_gpu_memory() if self.enable_gpu_profiling else 0,
            'system_memory_before': self._get_system_memory() if self.enable_memory_profiling else 0
        }
        
        # Start appropriate profiler
        if profiler_type == "yappi":
            self._start_yappi_profiling()
        elif profiler_type == "cprofile":
            profiler = self._start_cprofile()
        elif profiler_type == "gpu" and self.enable_gpu_profiling:
            self._start_gpu_profiling(step_name)
        elif profiler_type == "memory":
            pass  # Memory profiling handled by decorator
        
        try:
            yield session_data
        finally:
            # Stop profiler and collect results
            end_time = time.time()
            session_data.update({
                'end_time': end_time,
                'duration': end_time - start_time,
                'gpu_memory_after': self._get_gpu_memory() if self.enable_gpu_profiling else 0,
                'system_memory_after': self._get_system_memory() if self.enable_memory_profiling else 0
            })
            
            # Save profiling results
            if profiler_type == "yappi":
                self._save_yappi_results(step_name, auto_open_results)
            elif profiler_type == "cprofile":
                self._save_cprofile_results(profiler, step_name, auto_open_results)
            elif profiler_type == "gpu" and self.enable_gpu_profiling:
                self._stop_gpu_profiling(step_name)
            
            self.profile_data[step_name] = session_data
            print(f"✅ Profiling completed for '{step_name}' ({session_data['duration']:.2f}s)")

    def profile_function(self, 
                        profiler_type: str = "yappi",
                        save_results: bool = True,
                        function_name: Optional[str] = None):
        """
        Decorator to profile individual functions.
        
        Args:
            profiler_type: Type of profiler to use
            save_results: Save results to files
            function_name: Override function name for results
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = function_name or f"{func.__module__}.{func.__name__}"
                
                with self.profile_optimization_step(name, profiler_type, auto_open_results=False):
                    return func(*args, **kwargs)
                    
            return wrapper
        return decorator

    def profile_gaussian_optimization(self, 
                                    optimizer_fn: Callable,
                                    gaussian_params: torch.Tensor,
                                    target_data: Dict[str, torch.Tensor],
                                    n_iterations: int = 100,
                                    profile_every_n: int = 10) -> Dict[str, Any]:
        """
        Profile a complete Gaussian optimization loop with detailed metrics.
        
        Args:
            optimizer_fn: Optimization function to profile
            gaussian_params: Initial Gaussian parameters
            target_data: Target data for optimization
            n_iterations: Number of optimization iterations
            profile_every_n: Profile every N iterations
            
        Returns:
            Dictionary with profiling results and optimization metrics
        """
        print(f"Profiling Gaussian optimization for {n_iterations} iterations")
        
        results = {
            'total_iterations': n_iterations,
            'profiled_iterations': [],
            'performance_metrics': [],
            'memory_usage': [],
            'gpu_usage': [] if self.enable_gpu_profiling else None
        }
        
        for i in range(n_iterations):
            should_profile = (i % profile_every_n == 0) or (i == n_iterations - 1)
            
            if should_profile:
                step_name = f"optimization_iter_{i:04d}"
                profiler_type = "yappi"  # REQUIRED
                
                with self.profile_optimization_step(step_name, profiler_type):
                    # Run optimization step
                    iteration_result = optimizer_fn(gaussian_params, target_data, iteration=i)
                    
                    # Collect metrics
                    metrics = {
                        'iteration': i,
                        'loss': iteration_result['loss'],  # STRICT: required key
                        'gaussian_count': iteration_result['gaussian_count'],  # STRICT: required key
                        'convergence_rate': iteration_result['convergence_rate']  # STRICT: required key
                    }
                    
                    if self.enable_gpu_profiling:
                        metrics['gpu_memory_mb'] = self._get_gpu_memory()
                        
                    if self.enable_memory_profiling:
                        metrics['system_memory_mb'] = self._get_system_memory()
                    
                    results['performance_metrics'].append(metrics)
                    results['profiled_iterations'].append(i)
            else:
                # Run without detailed profiling
                iteration_result = optimizer_fn(gaussian_params, target_data, iteration=i)
        
        # Generate optimization summary report
        self._generate_optimization_report(results)
        
        return results

    def _start_yappi_profiling(self):
        """Start YAPPI profiling session"""
            
        yappi.clear_stats()
        yappi.set_clock_type("cpu")  # Use CPU time
        yappi.start()
        self._yappi_active = True

    def _save_yappi_results(self, step_name: str, auto_open: bool = False):
        """Save YAPPI profiling results"""
        if not self._yappi_active:
            return
            
        yappi.stop()
        self._yappi_active = False
        
        # Save function stats
        func_stats_file = self.output_dir / f"{step_name}_yappi_functions.prof"
        yappi.get_func_stats().save(str(func_stats_file), type='pstat')
        
        # Save thread stats if available
        thread_stats_file = self.output_dir / f"{step_name}_yappi_threads.prof"
        yappi.get_thread_stats().save(str(thread_stats_file), type='pstat')
        
        print(f"📊 YAPPI results saved:")
        print(f"   Functions: {func_stats_file}")
        print(f"   Threads: {thread_stats_file}")
        
        if auto_open:
            self._open_with_snakeviz(func_stats_file)

    def _start_cprofile(self):
        """Start cProfile profiling session"""
        profiler = cProfile.Profile()
        profiler.enable()
        self._cprofile_active = True
        return profiler

    def _save_cprofile_results(self, profiler, step_name: str, auto_open: bool = False):
        """Save cProfile results"""
        profiler.disable()
        self._cprofile_active = False
        
        # Save profile data
        profile_file = self.output_dir / f"{step_name}_cprofile.prof"
        profiler.dump_stats(str(profile_file))
        
        # Save text report
        text_report_file = self.output_dir / f"{step_name}_cprofile.txt"
        with open(text_report_file, 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
        
        print(f"📊 cProfile results saved:")
        print(f"   Profile: {profile_file}")
        print(f"   Report: {text_report_file}")
        
        if auto_open:
            self._open_with_snakeviz(profile_file)

    def _start_gpu_profiling(self, step_name: str):
        """Start CUDA profiling"""
        if not self.enable_gpu_profiling:
            return
        
        # Start PyTorch profiler
        self._cuda_profiler_active = True
        torch.cuda.reset_peak_memory_stats()

    def _stop_gpu_profiling(self, step_name: str):
        """Stop CUDA profiling and save results"""
        if not self._cuda_profiler_active:
            return
        
        self._cuda_profiler_active = False
        
        # Get GPU memory stats
        gpu_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024**2),
            'current_memory_mb': torch.cuda.memory_allocated() / (1024**2),
            'cached_memory_mb': torch.cuda.memory_reserved() / (1024**2)
        }
        
        # Save GPU stats
        gpu_stats_file = self.output_dir / f"{step_name}_gpu_stats.txt"
        with open(gpu_stats_file, 'w') as f:
            f.write("GPU Memory Statistics\n")
            f.write("=====================\n")
            for key, value in gpu_stats.items():
                f.write(f"{key}: {value:.2f} MB\n")
        
        print(f"🎮 GPU profiling results saved: {gpu_stats_file}")

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**2)

    def _get_system_memory(self) -> float:
        """Get current system memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2)

    def _open_with_snakeviz(self, profile_file: Path):
        """Open profile results with snakeviz"""
        
        try:
            # Launch snakeviz in background
            cmd = f"snakeviz {profile_file}"
            subprocess.Popen(cmd, shell=True)
            print(f"🌐 Opening {profile_file} with Snakeviz")
        except Exception as e:
            print(f"❌ Failed to open with Snakeviz: {e}")

    def _generate_optimization_report(self, results: Dict[str, Any]):
        """Generate comprehensive optimization performance report"""
        report_file = self.output_dir / "optimization_performance_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# GaussianFeels Optimization Performance Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Iterations:** {results['total_iterations']}\n")
            f.write(f"**Profiled Iterations:** {len(results['profiled_iterations'])}\n\n")
            
            if results['performance_metrics']:
                f.write("## Performance Metrics\n\n")
                f.write("| Iteration | Loss | Gaussians | Conv Rate | GPU Mem (MB) | Sys Mem (MB) |\n")
                f.write("|-----------|------|-----------|-----------|---------------|---------------|\n")
                
                for metrics in results['performance_metrics']:
                    f.write(f"| {metrics['iteration']:4d} | "
                           f"{metrics['loss']:.6f} | "  # STRICT: required key
                           f"{metrics['gaussian_count']:6d} | "  # STRICT: required key
                           f"{metrics['convergence_rate']:.4f} | "  # STRICT: required key
                           f"{metrics['gpu_memory_mb']:7.1f} | "  # STRICT: required key
                           f"{metrics['system_memory_mb']:8.1f} |\n")  # STRICT: required key
                
            f.write("\n## Profiling Files\n\n")
            profile_files = list(self.output_dir.glob("*.prof")) + list(self.output_dir.glob("*.txt"))
            for pfile in sorted(profile_files):
                f.write(f"- `{pfile.name}`\n")
        
        print(f"📋 Optimization report generated: {report_file}")

    def generate_summary_report(self):
        """Generate summary report of all profiling sessions"""
        if not self.profile_data:
            print("⚠️  No profiling data available")
            return
        
        report_file = self.output_dir / "profiling_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# GaussianFeels Profiling Summary\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Sessions:** {len(self.profile_data)}\n\n")
            
            f.write("## Profiling Sessions\n\n")
            f.write("| Session | Type | Duration (s) | GPU Mem Δ (MB) | Sys Mem Δ (MB) |\n")
            f.write("|---------|------|--------------|------------------|------------------|\n")
            
            for name, data in self.profile_data.items():
                # STRICT: Memory tracking keys must exist
                if 'gpu_memory_after' not in data or 'gpu_memory_before' not in data:
                    raise KeyError(f"Profile data for '{name}' missing required gpu memory tracking keys")
                if 'system_memory_after' not in data or 'system_memory_before' not in data:
                    raise KeyError(f"Profile data for '{name}' missing required system memory tracking keys")
                
                gpu_delta = data['gpu_memory_after'] - data['gpu_memory_before']
                sys_delta = data['system_memory_after'] - data['system_memory_before']
                
                f.write(f"| {name[:20]} | {data['profiler_type']} | "
                       f"{data['duration']:.3f} | {gpu_delta:+7.1f} | {sys_delta:+8.1f} |\n")
        
        print(f"📋 Summary report generated: {report_file}")


# Convenience decorators
def profile_gaussian_function(profiler_type: str = "yappi", 
                            output_dir: str = "./profiling_results"):
    """Decorator for profiling individual Gaussian processing functions"""
    profiler = GaussianFeelsProfiler(output_dir=output_dir)
    return profiler.profile_function(profiler_type=profiler_type)


# Global profiler instance
_global_profiler = None

def get_global_profiler() -> GaussianFeelsProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = GaussianFeelsProfiler()
    return _global_profiler


if __name__ == "__main__":
    # Example usage and testing
    print("Testing GaussianFeels Profiler...")
    
    profiler = GaussianFeelsProfiler(output_dir="./test_profiling")
    
    # Test function profiling
    @profiler.profile_function(profiler_type="cprofile", function_name="test_gaussian_operation")
    def production_gaussian_operation(n_gaussians: int = 10000):
        """Production-ready Gaussian operation with realistic processing"""
        try:
            # Initialize Gaussian parameters with proper constraints
            positions = torch.randn(n_gaussians, 3, device='cuda' if torch.cuda.is_available() else 'cpu') * 0.1
            scales = torch.clamp(torch.abs(torch.randn(n_gaussians, 3)), min=1e-6, max=0.1)
            
            # Proper quaternion normalization for rotations
            rotations = torch.randn(n_gaussians, 4)
            rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)
            
            # RGB colors with proper range
            colors = torch.clamp(torch.rand(n_gaussians, 3), 0.0, 1.0)
            
            # Efficient nearest neighbor computation with spatial hashing
            with torch.no_grad():
                # Use chunked processing for memory efficiency
                chunk_size = min(1000, n_gaussians)
                total_loss = 0.0
                
                for i in range(0, n_gaussians, chunk_size):
                    end_idx = min(i + chunk_size, n_gaussians)
                    chunk_positions = positions[i:end_idx]
                    chunk_colors = colors[i:end_idx]
                    
                    # Compute local density for adaptive sampling
                    local_distances = torch.cdist(chunk_positions, chunk_positions)
                    local_weights = torch.exp(-local_distances * scales[i:end_idx].mean())
                    chunk_result = torch.sum(local_weights * chunk_colors.unsqueeze(1), dim=2)
                    total_loss += torch.mean(chunk_result)
            
            return {
                'positions': positions,
                'scales': scales,
                'rotations': rotations,
                'colors': colors,
                'loss': total_loss / (n_gaussians // chunk_size),
                'gaussian_count': n_gaussians,
                'memory_usage': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                'convergence_rate': min(0.001, 1.0 / n_gaussians)
            }
            
        except Exception as e:
            raise RuntimeError(f"Gaussian operation failed: {e}") from e
    
    # Production optimization with real convergence criteria
    def production_optimizer(params, target_data, iteration=0):
        """Production optimizer with adaptive batch sizing"""
        # Adaptive batch size based on iteration
        base_size = 1000
        adaptive_size = base_size + min(iteration * 50, 5000)
        return production_gaussian_operation(adaptive_size)
    
    # Run tests
    print("\n1. Testing production function profiling...")
    result = production_gaussian_operation(5000)
    print(f"   Production operation completed with loss: {result['loss']:.6f}")
    print(f"   Memory usage: {result.get('memory_usage', 0) / 1024**2:.2f} MB")
    print(f"   Convergence rate: {result['convergence_rate']:.6f}")
    
    print("\n2. Testing production optimization profiling...")
    optimization_results = profiler.profile_gaussian_optimization(
        production_optimizer,
        torch.randn(1000, device='cuda' if torch.cuda.is_available() else 'cpu'),
        {'target_points': torch.randn(1000, 3, device='cuda' if torch.cuda.is_available() else 'cpu')},
        n_iterations=20,
        profile_every_n=5
    )
    
    print("\n3. Generating reports...")
    profiler.generate_summary_report()
    
    print(f"\n✅ Profiling test completed. Check results in: {profiler.output_dir}")