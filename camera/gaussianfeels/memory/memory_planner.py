# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory budget planning and allocation for GaussianFeels.
Implements 2-3x memory multiplier for optimizer state and rasterization buffers.
"""

import torch
import psutil
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MemoryMode(Enum):
    """Memory management modes - AGGRESSIVE ONLY"""
    AGGRESSIVE = "aggressive"     # 3x multiplier (only mode allowed)


@dataclass
class MemoryBudget:
    """Memory budget allocation for different components"""
    base_gaussians: int  # Base Gaussian parameters
    optimizer_state: int  # Adam optimizer state (2x params)
    gradients: int       # Gradient buffers
    rasterization: int   # Rasterization buffers
    cache: int           # Various caches
    overhead: int        # System overhead
    total: int           # Total budget
    
    def to_mb(self) -> Dict[str, float]:
        """Convert bytes to MB for display"""
        return {
            'base_gaussians': self.base_gaussians / (1024**2),
            'optimizer_state': self.optimizer_state / (1024**2),
            'gradients': self.gradients / (1024**2),
            'rasterization': self.rasterization / (1024**2),
            'cache': self.cache / (1024**2),
            'overhead': self.overhead / (1024**2),
            'total': self.total / (1024**2)
        }


class MemoryPlanner:
    """Plans and tracks memory allocation for GaussianFeels"""
    
    def __init__(self, target_gaussians: int = 300000, mode: MemoryMode = MemoryMode.AGGRESSIVE):
        self.target_gaussians = target_gaussians
        self.mode = mode
        self.base_memory_per_gaussian = self._calculate_base_memory()
        
    def _calculate_base_memory(self) -> int:
        """Calculate base memory per Gaussian (mixed precision)"""
        # Position: 3 * FP32 = 12 bytes
        # Rotation: 4 * FP32 = 16 bytes  
        # Color: 3 * FP16 = 6 bytes
        # Opacity: 1 * FP16 = 2 bytes
        # Scale: 3 * FP32 = 12 bytes
        # Total: 48 bytes per Gaussian
        return 48
        
    def calculate_budget(self, available_vram_gb: float = 8.0) -> MemoryBudget:
        """Calculate memory budget based on available VRAM"""
        available_bytes = int(available_vram_gb * 1024**3)
        
        # Base Gaussian memory
        base_gaussians = self.target_gaussians * self.base_memory_per_gaussian
        
        # Optimizer state (Adam: 2x parameters for momentum and velocity)
        optimizer_state = base_gaussians * 2
        
        # Gradient buffers (same size as parameters)
        gradients = base_gaussians
        
        # Rasterization buffers (depth, alpha, color accumulation)
        # Assume 1920x1080 rendering target
        H, W = 1080, 1920
        rasterization = H * W * (4 + 4 + 12)  # depth + alpha + color (FP32)
        
        # Cache and working memory
        cache = base_gaussians // 4  # 25% of base for spatial hash, etc.
        
        # System overhead (10%)
        overhead = int((base_gaussians + optimizer_state + gradients + rasterization + cache) * 0.1)
        
        # Apply memory mode multiplier - AGGRESSIVE ONLY
        multipliers = {
            MemoryMode.AGGRESSIVE: 3.0  # Only aggressive mode allowed
        }
        
        base_total = base_gaussians + optimizer_state + gradients + rasterization + cache + overhead
        total_with_multiplier = int(base_total * multipliers[self.mode])
        
        # Ensure we don't exceed available VRAM (leave 1GB for system)
        max_usable = available_bytes - (1024**3)
        if total_with_multiplier > max_usable:
            # Scale down Gaussian count to fit
            scale_factor = max_usable / total_with_multiplier
            self.target_gaussians = int(self.target_gaussians * scale_factor)
            return self.calculate_budget(available_vram_gb)
            
        return MemoryBudget(
            base_gaussians=base_gaussians,
            optimizer_state=optimizer_state,
            gradients=gradients,
            rasterization=rasterization,
            cache=cache,
            overhead=overhead,
            total=total_with_multiplier
        )
    
    def get_lod_thresholds(self, budget: MemoryBudget) -> Dict[str, int]:
        """Calculate Level of Detail thresholds based on memory budget"""
        total_mb = budget.total / (1024**2)
        
        if total_mb < 200:
            return {
                'max_gaussians': 200000,
                'densification_threshold': 0.01,
                'pruning_threshold': 0.005,
                'opacity_threshold': 0.1
            }
        elif total_mb < 400:
            return {
                'max_gaussians': 300000,
                'densification_threshold': 0.008,
                'pruning_threshold': 0.003,
                'opacity_threshold': 0.05
            }
        else:
            return {
                'max_gaussians': 500000,
                'densification_threshold': 0.005,
                'pruning_threshold': 0.001,
                'opacity_threshold': 0.02
            }
    
    def check_memory_constraints(self, current_gaussians: int) -> Tuple[bool, str]:
        """Check if current memory usage is within constraints"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            
            budget = self.calculate_budget()
            
            if current_memory > budget.total:
                return False, f"Memory usage {current_memory/(1024**2):.1f}MB exceeds budget {budget.total/(1024**2):.1f}MB"
            
            if current_gaussians > self.target_gaussians * 1.2:  # 20% tolerance
                return False, f"Gaussian count {current_gaussians} exceeds target {self.target_gaussians}"
                
        return True, "Memory constraints satisfied"
    
    def suggest_optimization(self, current_memory_mb: float, target_memory_mb: float) -> str:
        """Suggest memory optimization strategies"""
        if current_memory_mb > target_memory_mb:
            excess_mb = current_memory_mb - target_memory_mb
            suggestions = []
            
            if excess_mb > 100:
                suggestions.append("Enable aggressive pruning (opacity < 0.02)")
                suggestions.append("Reduce densification threshold to 0.003")
                
            if excess_mb > 50:
                suggestions.append("Enable Level of Detail (LOD) management")
                suggestions.append("Increase spatial hash bucket size")
                
            suggestions.append("Consider reducing render resolution")
            suggestions.append("Enable gradient checkpointing")
            
            return "; ".join(suggestions)
        
        return "Memory usage within budget"


def get_system_memory_info() -> Dict[str, float]:
    """Get system memory information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_cached = torch.cuda.memory_reserved() / (1024**3)
    else:
        gpu_memory = gpu_allocated = gpu_cached = 0.0
        
    system_memory = psutil.virtual_memory()
    
    return {
        'gpu_total_gb': gpu_memory,
        'gpu_allocated_gb': gpu_allocated,
        'gpu_cached_gb': gpu_cached,
        'gpu_free_gb': gpu_memory - gpu_allocated,
        'system_total_gb': system_memory.total / (1024**3),
        'system_available_gb': system_memory.available / (1024**3),
        'system_used_gb': system_memory.used / (1024**3)
    }


if __name__ == "__main__":
    # Example usage
    planner = MemoryPlanner(target_gaussians=300000, mode=MemoryMode.BALANCED)
    budget = planner.calculate_budget(available_vram_gb=8.0)
    
    print("Memory Budget Plan:")
    for component, mb in budget.to_mb().items():
        print(f"  {component}: {mb:.1f} MB")
        
    lod_thresholds = planner.get_lod_thresholds(budget)
    print(f"\nLOD Thresholds: {lod_thresholds}")
    
    memory_info = get_system_memory_info()
    print(f"\nSystem Memory: {memory_info}")