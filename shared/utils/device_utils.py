"""
Device management utilities for GaussianFeels

Consolidated device setup patterns - REQUIRES GPU availability.
Automatically detects GPU memory capacity for optimal configuration.
"""

import torch
from typing import Union, Optional
from .error_handling import DeviceError


def get_device(device_spec: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get appropriate PyTorch device - REQUIRES CUDA GPU.
    
    Args:
        device_spec: Device specification ('cuda', device object, or None)
    
    Returns:
        torch.device: CUDA device
        
    Raises:
        DeviceError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise DeviceError(
            "CUDA GPU is required but not available. "
            "Please ensure you have a CUDA-capable GPU and proper PyTorch installation."
        )
    
    if device_spec is None:
        return torch.device('cuda')
    
    if isinstance(device_spec, torch.device):
        if device_spec.type == 'cpu':
            raise DeviceError("CPU device not supported. CUDA GPU required.")
        return device_spec
    
    if isinstance(device_spec, str):
        if device_spec == 'cpu':
            raise DeviceError("CPU device not supported. CUDA GPU required.")
        elif device_spec == 'cuda':
            return torch.device('cuda')
        else:
            return torch.device(device_spec)
    
    raise ValueError(f"Invalid device specification: {device_spec}")


def get_gpu_memory_gb() -> float:
    """
    Get total GPU memory in GB for automatic optimization configuration.
    
    Returns:
        float: Total GPU memory in GB
        
    Raises:
        DeviceError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise DeviceError("CUDA GPU is required but not available.")
    
    # Get total memory for the current device
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return total_memory_gb


def get_optimal_memory_config() -> dict:
    """
    Automatically determine optimal memory configuration based on GPU capacity.
    
    Returns:
        dict: Optimal memory configuration parameters
    """
    gpu_memory_gb = get_gpu_memory_gb()
    
    # Leave 10% headroom for system and other processes
    usable_memory_gb = gpu_memory_gb * 0.9
    
    if gpu_memory_gb >= 24:  # RTX 4090/5090, A100, etc.
        return {
            'max_memory_gb': usable_memory_gb,
            'gaussian_batch_size': 4000,
            'memory_cleanup_interval': 200,
            'optimization_level': 'speed_optimized'
        }
    elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3090, etc.
        return {
            'max_memory_gb': usable_memory_gb,
            'gaussian_batch_size': 2000,
            'memory_cleanup_interval': 150,
            'optimization_level': 'balanced'
        }
    elif gpu_memory_gb >= 10:  # RTX 4070, RTX 3080, etc.
        return {
            'max_memory_gb': usable_memory_gb,
            'gaussian_batch_size': 1000,
            'memory_cleanup_interval': 100,
            'optimization_level': 'balanced'
        }
    else:  # RTX 4060, RTX 3060, etc.
        return {
            'max_memory_gb': usable_memory_gb,
            'gaussian_batch_size': 500,
            'memory_cleanup_interval': 50,
            'optimization_level': 'memory_efficient'
        }


def setup_device(config: Optional[object] = None, device_attr: str = 'device') -> torch.device:
    """
    Setup device from configuration object - REQUIRES CUDA GPU.
    
    Args:
        config: Configuration object with device attribute
        device_attr: Name of device attribute in config
    
    Returns:
        torch.device: CUDA device
        
    Raises:
        DeviceError: If CUDA is not available
    """
    if config is None or not hasattr(config, device_attr):
        return get_device()
    
    device_spec = getattr(config, device_attr)
    return get_device(device_spec)
