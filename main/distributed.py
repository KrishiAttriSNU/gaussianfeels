"""
GaussianFeels Distributed Training

Distributed training support for multi-GPU and multi-node Gaussian splatting.
Supports PyTorch DDP, gradient synchronization, and efficient communication.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import torch.distributed.launcher as launcher
    LAUNCHER_AVAILABLE = True
except ImportError:
    LAUNCHER_AVAILABLE = False

from .config import GaussianFeelsConfig
from .trainer import GaussianTrainer, GaussianField
from .datasets import BaseDataset

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Backend settings
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"  # or tcp://host:port
    
    # Multi-node settings
    world_size: int = 1  # Total number of processes
    rank: int = 0  # Global rank of this process
    local_rank: int = 0  # Local rank on this node
    
    # Communication settings
    timeout_seconds: int = 1800  # 30 minutes
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    
    # Optimization settings
    sync_batch_norm: bool = True
    broadcast_buffers: bool = True
    gradient_reduction: str = "mean"  # mean or sum
    
    # Memory optimization
    cpu_offload: bool = False
    use_zero: bool = False  # ZeRO optimization (future)
    
    # Debugging
    debug_mode: bool = False
    profile_communication: bool = False

class DistributedGaussianField(nn.Module):
    """Distributed wrapper for GaussianField"""
    
    def __init__(self, gaussian_field: GaussianField, config: DistributedConfig):
        super().__init__()
        self.gaussian_field = gaussian_field
        self.config = config
        self.device = gaussian_field.device
        
        # Wrap with DDP
        self.ddp_field = DDP(
            gaussian_field,
            device_ids=[config.local_rank] if torch.cuda.is_available() else None,
            output_device=config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            broadcast_buffers=config.broadcast_buffers
        )
        
        # Synchronization state
        self.sync_needed = False
        self.last_sync_step = 0
        
    def forward(self, *args, **kwargs):
        """Forward pass through DDP wrapper"""
        return self.ddp_field(*args, **kwargs)
    
    @property
    def module(self):
        """Access underlying module"""
        return self.ddp_field.module
    
    def get_gaussians(self):
        """Get Gaussians from underlying field"""
        return self.module.get_gaussians()
    
    def densify(self, positions: torch.Tensor):
        """Densify operation with distributed synchronization"""
        # Perform local densification
        self.module.densify(positions)
        
        # Mark for synchronization
        self.sync_needed = True
        
        # Broadcast new parameters to all ranks
        self._sync_parameters()
    
    def prune(self, opacity_threshold: float = 0.005):
        """Prune operation with distributed synchronization"""
        # Perform local pruning
        self.module.prune(opacity_threshold)
        
        # Mark for synchronization
        self.sync_needed = True
        
        # Broadcast updated parameters
        self._sync_parameters()
    
    def _sync_parameters(self):
        """Synchronize parameters across all ranks"""
        if not dist.is_initialized():
            return
        
        # Only sync if rank 0 (prevent multiple syncs)
        if dist.get_rank() == 0:
            # Broadcast parameter sizes and values
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        else:
            # Receive parameters from rank 0
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        
        self.sync_needed = False
        self.last_sync_step = getattr(self, 'current_step', 0)

class DistributedTrainer(GaussianTrainer):
    """Distributed version of GaussianTrainer"""
    
    def __init__(self, config: GaussianFeelsConfig, dataset: BaseDataset, 
                 distributed_config: DistributedConfig):
        self.distributed_config = distributed_config
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Initialize base trainer
        super().__init__(config, dataset)
        
        # Wrap Gaussian field for distributed training
        self.distributed_field = DistributedGaussianField(
            self.gaussian_field, distributed_config
        )
        
        # Setup distributed optimizers
        self._setup_distributed_optimizers()
        
        # Distributed state
        self.world_size = distributed_config.world_size
        self.rank = distributed_config.rank
        self.local_rank = distributed_config.local_rank
        
        # Communication tracking
        self.communication_time = 0.0
        self.sync_count = 0
        
        # Load balancing
        self.local_batch_size = max(1, len(dataset) // distributed_config.world_size)
        self.local_start_idx = self.rank * self.local_batch_size
        
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if dist.is_initialized():
            print(f"🔗 Distributed already initialized (rank {dist.get_rank()})")
            return
        
        try:
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.distributed_config.local_rank)
                device = f"cuda:{self.distributed_config.local_rank}"
            else:
                device = "cpu"
                print("⚠️ Using CPU for distributed training (not recommended)")
            
            # Initialize process group
            dist.init_process_group(
                backend=self.distributed_config.backend,
                init_method=self.distributed_config.init_method,
                world_size=self.distributed_config.world_size,
                rank=self.distributed_config.rank,
                timeout=torch.distributed.default_pg_timeout
            )
            
            print(f"✅ Initialized distributed training:")
            print(f"  Backend: {self.distributed_config.backend}")
            print(f"  World size: {self.distributed_config.world_size}")
            print(f"  Rank: {self.distributed_config.rank}/{self.distributed_config.world_size-1}")
            print(f"  Local rank: {self.distributed_config.local_rank}")
            print(f"  Device: {device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize distributed training: {e}")
            raise
    
    def _setup_distributed_optimizers(self):
        """Setup optimizers with distributed considerations"""
        # Scale learning rates by world size for gradient averaging
        if self.distributed_config.gradient_reduction == "mean":
            lr_scale = 1.0  # DDP automatically averages gradients
        else:
            lr_scale = self.world_size  # For gradient summing
        
        # Update learning rates
        for name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_scale
        
        print(f"📊 Scaled learning rates by {lr_scale} for distributed training")
    
    def step_map(self) -> float:
        """Distributed map optimization step"""
        start_time = time.time()
        
        # Calculate frame index for this rank
        frame_idx = (self.local_start_idx + self.current_frame_idx) % len(self.dataset)
        frame = self.dataset[frame_idx]
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass through distributed field
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            loss = self._compute_losses(frame)
        
        # Backward pass with gradient synchronization
        comm_start = time.time()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Synchronize gradients across ranks
            self._sync_gradients()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                self._clip_gradients_distributed()
            
            # Update parameters
            for optimizer in self.optimizers.values():
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Synchronize gradients
            self._sync_gradients()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                self._clip_gradients_distributed()
            
            # Update parameters
            for optimizer in self.optimizers.values():
                optimizer.step()
        
        self.communication_time += time.time() - comm_start
        
        # Gaussian field maintenance (coordinated across ranks)
        if self.step % self.config.gaussian_params.densify_interval == 0:
            self._densify_and_prune_distributed()
        
        self.step += 1
        self.current_frame_idx += 1
        
        map_loss = loss.item()
        self.map_losses.append(map_loss)
        
        step_time = time.time() - start_time
        self.timings["map_step"].append(step_time)
        
        return map_loss
    
    def _sync_gradients(self):
        """Synchronize gradients across all ranks"""
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        # DDP automatically handles gradient synchronization for model parameters
        # This method handles additional custom synchronization for Gaussian-specific needs
        
        # Synchronize Gaussian field gradients if not using DDP
        if hasattr(self, 'gaussian_field') and not isinstance(self.gaussian_field, torch.nn.parallel.DistributedDataParallel):
            # Manually synchronize gradients for Gaussian parameters
            for param_name in ['positions', 'colors', 'opacities', 'scales', 'rotations']:
                param = getattr(self.gaussian_field, f'_{param_name}', None)
                if param is not None and param.grad is not None:
                    # All-reduce gradients across all processes
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    # Average gradients
                    param.grad.data /= self.world_size
        
        # Synchronize any custom gradient accumulators
        if hasattr(self, 'gradient_accumulators'):
            for key, accumulator in self.gradient_accumulators.items():
                if accumulator is not None:
                    dist.all_reduce(accumulator, op=dist.ReduceOp.SUM)
                    accumulator /= self.world_size
    
    def _clip_gradients_distributed(self):
        """Distributed gradient clipping"""
        # Collect all parameters
        all_params = []
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                all_params.extend(group['params'])
        
        # Compute local gradient norm
        local_grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params, float('inf'), norm_type=2
        )
        
        # Synchronize gradient norms across ranks
        if dist.is_initialized() and self.world_size > 1:
            grad_norm_tensor = torch.tensor([local_grad_norm], device=self.device)
            dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.MAX)
            global_grad_norm = grad_norm_tensor.item()
        else:
            global_grad_norm = local_grad_norm
        
        # Apply clipping with global norm
        if global_grad_norm > self.config.training.gradient_clip:
            clip_ratio = self.config.training.gradient_clip / global_grad_norm
            for param in all_params:
                if param.grad is not None:
                    param.grad.mul_(clip_ratio)
    
    def _densify_and_prune_distributed(self):
        """Coordinated densification and pruning across ranks"""
        # Only rank 0 decides on densification/pruning
        if self.rank == 0:
            # Collect statistics from all ranks
            local_stats = self._collect_local_stats()
            
            # Make densification decisions
            densify_positions = self._decide_densification(local_stats)
            prune_threshold = self._decide_pruning(local_stats)
            
            # Broadcast decisions to all ranks
            self._broadcast_field_operations(densify_positions, prune_threshold)
        else:
            # Wait for instructions from rank 0
            self._receive_field_operations()
    
    def _collect_local_stats(self) -> Dict[str, Any]:
        """Collect local Gaussian field statistics"""
        gaussians = self.distributed_field.get_gaussians()
        
        if not gaussians:
            return {
                'num_gaussians': 0,
                'avg_opacity': 0.0,
                'max_gradient': 0.0,
                'positions': torch.tensor([])
            }
        
        # Extract statistics
        opacities = torch.stack([g["opacity"] for g in gaussians])
        positions = torch.stack([g["position"] for g in gaussians])
        
        # Gradient statistics
        max_grad = 0.0
        if hasattr(self.distributed_field.module.positions, 'grad') and \
           self.distributed_field.module.positions.grad is not None:
            grad_norms = torch.norm(self.distributed_field.module.positions.grad, dim=1)
            max_grad = torch.max(grad_norms).item()
        
        return {
            'num_gaussians': len(gaussians),
            'avg_opacity': torch.mean(torch.sigmoid(opacities)).item(),
            'max_gradient': max_grad,
            'positions': positions.detach()
        }
    
    def _decide_densification(self, stats: Dict[str, Any]) -> torch.Tensor:
        """Decide on densification based on global statistics"""
        # Simple densification strategy
        if stats['max_gradient'] > self.config.gaussian_params.split_threshold:
            new_positions = torch.randn(100, 3, device=self.device) * 0.01
            return new_positions
        else:
            return torch.tensor([], device=self.device).reshape(0, 3)
    
    def _decide_pruning(self, stats: Dict[str, Any]) -> float:
        """Decide on pruning threshold"""
        if stats['avg_opacity'] < 0.1:
            return self.config.gaussian_params.prune_threshold * 2
        else:
            return self.config.gaussian_params.prune_threshold
    
    def _broadcast_field_operations(self, densify_positions: torch.Tensor, 
                                   prune_threshold: float):
        """Broadcast field operations to all ranks"""
        if not dist.is_initialized():
            return
        
        # Broadcast densification
        if densify_positions.numel() > 0:
            # Broadcast number of new positions
            n_new = torch.tensor([densify_positions.shape[0]], device=self.device)
            dist.broadcast(n_new, src=0)
            
            # Broadcast positions
            dist.broadcast(densify_positions, src=0)
            
            # Apply densification
            self.distributed_field.densify(densify_positions)
        else:
            # Broadcast zero count
            n_new = torch.tensor([0], device=self.device)
            dist.broadcast(n_new, src=0)
        
        # Broadcast pruning threshold
        prune_tensor = torch.tensor([prune_threshold], device=self.device)
        dist.broadcast(prune_tensor, src=0)
        
        # Apply pruning
        self.distributed_field.prune(prune_tensor.item())
    
    def _receive_field_operations(self):
        """Receive and apply field operations from rank 0"""
        if not dist.is_initialized():
            return
        
        # Receive densification count
        n_new = torch.tensor([0], device=self.device)
        dist.broadcast(n_new, src=0)
        
        # Receive positions if any
        if n_new.item() > 0:
            new_positions = torch.zeros(n_new.item(), 3, device=self.device)
            dist.broadcast(new_positions, src=0)
            self.distributed_field.densify(new_positions)
        
        # Receive pruning threshold
        prune_tensor = torch.tensor([0.0], device=self.device)
        dist.broadcast(prune_tensor, src=0)
        
        # Apply pruning
        self.distributed_field.prune(prune_tensor.item())
    
    def save_checkpoint(self, path: Path):
        """Save distributed checkpoint (only from rank 0)"""
        if self.rank == 0:
            checkpoint = {
                "step": self.step,
                "gaussian_field": self.distributed_field.module.state_dict(),
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
                "config": self.config,
                "distributed_config": self.distributed_config,
                "pose_losses": self.pose_losses,
                "map_losses": self.map_losses,
                "communication_time": self.communication_time,
                "sync_count": self.sync_count,
            }
            
            torch.save(checkpoint, path)
            print(f"💾 Checkpoint saved by rank 0: {path}")
        
        # Synchronize to ensure all ranks wait for save completion
        if dist.is_initialized():
            dist.barrier()
    
    def load_checkpoint(self, path: Path):
        """Load distributed checkpoint"""
        # Load on all ranks
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint["step"]
        self.distributed_field.module.load_state_dict(checkpoint["gaussian_field"])
        
        for name, opt_state in checkpoint["optimizers"].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
        
        self.pose_losses = checkpoint.get("pose_losses", [])
        self.map_losses = checkpoint.get("map_losses", [])
        self.communication_time = checkpoint.get("communication_time", 0.0)
        self.sync_count = checkpoint.get("sync_count", 0)
        
        print(f"📂 Checkpoint loaded on rank {self.rank}: {path}")
    
    def get_distributed_metrics(self) -> Dict[str, Any]:
        """Get distributed training metrics"""
        base_metrics = self.get_performance_metrics()
        
        # Communication efficiency
        total_time = sum(self.timings["map_step"][-100:]) if self.timings["map_step"] else 1.0
        comm_ratio = self.communication_time / total_time if total_time > 0 else 0.0
        
        # Throughput scaling
        single_gpu_estimate = base_metrics.get("recent_fps", 0.0)
        scaling_efficiency = (single_gpu_estimate * self.world_size) / max(1.0, single_gpu_estimate) \
                           if single_gpu_estimate > 0 else 0.0
        
        distributed_metrics = {
            # Distributed state
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "backend": self.distributed_config.backend,
            
            # Communication metrics
            "communication_time": self.communication_time,
            "communication_ratio": comm_ratio,
            "sync_count": self.sync_count,
            "avg_sync_time": self.communication_time / max(1, self.sync_count),
            
            # Efficiency metrics
            "scaling_efficiency": scaling_efficiency,
            "local_batch_size": self.local_batch_size,
            "global_throughput": single_gpu_estimate * self.world_size,
            
            # Load balancing
            "frames_per_rank": len(self.dataset) // self.world_size,
            "local_frame_range": (self.local_start_idx, 
                                self.local_start_idx + self.local_batch_size),
        }
        
        # Merge with base metrics
        base_metrics.update(distributed_metrics)
        return base_metrics
    
    @staticmethod
    def cleanup():
        """Clean up distributed resources"""
        if dist.is_initialized():
            dist.destroy_process_group()
            print("🧹 Distributed resources cleaned up")

def setup_distributed_training(config: GaussianFeelsConfig, dataset: BaseDataset,
                              world_size: int = None, rank: int = None,
                              local_rank: int = None) -> DistributedTrainer:
    """Setup distributed training with automatic environment detection"""
    
    # Auto-detect distributed environment
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Create distributed config
    distributed_config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )
    
    print(f"🚀 Setting up distributed training:")
    print(f"  World size: {world_size}")
    print(f"  Rank: {rank}")
    print(f"  Local rank: {local_rank}")
    
    # Create distributed trainer
    trainer = DistributedTrainer(config, dataset, distributed_config)
    
    return trainer

def launch_distributed_training(config: GaussianFeelsConfig, dataset: BaseDataset,
                               num_gpus: int = None, 
                               nodes: int = 1,
                               node_rank: int = 0,
                               master_addr: str = "localhost",
                               master_port: str = "12355") -> None:
    """Launch distributed training with torchrun-style launcher"""
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        print("⚠️ Single GPU detected, using regular training")
        trainer = GaussianTrainer(config, dataset)
        return trainer
    
    # Set environment variables for distributed training
    os.environ.update({
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": master_port,
        "WORLD_SIZE": str(num_gpus * nodes),
        "NNODES": str(nodes),
        "NODE_RANK": str(node_rank),
        "NPROC_PER_NODE": str(num_gpus)
    })
    
    print(f"🌍 Launching distributed training:")
    print(f"  Nodes: {nodes}")
    print(f"  GPUs per node: {num_gpus}")
    print(f"  Total world size: {num_gpus * nodes}")
    print(f"  Master: {master_addr}:{master_port}")
    
    # Note: In a real implementation, this would use torch.multiprocessing.spawn
    # or torchrun to launch multiple processes. For now, we setup for single process.
    trainer = setup_distributed_training(config, dataset)
    
    return trainer

# Example usage and utilities
def create_distributed_config_from_env() -> DistributedConfig:
    """Create distributed config from environment variables"""
    return DistributedConfig(
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        rank=int(os.environ.get("RANK", 0)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        backend=os.environ.get("DIST_BACKEND", "nccl"),
        init_method=os.environ.get("DIST_INIT_METHOD", "env://"),
    )

def is_distributed_training() -> bool:
    """Check if we're in a distributed training environment"""
    return dist.is_initialized() or "WORLD_SIZE" in os.environ

def get_world_size() -> int:
    """Get world size safely"""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))

def get_rank() -> int:
    """Get rank safely"""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))

def barrier():
    """Synchronization barrier"""
    if dist.is_initialized():
        dist.barrier()