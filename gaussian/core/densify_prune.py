"""
Densification and pruning for Gaussian splatting with error- and memory-aware policies.
Supports gradient-driven splitting/cloning, opacity/scale pruning, and adaptive thresholds.
Designed to work with training loops that use differentiable rendering.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Set, Any
from dataclasses import dataclass
import numpy as np
from ..spatial.spatial_hash import CUDASpatialHash, SpatialHashConfig

@dataclass
class DensifyPruneConfig:
    """Configuration for densification and pruning (Industry Standards: 30K Training)"""
    # Densification parameters (Gaussian Splatting industry standard)
    densify_grad_threshold: float = 0.0002  # Standard 3DGS gradient threshold
    densify_split_threshold: float = 0.005  # Scale threshold for splitting
    densify_clone_threshold: float = 0.01   # Distance threshold (improved from 0.02)
    densify_interval: int = 100  # Industry standard: Every 100 iterations
    densify_start_iter: int = 500   # Industry standard warm-up
    densify_stop_iter: int = 30000  # Industry standard: 30K iterations
    
    # Pruning parameters (Enhanced for quality)
    prune_opacity_threshold: float = 0.005  # Minimum opacity (industry standard)
    prune_scale_threshold: float = 0.01     # Maximum scale (industry standard)
    prune_interval: int = 100  # Aligned with densification interval
    
    # Memory management
    max_gaussians: int = 300_000  # Hard limit on Gaussians
    memory_pressure_threshold: float = 0.8  # Start aggressive pruning
    target_memory_mb: float = 4096  # Target memory usage in MB
    
    # Adaptive parameters
    enable_adaptive_thresholds: bool = True
    adaptation_rate: float = 0.1
    error_history_length: int = 100

class DensifyPruneManager:
    """
    Manages Gaussian densification and pruning based on reconstruction error.
    
    Key features:
    - Error-driven splitting and cloning
    - Memory-aware pruning
    - Adaptive threshold adjustment
    - Spatial coherence preservation
    """
    
    def __init__(self, config: DensifyPruneConfig):
        self.config = config
        
        # Tracking variables
        self.step_count = 0
        self.gradient_accum = {}  # Accumulated gradients for densification
        self.error_history = []   # Recent reconstruction errors
        self.last_densify_step = 0
        self.last_prune_step = 0
        self.last_opacity_reset = 0
        
        # 3DGS standard opacity reset interval
        self.opacity_reset_interval = 3000
        
        # Spatial indexing for efficient operations
        self.spatial_hash = None
        
        # Adaptive thresholds
        self.adaptive_densify_threshold = config.densify_grad_threshold
        self.adaptive_prune_threshold = config.prune_opacity_threshold
        
    def step(self, gaussian_map, loss_info: Dict[str, torch.Tensor], 
             optimizer_state: Optional[Dict] = None) -> Dict[str, int]:
        """
        Perform one step of densification/pruning management.
        
        Args:
            gaussian_map: ObjectGaussianMap instance
            loss_info: Dictionary containing loss components and gradients
            optimizer_state: Optional optimizer state for gradient tracking
            
        Returns:
            Dictionary with counts of operations performed
        """
        self.step_count += 1
        operations = {'densified': 0, 'pruned': 0, 'split': 0, 'cloned': 0}
        
        # Update error history for adaptation
        if 'total_loss' in loss_info:
            self._update_error_history(loss_info['total_loss'].item())
        
        # Accumulate gradients for densification decisions
        if optimizer_state and gaussian_map.num_active_gaussians > 0:
            self._accumulate_gradients(gaussian_map, optimizer_state)
        
        # Perform densification if due
        if self._should_densify():
            densify_ops = self._densify_gaussians(gaussian_map)
            operations.update(densify_ops)
            self.last_densify_step = self.step_count
            
        # Perform opacity reset if due (3DGS standard: every 3000 iterations)
        if self._should_reset_opacity():
            self._reset_opacity(gaussian_map)
            self.last_opacity_reset = self.step_count
            
        # Perform pruning if due
        if self._should_prune():
            prune_count = self._prune_gaussians(gaussian_map)
            operations['pruned'] = prune_count
            self.last_prune_step = self.step_count
            
        # Adaptive threshold adjustment
        if self.config.enable_adaptive_thresholds:
            self._adapt_thresholds(operations)
            
        # Update spatial indexing
        self._update_spatial_hash(gaussian_map)
        
        return operations
    
    def _should_densify(self) -> bool:
        """Check if densification should be performed based on 3DGS schedule"""
        return (self.config.densify_start_iter <= self.step_count <= self.config.densify_stop_iter 
                and self.step_count % self.config.densify_interval == 0)
    
    def _should_prune(self) -> bool:
        """Check if pruning should be performed based on 3DGS schedule"""
        return self.step_count % self.config.prune_interval == 0
    
    def _accumulate_gradients(self, gaussian_map, optimizer_state: Dict):
        """Accumulate gradients for densification decisions"""
        if gaussian_map._positions is None or not gaussian_map._positions.requires_grad:
            return
            
        # Get position gradients
        pos_grad = gaussian_map._positions.grad
        if pos_grad is None:
            return
            
        # Compute gradient magnitudes
        grad_magnitudes = torch.norm(pos_grad, dim=1)
        
        # Accumulate in gradient buffer
        active_mask = gaussian_map._active_mask
        if active_mask is not None:
            grad_magnitudes = grad_magnitudes[active_mask]
        
        # Store for densification decision
        if 'position_gradients' not in self.gradient_accum:
            self.gradient_accum['position_gradients'] = grad_magnitudes.clone()
            self.gradient_accum['gradient_counts'] = torch.ones_like(grad_magnitudes)
        else:
            # Check if tensor sizes match - if not, clear buffer and restart
            if self.gradient_accum['position_gradients'].size(0) != grad_magnitudes.size(0):
                print(f"   🔄 Gradient buffer size mismatch ({self.gradient_accum['position_gradients'].size(0)} vs {grad_magnitudes.size(0)}), clearing buffer")
                self.gradient_accum['position_gradients'] = grad_magnitudes.clone()
                self.gradient_accum['gradient_counts'] = torch.ones_like(grad_magnitudes)
            else:
                # Simple accumulation (3DGS standard, not EMA)
                self.gradient_accum['position_gradients'] += grad_magnitudes
                self.gradient_accum['gradient_counts'] += 1
    
    def _densify_gaussians(self, gaussian_map) -> Dict[str, int]:
        """
        Perform error-driven densification through splitting and cloning.
        """
        operations = {'split': 0, 'cloned': 0, 'densified': 0}
        
        if gaussian_map.num_active_gaussians == 0:
            return operations
            
        # Get current memory usage
        memory_info = gaussian_map.get_memory_usage()
        if memory_info['total_mb'] > self.config.target_memory_mb * self.config.memory_pressure_threshold:
            # Skip densification under memory pressure
            return operations
        
        # Identify candidates for densification based on gradients
        densify_candidates = self._identify_densify_candidates(gaussian_map)
        
        if len(densify_candidates) == 0:
            return operations
        
        # Split large Gaussians with high gradients
        split_candidates = self._identify_split_candidates(gaussian_map, densify_candidates)
        if len(split_candidates) > 0:
            split_count = self._split_gaussians(gaussian_map, split_candidates)
            operations['split'] = split_count
            operations['densified'] += split_count * 2  # Each split creates 2 new Gaussians
        
        # Clone small Gaussians with high gradients
        clone_candidates = self._identify_clone_candidates(gaussian_map, densify_candidates)
        if len(clone_candidates) > 0:
            clone_count = self._clone_gaussians(gaussian_map, clone_candidates)
            operations['cloned'] = clone_count
            operations['densified'] += clone_count
        
        # Clear gradient accumulation after densification
        self.gradient_accum.clear()
        
        return operations
    
    def _identify_densify_candidates(self, gaussian_map) -> torch.Tensor:
        """Identify Gaussians that should be densified based on gradients"""
        if 'position_gradients' not in self.gradient_accum:
            return torch.empty(0, dtype=torch.long)
        
        grad_magnitudes = self.gradient_accum['position_gradients']
        grad_counts = self.gradient_accum['gradient_counts']
        
        # Calculate average gradients (3DGS standard)
        avg_grad_magnitudes = grad_magnitudes / grad_counts
        grad_threshold = self.adaptive_densify_threshold
        
        # Select Gaussians with high accumulated gradients
        high_grad_mask = avg_grad_magnitudes > grad_threshold
        
        # Additional filtering based on opacity (don't densify transparent Gaussians)
        opacity = gaussian_map.opacity.squeeze()
        opacity_mask = opacity > 0.01
        
        # Combine masks
        valid_mask = high_grad_mask & opacity_mask
        candidates = torch.where(valid_mask)[0]
        
        return candidates
    
    def _identify_split_candidates(self, gaussian_map, densify_candidates: torch.Tensor) -> torch.Tensor:
        """Identify large Gaussians that should be split"""
        if len(densify_candidates) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Get scales of candidate Gaussians
        scales = gaussian_map.scales[densify_candidates]  # [N, 3]
        max_scales = torch.max(torch.exp(scales), dim=1)[0]  # Convert from log scale
        
        # Split Gaussians that are too large
        split_mask = max_scales > self.config.densify_split_threshold
        split_candidates = densify_candidates[split_mask]
        
        return split_candidates
    
    def _identify_clone_candidates(self, gaussian_map, densify_candidates: torch.Tensor) -> torch.Tensor:
        """Identify small Gaussians that should be cloned"""
        if len(densify_candidates) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Get scales of candidate Gaussians
        scales = gaussian_map.scales[densify_candidates]  # [N, 3]
        max_scales = torch.max(torch.exp(scales), dim=1)[0]
        
        # Clone small Gaussians
        clone_mask = max_scales <= self.config.densify_split_threshold
        clone_candidates = densify_candidates[clone_mask]
        
        return clone_candidates
    
    def _split_gaussians(self, gaussian_map, split_indices: torch.Tensor) -> int:
        """
        Split large Gaussians into smaller ones.
        Each Gaussian is split into two along its largest principal axis.
        """
        if len(split_indices) == 0:
            return 0
        
        # No global count cap in core
        new_count = len(split_indices) * 2
        
        if len(split_indices) == 0:
            return 0
        
        # Get parameters of Gaussians to split
        positions = gaussian_map.positions[split_indices]  # [N, 3]
        rotations = gaussian_map.rotations[split_indices]  # [N, 4]
        scales = gaussian_map.scales[split_indices]        # [N, 3]
        opacity = gaussian_map.opacity[split_indices]      # [N, 1]
        
        # Create two new Gaussians for each split
        # Split along the largest scale axis
        exp_scales = torch.exp(scales)  # Convert from log scale
        largest_scale_idx = torch.argmax(exp_scales, dim=1)  # [N]
        
        # Compute split direction (largest principal axis)
        rotation_matrices = self._quaternion_to_rotation_matrix(rotations)
        split_directions = rotation_matrices[torch.arange(len(split_indices)), :, largest_scale_idx]
        
        # Reduce scale of the split axis (3DGS standard: 1/1.6 scaling)
        new_scales = scales.clone()
        new_scales[torch.arange(len(split_indices)), largest_scale_idx] -= math.log(1.6)
        
        # Compute new positions (offset along split direction)
        offset_distance = exp_scales[torch.arange(len(split_indices)), largest_scale_idx] * 0.5
        offset_vectors = split_directions * offset_distance.unsqueeze(-1)
        
        new_positions_1 = positions + offset_vectors
        new_positions_2 = positions - offset_vectors
        
        # Create new Gaussians
        all_new_positions = torch.cat([new_positions_1, new_positions_2], dim=0)
        all_new_rotations = torch.cat([rotations, rotations], dim=0)
        all_new_scales = torch.cat([new_scales, new_scales], dim=0)
        
        # Keep original opacity (3DGS standard - no opacity reduction for splits)
        all_new_opacity = torch.cat([opacity, opacity], dim=0)
        
        # Add new Gaussians to the map
        success = gaussian_map.add_gaussians(all_new_positions, all_new_rotations, all_new_scales)
        
        if success:
            # Remove original Gaussians by setting them inactive
            prune_mask = torch.zeros(gaussian_map.num_active_gaussians, dtype=torch.bool, device=positions.device)
            prune_mask[split_indices] = True
            gaussian_map.prune_gaussians(prune_mask)
            
            return len(split_indices)
        
        return 0
    
    def _clone_gaussians(self, gaussian_map, clone_indices: torch.Tensor) -> int:
        """
        Clone small Gaussians by creating copies with slight position offsets.
        """
        if len(clone_indices) == 0:
            return 0
        
        # No global count cap in core
        
        if len(clone_indices) == 0:
            return 0
        
        # Get parameters of Gaussians to clone
        positions = gaussian_map.positions[clone_indices]    # [N, 3]
        rotations = gaussian_map.rotations[clone_indices]    # [N, 4]
        scales = gaussian_map.scales[clone_indices]          # [N, 3]
        
        # Create small random offsets for cloned positions
        exp_scales = torch.exp(scales)
        avg_scales = torch.mean(exp_scales, dim=1, keepdim=True)  # [N, 1]
        random_offsets = torch.randn_like(positions) * avg_scales * 0.1
        new_positions = positions + random_offsets
        
        # Add cloned Gaussians
        success = gaussian_map.add_gaussians(new_positions, rotations, scales)
        
        return len(clone_indices) if success else 0
    
    def _prune_gaussians(self, gaussian_map) -> int:
        """
        Prune Gaussians based on opacity, scale, and memory pressure.
        """
        if gaussian_map.num_active_gaussians == 0:
            return 0
        
        # Update spatial hash once before all neighbor queries
        if hasattr(gaussian_map, '_update_spatial_hash'):
            gaussian_map._update_spatial_hash()
        
        # Get current parameters
        opacity = gaussian_map.opacity.squeeze()  # [N]
        scales = gaussian_map.scales              # [N, 3]
        positions = gaussian_map.positions        # [N, 3]
        
        # Standard pruning criteria
        prune_mask = torch.zeros(len(opacity), dtype=torch.bool, device=opacity.device)
        
        # 1. Prune by opacity
        opacity_mask = opacity < self.adaptive_prune_threshold
        prune_mask |= opacity_mask
        
        # 2. Prune by scale (too large Gaussians)
        max_scales = torch.max(torch.exp(scales), dim=1)[0]
        scale_mask = max_scales > self.config.prune_scale_threshold
        prune_mask |= scale_mask
        
        # 3. Memory pressure pruning
        memory_info = gaussian_map.get_memory_usage()
        if memory_info['total_mb'] > self.config.target_memory_mb * self.config.memory_pressure_threshold:
            # Aggressive pruning under memory pressure
            # Remove lowest opacity Gaussians first
            low_opacity_count = int(len(opacity) * 0.1)  # Remove 10% of lowest opacity
            _, low_opacity_indices = torch.topk(opacity, low_opacity_count, largest=False)
            prune_mask[low_opacity_indices] = True
        
        # 4. Conservative spatial redundancy pruning (3DGS standard)
        # Only prune Gaussians that are very close AND have significantly lower opacity
        if hasattr(gaussian_map, 'query_neighbors') and callable(gaussian_map.query_neighbors):
            spatial_redundancy_mask = self._identify_spatial_redundancy_conservative(gaussian_map, opacity)
            prune_mask |= spatial_redundancy_mask
        # Note: Most 3DGS implementations don't use spatial pruning - opacity/scale pruning is sufficient
        
        # Apply pruning
        prune_count = prune_mask.sum().item()
        if prune_count > 0:
            gaussian_map.prune_gaussians(prune_mask)
            
        return prune_count
    
    def _identify_spatial_redundancy_conservative(self, gaussian_map, opacity) -> torch.Tensor:
        """Conservative spatial redundancy pruning (3DGS industry standard approach).
        
        Only prunes Gaussians that are:
        1. Very close to another Gaussian (< 0.005 units)
        2. Have significantly lower opacity (< 50% of neighbor's opacity)
        3. Are smaller in scale
        """
        positions = gaussian_map.positions
        if positions is None or len(positions) == 0:
            return torch.zeros(0, dtype=torch.bool, device=opacity.device)
            
        device = positions.device
        redundant_mask = torch.zeros(len(positions), dtype=torch.bool, device=device)
        
        try:
            # Very conservative approach - only check immediate neighbors
            query_fn = gaussian_map.query_neighbors
            scales = gaussian_map.scales
            
            # Process in small batches to avoid memory issues
            batch_size = min(100, len(positions))
            for batch_start in range(0, len(positions), batch_size):
                batch_end = min(batch_start + batch_size, len(positions))
                
                for i in range(batch_start, batch_end):
                    if redundant_mask[i]:  # Already marked for removal
                        continue
                        
                    pos = positions[i]
                    # Use very small radius for conservative pruning
                    nbr_idx = query_fn(pos, radius=0.005)
                    
                    if nbr_idx is None or len(nbr_idx) <= 1:  # Only self or no neighbors
                        continue
                        
                    # Remove self from neighbors
                    nbr_idx = [idx for idx in nbr_idx if idx != i]
                    if len(nbr_idx) == 0:
                        continue
                        
                    current_opacity = opacity[i].item()
                    current_scale = torch.max(torch.exp(scales[i])).item()
                    
                    # Check if this Gaussian is significantly worse than any neighbor
                    for nbr in nbr_idx:
                        nbr_opacity = opacity[nbr].item()
                        nbr_scale = torch.max(torch.exp(scales[nbr])).item()
                        
                        # Very conservative criteria for redundancy
                        if (current_opacity < nbr_opacity * 0.5 and  # Much lower opacity
                            current_scale < nbr_scale * 0.8 and      # Smaller scale
                            current_opacity < 0.01):                 # Already quite transparent
                            redundant_mask[i] = True
                            break
                            
        except (AttributeError, RuntimeError, TypeError):
            # If neighbor queries fail, skip spatial redundancy pruning
            # This is fine - 3DGS works well with just opacity/scale pruning
            pass
            
        # Limit to maximum 1% of Gaussians to be very conservative
        redundant_count = redundant_mask.sum().item()
        max_redundant = max(1, int(len(positions) * 0.01))
        if redundant_count > max_redundant:
            # Keep only the most redundant ones (lowest opacity)
            redundant_indices = torch.where(redundant_mask)[0]
            redundant_opacities = opacity[redundant_indices]
            _, keep_indices = torch.topk(redundant_opacities, max_redundant, largest=False)
            
            # Reset mask and only keep the worst ones
            redundant_mask.fill_(False)
            redundant_mask[redundant_indices[keep_indices]] = True
            
        return redundant_mask
    
    def _update_error_history(self, error: float):
        """Update reconstruction error history for adaptation"""
        self.error_history.append(error)
        if len(self.error_history) > self.config.error_history_length:
            self.error_history.pop(0)
    
    def _adapt_thresholds(self, operations: Dict[str, int]):
        """Adapt thresholds based on recent performance"""
        if len(self.error_history) < 10:
            return
        
        # Compute error trend
        recent_errors = self.error_history[-10:]
        error_trend = (recent_errors[-1] - recent_errors[0]) / len(recent_errors)
        
        # Adapt densification threshold
        if error_trend > 0:  # Error increasing
            # Be more aggressive with densification
            self.adaptive_densify_threshold *= (1 - self.config.adaptation_rate)
        else:  # Error decreasing or stable
            # Be less aggressive
            self.adaptive_densify_threshold *= (1 + self.config.adaptation_rate * 0.5)
        
        # Clamp thresholds to reasonable ranges
        self.adaptive_densify_threshold = torch.clamp(
            torch.tensor(self.adaptive_densify_threshold),
            self.config.densify_grad_threshold * 0.1,
            self.config.densify_grad_threshold * 10.0
        ).item()
        
        # Adapt pruning threshold based on memory usage
        if operations['densified'] > operations['pruned'] * 2:
            # Too much densification, be more aggressive with pruning
            self.adaptive_prune_threshold *= (1 + self.config.adaptation_rate)
        else:
            # Balance is okay
            self.adaptive_prune_threshold *= (1 - self.config.adaptation_rate * 0.5)
        
        self.adaptive_prune_threshold = torch.clamp(
            torch.tensor(self.adaptive_prune_threshold),
            self.config.prune_opacity_threshold * 0.1,
            self.config.prune_opacity_threshold * 10.0
        ).item()
    
    def _update_spatial_hash(self, gaussian_map):
        """Update spatial hash for efficient neighbor queries"""
        # Skip spatial hash for now to avoid dependency issues
        pass
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices"""
        import torch.nn.functional as F
        
        q = F.normalize(quaternions, dim=1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrices = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=1)
        ], dim=1)
        
        return rotation_matrices
    
    def _should_reset_opacity(self) -> bool:
        """Check if opacity should be reset based on 3DGS schedule (every 3000 iterations)"""
        return (self.step_count > 0 and 
                self.step_count % self.opacity_reset_interval == 0 and
                self.step_count != self.last_opacity_reset)
    
    def _reset_opacity(self, gaussian_map):
        """Reset opacity for all Gaussians (3DGS standard: reset to 0.01)"""
        if hasattr(gaussian_map, '_opacity') and gaussian_map._opacity is not None:
            with torch.no_grad():
                # Reset opacity for Gaussians above threshold to 0.01 (3DGS standard)
                high_opacity_mask = gaussian_map._opacity.squeeze() > 0.01
                gaussian_map._opacity[high_opacity_mask] = 0.01
                print(f"   🔄 Reset opacity for {high_opacity_mask.sum().item()} Gaussians at step {self.step_count}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics and state"""
        stats = {
            'step_count': self.step_count,
            'adaptive_densify_threshold': self.adaptive_densify_threshold,
            'adaptive_prune_threshold': self.adaptive_prune_threshold,
            'last_densify_step': self.last_densify_step,
            'last_prune_step': self.last_prune_step,
            'error_history_length': len(self.error_history),
            'gradient_accum_size': len(self.gradient_accum),
        }
        
        if self.error_history:
            stats['current_error'] = self.error_history[-1]
            stats['avg_error'] = sum(self.error_history) / len(self.error_history)
        
        return stats
    
    def reset(self):
        """Reset manager state"""
        self.step_count = 0
        self.gradient_accum.clear()
        self.error_history.clear()
        self.last_densify_step = 0
        self.last_prune_step = 0
        self.spatial_hash = None
        
        # Reset adaptive thresholds
        self.adaptive_densify_threshold = self.config.densify_grad_threshold
        self.adaptive_prune_threshold = self.config.prune_opacity_threshold

class MemoryAwareDensifyPrune(DensifyPruneManager):
    """Enhanced version with more sophisticated memory management"""
    
    def __init__(self, config: DensifyPruneConfig):
        super().__init__(config)
        self.memory_monitor_interval = 10
        self.memory_history = []
        
    def step(self, gaussian_map, loss_info: Dict[str, torch.Tensor],
             optimizer_state: Optional[Dict] = None) -> Dict[str, int]:
        """Enhanced step with memory monitoring"""
        # Monitor memory usage
        if self.step_count % self.memory_monitor_interval == 0:
            memory_info = gaussian_map.get_memory_usage()
            self.memory_history.append(memory_info['total_mb'])
            
            # Trigger compaction if memory fragmented
            if len(self.memory_history) > 10:
                recent_memory = self.memory_history[-5:]
                if max(recent_memory) - min(recent_memory) > 100:  # 100MB fragmentation
                    gaussian_map.compact_gaussians()
        
        return super().step(gaussian_map, loss_info, optimizer_state)
    
    def _should_prune(self) -> bool:
        """Enhanced pruning decision with memory pressure"""
        base_should_prune = super()._should_prune()
        
        # Force pruning under high memory pressure
        if self.memory_history:
            current_memory = self.memory_history[-1]
            memory_pressure = current_memory / self.config.target_memory_mb
            
            if memory_pressure > 0.9:  # Emergency pruning
                return True
        
        return base_should_prune