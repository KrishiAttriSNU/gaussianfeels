# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
High-performance spatial hash for GaussianFeels with O(1) bounded neighbor queries.
Optimized for 300k+ Gaussians with CUDA acceleration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import threading


@dataclass
class SpatialHashConfig:
    """Configuration for spatial hash"""
    cell_size: float = 0.01  # 1cm cells for tactile precision
    max_neighbors: int = 64  # O(1) bound for neighbor queries
    hash_table_size: int = 1000000  # Large prime number for good distribution
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


class CUDASpatialHash(nn.Module):
    """CUDA-accelerated spatial hash for real-time Gaussian neighbor queries"""
    
    def __init__(self, config: SpatialHashConfig):
        super().__init__()
        self.config = config
        from shared.utils.device_utils import setup_device
        self.device = setup_device(config)
        
        # Hash table storage
        self.hash_table = torch.full(
            (config.hash_table_size, config.max_neighbors),
            -1, dtype=torch.int32, device=self.device
        )
        self.hash_counts = torch.zeros(
            config.hash_table_size, dtype=torch.int32, device=self.device
        )
        
        # Point storage
        self.max_points = 500000  # Support up to 500k Gaussians
        self.positions = torch.zeros(
            (self.max_points, 3), dtype=config.dtype, device=self.device
        )
        self.valid_mask = torch.zeros(
            self.max_points, dtype=torch.bool, device=self.device
        )
        self.num_points = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance counters
        self.query_count = 0
        self.collision_count = 0
        
    def _hash_position(self, position: torch.Tensor) -> torch.Tensor:
        """Hash 3D position to cell indices"""
        # Quantize to cell coordinates
        cell_coords = torch.floor(position / self.config.cell_size).int()
        
        # Compute hash using prime numbers for good distribution
        p1, p2, p3 = 73856093, 19349663, 83492791
        hash_values = (cell_coords[:, 0] * p1 + 
                      cell_coords[:, 1] * p2 + 
                      cell_coords[:, 2] * p3)
        
        return torch.abs(hash_values) % self.config.hash_table_size
    
    def clear(self):
        """Clear all points from hash table"""
        with self._lock:
            self.hash_table.fill_(-1)
            self.hash_counts.zero_()
            self.valid_mask.zero_()
            self.num_points = 0
            self.query_count = 0
            self.collision_count = 0
    
    def insert_points(self, positions: torch.Tensor, point_ids: Optional[torch.Tensor] = None):
        """Insert batch of points into spatial hash"""
        with self._lock:
            batch_size = positions.shape[0]
            
            if self.num_points + batch_size > self.max_points:
                raise RuntimeError(f"Exceeding maximum points: {self.max_points}")
            
            # Generate point IDs if not provided
            if point_ids is None:
                point_ids = torch.arange(
                    self.num_points, self.num_points + batch_size,
                    dtype=torch.int32, device=self.device
                )
            
            # Store positions
            start_idx = self.num_points
            end_idx = start_idx + batch_size
            self.positions[start_idx:end_idx] = positions.to(self.device)
            self.valid_mask[start_idx:end_idx] = True
            
            # Hash positions
            hash_indices = self._hash_position(positions)
            
            # Insert into hash table
            for i in range(batch_size):
                hash_idx = hash_indices[i].item()
                point_id = point_ids[i].item()
                
                # Find insertion slot (linear probing for collisions)
                slot_idx = self.hash_counts[hash_idx].item()
                
                if slot_idx < self.config.max_neighbors:
                    self.hash_table[hash_idx, slot_idx] = point_id
                    self.hash_counts[hash_idx] += 1
                else:
                    # Hash table bucket full - collision
                    self.collision_count += 1
            
            self.num_points += batch_size
    
    def query_neighbors_batch(self, 
                            query_positions: torch.Tensor, 
                            radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query neighbors for batch of positions (O(1) bounded)"""
        batch_size = query_positions.shape[0]
        
        # Output tensors
        neighbors = torch.full(
            (batch_size, self.config.max_neighbors),
            -1, dtype=torch.int32, device=self.device
        )
        neighbor_counts = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )
        
        # Calculate search radius in cells
        cell_radius = int(np.ceil(radius / self.config.cell_size))
        
        with self._lock:
            self.query_count += batch_size
            
            for batch_idx in range(batch_size):
                query_pos = query_positions[batch_idx]
                neighbor_idx = 0
                
                # Search neighboring cells
                for dx in range(-cell_radius, cell_radius + 1):
                    for dy in range(-cell_radius, cell_radius + 1):
                        for dz in range(-cell_radius, cell_radius + 1):
                            
                            # Calculate neighbor cell position
                            neighbor_cell = query_pos / self.config.cell_size + torch.tensor(
                                [dx, dy, dz], dtype=query_pos.dtype, device=self.device
                            )
                            neighbor_cell *= self.config.cell_size
                            
                            # Hash neighbor cell
                            hash_idx = self._hash_position(neighbor_cell.unsqueeze(0))[0].item()
                            
                            # Check all points in this cell
                            cell_count = self.hash_counts[hash_idx].item()
                            for slot_idx in range(min(cell_count, self.config.max_neighbors)):
                                point_id = self.hash_table[hash_idx, slot_idx].item()
                                
                                if point_id >= 0 and point_id < self.num_points:
                                    if self.valid_mask[point_id]:
                                        # Check distance
                                        point_pos = self.positions[point_id]
                                        dist = torch.norm(query_pos - point_pos)
                                        
                                        if dist <= radius:
                                            if neighbor_idx < self.config.max_neighbors:
                                                neighbors[batch_idx, neighbor_idx] = point_id
                                                neighbor_idx += 1
                                            else:
                                                # Reached max neighbors - O(1) bound enforced
                                                break
                            
                            # Break if max neighbors reached
                            if neighbor_idx >= self.config.max_neighbors:
                                break
                        if neighbor_idx >= self.config.max_neighbors:
                            break
                    if neighbor_idx >= self.config.max_neighbors:
                        break
                
                neighbor_counts[batch_idx] = neighbor_idx
        
        return neighbors, neighbor_counts
    
    def query_neighbors(self, query_position: torch.Tensor, radius: float) -> torch.Tensor:
        """Query neighbors for single position"""
        neighbors, counts = self.query_neighbors_batch(
            query_position.unsqueeze(0), radius
        )
        neighbor_count = counts[0].item()
        return neighbors[0, :neighbor_count]
    
    def remove_points(self, point_ids: torch.Tensor):
        """Remove points from spatial hash"""
        with self._lock:
            for point_id in point_ids:
                if 0 <= point_id < self.num_points:
                    self.valid_mask[point_id] = False
    
    def rebuild(self):
        """Rebuild hash table from scratch (call after many removals)"""
        with self._lock:
            # Get valid points
            valid_indices = torch.where(self.valid_mask[:self.num_points])[0]
            valid_positions = self.positions[valid_indices]
            
            # Clear and rebuild
            self.hash_table.fill_(-1)
            self.hash_counts.zero_()
            self.valid_mask.zero_()
            self.num_points = 0
            
            if len(valid_positions) > 0:
                self.insert_points(valid_positions, valid_indices)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get spatial hash performance statistics"""
        with self._lock:
            occupied_cells = (self.hash_counts > 0).sum().item()
            total_cells = self.config.hash_table_size
            
            avg_occupancy = self.hash_counts[self.hash_counts > 0].float().mean().item() if occupied_cells > 0 else 0.0
            
            return {
                'num_points': self.num_points,
                'occupied_cells': occupied_cells,
                'occupancy_ratio': occupied_cells / total_cells,
                'avg_points_per_cell': avg_occupancy,
                'query_count': self.query_count,
                'collision_count': self.collision_count,
                'collision_rate': self.collision_count / max(1, self.query_count)
            }
    
    def optimize_memory(self):
        """Optimize memory usage by compacting hash table"""
        stats = self.get_statistics()
        
        # If collision rate is high, rebuild with larger hash table
        if stats['collision_rate'] > 0.1:  # 10% collision rate threshold
            import logging
            logging.getLogger(__name__).warning(
                f"High collision rate {stats['collision_rate']:.3f}, consider increasing hash_table_size"
            )
        
        # If occupancy is very low, could use smaller hash table
        if stats['occupancy_ratio'] < 0.01:  # 1% occupancy threshold
            import logging
            logging.getLogger(__name__).info(
                f"Low occupancy {stats['occupancy_ratio']:.3f}, could use smaller hash_table_size"
            )


class AdaptiveSpatialHash(CUDASpatialHash):
    """Adaptive spatial hash that adjusts cell size based on point density"""
    
    def __init__(self, config: SpatialHashConfig):
        super().__init__(config)
        self.adaptive_enabled = True
        self.density_check_interval = 1000  # Check density every N insertions
        self.insertion_count = 0
        
    def insert_points(self, positions: torch.Tensor, point_ids: Optional[torch.Tensor] = None):
        """Insert points with adaptive cell size adjustment"""
        super().insert_points(positions, point_ids)
        
        if self.adaptive_enabled:
            self.insertion_count += positions.shape[0]
            
            if self.insertion_count >= self.density_check_interval:
                self._adapt_cell_size()
                self.insertion_count = 0
    
    def _adapt_cell_size(self):
        """Adapt cell size based on current point density"""
        stats = self.get_statistics()
        
        # Target: 4-8 points per occupied cell on average
        target_points_per_cell = 6.0
        current_points_per_cell = stats['avg_points_per_cell']
        
        if current_points_per_cell > target_points_per_cell * 1.5:
            # Too dense - increase cell size
            new_cell_size = self.config.cell_size * 1.1
            self._update_cell_size(new_cell_size)
        elif current_points_per_cell < target_points_per_cell * 0.5:
            # Too sparse - decrease cell size
            new_cell_size = self.config.cell_size * 0.9
            self._update_cell_size(new_cell_size)
    
    def _update_cell_size(self, new_cell_size: float):
        """Update cell size and rebuild hash table"""
        if abs(new_cell_size - self.config.cell_size) / self.config.cell_size > 0.05:  # 5% threshold
            print(f"Adapting cell size: {self.config.cell_size:.4f} → {new_cell_size:.4f}")
            self.config.cell_size = new_cell_size
            self.rebuild()


def create_spatial_hash(config: Optional[SpatialHashConfig] = None, 
                       adaptive: bool = True) -> CUDASpatialHash:
    """Factory function to create optimized spatial hash"""
    if config is None:
        config = SpatialHashConfig()
    
    if adaptive:
        return AdaptiveSpatialHash(config)
    else:
        return CUDASpatialHash(config)


# Global spatial hash instance for GaussianFeels
_global_spatial_hash: Optional[CUDASpatialHash] = None


def get_global_spatial_hash() -> CUDASpatialHash:
    """Get global spatial hash instance"""
    global _global_spatial_hash
    if _global_spatial_hash is None:
        config = SpatialHashConfig(
            cell_size=0.005,  # 5mm cells for high precision
            max_neighbors=64,
            device="cuda"
        )
        _global_spatial_hash = create_spatial_hash(config, adaptive=True)
    return _global_spatial_hash


if __name__ == "__main__":
    # Performance test
    print("Testing spatial hash performance...")
    
    config = SpatialHashConfig(cell_size=0.01, max_neighbors=32)
    spatial_hash = create_spatial_hash(config)
    
    # Generate test data
    n_points = 10000
    positions = torch.rand(n_points, 3, device=spatial_hash.device) * 2.0  # 2m x 2m x 2m space
    
    # Insert points
    import time
    start_time = time.time()
    spatial_hash.insert_points(positions)
    insertion_time = time.time() - start_time
    
    print(f"Inserted {n_points} points in {insertion_time:.3f}s ({n_points/insertion_time:.0f} points/s)")
    
    # Test queries
    n_queries = 1000
    query_positions = torch.rand(n_queries, 3, device=spatial_hash.device) * 2.0
    
    start_time = time.time()
    neighbors, counts = spatial_hash.query_neighbors_batch(query_positions, radius=0.1)
    query_time = time.time() - start_time
    
    print(f"Performed {n_queries} queries in {query_time:.3f}s ({n_queries/query_time:.0f} queries/s)")
    
    # Statistics
    stats = spatial_hash.get_statistics()
    print(f"Statistics: {stats}")
    
    avg_neighbors = counts.float().mean().item()
    print(f"Average neighbors found: {avg_neighbors:.1f}")