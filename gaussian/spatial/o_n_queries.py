"""
O(n) spatial queries using simple linear search.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ONSpatialQueries:
    """
    O(n) spatial query system using linear search instead of spatial acceleration structures.
    """
    
    def __init__(self):
        """Initialize O(n) spatial query system"""
        self.disable_spatial_hash = True
        self.query_method = "linear_search"
        
        logger.info("Initialized O(n) spatial queries")
    
    def find_neighbors(self, 
                      query_point: np.ndarray,
                      all_points: np.ndarray,
                      radius: float,
                      max_neighbors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors using O(n) linear search.
        
        Args:
            query_point: Point to query around [3]
            all_points: All points to search [N, 3]
            radius: Search radius
            max_neighbors: Maximum neighbors to return
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        if all_points.shape[0] == 0:
            return np.array([]), np.array([])
        
        # O(n) distance computation
        distances = np.linalg.norm(all_points - query_point, axis=1)
        
        # Find neighbors within radius
        neighbor_mask = distances <= radius
        neighbor_indices = np.where(neighbor_mask)[0]
        neighbor_distances = distances[neighbor_mask]
        
        # Limit number of neighbors if specified
        if max_neighbors is not None and len(neighbor_indices) > max_neighbors:
            # Sort by distance and take closest
            sorted_idx = np.argsort(neighbor_distances)[:max_neighbors]
            neighbor_indices = neighbor_indices[sorted_idx]
            neighbor_distances = neighbor_distances[sorted_idx]
        
        logger.debug(f"O(n) neighbor search found {len(neighbor_indices)} neighbors")
        return neighbor_indices, neighbor_distances
    
    def k_nearest_neighbors(self,
                           query_point: np.ndarray,
                           all_points: np.ndarray,
                           k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors using O(n) search.
        
        Args:
            query_point: Point to query around [3]
            all_points: All points to search [N, 3]
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        if all_points.shape[0] == 0:
            return np.array([]), np.array([])
        
        # O(n) distance computation
        distances = np.linalg.norm(all_points - query_point, axis=1)
        
        # Find k nearest neighbors
        k = min(k, len(distances))
        nearest_idx = np.argpartition(distances, k-1)[:k]
        nearest_distances = distances[nearest_idx]
        
        # Sort by distance
        sort_idx = np.argsort(nearest_distances)
        nearest_idx = nearest_idx[sort_idx]
        nearest_distances = nearest_distances[sort_idx]
        
        logger.debug(f"O(n) k-NN search found {k} nearest neighbors")
        return nearest_idx, nearest_distances
    
    def range_query(self,
                   query_points: np.ndarray,
                   all_points: np.ndarray,
                   radius: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform range queries for multiple points using O(n) search.
        
        Args:
            query_points: Points to query around [M, 3]
            all_points: All points to search [N, 3]
            radius: Search radius
            
        Returns:
            List of (neighbor_indices, distances) for each query point
        """
        results = []
        
        for query_point in query_points:
            neighbor_indices, distances = self.find_neighbors(
                query_point, all_points, radius
            )
            results.append((neighbor_indices, distances))
        
        logger.debug(f"O(n) range query completed for {len(query_points)} query points")
        return results
    
    def spatial_density_estimation(self,
                                 points: np.ndarray,
                                 radius: float) -> np.ndarray:
        """
        Estimate point density using O(n²) approach.
        
        Args:
            points: Points to estimate density for [N, 3]
            radius: Density estimation radius
            
        Returns:
            Density values for each point [N]
        """
        densities = np.zeros(len(points))
        
        for i, point in enumerate(points):
            neighbor_indices, _ = self.find_neighbors(point, points, radius)
            densities[i] = len(neighbor_indices) - 1  # Exclude self
        
        logger.debug(f"O(n²) density estimation completed for {len(points)} points")
        return densities


class BasicSpatialQueries:
    """
    Simple spatial query interface using O(n) methods.
    """
    
    def __init__(self):
        """Initialize queries"""
        self.spatial_queries = ONSpatialQueries()
        
        # Disable all spatial optimizations
        self.use_spatial_hash = False
        self.use_kd_tree = False
        self.use_octree = False
        
        logger.info("Initialized basic spatial queries")
    
    def query_neighbors(self,
                                        query_point: np.ndarray,
                                        scene_points: np.ndarray,
                                        search_params: dict) -> dict:
        """
        Query neighbors using O(n) approach.
        
        Args:
            query_point: Query point [3]
            scene_points: Scene points [N, 3]
            search_params: Search parameters
            
        Returns:
            Dictionary with neighbor information
        """
        radius = search_params.get('radius', 0.1)
        max_neighbors = search_params.get('max_neighbors', None)
        
        neighbor_indices, distances = self.spatial_queries.find_neighbors(
            query_point, scene_points, radius, max_neighbors
        )
        
        return {
            'neighbor_indices': neighbor_indices,
            'distances': distances,
            'num_neighbors': len(neighbor_indices),
            'query_method': 'o_n_linear_search',
            'spatial_optimization': False
        }
    
    def get_query_info(self):
        """
        Get information about the spatial query configuration.
        """
        return {
            'spatial_hash_disabled': True,  
            'reason': 'Use O(n) approach',
            'alternative': 'Use linear search'
        }