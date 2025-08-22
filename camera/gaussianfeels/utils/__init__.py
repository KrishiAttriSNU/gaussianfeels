"""Utility modules for spatial operations"""

from .spatial_hash import (
    CUDASpatialHash,
    SpatialHashConfig, 
    AdaptiveSpatialHash,
    create_spatial_hash,
    get_global_spatial_hash
)

# Backward compatibility alias
SpatialHashGrid = CUDASpatialHash

__all__ = [
    'CUDASpatialHash',
    'SpatialHashGrid',  # Backward compatibility
    'SpatialHashConfig', 
    'AdaptiveSpatialHash',
    'create_spatial_hash',
    'get_global_spatial_hash'
]