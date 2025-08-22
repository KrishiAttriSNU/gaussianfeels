"""Core Gaussian splatting modules with object-centric fusion"""

from .gaussian_field import ObjectGaussianMap, GaussianConfig
from .densify_prune import DensifyPruneManager, DensifyPruneConfig
from .fusion import ObjectCentricFusion, FusionConfig, IntegrationMode

__all__ = [
    'ObjectGaussianMap', 
    'GaussianConfig', 
    'DensifyPruneManager', 
    'DensifyPruneConfig',
    'ObjectCentricFusion',
    'FusionConfig', 
    'IntegrationMode'
]