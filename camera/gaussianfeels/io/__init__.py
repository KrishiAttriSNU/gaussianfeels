"""I/O module for GaussianFeels tactile and vision processing"""

from .tactile_predictors import make_predictor, TactilePredictorMode

__all__ = ['make_predictor', 'TactilePredictorMode']