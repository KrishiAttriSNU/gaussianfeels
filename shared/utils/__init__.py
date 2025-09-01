"""
Shared utility functions for GaussianFeels
"""

from .device_utils import get_device, setup_device
from .dataset_utils import (
    validate_dataset_structure,
    resolve_camera_name,
    get_file_list,
    create_dataset_config_validator
)
from .error_handling import (
    GaussianFeelsError,
    ConfigurationError,
    DatasetError,
    DeviceError,
    handle_file_not_found,
    handle_directory_not_found,
    validate_and_raise,
    safe_import,
    with_error_context,
    log_and_reraise
)

__all__ = [
    # Device utilities
    'get_device', 'setup_device',
    # Dataset utilities
    'validate_dataset_structure', 'resolve_camera_name', 
    'get_file_list', 'create_dataset_config_validator',
    # Error handling
    'GaussianFeelsError', 'ConfigurationError', 'DatasetError', 'DeviceError',
    'handle_file_not_found', 'handle_directory_not_found', 'validate_and_raise',
    'safe_import', 'with_error_context', 'log_and_reraise'
]