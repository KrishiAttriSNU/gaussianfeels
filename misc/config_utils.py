#!/usr/bin/env python3
"""
Configuration utilities for GFS1 pipeline
=========================================

Provides utilities for managing data paths and configuration across modules.
Replaces hardcoded paths with runtime configuration.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent


def get_data_root() -> Path:
    """Get the data root directory"""
    return get_project_root() / "data"


def get_default_output_dir(module_name: str) -> Path:
    """Get default output directory for a module"""
    return get_project_root() / "tests" / "output" / module_name


def resolve_data_path(path_input: Optional[str] = None) -> Optional[str]:
    """
    Resolve data path from various inputs
    
    Args:
        path_input: Input path (can be absolute, relative, or None)
        
    Returns:
        Resolved absolute path or None
    """
    if path_input is None:
        return None
        
    path = Path(path_input)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return str(path)
        
    # Try relative to current working directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)
        
    # Try relative to project root
    root_path = get_project_root() / path
    if root_path.exists():
        return str(root_path)
        
    # Try relative to data directory
    data_path = get_data_root() / path
    if data_path.exists():
        return str(data_path)
        
    # Return the original path and let the caller handle
    return str(path)


def update_config_paths(config: Dict[str, Any], 
                       data_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Update configuration with runtime paths
    
    Args:
        config: Configuration dictionary
        data_path: Override data path
        output_dir: Override output directory
        
    Returns:
        Updated configuration
    """
    config = config.copy()
    
    if 'paths' in config:
        if data_path:
            config['paths']['dataset_path'] = resolve_data_path(data_path)
            config['paths']['gfs1_root'] = str(get_project_root())
            
        if output_dir:
            config['paths']['output_directory'] = resolve_data_path(output_dir)
            
    return config


class PathManager:
    """Centralized path management for the pipeline"""
    
    def __init__(self, 
                 project_root: Optional[str] = None,
                 data_root: Optional[str] = None):
        """
        Initialize path manager
        
        Args:
            project_root: Override project root
            data_root: Override data root
        """
        self.project_root = Path(project_root) if project_root else get_project_root()
        self.data_root = Path(data_root) if data_root else (self.project_root / "data")
        
    def get_module_paths(self) -> Dict[str, str]:
        """Get paths for all modules"""
        return {
            'camera': str(self.project_root / 'camera'),
            'tactile': str(self.project_root / 'tactile'), 
            'fusion': str(self.project_root / 'fusion'),
            'tests': str(self.project_root / 'tests')
        }
        
    def get_output_dir(self, module_name: str) -> str:
        """Get output directory for a module"""
        return str(self.project_root / "tests" / "output" / module_name)
        
    def resolve_data_path(self, path: str) -> str:
        """Resolve a data path"""
        return resolve_data_path(path)