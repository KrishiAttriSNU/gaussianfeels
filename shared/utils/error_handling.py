"""
Error handling utilities for GaussianFeels

Standardized error handling patterns to eliminate duplication.
"""

import sys
import traceback
from typing import Optional, Any, Callable
from pathlib import Path


class GaussianFeelsError(Exception):
    """Base exception for GaussianFeels-specific errors"""
    pass


class ConfigurationError(GaussianFeelsError):
    """Error in configuration or setup"""
    pass


class DatasetError(GaussianFeelsError):
    """Error in dataset loading or validation"""
    pass


class DeviceError(GaussianFeelsError):
    """Error in device setup or usage"""
    pass


def handle_file_not_found(file_path: str, context: str = "file") -> None:
    """
    Standardized file not found error handling.
    
    Args:
        file_path: Path that was not found
        context: Context description for the error
    
    Raises:
        FileNotFoundError: With standardized message
    """
    raise FileNotFoundError(f"Required {context} not found: {file_path}")


def handle_directory_not_found(dir_path: str, context: str = "directory") -> None:
    """
    Standardized directory not found error handling.
    
    Args:
        dir_path: Directory path that was not found
        context: Context description for the error
    
    Raises:
        NotADirectoryError: With standardized message
    """
    raise NotADirectoryError(f"Required {context} not found: {dir_path}")


def validate_and_raise(condition: bool, error_message: str, 
                      error_type: type = ValueError) -> None:
    """
    Validate condition and raise error with message if false.
    
    Args:
        condition: Condition to validate
        error_message: Error message if condition fails
        error_type: Type of error to raise
    
    Raises:
        error_type: If condition is False
    """
    if not condition:
        raise error_type(error_message)


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module with error handling.
    
    Args:
        module_name: Name of module to import
        package: Package context for relative imports
    
    Returns:
        Module object or None if import fails
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")
        return None


def with_error_context(context: str):
    """
    Decorator to add context to errors.
    
    Args:
        context: Context description for errors
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise type(e)(f"{context}: {str(e)}") from e
        return wrapper
    return decorator


def log_and_reraise(error: Exception, context: str = "", 
                   log_traceback: bool = False) -> None:
    """
    Log error details and re-raise.
    
    Args:
        error: Exception to log and re-raise
        context: Additional context for logging
        log_traceback: Whether to log full traceback
    
    Raises:
        Exception: Re-raises the original exception
    """
    error_msg = f"Error in {context}: {str(error)}" if context else str(error)
    print(f"ERROR: {error_msg}", file=sys.stderr)
    
    if log_traceback:
        print("Traceback:", file=sys.stderr)
        traceback.print_exc()
    
    raise error