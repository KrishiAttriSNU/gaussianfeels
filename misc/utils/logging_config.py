"""
Centralized logging configuration for gaussianfeels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up standardized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(name)s:%(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    return logging.getLogger("gaussianfeels")

def get_logger(name: str) -> logging.Logger:
    """Get a logger with standardized naming convention."""
    return logging.getLogger(f"gaussianfeels.{name}")