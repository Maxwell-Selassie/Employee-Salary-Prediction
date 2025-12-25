"""
Time and timestamp utilities for production pipeline.
"""

import time
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_timestamp(format: str = "%Y%m%d%H%M%S") -> str:
    """
    Get current timestamp in format: YYYYMMDD_HHMMSS
    
    Returns:
        Timestamp string
        
    Example:
        >>> get_timestamp()
        '20250108_143025'
    """
    return datetime.now().strftime(format)


def get_date() -> str:
    """
    Get current date in format: YYYYMMDD
    
    Returns:
        Date string
        
    Example:
        >>> get_date()
        '20250108'
    """
    return datetime.now().strftime("%Y%m%d")


def get_datetime_str(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current datetime with custom format.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> get_datetime_str("%d/%m/%Y")
        '08/01/2025'
    """
    return datetime.now().strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Example:
        >>> format_duration(3725.5)
        '1h 2m 5.50s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class Timer:
    """
    Context manager for timing code execution.
    
    Example:
        >>> with Timer("Data loading"):
        ...     df = pd.read_csv("data.csv")
        Data loading completed in 2.34s
    """
    
    def __init__(self, name: str = "Operation", logger_instance: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            logger_instance: Logger instance to use
        """
        self.name = name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        """Start timer."""
        self.start_time = time.time()
        self.logger.info(f"{self.name} started...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timer and log duration."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.name} completed in {format_duration(self.elapsed)}")
        else:
            self.logger.error(f"{self.name} failed after {format_duration(self.elapsed)}")
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            raise RuntimeError("Timer has not completed yet")
        return self.elapsed