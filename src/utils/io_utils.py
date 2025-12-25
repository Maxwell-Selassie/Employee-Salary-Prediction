"""
I/O utilities for reading/writing CSV, YAML, JSON, and Joblib files.
Optimized for performance and security.
"""

import os
import json
import joblib
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object of the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        return path
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def read_csv(
    filepath: Union[str, Path],
    optimize_dtypes: bool = True,
    categorical_columns: Optional[list[str]] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Read CSV file with optimization and error handling.
    
    Args:
        filepath: Path to CSV file
        optimize_dtypes: Whether to optimize data types
        categorical_columns: Columns to convert to category dtype
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or corrupted
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        logger.info(f"Reading CSV file: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {filepath}")
        
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        
        # Optimize data types
        if optimize_dtypes:
            df = optimize_dataframe_dtypes(df, categorical_columns)
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or corrupted: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading CSV: {e}")
        raise


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    categorical_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        categorical_columns: Columns to convert to category dtype
        
    Returns:
        Optimized DataFrame
    """
    logger.info("Optimizing DataFrame dtypes...")
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert to category dtype
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - final_memory / initial_memory) * 100
    
    logger.info(f"Memory reduced by {reduction:.2f}% ({initial_memory:.2f}MB → {final_memory:.2f}MB)")
    
    return df


def write_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    create_dirs: bool = True,
    index: bool= True,
    **kwargs: Any
) -> None:
    """
    Write DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to write
        filepath: Output path
        create_dirs: Whether to create parent directories
        **kwargs: Additional arguments for df.to_csv
    """
    filepath = Path(filepath)
    
    if create_dirs:
        ensure_directory(filepath.parent)
    
    try:
        logger.info(f"Writing CSV to: {filepath}")
        df.to_csv(filepath, index=index, **kwargs)
        logger.info(f"Successfully wrote {len(df):,} rows to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")
        raise


def read_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read YAML configuration file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary of configuration
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    try:
        logger.info(f"Reading YAML file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"YAML file is empty: {filepath}")
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading YAML: {e}")
        raise


def write_yaml(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Write dictionary to YAML file.
    
    Args:
        data: Dictionary to write
        filepath: Output path
        create_dirs: Whether to create parent directories
    """
    filepath = Path(filepath)
    
    if create_dirs:
        ensure_directory(filepath.parent)
    
    try:
        logger.info(f"Writing YAML to: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Successfully wrote YAML to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write YAML: {e}")
        raise


def read_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        logger.info(f"Reading JSON file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading JSON: {e}")
        raise


def write_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    create_dirs: bool = True,
    indent: int = 2
) -> None:
    """
    Write dictionary to JSON file.
    
    Args:
        data: Dictionary to write
        filepath: Output path
        create_dirs: Whether to create parent directories
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    
    if create_dirs:
        ensure_directory(filepath.parent)
    
    try:
        logger.info(f"Writing JSON to: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Successfully wrote JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")
        raise


def load_joblib(filepath: Union[str, Path]) -> Any:
    """
    Load object from joblib file.
    
    Args:
        filepath: Path to joblib file
        
    Returns:
        Loaded object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Joblib file not found: {filepath}")
    
    try:
        logger.info(f"Loading joblib file: {filepath}")
        obj = joblib.load(filepath)
        return obj
    except Exception as e:
        logger.error(f"Failed to load joblib: {e}")
        raise


def save_joblib(
    obj: Any,
    filepath: Union[str, Path],
    create_dirs: bool = True,
    compress: int = 3
) -> None:
    """
    Save object to joblib file.
    
    Args:
        obj: Object to save
        filepath: Output path
        create_dirs: Whether to create parent directories
        compress: Compression level (0-9)
    """
    filepath = Path(filepath)
    
    if create_dirs:
        ensure_directory(filepath.parent)
    
    try:
        logger.info(f"Saving joblib to: {filepath}")
        joblib.dump(obj, filepath, compress=compress)
        logger.info(f"Successfully saved joblib to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save joblib: {e}")
        raise