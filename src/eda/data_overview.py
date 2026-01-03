"""
Data Overview Module - Initial data exploration and validation
FIXED: Improved error handling, added type hints, better validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import mlflow
import sys

# NEW: Import LoggerMixin for consistent logging
from utils import read_yaml, ensure_directory, read_csv, write_json
from utils.logging_mixin import LoggerMixin  


class DataExecutionError(Exception):
    """Custom exception for data execution errors"""
    pass


class DataOverview(LoggerMixin):  
    """
    Perform initial data overview and validation.
    
    Validates data schema, converts data types, and analyzes
    numeric and categorical columns.
    """
    
    def __init__(self, config_path: str | Path):
        """
        Initialize DataOverview.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self.setup_class_logger('data_overview', self.config, 'logging')
 
    
    def _load_config(self, config_path: str | Path) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        try:
            config = read_yaml(config_path)
            # log eda yaml configuration
            mlflow.log_artifact(config_path)

            self._validate_config(config)  
            return config
        except FileNotFoundError:
            print(f'ERROR: Config file not found: {config_path}')
            sys.exit(1)
        except Exception as e:
            print(f'Failed to load config: {e}')
            sys.exit(1)
    
    # NEW FUNCTION: Config validation
    def _validate_config(self, config: Dict) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required keys are missing
        """
        required_keys = ['data', 'logging']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        # Validate data section
        if 'expected_columns' not in config['data']:
            raise ValueError("Config missing 'data.expected_columns'")
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from file path.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            file_path = Path(self.config.get('file_path', 'data/raw/Employee_Complete_Dataset.csv'))
            
            if not file_path.exists():
                raise FileNotFoundError(f'Data file not found: {file_path}')
            
            df = read_csv(file_path, optimize_dtypes=True)
            
            
            if df.empty:
                raise pd.errors.EmptyDataError('Loaded DataFrame is empty')
            
            self.logger.info(f'DataFrame successfully loaded: {df.shape[0]} rows × {df.shape[1]} columns')

            # log dataset metadata to MLflow
            mlflow.log_param('Dataset_name', 'Employee_Complete_Dataset.csv(1)')
            mlflow.log_param('n_rows', len(df))
            mlflow.log_param('n_columns', len(df.columns))
 
            return df
        
        except FileNotFoundError as e:
            self.logger.error(f'File not found: {e}')
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f'Empty data file: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Error loading CSV: {e}', exc_info=True)
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame schema and structure.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # FIXED: Better validation logic
        if len(df) == 0:
            self.logger.error('DataFrame is empty')
            raise ValueError('Cannot validate empty DataFrame')
        
        expected_columns = self.config['data']['expected_columns']
        
        try:
            # Check for missing columns
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f'Missing expected columns: {missing_cols}')
                raise ValueError(f'Missing columns: {missing_cols}')
            
            # Check for unexpected columns
            unexpected_cols = [col for col in df.columns if col not in expected_columns]
            if unexpected_cols:
                self.logger.warning(f'Unexpected columns detected: {unexpected_cols}')
                raise ValueError(f'Unexpected columns: {unexpected_cols}')
            
            # NEW: Check for duplicate columns
            if df.columns.duplicated().any():
                dup_cols = df.columns[df.columns.duplicated()].tolist()
                self.logger.error(f'Duplicate column names: {dup_cols}')
                raise ValueError(f'Duplicate columns: {dup_cols}')
            
            self.logger.info('✓ Data validation passed')
            
            # Save feature names if configured
            if self.config['data'].get('save_expected_columns', True):
                feature_names_path = self.config['data']['feature_store_path']
                write_json(expected_columns, feature_names_path, indent=4)
                self.logger.info(f'Saved feature names to {feature_names_path}')

            # log expected feature names to MLflow
            mlflow.log_artifact(feature_names_path)
        
        except ValueError as e:
            self.logger.error(f'Validation error: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected validation error: {e}', exc_info=True)
            raise

    
    def _analyze_numeric_cols(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]: 
        """
        Analyze numerical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (summary DataFrame, list of numeric columns)
        """
        self.logger.info('Analyzing numerical columns...')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID column if present
        id_column = self.config['data'].get('id_column')
        if id_column and id_column in numeric_cols:
            numeric_cols.remove(id_column)
            self.logger.debug(f'Excluded ID column: {id_column}')
        
        if not numeric_cols:
            self.logger.warning('No numerical columns found')
            return pd.DataFrame(), []
        
        # Generate summary statistics
        num_summary = df[numeric_cols].describe().T
        num_summary['range'] = num_summary['max'] - num_summary['min']
        num_summary['missing_count'] = df[numeric_cols].isnull().sum()
        num_summary['missing_pct'] = (num_summary['missing_count'] / len(df) * 100).round(2)
        
        # Add skewness and kurtosis
        num_summary['skewness'] = df[numeric_cols].skew()
        num_summary['kurtosis'] = df[numeric_cols].kurtosis()
        
        self.logger.info(f'✓ Analyzed {len(numeric_cols)} numeric columns')
        mlflow.log_param('no_of_numeric_cols', len(numeric_cols))
        
        # Save summary
        summary_path = Path(self.config.get('output', {}).get('artifacts_dir', 'artifacts')) / 'numeric_summary.csv'
        ensure_directory(summary_path.parent)
        num_summary.to_csv(summary_path)
        self.logger.info(f'Saved numeric summary to {summary_path}')

        # log numeric summary to mlflow
        mlflow.log_artifact(summary_path)
        
        return num_summary, numeric_cols
    
    def _analyze_categorical_cols(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:  
        """
        Analyze categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (summary DataFrame, list of categorical columns)
        """
        self.logger.info('Analyzing categorical columns...')
        
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        mlflow.log_param('no_of_cat_cols', len(cat_cols))
        
        if not cat_cols:
            self.logger.warning('No categorical columns found')
            return pd.DataFrame(), []
        
        # Generate summary
        cat_summary = df[cat_cols].describe(include='all').T
        cat_summary['missing_count'] = df[cat_cols].isnull().sum()
        cat_summary['missing_pct'] = (cat_summary['missing_count'] / len(df) * 100).round(2)
        
        # Add cardinality info
        cat_summary['unique_count'] = df[cat_cols].nunique()
        cat_summary['cardinality_ratio'] = (cat_summary['unique_count'] / len(df) * 100).round(2)
        
        self.logger.info(f'✓ Analyzed {len(cat_cols)} categorical columns')
        
        # Save summary
        summary_path = Path(self.config.get('output', {}).get('artifacts_dir', 'artifacts')) / 'categorical_summary.csv'
        ensure_directory(summary_path.parent)
        cat_summary.to_csv(summary_path)
        self.logger.info(f'Saved categorical summary to {summary_path}')

        # log categorical summary to MLflow
        mlflow.log_artifact(summary_path)
        
        return cat_summary, cat_cols
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return self.config
        
    
    def run_data_overview(self) -> Dict[str, Any]:
        """
        Execute all data overview functions.
        
        Returns:
            Dictionary with results from each step
        """
        results = {}
        
        self.logger.info('Starting data overview...')
        
        try:
            # Load data
            results['dataframe'] = self._load_data()
            df = results['dataframe']
            
            # Validate
            self._validate_data(df)
            results['validation'] = 'passed'
            
            # Analyze columns
            num_summary, num_cols = self._analyze_numeric_cols(df)
            results['numeric_summary'] = num_summary
            results['numeric_columns'] = num_cols
            
            cat_summary, cat_cols = self._analyze_categorical_cols(df)
            results['categorical_summary'] = cat_summary
            results['categorical_columns'] = cat_cols
            
            self.logger.info('✓ Data overview completed successfully')

            # log final results of eda overview
            mlflow.log_params(results)
            
            return results
        
        
        except Exception as e:
            self.logger.error(f'Data overview failed: {e}', exc_info=True)
            raise DataExecutionError(f'Data overview failed: {e}')