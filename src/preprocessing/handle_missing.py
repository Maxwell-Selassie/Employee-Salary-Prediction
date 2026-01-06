

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.logging_mixin import LoggerMixin


# UPDATED: Inherit from LoggerMixin
class MissingHandler(LoggerMixin):
    """
    Handle missing values with proper fit/transform pattern.
    
    Separates numeric and categorical imputation
    Caches imputation values from training set
    """
    
    def __init__(self, config: dict):
        self.config = config['missing_values']
        self.logger = self.setup_class_logger('missing_handler', config, 'logging')
        
        self.impute_values_numeric = {}
        self.impute_values_categorical = {}
    
    def handle_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values with separate strategies for numeric/categorical.
        
        Properly separates numeric and categorical columns
        Caches imputation values when fit=True
        
        Args:
            df: Input DataFrame
            fit: If True, compute and cache imputation values from this data
        
        Returns:
            DataFrame with imputed values
        
        Example:
            >>> handler = MissingHandler(config)
            >>> train = handler.handle_missing(train_df, fit=True)
            >>> dev = handler.handle_missing(dev_df, fit=False)
        """
        try:
            self.logger.info(f'Handling missing values - Before: {len(df)} rows')
            
            if not self.config.get('enabled', True):
                self.logger.warning('Missing value handling disabled (skipping...)')
                return df
            
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Handle numeric missing values
            df = self._handle_numeric_missing(df, numeric_cols, fit)
            
            # Handle categorical missing values
            df = self._handle_categorical_missing(df, categorical_cols, fit)
            
            # Report remaining missing values
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                self.logger.warning(
                    f'⚠️  {remaining_missing} missing values remain after imputation'
                )
                missing_by_col = df.isnull().sum()
                missing_by_col = missing_by_col[missing_by_col > 0]
                self.logger.warning(f'Columns with remaining nulls:\n{missing_by_col}')
            else:
                self.logger.info('✓ No missing values remaining')
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling missing values: {e}', exc_info=True)
            raise
    
    def _handle_numeric_missing(self, df: pd.DataFrame, 
                                numeric_cols: list, fit: bool) -> pd.DataFrame:
        """
        Handle numeric missing values with caching.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric column names
            fit: Whether to compute imputation values
        
        Returns:
            DataFrame with imputed numeric columns
        """
        numeric_missing = [col for col in numeric_cols if df[col].isnull().any()]
        
        if not numeric_missing:
            return df
        
        numeric_strategy = self.config.get('numeric', {}).get('strategy', 'mean')
        
        if fit:
            # Compute and cache imputation values
            self.logger.info(f'Computing {numeric_strategy} for {len(numeric_missing)} numeric columns')
            
            for col in numeric_missing:
                if numeric_strategy == 'mean':
                    self.impute_values_numeric[col] = df[col].mean()
                elif numeric_strategy == 'median':
                    self.impute_values_numeric[col] = df[col].median()
                elif numeric_strategy == 'mode':
                    mode_val = df[col].mode()
                    self.impute_values_numeric[col] = mode_val[0] if len(mode_val) > 0 else 0
                else:
                    raise ValueError(f"Unknown numeric strategy: {numeric_strategy}")
        
        # Apply imputation
        for col in numeric_missing:
            if col in self.impute_values_numeric:
                fill_value = self.impute_values_numeric[col]
                df[col].fillna(fill_value, inplace=True)
                self.logger.debug(f'Imputed {col} with {numeric_strategy}={fill_value:.2f}')
            else:
                self.logger.warning(f'No cached value for {col}, skipping imputation')
        
        return df
    
    def _handle_categorical_missing(self, df: pd.DataFrame, 
                                    categorical_cols: list, fit: bool) -> pd.DataFrame:
        """
        ✨ NEW: Handle categorical missing values with caching.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            fit: Whether to compute imputation values
        
        Returns:
            DataFrame with imputed categorical columns
        """
        categorical_missing = [col for col in categorical_cols if df[col].isnull().any()]
        
        if not categorical_missing:
            return df
        
        categorical_strategy = self.config.get('categorical', {}).get('strategy', 'mode')
        
        if fit:
            # Compute and cache imputation values
            self.logger.info(f'Computing {categorical_strategy} for {len(categorical_missing)} categorical columns')
            
            for col in categorical_missing:
                if categorical_strategy == 'mode':
                    mode_val = df[col].mode()
                    self.impute_values_categorical[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                elif categorical_strategy == 'constant':
                    self.impute_values_categorical[col] = self.config['categorical'].get('fill_value', 'Unknown')
                else:
                    raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")
        
        # Apply imputation
        for col in categorical_missing:
            if col in self.impute_values_categorical:
                fill_value = self.impute_values_categorical[col]
                df[col].fillna(fill_value, inplace=True)
                self.logger.debug(f'Imputed {col} with {categorical_strategy}={fill_value}')
            else:
                self.logger.warning(f'No cached value for {col}, skipping imputation')
        
        return df