import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.logging_mixin import LoggerMixin


# UPDATED: Inherit from LoggerMixin
class OutlierHandler(LoggerMixin):
    """
    Flag outliers without removing them (using IQR method).

    """
    
    def __init__(self, config: dict):
        self.config = config['outliers']
        self.logger = self.setup_class_logger('outlier_handler', config, 'logging')
        self.outlier_bounds = {}
    
    def handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Flag outliers using IQR method.
        
        Computes bounds on training set, applies to all sets.
        
        Args:
            df: Input DataFrame
            fit: If True, compute outlier bounds from this data
        
        Returns:
            DataFrame with 'is_outlier' column added
        
        Example:
            >>> handler = OutlierHandler(config)
            >>> train = handler.handle_outliers(train_df, fit=True)
            >>> dev = handler.handle_outliers(dev_df, fit=False)
        """
        try:
            self.logger.info('Processing outliers...')
            
            cols_to_flag = self.config.get('cols_to_flag', [])
            
            if not cols_to_flag:
                self.logger.warning('No columns specified for outlier detection')
                df['is_outlier'] = 0
                return df
            
            if fit:
                self.logger.info(f'Computing outlier bounds from training data...')
                self.outlier_bounds = self._compute_bounds(df, cols_to_flag)
            
            df = self._flag_outliers(df, cols_to_flag)
            
            outlier_count = (df['is_outlier'] == 1).sum()
            self.logger.info(
                f'Rows flagged as outliers: {outlier_count} '
                f'({outlier_count/len(df)*100:.2f}%)'
            )
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling outliers: {e}', exc_info=True)
            raise
    
    def _compute_bounds(self, df: pd.DataFrame, cols_to_flag: list) -> Dict[str, Dict[str, float]]:
        """
        Compute IQR bounds for each column.
        
        Args:
            df: Input DataFrame
            cols_to_flag: List of column names
        
        Returns:
            Dictionary mapping column names to {'lower': float, 'upper': float}
        """
        bounds = {}
        multiplier = self.config.get('multiplier', 1.5)
        
        for col in cols_to_flag:
            if col not in df.columns:
                self.logger.warning(f'Column {col} not found, skipping')
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
            
            self.logger.debug(
                f'{col}: bounds=({lower_bound:.2f}, {upper_bound:.2f}), '
                f'IQR={IQR:.2f}'
            )
        
        self.logger.debug(f'Computed outlier bounds for {len(bounds)} columns')
        return bounds
    
    def _flag_outliers(self, df: pd.DataFrame, cols_to_flag: list) -> pd.DataFrame:
        """
        Flag rows as outliers based on training bounds.
        
        Args:
            df: Input DataFrame
            cols_to_flag: List of column names
        
        Returns:
            DataFrame with 'is_outlier' column
        """
        is_outlier = pd.Series(0, index=df.index)
        
        for col in cols_to_flag:
            if col not in self.outlier_bounds:
                continue
            
            if col not in df.columns:
                self.logger.warning(f'Column {col} not in DataFrame, skipping')
                continue
            
            lower = self.outlier_bounds[col]['lower']
            upper = self.outlier_bounds[col]['upper']
            
            col_outliers = (df[col] < lower) | (df[col] > upper)
            is_outlier = is_outlier | col_outliers.astype(int)
        
        df['is_outlier'] = is_outlier
        return df