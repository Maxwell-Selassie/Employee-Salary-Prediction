import pandas as pd
import numpy as np
from typing import List
from pathlib import Path


from utils.logging_mixin import LoggerMixin


# UPDATED: Inherit from LoggerMixin
class FeatureTransformer(LoggerMixin):
    """
    Apply mathematical transformations (log, sqrt, etc.).
    
    Configurable original column dropping
    """
    
    def __init__(self, config: dict):
        self.config = config['transformations']

        self.logger = self.setup_class_logger('feature_transformer', config, 'logging')
    
    def transform_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply log transformations to reduce skewness.
        
        Option to keep or drop original columns
        
        Args:
            df: Input DataFrame
            fit: Unused (kept for interface consistency)
        
        Returns:
            DataFrame with transformed features
        
        Example:
            >>> transformer = FeatureTransformer(config)
            >>> df = transformer.transform_features(df)
        """
        try:
            self.logger.info('Applying feature transformations...')
            
            log_cols = self.config.get('log_columns', [])
            drop_original = self.config.get('drop_original', True)
            
            if not log_cols:
                self.logger.info('No columns specified for log transformation')
                return df
            
            for col in log_cols:
                if col not in df.columns:
                    self.logger.warning(f'Column {col} not found for log transformation')
                    continue
                
                # Apply log1p transformation
                new_col_name = f'{col}_log'
                df[new_col_name] = np.log1p(df[col])
                
                #  Optionally drop original
                if drop_original:
                    df = df.drop(columns=[col])
                    self.logger.debug(f'Log transformed {col} → {new_col_name} (original dropped)')
                else:
                    self.logger.debug(f'Log transformed {col} → {new_col_name} (original kept)')
            
            self.logger.info('Feature transformations completed')
            return df
        
        except Exception as e:
            self.logger.error(f'Error in feature transformation: {e}', exc_info=True)
            raise