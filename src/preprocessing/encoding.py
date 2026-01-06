import pandas as pd
import numpy as np
from typing import Dict, List, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


from utils.logging_mixin import LoggerMixin

class FeatureEncoder(LoggerMixin):
    """
    Encode categorical features with proper fit/transform pattern.
    
    Handles unseen categories in dev/test
    Proper fit/transform caching
    """
    
    def __init__(self, config: dict):
        self.config = config['encoding']

        self.logger = self.setup_class_logger('feature_encoder', config, 'logging')
        

        self.seen_categories = {}
        self.encoded_columns = []
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply one-hot encoding with proper fit/transform.
        
        Handles unseen categories gracefully
        
        Args:
            df: Input DataFrame
            fit: If True, learn categories from this data
        
        Returns:
            DataFrame with encoded features
        
        Example:
            >>> encoder = FeatureEncoder(config)
            >>> train = encoder.encode_features(train_df, fit=True)
            >>> dev = encoder.encode_features(dev_df, fit=False)
        """
        try:
            self.logger.info('Encoding categorical features...')
            
            # One-hot encoding
            df = self._one_hot_encode(df, fit)
            
            self.logger.info('Feature encoding completed')
            return df
        
        except Exception as e:
            self.logger.error(f'Error encoding features: {e}', exc_info=True)
            raise
    
    def _one_hot_encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        UPDATED: One-hot encode with unseen category handling.
        
        Args:
            df: Input DataFrame
            fit: Whether to learn categories
        
        Returns:
            DataFrame with one-hot encoded features
        """
        try:
            one_hot_config = self.config.get('one_hot_columns', True)
            
            # If False/empty, skip encoding
            if not one_hot_config:
                self.logger.warning('One-hot encoding disabled')
                return df
            
            # Get categorical columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            if not categorical_cols:
                self.logger.info('No categorical columns found')
                return df
            
            if fit:
                # Learn categories from training data
                self.logger.info(f'Learning categories from {len(categorical_cols)} columns')
                
                for col in categorical_cols:
                    self.seen_categories[col] = set(df[col].dropna().unique())
                    
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    
                    # Track encoded column names
                    self.encoded_columns.extend(dummies.columns.tolist())
                    
                    self.logger.debug(
                        f'One-hot encoded {col}: {len(self.seen_categories[col])} categories '
                        f'→ {len(dummies.columns)} features'
                    )
            
            else:
                # Apply learned categories to dev/test
                for col in categorical_cols:
                    if col not in self.seen_categories:
                        self.logger.warning(
                            f'Column {col} was not seen during training, skipping'
                        )
                        continue
                    
                    # Check for unseen categories
                    current_categories = set(df[col].dropna().unique())
                    unseen = current_categories - self.seen_categories[col]
                    
                    if unseen:
                        self.logger.warning(
                            f'Unseen categories in {col}: {unseen}. '
                            f'Setting to NaN (will create all-zero encoding)'
                        )
                        # Replace unseen categories with NaN
                        df.loc[df[col].isin(unseen), col] = np.nan
                    
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    
                    # Ensure same columns as training
                    for encoded_col in self.encoded_columns:
                        if encoded_col.startswith(f'{col}_') and encoded_col not in dummies.columns:
                            dummies[encoded_col] = 0
                    
                    # Keep only columns from training
                    training_cols = [c for c in self.encoded_columns if c.startswith(f'{col}_')]
                    dummies = dummies[training_cols]
                    
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error in one-hot encoding: {e}', exc_info=True)
            raise