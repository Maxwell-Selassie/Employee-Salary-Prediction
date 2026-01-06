import pandas as pd
from typing import Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from utils.logging_mixin import LoggerMixin

class DuplicateHandler(LoggerMixin):

    def __init__(self, config: dict):
        self.config = config['duplicates']
        self.logger = self.setup_class_logger('duplicate_handler', config, 'logging')

    def handle_duplicates(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Detect and remove exact duplicates.
        
        Logic error (was calling .sum() on int)
        
        Args:
            df: Input DataFrame
            fit: Unused (kept for interface consistency)
        
        Returns:
            DataFrame with duplicates removed
        
        Example:
            >>> handler = DuplicateHandler(config)
            >>> df = handler.handle_duplicates(df)
        """
        try:
            self.logger.info(f'Checking for duplicates - Before: {len(df)} rows')
            
            if not self.config.get('check_duplicates', True):
                self.logger.info('Duplicate checking disabled (skipping...)')
                return df
            
            # Count duplicates
            duplicates_count = df.duplicated().sum()  
            self.logger.info(f'Exact duplicates found: {duplicates_count}')
            

            if duplicates_count > 0:
                df = df.drop_duplicates()
                self.logger.info(f'Duplicates removed - After: {len(df)} rows')
            else:
                self.logger.info('No duplicates detected')
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling duplicates: {e}', exc_info=True)
            raise