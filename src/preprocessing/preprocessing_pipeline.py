import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Existing imports
from preprocessing.data_splitter import DataSplitter
from preprocessing.handle_missing import MissingHandler
from preprocessing.handle_duplicates import DuplicateHandler
from preprocessing.handle_outliers import OutlierHandler
from preprocessing.encoding import FeatureEncoder
from preprocessing.feature_transformations import FeatureTransformer

# NEW: Import new modules
from preprocessing.config_validator import ConfigValidator
from preprocessing.data_validator import DataValidator

from utils import (
    ensure_directory, setup_logger, Timer, 
    read_yaml, write_csv, read_csv
)


class PreprocessingPipeline:
    """
    Orchestrate all preprocessing steps with zero data leakage.
    
    Encoding now happens AFTER split
    UPDATED: Data validation at multiple stages
    NEW: Config validation before execution
    """
    
    def __init__(self):
        self.logger = setup_logger(
            name='preprocessing_pipeline',
            log_dir='logs/'
        )
        
        # Load and validate config
        self.config = self._load_and_validate_config()
        
        self.config_validator = ConfigValidator()
        self.data_validator = DataValidator(self.config)
        
        # Initialize preprocessors
        self.splitter = DataSplitter(self.config)
        self.missing_handler = MissingHandler(self.config)
        self.duplicate_handler = DuplicateHandler(self.config)
        self.outlier_handler = OutlierHandler(self.config)
        self.feature_encoder = FeatureEncoder(self.config)
        self.feature_transformer = FeatureTransformer(self.config)
    
    def _load_and_validate_config(self, 
                                config_path: str = 'config/preprocessing_config.yaml') -> Dict:
        """
        NEW: Load and validate configuration.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Validated configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = read_yaml(config_path)
            self.logger.info(f'Configuration loaded from {config_path}')
            
            # Validate config structure
            validator = ConfigValidator()
            validator.validate_preprocessing_config(config)
            
            return config
        
        except Exception as e:
            self.logger.error(f'Error loading/validating config: {e}', exc_info=True)
            raise
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data with validation.
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data is invalid
        """
        try:
            file_path = Path(self.config['data'].get('file_path', 'data/raw/Employee_Complete_Dataset.csv'))
            
            if not file_path.exists():
                raise FileNotFoundError(f'File not found: {file_path}')
            
            self.logger.info(f'Loading data from {file_path}')
            df = read_csv(file_path, optimize_dtypes=True)
            
            self.logger.info(f'DataFrame loaded: {df.shape}')
            
            #  Validate loaded data
            if 'expected_columns' in self.config:
                self.data_validator.validate_schema(df, self.config['expected_columns'])
            
            #  Validate data integrity
            self.data_validator.validate_integrity(df)
            
            #  Validate ranges if specified
            if 'value_ranges' in self.config:
                self.data_validator.validate_ranges(df, self.config['value_ranges'])
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error loading data: {e}', exc_info=True)
            raise
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns (e.g., IDs).
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with columns dropped
        """
        cols_to_drop = self.config.get('columns_to_drop', [])
        
        if not cols_to_drop:
            return df
        
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            self.logger.info(f'Dropped columns: {existing_cols}')
        
        return df
    
    def fit_transform(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline with proper ordering.
        
        Key changes:
        - Encoding happens AFTER split (not before)
        - Enhanced validation at each stage
        - Better error handling and logging
        
        Returns:
            Tuple of (train_set, dev_set, test_set)
        
        Raises:
            Various exceptions if preprocessing fails
        """
        try:
            self.logger.info('=' * 80)
            self.logger.info('STARTING PREPROCESSING PIPELINE')
            self.logger.info('=' * 80)
            
            # ================================================================
            # STAGE 0: Load Data
            # ================================================================
            self.logger.info('\n[Stage 0] Loading Data...')
            df = self._load_data()
            
            # ================================================================
            # STAGE 1: Pre-Split Cleaning
            # ================================================================
            self.logger.info('\n[Stage 1] Pre-split Data Cleaning...')
            
            initial_rows = len(df)
            
            df = self.duplicate_handler.handle_duplicates(df)
            df = self.missing_handler.handle_missing(df, fit=True)  
            
            final_rows = len(df)
            self.logger.info(
                f'Cleaning complete: {initial_rows} → {final_rows} rows '
                f'({(initial_rows - final_rows) / initial_rows * 100:.1f}% removed)'
            )
            
            # ================================================================
            # STAGE 2: Split Data
            # ================================================================
            self.logger.info('\n[Stage 2] Splitting Data...')
            train_set, dev_set, test_set = self.splitter.split_data(df)
            
            #  Validate splits
            self.logger.info('Validating data splits...')
            if train_set.shape[1] != dev_set.shape[1] or train_set.shape[1] != test_set.shape[1]:
                raise ValueError('Feature count mismatch across splits')
            
            # ================================================================
            # STAGE 3: Fit Transformers on Training Set
            # ================================================================
            self.logger.info('\n[Stage 3] Fitting Transformers on Training Data...')
            
            train_set = self.outlier_handler.handle_outliers(train_set, fit=True)
            # Encoding now happens AFTER split
            train_set = self.feature_encoder.encode_features(train_set, fit=True)
            train_set = self.feature_transformer.transform_features(train_set, fit=True)
            train_set = self._drop_columns(train_set)
            
            # ================================================================
            # STAGE 4: Transform Dev Set
            # ================================================================
            self.logger.info('\n[Stage 4] Transforming Dev Set...')
            
            dev_set = self.outlier_handler.handle_outliers(dev_set, fit=False)
            dev_set = self.feature_encoder.encode_features(dev_set, fit=False)
            dev_set = self.feature_transformer.transform_features(dev_set, fit=False)
            dev_set = self._drop_columns(dev_set)
            
            # ================================================================
            # STAGE 5: Transform Test Set
            # ================================================================
            self.logger.info('\n[Stage 5] Transforming Test Set...')
            
            test_set = self.outlier_handler.handle_outliers(test_set, fit=False)
            test_set = self.feature_encoder.encode_features(test_set, fit=False)
            test_set = self.feature_transformer.transform_features(test_set, fit=False)
            test_set = self._drop_columns(test_set)
            
            # ================================================================
            # STAGE 6: Final Validation
            # ================================================================
            self.logger.info('\n[Stage 6] Final Validation...')
            
            #  Validate processed data
            self.data_validator.validate_processed_data(train_set, dev_set, test_set)
            
            # ================================================================
            # STAGE 7: Save Outputs
            # ================================================================
            self.logger.info('\n[Stage 7] Saving Outputs...')
            self._save_datasets(train_set, dev_set, test_set)
            self._save_pipeline()
            #  Save processing metadata
            self._save_metadata(train_set, dev_set, test_set)
            
            self.logger.info('\n' + '=' * 80)
            self.logger.info('PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY')
            self.logger.info('=' * 80)
            
            return train_set, dev_set, test_set
        
        except Exception as e:
            self.logger.error(f'Pipeline failed: {e}', exc_info=True)
            raise
    
    def _save_datasets(self, train: pd.DataFrame, dev: pd.DataFrame, 
                    test: pd.DataFrame) -> None:
        """
        Save preprocessed datasets.
        
        Args:
            train: Training set
            dev: Development set
            test: Test set
        """
        try:
            processed_dir = Path(self.config['output']['processed_dir'])
            ensure_directory(processed_dir)
            
            write_csv(train, processed_dir / 'train_set.csv')
            write_csv(dev, processed_dir / 'dev_set.csv')
            write_csv(test, processed_dir / 'test_set.csv')
            
            self.logger.info(f'✓ All datasets saved to {processed_dir}')
        
        except Exception as e:
            self.logger.error(f'Error saving datasets: {e}', exc_info=True)
            raise
    
    def _save_pipeline(self) -> None:
        """
        Save pipeline with version info.
        """
        try:
            pipeline_path = Path(self.config['output']['pipeline_file'])
            ensure_directory(pipeline_path.parent)
            
            import sklearn
            from datetime import datetime
            
            pipeline_obj = {
                'version': '2.0-fixed',  
                'created_at': datetime.now().isoformat(),  
                'sklearn_version': sklearn.__version__,  
                'pandas_version': pd.__version__,
                'outlier_handler': self.outlier_handler,
                'encoder': self.feature_encoder,
                'transformer': self.feature_transformer,
                'missing_handler': self.missing_handler,
                'config': self.config
            }
            
            joblib.dump(pipeline_obj, pipeline_path)
            self.logger.info(f'✓ Pipeline saved to {pipeline_path}')
        
        except Exception as e:
            self.logger.error(f'Error saving pipeline: {e}', exc_info=True)
            raise
    
    def _save_metadata(self, train: pd.DataFrame, dev: pd.DataFrame, 
                    test: pd.DataFrame) -> None:
        """
         Save preprocessing metadata for reproducibility.
        
        Args:
            train: Training set
            dev: Development set
            test: Test set
        """
        try:
            from datetime import datetime
            import json
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0-fixed',
                'shapes': {
                    'train': list(train.shape),
                    'dev': list(dev.shape),
                    'test': list(test.shape)
                },
                'features': {
                    'total': int(train.shape[1]),
                    'names': train.columns.tolist()
                },
                'target_distribution': {
                    'train': train['stroke'].value_counts().to_dict() if 'stroke' in train.columns else None,
                    'dev': dev['stroke'].value_counts().to_dict() if 'stroke' in dev.columns else None,
                    'test': test['stroke'].value_counts().to_dict() if 'stroke' in test.columns else None
                },
                'data_quality': {
                    'train_missing': int(train.isnull().sum().sum()),
                    'dev_missing': int(dev.isnull().sum().sum()),
                    'test_missing': int(test.isnull().sum().sum())
                }
            }
            
            metadata_path = Path(self.config['output']['processed_dir']) / 'preprocessing_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f'✓ Metadata saved to {metadata_path}')
        
        except Exception as e:
            self.logger.warning(f'Could not save metadata: {e}')


def main():
    """
    Main execution function.
    
    Returns:
        Tuple of (train_set, dev_set, test_set)
    """
    try:
        # Initialize pipeline
        pipeline = PreprocessingPipeline()
        
        # Run preprocessing
        train_set, dev_set, test_set = pipeline.fit_transform()
        
        print('\n✓ Preprocessing completed successfully')
        print(f'Train shape: {train_set.shape}')
        print(f'Dev shape: {dev_set.shape}')
        print(f'Test shape: {test_set.shape}')
        
        return train_set, dev_set, test_set
    
    except Exception as e:
        print(f'\n❌ Pipeline execution failed: {e}')
        raise


if __name__ == '__main__':
    train, dev, test = main()