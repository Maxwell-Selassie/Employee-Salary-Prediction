"""
Data loading module with MLflow dataset tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import mlflow
from mlflow.data.pandas_dataset import PandasDataset

from utils import read_csv, LoggerMixin


class TrainingDataLoader(LoggerMixin):
    """
    Load preprocessed data with MLflow dataset tracking.
    
    Features:
    - Loads train/val/test splits
    - Validates data integrity
    - Logs datasets to MLflow with mlflow.log_input()
    - Tracks dataset metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrainingDataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data', {})
        self.logger = self.setup_class_logger('data_loader', config, 'logging')
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.n_features = None
        
        # Store raw dataframes for MLflow dataset logging
        self.df_train = None
        self.df_val = None
        self.df_test = None
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train/val/test datasets with MLflow tracking.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.logger.info("="*80)
        self.logger.info("LOADING TRAINING DATA")
        self.logger.info("="*80)
        
        target_col = self.config.get('target_column', 'Current_Salary_log')
        
        # Load train
        train_path = self.config.get('train_data')
        self.logger.info(f"Loading training data: {train_path}")
        self.df_train = read_csv(train_path)
        
        if target_col not in self.df_train.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        self.X_train = self.df_train.drop(columns=[target_col]).values
        self.y_train = self.df_train[target_col].values
        self.feature_names = self.df_train.drop(columns=[target_col]).columns.tolist()
        self.n_features = len(self.feature_names)
        
        self.logger.info(f"  Train: X={self.X_train.shape}, y={self.y_train.shape}")
        
        # Load validation
        val_path = self.config.get('val_data')
        self.logger.info(f"Loading validation data: {val_path}")
        self.df_val = read_csv(val_path)
        
        self.X_val = self.df_val.drop(columns=[target_col]).values
        self.y_val = self.df_val[target_col].values
        
        self.logger.info(f"  Val: X={self.X_val.shape}, y={self.y_val.shape}")
        
        # Load test
        test_path = self.config.get('test_data')
        self.logger.info(f"Loading test data: {test_path}")
        self.df_test = read_csv(test_path)
        
        self.X_test = self.df_test.drop(columns=[target_col]).values
        self.y_test = self.df_test[target_col].values
        
        self.logger.info(f"  Test: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # Validate shapes
        if not (self.X_train.shape[1] == self.X_val.shape[1] == self.X_test.shape[1]):
            raise ValueError("Feature count mismatch between datasets")
        
        self.logger.info(f"✓ Data loaded successfully")
        self.logger.info(f"  Features: {self.n_features}")
        self.logger.info(f"  Total samples: {len(self.X_train) + len(self.X_val) + len(self.X_test)}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def log_datasets_to_mlflow(self) -> None:
        """
        Log datasets to MLflow using mlflow.log_input().
        
        This tracks:
        - Dataset schema
        - Dataset statistics
        - Data lineage
        """
        self.logger.info("Logging datasets to MLflow...")
        
        try:
            # Create MLflow datasets
            train_dataset = mlflow.data.from_pandas(
                self.df_train,
                source=self.config.get('train_data'),
                name="train_data"
            )
            
            val_dataset = mlflow.data.from_pandas(
                self.df_val,
                source=self.config.get('val_data'),
                name="val_data"
            )
            
            test_dataset = mlflow.data.from_pandas(
                self.df_test,
                source=self.config.get('test_data'),
                name="test_data"
            )
            
            # Log datasets to current MLflow run
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")
            mlflow.log_input(test_dataset, context="testing")
            
            self.logger.info("✓ Datasets logged to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log datasets to MLflow: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata for logging.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test),
            'target_column': self.config.get('target_column'),
            'train_mean': float(self.y_train.mean()),
            'train_std': float(self.y_train.std()),
            'val_mean': float(self.y_val.mean()),
            'val_std': float(self.y_val.std())
        }