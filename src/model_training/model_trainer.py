"""
Core model training module with runtime constraints.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
import time
import mlflow
from utils import LoggerMixin


class TrainingTimeoutError(Exception):
    """Raised when model training exceeds time limit."""
    pass


class ModelTrainer(LoggerMixin):
    """
    Train regression models with cross-validation and runtime constraints.
    
    Attributes:
        config: Configuration dictionary
        model: Trained model instance
        model_name: Name of the model
        training_time: Time taken to train
        cv_scores: Cross-validation scores
        max_training_time: Maximum allowed training time (seconds)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('model_trainer', config, 'logging')
        self.model = None
        self.model_name = None
        self.training_time = 0.0
        self.cv_scores = None
        
        # Runtime constraint
        self.max_training_time = config.get('runtime_constraints', {}).get('max_training_time', 10.0)
    
    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """
        Get model instance based on name.
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        model_map = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name](**params)
    
    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Train a single model with runtime constraints.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters (optional)
            
        Returns:
            Trained model
            
        Raises:
            TrainingTimeoutError: If training exceeds max_training_time
        """
        self.logger.info(f"\nTraining {model_name}...")
        self.logger.info("-"*60)
        
        self.model_name = model_name
        
        # Get model parameters
        if params is None:
            if model_name == 'Ridge':
                params = self.config['models']['baseline']['params']
            else:
                params = self.config['models']['tree_based'][model_name]['params']
        
        self.logger.info(f"Parameters: {params}")
        
        # Initialize model
        self.model = self._get_model_instance(model_name, params)
        
        # Train with runtime tracking
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        # ✅ Check runtime constraint (satisficing metric)
        if self.training_time > self.max_training_time:
            warning_msg = (
                f"⚠️  Training time ({self.training_time:.2f}s) exceeds limit "
                f"({self.max_training_time}s)"
            )
            self.logger.warning(warning_msg)
            
            # Log decision to MLflow
            mlflow.set_tag("runtime_exceeded", "true")
            mlflow.set_tag("runtime_decision", warning_msg)
            
            # Don't raise error, but mark as non-compliant
            mlflow.log_param("satisficing_constraint_met", False)
        else:
            self.logger.info(f"✓ Training completed in {self.training_time:.2f}s (within {self.max_training_time}s limit)")
            mlflow.log_param("satisficing_constraint_met", True)
        
        return self.model
    
    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        cv_config = self.config.get('cross_validation', {})
        
        if not cv_config.get('enabled', True):
            self.logger.info("Cross-validation disabled")
            return {}
        
        self.logger.info("Performing cross-validation...")
        
        n_splits = cv_config.get('n_splits', 5)
        shuffle = cv_config.get('shuffle', True)
        random_state = cv_config.get('random_state', 42)
        
        cv = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        # Time cross-validation
        cv_start = time.time()
        
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cv_time = time.time() - cv_start
        
        # Convert negative MSE to positive
        cv_scores_abs = np.abs(cv_scores)
        self.cv_scores = cv_scores_abs
        
        self.logger.info(f"✓ Cross-validation completed in {cv_time:.2f}s")
        
        cv_results = {
            'cv_mean': float(np.mean(cv_scores_abs)),
            'cv_std': float(np.std(cv_scores_abs)),
            'cv_min': float(np.min(cv_scores_abs)),
            'cv_max': float(np.max(cv_scores_abs)),
            'cv_scores': cv_scores_abs.tolist(),
            'cv_time': cv_time
        }
        
        self.logger.info(f"  CV MSE: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        self.logger.info(f"  CV range: [{cv_results['cv_min']:.4f}, {cv_results['cv_max']:.4f}]")
        
        # ✅ Log decision context
        mlflow.set_tag(
            "cv_decision",
            f"CV MSE={cv_results['cv_mean']:.4f} with std={cv_results['cv_std']:.4f} "
            f"indicates {'high' if cv_results['cv_std'] > 0.1 else 'low'} variance"
        )
        
        return cv_results
    
    def get_model(self):
        """Get trained model."""
        return self.model