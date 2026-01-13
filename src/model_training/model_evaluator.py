"""
Comprehensive model evaluation with multiple metrics.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
)
import logging
from utils import LoggerMixin


class ModelEvaluator(LoggerMixin):
    """
    Evaluate trained models with comprehensive metrics.
    
    Attributes:
        config: Configuration dictionary
        metrics_results: Dictionary storing evaluation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('metrics', {})
        self.logger = self.setup_class_logger('model_evaluator', config, 'logging')
        self.metrics_results = {}
    
    def evaluate(
        self,
        model,
        X: np.ndarray,
        y_true: np.ndarray,
        dataset_name: str = 'validation'
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X: Features
            y_true: True labels
            dataset_name: Name of dataset (train/val/test)
            
        Returns:
            Dictionary with all metrics
        """
        self.logger.info(f"\nEvaluating on {dataset_name} set...")
        self.logger.info("-"*60)
        
        # Predictions
        y_pred = model.predict(X)

        
        results = {}

        # Regression metrics
        regression_metrics = self.config.get('regression_metrics', [])
        
        for metric_name in regression_metrics:
            if metric_name == 'mean_squared_error':
                score = mean_squared_error(y_true, y_pred)
            elif metric_name == 'mean_absolute_error':
                score = mean_absolute_error(y_true, y_pred, zero_division=0)
            elif metric_name == 'median_absolute_error':
                score = median_absolute_error(y_true, y_pred, zero_division=0)
            elif metric_name == 'root_mean_squared_error':
                score = root_mean_squared_error(y_true, y_pred, zero_division=0)
            elif metric_name == 'mean_absolute_percentage_error':
                score = mean_absolute_percentage_error(y_true, y_pred, zero_division=0)
            elif metric_name == 'r2_score':
                score = r2_score(y_true, y_pred)
            else:
                continue
            
            results[metric_name] = float(score)
            self.logger.info(f"  {metric_name}: {score:.4f}")
        
        
        self.metrics_results[dataset_name] = results
        
        return results
    
    def compare_train_val(
        self,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare training and validation metrics for generalization.
        
        Args:
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("\nGeneralization Analysis:")
        self.logger.info("-"*60)
        
        primary_metric = self.config.get('primary_metric', 'mean_squared_error')
        
        train_score = train_metrics.get(primary_metric, 0)
        val_score = val_metrics.get(primary_metric, 0)
        
        gap = train_score - val_score
        gap_pct = (gap / train_score * 100) if train_score > 0 else 0
        
        self.logger.info(f"  Train {primary_metric}: {train_score:.4f}")
        self.logger.info(f"  Val {primary_metric}: {val_score:.4f}")
        self.logger.info(f"  Gap: {gap:.4f} ({gap_pct:.2f}%)")
        
        if gap > 0.10:
            self.logger.warning(f"  ⚠ Large gap detected - possible overfitting!")
        elif gap < -0.05:
            self.logger.warning(f"  ⚠ Negative gap - unusual, check data!")
        else:
            self.logger.info(f"  ✓ Acceptable generalization")
        
        return {
            'train_score': train_score,
            'val_score': val_score,
            'gap': gap,
            'gap_percentage': gap_pct,
            'generalization_status': 'good' if abs(gap) <= 0.10 else 'poor'
        }