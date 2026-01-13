"""
Feature importance analysis for trained models.
"""

import numpy as np
from typing import Dict, Any
from sklearn.inspection import permutation_importance
import logging
from utils import LoggerMixin


class FeatureImportanceAnalyzer(LoggerMixin):
    """
    Analyze feature importance using multiple methods.
    
    Attributes:
        config: Configuration dictionary
        feature_names: List of feature names
        importance_results: Dictionary storing importance results
    """
    
    def __init__(self, config: Dict[str, Any], feature_names: list):
        """
        Initialize FeatureImportanceAnalyzer.
        
        Args:
            config: Configuration dictionary
            feature_names: List of feature names
        """
        self.config = config.get('feature_importance', {})
        self.logger = self.setup_class_logger('feature_imp_analyzer', config, 'logging')
        self.feature_names = feature_names
        self.importance_results = {}
    
    def analyze(
        self,
        model,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze feature importance.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X: Features
            y: Labels
            
        Returns:
            Dictionary with importance scores
        """
        if not self.config.get('enabled', True):
            self.logger.info("Feature importance analysis disabled")
            return {}
        
        self.logger.info("\nAnalyzing feature importance...")
        self.logger.info("-"*60)
        
        results = {}
        
        # Native feature importance (for tree-based models)
        native_config = self.config.get('methods', {}).get('native', {})
        if native_config.get('enabled', True):
            applicable_models = native_config.get('applicable_to', [])
            
            if model_name in applicable_models:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Sort by importance
                    indices = np.argsort(importances)[::-1]
                    
                    results['native'] = {
                        'importances': importances.tolist(),
                        'feature_ranking': indices.tolist(),
                        'top_features': [
                            {
                                'feature': self.feature_names[i],
                                'importance': float(importances[i])
                            }
                            for i in indices[:20]
                        ]
                    }
                    
                    self.logger.info("  Native feature importance (top 10):")
                    for i in range(min(10, len(indices))):
                        idx = indices[i]
                        self.logger.info(f"    {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # Permutation importance
        perm_config = self.config.get('methods', {}).get('permutation', {})
        if perm_config.get('enabled', True):
            self.logger.info("\n  Computing permutation importance...")
            
            n_repeats = perm_config.get('n_repeats', 10)
            random_state = perm_config.get('random_state', 42)
            
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Sort by importance
            indices = np.argsort(perm_importance.importances_mean)[::-1]
            
            results['permutation'] = {
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist(),
                'feature_ranking': indices.tolist(),
                'top_features': [
                    {
                        'feature': self.feature_names[i],
                        'importance_mean': float(perm_importance.importances_mean[i]),
                        'importance_std': float(perm_importance.importances_std[i])
                    }
                    for i in indices[:20]
                ]
            }
            
            self.logger.info("  Permutation importance (top 10):")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                mean_imp = perm_importance.importances_mean[idx]
                std_imp = perm_importance.importances_std[idx]
                self.logger.info(f"    {i+1}. {self.feature_names[idx]}: {mean_imp:.4f} (+/- {std_imp:.4f})")
        
        self.importance_results = results
        
        return results