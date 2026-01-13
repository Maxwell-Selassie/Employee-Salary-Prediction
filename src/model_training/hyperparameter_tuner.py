"""
Hyperparameter tuning using Optuna.
"""

import numpy as np
from typing import Dict, Any
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from utils import LoggerMixin

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner(LoggerMixin):
    """
    Hyperparameter tuning with Optuna.
    
    Attributes:
        config: Configuration dictionary
        study: Optuna study object
        best_params: Best hyperparameters found
        best_score: Best score achieved
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HyperparameterTuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('hyperparameter_tuning', {})
        self.logger = self.setup_class_logger('hyperparameter_tuner', config, 'logging')
        self.study = None
        self.best_params = None
        self.best_score = None
    
    def _suggest_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = self.config['search_spaces'].get(model_name, {})
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            
            elif param_type == 'uniform':
                params[param_name] = trial.suggest_uniform(param_name, param_config[1], param_config[2])
            
            elif param_type == 'log_uniform':
                params[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
            
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1], param_config[2])
        
        return params
    
    def _objective_function(
        self,
        trial: optuna.Trial,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
            model_name: Model name
            X_train: Training features
            y_train: Training labels
            scoring: Scoring metric
            
        Returns:
            Score to optimize
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial, model_name)
        
        # Add fixed params (class_weight, random_state, etc.)
        if model_name == 'Ridge':
            params['random_state'] = 42
            params['max_iter'] = 1000
            model = Ridge(**params)
        
        elif model_name == 'RandomForest':
            params['random_state'] = 42
            params['n_jobs'] = -1
            model = RandomForestRegressor(**params)
        
        elif model_name == 'XGBoost':
            params['random_state'] = 42
            params['n_jobs'] = -1
            model = XGBRegressor(**params)
        
        elif model_name == 'LightGBM':
            params['random_state'] = 42
            params['class_weight'] = 'balanced'
            params['n_jobs'] = -1
            params['verbose'] = -1
            model = LGBMRegressor(**params)
        
        # Cross-validation
        cv_splits = self.config.get('cv_splits', 5)
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        return scores.mean()
    
    def tune(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with best parameters and score
        """
        self.logger.info(f"\nTuning hyperparameters for {model_name}...")
        self.logger.info("="*60)
        
        n_trials = self.config.get('n_trials', 50)
        timeout = self.config.get('timeout', 3600)
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        
        self.study = optuna.create_study(
            direction='minimize' if 'neg_' in scoring or 'error' in scoring else 'maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        self.logger.info(f"Running {n_trials} trials...")
        
        self.study.optimize(
            lambda trial: self._objective_function(trial, model_name, X_train, y_train, scoring),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"\n✓ Tuning completed")
        self.logger.info(f"  Best {scoring}: {self.best_score:.4f}")
        self.logger.info(f"  Best parameters:")
        for param, value in self.best_params.items():
            self.logger.info(f"    {param}: {value}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number
        }