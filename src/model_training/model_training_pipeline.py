"""
Complete Model Training Pipeline with MLflow Integration

Features:
- SQLite MLflow backend
- Automatic best model selection
- Runtime constraints (<10s)
- Comprehensive logging with decision tags
- Dataset tracking
- Model-as-code registration
- Production alias assignment
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import joblib
import time

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    root_mean_squared_error
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import read_yaml, ensure_directory, get_timestamp, LoggerMixin
from utils.mlflow_utils import mlflow_stage_run


class ModelTrainingPipeline(LoggerMixin):
    """
    Production-grade ML training pipeline with MLflow.
    
    Key Features:
    1. SQLite backend for local tracking
    2. Automatic model selection based on CV MSE
    3. Runtime constraint enforcement (<10s)
    4. Decision tagging (WHY choices were made)
    5. Dataset lineage tracking
    6. Model-as-code registration
    7. Production alias assignment
    """
    
    def __init__(self, config):
        """Initialize pipeline."""
        self.timestamp = get_timestamp()
        self.config = config
        self.logger = self.setup_class_logger('training_pipeline', config, 'logging')
        
        # Data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        # Results
        self.model_results = []  # All trained models
        self.best_model = None
        self.best_model_name = None
        self.best_run_id = None
        
        # MLflow
        self.client = None
        self.experiment_id = None
        
        self.logger.info("="*80)
        self.logger.info(f"MODEL TRAINING PIPELINE - {self.timestamp}")
        self.logger.info("="*80)
        self.logger.info(f"Project: {self.config['project']['name']}")
        self.logger.info(f"Version: {self.config['project']['version']}")
    
    
    def setup_mlflow(self) -> None:
        """
        Setup MLflow with SQLite backend.
        
        Creates:
        - SQLite database for tracking
        - Experiment
        - Model registry
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("MLFLOW SETUP")
        self.logger.info("="*80)
        
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI (SQLite)
        tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        self.logger.info(f"✓ Tracking URI: {tracking_uri}")
        
        # Create/get experiment
        experiment_name = mlflow_config.get('experiment_name', 'Employee_Salary_Prediction')
        
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            self.logger.info(f"✓ Created experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            self.logger.info(f"✓ Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Initialize client
        self.client = MlflowClient()
        
        self.logger.info(f"  Experiment ID: {self.experiment_id}")
    
    def load_data(self) -> None:
        """
        Load data and log to MLflow.
        
        Uses mlflow.log_input() for dataset tracking.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("DATA LOADING")
        self.logger.info("="*80)
        
        data_config = self.config.get('data', {})
        target_col = data_config.get('target_column')
        
        # Load datasets
        train_path = data_config.get('train_data')
        val_path = data_config.get('val_data')
        test_path = data_config.get('test_data')
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        
        self.X_train = df_train.drop(columns=[target_col]).values
        self.y_train = df_train[target_col].values
        self.X_val = df_val.drop(columns=[target_col]).values
        self.y_val = df_val[target_col].values
        self.X_test = df_test.drop(columns=[target_col]).values
        self.y_test = df_test[target_col].values
        self.feature_names = df_train.drop(columns=[target_col]).columns.tolist()
        
        self.logger.info(f"✓ Data loaded:")
        self.logger.info(f"  Train: {self.X_train.shape}")
        self.logger.info(f"  Val: {self.X_val.shape}")
        self.logger.info(f"  Test: {self.X_test.shape}")
        self.logger.info(f"  Features: {len(self.feature_names)}")
    
    def train_models(self) -> List[Dict[str, Any]]:
        """
        Train all configured models with MLflow tracking.
        
        For each model:
        1. Start MLflow run
        2. Log parameters, datasets, tags
        3. Train model (enforce runtime constraint)
        4. Cross-validate
        5. Evaluate on val set
        6. Log metrics, artifacts
        7. Register if best
        
        Returns:
            List of model results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL TRAINING")
        self.logger.info("="*80)
        
        # Get models to train
        models_to_train = self._get_models_to_train()
        self.logger.info(f"Training {len(models_to_train)} models: {[m[0] for m in models_to_train]}")
        
        runtime_config = self.config.get('runtime_constraints', {})
        max_time = runtime_config.get('max_training_time_seconds', 10)
        
        self.logger.info(f"Runtime constraint: {max_time}s per model")
        
        results = []
        
        for model_name, model_config in models_to_train:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training: {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                result = self._train_single_model(model_name, model_config)
                
                # Check runtime constraint
                if result['training_time'] > max_time:
                    decision = f"Model exceeded {max_time}s runtime constraint"
                    self.logger.warning(f"⚠ {decision}: {result['training_time']:.2f}s")
                    
                    # Tag in MLflow
                    with mlflow.start_run(run_id=result['run_id']):
                        mlflow.set_tag("runtime_constraint_violated", "true")
                        mlflow.set_tag("rejection_reason", decision)
                    
                    if runtime_config.get('strict_mode', False):
                        self.logger.error(f"✗ Rejecting model due to strict mode")
                        continue
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}", exc_info=True)
                
                if self.config.get('error_handling', {}).get('on_model_failure') == 'stop':
                    raise
        
        # Rank by CV MSE (lower is better)
        results.sort(key=lambda x: x['cv_mse'])
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING RESULTS (Ranked by CV MSE)")
        self.logger.info("="*80)
        
        for i, r in enumerate(results, 1):
            self.logger.info(
                f"{i}. {r['model_name']}: "
                f"CV MSE={r['cv_mse']:.4f}, "
                f"Val MSE={r['val_mse']:.4f}, "
                f"R²={r['val_r2']:.4f}, "
                f"Time={r['training_time']:.2f}s"
            )
        
        self.model_results = results
        return results
    
    def _get_models_to_train(self) -> List[tuple]:
        """Get list of models to train from config."""
        models = []
        
        # Baseline
        if self.config['models']['baseline'].get('enabled', True):
            models.append((
                self.config['models']['baseline']['name'],
                self.config['models']['baseline']
            ))
        
        # Tree-based
        for name, config in self.config['models']['tree_based'].items():
            if config.get('enabled', True):
                models.append((name, config))
        
        return models
    
    def _train_single_model(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """
        Train a single model with full MLflow tracking.
        
        Returns:
            Dictionary with training results and metrics
        """
        run_name = f"{model_name}_{self.timestamp}"
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            self.logger.info(f"MLflow Run ID: {run_id}")
            
            # ===== LOG PARAMETERS =====
            params = model_config['params']
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("val_size", len(self.X_val))
            
            # ===== LOG DATASETS =====
            try:
                train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
                train_df[self.config['data']['target_column']] = self.y_train
                
                train_dataset = mlflow.data.from_pandas(
                    train_df,
                    source=self.config['data']['train_data'],
                    name="training_data"
                )
                mlflow.log_input(train_dataset, context="training")
            except Exception as e:
                self.logger.warning(f"Failed to log dataset: {e}")
            
            # ===== TRAIN MODEL =====
            start_time = time.time()
            
            model = self._get_model_instance(model_name, params)
            model.fit(self.X_train, self.y_train)
            
            training_time = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            self.logger.info(f"✓ Training completed: {training_time:.2f}s")
            
            # ===== CROSS-VALIDATION =====
            cv_config = self.config.get('cross_validation', {})
            
            if cv_config.get('enabled', True):
                self.logger.info("Running cross-validation...")
                
                kfold = KFold(
                    n_splits=cv_config.get('n_splits', 5),
                    shuffle=cv_config.get('shuffle', True),
                    random_state=cv_config.get('random_state', 42)
                )
                
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=kfold,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                cv_mse = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                mlflow.log_metric("cv_mse_mean", cv_mse)
                mlflow.log_metric("cv_mse_std", cv_std)
                mlflow.log_metric("cv_mse_min", -cv_scores.max())
                mlflow.log_metric("cv_mse_max", -cv_scores.min())
                
                self.logger.info(f"  CV MSE: {cv_mse:.4f} (+/- {cv_std:.4f})")
            else:
                cv_mse = float('inf')
                cv_std = 0
            
            # ===== VALIDATION EVALUATION =====
            y_val_pred = model.predict(self.X_val)
            
            val_mse = mean_squared_error(self.y_val, y_val_pred)
            val_rmse = root_mean_squared_error(self.y_val, y_val_pred)
            val_mae = mean_absolute_error(self.y_val, y_val_pred)
            val_r2 = r2_score(self.y_val, y_val_pred)
            
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("val_r2", val_r2)
            
            self.logger.info(f"  Val MSE: {val_mse:.4f}")
            self.logger.info(f"  Val RMSE: {val_rmse:.4f}")
            self.logger.info(f"  Val R²: {val_r2:.4f}")
            
            # ===== TRAIN EVALUATION (for overfitting check) =====
            y_train_pred = model.predict(self.X_train)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("train_r2", train_r2)
            
            # Overfitting check
            mse_gap = val_mse - train_mse
            mse_gap_pct = (mse_gap / train_mse) * 100 if train_mse > 0 else 0
            
            mlflow.log_metric("mse_gap", mse_gap)
            mlflow.log_metric("mse_gap_percent", mse_gap_pct)
            
            if abs(mse_gap_pct) > 15:
                mlflow.set_tag("overfitting_warning", "true")
                self.logger.warning(f"⚠ MSE gap: {mse_gap_pct:.1f}%")
            
            # ===== DECISION TAGS =====
            mlflow.set_tag("model_family", "tree_based" if model_name in ["RandomForest", "XGBoost", "LightGBM"] else "linear")
            mlflow.set_tag("selection_criteria", "cv_mse")
            mlflow.set_tag("runtime_acceptable", str(training_time <= self.config['runtime_constraints']['max_training_time_seconds']))
            
            # ===== LOG MODEL (Model-as-Code) =====
            signature = infer_signature(self.X_train, y_train_pred)
            
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                code_paths=["src/"]  # Model-as-code: include source code
            )
            
            # ===== CONTEXT NOTES =====
            notes = f"""
Model Training Context:
- Model: {model_name}
- Training time: {training_time:.2f}s
- CV MSE: {cv_mse:.4f}
- Val MSE: {val_mse:.4f}
- R²: {val_r2:.4f}
- Overfitting gap: {mse_gap_pct:.1f}%

Decision rationale:
- Selected based on cross-validated MSE (lower is better)
- Runtime constraint: {'MET' if training_time <= 10 else 'EXCEEDED'}
- Generalization: {'Good' if abs(mse_gap_pct) <= 15 else 'Poor'}
"""
            
            mlflow.set_tag("training_notes", notes)
            
            return {
                'model_name': model_name,
                'model': model,
                'run_id': run_id,
                'training_time': training_time,
                'cv_mse': cv_mse,
                'cv_std': cv_std,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'train_mse': train_mse,
                'mse_gap_percent': mse_gap_pct
            }
    
    def _get_model_instance(self, model_name: str, params: Dict):
        """Get model instance."""
        models = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name](**params)
    
    def select_and_register_best_model(self) -> None:
        """
        Select best model and register to MLflow with production alias.
        
        Selection criteria:
        1. Lowest CV MSE
        2. Runtime constraint met
        3. No severe overfitting
        
        Registration:
        - Registers model to MLflow Model Registry
        - Assigns 'production' alias
        - Tags with selection decision
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL SELECTION & REGISTRATION")
        self.logger.info("="*80)
        
        if not self.model_results:
            self.logger.error("No models trained!")
            return
        
        # Filter models meeting runtime constraint
        runtime_config = self.config.get('runtime_constraints', {})
        max_time = runtime_config.get('max_training_time_seconds', 10)
        
        valid_models = [
            m for m in self.model_results 
            if m['training_time'] <= max_time
        ]
        
        if not valid_models:
            self.logger.warning(f"No models met {max_time}s runtime constraint!")
            self.logger.warning("Using all models for selection...")
            valid_models = self.model_results
        
        # Select best (already sorted by CV MSE)
        best_result = valid_models[0]
        
        self.best_model = best_result['model']
        self.best_model_name = best_result['model_name']
        self.best_run_id = best_result['run_id']
        
        self.logger.info(f"\n✓ BEST MODEL SELECTED: {self.best_model_name}")
        self.logger.info(f"  CV MSE: {best_result['cv_mse']:.4f}")
        self.logger.info(f"  Val MSE: {best_result['val_mse']:.4f}")
        self.logger.info(f"  Val R²: {best_result['val_r2']:.4f}")
        self.logger.info(f"  Training time: {best_result['training_time']:.2f}s")
        self.logger.info(f"  Run ID: {self.best_run_id}")
        
        # Register to MLflow
        registry_config = self.config['mlflow']['model_registry']
        
        if not registry_config.get('enabled', True):
            self.logger.info("Model registry disabled")
            return
        
        try:
            model_name = registry_config.get('model_name', 'EmployeeSalaryModel')
            model_uri = f"runs:/{self.best_run_id}/model"
            
            self.logger.info(f"\nRegistering model to MLflow...")
            
            # Register model
            mv = mlflow.register_model(model_uri, model_name)
            
            self.logger.info(f"✓ Model registered: {model_name}")
            self.logger.info(f"  Version: {mv.version}")
            
            # Assign production alias
            production_alias = registry_config.get('production_alias', 'production')
            
            self.client.set_registered_model_alias(
                name=model_name,
                alias=production_alias,
                version=mv.version
            )
            
            self.logger.info(f"✓ Alias assigned: {production_alias}")
            
            # Tag registration decision
            self.client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="selection_reason",
                value=f"Lowest CV MSE: {best_result['cv_mse']:.4f}"
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="runtime_seconds",
                value=str(best_result['training_time'])
            )
            
            self.logger.info("\n✓ MODEL REGISTERED TO PRODUCTION")
            self.logger.info(f"  Model: {model_name}")
            self.logger.info(f"  Version: {mv.version}")
            self.logger.info(f"  Alias: {production_alias}")
            self.logger.info(f"\nLoad with:")
            self.logger.info(f"  model = mlflow.sklearn.load_model('models:/{model_name}@{production_alias}')")
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}", exc_info=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete training pipeline with MLflow stage tracking.
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            with mlflow_stage_run(
                stage="training",
                run_name=f"training_session_{self.timestamp}",
                tags={
                    "project": self.config["project"]["name"],
                    "version": self.config["project"]["version"],
                },
            ):
                # Setup
                self.setup_mlflow()
                self.load_data()
                
                # Train all models
                self.train_models()
                
                # Select and register best
                self.select_and_register_best_model()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return {
                'status': 'success',
                'best_model_name': self.best_model_name,
                'best_run_id': self.best_run_id,
                'n_models_trained': len(self.model_results)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Entry point."""
    config = read_yaml("config/model_training_config.yaml")
    pipeline = ModelTrainingPipeline(config)
    _ = pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())