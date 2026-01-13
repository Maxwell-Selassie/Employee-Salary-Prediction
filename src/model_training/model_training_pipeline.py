"""
Main Model Training Pipeline with MLflow Integration (Train/val Only)
Production-grade ML training with experiment tracking.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    read_yaml, ensure_directory, get_timestamp,
    Timer, setup_logger
)
from model_training import (
    TrainingDataLoader,
    ModelTrainer,
    HyperparameterTuner,
    ModelEvaluator,
    FeatureImportanceAnalyzer,
    ModelValidator
)


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class ModelTrainingPipeline:
    """
    Complete model training pipeline with MLflow tracking (Train/val split).
    
    Pipeline Stages:
    1. Data loading (train/val)
    2. MLflow experiment setup
    3. Baseline model training (evaluated on val set)
    4. Model evaluation & selection
    5. Hyperparameter tuning (top 3 models, using CV on train set)
    6. Final evaluation on val set
    7. Model registration
    """
    
    def __init__(self, config_path: str = "config/model_training_config.yaml"):
        """
        Initialize Model Training Pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.timestamp = get_timestamp()
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = None
        self.trainer = None
        self.tuner = None
        self.evaluator = None
        self.feature_analyzer = None
        self.validator = None
        
        # Data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = None
        
        # Results
        self.baseline_results = []
        self.tuned_results = []
        self.best_model = None
        self.best_model_name = None
        self.best_model_score = 0.0
        
        # MLflow
        self.experiment_id = None
        
        self.logger.info("="*80)
        self.logger.info(f"MODEL TRAINING PIPELINE INITIALIZED - {self.timestamp}")
        self.logger.info("="*80)
        self.logger.info(f"Author: {self.config['project']['name']}")
        self.logger.info(f"Version: {self.config['project']['version']}")
        self.logger.info(f"Note: Using Train/val split (no validation set)")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config = read_yaml(config_path)
            return config
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> Any:
        """Setup logging system."""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('log_dir', 'logs/'))
        
        ensure_directory(log_dir)
        
        logger = setup_logger(
            name='model_training_pipeline',
            log_dir=log_dir,
            log_level=log_config.get('log_level', 'INFO'),
            max_bytes=log_config.get('max_bytes', 10485760),
            backup_count=log_config.get('backup_count', 7)
        )
        
        return logger
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Set experiment
        experiment_name = mlflow_config.get('experiment_name', 'Loan_Eligibility_Prediction')
        
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _ensure_output_directories(self) -> None:
        """Ensure all output directories exist."""
        self.logger.info("Creating output directories...")
        
        # Models directory
        model_path = Path(self.config['model_persistence']['best_model_path'])
        ensure_directory(model_path.parent)
        
        # Artifacts directory
        artifacts_path = Path(self.config['model_persistence']['artifacts_path'])
        ensure_directory(artifacts_path)
        
        # Plots directory
        if self.config.get('plotting', {}).get('enabled', True):
            plots_dir = Path(self.config['plotting']['plots_dir'])
            ensure_directory(plots_dir)
    
    def load_data(self) -> None:
        """Load preprocessed training data."""
        with Timer("Data loading", self.logger):
            try:
                self.data_loader = TrainingDataLoader(self.config)
                
                self.X_train, self.y_train, self.X_val, self.y_val = self.data_loader.load()
                
                self.feature_names = self.data_loader.feature_names
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                raise ModelTrainingError(f"Failed to load data: {e}")
    
    def train_baseline_models(self) -> List[Dict[str, Any]]:
        """
        Train all baseline models (no hyperparameter tuning).
        Evaluate on val set since we don't have a validation set.
        
        Returns:
            List of baseline results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1: BASELINE MODEL TRAINING")
        self.logger.info("="*80)
        
        # Get models to train
        models_to_train = []
        
        # Baseline model
        if self.config['models']['baseline']['enabled']:
            models_to_train.append(('LogisticRegression', self.config['models']['baseline']))
        
        # Tree-based models
        for model_name, model_config in self.config['models']['tree_based'].items():
            if model_config.get('enabled', True):
                models_to_train.append((model_name, model_config))
        
        self.logger.info(f"Training {len(models_to_train)} baseline models: {[m[0] for m in models_to_train]}")
        self.logger.info("Note: Models evaluated on val set (no separate validation set)")
        
        baseline_results = []
        
        for model_name, model_config in models_to_train:
            try:
                result = self._train_single_model(
                    model_name=model_name,
                    params=model_config['params'],
                    is_tuned=False
                )
                baseline_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Training {model_name} failed: {e}", exc_info=True)
                
                # Stop pipeline on failure
                if self.config.get('error_handling', {}).get('on_model_failure') == 'stop':
                    raise ModelTrainingError(f"Model training failed: {e}")
        
        # Rank models by val performance
        baseline_results.sort(key=lambda x: x['val_metrics'][self.config['metrics']['primary_metric']], reverse=True)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("BASELINE RESULTS SUMMARY (Ranked by val Performance)")
        self.logger.info("="*80)
        
        primary_metric = self.config['metrics']['primary_metric']
        for i, result in enumerate(baseline_results, 1):
            score = result['val_metrics'][primary_metric]
            cv_mean = result['cv_results'].get('cv_mean', 0)
            self.logger.info(f"{i}. {result['model_name']}: val_{primary_metric}={score:.4f}, cv_{primary_metric}={cv_mean:.4f}")
        
        self.baseline_results = baseline_results
        
        return baseline_results
    
    def _train_single_model(
        self,
        model_name: str,
        params: Dict[str, Any],
        is_tuned: bool = False
    ) -> Dict[str, Any]:
        """
        Train a single model and log to MLflow.
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            is_tuned: Whether this is a tuned model
            
        Returns:
            Dictionary with training results
        """
        run_name = f"{'tuned_' if is_tuned else ''}{model_name}_{self.timestamp}"
        
        with mlflow.start_run(run_name=run_name) as run:
            self.logger.info(f"\nMLflow Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("is_tuned", is_tuned)
            mlflow.log_params(params)
            
            # Log dataset metadata
            metadata = self.data_loader.get_metadata()
            for key, value in metadata.items():
                if not isinstance(value, (list, dict)):
                    mlflow.log_param(f"data_{key}", value)
            
            # Initialize trainer
            self.trainer = ModelTrainer(self.config)
            
            # Train model
            model = self.trainer.train(model_name, self.X_train, self.y_train, params)
            
            # Log training time
            mlflow.log_metric("training_time", self.trainer.training_time)
            
            # Cross-validation (on train set only)
            cv_results = self.trainer.cross_validate(
                self.X_train, self.y_train,
                scoring='f1'
            )
            
            for key, value in cv_results.items():
                if key != 'cv_scores':
                    mlflow.log_metric(f"cv_{key}", value)
            
            # Evaluate on train and val
            self.evaluator = ModelEvaluator(self.config)
            
            train_metrics = self.evaluator.evaluate(model, self.X_train, self.y_train, 'train')
            val_metrics = self.evaluator.evaluate(model, self.X_val, self.y_val, 'val')
            
            # Log all metrics to MLflow
            for metric_name, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"train_{metric_name}", value)
            
            for metric_name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{metric_name}", value)
            
            # Generalization analysis (train vs val)
            gen_analysis = self.evaluator.compare_train_val(train_metrics, val_metrics)
            mlflow.log_metric("train_val_gap", gen_analysis['gap'])
            mlflow.log_metric("train_val_gap_pct", gen_analysis['gap_percentage'])
            
            # Feature importance (for applicable models)
            feature_importance_results = {}
            if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                self.feature_analyzer = FeatureImportanceAnalyzer(self.config, self.feature_names)
                feature_importance_results = self.feature_analyzer.analyze(
                    model, model_name, self.X_val, self.y_val
                )
            
            # Generate and log plots
            if self.config.get('plotting', {}).get('enabled', True):
                self._generate_and_log_plots(model, model_name, val_metrics, feature_importance_results)
            
            # Log model to MLflow
            signature = infer_signature(self.X_train, model.predict(self.X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Compile results
            result = {
                'model_name': model_name,
                'model': model,
                'params': params,
                'is_tuned': is_tuned,
                'training_time': self.trainer.training_time,
                'cv_results': cv_results,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'generalization': gen_analysis,
                'feature_importance': feature_importance_results,
                'mlflow_run_id': run.info.run_id
            }
            
            return result
    
    def _generate_and_log_plots(
        self,
        model,
        model_name: str,
        val_metrics: Dict[str, Any],
        feature_importance: Dict[str, Any]
    ) -> None:
        """
        Generate and log plots to MLflow.
        
        Args:
            model: Trained model
            model_name: Name of the model
            val_metrics: val metrics
            feature_importance: Feature importance results
        """
        plots_dir = Path(self.config['plotting']['plots_dir'])
        
        # Confusion Matrix
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = val_metrics.get('confusion_matrix')
            if cm is not None:
                sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                
                cm_path = plots_dir / f"{model_name}_confusion_matrix.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(cm_path)
                plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to generate confusion matrix plot: {e}")
        
        
        # Feature Importance (if available)
        if feature_importance and 'native' in feature_importance:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                top_features = feature_importance['native']['top_features'][:20]
                features = [f['feature'] for f in top_features]
                importances = [f['importance'] for f in top_features]
                
                ax.barh(features, importances)
                ax.set_xlabel('Importance')
                ax.set_title(f'Top 20 Features - {model_name}')
                ax.invert_yaxis()
                
                fi_path = plots_dir / f"{model_name}_feature_importance.png"
                plt.tight_layout()
                plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(fi_path)
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to generate feature importance plot: {e}")
    
    def tune_top_models(self) -> List[Dict[str, Any]]:
        """
        Tune hyperparameters for top N models.
        Uses cross-validation on train set only.
        
        Returns:
            List of tuned model results
        """
        if not self.config.get('hyperparameter_tuning', {}).get('enabled', True):
            self.logger.info("Hyperparameter tuning disabled")
            return []
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: HYPERPARAMETER TUNING")
        self.logger.info("="*80)
        self.logger.info("Note: Tuning uses CV on train set, final evaluation on val set")
        
        tune_top_n = self.config['hyperparameter_tuning'].get('tune_top_n_models', 3)
        top_models = self.baseline_results[:tune_top_n]
        
        self.logger.info(f"Tuning top {tune_top_n} models: {[m['model_name'] for m in top_models]}")
        
        tuned_results = []
        
        for baseline_result in top_models:
            model_name = baseline_result['model_name']
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Tuning {model_name}")
            self.logger.info(f"{'='*80}")
            
            try:
                # Initialize tuner
                self.tuner = HyperparameterTuner(self.config)
                
                # Tune (uses CV on train set)
                tuning_results = self.tuner.tune(
                    model_name,
                    self.X_train,
                    self.y_train,
                    scoring='f1'
                )
                
                # Train model with best params and evaluate on val
                best_params = tuning_results['best_params']
                
                result = self._train_single_model(
                    model_name=model_name,
                    params=best_params,
                    is_tuned=True
                )
                
                result['tuning_results'] = tuning_results
                tuned_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Tuning {model_name} failed: {e}", exc_info=True)
                
                if self.config.get('error_handling', {}).get('on_model_failure') == 'stop':
                    raise ModelTrainingError(f"Hyperparameter tuning failed: {e}")
        
        # Rank tuned models by val performance
        tuned_results.sort(key=lambda x: x['val_metrics'][self.config['metrics']['primary_metric']], reverse=True)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TUNED RESULTS SUMMARY (Ranked by val Performance)")
        self.logger.info("="*80)
        
        primary_metric = self.config['metrics']['primary_metric']
        for i, result in enumerate(tuned_results, 1):
            score = result['val_metrics'][primary_metric]
            cv_mean = result['cv_results'].get('cv_mean', 0)
            self.logger.info(f"{i}. {result['model_name']}: val_{primary_metric}={score:.4f}, cv_{primary_metric}={cv_mean:.4f}")
        
        self.tuned_results = tuned_results
        
        return tuned_results
    
    def select_best_model(self) -> Dict[str, Any]:
        """
        Select best model based on val performance.
        
        Returns:
            Best model result dictionary
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL SELECTION")
        self.logger.info("="*80)
        
        # Combine baseline and tuned results
        all_results = self.baseline_results + self.tuned_results
        
        # Sort by val performance (primary metric)
        primary_metric = self.config['metrics']['primary_metric']
        all_results.sort(key=lambda x: x['val_metrics'][primary_metric], reverse=True)
        
        # Auto-select best
        best_result = all_results[0]
        
        self.best_model = best_result['model']
        self.best_model_name = best_result['model_name']
        self.best_model_score = best_result['val_metrics'][primary_metric]
        
        self.logger.info(f"\n✓ Best Model Selected: {self.best_model_name}")
        self.logger.info(f"  val {primary_metric}: {self.best_model_score:.4f}")
        self.logger.info(f"  CV {primary_metric}: {best_result['cv_results'].get('cv_mean', 0):.4f}")
        self.logger.info(f"  Tuned: {best_result['is_tuned']}")
        self.logger.info(f"  Training time: {best_result['training_time']:.2f}s")
        
        # Log selection criteria
        self.logger.info(f"\nSelection Criteria:")
        self.logger.info(f"  Primary metric (val {primary_metric}): {best_result['val_metrics'][primary_metric]:.4f}")
        self.logger.info(f"  Cross-validation {primary_metric}: {best_result['cv_results'].get('cv_mean', 0):.4f}")
        self.logger.info(f"  Training time: {best_result['training_time']:.2f}s")
        self.logger.info(f"  Generalization gap: {best_result['generalization']['gap']:.4f}")
        
        # Show top 5 models for manual verification
        self.logger.info(f"\nTop 5 Models (for manual verification):")
        for i, result in enumerate(all_results[:5], 1):
            val_score = result['val_metrics'][primary_metric]
            cv_score = result['cv_results'].get('cv_mean', 0)
            self.logger.info(f"  {i}. {result['model_name']} ({'tuned' if result['is_tuned'] else 'baseline'}):")
            self.logger.info(f"      val {primary_metric}: {val_score:.4f}")
            self.logger.info(f"      CV {primary_metric}: {cv_score:.4f}")
            self.logger.info(f"      Training time: {result['training_time']:.2f}s")
            self.logger.info(f"      Gap: {result['generalization']['gap']:.4f}")
        
        return best_result

    def validate_best_model(self, best_result: Dict[str, Any]) -> bool:
        """
        Validate best model before registration.
        
        Args:
            best_result: Best model result dictionary
            
        Returns:
            True if validation passed, False otherwise
        """
        self.validator = ModelValidator(self.config)
        
        validation_results = self.validator.validate_model(
            train_metrics=best_result['train_metrics'],
            val_metrics=best_result['val_metrics'],
            confusion_matrix=best_result['val_metrics'].get('confusion_matrix'),
            primary_metric=self.config['metrics']['primary_metric']
        )
        
        # Log to MLflow
        with mlflow.start_run(run_id=best_result['mlflow_run_id']):
            mlflow.log_param("validation_passed", validation_results['overall_passed'])
        
        return validation_results['overall_passed']
    
    def save_best_model(self, best_result: Dict[str, Any]) -> None:
        """
        Save best model to disk.
        
        Args:
            best_result: Best model result dictionary
        """
        with Timer("Saving best model", self.logger):
            try:
                model_path = self.config['model_persistence']['best_model_path']
                
                # Save model
                joblib.dump(best_result['model'], model_path)
                self.logger.info(f"✓ Best model saved: {model_path}")
                
                # Save artifacts
                if self.config['model_persistence']['save_artifacts']:
                    artifacts_dir = Path(self.config['model_persistence']['artifacts_path'])
                    
                    # Feature names
                    feature_names_path = artifacts_dir / 'feature_names.txt'
                    with open(feature_names_path, 'w') as f:
                        for feat in self.feature_names:
                            f.write(f"{feat}\n")
                    
                    # Metrics summary
                    metrics_path = artifacts_dir / 'metrics_summary.json'
                    import json
                    with open(metrics_path, 'w') as f:
                        json.dump({
                            'model_name': best_result['model_name'],
                            'is_tuned': best_result['is_tuned'],
                            'val_metrics': {k: v for k, v in best_result['val_metrics'].items() if isinstance(v, (int, float))},
                            'cv_metrics': best_result['cv_results'],
                            'training_time': best_result['training_time']
                        }, f, indent=2)
                    
                    self.logger.info(f"✓ Artifacts saved: {artifacts_dir}")
                
            except Exception as e:
                self.logger.error(f"Failed to save model: {e}", exc_info=True)
    
    def register_model_to_mlflow(self, best_result: Dict[str, Any]) -> None:
        """
        Register model with validation (Staging, not Production).
        """
        registry_config = self.config.get('mlflow', {}).get('model_registry', {})
        
        if not registry_config.get('enabled', True):
            self.logger.info("MLflow Model Registry disabled")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL REGISTRATION")
        self.logger.info("="*80)
        
        # ✅ NEW: Validate before registration
        if not self.validate_best_model(best_result):
            self.logger.error("✗ Model validation failed. NOT registering.")
            return
        
        try:
            run_id = best_result['mlflow_run_id']
            model_name = registry_config.get('model_name', 'LoanEligibilityClassifier')
            
            # ✅ CHANGED: Staging, not Production
            auto_stage = registry_config.get('auto_register_stage', 'Staging')
            
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            registered_model = mlflow.register_model(model_uri, model_name)
            
            self.logger.info(f"✓ Model registered: {model_name}")
            self.logger.info(f"  Version: {registered_model.version}")
            
            # Transition to Staging (not Production!)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            if auto_stage and auto_stage != 'None':
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage=auto_stage  # ← Staging!
                )
                
                self.logger.info(f"✓ Model transitioned to: {auto_stage}")
                
                # ✅ NEW: Show promotion instructions
                if auto_stage == 'Staging':
                    self.logger.info("\n" + "="*80)
                    self.logger.info("NEXT STEPS FOR PRODUCTION")
                    self.logger.info("="*80)
                    self.logger.info("1. Review model in MLflow UI")
                    self.logger.info("2. Test in staging environment")
                    self.logger.info("3. Manually promote:")
                    self.logger.info(f"   python scripts/promote_to_production.py --model-name {model_name} --version {registered_model.version}")
        
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}", exc_info=True)
    
    def execute(self) -> Dict[str, Any]:
        """Execute pipeline with validation."""
        try:
            with Timer("Complete Model Training Pipeline", self.logger):
                # ... existing setup code ...
                
                # Stages 1-4: Same as before
                self.load_data()
                self.train_baseline_models()
                self.tune_top_models()
                best_result = self.select_best_model()
                
                # ✅ NEW: Stage 5 - Validate
                validation_passed = self.validate_best_model(best_result)
                
                if not validation_passed:
                    self.logger.error("MODEL VALIDATION FAILED - NOT SAVING")
                    return {
                        'status': 'validation_failed',
                        'best_result': best_result
                    }
                
                # Only save/register if validated
                self.save_best_model(best_result)
                self.register_model_to_mlflow(best_result)
                
                self.logger.info("✓ Model registered to Staging")
                self.logger.info("  Manual promotion required for Production")
                
                return {
                    'status': 'success',
                    'validation_passed': validation_passed,
                    'best_model': self.best_model,
                    # ... rest of return dict
                }
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise ModelTrainingError(f"Pipeline failed: {e}")


def main():
    """Main entry point for model training pipeline."""
    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline(config_path="config/model_training_config.yaml")
        
        # Execute
        results = pipeline.execute()
        
        return 0
        
    except ModelTrainingError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())