
"""
EDA Pipeline Orchestrator
ENHANCED: Better error handling, result tracking, report generation
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict, Any
import sys
import pandas as pd
import numpy as np
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from eda import DataQuality, DataOverview, Visualizations
from utils import ensure_directory, setup_logger, Timer, write_json  
from utils.mlflow_utils import mlflow_stage_run


class EDAPipelineError(Exception):
    """Custom exception for EDA pipeline errors"""
    pass


class EDAPipeline:
    """
    Orchestrate complete EDA workflow.
    
    Coordinates data overview, quality checks, and visualizations.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize EDA Pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.logger = setup_logger(
            name='eda_pipeline',
            log_dir=Path('logs/')
        )
        self.results = {}  
        
        self.logger.info('='*80)
        self.logger.info('EDA PIPELINE INITIALIZED')
        self.logger.info('='*80)
    
    def execute(self) -> Dict[str, Any]:  
        """
        Execute complete EDA pipeline with MLflow tracking.
        
        Returns:
            Dictionary with results from all stages
            
        Raises:
            EDAPipelineError: If pipeline execution fails
        """
        try:
            # top-level MLflow run for the EDA stage
            with mlflow_stage_run(
                stage="eda",
                run_name="complete_eda_pipeline",
                tags={"pipeline": "eda", "config_path": self.config_path},
            ):
                # ===== STAGE 1: DATA OVERVIEW =====
                with mlflow.start_run(run_name='Data_Overview', nested=True):
                    with Timer('Data overview', self.logger):
                        overview = DataOverview(self.config_path)
                        overview_results = overview.run_data_overview()
                
                        df = overview_results['dataframe']
                        config = overview.get_config()
                        
                        self.results['overview'] = {
                            'shape': df.shape,
                            'numeric_columns': overview_results.get('numeric_columns', []),
                            'categorical_columns': overview_results.get('categorical_columns', [])
                        }
                        mlflow.log_metric("n_rows", df.shape[0])
                        mlflow.log_metric("n_columns", df.shape[1])
                    
                # ===== STAGE 2: DATA QUALITY CHECKS =====
                with mlflow.start_run(run_name='Data_Quality', nested=True):
                    with Timer('Data Quality Checks', self.logger):
                        quality_checks = DataQuality(config)
                        quality_results = quality_checks.run_all_checks(df) 
                        
                        self.results['quality'] = {
                            'missing_columns': len(quality_results.get('missing_values', [])),
                            'duplicate_count': quality_results.get('duplicate_count', 0),
                            'outlier_columns': len(quality_results.get('outliers', {}))
                        }
                        mlflow.log_metric("missing_columns", self.results['quality']['missing_columns'])
                        mlflow.log_metric("duplicate_count", self.results['quality']['duplicate_count'])
                        mlflow.log_metric("outlier_columns", self.results['quality']['outlier_columns'])
                        mlflow.set_tag(
                            "data_quality_note",
                            "Baseline EDA checks; no irreversible schema changes applied.",
                        )
                    
                # ===== STAGE 3: VISUALIZATIONS =====
                with mlflow.start_run(run_name='Data_visualizations', nested=True):
                    with Timer('Data visualizations', self.logger):
                        visuals = Visualizations(config)
                        visual_results = visuals.run_visualizations(df)
                        
                        self.results['visualizations'] = visual_results
            
                # ===== GENERATE SUMMARY REPORT =====
                with mlflow.start_run(run_name='Generate_summary_report', nested=True):
                    self._generate_summary_report(df, config)
                    mlflow.set_tag(
                        "decision_note",
                        "EDA summary generated to support downstream preprocessing decisions.",
                    )
            
            self.logger.info("=" * 80)
            self.logger.info("✓ EDA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return self.results
        
        except Exception as e:
            self.logger.error(f'EDA pipeline failed: {e}', exc_info=True)
            raise EDAPipelineError(f'Pipeline execution failed: {e}')
    

    def _generate_summary_report(self, df: pd.DataFrame, config: Dict) -> None:
        """
        Generate comprehensive EDA summary report.
        
        Args:
            df: DataFrame being analyzed
            config: Configuration dictionary
        """
        try:
            self.logger.info('Generating EDA summary report...')
            
            report = {
                'dataset_info': {
                    'rows': int(df.shape[0]),
                    'columns': int(df.shape[1]),
                    'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                    'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
                    'categorical_columns': int(len(df.select_dtypes(exclude=[np.number]).columns))
                },
                'data_quality': {
                    'missing_percentage': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
                    'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2)
                },
                'pipeline_results': self.results
            }
            
            # Save report
            report_path = Path(config.get('output', {}).get('artifacts_dir', 'artifacts')) / 'eda_summary_report.json'
            ensure_directory(report_path.parent)
            write_json(report, report_path, indent=2)
            
            self.logger.info(f'✓ Summary report saved to {report_path}')

            mlflow.log_artifact(report_path)
        
        except Exception as e:
            self.logger.warning(f'Could not generate summary report: {e}')


if __name__ == '__main__':
    pipeline = EDAPipeline(config_path='config/eda_config.yaml')
    results = pipeline.execute()