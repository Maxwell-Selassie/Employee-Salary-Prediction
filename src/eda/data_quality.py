"""
Data Quality Module - Check for missing values, duplicates, and outliers
FIXED: Logic errors, better reporting, improved outlier detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logger, ensure_directory, write_csv
from utils.logging_mixin import LoggerMixin  


class DataQualityError(Exception):
    """Custom exception for data quality errors"""
    pass


class DataQuality(LoggerMixin):  
    """
    Perform data quality checks including missing values, duplicates, and outliers.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataQuality checker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('data_quality', config, 'logging')  
    
    def _check_for_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values with severity thresholds.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value analysis
        """
        warning_threshold = self.config['data_quality'].get('missing_threshold_warning', 0.05)
        critical_threshold = self.config['data_quality'].get('missing_threshold_critical', 0.30)
        
        self.logger.info('Analyzing missing values...')
        
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        # FIXED: Correct logic
        if len(missing) == 0:
            self.logger.info('✓ No missing values found!')
            return pd.DataFrame()  # Return empty df
        
        # Calculate percentages
        missing_pct = (missing / len(df)).round(4)
        
        # Categorize severity
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct,
            'severity': [
                'CRITICAL' if pct >= critical_threshold
                else 'WARNING' if pct >= warning_threshold
                else 'INFO'
                for pct in missing_pct
            ]
        })
        
        # Save artifacts
        artifacts_path = self.config['data'].get(
            'artifact_data_path_missing',
            'artifacts/missing_values.csv'
        )
        ensure_directory(Path(artifacts_path).parent)
        write_csv(missing_df, artifacts_path, index=True) 
        
        # Log findings
        critical_cols = missing_df[missing_df['severity'] == 'CRITICAL']
        warning_cols = missing_df[missing_df['severity'] == 'WARNING']
        
        self.logger.info(f'Missing values found in {len(missing_df)} columns')
        
        if len(critical_cols) > 0:
            self.logger.warning(
                f'⚠️  CRITICAL: {len(critical_cols)} columns exceed {critical_threshold*100}% missing'
            )
            self.logger.warning(f'Critical columns: {critical_cols.index.tolist()}')
        
        if len(warning_cols) > 0:
            self.logger.warning(
                f'⚠️  WARNING: {len(warning_cols)} columns exceed {warning_threshold*100}% missing'
            )
        
        return missing_df
    
    def _check_for_duplicates(self, df: pd.DataFrame) -> int:
        """
        Check for duplicate rows in dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Number of duplicate rows
        """
        if not self.config['data_quality'].get('check_duplicates', True):
            self.logger.info('Duplicate checking disabled (skipping...)')
            return 0
        
        self.logger.info('Checking for duplicates...')
        
        n_duplicates = df.duplicated().sum()
        
        # FIXED: Correct logic
        if n_duplicates == 0:
            self.logger.info('✓ No duplicates found')
            return 0
        
        duplicates_pct = (n_duplicates / len(df)) * 100
        self.logger.warning(
            f'⚠️  Found {n_duplicates} duplicate rows ({duplicates_pct:.2f}%)'
        )
        
        # NEW: Save duplicate rows for inspection
        if self.config['data_quality'].get('save_duplicates', False):
            dup_rows = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
            dup_path = Path(self.config['data'].get('artifact_data_path_duplicates', 'artifacts/duplicate_rows.csv'))
            ensure_directory(dup_path.parent)
            write_csv(dup_rows, dup_path, index=False)
            self.logger.info(f'Saved duplicate rows to {dup_path}')
        
        return n_duplicates
    
    def detect_outliers_iqr(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier statistics per column
        """
        if not self.config['data_quality'].get('check_outliers', True):
            self.logger.warning('Outlier detection disabled (skipping...)')
            return {}
        
        self.logger.info('Detecting outliers using IQR method...')
        
        # Columns to exclude from outlier detection
        cols_not_for_outliers = self.config['data_quality'].get(
            'outlier_exclude_columns',
            ['id', 'stroke', 'hypertension', 'heart_disease']
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_check = [col for col in numeric_cols if col not in cols_not_for_outliers]
        
        if not cols_to_check:
            self.logger.warning('No numeric columns available for outlier detection')
            return {}
        
        outlier_summary = {}
        
        for col in cols_to_check:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outlier_mask.sum()
            outlier_pct = (n_outliers / len(df)) * 100
            
            outlier_summary[col] = {
                'count': int(n_outliers),
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'Q1': round(Q1, 2),
                'Q3': round(Q3, 2),
                'IQR': round(IQR, 2)
            }
            
            self.logger.debug(
                f'{col}: {n_outliers} outliers ({outlier_pct:.1f}%) '
                f'outside [{lower_bound:.2f}, {upper_bound:.2f}]'
            )
        
        # Save summary
        if self.config['data_quality'].get('outlier_report', True):
            summary_df = pd.DataFrame(outlier_summary).T
            outlier_path = self.config['data'].get(
                'artifact_data_path_outliers',
                'artifacts/outlier_summary.csv'
            )
            ensure_directory(Path(outlier_path).parent)
            write_csv(summary_df, outlier_path, index=True)  
            self.logger.info(f'✓ Saved outlier summary to {outlier_path}')
        
        # Log high outlier columns
        high_outlier_cols = [
            col for col, stats in outlier_summary.items()
            if stats['percentage'] > 10
        ]
        
        if high_outlier_cols:
            self.logger.warning(
                f'⚠️  Columns with >10% outliers: {high_outlier_cols}'
            )
        
        return outlier_summary
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Any]:  
        """
        Run all data quality checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with results from all checks
        """
        results = {}
        
        try:
            self.logger.info('='*60)
            self.logger.info('STARTING DATA QUALITY CHECKS')
            self.logger.info('='*60)
            
            results['missing_values'] = self._check_for_missing_values(df)
            results['duplicate_count'] = self._check_for_duplicates(df)
            results['outliers'] = self.detect_outliers_iqr(df)
            
            self.logger.info('='*60)
            self.logger.info('✓ DATA QUALITY CHECKS COMPLETED')
            self.logger.info('='*60)
            
            return results
        
        except Exception as e:
            self.logger.error(f'Data quality checks failed: {e}', exc_info=True)
            raise DataQualityError(f'Quality checks failed: {e}')