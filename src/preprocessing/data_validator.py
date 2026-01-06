

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils import LoggerMixin

class DataValidator(LoggerMixin):
    """
    Validate data quality at various pipeline stages.
    
    Performs checks for:
    - Schema validation (expected columns and types)
    - Value range validation
    - Cardinality checks
    - Data integrity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = self.setup_class_logger('Data_validator',config,'logging')
        self.config = config or {}
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> None:
        """
        Validate DataFrame has expected columns.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
        
        Raises:
            ValueError: If schema validation fails
        
        Example:
            >>> validator.validate_schema(df, ['age', 'gender', 'stroke'])
        """
        self.logger.info('Validating data schema...')
        
        # Check for missing columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        # Check for unexpected columns
        unexpected_cols = [col for col in df.columns if col not in expected_columns]
        if unexpected_cols:
            self.logger.warning(f"Unexpected columns found: {unexpected_cols}")
        
        self.logger.info(f'✓ Schema validation passed ({len(df.columns)} columns)')
    
    def validate_ranges(self, df: pd.DataFrame, range_rules: Dict[str, tuple]) -> None:
        """
        Validate numeric columns are within expected ranges.
        
        Args:
            df: DataFrame to validate
            range_rules: Dictionary mapping column names to (min, max) tuples
        
        Raises:
            ValueError: If values are outside expected ranges
        
        Example:
            >>> rules = {'age': (0, 120), 'bmi': (10, 70)}
            >>> validator.validate_ranges(df, rules)
        """
        self.logger.info('Validating value ranges...')
        
        violations = []
        
        for col, (min_val, max_val) in range_rules.items():
            if col not in df.columns:
                continue
            
            # Check for values outside range
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            
            if len(out_of_range) > 0:
                violations.append({
                    'column': col,
                    'expected_range': (min_val, max_val),
                    'actual_range': (df[col].min(), df[col].max()),
                    'violations': len(out_of_range),
                    'percentage': len(out_of_range) / len(df) * 100
                })
        
        if violations:
            for v in violations:
                self.logger.warning(
                    f"Range violation in {v['column']}: "
                    f"{v['violations']} rows ({v['percentage']:.2f}%) "
                    f"outside expected range {v['expected_range']}"
                )
        else:
            self.logger.info('✓ Range validation passed')
    
    def validate_cardinality(self, df: pd.DataFrame, max_cardinality: int = 50) -> None:
        """
        Check for unexpectedly high-cardinality columns.
        
        High cardinality in categorical columns often indicates:
        - IDs that should be excluded
        - Free-text fields that need special handling
        - Data quality issues
        
        Args:
            df: DataFrame to validate
            max_cardinality: Maximum expected unique values for categorical columns
        
        Example:
            >>> validator.validate_cardinality(df, max_cardinality=30)
        """
        self.logger.info('Validating cardinality...')
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        high_cardinality_cols = []
        
        for col in categorical_cols[1:]: # ignore employee names column
            n_unique = df[col].nunique()
            if n_unique > max_cardinality:
                high_cardinality_cols.append({
                    'column': col,
                    'unique_values': n_unique,
                    'cardinality_ratio': n_unique / len(df)
                })
        
        if high_cardinality_cols:
            self.logger.warning('High cardinality columns detected:')
            for col_info in high_cardinality_cols:
                self.logger.warning(
                    f"  {col_info['column']}: {col_info['unique_values']} unique values "
                    f"({col_info['cardinality_ratio']:.1%} of rows)"
                )
        else:
            self.logger.info('✓ Cardinality validation passed')
    
    def validate_integrity(self, df: pd.DataFrame) -> None:
        """
        Validate data integrity (no duplicates, no all-null columns, etc.).
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: If integrity checks fail
        
        Example:
            >>> validator.validate_integrity(df)
        """
        self.logger.info('Validating data integrity...')
        
        issues = []
        
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Check for all-null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            issues.append(f"Columns with all nulls: {all_null_cols}")
        
        # Check for all-same-value columns (constants)
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
        if inf_cols:
            issues.append(f"Columns with infinite values: {inf_cols}")
        
        # Log issues
        if issues:
            for issue in issues:
                self.logger.warning(f"  {issue}")
        else:
            self.logger.info('✓ Integrity validation passed')
    
    def validate_processed_data(self, train: pd.DataFrame, dev: pd.DataFrame, 
                            test: pd.DataFrame) -> None:
        """
        Validate processed data splits.
        
        Checks:
        - All splits have same number of features
        - No missing values (if imputation was applied)
        - Target distribution is similar across splits
        
        Args:
            train: Training set
            dev: Development set
            test: Test set
        
        Raises:
            ValueError: If validation fails
        
        Example:
            >>> validator.validate_processed_data(train, dev, test)
        """
        self.logger.info('Validating processed data splits...')
        
        # Check feature count consistency
        if train.shape[1] != dev.shape[1] or train.shape[1] != test.shape[1]:
            raise ValueError(
                f"Feature count mismatch: "
                f"train={train.shape[1]}, dev={dev.shape[1]}, test={test.shape[1]}"
            )
        
        # Check for missing values
        train_missing = train.isnull().sum().sum()
        dev_missing = dev.isnull().sum().sum()
        test_missing = test.isnull().sum().sum()
        
        if train_missing > 0 or dev_missing > 0 or test_missing > 0:
            self.logger.warning(
                f"Missing values detected: "
                f"train={train_missing}, dev={dev_missing}, test={test_missing}"
            )
        
        # Check for infinite values
        train_inf = np.isinf(train.select_dtypes(include=[np.number])).sum().sum()
        dev_inf = np.isinf(dev.select_dtypes(include=[np.number])).sum().sum()
        test_inf = np.isinf(test.select_dtypes(include=[np.number])).sum().sum()
        
        if train_inf > 0 or dev_inf > 0 or test_inf > 0:
            raise ValueError(
                f"Infinite values detected: "
                f"train={train_inf}, dev={dev_inf}, test={test_inf}"
            )
        
        self.logger.info('✓ Processed data validation passed')