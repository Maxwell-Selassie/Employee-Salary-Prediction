

from typing import Dict, List, Any
import logging


class ConfigValidator:
    """
    Validate preprocessing configuration structure and values.
    
    Ensures all required keys exist and values are valid before
    pipeline execution begins.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_preprocessing_config(self, config: Dict[str, Any]) -> None:
        """
        Validate complete preprocessing configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        
        Example:
            >>> validator = ConfigValidator()
            >>> validator.validate_preprocessing_config(config)
        """
        try:
            self.logger.info('Validating preprocessing configuration...')
            
            # Check required top-level keys
            self._validate_required_keys(config)
            
            # Validate data split config
            self._validate_data_split_config(config.get('data_split', {}))
            
            # Validate missing values config
            self._validate_missing_values_config(config.get('missing_values', {}))
            
            # Validate outliers config
            self._validate_outliers_config(config.get('outliers', {}))
            
            # Validate encoding config
            self._validate_encoding_config(config.get('encoding', {}))
            
            # Validate transformations config
            self._validate_transformations_config(config.get('transformations', {}))
            
            # Validate output config
            self._validate_output_config(config.get('output', {}))
            
            self.logger.info('✓ Configuration validation passed')
        
        except Exception as e:
            self.logger.error(f'Configuration validation failed: {e}')
            raise
    
    def _validate_required_keys(self, config: Dict) -> None:
        """Validate required top-level configuration keys exist"""
        required_keys = [
            'data','expected_columns', 'value_ranges', 'data_split', 'missing_values', 
            'duplicates', 'outliers', 'encoding', 
            'transformations', 'output', 'logging', 'columns_to_drop'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
    
    def _validate_data_split_config(self, config: Dict) -> None:
        """Validate data split configuration"""
        required = ['test_size', 'dev_size', 'random_state']
        
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required key in data_split: {key}")
        
        # Validate test_size
        test_size = config['test_size']
        if not isinstance(test_size, int) or test_size <= 0:
            raise ValueError(f"test_size must be positive integer, got {test_size}")
        
        # Validate dev_size
        dev_size = config['dev_size']
        if not isinstance(dev_size, int) or dev_size <= 0:
            raise ValueError(f"dev_size must be positive integer, got {dev_size}")
        
        # Validate random_state
        random_state = config['random_state']
        if not isinstance(random_state, int):
            raise ValueError(f"random_state must be integer, got {type(random_state)}")
    
    def _validate_missing_values_config(self, config: Dict) -> None:
        """Validate missing values configuration"""
        if 'enabled' not in config:
            raise ValueError("Missing 'enabled' key in missing_values config")
        
        if config['enabled']:
            if 'numeric' not in config or 'strategy' not in config['numeric']:
                raise ValueError("Missing numeric strategy in missing_values config")
            
            valid_strategies = ['mean', 'median', 'mode', 'drop']
            strategy = config['numeric']['strategy']
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid numeric strategy: {strategy}. Must be one of {valid_strategies}")
    
    def _validate_outliers_config(self, config: Dict) -> None:
        """Validate outliers configuration"""
        if 'cols_to_flag' not in config:
            raise ValueError("Missing 'cols_to_flag' in outliers config")
        
        if not isinstance(config['cols_to_flag'], list):
            raise ValueError("cols_to_flag must be a list")
    
    def _validate_encoding_config(self, config: Dict) -> None:
        """Validate encoding configuration"""
        if 'one_hot_columns' not in config:
            raise ValueError("Missing 'one_hot_columns' in encoding config")
        
        if not isinstance(config['one_hot_columns'], (list, bool)):
            raise ValueError("one_hot_columns must be list or boolean")
    
    def _validate_transformations_config(self, config: Dict) -> None:
        """Validate transformations configuration"""
        if 'log_columns' not in config:
            raise ValueError("Missing 'log_columns' in transformations config")
        
        if not isinstance(config['log_columns'], list):
            raise ValueError("log_columns must be a list")
    
    def _validate_output_config(self, config: Dict) -> None:
        """Validate output configuration"""
        required = ['processed_dir', 'pipeline_file']
        
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required key in output config: {key}")