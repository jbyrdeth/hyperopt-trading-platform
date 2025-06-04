"""
Configuration manager for the trading optimization system.
Handles loading and validation of YAML configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    config_file: str
    error_message: str
    
    def __str__(self):
        return f"Configuration error in {self.config_file}: {self.error_message}"


class ConfigManager:
    """
    Configuration manager for the trading optimization system.
    
    Handles loading, validation, and access to configuration files.
    Supports environment variable substitution and configuration merging.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
        
        # Load all configuration files
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML configuration files from the config directory."""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        if not config_files:
            raise ConfigValidationError("config_dir", "No YAML configuration files found")
        
        for config_file in config_files:
            config_name = config_file.stem
            try:
                self.configs[config_name] = self._load_config_file(config_file)
                self.logger.debug(f"Loaded configuration: {config_name}")
            except Exception as e:
                raise ConfigValidationError(str(config_file), str(e))
        
        self.logger.info(f"Loaded {len(self.configs)} configuration files")
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """
        Load a single YAML configuration file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse YAML
            config = yaml.safe_load(content)
            
            if config is None:
                config = {}
            
            # Validate configuration
            self._validate_config(config_file.name, config)
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigValidationError(str(config_file), f"YAML parsing error: {e}")
        except Exception as e:
            raise ConfigValidationError(str(config_file), f"Error loading file: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables substituted
        """
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                var_value = os.getenv(var_expr)
                if var_value is None:
                    self.logger.warning(f"Environment variable {var_expr} not found")
                    return match.group(0)  # Return original if not found
                return var_value
        
        # Replace ${VAR_NAME} and ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _validate_config(self, config_name: str, config: Dict[str, Any]):
        """
        Validate configuration structure and required fields.
        
        Args:
            config_name: Name of the configuration file
            config: Configuration dictionary to validate
        """
        # Basic validation rules for each config type
        validation_rules = {
            'strategies.yaml': self._validate_strategies_config,
            'optimization.yaml': self._validate_optimization_config,
            'exchanges.yaml': self._validate_exchanges_config,
        }
        
        if config_name in validation_rules:
            validation_rules[config_name](config)
    
    def _validate_strategies_config(self, config: Dict[str, Any]):
        """Validate strategies configuration."""
        required_sections = ['trend_following', 'mean_reversion', 'momentum']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(
                    'strategies.yaml',
                    f"Required section '{section}' not found"
                )
        
        # Validate global settings
        if 'global_settings' not in config:
            raise ConfigValidationError(
                'strategies.yaml',
                "Required section 'global_settings' not found"
            )
    
    def _validate_optimization_config(self, config: Dict[str, Any]):
        """Validate optimization configuration."""
        required_sections = ['hyperopt', 'validation', 'performance_thresholds']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(
                    'optimization.yaml',
                    f"Required section '{section}' not found"
                )
        
        # Validate hyperopt settings
        hyperopt_config = config.get('hyperopt', {})
        if 'algorithm' not in hyperopt_config:
            raise ConfigValidationError(
                'optimization.yaml',
                "hyperopt.algorithm is required"
            )
    
    def _validate_exchanges_config(self, config: Dict[str, Any]):
        """Validate exchanges configuration."""
        if 'exchanges' not in config:
            raise ConfigValidationError(
                'exchanges.yaml',
                "Required section 'exchanges' not found"
            )
        
        exchanges = config['exchanges']
        if not exchanges:
            raise ConfigValidationError(
                'exchanges.yaml',
                "At least one exchange must be configured"
            )
        
        # Validate each exchange has required fields
        for exchange_name, exchange_config in exchanges.items():
            required_fields = ['name', 'enabled', 'priority']
            for field in required_fields:
                if field not in exchange_config:
                    raise ConfigValidationError(
                        'exchanges.yaml',
                        f"Exchange '{exchange_name}' missing required field '{field}'"
                    )
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration (without .yaml extension)
            
        Returns:
            Configuration dictionary
            
        Raises:
            KeyError: If configuration not found
        """
        if config_name not in self.configs:
            available_configs = list(self.configs.keys())
            raise KeyError(
                f"Configuration '{config_name}' not found. "
                f"Available configurations: {available_configs}"
            )
        
        return self.configs[config_name].copy()  # Return copy to prevent modification
    
    def get_config_value(
        self,
        config_name: str,
        key_path: str,
        default: Any = None
    ) -> Any:
        """
        Get a specific value from configuration using dot notation.
        
        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value (e.g., 'hyperopt.max_evals')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            config = self.get_config(config_name)
            keys = key_path.split('.')
            
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except KeyError:
            return default
    
    def get_available_configs(self) -> list:
        """Get list of available configuration names."""
        return list(self.configs.keys()) 