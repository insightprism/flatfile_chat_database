"""
Configuration loading system for Flatfile-PrismMind integration.

Supports loading from multiple sources with precedence hierarchy following
PrismMind's configuration-driven philosophy.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import asdict

from flatfile_chat_database.prismmind_integration.config import (
    FlatfilePrismMindConfig,
    FlatfileDocumentProcessingConfig,
    FlatfileEngineSelectionConfig,
    FlatfileHandlerStrategiesConfig,
    FlatfileIntegrationConfig
)
from flatfile_chat_database.config import StorageConfig


class FlatfilePrismMindConfigLoader:
    """Centralized configuration loading with multiple sources"""
    
    @staticmethod
    def from_file(config_path: Union[str, Path]) -> FlatfilePrismMindConfig:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Complete configuration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if suffix in ['.json']:
                    config_data = json.load(f)
                elif suffix in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    # Try JSON first, then YAML
                    content = f.read()
                    try:
                        config_data = json.loads(content)
                    except json.JSONDecodeError:
                        config_data = yaml.safe_load(content)
            
            return FlatfilePrismMindConfigLoader.from_dict(config_data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    @staticmethod
    def from_environment(prefix: str = "FLATFILE_PM_") -> FlatfilePrismMindConfig:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration object with environment overrides
        """
        # Start with default configuration
        config_data = FlatfilePrismMindConfigLoader._get_default_config_dict()
        
        # Override with environment variables
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable name to config path
                # e.g., FLATFILE_PM_DOCUMENT_PROCESSING_TIMEOUT_SECONDS -> document_processing.timeout_seconds
                config_path = key[len(prefix):].lower().split('_')
                
                # Set nested value in config
                FlatfilePrismMindConfigLoader._set_nested_value(env_overrides, config_path, value)
        
        # Merge environment overrides
        config_data = FlatfilePrismMindConfigLoader._deep_merge(config_data, env_overrides)
        
        return FlatfilePrismMindConfigLoader.from_dict(config_data)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> FlatfilePrismMindConfig:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration object
        """
        try:
            # Extract flatfile storage config
            flatfile_config_data = config_dict.get("flatfile_config", {})
            flatfile_config = StorageConfig(**flatfile_config_data)
            
            # Extract document processing config
            doc_processing_data = config_dict.get("document_processing", {})
            doc_processing_config = FlatfileDocumentProcessingConfig(**doc_processing_data)
            
            # Extract engine selection config
            engine_selection_data = config_dict.get("engine_selection", {})
            engine_selection_config = FlatfileEngineSelectionConfig(**engine_selection_data)
            
            # Extract handler strategies config
            handler_strategies_data = config_dict.get("handler_strategies", {})
            handler_strategies_config = FlatfileHandlerStrategiesConfig(**handler_strategies_data)
            
            # Extract integration config
            integration_data = config_dict.get("integration_settings", {})
            integration_config = FlatfileIntegrationConfig(**integration_data)
            
            # Extract environment
            environment = config_dict.get("environment", "development")
            
            return FlatfilePrismMindConfig(
                flatfile_config=flatfile_config,
                document_processing=doc_processing_config,
                engine_selection=engine_selection_config,
                handler_strategies=handler_strategies_config,
                integration_settings=integration_config,
                environment=environment
            )
            
        except Exception as e:
            raise ValueError(f"Failed to create configuration from dictionary: {e}")
    
    @staticmethod
    def merge_configs(
        base_config: FlatfilePrismMindConfig,
        override_config: Dict[str, Any]
    ) -> FlatfilePrismMindConfig:
        """
        Merge configurations with override precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        base_dict = asdict(base_config)
        
        # Deep merge with overrides
        merged_dict = FlatfilePrismMindConfigLoader._deep_merge(base_dict, override_config)
        
        return FlatfilePrismMindConfigLoader.from_dict(merged_dict)
    
    @staticmethod
    def save_config(
        config: FlatfilePrismMindConfig,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
            format: Output format ("json" or "yaml")
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            config_dict = asdict(config)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == "json":
                    json.dump(config_dict, f, indent=2, default=str)
                elif format.lower() in ["yaml", "yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    @staticmethod
    def create_default_config() -> FlatfilePrismMindConfig:
        """
        Create default configuration with sensible defaults.
        
        Returns:
            Default configuration object
        """
        # Create default flatfile storage config
        flatfile_config = StorageConfig()
        
        # Create configuration with all defaults
        return FlatfilePrismMindConfig(flatfile_config=flatfile_config)
    
    @staticmethod
    def load_with_fallbacks(
        primary_path: Optional[Union[str, Path]] = None,
        fallback_paths: Optional[List[Union[str, Path]]] = None,
        environment_prefix: str = "FLATFILE_PM_"
    ) -> FlatfilePrismMindConfig:
        """
        Load configuration with multiple fallback sources.
        
        Precedence order:
        1. Primary configuration file
        2. Fallback configuration files (in order)
        3. Environment variables
        4. Default configuration
        
        Args:
            primary_path: Primary configuration file path
            fallback_paths: List of fallback configuration file paths
            environment_prefix: Environment variable prefix
            
        Returns:
            Configuration loaded from available sources
        """
        config = None
        
        # Try primary path
        if primary_path and Path(primary_path).exists():
            try:
                config = FlatfilePrismMindConfigLoader.from_file(primary_path)
                print(f"Loaded configuration from primary source: {primary_path}")
            except Exception as e:
                print(f"Failed to load primary config {primary_path}: {e}")
        
        # Try fallback paths
        if not config and fallback_paths:
            for fallback_path in fallback_paths:
                if Path(fallback_path).exists():
                    try:
                        config = FlatfilePrismMindConfigLoader.from_file(fallback_path)
                        print(f"Loaded configuration from fallback: {fallback_path}")
                        break
                    except Exception as e:
                        print(f"Failed to load fallback config {fallback_path}: {e}")
                        continue
        
        # Use default config if no file loaded
        if not config:
            config = FlatfilePrismMindConfigLoader.create_default_config()
            print("Using default configuration")
        
        # Apply environment variable overrides
        try:
            env_overrides = FlatfilePrismMindConfigLoader._get_environment_overrides(environment_prefix)
            if env_overrides:
                config = FlatfilePrismMindConfigLoader.merge_configs(config, env_overrides)
                print(f"Applied {len(env_overrides)} environment overrides")
        except Exception as e:
            print(f"Failed to apply environment overrides: {e}")
        
        return config
    
    @staticmethod
    def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate configuration file without loading full configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation result with status and errors
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "file_exists": False,
            "format_valid": False,
            "structure_valid": False
        }
        
        try:
            config_path = Path(config_path)
            
            # Check file existence
            if not config_path.exists():
                validation_result["errors"].append(f"Configuration file does not exist: {config_path}")
                return validation_result
            
            validation_result["file_exists"] = True
            
            # Check file format
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Try parsing as JSON or YAML
                try:
                    json.loads(content)
                    validation_result["format_valid"] = True
                except json.JSONDecodeError:
                    try:
                        yaml.safe_load(content)
                        validation_result["format_valid"] = True
                    except yaml.YAMLError as e:
                        validation_result["errors"].append(f"Invalid file format: {e}")
                        return validation_result
            
            except Exception as e:
                validation_result["errors"].append(f"Cannot read file: {e}")
                return validation_result
            
            # Check structure by attempting to load
            try:
                config = FlatfilePrismMindConfigLoader.from_file(config_path)
                validation_result["structure_valid"] = True
                
                # Run configuration validation
                from flatfile_chat_database.factory import FlatfilePrismMindConfigFactory
                config_errors = FlatfilePrismMindConfigFactory.validate_configuration(config)
                
                if config_errors:
                    validation_result["warnings"].extend(config_errors)
                else:
                    validation_result["valid"] = True
                    
            except Exception as e:
                validation_result["errors"].append(f"Configuration structure invalid: {e}")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {e}")
        
        return validation_result
    
    @staticmethod
    def _get_default_config_dict() -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        default_config = FlatfilePrismMindConfigLoader.create_default_config()
        return asdict(default_config)
    
    @staticmethod
    def _get_environment_overrides(prefix: str) -> Dict[str, Any]:
        """Extract environment variable overrides"""
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable to config path
                config_path = key[len(prefix):].lower().split('_')
                
                # Convert value to appropriate type
                converted_value = FlatfilePrismMindConfigLoader._convert_env_value(value)
                
                # Set nested value
                FlatfilePrismMindConfigLoader._set_nested_value(overrides, config_path, converted_value)
        
        return overrides
    
    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Handle boolean values
        if value.lower() in ["true", "yes", "1", "on"]:
            return True
        elif value.lower() in ["false", "no", "0", "off"]:
            return False
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Handle JSON values
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value
    
    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set nested value in dictionary using path list"""
        current = data
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[path[-1]] = value
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries with override precedence"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = FlatfilePrismMindConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result