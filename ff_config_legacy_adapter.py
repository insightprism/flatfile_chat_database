"""
Legacy configuration adapter for backward compatibility.

This module provides backward compatibility with the old configuration system.
"""

from ff_class_configs.ff_configuration_manager_config import (
    FFConfigurationManagerConfigDTO, create_default_config, load_config as new_load_config
)

# Legacy StorageConfig class for backward compatibility
class StorageConfig:
    """Legacy storage configuration class. Use FFConfigurationManagerConfigDTO instead."""
    
    def __init__(self, base_path: str = "./dev_data"):
        self._config = create_default_config("development")
        self._config.storage.base_path = base_path
        
    @property
    def base_path(self) -> str:
        return self._config.storage.base_path
        
    @base_path.setter
    def base_path(self, value: str):
        self._config.storage.base_path = value
        
    def to_new_config(self) -> FFConfigurationManagerConfigDTO:
        """Convert to new configuration format."""
        return self._config


def ff_load_config(config_path: str = None, environment: str = "development") -> StorageConfig:
    """
    Legacy config loading function. Use load_config() instead.
    
    Args:
        config_path: Path to config file (ignored for now)
        environment: Environment name
        
    Returns:
        StorageConfig: Legacy configuration object
    """
    new_config = create_default_config(environment)
    legacy_config = StorageConfig(new_config.storage.base_path)
    legacy_config._config = new_config
    return legacy_config