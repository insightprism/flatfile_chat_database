"""
Configuration system for flatfile chat database.

This module provides backward compatibility with the old configuration system
while using the new modular configuration architecture internally.

For new code, prefer importing from config.manager directly:
    from config.manager import ConfigurationManager, load_config

For legacy compatibility, this module provides the old StorageConfig interface.
"""

from typing import Optional
from pathlib import Path

# Import new configuration system
from config_new.manager import ConfigurationManager, load_config as load_new_config
from config_new.compat import LegacyStorageConfig

# Re-export legacy interface for backward compatibility
StorageConfig = LegacyStorageConfig


# Legacy compatibility classes
class DevelopmentConfig(LegacyStorageConfig):
    """Development environment configuration (legacy compatibility)."""
    def __init__(self):
        manager = ConfigurationManager.from_environment("development")
        super().__init__(manager)


class ProductionConfig(LegacyStorageConfig):
    """Production environment configuration (legacy compatibility)."""
    def __init__(self):
        manager = ConfigurationManager.from_environment("production")
        super().__init__(manager)


def load_config(config_path: Optional[str] = None, 
                environment: Optional[str] = None) -> StorageConfig:
    """
    Load configuration from file and environment (legacy interface).
    
    Priority (highest to lowest):
    1. Environment variables (CHATDB_*)
    2. Environment-specific config class
    3. Config file
    4. Default values
    
    Args:
        config_path: Path to JSON configuration file
        environment: Environment name ('development', 'production')
        
    Returns:
        Validated StorageConfig instance (legacy adapter)
    """
    manager = load_new_config(config_path, environment)
    return LegacyStorageConfig(manager)


def save_config(config: StorageConfig, path: str) -> None:
    """
    Save configuration to JSON file (legacy interface).
    
    Args:
        config: StorageConfig instance to save
        path: Path where to save the configuration
    """
    if hasattr(config, '_manager'):
        # New config with manager
        config._manager.save_to_file(Path(path))
    else:
        # Old-style config - convert to dict and save
        import json
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)


def get_default_config(environment: Optional[str] = None) -> StorageConfig:
    """
    Get default configuration for the specified environment (legacy interface).
    
    Args:
        environment: Environment name or None for base config
        
    Returns:
        Default StorageConfig for the environment
    """
    if environment == "development":
        return DevelopmentConfig()
    elif environment == "production":
        return ProductionConfig()
    else:
        return LegacyStorageConfig()


# Import guard to prevent circular imports
import sys
if 'config' not in sys.modules:
    # First import - set up proper structure
    pass
else:
    # Module already imported - maintain compatibility
    from typing import Optional, List, Dict, Any
    from pathlib import Path