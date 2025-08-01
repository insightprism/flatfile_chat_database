"""
LEGACY CONFIGURATION ADAPTER - DEPRECATED

⚠️  THIS FILE IS DEPRECATED AND MOSTLY COMMENTED OUT ⚠️

Migration completed: New configuration system now active.

TO ROLLBACK (if needed):
1. Uncomment the large commented section below
2. Uncomment ff_class_configs/ff_legacy_storage_config.py  
3. Change imports back to: from ff_config_legacy_adapter import StorageConfig
4. Revert attribute access patterns: config.storage.base_path -> config.storage_base_path

FOR NEW CODE, USE:
    from ff_class_configs.ff_configuration_manager_config import load_config
    config = load_config()
    base_path = config.storage.base_path  # Domain-organized access

BENEFITS OF NEW SYSTEM:
- Better organization (storage, search, vector, document domains)
- Type safety with domain-specific configs  
- Cleaner attribute names
- No 300+ line mapping overhead
- Easier to extend and maintain

  Phase 2: Clean Removal

  # Remove legacy files
  rm ff_config_legacy_adapter.py
  rm ff_class_configs/ff_legacy_storage_config.py

  # Update __init__.py to remove legacy exports
  # (Manual edit required)

  Phase 3: Verify After Cleanup

  # Test that nothing broke after deletion
  python3 -c "
  from ff_storage_manager import FFStorageManager
  from ff_class_configs.ff_configuration_manager_config import load_config
  print('✅ Cleanup successful - no legacy dependencies remain')

"""
#
# # TEMPORARY COMPATIBILITY STUBS - Keep imports working during transition
# from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
# from ff_class_configs.ff_legacy_storage_config import FFLegacyStorageConfigDTO
#
# # Minimal compatibility exports
# StorageConfig = FFLegacyStorageConfigDTO
# StorageConfigDTO = FFLegacyStorageConfigDTO
#
# def ff_load_config(config_path=None, environment=None):
#     """Compatibility stub - redirects to new system"""
#     manager = load_config(config_path, environment)
#     return FFLegacyStorageConfigDTO(manager)
#
# def ff_load_configDTO(config_path=None, environment=None):
#     """Compatibility stub with DTO suffix"""
#     return ff_load_config(config_path, environment)
#
# """
# # COMMENTED OUT - Original legacy adapter code
# # Uncomment this entire section to rollback to legacy system
#
# # from typing import Optional
# from pathlib import Path

# Import new configuration system
# from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config as load_new_config
# from ff_class_configs.ff_legacy_storage_config import FFLegacyStorageConfig

# Re-export legacy interface for backward compatibility
# StorageConfig = FFLegacyStorageConfig


# Legacy compatibility classes
# class FFDevelopmentConfig(FFLegacyStorageConfig):
    # Development environment configuration (legacy compatibility).
    # def __init__(self):
    #     manager = FFConfigurationManagerConfigDTO.from_environment("development")
    #     super().__init__(manager)


# class FFProductionConfig(FFLegacyStorageConfig):
    # Production environment configuration (legacy compatibility).
    # def __init__(self):
    #     manager = FFConfigurationManagerConfigDTO.from_environment("production")
    #     super().__init__(manager)


# def ff_load_config(config_path: Optional[str] = None, 
#                 environment: Optional[str] = None) -> StorageConfig:
    # Load configuration from file and environment (legacy interface).
    
    # Priority (highest to lowest):
    # 1. Environment variables (CHATDB_*)
    # 2. Environment-specific config class
    # 3. Config file
    # 4. Default values
    
    # Args:
    #     config_path: Path to JSON configuration file
    #     environment: Environment name ('development', 'production')
    #     
    # Returns:
    #     Validated StorageConfig instance (legacy adapter)
    
    # manager = load_new_config(config_path, environment)
    # return FFLegacyStorageConfig(manager)


# def ff_save_config(config: StorageConfig, path: str) -> None:
    # Save configuration to JSON file (legacy interface).
    
    # Args:
    #     config: StorageConfig instance to save
    #     path: Path where to save the configuration
    
    # if hasattr(config, '_manager'):
    #     # New config with manager
    #     config._manager.save_to_file(Path(path))
    # else:
    #     # Old-style config - convert to dict and save
    #     import json
    #     with open(path, 'w') as f:
    #         json.dump(config.to_dict(), f, indent=2)


def ff_get_default_config(environment: Optional[str] = None) -> StorageConfig:
    # Get default configuration for the specified environment (legacy interface).
    
    # Args:
    #     environment: Environment name or None for base config
    #     
    # Returns:
    #     Default StorageConfig for the environment
    
    # if environment == "development":
    #     return FFDevelopmentConfig()
    # elif environment == "production":
    #     return FFProductionConfig()
    # else:
    #     return StorageConfig()


# Import guard to prevent circular imports
# import sys
# if 'config' not in sys.modules:
#     # First import - set up proper structure
#     pass
# else:
#     # Module already imported - maintain compatibility
#     from typing import Optional, List, Dict, Any
#     from pathlib import Path
"""