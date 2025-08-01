"""
Correct imports and initialization for the new configuration system.

Replace your current demo imports with these:
"""
import sys
sys.path.append('..')

# CORRECT IMPORTS - Use these instead of legacy adapter
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config  # NEW CONFIG SYSTEM
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFUserProfileDTO, MessageRole
from ff_search_manager import FFSearchQuery, FFSearchManager

# CORRECT INITIALIZATION - Use this instead of legacy config
config = load_config()  # This loads the new configuration system
storage_manager = FFStorageManager(config)

print("✅ FFStorageManager initialized successfully!")
print(f"🏠 Base path: {config.storage.base_path}")  # Note: config.storage.base_path (new structure)
print(f"📁 User directory: {config.storage.user_data_directory}")
print(f"🔍 Default search limit: {config.search.default_limit}")

# Example of accessing new configuration structure:
# config.storage.base_path              # Storage settings
# config.search.default_limit           # Search settings  
# config.vector.default_embedding_provider  # Vector settings
# config.document.allowed_extensions    # Document settings
# config.locking.enabled               # Locking settings
# config.panel.max_personas_per_panel  # Panel settings

# OLD WAY (don't use this):
# from ff_config_legacy_adapter import StorageConfigDTO, ff_load_configDTO
# config = ff_load_configDTO()  # This was causing the error

# NEW WAY (use this):
# from ff_class_configs.ff_configuration_manager_config import load_config
# config = load_config()