"""
Configuration system for flatfile chat database.

Provides comprehensive configuration with meaningful names, environment support,
and type safety through dataclasses.
"""

import os
import json
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StorageConfig:
    """Main configuration for storage system"""
    
    # Storage locations
    storage_base_path: str = "./data"
    user_data_directory_name: str = "users"  # Not generic "users" - be specific
    panel_sessions_directory_name: str = "panel_sessions"
    global_personas_directory_name: str = "personas_global"
    system_config_directory_name: str = "system"
    
    # File naming patterns
    session_id_prefix: str = "chat_session"
    panel_id_prefix: str = "panel"
    session_timestamp_format: str = "%Y%m%d_%H%M%S"
    context_snapshot_timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # File names
    user_profile_filename: str = "profile.json"
    session_metadata_filename: str = "session.json"
    messages_filename: str = "messages.jsonl"
    situational_context_filename: str = "situational_context.json"
    document_metadata_filename: str = "metadata.json"
    panel_metadata_filename: str = "panel.json"
    
    # Storage limits
    max_message_size_bytes: int = 1_048_576  # 1MB
    max_document_size_bytes: int = 104_857_600  # 100MB
    max_messages_per_session: int = 100_000
    max_sessions_per_user: int = 1_000
    max_context_history_snapshots: int = 100
    
    # Performance settings
    message_pagination_default_limit: int = 100
    session_list_default_limit: int = 50
    search_results_default_limit: int = 20
    jsonl_read_buffer_size: int = 8192
    
    # Behavior settings
    atomic_write_temp_suffix: str = ".tmp"
    backup_before_delete: bool = True
    auto_cleanup_empty_directories: bool = True
    validate_json_on_read: bool = True
    create_parent_directories: bool = True
    
    # Search configuration
    search_include_message_content: bool = True
    search_include_context: bool = True
    search_include_metadata: bool = False
    full_text_search_min_word_length: int = 3
    
    # Panel configuration
    panel_max_personas: int = 10
    panel_insight_retention_days: int = 90
    panel_message_threading_enabled: bool = True
    panel_insights_directory_name: str = "insights"
    panel_personas_directory_name: str = "personas"
    
    # Document handling
    allowed_document_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".md", ".json", ".csv", 
        ".png", ".jpg", ".jpeg", ".gif", ".webp"
    ])
    document_storage_subdirectory_name: str = "documents"
    document_analysis_subdirectory_name: str = "analysis"
    
    # Context management
    context_history_subdirectory_name: str = "context_history"
    context_summary_max_length: int = 500
    context_key_points_max_count: int = 10
    context_confidence_threshold: float = 0.7
    
    def validate(self) -> None:
        """Validate configuration values"""
        if self.max_message_size_bytes <= 0:
            raise ValueError("max_message_size_bytes must be positive")
        if self.max_document_size_bytes <= 0:
            raise ValueError("max_document_size_bytes must be positive")
        if self.message_pagination_default_limit <= 0:
            raise ValueError("message_pagination_default_limit must be positive")
        if not self.storage_base_path:
            raise ValueError("storage_base_path cannot be empty")
        if self.context_confidence_threshold < 0 or self.context_confidence_threshold > 1:
            raise ValueError("context_confidence_threshold must be between 0 and 1")


@dataclass
class DevelopmentConfig(StorageConfig):
    """Development environment overrides"""
    storage_base_path: str = "./dev_data"
    validate_json_on_read: bool = True
    backup_before_delete: bool = True
    max_messages_per_session: int = 10_000  # Lower for testing
    

@dataclass
class ProductionConfig(StorageConfig):
    """Production environment overrides"""
    storage_base_path: str = "/var/lib/chatdb/data"
    validate_json_on_read: bool = False  # Performance optimization
    message_pagination_default_limit: int = 50
    search_results_default_limit: int = 10
    jsonl_read_buffer_size: int = 16384  # Larger buffer for production


def load_config(config_path: Optional[str] = None, 
                environment: Optional[str] = None) -> StorageConfig:
    """
    Load configuration from file and environment.
    
    Priority (highest to lowest):
    1. Environment variables (CHATDB_*)
    2. Environment-specific config class
    3. Config file
    4. Default values
    
    Args:
        config_path: Path to JSON configuration file
        environment: Environment name ('development', 'production')
        
    Returns:
        Validated StorageConfig instance
    """
    # Start with base config
    config = StorageConfig()
    
    # Load from JSON file if provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = json.load(f)
            # Only update fields that exist in the dataclass
            valid_fields = {f.name for f in dataclasses.fields(StorageConfig)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            config = StorageConfig(**filtered_dict)
    
    # Apply environment-specific overrides
    if environment == "development":
        config = DevelopmentConfig()
    elif environment == "production":
        config = ProductionConfig()
    
    # Allow environment variables to override any setting
    # Format: CHATDB_SETTING_NAME (uppercase)
    for field in dataclasses.fields(config):
        env_var = f"CHATDB_{field.name.upper()}"
        if env_var in os.environ:
            value = os.environ[env_var]
            # Convert to appropriate type
            field_type = field.type
            
            # Handle basic type conversions
            if field_type == int:
                value = int(value)
            elif field_type == float:
                value = float(value)
            elif field_type == bool:
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                # Handle list types (like allowed_document_extensions)
                value = json.loads(value) if value.startswith('[') else value.split(',')
            
            setattr(config, field.name, value)
    
    # Validate configuration
    config.validate()
    
    return config


def save_config(config: StorageConfig, path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: StorageConfig instance to save
        path: Path where to save the configuration
    """
    config_dict = dataclasses.asdict(config)
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def get_default_config(environment: Optional[str] = None) -> StorageConfig:
    """
    Get default configuration for the specified environment.
    
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
        return StorageConfig()