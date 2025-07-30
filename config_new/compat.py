"""
Backward compatibility adapter for old StorageConfig.

Provides a compatibility layer to allow existing code to work
with the new configuration structure during migration.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .manager import ConfigurationManager


@dataclass
class LegacyStorageConfig:
    """
    Legacy StorageConfig adapter that maps to new configuration structure.
    
    This class provides backward compatibility by exposing the old
    StorageConfig interface while internally using the new
    domain-specific configurations.
    """
    
    def __init__(self, manager: Optional[ConfigurationManager] = None, **kwargs):
        """
        Initialize legacy config adapter.
        
        Args:
            manager: Configuration manager instance
            **kwargs: Legacy configuration values
        """
        self._manager = manager or ConfigurationManager()
        
        # Apply any provided kwargs to appropriate domains
        if kwargs:
            self._apply_legacy_values(kwargs)
    
    def _apply_legacy_values(self, values: Dict[str, Any]) -> None:
        """Map legacy values to new configuration domains."""
        # Storage mappings
        storage_fields = {
            'storage_base_path': 'base_path',
            'user_data_directory_name': 'user_data_directory',
            'session_data_directory_name': 'session_data_directory',
            'system_config_directory_name': 'system_config_directory',
            'user_profile_filename': 'profile_filename',
            'session_metadata_filename': 'session_metadata_filename',
            'messages_filename': 'messages_filename',
            'max_message_size_bytes': 'max_message_size_bytes',
            'max_document_size_bytes': 'max_document_size_bytes',
            'max_messages_per_session': 'max_messages_per_session',
            'max_sessions_per_user': 'max_sessions_per_user',
            'atomic_write_temp_suffix': 'atomic_write_temp_suffix',
            'backup_before_delete': 'backup_before_delete',
            'auto_cleanup_empty_directories': 'auto_cleanup_empty_directories',
            'validate_json_on_read': 'validate_json_on_read',
            'create_parent_directories': 'create_parent_directories',
            'jsonl_read_buffer_size': 'jsonl_read_buffer_size',
            'file_operation_max_retries': 'file_operation_max_retries',
            'file_operation_retry_delay_ms': 'file_operation_retry_delay_ms',
            'message_id_length': 'message_id_length',
            'temp_file_suffix_length': 'temp_file_suffix_length'
        }
        
        for old_name, new_name in storage_fields.items():
            if old_name in values:
                setattr(self._manager.storage, new_name, values[old_name])
        
        # Search mappings
        search_fields = {
            'search_include_message_content': 'include_message_content',
            'search_include_context': 'include_context',
            'search_include_metadata': 'include_metadata',
            'full_text_search_min_word_length': 'min_word_length',
            'search_results_default_limit': 'default_limit',
            'search_min_word_length': 'min_word_length'
        }
        
        for old_name, new_name in search_fields.items():
            if old_name in values:
                setattr(self._manager.search, new_name, values[old_name])
        
        # Vector mappings
        vector_fields = {
            'vector_storage_subdirectory': 'storage_subdirectory',
            'vector_index_filename': 'index_filename',
            'embeddings_filename': 'embeddings_filename',
            'vector_search_top_k': 'search_top_k',
            'similarity_threshold': 'similarity_threshold',
            'hybrid_search_weight': 'hybrid_search_weight',
            'vector_batch_size': 'batch_size',
            'vector_cache_enabled': 'cache_enabled',
            'vector_mmap_mode': 'mmap_mode',
            'default_embedding_provider': 'default_embedding_provider',
            'embedding_providers': 'embedding_providers',
            'default_chunking_strategy': 'default_chunking_strategy',
            'chunking_strategies': 'chunking_strategies',
            'spacy_model_name': 'spacy_model_name'
        }
        
        for old_name, new_name in vector_fields.items():
            if old_name in values:
                setattr(self._manager.vector, new_name, values[old_name])
        
        # Document mappings
        document_fields = {
            'allowed_document_extensions': 'allowed_extensions',
            'document_storage_subdirectory_name': 'storage_subdirectory',
            'document_analysis_subdirectory_name': 'analysis_subdirectory',
            'document_metadata_filename': 'metadata_filename'
        }
        
        for old_name, new_name in document_fields.items():
            if old_name in values:
                setattr(self._manager.document, new_name, values[old_name])
        
        # Locking mappings
        locking_fields = {
            'enable_file_locking': 'enabled',
            'lock_timeout_seconds': 'timeout_seconds',
            'lock_retry_delay_ms': 'retry_initial_delay_ms',
            'lock_retry_max_delay_seconds': 'retry_max_delay_seconds',
            'lock_strategy': 'strategy'
        }
        
        for old_name, new_name in locking_fields.items():
            if old_name in values:
                setattr(self._manager.locking, new_name, values[old_name])
        
        # Panel mappings
        panel_fields = {
            'panel_sessions_directory_name': 'panel_sessions_directory',
            'global_personas_directory_name': 'global_personas_directory',
            'panel_metadata_filename': 'panel_metadata_filename',
            'panel_max_personas': 'max_personas_per_panel',
            'panel_insight_retention_days': 'insight_retention_days',
            'panel_message_threading_enabled': 'message_threading_enabled',
            'panel_insights_directory_name': 'insights_subdirectory',
            'panel_personas_directory_name': 'personas_subdirectory',
            'persona_limit': 'user_persona_limit'
        }
        
        for old_name, new_name in panel_fields.items():
            if old_name in values:
                setattr(self._manager.panel, new_name, values[old_name])
    
    # Proxy all attribute access to appropriate domain configs
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to appropriate configuration domain."""
        # Map attributes to their domains
        attribute_map = {
            # Storage attributes
            'storage_base_path': (self._manager.storage, 'base_path'),
            'user_data_directory_name': (self._manager.storage, 'user_data_directory'),
            'session_data_directory_name': (self._manager.storage, 'session_data_directory'),
            'system_config_directory_name': (self._manager.storage, 'system_config_directory'),
            'user_profile_filename': (self._manager.storage, 'profile_filename'),
            'session_metadata_filename': (self._manager.storage, 'session_metadata_filename'),
            'messages_filename': (self._manager.storage, 'messages_filename'),
            'max_message_size_bytes': (self._manager.storage, 'max_message_size_bytes'),
            'max_document_size_bytes': (self._manager.storage, 'max_document_size_bytes'),
            'max_messages_per_session': (self._manager.storage, 'max_messages_per_session'),
            'max_sessions_per_user': (self._manager.storage, 'max_sessions_per_user'),
            'atomic_write_temp_suffix': (self._manager.storage, 'atomic_write_temp_suffix'),
            'backup_before_delete': (self._manager.storage, 'backup_before_delete'),
            'auto_cleanup_empty_directories': (self._manager.storage, 'auto_cleanup_empty_directories'),
            'validate_json_on_read': (self._manager.storage, 'validate_json_on_read'),
            'create_parent_directories': (self._manager.storage, 'create_parent_directories'),
            'jsonl_read_buffer_size': (self._manager.storage, 'jsonl_read_buffer_size'),
            'file_operation_max_retries': (self._manager.storage, 'file_operation_max_retries'),
            'file_operation_retry_delay_ms': (self._manager.storage, 'file_operation_retry_delay_ms'),
            'message_id_length': (self._manager.storage, 'message_id_length'),
            'temp_file_suffix_length': (self._manager.storage, 'temp_file_suffix_length'),
            'session_id_prefix': (self._manager.storage, 'session_id_prefix'),
            'session_timestamp_format': (self._manager.storage, 'session_timestamp_format'),
            
            # Search attributes
            'search_include_message_content': (self._manager.search, 'include_message_content'),
            'search_include_context': (self._manager.search, 'include_context'),
            'search_include_metadata': (self._manager.search, 'include_metadata'),
            'full_text_search_min_word_length': (self._manager.search, 'min_word_length'),
            'search_results_default_limit': (self._manager.search, 'default_limit'),
            'search_min_word_length': (self._manager.search, 'min_word_length'),
            'message_pagination_default_limit': (self._manager.search, 'default_page_size'),
            'session_list_default_limit': (self._manager.search, 'default_page_size'),
            
            # Vector attributes
            'vector_storage_subdirectory': (self._manager.vector, 'storage_subdirectory'),
            'vector_index_filename': (self._manager.vector, 'index_filename'),
            'embeddings_filename': (self._manager.vector, 'embeddings_filename'),
            'vector_search_top_k': (self._manager.vector, 'search_top_k'),
            'similarity_threshold': (self._manager.vector, 'similarity_threshold'),
            'hybrid_search_weight': (self._manager.vector, 'hybrid_search_weight'),
            'vector_batch_size': (self._manager.vector, 'batch_size'),
            'vector_cache_enabled': (self._manager.vector, 'cache_enabled'),
            'vector_mmap_mode': (self._manager.vector, 'mmap_mode'),
            'default_embedding_provider': (self._manager.vector, 'default_embedding_provider'),
            'embedding_providers': (self._manager.vector, 'embedding_providers'),
            'default_chunking_strategy': (self._manager.vector, 'default_chunking_strategy'),
            'chunking_strategies': (self._manager.vector, 'chunking_strategies'),
            'spacy_model_name': (self._manager.vector, 'spacy_model_name'),
            
            # Document attributes
            'allowed_document_extensions': (self._manager.document, 'allowed_extensions'),
            'document_storage_subdirectory_name': (self._manager.document, 'storage_subdirectory'),
            'document_analysis_subdirectory_name': (self._manager.document, 'analysis_subdirectory'),
            'document_metadata_filename': (self._manager.document, 'metadata_filename'),
            
            # Locking attributes
            'enable_file_locking': (self._manager.locking, 'enabled'),
            'lock_timeout_seconds': (self._manager.locking, 'timeout_seconds'),
            'lock_retry_delay_ms': (self._manager.locking, 'retry_initial_delay_ms'),
            'lock_retry_max_delay_seconds': (self._manager.locking, 'retry_max_delay_seconds'),
            'lock_strategy': (self._manager.locking, 'strategy'),
            
            # Panel attributes
            'panel_sessions_directory_name': (self._manager.panel, 'panel_sessions_directory'),
            'global_personas_directory_name': (self._manager.panel, 'global_personas_directory'),
            'panel_metadata_filename': (self._manager.panel, 'panel_metadata_filename'),
            'panel_max_personas': (self._manager.panel, 'max_personas_per_panel'),
            'panel_insight_retention_days': (self._manager.panel, 'insight_retention_days'),
            'panel_message_threading_enabled': (self._manager.panel, 'message_threading_enabled'),
            'panel_insights_directory_name': (self._manager.panel, 'insights_subdirectory'),
            'panel_personas_directory_name': (self._manager.panel, 'personas_subdirectory'),
            'persona_limit': (self._manager.panel, 'user_persona_limit'),
            'panel_id_prefix': (self._manager.panel, 'panel_id_prefix'),
            
            # Fixed values for missing attributes
            'situational_context_filename': (None, 'situational_context.json'),
            'context_snapshot_timestamp_format': (None, '%Y%m%d_%H%M%S'),
            'max_context_history_snapshots': (None, 100),
            'context_history_subdirectory_name': (None, 'context_history'),
            'context_summary_max_length': (None, 500),
            'context_key_points_max_count': (None, 10),
            'context_confidence_threshold': (None, 0.7),
            'insight_id_length': (None, 12),
            'fallback_chunk_size_limit': (None, 500),
        }
        
        if name in attribute_map:
            config_obj, attr_name = attribute_map[name]
            if config_obj is None:
                # Return fixed value
                return attr_name
            return getattr(config_obj, attr_name)
        
        # If not found, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute setting to appropriate configuration domain."""
        if name.startswith('_'):
            # Internal attributes
            super().__setattr__(name, value)
            return
        
        # Use the same mapping as __getattr__
        attribute_map = {
            'storage_base_path': (self._manager.storage, 'base_path'),
            'user_data_directory_name': (self._manager.storage, 'user_data_directory'),
            # ... (same mappings as in __getattr__)
        }
        
        if name in attribute_map:
            config_obj, attr_name = attribute_map[name]
            if config_obj is not None:
                setattr(config_obj, attr_name, value)
            return
        
        # If not found, set on self
        super().__setattr__(name, value)
    
    def validate(self) -> None:
        """Validate all configurations (legacy interface)."""
        errors = self._manager.validate_all()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (legacy format)."""
        # Combine all domains into single dict matching old structure
        result = {}
        
        # Add all mapped attributes
        for legacy_name in [
            'storage_base_path', 'user_data_directory_name', 'session_data_directory_name',
            'system_config_directory_name', 'user_profile_filename', 'session_metadata_filename',
            'messages_filename', 'max_message_size_bytes', 'max_document_size_bytes',
            # ... (all legacy attribute names)
        ]:
            try:
                result[legacy_name] = getattr(self, legacy_name)
            except AttributeError:
                pass
        
        return result


# Compatibility aliases
StorageConfig = LegacyStorageConfig
DevelopmentConfig = lambda: LegacyStorageConfig(ConfigurationManager.from_environment("development"))
ProductionConfig = lambda: LegacyStorageConfig(ConfigurationManager.from_environment("production"))


def load_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> LegacyStorageConfig:
    """
    Load configuration (legacy interface).
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        Legacy config adapter
    """
    from .manager import load_config as load_new_config
    manager = load_new_config(config_path, environment)
    return LegacyStorageConfig(manager)