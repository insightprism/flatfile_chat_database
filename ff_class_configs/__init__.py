"""
Configuration package for flatfile chat database.

This package provides domain-specific configuration classes following
clean architecture principles with clear separation of concerns.
"""

from .ff_base_config import FFBaseConfig
from .ff_storage_config import FFStorageConfig
from .ff_search_config import FFSearchConfig
from .ff_vector_storage_config import FFVectorStorageConfig
from .ff_document_config import FFDocumentConfig
from .ff_locking_config import FFLockingConfig
from .ff_persona_panel_config import FFPersonaPanelConfig
from .ff_configuration_manager_config import FFConfigurationManagerConfig
from .ff_chat_entities_config import (
    MessageRole, PanelType, SearchType,
    FFMessage, FFSession, FFUserProfile, FFPanel, FFPanelMessage,
    FFPersona, FFPanelInsight, FFDocument, FFSituationalContext,
    FFVectorSearchResult, FFVectorMetadata, FFProcessingResult,
    generate_message_id, generate_insight_id, current_timestamp
)

__all__ = [
    'FFBaseConfig',
    'FFStorageConfig',
    'FFSearchConfig',
    'FFVectorStorageConfig',
    'FFDocumentConfig',
    'FFLockingConfig',
    'FFPersonaPanelConfig',
    'FFConfigurationManagerConfig',
    # Chat entities
    'MessageRole',
    'PanelType', 
    'SearchType',
    'FFMessage',
    'FFSession',
    'FFUserProfile',
    'FFPanel',
    'FFPanelMessage',
    'FFPersona',
    'FFPanelInsight',
    'FFDocument',
    'FFSituationalContext',
    'FFVectorSearchResult',
    'FFVectorMetadata',
    'FFProcessingResult',
    # Utility functions
    'generate_message_id',
    'generate_insight_id',
    'current_timestamp'
]