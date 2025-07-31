"""
Configuration package for flatfile chat database.

This package provides domain-specific configuration classes following
clean architecture principles with clear separation of concerns.
"""

from .ff_base_config import FFBaseConfigDTO
from .ff_storage_config import FFStorageConfigDTO
from .ff_search_config import FFSearchConfigDTO
from .ff_vector_storage_config import FFVectorStorageConfigDTO
from .ff_document_config import FFDocumentConfigDTO
from .ff_locking_config import FFLockingConfigDTO
from .ff_persona_panel_config import FFPersonaPanelConfigDTO
from .ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from .ff_chat_entities_config import (
    MessageRole, PanelType, SearchType,
    FFMessageDTO, FFSessionDTO, FFUserProfileDTO, FFPanelDTO, FFPanelMessageDTO,
    FFPersonaDTO, FFPanelInsightDTO, FFDocumentDTO, FFSituationalContextDTO,
    FFVectorSearchResultDTO, FFVectorMetadataDTO, FFProcessingResultDTO,
    generate_message_id, generate_insight_id, current_timestamp
)

__all__ = [
    'FFBaseConfigDTO',
    'FFStorageConfigDTO',
    'FFSearchConfigDTO',
    'FFVectorStorageConfigDTO',
    'FFDocumentConfigDTO',
    'FFLockingConfigDTO',
    'FFPersonaPanelConfigDTO',
    'FFConfigurationManagerConfigDTO',
    # Chat entities
    'MessageRole',
    'PanelType', 
    'SearchType',
    'FFMessageDTO',
    'FFSessionDTO',
    'FFUserProfileDTO',
    'FFPanelDTO',
    'FFPanelMessageDTO',
    'FFPersonaDTO',
    'FFPanelInsightDTO',
    'FFDocumentDTO',
    'FFSituationalContextDTO',
    'FFVectorSearchResultDTO',
    'FFVectorMetadataDTO',
    'FFProcessingResultDTO',
    # Utility functions
    'generate_message_id',
    'generate_insight_id',
    'current_timestamp'
]