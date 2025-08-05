"""
Flatfile Chat Database System

A simple, modular file-based storage solution for AI chat applications.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFPanelDTO, FFSituationalContextDTO, FFDocumentDTO,
    FFUserProfileDTO, FFPersonaDTO, FFPanelMessageDTO, FFPanelInsightDTO,
    FFVectorSearchResultDTO, FFVectorMetadataDTO, FFProcessingResultDTO, SearchType
)
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchQueryDTO, FFSearchResultDTO, FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_chunking_manager import FFChunkingManager, FFChunkingConfigDTO
from ff_embedding_manager import FFEmbeddingManager, FFEmbeddingProviderDTO
from ff_document_processing_manager import FFDocumentProcessingManager

__all__ = [
    # Main API
    "FFStorageManager",
    "FFDocumentProcessingManager",
    # Configuration
    "FFConfigurationManagerConfigDTO",
    "load_config",
    # Models
    "FFMessageDTO",
    "FFSessionDTO", 
    "FFPanelDTO",
    "FFSituationalContextDTO",
    "FFDocumentDTO",
    "FFUserProfileDTO",
    "FFPersonaDTO",
    "FFPanelMessageDTO",
    "FFPanelInsightDTO",
    "FFVectorSearchResultDTO",
    "FFVectorMetadataDTO",
    "FFProcessingResultDTO",
    "SearchType",
    # Search
    "FFSearchQueryDTO",
    "FFSearchResultDTO",
    "FFSearchManager",
    # Vector Components
    "FFVectorStorageManager",
    "FFChunkingManager",
    "FFChunkingConfigDTO",
    "FFEmbeddingManager",
    "FFEmbeddingProviderDTO"
]