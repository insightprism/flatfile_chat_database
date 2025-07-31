"""
Flatfile Chat Database System

A simple, modular file-based storage solution for AI chat applications.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from ff_config_legacy_adapter import StorageConfig, ff_load_config
from ff_class_configs.ff_chat_entities_config import (
    FFMessage, FFSession, FFPanel, FFSituationalContext, FFDocument,
    FFUserProfile, FFPersona, FFPanelMessage, FFPanelInsight,
    FFVectorSearchResult, FFVectorMetadata, FFProcessingResult, SearchType
)
from ff_storage_manager import FFStorageManager
from ff_search_manager import SearchQuery, SearchResult, FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_chunking_manager import FFChunkingManager, FFChunkingConfig
from ff_embedding_manager import FFEmbeddingManager, FFEmbeddingProvider
from ff_document_processing_manager import FFDocumentProcessingManager

__all__ = [
    # Main API
    "FFStorageManager",
    "FFDocumentProcessingManager",
    # Configuration
    "StorageConfig",
    "ff_load_config",
    # Models
    "FFMessage",
    "FFSession", 
    "FFPanel",
    "FFSituationalContext",
    "FFDocument",
    "FFUserProfile",
    "FFPersona",
    "FFPanelMessage",
    "FFPanelInsight",
    "FFVectorSearchResult",
    "FFVectorMetadata",
    "FFProcessingResult",
    "SearchType",
    # Search
    "SearchQuery",
    "SearchResult",
    "FFSearchManager",
    # Vector Components
    "FFVectorStorageManager",
    "FFChunkingManager",
    "FFChunkingConfig",
    "FFEmbeddingManager",
    "FFEmbeddingProvider"
]