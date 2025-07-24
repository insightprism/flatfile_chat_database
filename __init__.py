"""
Flatfile Chat Database System

A simple, modular file-based storage solution for AI chat applications.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from flatfile_chat_database.config import StorageConfig, load_config
from flatfile_chat_database.models import (
    Message, Session, Panel, SituationalContext, Document,
    UserProfile, Persona, PanelMessage, PanelInsight,
    VectorSearchResult, VectorMetadata, ProcessingResult, SearchType
)
from flatfile_chat_database.storage import StorageManager
from flatfile_chat_database.search import SearchQuery, SearchResult, AdvancedSearchEngine
from flatfile_chat_database.vector_storage import FlatfileVectorStorage
from flatfile_chat_database.chunking import ChunkingEngine, ChunkingStrategy
from flatfile_chat_database.embedding import EmbeddingEngine, EmbeddingProvider
from flatfile_chat_database.document_pipeline import DocumentRAGPipeline

__all__ = [
    # Main API
    "StorageManager",
    "DocumentRAGPipeline",
    # Configuration
    "StorageConfig",
    "load_config",
    # Models
    "Message",
    "Session", 
    "Panel",
    "SituationalContext",
    "Document",
    "UserProfile",
    "Persona",
    "PanelMessage",
    "PanelInsight",
    "VectorSearchResult",
    "VectorMetadata",
    "ProcessingResult",
    "SearchType",
    # Search
    "SearchQuery",
    "SearchResult",
    "AdvancedSearchEngine",
    # Vector Components
    "FlatfileVectorStorage",
    "ChunkingEngine",
    "ChunkingStrategy",
    "EmbeddingEngine",
    "EmbeddingProvider"
]