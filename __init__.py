"""
Flatfile Chat Database System

A simple, modular file-based storage solution for AI chat applications.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from config import StorageConfig, load_config
from models import (
    Message, Session, Panel, SituationalContext, Document,
    UserProfile, Persona, PanelMessage, PanelInsight,
    VectorSearchResult, VectorMetadata, ProcessingResult, SearchType
)
from storage import StorageManager
from search import SearchQuery, SearchResult, AdvancedSearchEngine
from vector_storage import FlatfileVectorStorage
from chunking import ChunkingEngine, ChunkingStrategy
from embedding import EmbeddingEngine, EmbeddingProvider
from document_pipeline import DocumentRAGPipeline

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