"""
Interfaces package for flatfile chat database.

Provides protocol-based interfaces for all major components to enable
dependency injection, testing, and loose coupling between modules.
"""

from .storage import StorageProtocol
from .search import SearchProtocol, SearchQueryProtocol, SearchResultProtocol
from .vector_store import VectorStoreProtocol
from .processor import DocumentProcessorProtocol, ProcessingResultProtocol
from .backend import BackendProtocol
from .file_operations import FileOperationsProtocol

__all__ = [
    'StorageProtocol',
    'SearchProtocol',
    'SearchQueryProtocol',
    'SearchResultProtocol',
    'VectorStoreProtocol',
    'DocumentProcessorProtocol',
    'ProcessingResultProtocol',
    'BackendProtocol',
    'FileOperationsProtocol'
]