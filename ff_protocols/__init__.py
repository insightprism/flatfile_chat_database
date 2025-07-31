"""
Protocols package for flatfile chat database.

Provides protocol-based interfaces for all major components to enable
dependency injection, testing, and loose coupling between modules.
"""

from .ff_storage_protocol import FFStorageProtocol
from .ff_search_protocol import FFSearchProtocol, FFSearchQueryProtocol, FFSearchResultProtocol, FFSearchType
from .ff_vector_store_protocol import FFVectorStoreProtocol
from .ff_processor_protocol import FFDocumentProcessorProtocol, FFProcessingResultProtocol, FFProcessingStatus
from .ff_backend_protocol import FFBackendProtocol
from .ff_file_operations_protocol import FFFileOperationsProtocol

# Backward compatibility aliases
StorageProtocol = FFStorageProtocol
SearchProtocol = FFSearchProtocol
SearchQueryProtocol = FFSearchQueryProtocol
SearchResultProtocol = FFSearchResultProtocol
VectorStoreProtocol = FFVectorStoreProtocol
DocumentProcessorProtocol = FFDocumentProcessorProtocol
ProcessingResultProtocol = FFProcessingResultProtocol
BackendProtocol = FFBackendProtocol
FileOperationsProtocol = FFFileOperationsProtocol

__all__ = [
    # New FF-prefixed protocols
    'FFStorageProtocol',
    'FFSearchProtocol',
    'FFSearchQueryProtocol', 
    'FFSearchResultProtocol',
    'FFSearchType',
    'FFVectorStoreProtocol',
    'FFDocumentProcessorProtocol',
    'FFProcessingResultProtocol',
    'FFProcessingStatus',
    'FFBackendProtocol',
    'FFFileOperationsProtocol',
    # Backward compatibility aliases
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