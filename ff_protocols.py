"""
Protocol definitions for dependency injection and loose coupling.

Defines abstract interfaces that concrete implementations must follow,
enabling dependency injection and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from pathlib import Path

from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFUserProfileDTO,
    FFSituationalContextDTO, FFPersonaDTO, FFPanelDTO, FFPanelMessageDTO
)


class StorageProtocol(ABC):
    """Protocol for storage operations."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize storage backend."""
        pass
    
    @abstractmethod
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new user."""
        pass
    
    @abstractmethod
    async def create_session(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new session."""
        pass
    
    @abstractmethod
    async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool:
        """Add a message to a session."""
        pass
    
    @abstractmethod
    async def get_messages(self, user_id: str, session_id: str, 
                          limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]:
        """Get messages from a session."""
        pass


class SearchProtocol(ABC):
    """Protocol for search operations."""
    
    @abstractmethod
    async def search(self, query: str, user_id: Optional[str] = None, 
                    session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform search across sessions."""
        pass
    
    @abstractmethod
    async def build_search_index(self, user_id: str) -> Dict[str, Any]:
        """Build search index for user."""
        pass


class VectorStoreProtocol(ABC):
    """Protocol for vector storage operations."""
    
    @abstractmethod
    async def store_embeddings(self, session_id: str, embeddings: List[Dict[str, Any]]) -> bool:
        """Store embeddings for a session."""
        pass
    
    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], 
                               session_id: Optional[str] = None,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        pass


class DocumentProcessorProtocol(ABC):
    """Protocol for document processing operations."""
    
    @abstractmethod
    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a document and extract metadata."""
        pass
    
    @abstractmethod
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from a document."""
        pass


class BackendProtocol(ABC):
    """Protocol for storage backend operations."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize backend."""
        pass
    
    @abstractmethod
    async def read(self, key: str) -> Optional[bytes]:
        """Read data from storage."""
        pass
    
    @abstractmethod
    async def write(self, key: str, data: bytes) -> bool:
        """Write data to storage."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from storage."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "", pattern: Optional[str] = None) -> List[str]:
        """List keys with optional prefix/pattern."""
        pass


class FileOperationsProtocol(ABC):
    """Protocol for file operations."""
    
    @abstractmethod
    async def read_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file."""
        pass
    
    @abstractmethod
    async def write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file."""
        pass
    
    @abstractmethod
    async def append_jsonl(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Append to JSONL file."""
        pass
    
    @abstractmethod
    async def read_jsonl(self, file_path: Path, limit: Optional[int] = None, 
                        offset: int = 0) -> List[Dict[str, Any]]:
        """Read JSONL file."""
        pass


class CompressionProtocol(ABC):
    """Protocol for compression operations."""
    
    @abstractmethod
    async def compress_data(self, data: bytes) -> bytes:
        """Compress data."""
        pass
    
    @abstractmethod
    async def decompress_data(self, data: bytes) -> bytes:
        """Decompress data."""
        pass
    
    @abstractmethod
    async def get_compression_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get compression statistics."""
        pass


class StreamingProtocol(ABC):
    """Protocol for streaming operations."""
    
    @abstractmethod
    async def stream_messages(self, user_id: str, session_id: str,
                             start_offset: int = 0,
                             max_messages: Optional[int] = None) -> AsyncIterator[List[FFMessageDTO]]:
        """Stream messages from a session."""
        pass
    
    @abstractmethod
    async def stream_session_export(self, user_id: str, session_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream session data for export."""
        pass


class EmbeddingProtocol(ABC):
    """Protocol for embedding operations."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass


class ChunkingProtocol(ABC):
    """Protocol for text chunking operations."""
    
    @abstractmethod
    async def chunk_text(self, text: str, strategy: str = "optimized_summary") -> List[str]:
        """Chunk text using specified strategy."""
        pass
    
    @abstractmethod
    async def estimate_chunks(self, text: str, strategy: Optional[str] = None) -> int:
        """Estimate number of chunks for text."""
        pass