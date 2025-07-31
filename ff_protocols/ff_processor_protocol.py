"""
Document processor protocol interfaces.

Defines contracts for document processing and analysis operations.
"""

from typing import Protocol, Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class FFProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class FFProcessingResultProtocol(Protocol):
    """
    Processing result interface.
    
    Defines the structure of document processing results.
    """
    
    status: FFProcessingStatus
    document_id: str
    file_path: str
    file_type: str
    processing_time: float
    
    # Extracted content
    text_content: Optional[str]
    metadata: Dict[str, Any]
    
    # Analysis results
    chunks: Optional[List[Dict[str, Any]]]
    embeddings: Optional[List[List[float]]]
    entities: Optional[Dict[str, List[str]]]
    keywords: Optional[List[str]]
    summary: Optional[str]
    
    # Error information
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...


class FFDocumentProcessorProtocol(Protocol):
    """
    Document processor interface.
    
    All document processors must implement this protocol.
    """
    
    async def process(self, file_path: str, 
                     metadata: Optional[Dict[str, Any]] = None,
                     options: Optional[Dict[str, Any]] = None) -> FFProcessingResultProtocol:
        """
        Process a document.
        
        Args:
            file_path: Path to document
            metadata: Optional metadata
            options: Processing options
            
        Returns:
            Processing result
        """
        ...
    
    async def process_batch(self, file_paths: List[str],
                          metadata: Optional[List[Dict[str, Any]]] = None,
                          options: Optional[Dict[str, Any]] = None) -> List[FFProcessingResultProtocol]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of document paths
            metadata: Optional metadata for each document
            options: Processing options
            
        Returns:
            List of processing results
        """
        ...
    
    async def extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content from document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Tuple of (text_content, metadata)
        """
        ...
    
    async def chunk_text(self, text: str, 
                       strategy: Optional[str] = None,
                       options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy
            options: Strategy-specific options
            
        Returns:
            List of chunks with metadata
        """
        ...
    
    async def generate_embeddings(self, texts: List[str],
                                provider: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for text.
        
        Args:
            texts: List of texts to embed
            provider: Embedding provider
            
        Returns:
            List of embedding vectors
        """
        ...
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to entity values
        """
        ...
    
    async def extract_keywords(self, text: str, 
                             max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum keywords to extract
            
        Returns:
            List of keywords
        """
        ...
    
    async def generate_summary(self, text: str,
                             max_length: int = 500) -> str:
        """
        Generate summary of text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        ...
    
    async def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze document structure and content.
        
        Args:
            file_path: Path to document
            
        Returns:
            Analysis results including:
            - file_type: Detected file type
            - file_size: Size in bytes
            - page_count: Number of pages (if applicable)
            - has_images: Whether document contains images
            - has_tables: Whether document contains tables
            - language: Detected language
            - encoding: Text encoding
        """
        ...
    
    async def validate_document(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate document for processing.
        
        Args:
            file_path: Path to document
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported MIME types
        """
        ...
    
    def get_processing_options(self) -> Dict[str, Any]:
        """
        Get available processing options.
        
        Returns:
            Dictionary of option names to descriptions
        """
        ...
    
    async def cleanup(self) -> None:
        """
        Cleanup any resources used by processor.
        """
        ...