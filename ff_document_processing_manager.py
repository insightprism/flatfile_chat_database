"""
Document Processing Pipeline for Flatfile Chat Database

NEW: Integrated with PrismMind engine architecture for maximum code reuse.
Provides both legacy API compatibility and new PrismMind-powered processing.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
from ff_class_configs.ff_chat_entities_config import FFDocumentDTO, FFProcessingResult

# Import PrismMind integration
try:
    from prismmind_integration import (
        FFDocumentProcessor,
        FFPrismMindConfig,
        FFPrismMindConfigLoader
    )
    PRISMMIND_INTEGRATION_AVAILABLE = True
except ImportError:
    FFDocumentProcessor = None
    FFPrismMindConfig = None
    FFPrismMindConfigLoader = None
    PRISMMIND_INTEGRATION_AVAILABLE = False


class FFDocumentProcessingManager:
    """
    Document processing pipeline with PrismMind integration.
    
    DEPRECATED: This class provides legacy compatibility. 
    NEW CODE SHOULD USE: FlatfileDocumentProcessor from prismmind_integration
    
    Benefits of new PrismMind integration:
    - Universal file support (PDF, images, URLs)
    - Configuration-driven processing
    - Proven engine architecture
    - Better error handling and performance
    """
    
    def __init__(self, config: Optional[FFConfigurationManagerConfigDTO] = None, use_prismmind: bool = True):
        """
        Initialize pipeline with optional PrismMind integration.
        
        Args:
            config: Storage configuration
            use_prismmind: Whether to use PrismMind integration (recommended)
        """
        self.config = config or load_config()
        self.storage = FFStorageManager(self.config)
        
        # Default settings (legacy)
        self.chunking_strategy = self.config.vector.default_chunking_strategy
        self.embedding_provider = self.config.vector.default_embedding_provider
        
        # PrismMind integration
        self.use_prismmind = use_prismmind and PRISMMIND_INTEGRATION_AVAILABLE
        self.prismmind_processor = None
        
        if self.use_prismmind:
            try:
                # Create PrismMind configuration from flatfile config
                prismmind_config = FFPrismMindConfig(flatfile_config=self.config)
                self.prismmind_processor = FFDocumentProcessor(prismmind_config)
            except Exception as e:
                from ff_utils.ff_logging import get_logger
                logger = get_logger(__name__)
                logger.error(f"Failed to initialize PrismMind integration: {e}", exc_info=True)
                self.use_prismmind = False
        
        if not self.use_prismmind:
            warnings.warn(
                "Using legacy document processing. Consider upgrading to PrismMind integration "
                "for better file support and performance.",
                DeprecationWarning,
                stacklevel=2
            )
    
    async def process_document(
        self,
        document_path: str,
        user_id: str,
        session_id: str,
        document_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FFProcessingResult:
        """
        Process a document through the complete RAG pipeline.
        
        Routes to PrismMind processor when available for enhanced file support,
        falls back to legacy processing for compatibility.
        
        Args:
            document_path: Path to document file
            user_id: User identifier
            session_id: Session identifier
            document_id: Optional document ID (auto-generated if not provided)
            chunking_strategy: Override default chunking
            embedding_provider: Override default embedding
            api_key: API key if required
            metadata: Additional document metadata
            
        Returns:
            FFProcessingResult with pipeline status
        """
        # Route to PrismMind processor if available
        if self.use_prismmind and self.prismmind_processor:
            try:
                return await self.prismmind_processor.process_document(
                    document_path=document_path,
                    user_id=user_id,
                    session_id=session_id,
                    document_id=document_id,
                    chunking_strategy=chunking_strategy,
                    embedding_provider=embedding_provider,
                    api_key=api_key,
                    metadata=metadata
                )
            except Exception as e:
                warnings.warn(
                    f"PrismMind processing failed, falling back to legacy: {e}",
                    UserWarning
                )
                # Fall through to legacy processing
        
        # Legacy processing
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Read document
            doc_path = Path(document_path)
            if not doc_path.exists():
                return FFProcessingResult(
                    success=False,
                    document_id="",
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error=f"Document not found: {document_path}"
                )
            
            # Extract text based on file type
            content = await self._extract_text(doc_path)
            
            if not content:
                return FFProcessingResult(
                    success=False,
                    document_id="",
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error="Failed to extract text from document"
                )
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"doc_{doc_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare metadata
            doc_metadata = {
                "original_filename": doc_path.name,
                "file_type": doc_path.suffix,
                "file_size": doc_path.stat().st_size,
                "processed_at": datetime.now().isoformat(),
                "processing_method": "legacy",
                **(metadata or {})
            }
            
            # Process with vectors
            success = await self.storage.store_document_with_vectors(
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                content=content,
                metadata=doc_metadata,
                chunking_strategy=chunking_strategy or self.chunking_strategy,
                embedding_provider=embedding_provider or self.embedding_provider,
                api_key=api_key
            )
            
            if success:
                # Get chunk count
                chunks = await self.storage.chunking_engine.ff_chunk_text(
                    content,
                    strategy=chunking_strategy or self.chunking_strategy
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                return FFProcessingResult(
                    success=True,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    vector_count=len(chunks),
                    processing_time=processing_time,
                    metadata={
                        "chunking_strategy": chunking_strategy or self.chunking_strategy,
                        "embedding_provider": embedding_provider or self.embedding_provider,
                        "content_length": len(content),
                        "processing_method": "legacy"
                    }
                )
            else:
                return FFProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error="Failed to store document with vectors"
                )
                
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            return FFProcessingResult(
                success=False,
                document_id=document_id or "",
                chunk_count=0,
                vector_count=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def process_text(
        self,
        text: str,
        user_id: str,
        session_id: str,
        document_id: Optional[str] = None,
        **kwargs
    ) -> FFProcessingResult:
        """
        Process raw text through the pipeline.
        
        Args:
            text: Text content to process
            user_id: User identifier
            session_id: Session identifier
            document_id: Optional document ID
            **kwargs: Additional arguments for processing
            
        Returns:
            FFProcessingResult with pipeline status
        """
        start_time = asyncio.get_event_loop().time()
        
        # Generate document ID if not provided
        if not document_id:
            document_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary metadata
        metadata = {
            "source": "direct_text",
            "text_length": len(text),
            "processed_at": datetime.now().isoformat(),
            **kwargs.get("metadata", {})
        }
        
        try:
            # Process with vectors
            success = await self.storage.store_document_with_vectors(
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                content=text,
                metadata=metadata,
                chunking_strategy=kwargs.get("chunking_strategy", self.chunking_strategy),
                embedding_provider=kwargs.get("embedding_provider", self.embedding_provider),
                api_key=kwargs.get("api_key")
            )
            
            if success:
                chunks = await self.storage.chunking_engine.ff_chunk_text(
                    text,
                    strategy=kwargs.get("chunking_strategy", self.chunking_strategy)
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                return FFProcessingResult(
                    success=True,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    vector_count=len(chunks),
                    processing_time=processing_time,
                    metadata={
                        "chunking_strategy": kwargs.get("chunking_strategy", self.chunking_strategy),
                        "embedding_provider": kwargs.get("embedding_provider", self.embedding_provider),
                        "content_length": len(text)
                    }
                )
            else:
                raise Exception("Failed to store document with vectors")
                
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            return FFProcessingResult(
                success=False,
                document_id=document_id,
                chunk_count=0,
                vector_count=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def process_batch(
        self,
        document_paths: List[str],
        user_id: str,
        session_id: str,
        **kwargs
    ) -> List[FFProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            document_paths: List of paths to documents
            user_id: User identifier
            session_id: Session identifier
            **kwargs: Additional arguments for processing
            
        Returns:
            List of FFProcessingResult for each document
        """
        results = []
        
        for path in document_paths:
            result = await self.process_document(
                document_path=path,
                user_id=user_id,
                session_id=session_id,
                **kwargs
            )
            results.append(result)
        
        return results
    
    async def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from various file types.
        Can be extended to use PrismMind's ingestion engines.
        """
        if file_path.suffix.lower() in [".txt", ".md"]:
            return file_path.read_text(encoding='utf-8')
        elif file_path.suffix.lower() == ".json":
            import json
            data = json.loads(file_path.read_text(encoding='utf-8'))
            # Convert JSON to readable text
            return json.dumps(data, indent=2)
        elif file_path.suffix.lower() == ".csv":
            # Simple CSV reading
            import csv
            text_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
            for row in reader:
                    text_lines.append(', '.join(row))
            return '\n'.join(text_lines)
        else:
            # For unsupported types, raise an error or return empty
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    async def reprocess_document(
        self,
        user_id: str,
        session_id: str,
        document_id: str,
        **kwargs
    ) -> FFProcessingResult:
        """
        Reprocess an existing document with new settings.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document to reprocess
            **kwargs: New processing parameters
            
        Returns:
            FFProcessingResult with reprocessing status
        """
        # First, get the existing document
        document = await self.storage.get_document(user_id, session_id, document_id)
        
        if not document:
            return FFProcessingResult(
                success=False,
                document_id=document_id,
                chunk_count=0,
                vector_count=0,
                processing_time=0,
                error="Document not found"
            )
        
        # Delete existing vectors
        await self.storage.delete_document_vectors(user_id, session_id, document_id)
        
        # Reprocess with new settings
        content = document.content.decode('utf-8') if isinstance(document.content, bytes) else document.content
        
        return await self.process_text(
            text=content,
            user_id=user_id,
            session_id=session_id,
            document_id=document_id,
            metadata=document.metadata,
            **kwargs
        )