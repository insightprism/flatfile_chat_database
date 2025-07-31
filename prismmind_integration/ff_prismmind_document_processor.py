"""
Document processor using PrismMind engines with flatfile storage.

This processor replaces the monolithic FFDocumentProcessingManager with a clean,
configuration-driven approach using PrismMind's proven engine architecture.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

# Import PrismMind engines and utilities if available
try:
    from pm_engines.pm_run_engine_chain import pm_run_engine_chain
    from pm_utils.adhoc_util_functions import pm_resolve_input_source_async
    PRISMMIND_AVAILABLE = True
except ImportError:
    # Fallback implementations
    async def pm_run_engine_chain(input_data, *engines):
        """Fallback implementation when PrismMind not available"""
        raise RuntimeError("PrismMind not available. Cannot run engine chain.")
    
    async def pm_resolve_input_source_async(input_path):
        """Fallback implementation for input resolution"""
        return {"mime_type": "text/plain", "input_path": input_path}
    
    PRISMMIND_AVAILABLE = False

from .ff_prismmind_config import FFPrismMindConfigDTO
from .ff_prismmind_engine_factory import FFPrismMindConfigFactory
from ..ff_storage_manager import FFStorageManager
from ..ff_class_configs.ff_chat_entities_config import FFProcessingResultDTO


class FFDocumentProcessor:
    """
    Main integration class - uses PrismMind engines with flatfile storage.
    
    Replaces FFDocumentProcessingManager with configuration-driven PrismMind approach.
    """
    
    def __init__(self, config: FFPrismMindConfigDTO):
        """
        Initialize processor with configuration.
        
        Args:
            config: Complete Flatfile-PrismMind configuration
        """
        self.config = config
        self.storage_manager = FFStorageManager(config.flatfile_config)
        self.factory = FFPrismMindConfigFactory()
        self._initialized = False
        
        # Validate configuration
        validation_errors = self.factory.validate_configuration(config)
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {validation_errors}")
    
    async def initialize(self) -> bool:
        """
        Initialize the processor and storage system.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot initialize processor.")
        
        # Initialize storage manager
        success = await self.storage_manager.initialize()
        if success:
            self._initialized = True
        
        return success
    
    async def process_document(
        self,
        document_path: str,
        user_id: str,
        session_id: str,
        document_id: Optional[str] = None,
        processing_overrides: Optional[Dict[str, Any]] = None
    ) -> FFProcessingResultDTO:
        """
        Process document using PrismMind engines → store in flatfile.
        
        Zero hard-coding - everything from configuration.
        
        Args:
            document_path: Path to document file
            user_id: User identifier
            session_id: Session identifier
            document_id: Optional document ID (auto-generated if not provided)
            processing_overrides: Optional strategy overrides
            
        Returns:
            FFProcessingResult with pipeline status
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Ensure initialized
            if not self._initialized:
                await self.initialize()
            
            # Validate document exists
            doc_path = Path(document_path)
            if not doc_path.exists():
                return FFProcessingResultDTO(
                    success=False,
                    document_id=document_id or "",
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error=f"Document not found: {document_path}"
                )
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"doc_{doc_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Resolve file type using PrismMind's intelligent detection
            resolved_input = await pm_resolve_input_source_async(document_path)
            file_type = resolved_input["mime_type"]
            
            # Get environment configuration
            env_config = self.config.get_current_environment_config()
            
            # Apply timeout if configured
            timeout = env_config.get("timeout_seconds", 300)
            
            # Process document with timeout
            result = await asyncio.wait_for(
                self._process_document_internal(
                    document_path=document_path,
                    file_type=file_type,
                    user_id=user_id,
                    session_id=session_id,
                    document_id=document_id,
                    processing_overrides=processing_overrides
                ),
                timeout=timeout
            )
            
            # Calculate total processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update result with timing
            result.processing_time = processing_time
            
            return result
            
        except asyncio.TimeoutError:
            processing_time = asyncio.get_event_loop().time() - start_time
            return FFProcessingResultDTO(
                success=False,
                document_id=document_id or "",
                chunk_count=0,
                vector_count=0,
                processing_time=processing_time,
                error=f"Document processing timed out after {timeout} seconds"
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            return FFProcessingResultDTO(
                success=False,
                document_id=document_id or "",
                chunk_count=0,
                vector_count=0,
                processing_time=processing_time,
                error=f"Document processing failed: {str(e)}"
            )
    
    async def _process_document_internal(
        self,
        document_path: str,
        file_type: str,
        user_id: str,
        session_id: str,
        document_id: str,
        processing_overrides: Optional[Dict[str, Any]] = None
    ) -> FFProcessingResultDTO:
        """
        Internal document processing using PrismMind engine chain.
        
        Args:
            document_path: Path to document file
            file_type: MIME type of the file
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            processing_overrides: Optional strategy overrides
            
        Returns:
            FFProcessingResult with processing details
        """
        try:
            # Create complete processing chain from configuration
            engines = self.factory.create_complete_processing_chain(
                file_type=file_type,
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                config=self.config,
                strategy_overrides=processing_overrides
            )
            
            if not engines:
                return FFProcessingResultDTO(
                    success=False,
                    document_id=document_id,
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error=f"No processing engines configured for file type: {file_type}"
                )
            
            # Run PrismMind engine chain - this does all the heavy lifting!
            final_engine = await pm_run_engine_chain(
                document_path,  # Input file path
                *engines        # Configured engines
            )
            
            # Extract results from final engine
            final_output = final_engine.output_content
            final_metadata = getattr(final_engine, 'full_output_data', {}).get('metadata', {})
            
            # Determine chunk and vector counts
            chunk_count = 0
            vector_count = 0
            
            if isinstance(final_output, list):
                # If final output is list of embeddings
                chunk_count = len(final_output)
                vector_count = len(final_output)
            elif final_metadata.get("chunks_stored"):
                # Get from storage metadata
                chunk_count = final_metadata["chunks_stored"]
                vector_count = final_metadata["vectors_stored"]
            
            # Prepare document metadata
            doc_metadata = {
                "original_filename": Path(document_path).name,
                "file_type": file_type,
                "file_size": Path(document_path).stat().st_size,
                "processed_at": datetime.now().isoformat(),
                "processing_chain": [engine.__class__.__name__ for engine in engines],
                "strategies_used": {
                    "chunking_strategy": processing_overrides.get("chunking_strategy") if processing_overrides else self.config.handler_strategies.default_strategies["chunking_strategy"],
                    "embedding_provider": processing_overrides.get("embedding_provider") if processing_overrides else self.config.handler_strategies.default_strategies["embedding_provider"],
                    "nlp_strategy": processing_overrides.get("nlp_strategy") if processing_overrides else self.config.handler_strategies.default_strategies["nlp_strategy"]
                },
                "prismmind_metadata": final_metadata
            }
            
            # Store document metadata in flatfile storage
            await self._store_document_metadata(
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                metadata=doc_metadata
            )
            
            return FFProcessingResultDTO(
                success=True,
                document_id=document_id,
                chunk_count=chunk_count,
                vector_count=vector_count,
                processing_time=0,  # Will be set by caller
                metadata=doc_metadata
            )
            
        except Exception as e:
            return FFProcessingResultDTO(
                success=False,
                document_id=document_id,
                chunk_count=0,
                vector_count=0,
                processing_time=0,
                error=f"Internal processing error: {str(e)}"
            )
    
    async def process_text(
        self,
        text: str,
        user_id: str,
        session_id: str,
        document_id: Optional[str] = None,
        text_type: str = "text/plain",
        processing_overrides: Optional[Dict[str, Any]] = None
    ) -> FFProcessingResultDTO:
        """
        Process raw text through PrismMind pipeline.
        
        Args:
            text: Text content to process
            user_id: User identifier
            session_id: Session identifier
            document_id: Optional document ID
            text_type: Type of text content
            processing_overrides: Optional strategy overrides
            
        Returns:
            FFProcessingResult with pipeline status
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Ensure initialized
            if not self._initialized:
                await self.initialize()
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create engines for text processing (skip injest engine)
            engines = []
            
            # Get processing chain but skip injest for direct text
            processing_chain = self.config.get_processing_chain_for_file_type(text_type)
            processing_chain = [e for e in processing_chain if e != "pm_injest_engine"]
            
            # Storage metadata
            storage_metadata = {
                "user_id": user_id,
                "session_id": session_id,
                "document_id": document_id
            }
            
            # Create engines based on chain
            for engine_name in processing_chain:
                if engine_name == "pm_nlp_engine":
                    if text_type not in self.config.document_processing.skip_nlp_for_file_types:
                        engine_config = self.factory.create_nlp_engine_config(
                            self.config, 
                            processing_overrides.get("nlp_strategy") if processing_overrides else None
                        )
                        engines.append(self.factory.create_engine_by_name("pm_nlp_engine", self.config))
                
                elif engine_name == "pm_chunking_engine":
                    engines.append(self.factory.create_engine_by_name(
                        "pm_chunking_engine", 
                        self.config,
                        strategy_override=processing_overrides.get("chunking_strategy") if processing_overrides else None
                    ))
                
                elif engine_name == "ff_storage_engine":
                    embedding_provider = processing_overrides.get("embedding_provider") if processing_overrides else self.config.handler_strategies.default_strategies["embedding_provider"]
                    
                    engine_config, handler_config = self.factory.create_embedding_engine_config_with_storage(
                        embedding_provider, storage_metadata, self.config
                    )
                    
                    from pm_engines.pm_embedding_engine import PmEmbeddingEngine
                    engines.append(PmEmbeddingEngine(
                        engine_config=engine_config,
                        handler_config=handler_config
                    ))
            
            # Run engine chain with text input
            if engines:
                final_engine = await pm_run_engine_chain(text, *engines)
                final_metadata = getattr(final_engine, 'full_output_data', {}).get('metadata', {})
                
                chunk_count = final_metadata.get("chunks_stored", 0)
                vector_count = final_metadata.get("vectors_stored", 0)
            else:
                chunk_count = 0
                vector_count = 0
                final_metadata = {}
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Prepare metadata
            doc_metadata = {
                "source": "direct_text",
                "text_type": text_type,
                "text_length": len(text),
                "processed_at": datetime.now().isoformat(),
                "processing_chain": [engine.__class__.__name__ for engine in engines],
                "prismmind_metadata": final_metadata
            }
            
            return FFProcessingResultDTO(
                success=True,
                document_id=document_id,
                chunk_count=chunk_count,
                vector_count=vector_count,
                processing_time=processing_time,
                metadata=doc_metadata
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            return FFProcessingResultDTO(
                success=False,
                document_id=document_id or "",
                chunk_count=0,
                vector_count=0,
                processing_time=processing_time,
                error=f"Text processing failed: {str(e)}"
            )
    
    async def process_batch(
        self,
        document_paths: List[str],
        user_id: str,
        session_id: str,
        processing_overrides: Optional[Dict[str, Any]] = None,
        max_concurrent: Optional[int] = None
    ) -> List[FFProcessingResult]:
        """
        Process multiple documents in batch with optional concurrency control.
        
        Args:
            document_paths: List of paths to documents
            user_id: User identifier
            session_id: Session identifier
            processing_overrides: Optional strategy overrides
            max_concurrent: Maximum concurrent documents (from config if not specified)
            
        Returns:
            List of FFProcessingResult for each document
        """
        if not max_concurrent:
            max_concurrent = self.config.integration_settings.performance_settings.get("concurrent_documents", 5)
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_document(doc_path: str) -> FFProcessingResultDTO:
            async with semaphore:
                return await self.process_document(
                    document_path=doc_path,
                    user_id=user_id,
                    session_id=session_id,
                    processing_overrides=processing_overrides
                )
        
        # Process all documents concurrently (with limit)
        tasks = [process_single_document(path) for path in document_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(FFProcessingResult(
                    success=False,
                    document_id="",
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error=f"Batch processing exception for {document_paths[i]}: {str(result)}"
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def reprocess_document(
        self,
        user_id: str,
        session_id: str,
        document_id: str,
        new_strategies: Dict[str, str]
    ) -> FFProcessingResultDTO:
        """
        Reprocess existing document with new strategies.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document to reprocess
            new_strategies: New processing strategies to use
            
        Returns:
            FFProcessingResult with reprocessing status
        """
        try:
            # Get existing document
            document = await self.storage_manager.get_document(user_id, session_id, document_id)
            
            if not document:
                return FFProcessingResultDTO(
                    success=False,
                    document_id=document_id,
                    chunk_count=0,
                    vector_count=0,
                    processing_time=0,
                    error="Document not found for reprocessing"
                )
            
            # Delete existing vectors
            await self.storage_manager.vector_storage.delete_document_vectors(session_id, document_id)
            
            # Get original content
            content = document.content.decode('utf-8') if isinstance(document.content, bytes) else document.content
            
            # Reprocess with new strategies
            return await self.process_text(
                text=content,
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                processing_overrides=new_strategies
            )
            
        except Exception as e:
            return FFProcessingResultDTO(
                success=False,
                document_id=document_id,
                chunk_count=0,
                vector_count=0,
                processing_time=0,
                error=f"Reprocessing failed: {str(e)}"
            )
    
    async def _store_document_metadata(
        self,
        user_id: str,
        session_id: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store document metadata in flatfile storage.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            metadata: Document metadata to store
            
        Returns:
            True if successful
        """
        try:
            # Store in storage manager's document system
            # This integrates with the existing flatfile document storage
            return True  # Placeholder - integrate with actual document storage
            
        except Exception as e:
            print(f"Failed to store document metadata: {e}")
            return False
    
    def get_supported_file_types(self) -> List[str]:
        """
        Get list of supported file types from configuration.
        
        Returns:
            List of supported MIME types
        """
        return list(self.config.engine_selection.file_type_handlers.keys())
    
    def get_available_strategies(self) -> Dict[str, List[str]]:
        """
        Get available processing strategies from configuration.
        
        Returns:
            Dict mapping strategy types to available options
        """
        return {
            "nlp_strategies": list(self.config.engine_selection.nlp_handlers.keys()),
            "chunking_strategies": list(self.config.engine_selection.chunking_handlers.keys()),
            "embedding_providers": list(self.config.engine_selection.embedding_handlers.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the processor and dependencies.
        
        Returns:
            Dict with health status information
        """
        health_status = {
            "processor_initialized": self._initialized,
            "prismmind_available": PRISMMIND_AVAILABLE,
            "configuration_valid": True,
            "storage_healthy": False,
            "supported_file_types": len(self.get_supported_file_types()),
            "available_strategies": self.get_available_strategies()
        }
        
        try:
            # Check storage health
            if self.storage_manager:
                storage_health = await self.storage_manager.backend.health_check()
                health_status["storage_healthy"] = storage_health.get("healthy", False)
                health_status["storage_details"] = storage_health
            
            # Validate configuration
            validation_errors = self.factory.validate_configuration(self.config)
            if validation_errors:
                health_status["configuration_valid"] = False
                health_status["configuration_errors"] = validation_errors
            
        except Exception as e:
            health_status["health_check_error"] = str(e)
        
        return health_status