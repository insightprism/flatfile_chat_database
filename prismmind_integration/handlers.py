"""
Custom handlers for Flatfile-PrismMind integration.

This module contains the minimal custom handlers needed to bridge PrismMind
engines with flatfile storage. Follows PrismMind handler patterns exactly.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# Import PrismMind handler decoration if available
try:
    from pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec
    PRISMMIND_AVAILABLE = True
except ImportError:
    # Fallback decorator if PrismMind not available
    def pm_trace_handler_log_dec(func):
        return func
    PRISMMIND_AVAILABLE = False

# Import PrismMind config types if available
try:
    from pm_config.pm_embedding_engine_config import (
        pm_embedding_engine_config_dto, 
        pm_embedding_handler_config_dto
    )
except ImportError:
    # Fallback types
    pm_embedding_engine_config_dto = object
    pm_embedding_handler_config_dto = object

from flatfile_chat_database.vector_storage import FlatfileVectorStorage
from flatfile_chat_database.config import StorageConfig


@pm_trace_handler_log_dec
async def ff_store_vectors_handler_async(
    input_data: Dict[str, Any],
    engine_config: pm_embedding_engine_config_dto,
    handler_config: pm_embedding_handler_config_dto,
    rag_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    PrismMind handler that stores embedding results in flatfile storage.
    
    This is the ONLY custom handler needed - bridges PrismMind → Flatfile.
    Follows PrismMind handler signature exactly.
    
    Args:
        input_data: Dict containing embedding results from previous engine
        engine_config: PrismMind embedding engine configuration
        handler_config: Handler-specific configuration with flatfile metadata
        rag_data: Optional RAG context data
        
    Returns:
        Dict with standard PrismMind output format
    """
    start_time = datetime.now()
    
    try:
        # Extract embeddings from PrismMind format
        embeddings = input_data.get("input_content", [])
        
        if not embeddings:
            return {
                "output_content": [],
                "metadata": {
                    "handler_name": "ff_store_vectors_handler_async",
                    "error": "No embeddings found in input_data",
                    "success": False
                },
                "success": False
            }
        
        # Extract flatfile-specific metadata from handler_config
        flatfile_metadata = getattr(handler_config, 'metadata', {})
        
        user_id = flatfile_metadata.get("user_id")
        session_id = flatfile_metadata.get("session_id") 
        document_id = flatfile_metadata.get("document_id")
        flatfile_config = flatfile_metadata.get("flatfile_config")
        
        if not all([user_id, session_id, document_id, flatfile_config]):
            return {
                "output_content": embeddings,
                "metadata": {
                    "handler_name": "ff_store_vectors_handler_async",
                    "error": "Missing required metadata: user_id, session_id, document_id, or flatfile_config",
                    "success": False
                },
                "success": False
            }
        
        # Initialize flatfile vector storage
        vector_storage = FlatfileVectorStorage(flatfile_config)
        
        # Extract chunks and vectors from PrismMind embedding format
        chunks = []
        vectors = []
        
        for embedding in embeddings:
            if isinstance(embedding, dict):
                # PrismMind format: {"text": "...", "embedding_vector": [...], "index": 0}
                chunk_text = embedding.get("text", "")
                embedding_vector = embedding.get("embedding_vector", [])
                
                if chunk_text and embedding_vector:
                    chunks.append(chunk_text)
                    vectors.append(embedding_vector)
        
        if not chunks or not vectors:
            return {
                "output_content": embeddings,
                "metadata": {
                    "handler_name": "ff_store_vectors_handler_async",
                    "error": "No valid chunks or vectors found in embeddings",
                    "success": False
                },
                "success": False
            }
        
        # Store vectors in flatfile storage
        storage_metadata = {
            "provider": getattr(engine_config, 'embedding_provider', 'unknown'),
            "model": getattr(engine_config, 'embedding_model_name', 'unknown'),
            "processing_timestamp": start_time.isoformat(),
            "handler_name": "ff_store_vectors_handler_async"
        }
        
        success = await vector_storage.store_vectors(
            session_id=session_id,
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            metadata=storage_metadata
        )
        
        if not success:
            return {
                "output_content": embeddings,
                "metadata": {
                    "handler_name": "ff_store_vectors_handler_async",
                    "error": "Failed to store vectors in flatfile storage",
                    "success": False
                },
                "success": False
            }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Return successful result in PrismMind format
        result_metadata = {
            "handler_name": "ff_store_vectors_handler_async",
            "stored_in_flatfile": True,
            "user_id": user_id,
            "session_id": session_id,
            "document_id": document_id,
            "chunks_stored": len(chunks),
            "vectors_stored": len(vectors),
            "processing_time_ms": processing_time,
            "storage_provider": "flatfile",
            "embedding_provider": getattr(engine_config, 'embedding_provider', 'unknown'),
            "success": True
        }
        
        # Merge with any existing metadata
        input_metadata = input_data.get("metadata", {})
        result_metadata.update(input_metadata)
        
        return {
            "output_content": embeddings,  # Pass through unchanged for chaining
            "metadata": result_metadata,
            "success": True,
            "output_format": "embeddings_with_flatfile_storage"
        }
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "output_content": input_data.get("input_content", []),
            "metadata": {
                "handler_name": "ff_store_vectors_handler_async",
                "error": f"Exception in flatfile storage: {str(e)}",
                "processing_time_ms": processing_time,
                "success": False
            },
            "success": False
        }


# Declare input key for PrismMind chaining
ff_store_vectors_handler_async.__input_key__ = "input_content"


async def ff_document_metadata_handler_async(
    input_data: Dict[str, Any],
    engine_config: object,
    handler_config: object,
    rag_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optional handler to enrich document metadata before storage.
    
    This handler can be inserted in the chain to add additional metadata
    or perform document-level processing before vector storage.
    """
    try:
        document_content = input_data.get("input_content", "")
        
        # Add document-level metadata
        metadata = {
            "document_length": len(document_content) if isinstance(document_content, str) else 0,
            "processing_timestamp": datetime.now().isoformat(),
            "content_type": type(document_content).__name__,
            "handler_name": "ff_document_metadata_handler_async"
        }
        
        # Add word and sentence counts for text content
        if isinstance(document_content, str):
            metadata.update({
                "word_count": len(document_content.split()),
                "character_count": len(document_content),
                "estimated_reading_time_minutes": len(document_content.split()) / 200  # 200 WPM average
            })
        
        return {
            "output_content": document_content,
            "metadata": metadata,
            "success": True,
            "output_format": "enriched_document"
        }
        
    except Exception as e:
        return {
            "output_content": input_data.get("input_content", ""),
            "metadata": {
                "handler_name": "ff_document_metadata_handler_async",
                "error": f"Exception in metadata enrichment: {str(e)}",
                "success": False
            },
            "success": False
        }


# Declare input key for chaining
ff_document_metadata_handler_async.__input_key__ = "input_content"


async def ff_batch_storage_handler_async(
    input_data: Dict[str, Any],
    engine_config: object,
    handler_config: object,
    rag_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handler for batch processing multiple documents with optimized storage.
    
    Useful for processing multiple documents efficiently with batched
    vector storage operations.
    """
    try:
        batch_data = input_data.get("input_content", [])
        
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        
        stored_count = 0
        batch_metadata = []
        
        for item in batch_data:
            if isinstance(item, dict) and "embeddings" in item:
                # Process individual document in batch
                result = await ff_store_vectors_handler_async(
                    {"input_content": item["embeddings"]},
                    engine_config,
                    handler_config,
                    rag_data
                )
                
                if result.get("success", False):
                    stored_count += 1
                    batch_metadata.append(result.get("metadata", {}))
        
        return {
            "output_content": batch_data,
            "metadata": {
                "handler_name": "ff_batch_storage_handler_async",
                "documents_processed": len(batch_data),
                "documents_stored": stored_count,
                "batch_metadata": batch_metadata,
                "success": stored_count > 0
            },
            "success": stored_count > 0,
            "output_format": "batch_storage_results"
        }
        
    except Exception as e:
        return {
            "output_content": input_data.get("input_content", []),
            "metadata": {
                "handler_name": "ff_batch_storage_handler_async",
                "error": f"Exception in batch storage: {str(e)}",
                "success": False
            },
            "success": False
        }


# Declare input key for chaining
ff_batch_storage_handler_async.__input_key__ = "input_content"