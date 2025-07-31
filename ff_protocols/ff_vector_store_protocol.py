"""
Vector store protocol interface.

Defines the contract for vector storage and similarity search operations.
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
import numpy as np


class FFVectorStoreProtocol(Protocol):
    """
    Vector storage interface.
    
    All vector storage implementations must follow this protocol.
    """
    
    async def initialize(self) -> bool:
        """
        Initialize the vector store.
        
        Returns:
            True if successful
        """
        ...
    
    async def store_embedding(self, doc_id: str, embedding: List[float],
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a single embedding.
        
        Args:
            doc_id: Document identifier
            embedding: Embedding vector
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        ...
    
    async def store_embeddings_batch(self, embeddings: List[Tuple[str, List[float], Dict[str, Any]]]) -> Dict[str, bool]:
        """
        Store multiple embeddings in batch.
        
        Args:
            embeddings: List of (doc_id, embedding, metadata) tuples
            
        Returns:
            Dictionary mapping doc_ids to success status
        """
        ...
    
    async def get_embedding(self, doc_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Retrieve an embedding by document ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        ...
    
    async def delete_embedding(self, doc_id: str) -> bool:
        """
        Delete an embedding.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful
        """
        ...
    
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 5,
                           threshold: float = 0.0,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find similar embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (doc_id, similarity_score, metadata) tuples
        """
        ...
    
    async def search_by_metadata(self, metadata_filter: Dict[str, Any],
                               limit: int = 100) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search embeddings by metadata.
        
        Args:
            metadata_filter: Metadata to match
            limit: Maximum results
            
        Returns:
            List of (doc_id, metadata) tuples
        """
        ...
    
    async def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an embedding.
        
        Args:
            doc_id: Document identifier
            metadata: New metadata
            
        Returns:
            True if successful
        """
        ...
    
    async def get_all_ids(self, offset: int = 0, limit: int = 1000) -> List[str]:
        """
        Get all document IDs in the store.
        
        Args:
            offset: Skip this many IDs
            limit: Maximum IDs to return
            
        Returns:
            List of document IDs
        """
        ...
    
    async def count(self) -> int:
        """
        Get total number of embeddings.
        
        Returns:
            Number of stored embeddings
        """
        ...
    
    async def clear(self) -> bool:
        """
        Clear all embeddings from store.
        
        Returns:
            True if successful
        """
        ...
    
    async def optimize(self) -> bool:
        """
        Optimize the vector index for better performance.
        
        Returns:
            True if successful
        """
        ...
    
    async def rebuild_index(self) -> bool:
        """
        Rebuild the vector index.
        
        Returns:
            True if successful
        """
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary including:
            - total_embeddings: Total number of embeddings
            - index_size_bytes: Size of index in bytes
            - embedding_dimension: Dimension of embeddings
            - index_type: Type of index being used
        """
        ...
    
    async def export_embeddings(self, output_path: str, 
                              format: str = "numpy") -> bool:
        """
        Export all embeddings to file.
        
        Args:
            output_path: Path to save embeddings
            format: Export format ("numpy", "json", "parquet")
            
        Returns:
            True if successful
        """
        ...
    
    async def import_embeddings(self, input_path: str,
                              format: str = "numpy",
                              clear_existing: bool = False) -> int:
        """
        Import embeddings from file.
        
        Args:
            input_path: Path to embeddings file
            format: Import format ("numpy", "json", "parquet")
            clear_existing: Clear existing embeddings before import
            
        Returns:
            Number of embeddings imported
        """
        ...