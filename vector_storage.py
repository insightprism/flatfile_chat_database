
"""
Vector Storage Module for Flatfile Chat Database

Manages vector storage using NumPy arrays and JSONL indices.
Provides efficient storage and retrieval of embedding vectors.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime

from flatfile_chat_database.utils.file_ops import atomic_write, ensure_directory
from flatfile_chat_database.utils.json_utils import read_jsonl, append_jsonl
from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.models import VectorSearchResult


class FlatfileVectorStorage:
    """
    Manages vector storage using NumPy arrays and JSONL indices.
    Provides efficient storage and retrieval of embedding vectors.
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.storage_base_path)
    
    async def store_vectors(
        self, 
        session_id: str, 
        document_id: str,
        chunks: List[str], 
        vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store embedding vectors and their metadata.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            chunks: Original text chunks
            vectors: Embedding vectors (must be same length as chunks)
            metadata: Optional metadata for the entire document
            
        Returns:
            True if successful
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")
        
        # Get vector storage path
        vector_dir = self._get_vector_path(session_id)
        await ensure_directory(vector_dir)
        
        # Load existing vectors if any
        embeddings_path = vector_dir / self.config.embeddings_filename
        index_path = vector_dir / self.config.vector_index_filename
        
        if embeddings_path.exists():
            existing_embeddings = np.load(embeddings_path)
            existing_index = await read_jsonl(index_path, self.config)
        else:
            existing_embeddings = np.array([]).reshape(0, len(vectors[0]))
            existing_index = []
        
        # Prepare new data
        new_embeddings = np.array(vectors)
        start_index = len(existing_embeddings)
        
        # Create index entries
        new_index_entries = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            entry = {
                "chunk_id": f"{document_id}_chunk_{i}",
                "vector_index": start_index + i,
                "document_id": document_id,
                "session_id": session_id,
                "chunk_text": chunk,
                "chunk_metadata": {
                    "position": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                },
                "embedding_metadata": {
                    "provider": metadata.get("provider", self.config.default_embedding_provider) if metadata else self.config.default_embedding_provider,
                    "model": metadata.get("model", "nomic-embed-text-v1.5") if metadata else "nomic-embed-text-v1.5",
                    "dimensions": len(vector),
                    "normalized": metadata.get("normalized", True) if metadata else True,
                    "timestamp": datetime.now().isoformat()
                }
            }
            new_index_entries.append(entry)
        
        # Append to embeddings array
        if existing_embeddings.size > 0:
            all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings
        
        # Save embeddings
        np.save(embeddings_path, all_embeddings)
        
        # Append to index
        for entry in new_index_entries:
            await append_jsonl(index_path, entry, self.config)
        
        return True
    
    async def load_vectors(
        self, 
        session_id: str
    ) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Load all vectors and metadata for a session.
        
        Returns:
            Tuple of (embeddings array, index entries)
        """
        vector_dir = self._get_vector_path(session_id)
        embeddings_path = vector_dir / self.config.embeddings_filename
        index_path = vector_dir / self.config.vector_index_filename
        
        if not embeddings_path.exists():
            return None, []
        
        embeddings = np.load(embeddings_path, mmap_mode='r')
        index = await read_jsonl(index_path, self.config)
        
        return embeddings, index
    
    async def search_similar(
        self, 
        session_id: str, 
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            session_id: Session to search in
            query_vector: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results sorted by similarity
        """
        embeddings, index = await self.load_vectors(session_id)
        
        if embeddings is None or len(index) == 0:
            return []
        
        # Normalize query vector
        query_norm = np.array(query_vector)
        query_norm = query_norm / np.linalg.norm(query_norm)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings, query_norm)
        
        # Get top-k indices above threshold
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []
        
        top_indices = valid_indices[np.argsort(-similarities[valid_indices])][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(index):
                entry = index[idx]
                result = VectorSearchResult(
                    chunk_id=entry["chunk_id"],
                    chunk_text=entry["chunk_text"],
                    similarity_score=float(similarities[idx]),
                    document_id=entry["document_id"],
                    session_id=entry["session_id"],
                    metadata=entry.get("chunk_metadata", {})
                )
                results.append(result)
        
        return results
    
    async def search_similar_across_sessions(
        self,
        session_ids: List[str],
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors across multiple sessions.
        
        Args:
            session_ids: List of sessions to search
            query_vector: Query embedding vector
            top_k: Total number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results sorted by similarity across all sessions
        """
        all_results = []
        
        # Search each session
        for session_id in session_ids:
            try:
                results = await self.search_similar(
                    session_id=session_id,
                    query_vector=query_vector,
                    top_k=top_k,  # Get top_k from each session
                    threshold=threshold
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error searching session {session_id}: {e}")
                continue
        
        # Sort all results by similarity and return top_k
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:top_k]
    
    async def delete_document_vectors(
        self, 
        session_id: str, 
        document_id: str
    ) -> bool:
        """Delete all vectors associated with a document."""
        embeddings, index = await self.load_vectors(session_id)
        
        if embeddings is None:
            return True
        
        # Find indices to keep
        keep_indices = []
        new_index = []
        
        for i, entry in enumerate(index):
            if entry["document_id"] != document_id:
                keep_indices.append(i)
                # Update vector index
                entry["vector_index"] = len(new_index)
                new_index.append(entry)
        
        if len(keep_indices) == len(index):
            return True  # Document not found
        
        # Save filtered data
        vector_dir = self._get_vector_path(session_id)
        embeddings_path = vector_dir / self.config.embeddings_filename
        index_path = vector_dir / self.config.vector_index_filename
        
        if len(keep_indices) > 0:
            new_embeddings = embeddings[keep_indices]
            np.save(embeddings_path, new_embeddings)
            
            # Rewrite index
            await atomic_write(index_path, "", self.config)  # Clear file
            for entry in new_index:
                await append_jsonl(index_path, entry, self.config)
        else:
            # No vectors left, remove files
            embeddings_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
        
        return True
    
    async def get_vector_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about vectors in a session."""
        embeddings, index = await self.load_vectors(session_id)
        
        if embeddings is None:
            return {
                "total_vectors": 0,
                "total_documents": 0,
                "vector_dimensions": 0,
                "storage_size_bytes": 0
            }
        
        # Get unique document IDs
        document_ids = set(entry["document_id"] for entry in index)
        
        # Calculate storage size
        vector_dir = self._get_vector_path(session_id)
        embeddings_path = vector_dir / self.config.embeddings_filename
        index_path = vector_dir / self.config.vector_index_filename
        
        storage_size = 0
        if embeddings_path.exists():
            storage_size += embeddings_path.stat().st_size
        if index_path.exists():
            storage_size += index_path.stat().st_size
        
        return {
            "total_vectors": len(index),
            "total_documents": len(document_ids),
            "vector_dimensions": embeddings.shape[1] if embeddings.size > 0 else 0,
            "storage_size_bytes": storage_size,
            "providers_used": list(set(entry.get("embedding_metadata", {}).get("provider", "unknown") for entry in index))
        }
    
    def _get_vector_path(self, session_id: str) -> Path:
        """Get the vector storage path for a session."""
        # We need to work with the session structure
        # Sessions are stored under users/{user_id}/sessions/{session_id}/
        # We'll need to find the user directory that contains this session
        
        # Search for the session in all user directories
        users_dir = self.base_path / self.config.user_data_directory_name
        if not users_dir.exists():
            users_dir.mkdir(parents=True, exist_ok=True)
        
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                session_path = user_dir / self.config.session_data_directory_name / session_id
                if session_path.exists():
                    return session_path / self.config.vector_storage_subdirectory
        
        # If session not found, create a default path
        # This handles new sessions
        default_user = "default"
        return (
            self.base_path / 
            self.config.user_data_directory_name /
            default_user /
            self.config.session_data_directory_name /
            session_id /
            self.config.vector_storage_subdirectory
        )