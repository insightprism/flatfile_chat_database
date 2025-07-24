"""
Embedding Generation Module for Flatfile Chat Database

Generate embeddings using multiple providers.
Based on PrismMind's embedding architecture.
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.utils.pm_get_vector import pm_get_vector


@dataclass
class EmbeddingProvider:
    """Configuration for an embedding provider"""
    model_name: str
    embedding_dimension: int
    requires_api_key: bool
    normalize_vectors: bool = True


class EmbeddingEngine:
    """
    Generate embeddings using multiple providers.
    Based on PrismMind's embedding architecture.
    """
    
    # Provider configurations
    PROVIDERS = {
        "nomic-ai": EmbeddingProvider(
            model_name="nomic-ai/nomic-embed-text-v1",
            embedding_dimension=768,
            requires_api_key=False,
            normalize_vectors=True
        ),
        "openai": EmbeddingProvider(
            model_name="text-embedding-ada-002",
            embedding_dimension=1536,
            requires_api_key=True,
            normalize_vectors=True
        ),
        "openai-3-small": EmbeddingProvider(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            requires_api_key=True,
            normalize_vectors=True
        ),
        "openai-3-large": EmbeddingProvider(
            model_name="text-embedding-3-large",
            embedding_dimension=3072,
            requires_api_key=True,
            normalize_vectors=True
        )
    }
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.default_provider = config.default_embedding_provider
    
    async def generate_embeddings(
        self,
        texts: List[str],
        provider: str = None,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            provider: Embedding provider (default: "nomic-ai")
            api_key: API key if required
            batch_size: Optional batch size for processing
            
        Returns:
            List of embedding dictionaries with vectors and metadata
        """
        provider = provider or self.default_provider
        batch_size = batch_size or self.config.vector_batch_size
        
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.PROVIDERS.keys())}")
        
        provider_config = self.PROVIDERS[provider]
        
        # Validate API key if required
        if provider_config.requires_api_key and not api_key:
            raise ValueError(f"API key required for provider: {provider}")
        
        # Process in batches if needed
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare embedding payload (PrismMind format)
            embedding_payload = {
                "embedding_provider": provider,
                "embedding_model_name": provider_config.model_name,
                "input_content": batch_texts,
                "api_key": api_key,
                "embedding_provider_url": self._get_provider_url(provider),
                "metadata": {
                    "batch_index": i // batch_size,
                    "batch_size": len(batch_texts)
                }
            }
            
            # Call PrismMind's vector generation
            try:
                results = await pm_get_vector(embedding_payload)
                
                # Ensure normalization if needed
                if provider_config.normalize_vectors:
                    for result in results:
                        vector = np.array(result["embedding_vector"])
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            result["embedding_vector"] = (vector / norm).tolist()
                
                # Add provider metadata
                for result in results:
                    result["metadata"] = {
                        "provider": provider,
                        "model": provider_config.model_name,
                        "dimensions": provider_config.embedding_dimension,
                        "normalized": provider_config.normalize_vectors,
                        **result.get("metadata", {})
                    }
                
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size}: {e}")
                # Add empty results for failed batch
                for text in batch_texts:
                    all_results.append({
                        "embedding_vector": [0.0] * provider_config.embedding_dimension,
                        "input_content": text,
                        "metadata": {
                            "provider": provider,
                            "model": provider_config.model_name,
                            "dimensions": provider_config.embedding_dimension,
                            "error": str(e)
                        }
                    })
        
        return all_results
    
    async def generate_single_embedding(
        self,
        text: str,
        provider: str = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            provider: Embedding provider
            api_key: API key if required
            
        Returns:
            Embedding dictionary with vector and metadata
        """
        results = await self.generate_embeddings([text], provider, api_key)
        return results[0] if results else None
    
    def get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """
        Get information about an embedding provider.
        
        Args:
            provider: Provider name (default: configured default)
            
        Returns:
            Dictionary with provider information
        """
        provider = provider or self.default_provider
        
        if provider not in self.PROVIDERS:
            return {
                "error": f"Unknown provider: {provider}",
                "available_providers": list(self.PROVIDERS.keys())
            }
        
        config = self.PROVIDERS[provider]
        return {
            "provider": provider,
            "model_name": config.model_name,
            "embedding_dimension": config.embedding_dimension,
            "requires_api_key": config.requires_api_key,
            "normalize_vectors": config.normalize_vectors
        }
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available embedding providers."""
        providers = []
        for name, config in self.PROVIDERS.items():
            providers.append({
                "name": name,
                "model": config.model_name,
                "dimensions": config.embedding_dimension,
                "requires_api_key": config.requires_api_key,
                "is_default": name == self.default_provider
            })
        return providers
    
    def _get_provider_url(self, provider: str) -> str:
        """Get the API URL for a provider."""
        urls = {
            "openai": "https://api.openai.com/v1/embeddings",
            "openai-3-small": "https://api.openai.com/v1/embeddings",
            "openai-3-large": "https://api.openai.com/v1/embeddings",
            "nomic-ai": ""  # Local, no URL needed
        }
        return urls.get(provider, "")
    
    async def validate_embeddings(
        self,
        embeddings: List[List[float]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        Validate embeddings match expected dimensions for provider.
        
        Args:
            embeddings: List of embedding vectors
            provider: Provider name to validate against
            
        Returns:
            Validation result dictionary
        """
        provider = provider or self.default_provider
        
        if provider not in self.PROVIDERS:
            return {
                "valid": False,
                "error": f"Unknown provider: {provider}"
            }
        
        expected_dim = self.PROVIDERS[provider].embedding_dimension
        
        issues = []
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                issues.append({
                    "index": i,
                    "expected_dimensions": expected_dim,
                    "actual_dimensions": len(embedding)
                })
        
        return {
            "valid": len(issues) == 0,
            "total_embeddings": len(embeddings),
            "expected_dimensions": expected_dim,
            "issues": issues
        }
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Ensure result is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))