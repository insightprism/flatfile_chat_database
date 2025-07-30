"""
Vector storage and embedding configuration.

Manages vector search, embeddings, and chunking strategies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .base import BaseConfig, validate_positive, validate_range, validate_non_empty


@dataclass
class VectorConfig(BaseConfig):
    """
    Vector storage and embedding configuration.
    
    Controls vector operations, embedding providers, and chunking strategies.
    """
    
    # Storage settings
    storage_subdirectory: str = "vectors"
    index_filename: str = "vector_index.jsonl"
    embeddings_filename: str = "embeddings.npy"
    metadata_filename: str = "vector_metadata.json"
    
    # Vector search parameters
    search_top_k: int = 5
    similarity_threshold: float = 0.7
    hybrid_search_weight: float = 0.5  # Balance between text and vector search
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # Performance settings
    batch_size: int = 32
    cache_enabled: bool = True
    cache_max_size: int = 10000
    mmap_mode: str = "r"  # Memory-mapped file mode
    use_gpu: bool = False
    
    # Default embedding provider
    default_embedding_provider: str = "nomic-ai"
    
    # Embedding provider configurations
    embedding_providers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "nomic-ai": {
            "model_name": "nomic-ai/nomic-embed-text-v1",
            "embedding_dimension": 768,
            "requires_api_key": False,
            "normalize_vectors": True,
            "max_tokens": 8192
        },
        "openai": {
            "model_name": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "requires_api_key": True,
            "normalize_vectors": True,
            "api_url": "https://api.openai.com/v1/embeddings",
            "max_tokens": 8191
        },
        "sentence-transformers": {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "requires_api_key": False,
            "normalize_vectors": True,
            "max_tokens": 512
        }
    })
    
    # Default chunking strategy
    default_chunking_strategy: str = "optimized_summary"
    
    # Chunking strategies
    chunking_strategies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "optimized_summary": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "sentence_per_chunk": 5,
            "sentence_overlap": 1,
            "sentence_buffer": 2,
            "max_tokens_per_chunk": 800,
            "min_tokens_per_chunk": 128,
            "preserve_sentences": True
        },
        "fixed_size": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "preserve_words": True
        },
        "sentence_based": {
            "sentence_per_chunk": 3,
            "sentence_overlap": 1,
            "min_chunk_size": 50,
            "max_chunk_size": 1000
        },
        "semantic": {
            "target_chunk_size": 500,
            "similarity_threshold": 0.8,
            "min_chunk_size": 100,
            "max_chunk_size": 1000
        }
    })
    
    # SpaCy model settings
    spacy_model_name: str = "en_core_web_sm"
    download_spacy_model: bool = True
    
    # Index settings
    index_type: str = "flat"  # flat, hnsw, ivf
    index_rebuild_threshold: int = 10000  # Rebuild index after this many additions
    
    def validate(self) -> List[str]:
        """
        Validate vector configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate storage settings
        if error := validate_non_empty(self.storage_subdirectory, "storage_subdirectory"):
            errors.append(error)
        if error := validate_non_empty(self.index_filename, "index_filename"):
            errors.append(error)
        if error := validate_non_empty(self.embeddings_filename, "embeddings_filename"):
            errors.append(error)
        
        # Validate search parameters
        if error := validate_positive(self.search_top_k, "search_top_k"):
            errors.append(error)
        if error := validate_range(self.similarity_threshold, 0.0, 1.0, "similarity_threshold"):
            errors.append(error)
        if error := validate_range(self.hybrid_search_weight, 0.0, 1.0, "hybrid_search_weight"):
            errors.append(error)
        
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if self.distance_metric not in valid_metrics:
            errors.append(f"distance_metric must be one of {valid_metrics}, got {self.distance_metric}")
        
        # Validate performance settings
        if error := validate_positive(self.batch_size, "batch_size"):
            errors.append(error)
        if error := validate_positive(self.cache_max_size, "cache_max_size"):
            errors.append(error)
        
        # Validate mmap mode
        valid_mmap_modes = ["r", "r+", "w+", "c"]
        if self.mmap_mode not in valid_mmap_modes:
            errors.append(f"mmap_mode must be one of {valid_mmap_modes}, got {self.mmap_mode}")
        
        # Validate embedding provider
        if self.default_embedding_provider not in self.embedding_providers:
            errors.append(f"default_embedding_provider '{self.default_embedding_provider}' not found in embedding_providers")
        
        # Validate chunking strategy
        if self.default_chunking_strategy not in self.chunking_strategies:
            errors.append(f"default_chunking_strategy '{self.default_chunking_strategy}' not found in chunking_strategies")
        
        # Validate index settings
        valid_index_types = ["flat", "hnsw", "ivf"]
        if self.index_type not in valid_index_types:
            errors.append(f"index_type must be one of {valid_index_types}, got {self.index_type}")
        
        if error := validate_positive(self.index_rebuild_threshold, "index_rebuild_threshold"):
            errors.append(error)
        
        return errors
    
    def get_embedding_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for an embedding provider.
        
        Args:
            provider: Provider name (uses default if None)
            
        Returns:
            Provider configuration dictionary
        """
        provider = provider or self.default_embedding_provider
        return self.embedding_providers.get(provider, {})
    
    def get_chunking_config(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a chunking strategy.
        
        Args:
            strategy: Strategy name (uses default if None)
            
        Returns:
            Strategy configuration dictionary
        """
        strategy = strategy or self.default_chunking_strategy
        return self.chunking_strategies.get(strategy, {})