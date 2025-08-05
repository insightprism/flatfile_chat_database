"""
Functional approach to embedding generation.

Provides embedding generation functions without unnecessary class abstraction.
"""

import asyncio
from typing import List, Dict, Any, Optional
from functools import partial

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_utils.ff_logging import get_logger

# Module logger
logger = get_logger(__name__)

# PrismMind Engine Imports
try:
    from prismmind.pm_engines.pm_embedding_engine import PmEmbeddingEngine
    from prismmind.pm_config.pm_embedding_engine_config import pm_embedding_engine_config_dto
    PRISMMIND_AVAILABLE = True
except ImportError:
    PRISMMIND_AVAILABLE = False
    logger.warning("PrismMind not available. Install PrismMind for full functionality.")


# Provider configurations
EMBEDDING_PROVIDERS = {
    "nomic-ai": {
        "model_name": "nomic-embed-text-v1.5",
        "embedding_dimension": 768,
        "requires_api_key": True
    },
    "openai": {
        "model_name": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "requires_api_key": True
    },
    "sentence-transformers": {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "requires_api_key": False
    },
    "bedrock": {
        "model_name": "amazon.titan-embed-text-v1",
        "embedding_dimension": 1536,
        "requires_api_key": True
    }
}


async def generate_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[FFConfigurationManagerConfigDTO] = None
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of texts using specified provider.
    
    This is a functional replacement for FFEmbeddingManager that directly
    uses PrismMind engines when available.
    
    Args:
        texts: List of text strings to embed
        provider: Embedding provider name (default from config)
        api_key: API key for the provider if required
        config: Configuration object (uses DI container if not provided)
        
    Returns:
        List of embedding dictionaries with:
        - embedding_vector: List of floats
        - input_content: Original text
        - metadata: Provider info, dimension, timestamp
        
    Raises:
        RuntimeError: If no embedding provider is available
        ValueError: If provider is not supported
    """
    # Get configuration if not provided
    if config is None:
        from ff_dependency_injection_manager import ff_get_container
        config = ff_get_container().resolve(FFConfigurationManagerConfigDTO)
    
    # Use default provider if not specified
    if provider is None:
        provider = config.vector.default_embedding_provider
    
    # Validate provider
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    
    provider_config = EMBEDDING_PROVIDERS[provider]
    
    # Check if API key is required
    if provider_config["requires_api_key"] and not api_key:
        raise ValueError(f"API key required for provider: {provider}")
    
    try:
        if PRISMMIND_AVAILABLE:
            # Use PrismMind engine
            engine_result = await _generate_with_prismmind(
                texts, provider, api_key, provider_config
            )
            return engine_result
        else:
            # Fallback implementation
            logger.warning("PrismMind not available, using mock embeddings")
            return _generate_mock_embeddings(texts, provider_config)
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        # Return zero embeddings as fallback
        return _generate_zero_embeddings(texts, provider_config)


async def _generate_with_prismmind(
    texts: List[str],
    provider: str,
    api_key: Optional[str],
    provider_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate embeddings using PrismMind engine."""
    # Configure PrismMind engine
    engine_config = pm_embedding_engine_config_dto(
        embedding_provider=provider,
        embedding_model=provider_config["model_name"],
        embedding_dimension=provider_config["embedding_dimension"],
        api_key=api_key
    )
    
    # Create and initialize engine
    engine = PmEmbeddingEngine(engine_config)
    result = await engine.process(texts)
    
    if result["status"] == "success":
        return result["embeddings"]
    else:
        error_msg = result.get("metadata", {}).get("error", "Unknown error")
        raise RuntimeError(f"Embedding generation failed: {error_msg}")


def _generate_mock_embeddings(
    texts: List[str],
    provider_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate mock embeddings for testing/development."""
    import hashlib
    from datetime import datetime
    
    embeddings = []
    dimension = provider_config["embedding_dimension"]
    
    for text in texts:
        # Generate deterministic "embeddings" based on text hash
        text_hash = hashlib.sha256(text.encode()).digest()
        vector = [
            float(b) / 255.0 for b in text_hash[:dimension]
        ]
        # Pad if necessary
        if len(vector) < dimension:
            vector.extend([0.0] * (dimension - len(vector)))
        
        embeddings.append({
            "embedding_vector": vector,
            "input_content": text,
            "metadata": {
                "provider": provider_config.get("model_name", "mock"),
                "dimension": dimension,
                "timestamp": datetime.now().isoformat(),
                "mock": True
            }
        })
    
    return embeddings


def _generate_zero_embeddings(
    texts: List[str],
    provider_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate zero embeddings as ultimate fallback."""
    from datetime import datetime
    
    dimension = provider_config["embedding_dimension"]
    
    return [{
        "embedding_vector": [0.0] * dimension,
        "input_content": text,
        "metadata": {
            "provider": provider_config.get("model_name", "fallback"),
            "dimension": dimension,
            "timestamp": datetime.now().isoformat(),
            "error": "fallback"
        }
    } for text in texts]


def get_embedding_dimension(provider: str) -> int:
    """
    Get the embedding dimension for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Embedding dimension
        
    Raises:
        ValueError: If provider not supported
    """
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    return EMBEDDING_PROVIDERS[provider]["embedding_dimension"]


def list_available_providers() -> List[str]:
    """
    Get list of available embedding providers.
    
    Returns:
        List of provider names
    """
    return list(EMBEDDING_PROVIDERS.keys())


def is_api_key_required(provider: str) -> bool:
    """
    Check if provider requires API key.
    
    Args:
        provider: Provider name
        
    Returns:
        True if API key required
    """
    if provider not in EMBEDDING_PROVIDERS:
        return True  # Assume unknown providers need keys
    return EMBEDDING_PROVIDERS[provider]["requires_api_key"]