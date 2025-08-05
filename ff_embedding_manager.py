"""
Embedding Generation Module for Flatfile Chat Database

Generate embeddings using PrismMind's proven embedding engine.
Replaced 291 lines of custom wrapper code with direct engine usage.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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


@dataclass
class FFEmbeddingProviderDTO:
    """Configuration for an embedding provider"""
    model_name: str
    embedding_dimension: int
    requires_api_key: bool
    normalize_vectors: bool = True


class FFEmbeddingManager:
    """
    Generate embeddings using PrismMind's proven embedding engine.
    Simplified from 291 lines to ~35 lines of actual logic.
    """
    
    # Provider configurations (used for validation and metadata)
    PROVIDERS = {
        "nomic-ai": FFEmbeddingProviderDTO(
            model_name="nomic-ai/nomic-embed-text-v1",
            embedding_dimension=768,
            requires_api_key=False,
            normalize_vectors=True
        ),
        "openai": FFEmbeddingProviderDTO(
            model_name="text-embedding-ada-002",
            embedding_dimension=1536,
            requires_api_key=True,
            normalize_vectors=True
        ),
        "openai-3-small": FFEmbeddingProviderDTO(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            requires_api_key=True,
            normalize_vectors=True
        ),
        "openai-3-large": FFEmbeddingProviderDTO(
            model_name="text-embedding-3-large",
            embedding_dimension=3072,
            requires_api_key=True,
            normalize_vectors=True
        )
    }
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        self.config = config
        self.default_provider = config.vector.default_embedding_provider
    
    async def ff_generate_embeddings(
        self,
        texts: List[str],
        provider: str = None,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate embeddings using PrismMind engine."""
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind is required for embedding generation")
        
        provider = provider or self.default_provider
        
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.PROVIDERS.keys())}")
        
        provider_config = self.PROVIDERS[provider]
        
        # Validate API key if required
        if provider_config.requires_api_key and not api_key:
            raise ValueError(f"API key required for provider: {provider}")
        
        # Create PrismMind embedding engine
        engine = PmEmbeddingEngine(
            engine_config=pm_embedding_engine_config_dto(
                embedding_provider=provider,
                embedding_model_name=provider_config.model_name,
                handler_name="pm_embed_batch_handler_async",
                api_key=api_key
            )
        )
        
        try:
            # Process with PrismMind engine
            result = await engine(texts)
            
            # PrismMind returns {"output_content": [...], "success": True, ...}
            if result.get("success", True):
                return result["output_content"]
            else:
                raise RuntimeError(f"Embedding generation failed: {result.get('metadata', {}).get('error')}")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            # Return empty embeddings as fallback
            return [{
                "embedding_vector": [0.0] * provider_config.embedding_dimension,
                "input_content": text,
                "metadata": {
                    "provider": provider,
                    "model": provider_config.model_name,
                    "dimensions": provider_config.embedding_dimension,
                    "error": str(e)
                }
            } for text in texts]
    
    async def ff_generate_single_embedding(
        self,
        text: str,
        provider: str = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        results = await self.ff_generate_embeddings([text], provider, api_key)
        return results[0] if results else None
    
    def ff_get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """Get information about an embedding provider."""
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
    
    def ff_list_providers(self) -> List[Dict[str, Any]]:
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


# Simplified convenience functions
async def ff_generate_embeddings(texts: List[str], provider: str = "nomic-ai", api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate embeddings using PrismMind engine - simplified interface."""
    if not PRISMMIND_AVAILABLE:
        raise RuntimeError("PrismMind is required for embedding generation")
    
    engine = PmEmbeddingEngine(
        engine_config=pm_embedding_engine_config_dto(
            embedding_provider=provider,
            handler_name="pm_embed_batch_handler_async",
            api_key=api_key
        )
    )
    
    result = await engine(texts)
    return result["output_content"] if result.get("success", True) else []


async def ff_generate_single_embedding(text: str, provider: str = "nomic-ai", api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate single embedding using PrismMind engine."""
    results = await ff_generate_embeddings([text], provider, api_key)
    return results[0] if results else None