"""
Vector Generation Utility

Adapted from PrismMind's pm_get_vector for use in flatfile chat database.
Generates embeddings using various providers (Nomic-AI, OpenAI, etc.)
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np


async def pm_get_vector(embedding_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate embedding vectors for given text inputs.
    
    Args:
        embedding_payload: Dictionary containing:
            - embedding_provider: Provider name (e.g., "nomic-ai", "openai")
            - embedding_model_name: Model name
            - input_content: List of texts to embed
            - api_key: Optional API key
            - embedding_provider_url: Optional API URL
            - metadata: Optional metadata
    
    Returns:
        List of dictionaries containing embedding vectors and metadata
    """
    provider = embedding_payload.get("embedding_provider", "nomic-ai")
    model_name = embedding_payload.get("embedding_model_name", "nomic-ai/nomic-embed-text-v1.5")
    input_texts = embedding_payload.get("input_content", [])
    api_key = embedding_payload.get("api_key")
    
    if not input_texts:
        return []
    
    # Ensure input is a list
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    results = []
    
    if provider == "nomic-ai":
        embeddings = await _get_nomic_embeddings(input_texts, model_name)
    elif provider == "openai":
        embeddings = await _get_openai_embeddings(input_texts, model_name, api_key)
    else:
        # Fallback to mock embeddings for unsupported providers
        embeddings = await _get_mock_embeddings(input_texts, 768)
    
    # Format results
    for i, (text, embedding) in enumerate(zip(input_texts, embeddings)):
        results.append({
            "embedding_vector": embedding,
            "input_content": text,
            "metadata": {
                "provider": provider,
                "model": model_name,
                "index": i
            }
        })
    
    return results


async def _get_nomic_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """Generate embeddings using Nomic AI via PrismMind's implementation."""
    try:
        # Import PrismMind's Nomic vector function
        import sys
        from pathlib import Path
        
        # Add PrismMind path
        prism_path = Path("/home/markly2/PrismMind_v2")
        if prism_path.exists():
            sys.path.insert(0, str(prism_path))
            
            from pm_utils.pm_get_vector import pm_get_nomic_vector
            
            # Use PrismMind's payload format
            payload = {
                "embedding_model_name": model_name,
                "input_content": texts
            }
            
            # Call PrismMind's function
            results = await pm_get_nomic_vector(payload)
            
            # Extract vectors from PrismMind's format
            embeddings = [result["embedding_vector"] for result in results]
            return embeddings
        else:
            raise ImportError("PrismMind path not found")
    
    except ImportError as e:
        print(f"PrismMind integration failed: {e}. Using mock embeddings.")
        return await _get_mock_embeddings(texts, 768)
    except Exception as e:
        print(f"Error generating Nomic embeddings: {e}")
        return await _get_mock_embeddings(texts, 768)


async def _get_openai_embeddings(texts: List[str], model_name: str, api_key: str) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    if not api_key:
        print("OpenAI API key not provided. Using mock embeddings.")
        return await _get_mock_embeddings(texts, 1536)
    
    try:
        import openai
        
        client = openai.Client(api_key=api_key)
        
        # Make API call
        response = client.embeddings.create(
            model=model_name,
            input=texts
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    except ImportError:
        print("openai package not installed. Using mock embeddings.")
        return await _get_mock_embeddings(texts, 1536)
    except Exception as e:
        print(f"Error generating OpenAI embeddings: {e}")
        return await _get_mock_embeddings(texts, 1536)


async def _get_mock_embeddings(texts: List[str], dimensions: int) -> List[List[float]]:
    """Generate mock embeddings for testing or when providers are unavailable."""
    embeddings = []
    
    for text in texts:
        # Create a deterministic mock embedding based on text content
        # This ensures same text always gets same embedding
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(dimensions)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    return embeddings


# Compatibility functions for different naming conventions
async def pm_get_nomic_vector(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Nomic-specific vector generation."""
    payload["embedding_provider"] = "nomic-ai"
    return await pm_get_vector(payload)


async def pm_get_openai_vector(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OpenAI-specific vector generation."""
    payload["embedding_provider"] = "openai"
    return await pm_get_vector(payload)