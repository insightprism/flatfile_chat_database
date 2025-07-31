"""
Text Chunking Module for Flatfile Chat Database

Uses PrismMind's proven chunking engine instead of custom implementation.
Replaced 323 lines of custom logic with direct engine usage.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO

# PrismMind Engine Imports
try:
    from prismmind.pm_engines.pm_chunking_engine import PmChunkingEngine
    from prismmind.pm_config.pm_chunking_engine_config import pm_chunking_engine_config_dto
    PRISMMIND_AVAILABLE = True
except ImportError:
    PRISMMIND_AVAILABLE = False
    print("PrismMind not available. Install PrismMind for full functionality.")


@dataclass
class FFChunkingConfigDTO:
    """Configuration for a chunking strategy - kept for backward compatibility"""
    chunk_strategy: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    sentence_per_chunk: int = 5
    sentence_overlap: int = 1
    sentence_buffer: int = 2
    max_tokens_per_chunk: int = 800
    min_tokens_per_chunk: int = 128
    chunk_overlap_sentences: int = 1
    spacy_model_name: str = "en_core_web_sm"


class FFChunkingManager:
    """
    Text chunking using PrismMind's proven chunking engine.
    Simplified from 323 lines to ~50 lines of actual logic.
    """
    
    # Strategy mapping to PrismMind handlers
    STRATEGY_HANDLERS = {
        "optimized_summary": "pm_optimize_chunk_handler_async",
        "default_fixed": "pm_fixed_chunk_handler_async",
        "sentence_short": "pm_sentence_chunk_handler_async",
        "sentence_medium": "pm_sentence_chunk_handler_async",
        "large_context": "pm_optimize_chunk_handler_async",
        "optimize": "pm_optimize_chunk_handler_async",
        "fixed": "pm_fixed_chunk_handler_async",
        "sentence": "pm_sentence_chunk_handler_async"
    }
    
    # Legacy strategy configs (kept for validation/metadata)
    STRATEGIES = {
        "optimized_summary": FFChunkingConfigDTO(
            chunk_strategy="optimize",
            chunk_size=800,
            chunk_overlap=100,
            sentence_per_chunk=5,
            sentence_overlap=1,
            sentence_buffer=2,
            max_tokens_per_chunk=800,
            min_tokens_per_chunk=128,
            chunk_overlap_sentences=1
        ),
        "default_fixed": FFChunkingConfigDTO(
            chunk_strategy="fixed",
            chunk_size=512,
            chunk_overlap=64
        ),
        "sentence_short": FFChunkingConfigDTO(
            chunk_strategy="sentence",
            sentence_per_chunk=2,
            sentence_overlap=0
        ),
        "sentence_medium": FFChunkingConfigDTO(
            chunk_strategy="sentence",
            sentence_per_chunk=5,
            sentence_overlap=1
        ),
        "large_context": FFChunkingConfigDTO(
            chunk_strategy="optimize",
            chunk_size=1200,
            chunk_overlap=200,
            sentence_per_chunk=8,
            max_tokens_per_chunk=1200,
            min_tokens_per_chunk=200,
            chunk_overlap_sentences=2
        )
    }
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        self.config = config
        self.default_strategy = config.vector.default_chunking_strategy
    
    async def chunk_text(
        self, 
        text: str, 
        strategy: str = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Chunk text using PrismMind engine."""
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind is required for text chunking")
        
        strategy = strategy or self.default_strategy
        
        # Validate strategy
        if strategy not in self.STRATEGY_HANDLERS:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGY_HANDLERS.keys())}")
        
        # Get PrismMind handler name
        handler_name = self.STRATEGY_HANDLERS[strategy]
        
        # Create chunking engine
        engine = PmChunkingEngine(
            engine_config=pm_chunking_engine_config_dto(
                handler_name=handler_name
            )
        )
        
        try:
            # Process with PrismMind engine
            result = await engine(text)
            
            # PrismMind returns {"output_content": [...], "success": True, ...}
            if result.get("success", True):
                chunks = result["output_content"]
                # Ensure we return a list of strings
                if isinstance(chunks, list):
                    return [str(chunk) for chunk in chunks]
                else:
                    return [str(chunks)]
            else:
                raise RuntimeError(f"Chunking failed: {result.get('metadata', {}).get('error')}")
                
        except Exception as e:
            print(f"Error chunking text: {e}")
            # Fallback to simple splitting
            return await self._fallback_chunk(text, strategy)
    
    async def _fallback_chunk(self, text: str, strategy: str) -> List[str]:
        """Simple fallback chunking when PrismMind is unavailable."""
        strategy_config = self.STRATEGIES.get(strategy, self.STRATEGIES["default_fixed"])
        
        if strategy_config.chunk_strategy == "fixed":
            # Simple character-based chunking
            chunk_size = strategy_config.chunk_size
            overlap = strategy_config.chunk_overlap
            
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end].strip())
                if end == len(text):
                    break
                start = start + chunk_size - overlap
            
            return [chunk for chunk in chunks if chunk]
        else:
            # Fallback to simple sentence splitting
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Use configured fallback chunk size limit
                limit = self.config.fallback_chunk_size_limit
                if len(current_chunk) + len(sentence) > limit:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
    
    async def chunk_with_metadata(
        self, 
        text: str, 
        strategy: str = None,
        document_id: str = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text and return with metadata."""
        chunks = await self.chunk_text(text, strategy, custom_config)
        
        strategy = strategy or self.default_strategy
        metadata = {
            "strategy": strategy,
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunking_engine": "prismmind"
        }
        
        return [
            {
                "chunk_id": f"{document_id}_{i}" if document_id else f"chunk_{i}",
                "chunk_text": chunk,
                "chunk_index": i,
                "metadata": {**metadata, "chunk_index": i}
            }
            for i, chunk in enumerate(chunks)
        ]
    
    async def estimate_chunks(self, text: str, strategy: str = None) -> int:
        """Estimate number of chunks for given text and strategy."""
        strategy = strategy or self.default_strategy
        strategy_config = self.STRATEGIES.get(strategy, self.STRATEGIES["default_fixed"])
        
        if strategy_config.chunk_strategy == "fixed":
            return max(1, len(text) // strategy_config.chunk_size)
        else:
            # Rough estimate based on sentences
            sentence_count = len(text.split('. '))
            sentences_per_chunk = strategy_config.sentence_per_chunk
            return max(1, sentence_count // sentences_per_chunk)


# Simplified convenience functions
async def ff_chunk_text(text: str, strategy: str = "optimized_summary") -> List[str]:
    """Chunk text using PrismMind engine - simplified interface."""
    if not PRISMMIND_AVAILABLE:
        raise RuntimeError("PrismMind is required for text chunking")
    
    # Strategy to handler mapping
    handler_map = {
        "optimized_summary": "pm_optimize_chunk_handler_async",
        "optimize": "pm_optimize_chunk_handler_async",
        "fixed": "pm_fixed_chunk_handler_async",
        "sentence": "pm_sentence_chunk_handler_async"
    }
    
    handler_name = handler_map.get(strategy, "pm_optimize_chunk_handler_async")
    
    engine = PmChunkingEngine(
        engine_config=pm_chunking_engine_config_dto(
            handler_name=handler_name
        )
    )
    
    result = await engine(text)
    if result.get("success", True):
        chunks = result["output_content"]
        return [str(chunk) for chunk in chunks] if isinstance(chunks, list) else [str(chunks)]
    else:
        return [text]  # Fallback to original text