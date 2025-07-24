"""
Text Chunking Module for Flatfile Chat Database

Provides multiple text chunking strategies based on PrismMind patterns.
Supports fixed-size, sentence-based, and optimized chunking.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import spacy
import re

from flatfile_chat_database.config import StorageConfig


# SpaCy model cache
_spacy_model_cache = {}


@dataclass
class ChunkingStrategy:
    """Configuration for a chunking strategy"""
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


class ChunkingEngine:
    """
    Text chunking engine with multiple strategies.
    Based on PrismMind's proven chunking patterns.
    """
    
    # Default strategies matching PrismMind
    STRATEGIES = {
        "optimized_summary": ChunkingStrategy(
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
        "default_fixed": ChunkingStrategy(
            chunk_strategy="fixed",
            chunk_size=512,
            chunk_overlap=64
        ),
        "sentence_short": ChunkingStrategy(
            chunk_strategy="sentence",
            sentence_per_chunk=2,
            sentence_overlap=0
        ),
        "sentence_medium": ChunkingStrategy(
            chunk_strategy="sentence",
            sentence_per_chunk=5,
            sentence_overlap=1
        ),
        "large_context": ChunkingStrategy(
            chunk_strategy="optimize",
            chunk_size=1200,
            chunk_overlap=200,
            sentence_per_chunk=8,
            max_tokens_per_chunk=1200,
            min_tokens_per_chunk=200,
            chunk_overlap_sentences=2
        )
    }
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.default_strategy = config.default_chunking_strategy
    
    async def chunk_text(
        self, 
        text: str, 
        strategy: str = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Strategy name (default: "optimized_summary")
            custom_config: Optional custom configuration
            
        Returns:
            List of text chunks
        """
        strategy = strategy or self.default_strategy
        
        if strategy not in self.STRATEGIES and not custom_config:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategy_config = custom_config or self.STRATEGIES[strategy]
        
        if isinstance(strategy_config, dict):
            strategy_config = ChunkingStrategy(**strategy_config)
        
        # Clean text first
        text = self._clean_text(text)
        
        # Route to appropriate handler
        if strategy_config.chunk_strategy == "fixed":
            return await self._fixed_chunk(text, strategy_config)
        elif strategy_config.chunk_strategy == "sentence":
            return await self._sentence_chunk(text, strategy_config)
        elif strategy_config.chunk_strategy == "optimize":
            return await self._optimized_chunk(text, strategy_config)
        else:
            raise ValueError(f"Unknown chunk strategy: {strategy_config.chunk_strategy}")
    
    async def _fixed_chunk(
        self, 
        text: str, 
        config: ChunkingStrategy
    ) -> List[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + config.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            
            if end == len(text):
                break
                
            start = start + config.chunk_size - config.chunk_overlap
        
        return [chunk for chunk in chunks if chunk]  # Filter empty chunks
    
    async def _sentence_chunk(
        self, 
        text: str, 
        config: ChunkingStrategy
    ) -> List[str]:
        """Sentence-based chunking using SpaCy."""
        nlp = await self._get_spacy_model(config.spacy_model_name)
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            # Get the chunk sentences
            end_idx = min(i + config.sentence_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk = " ".join(chunk_sentences)
            
            if chunk:
                chunks.append(chunk)
            
            # Move forward with overlap
            i += config.sentence_per_chunk - config.sentence_overlap
            
            # Ensure we don't get stuck in a loop
            if i <= 0 and len(chunks) > 0:
                i = config.sentence_per_chunk
        
        return chunks
    
    async def _optimized_chunk(
        self, 
        text: str, 
        config: ChunkingStrategy
    ) -> List[str]:
        """
        Optimized chunking using sentence boundaries and token counts.
        This is the default strategy for best semantic coherence.
        """
        nlp = await self._get_spacy_model(config.spacy_model_name)
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            # Start with target number of sentences
            sentence_count = config.sentence_per_chunk
            candidate = sentences[i:i + sentence_count]
            chunk = " ".join(candidate)
            token_count = len(chunk.split())
            
            # Expand if below minimum
            while (token_count < config.min_tokens_per_chunk and 
                   (i + sentence_count) < len(sentences)):
                sentence_count += 1
                candidate = sentences[i:i + sentence_count]
                chunk = " ".join(candidate)
                token_count = len(chunk.split())
            
            # Truncate if above maximum
            if token_count > config.max_tokens_per_chunk:
                words = chunk.split()[:config.max_tokens_per_chunk]
                chunk = " ".join(words)
            
            if chunk:
                chunks.append(chunk)
            
            # Move forward with overlap
            advance = max(1, sentence_count - config.chunk_overlap_sentences)
            i += advance
        
        return chunks
    
    async def _get_spacy_model(self, model_name: str = "en_core_web_sm"):
        """Get or load SpaCy model with caching."""
        if model_name not in _spacy_model_cache:
            try:
                _spacy_model_cache[model_name] = spacy.load(model_name)
            except OSError:
                # Model not installed, use a simple fallback
                print(f"SpaCy model '{model_name}' not found. Using simple sentence splitting.")
                return None
        return _spacy_model_cache[model_name]
    
    def _clean_text(self, text: str) -> str:
        """Clean text before chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    async def chunk_with_metadata(
        self,
        text: str,
        strategy: str = None,
        document_id: str = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and return chunks with metadata.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = await self.chunk_text(text, strategy, custom_config)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "document_id": document_id,
                "strategy": strategy or self.default_strategy,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "position": {
                    "start": text.find(chunk),
                    "end": text.find(chunk) + len(chunk)
                }
            }
            result.append(chunk_metadata)
        
        return result
    
    async def estimate_chunks(
        self,
        text: str,
        strategy: str = None
    ) -> int:
        """
        Estimate the number of chunks without actually chunking.
        Useful for planning and progress indication.
        """
        strategy = strategy or self.default_strategy
        strategy_config = self.STRATEGIES.get(strategy)
        
        if not strategy_config:
            return 0
        
        if strategy_config.chunk_strategy == "fixed":
            # Simple calculation for fixed chunks
            chunk_size = strategy_config.chunk_size
            overlap = strategy_config.chunk_overlap
            text_len = len(text)
            
            if text_len <= chunk_size:
                return 1
            
            # Account for overlap
            effective_chunk_size = chunk_size - overlap
            return ((text_len - chunk_size) // effective_chunk_size) + 1
        
        elif strategy_config.chunk_strategy in ["sentence", "optimize"]:
            # Rough estimate based on average sentence length
            # Assuming average sentence is ~20 words, ~100 characters
            estimated_sentences = len(text) // 100
            if estimated_sentences == 0:
                return 1
            
            sentences_per_chunk = strategy_config.sentence_per_chunk
            overlap = strategy_config.sentence_overlap if hasattr(strategy_config, 'sentence_overlap') else 0
            
            if overlap >= sentences_per_chunk:
                overlap = sentences_per_chunk - 1
            
            effective_sentences_per_chunk = sentences_per_chunk - overlap
            return max(1, (estimated_sentences // effective_sentences_per_chunk))
        
        return 1  # Default to at least one chunk