# Implementation Context for Vector Support

## Critical Context Summary

This document captures essential context needed to implement vector support in the flatfile chat database based on PrismMind architecture.

## Key Design Decisions

### 1. Vector Storage Approach
- Store vectors in **NumPy .npy files** (binary format, efficient)
- Use **JSONL for metadata** (vector_index.jsonl)
- **Per-session storage** in `sessions/{session_id}/vectors/` directory
- This replaces ChromaDB while maintaining the same functionality

### 2. Default Configurations (CRITICAL)
- **Embedding Provider**: `nomic-ai` (local, no API key required)
- **Chunking Strategy**: `optimized_summary` (best semantic coherence)
- **Vector Dimensions**: 768 (Nomic-AI default)
- **Chunk Size**: 800 tokens (optimized_summary default)

### 3. PrismMind Integration Points

#### Key Files to Reference:
```python
# Chunking configuration and strategies
/home/markly2/PrismMind_v2/pm_config/pm_chunking_engine_config.py
# Contains PM_CHUNKING_STRATEGY_MAP with optimized_summary definition

# Chunking engine implementation
/home/markly2/PrismMind_v2/pm_engines/pm_chunking_engine.py
# Contains pm_optimize_chunk_handler_async function

# Embedding configuration
/home/markly2/PrismMind_v2/pm_config/pm_embedding_engine_config.py
# Contains PM_EMBEDDING_DEFAULT_CONFIG_MAP

# Embedding engine implementation
/home/markly2/PrismMind_v2/pm_engines/pm_embedding_engine.py
# Contains pm_embed_batch_handler_async function

# Vector generation utility
/home/markly2/PrismMind_v2/pm_utils/pm_get_vector.py
# Contains pm_get_vector function for embedding generation
```

### 4. Integration Architecture

```
StorageManager (storage.py)
    ├── FlatfileVectorStorage (NEW: vector_storage.py)
    ├── ChunkingEngine (NEW: chunking.py) 
    ├── EmbeddingEngine (NEW: embedding.py)
    └── AdvancedSearchEngine (ENHANCE: search.py)
```

### 5. Implementation Order

1. **Create base modules first**:
   - `vector_storage.py` - Core vector storage functionality
   - `chunking.py` - Text chunking with strategies
   - `embedding.py` - Embedding generation

2. **Integrate with existing system**:
   - Update `storage.py` - Add vector methods to StorageManager
   - Update `search.py` - Add vector search to AdvancedSearchEngine
   - Update `config.py` - Add vector configuration fields

3. **Create pipeline**:
   - `document_pipeline.py` - Automated processing

### 6. Key Implementation Details

#### Vector Index Entry Format:
```json
{
    "chunk_id": "doc123_chunk_0",
    "vector_index": 0,
    "document_id": "doc123",
    "session_id": "sess_001",
    "chunk_text": "Original chunk text...",
    "chunk_metadata": {
        "strategy": "optimized_summary",
        "position": {"start": 0, "end": 800}
    },
    "embedding_metadata": {
        "provider": "nomic-ai",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "normalized": true
    }
}
```

#### Similarity Search Algorithm:
```python
# 1. Load embeddings as memory-mapped array
embeddings = np.load(embeddings_path, mmap_mode='r')

# 2. Normalize query vector
query_norm = query_vector / np.linalg.norm(query_vector)

# 3. Compute cosine similarities
similarities = np.dot(embeddings, query_norm)

# 4. Get top-k results
top_indices = np.argpartition(similarities, -top_k)[-top_k:]
```

### 7. Configuration to Add to config.py

```python
# Vector storage
vector_storage_subdirectory: str = "vectors"
vector_index_filename: str = "vector_index.jsonl"
embeddings_filename: str = "embeddings.npy"

# Defaults
default_chunking_strategy: str = "optimized_summary"
default_embedding_provider: str = "nomic-ai"

# Search
vector_search_top_k: int = 5
similarity_threshold: float = 0.7
```

### 8. Dependencies to Install

```bash
pip install numpy sentence-transformers torch spacy
python -m spacy download en_core_web_sm
```

### 9. Testing Strategy

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test StorageManager vector methods
3. **Comparison Tests**: Compare with ChromaDB results
4. **Performance Tests**: Measure search speed with different vector counts

### 10. Migration Considerations

- The system should be **drop-in compatible** with existing ChromaDB usage
- Provide a compatibility layer if needed
- Same API surface for easy migration

## Important Notes

1. **Use async/await throughout** - Matches existing codebase pattern
2. **Follow existing error handling** - Use try/except with specific error types
3. **Maintain type hints** - All functions should have complete type annotations
4. **Use dataclasses** - For all data structures (VectorSearchResult, etc.)
5. **Follow naming conventions** - Lowercase with underscores for files/functions

## Quick Reference for Implementation

When implementing:
1. Start with `vector_storage.py` using the NumPy storage pattern
2. Copy chunking logic from PrismMind's `pm_optimize_chunk_handler_async`
3. Use PrismMind's `pm_get_vector` for embedding generation
4. Integrate with StorageManager by adding new methods
5. Enhance search.py to support vector and hybrid search
6. Test with both Nomic-AI (default) and OpenAI providers

This context, combined with the VECTOR_IMPLEMENTATION_PLAN.md, provides everything needed to implement vector support effectively.