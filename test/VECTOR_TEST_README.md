# Vector Implementation Tests

This directory contains test files for the vector functionality in the flatfile chat database.

## Test Files

### 1. `test_flatfile_vectors.py`
Comprehensive test suite that covers:
- Basic vector storage and retrieval
- Document processing with chunking and embedding
- Vector search functionality
- Hybrid search (text + vector)
- Multiple chunking strategies
- Error handling
- Performance metrics

**To run:**
```bash
cd /home/markly2/claude_code
python3 flatfile_chat_database_test/test_flatfile_vectors.py
```

### 2. `example_vector_usage.py`
Simple demonstration of vector features showing:
- How to store documents with automatic vectorization
- Performing vector similarity searches
- Using hybrid search
- Processing documents through the pipeline
- Real-world usage patterns

**To run:**
```bash
cd /home/markly2/claude_code
python3 flatfile_chat_database_test/example_vector_usage.py
```

## Important Notes

1. **Run from parent directory**: Both tests must be run from `/home/markly2/claude_code` to avoid import issues
2. **Dependencies**: Requires numpy, sentence-transformers, spacy (see requirements.txt)
3. **Mock embeddings**: If sentence-transformers is not installed, the system will use mock embeddings
4. **Temporary data**: Tests create and clean up temporary directories automatically

## Vector Features Tested

- **Storage**: Documents are chunked and embedded automatically
- **Search**: Semantic similarity search using cosine similarity
- **Hybrid**: Combines traditional text search with vector search
- **Chunking**: Multiple strategies (optimized_summary, fixed, sentence-based)
- **Providers**: Nomic-AI (default, local) and OpenAI (requires API key)
- **Pipeline**: Complete document processing workflow

## Default Configuration

- Embedding Provider: `nomic-ai` (local, no API key required)
- Chunking Strategy: `optimized_summary` (best semantic coherence)
- Vector Dimensions: 768
- Storage Format: NumPy arrays (.npy) + JSONL metadata