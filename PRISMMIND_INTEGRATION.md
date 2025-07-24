# PrismMind Integration for Flatfile Chat Database

## Overview

The flatfile chat database now includes comprehensive integration with PrismMind's proven engine architecture, providing universal file support and configuration-driven document processing.

## Key Benefits

- **Universal File Support**: Process PDFs, images, URLs, and more using PrismMind's mature handlers
- **Configuration-Driven**: All processing options controlled via JSON configuration files
- **Maximum Code Reuse**: Leverages PrismMind's existing, tested engine architecture
- **Backward Compatibility**: Legacy processing still available as fallback
- **Performance**: Optimized engine chaining and parallel processing

## Usage

### Basic Usage with StorageManager

```python
from storage import StorageManager

# Initialize with PrismMind integration enabled (default)
storage = StorageManager()

# Process document using PrismMind engines
result = await storage.process_document_with_prismmind(
    document_path="/path/to/document.pdf",
    user_id="user123",
    session_id="session456"
)

# Check if PrismMind is available
if storage.is_prismmind_available():
    print("PrismMind integration active")
```

### Using DocumentRAGPipeline

```python
from document_pipeline import DocumentRAGPipeline

# Initialize with PrismMind integration (recommended)
pipeline = DocumentRAGPipeline(use_prismmind=True)

# Process document - automatically routes to PrismMind when available
result = await pipeline.process_document(
    document_path="/path/to/document.pdf",
    user_id="user123", 
    session_id="session456"
)
```

### Direct PrismMind Processor

```python
from prismmind_integration import FlatfileDocumentProcessor, FlatfilePrismMindConfigLoader

# Load configuration
config = FlatfilePrismMindConfigLoader.from_file("configs/flatfile_prismmind_config.json")

# Initialize processor
processor = FlatfileDocumentProcessor(config)

# Process document
result = await processor.process_document(
    document_path="/path/to/document.pdf",
    user_id="user123",
    session_id="session456"
)
```

## Configuration

Configuration files are located in the `configs/` directory:

- `flatfile_prismmind_config.json` - Main configuration with all options
- `development_config.json` - Development environment settings
- `production_config.json` - Production optimized settings  
- `test_config.json` - Test environment settings

### Key Configuration Sections

1. **File Type Chains**: Define processing chains per file type
2. **Handler Selection**: Map file types to specific PrismMind handlers
3. **Strategy Parameters**: Configure chunking, embedding, and NLP strategies
4. **Performance Settings**: Control concurrency, memory, and timeouts
5. **Error Handling**: Retry logic and error recovery

## File Support

PrismMind integration supports:

- **Text Files**: TXT, MD, JSON, CSV, XML, HTML
- **Documents**: PDF, DOCX, DOC  
- **Images**: PNG, JPG, GIF, BMP (with OCR)
- **Web Content**: URLs with full page extraction
- **Archives**: ZIP, TAR (planned)

## Engine Architecture

The integration follows PrismMind's engine chaining pattern:

```
File Input → Ingestion Engine → NLP Engine → Chunking Engine → Embedding Engine → Storage Engine
```

Each engine is configurable and replaceable based on file type and requirements.

## Migration from Legacy

Existing code continues to work unchanged. To use PrismMind features:

1. Update configuration to include PrismMind settings
2. Set `use_prismmind=True` in DocumentRAGPipeline 
3. Call `storage.process_document_with_prismmind()` for direct access
4. Legacy fallback ensures compatibility if PrismMind unavailable

## Environment Setup

Ensure PrismMind engines are accessible in your Python path:

```python
import sys
sys.path.append('/path/to/PrismMind_v2/pm_engines')
```

## Performance Notes

- Production configurations use optimized settings
- Parallel processing enabled for multiple documents
- Vector compression and caching available
- Memory limits and timeouts configurable per environment