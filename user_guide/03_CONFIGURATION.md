# Configuration Guide

Complete guide to configuring the Flatfile Chat Database system for your specific needs.

## 🎯 Configuration Overview

The system uses a hierarchical configuration approach with multiple layers:

```
Built-in Defaults → Environment Config → Runtime Overrides → User Customization
```

## 🏗️ Configuration Architecture

### Configuration Loading Order
1. **Built-in defaults** - Sensible defaults for all settings
2. **Environment-specific configs** - JSON files for dev/test/prod
3. **Environment variables** - OS-level configuration
4. **Runtime overrides** - Programmatic configuration changes

### Configuration Types
- **Storage Configuration** - File system, paths, caching
- **Search Configuration** - Full-text search settings
- **Vector Configuration** - Embedding and similarity search
- **Document Configuration** - Document processing settings
- **Streaming Configuration** - Real-time streaming settings
- **Compression Configuration** - Data compression options

## 📝 Basic Configuration

### Loading Default Configuration
```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Load with built-in defaults
config = load_config()

# Load environment-specific
config = load_config("development")  # or "test", "production"
```

### Inspecting Configuration
```python
# View current configuration
print(f"Base path: {config.storage.base_path}")
print(f"Environment: {config.environment}")
print(f"Debug mode: {config.debug_mode}")

# View all storage settings
print(config.storage)
```

## 🗂️ Configuration Files

### File Locations
```
ff_preset_configs/
├── ff_development_config.json    # Development settings
├── ff_test_config.json          # Testing settings
├── ff_production_config.json    # Production settings
└── ff_flatfile_prismmind_config.json  # PrismMind integration
```

### Environment-Specific Loading
```python
import os

# Set custom config directory
os.environ['FF_CONFIG_PATH'] = '/path/to/custom/configs'

# Load from custom location
config = load_config("production")
```

## 🗄️ Storage Configuration

### Basic Storage Settings
```python
# Access storage configuration
storage_config = config.storage

print(f"Base path: {storage_config.base_path}")
print(f"User data path: {storage_config.user_data_path}")
print(f"Max message size: {storage_config.max_message_size_bytes}")
```

### Storage Configuration Options
```json
{
  "storage": {
    "base_path": "./data",
    "user_data_path": "users",
    "system_data_path": "system",
    "session_id_prefix": "chat_session_",
    "message_id_length": 8,
    "max_message_size_bytes": 1048576,
    "max_messages_per_session": 10000,
    "enable_message_compression": true,
    "message_cache_size": 1000,
    "batch_write_size": 50,
    "validate_json_on_read": false,
    "backup_on_write": false,
    "auto_cleanup_old_sessions": false,
    "cleanup_threshold_days": 90
  }
}
```

### Customizing Storage Paths
```python
config = load_config()

# Customize paths
config.storage.base_path = "/custom/storage/location"
config.storage.user_data_path = "user_profiles"
config.storage.system_data_path = "system_files"

# Apply configuration
storage = FFStorageManager(config)
```

### Advanced Storage Settings
```python
# Performance tuning
config.storage.message_cache_size = 2000  # Cache more messages
config.storage.batch_write_size = 100     # Larger batch writes
config.storage.enable_message_compression = True

# Validation settings
config.storage.validate_json_on_read = True   # Slower but safer
config.storage.backup_on_write = True        # Create backups

# Cleanup settings
config.storage.auto_cleanup_old_sessions = True
config.storage.cleanup_threshold_days = 30
```

## 🔍 Search Configuration

### Search Settings
```json
{
  "search": {
    "enable_full_text_search": true,
    "search_index_path": "search_indices",
    "max_search_results": 1000,
    "search_result_snippet_length": 200,
    "enable_search_cache": true,
    "search_cache_size": 500,
    "search_cache_ttl_seconds": 300,
    "indexing_batch_size": 100,
    "enable_fuzzy_search": true,
    "fuzzy_search_threshold": 0.8,
    "search_ranking_algorithm": "tf_idf"
  }
}
```

### Customizing Search
```python
# Enable advanced search features
config.search.enable_fuzzy_search = True
config.search.fuzzy_search_threshold = 0.7

# Performance tuning
config.search.max_search_results = 500
config.search.search_cache_size = 1000
config.search.indexing_batch_size = 200

# Search result formatting
config.search.search_result_snippet_length = 150
```

## 🧠 Vector Configuration

### Vector Storage Settings
```json
{
  "vector": {
    "enable_vector_storage": true,
    "vector_storage_path": "vectors",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "similarity_threshold": 0.7,
    "max_similar_results": 50,
    "cache_embeddings": true,
    "embedding_cache_size": 1000,
    "vector_index_type": "flat",
    "enable_incremental_indexing": true,
    "batch_embedding_size": 32
  }
}
```

### Vector Configuration
```python
# Embedding model selection
config.vector.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
config.vector.embedding_dimension = 384

# Similarity search tuning
config.vector.similarity_threshold = 0.75
config.vector.max_similar_results = 25

# Performance optimization
config.vector.cache_embeddings = True
config.vector.batch_embedding_size = 64
```

## 📄 Document Configuration

### Document Processing Settings
```json
{
  "document": {
    "enable_document_processing": true,
    "document_storage_path": "documents",
    "max_document_size_bytes": 10485760,
    "supported_document_types": [".txt", ".md", ".pdf", ".docx"],
    "enable_document_indexing": true,
    "cache_processed_documents": true,
    "document_cache_size": 200,
    "text_extraction_timeout_seconds": 30,
    "enable_document_compression": true,
    "document_chunk_size": 1000,
    "document_chunk_overlap": 200
  }
}
```

### Document Settings
```python
# File type restrictions
config.document.supported_document_types = [".txt", ".md", ".pdf"]
config.document.max_document_size_bytes = 5 * 1024 * 1024  # 5MB

# Processing settings
config.document.text_extraction_timeout_seconds = 60
config.document.document_chunk_size = 500
config.document.document_chunk_overlap = 100

# Caching
config.document.cache_processed_documents = True
config.document.document_cache_size = 500
```

## 🔄 Streaming Configuration

### Real-time Streaming Settings
```json
{
  "streaming": {
    "enable_streaming": true,
    "stream_buffer_size": 1000,
    "stream_timeout_seconds": 30,
    "max_concurrent_streams": 100,
    "enable_stream_compression": false,
    "stream_heartbeat_interval": 10,
    "enable_stream_persistence": true,
    "stream_batch_size": 10
  }
}
```

### Streaming Configuration
```python
# Performance tuning
config.streaming.stream_buffer_size = 2000
config.streaming.max_concurrent_streams = 200
config.streaming.stream_batch_size = 20

# Reliability settings
config.streaming.stream_timeout_seconds = 60
config.streaming.stream_heartbeat_interval = 5
config.streaming.enable_stream_persistence = True
```

## 🗜️ Compression Configuration

### Compression Settings
```json
{
  "compression": {
    "enable_compression": true,
    "compression_algorithm": "gzip",
    "compression_level": 6,
    "compress_messages": true,
    "compress_documents": true,
    "compress_search_indices": false,
    "compression_threshold_bytes": 1024,
    "decompression_cache_size": 100
  }
}
```

### Compression Tuning
```python
# Enable compression for space saving
config.compression.enable_compression = True
config.compression.compression_level = 9  # Maximum compression

# Selective compression
config.compression.compress_messages = True
config.compression.compress_documents = True
config.compression.compression_threshold_bytes = 512  # Compress files > 512 bytes

# Performance vs. space trade-off
config.compression.compression_level = 3  # Faster compression
config.compression.decompression_cache_size = 200
```

## 🔒 Locking Configuration

### File Locking Settings
```json
{
  "locking": {
    "enabled": true,
    "timeout_seconds": 10.0,
    "retry_interval_seconds": 0.1,
    "max_retries": 100,
    "lock_file_suffix": ".lock",
    "enable_process_locking": true,
    "enable_thread_locking": true,
    "lock_cleanup_interval": 60
  }
}
```

### Locking Configuration
```python
# Adjust for high-concurrency scenarios
config.locking.timeout_seconds = 30.0
config.locking.max_retries = 200
config.locking.retry_interval_seconds = 0.05

# Enable advanced locking
config.locking.enable_process_locking = True
config.locking.enable_thread_locking = True
```

## 🌍 Environment Variables

### Supported Environment Variables
```bash
# Configuration
export FF_CONFIG_PATH="/path/to/configs"
export FF_ENVIRONMENT="production"

# Storage
export FF_DATA_PATH="/custom/data/path"
export FF_BASE_PATH="/custom/base/path"

# Logging
export FF_LOG_LEVEL="INFO"
export FF_LOG_FILE="/var/log/ff_chat.log"

# Performance
export FF_CACHE_SIZE="2000"
export FF_BATCH_SIZE="100"

# Features
export FF_ENABLE_SEARCH="true"
export FF_ENABLE_VECTORS="true"
export FF_ENABLE_COMPRESSION="true"
```

### Using Environment Variables
```python
import os

# Set environment variables programmatically
os.environ['FF_ENVIRONMENT'] = 'production'
os.environ['FF_LOG_LEVEL'] = 'WARNING'

# Load configuration with environment overrides
config = load_config()
```

## 🔧 Runtime Configuration

### Modifying Configuration at Runtime
```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Load base configuration
config = load_config("production")

# Runtime modifications
config.storage.base_path = "/runtime/custom/path"
config.storage.message_cache_size = 5000
config.search.enable_fuzzy_search = False

# Create storage manager with modified config
storage = FFStorageManager(config)
```

### Configuration Validation
```python
# The system automatically validates configuration
try:
    config = load_config("production")
    # Configuration is automatically validated
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 📊 Performance Configuration

### Development Configuration
```python
# Optimized for development speed
config = load_config("development")

# Typical development settings:
# - Fast startup
# - Verbose logging
# - Minimal caching
# - No compression
# - Relaxed validation
```

### Production Configuration
```python
# Optimized for production performance
config = load_config("production")

# Typical production settings:
# - Maximum performance
# - Minimal logging
# - Aggressive caching
# - Compression enabled
# - Strict validation
```

### High-Performance Configuration
```python
config = load_config("production")

# Memory optimization
config.storage.message_cache_size = 10000
config.search.search_cache_size = 5000
config.vector.embedding_cache_size = 2000

# I/O optimization
config.storage.batch_write_size = 200
config.document.document_cache_size = 1000

# Compression for space efficiency
config.compression.enable_compression = True
config.compression.compression_level = 6

# Disable expensive operations
config.storage.validate_json_on_read = False
config.storage.backup_on_write = False
```

## 📁 Custom Configuration Files

### Creating Custom Configuration
```json
{
  "environment": "custom",
  "debug_mode": false,
  "storage": {
    "base_path": "/my/custom/path",
    "message_cache_size": 2000,
    "enable_message_compression": true
  },
  "search": {
    "enable_full_text_search": true,
    "max_search_results": 500
  },
  "vector": {
    "enable_vector_storage": true,
    "similarity_threshold": 0.8
  }
}
```

### Loading Custom Configuration
```python
# Save as my_custom_config.json in ff_preset_configs/
config = load_config("custom")

# Or specify full path
import os
os.environ['FF_CONFIG_PATH'] = '/path/to/my/configs'
config = load_config("my_config_name")
```

## 🧪 Testing Configuration

### Test-Specific Settings
```python
# Test configuration optimizations
config = load_config("test")

# Typical test settings:
# - Fast execution
# - In-memory operations where possible
# - Minimal caching
# - No compression
# - Temporary directories
```

### Configuration for Unit Tests
```python
import tempfile
from pathlib import Path

# Create isolated test configuration
config = load_config("test")
config.storage.base_path = tempfile.mkdtemp()
config.storage.validate_json_on_read = True  # Strict validation in tests
config.locking.timeout_seconds = 1.0        # Quick timeouts in tests
```

## 🔍 Configuration Debugging

### Debugging Configuration Issues
```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Enable verbose configuration loading
import logging
logging.basicConfig(level=logging.DEBUG)

config = load_config("development")

# Inspect configuration values
print("=== Storage Configuration ===")
for key, value in config.storage.__dict__.items():
    print(f"{key}: {value}")

print("\n=== Search Configuration ===")
for key, value in config.search.__dict__.items():
    print(f"{key}: {value}")
```

### Configuration Validation
```python
def validate_custom_configuration(config):
    """Validate custom configuration settings."""
    
    # Check required paths exist
    base_path = Path(config.storage.base_path)
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    # Check reasonable limits
    if config.storage.max_message_size_bytes > 100 * 1024 * 1024:  # 100MB
        raise ValueError("Message size limit too high")
    
    # Check cache sizes
    if config.storage.message_cache_size < 0:
        raise ValueError("Cache size cannot be negative")
    
    print("✅ Configuration validation passed")

# Use validation
config = load_config("production")
validate_custom_configuration(config)
```

## 📖 Configuration Best Practices

### 1. Environment Separation
- Use different configurations for dev/test/prod
- Never commit sensitive configuration to version control
- Use environment variables for deployment-specific settings

### 2. Performance Tuning
- Start with defaults and measure performance
- Increase cache sizes gradually based on memory availability
- Enable compression in production for space savings
- Adjust batch sizes based on I/O characteristics

### 3. Security Considerations
- Restrict file permissions on configuration files
- Use environment variables for sensitive settings
- Validate all configuration inputs
- Enable appropriate logging levels

### 4. Monitoring Configuration
```python
# Add configuration monitoring
def log_configuration_summary(config):
    print(f"Environment: {config.environment}")
    print(f"Base path: {config.storage.base_path}")
    print(f"Cache sizes: msg={config.storage.message_cache_size}, "
          f"search={config.search.search_cache_size}")
    print(f"Features: search={config.search.enable_full_text_search}, "
          f"vectors={config.vector.enable_vector_storage}")

config = load_config()
log_configuration_summary(config)
```

## 🎯 Configuration Recipes

### High-Throughput Configuration
```python
config = load_config("production")

# Optimize for high message throughput
config.storage.message_cache_size = 20000
config.storage.batch_write_size = 500
config.locking.timeout_seconds = 1.0
config.compression.enable_compression = False  # CPU vs space trade-off
```

### Low-Memory Configuration
```python
config = load_config("production")

# Optimize for low memory usage
config.storage.message_cache_size = 100
config.search.search_cache_size = 50
config.vector.embedding_cache_size = 25
config.document.document_cache_size = 10
config.compression.enable_compression = True
```

### High-Security Configuration
```python
config = load_config("production")

# Maximum security settings
config.storage.validate_json_on_read = True
config.storage.backup_on_write = True
config.locking.enabled = True
config.locking.timeout_seconds = 30.0
```

## 🆘 Configuration Troubleshooting

### Common Configuration Issues

1. **Path Not Found Errors**
   ```python
   # Ensure paths exist
   from pathlib import Path
   Path(config.storage.base_path).mkdir(parents=True, exist_ok=True)
   ```

2. **Permission Errors**
   ```bash
   # Fix permissions
   chmod -R 755 /path/to/data
   chown -R user:group /path/to/data
   ```

3. **Invalid JSON Configuration**
   ```python
   # Validate JSON files
   import json
   with open('config.json') as f:
       config_data = json.load(f)  # Will raise error if invalid
   ```

4. **Environment Variable Issues**
   ```python
   import os
   print("FF_CONFIG_PATH:", os.environ.get('FF_CONFIG_PATH', 'not set'))
   print("FF_ENVIRONMENT:", os.environ.get('FF_ENVIRONMENT', 'not set'))
   ```

## 🎉 Configuration Summary

The Flatfile Chat Database configuration system provides:

- ✅ **Flexible Configuration**: Multiple layers with sensible defaults
- ✅ **Environment-Specific**: Different settings for different environments
- ✅ **Runtime Modification**: Change settings programmatically
- ✅ **Validation**: Automatic validation of configuration values
- ✅ **Performance Tuning**: Extensive options for optimization
- ✅ **Security**: Secure configuration management practices

The system is designed to work out-of-the-box with minimal configuration while providing extensive customization options for advanced use cases.

**Next Steps**: Now that you understand configuration, continue to [Basic Usage](04_BASIC_USAGE.md) to learn how to use the configured system.