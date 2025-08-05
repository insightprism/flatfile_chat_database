# Configuration Guide - Current Implementation

This guide provides comprehensive information about configuring the flatfile chat database system, including all configuration options, environment-specific settings, and best practices.

## Table of Contents
1. [Configuration Overview](#configuration-overview)
2. [Configuration Structure](#configuration-structure)
3. [Environment-Specific Configuration](#environment-specific-configuration)
4. [Configuration Loading](#configuration-loading)
5. [Validation and Business Rules](#validation-and-business-rules)
6. [Performance Tuning](#performance-tuning)
7. [Security Configuration](#security-configuration)
8. [Troubleshooting](#troubleshooting)

## Configuration Overview

The system uses a hierarchical configuration structure with domain-specific configuration classes that can be customized per environment. All configuration is externalized from code, following the principle of configuration-driven behavior.

### Key Benefits
- **Environment-aware**: Different settings for development, testing, production
- **Validation**: Configurable business rules and validation constraints
- **Performance**: Tunable thresholds and limits
- **Security**: Configurable security and file locking settings
- **Flexibility**: Override any setting via environment variables

## Configuration Structure

The main configuration is composed of domain-specific configuration classes:

```python
FFConfigurationManagerConfigDTO
├── storage: FFStorageConfig          # File paths and storage settings
├── search: FFSearchConfig            # Search behavior and indexing
├── vector: FFVectorConfig            # Vector embeddings and similarity
├── document: FFDocumentConfig        # Document processing settings
├── locking: FFLockingConfig          # File locking configuration
├── panel: FFPanelConfig              # Panel and persona settings
└── runtime: FFRuntimeConfig          # Runtime behavior and validation
```

### Configuration Files Structure

```
ff_preset_configs/
├── ff_development_config.json        # Development environment
├── ff_production_config.json         # Production environment
├── ff_test_config.json              # Testing environment
└── ff_flatfile_prismmind_config.json # PrismMind integration config
```

## Configuration Classes

### 1. FFStorageConfig

Core storage and file system settings.

```python
@dataclass
class FFStorageConfig:
    # File System Settings
    base_path: str = "./data"                          # Root data directory
    user_data_directory_name: str = "users"           # User data subdirectory
    session_id_prefix: str = "chat_session_"          # Session ID prefix
    
    # File Names
    messages_filename: str = "messages.jsonl"         # Message file name
    session_metadata_filename: str = "session.json"   # Session metadata file
    user_profile_filename: str = "profile.json"       # User profile file
    document_metadata_filename: str = "documents.json" # Document metadata file
    situational_context_filename: str = "context.json" # Context file
    
    # Directory Names
    document_storage_subdirectory_name: str = "documents"      # Documents subdirectory
    context_history_subdirectory_name: str = "context_history" # Context history subdirectory
    vector_storage_subdirectory_name: str = "vectors"          # Vector storage subdirectory
    
    # File Locking and Safety
    enable_file_locking: bool = True                   # Enable file locking
    backup_on_write: bool = False                      # Create backups on write
    
    # Size Limits
    max_message_size_bytes: int = 1_000_000           # 1MB per message
    max_document_size_bytes: int = 10_000_000         # 10MB per document
    max_session_size_bytes: int = 100_000_000         # 100MB per session
```

**Configuration Example:**
```json
{
    "storage": {
        "base_path": "/var/lib/chatdb/data",
        "enable_file_locking": true,
        "max_message_size_bytes": 2000000,
        "max_document_size_bytes": 20000000,
        "backup_on_write": true
    }
}
```

### 2. FFRuntimeConfig

Runtime behavior and validation rules.

```python
@dataclass
class FFRuntimeConfig:
    # Business Logic Thresholds
    large_session_threshold_bytes: int = 10_000_000        # 10MB
    storage_default_message_limit: int = 1000              # Default message limit
    storage_default_session_limit: int = 100               # Default session limit
    cache_size_limit: int = 100                            # Cache size limit
    
    # Disk Usage Monitoring
    disk_usage_threshold_percent: float = 85.0             # Disk usage warning threshold
    disk_usage_critical_percent: float = 95.0              # Disk usage critical threshold
    
    # Validation Rules - User IDs
    user_id_min_length: int = 3                            # Minimum user ID length
    user_id_max_length: int = 50                           # Maximum user ID length
    user_id_allowed_chars: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    
    # Validation Rules - Session Names
    session_name_min_length: int = 1                       # Minimum session name length
    session_name_max_length: int = 200                     # Maximum session name length
    
    # Validation Rules - File Names
    filename_min_length: int = 1                           # Minimum filename length
    filename_max_length: int = 255                         # Maximum filename length
    
    # Validation Rules - Content
    message_content_min_length: int = 1                    # Minimum message content length
    message_content_max_length: int = 100_000              # Maximum message content length
    document_content_max_length: int = 10_000_000          # Maximum document content length
    
    # File Extensions
    document_content_extension: str = ".txt"               # Document content file extension
    context_file_extension: str = ".json"                  # Context file extension
    persona_file_extension: str = ".json"                  # Persona file extension
    insight_file_extension: str = ".json"                  # Insight file extension
    
    # Logging Configuration
    log_level: str = "INFO"                                # Logging level
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_performance_logging: bool = False               # Enable performance logging
```

**Configuration Example:**
```json
{
    "runtime": {
        "large_session_threshold_bytes": 20000000,
        "storage_default_message_limit": 2000,
        "user_id_min_length": 4,
        "user_id_max_length": 30,
        "log_level": "DEBUG",
        "enable_performance_logging": true,
        "disk_usage_threshold_percent": 80.0
    }
}
```

### 3. FFSearchConfig

Search and indexing configuration.

```python
@dataclass
class FFSearchConfig:
    # Search Behavior
    default_search_limit: int = 100                        # Default search result limit
    max_search_results: int = 1000                         # Maximum search results
    min_query_length: int = 2                              # Minimum search query length
    max_query_length: int = 500                            # Maximum search query length
    
    # Indexing Settings
    enable_automatic_indexing: bool = True                 # Auto-index new content
    index_rebuild_threshold: int = 1000                    # Messages before index rebuild
    enable_fuzzy_search: bool = True                       # Enable fuzzy matching
    fuzzy_threshold: float = 0.8                          # Fuzzy match threshold
    
    # Performance Settings
    search_timeout_seconds: int = 30                       # Search operation timeout
    indexing_batch_size: int = 100                        # Batch size for indexing
    
    # Entity Extraction
    enable_entity_extraction: bool = True                  # Enable entity extraction
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY"
    ])
```

**Configuration Example:**
```json
{
    "search": {
        "default_search_limit": 50,
        "max_search_results": 500,
        "enable_fuzzy_search": true,
        "fuzzy_threshold": 0.85,
        "search_timeout_seconds": 15,
        "enable_entity_extraction": true
    }
}
```

### 4. FFVectorConfig

Vector embeddings and similarity search configuration.

```python
@dataclass
class FFVectorConfig:
    # Embedding Settings
    default_embedding_provider: str = "nomic-ai"           # Default embedding provider
    embedding_model: str = "nomic-embed-text-v1"          # Embedding model name
    embedding_dimensions: int = 768                        # Embedding vector dimensions
    
    # Vector Storage
    vector_index_filename: str = "vector_index.jsonl"     # Vector index file
    embeddings_filename: str = "embeddings.npy"           # Embeddings file (numpy)
    enable_vector_compression: bool = True                 # Compress vector storage
    
    # Similarity Search
    similarity_threshold: float = 0.7                      # Default similarity threshold
    max_similarity_results: int = 50                       # Maximum similarity results
    similarity_search_timeout: int = 10                    # Similarity search timeout
    
    # Performance
    embedding_batch_size: int = 10                         # Batch size for embeddings
    vector_cache_size: int = 1000                         # Vector cache size
    enable_async_embedding: bool = True                    # Async embedding generation
```

**Configuration Example:**
```json
{
    "vector": {
        "default_embedding_provider": "openai",
        "embedding_model": "text-embedding-ada-002",
        "embedding_dimensions": 1536,
        "similarity_threshold": 0.75,
        "max_similarity_results": 25,
        "enable_vector_compression": true
    }
}
```

### 5. FFDocumentConfig

Document processing configuration.

```python
@dataclass
class FFDocumentConfig:
    # Allowed File Types
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".json", ".csv", ".tsv", ".log"
    ])
    
    # Processing Settings
    enable_text_extraction: bool = True                    # Enable text extraction
    enable_document_analysis: bool = True                  # Enable document analysis
    max_extraction_size_bytes: int = 50_000_000           # Max size for text extraction
    
    # Content Processing
    normalize_whitespace: bool = True                      # Normalize whitespace in content
    remove_empty_lines: bool = True                        # Remove empty lines
    max_line_length: int = 10000                          # Maximum line length
    
    # Metadata
    store_file_metadata: bool = True                       # Store file metadata
    compute_content_hash: bool = True                      # Compute content hashes
    
    # Integration Settings
    prismmind_integration: bool = False                    # Enable PrismMind integration
    prismmind_analysis_timeout: int = 60                   # PrismMind analysis timeout
```

**Configuration Example:**
```json
{
    "document": {
        "allowed_extensions": [".txt", ".md", ".json", ".pdf", ".docx"],
        "enable_document_analysis": true,
        "max_extraction_size_bytes": 100000000,
        "normalize_whitespace": true,
        "store_file_metadata": true,
        "prismmind_integration": true
    }
}
```

### 6. FFPanelConfig

Panel and persona management configuration.

```python
@dataclass
class FFPanelConfig:
    # Panel Settings
    max_personas_per_panel: int = 10                       # Maximum personas per panel
    panel_sessions_directory: str = "panel_sessions"       # Panel sessions directory
    default_panel_timeout: int = 3600                      # Default panel timeout (seconds)
    
    # Persona Settings
    global_personas_directory: str = "personas_global"     # Global personas directory
    user_personas_directory: str = "personas_user"         # User personas directory
    max_persona_description_length: int = 2000             # Max persona description length
    
    # Panel Messages
    max_panel_message_length: int = 10000                  # Max panel message length
    panel_message_history_limit: int = 1000                # Panel message history limit
    
    # Insights
    enable_panel_insights: bool = True                      # Enable panel insights
    max_insight_length: int = 5000                          # Maximum insight length
    insight_retention_days: int = 30                        # Insight retention period
```

**Configuration Example:**
```json
{
    "panel": {
        "max_personas_per_panel": 5,
        "max_persona_description_length": 1500,
        "enable_panel_insights": true,
        "insight_retention_days": 60,
        "panel_message_history_limit": 500
    }
}
```

### 7. FFLockingConfig

File locking and concurrency configuration.

```python
@dataclass
class FFLockingConfig:
    # Locking Behavior
    enable_file_locking: bool = True                        # Enable file locking
    lock_timeout_seconds: float = 30.0                     # Lock acquisition timeout
    lock_retry_interval_ms: int = 100                      # Retry interval (milliseconds)
    max_lock_retries: int = 50                             # Maximum lock retries
    
    # Lock Files
    lock_file_extension: str = ".lock"                     # Lock file extension
    cleanup_stale_locks: bool = True                       # Clean up stale locks
    stale_lock_timeout_seconds: int = 300                  # Stale lock timeout (5 minutes)
    
    # Concurrency
    max_concurrent_operations: int = 10                    # Max concurrent operations
    enable_read_write_locks: bool = True                   # Enable read/write lock distinction
    
    # Performance
    lock_check_interval_ms: int = 50                       # Lock status check interval
    enable_lock_monitoring: bool = False                   # Enable lock monitoring/logging
```

**Configuration Example:**
```json
{
    "locking": {
        "enable_file_locking": true,
        "lock_timeout_seconds": 15.0,
        "max_lock_retries": 30,
        "cleanup_stale_locks": true,
        "max_concurrent_operations": 20,
        "enable_lock_monitoring": true
    }
}
```

## Environment-Specific Configuration

### Development Environment

**File: `ff_preset_configs/ff_development_config.json`**

```json
{
    "storage": {
        "base_path": "./dev_data",
        "enable_file_locking": false,
        "backup_on_write": false,
        "max_message_size_bytes": 500000,
        "max_document_size_bytes": 5000000
    },
    "runtime": {
        "large_session_threshold_bytes": 1000000,
        "storage_default_message_limit": 100,
        "log_level": "DEBUG",
        "enable_performance_logging": true,
        "user_id_min_length": 1,
        "disk_usage_threshold_percent": 95.0
    },
    "search": {
        "default_search_limit": 20,
        "search_timeout_seconds": 5,
        "enable_automatic_indexing": true,
        "indexing_batch_size": 50
    },
    "vector": {
        "similarity_threshold": 0.6,
        "max_similarity_results": 10,
        "enable_vector_compression": false,
        "vector_cache_size": 100
    },
    "document": {
        "allowed_extensions": [".txt", ".md", ".json"],
        "enable_document_analysis": false,
        "prismmind_integration": false
    },
    "locking": {
        "enable_file_locking": false,
        "lock_timeout_seconds": 5.0,
        "max_lock_retries": 10
    }
}
```

### Production Environment

**File: `ff_preset_configs/ff_production_config.json`**

```json
{
    "storage": {
        "base_path": "/var/lib/chatdb/data",
        "enable_file_locking": true,
        "backup_on_write": true,
        "max_message_size_bytes": 2000000,
        "max_document_size_bytes": 50000000
    },
    "runtime": {
        "large_session_threshold_bytes": 50000000,
        "storage_default_message_limit": 1000,
        "log_level": "INFO",
        "enable_performance_logging": false,
        "user_id_min_length": 3,
        "disk_usage_threshold_percent": 80.0,
        "disk_usage_critical_percent": 90.0
    },
    "search": {
        "default_search_limit": 100,
        "max_search_results": 1000,
        "search_timeout_seconds": 30,
        "enable_automatic_indexing": true,
        "indexing_batch_size": 500
    },
    "vector": {
        "similarity_threshold": 0.75,
        "max_similarity_results": 50,
        "enable_vector_compression": true,
        "vector_cache_size": 5000,
        "enable_async_embedding": true
    },
    "document": {
        "allowed_extensions": [".txt", ".md", ".json", ".csv", ".log", ".xml"],
        "enable_document_analysis": true,
        "max_extraction_size_bytes": 100000000,
        "prismmind_integration": true,
        "prismmind_analysis_timeout": 120
    },
    "locking": {
        "enable_file_locking": true,
        "lock_timeout_seconds": 30.0,
        "max_lock_retries": 100,
        "cleanup_stale_locks": true,
        "max_concurrent_operations": 50,
        "enable_lock_monitoring": true
    }
}
```

### Test Environment

**File: `ff_preset_configs/ff_test_config.json`**

```json
{
    "storage": {
        "base_path": "./test_data",
        "enable_file_locking": false,
        "backup_on_write": false,
        "max_message_size_bytes": 100000,
        "max_document_size_bytes": 1000000
    },
    "runtime": {
        "large_session_threshold_bytes": 100000,
        "storage_default_message_limit": 10,
        "log_level": "WARNING",
        "enable_performance_logging": false,
        "user_id_min_length": 1,
        "session_name_min_length": 1
    },
    "search": {
        "default_search_limit": 5,
        "max_search_results": 50,
        "search_timeout_seconds": 2,
        "enable_automatic_indexing": false
    },
    "vector": {
        "similarity_threshold": 0.5,
        "max_similarity_results": 5,
        "enable_vector_compression": false,
        "vector_cache_size": 10
    },
    "document": {
        "allowed_extensions": [".txt"],
        "enable_document_analysis": false,
        "prismmind_integration": false
    },
    "locking": {
        "enable_file_locking": false,
        "lock_timeout_seconds": 1.0,
        "max_lock_retries": 3
    }
}
```

## Configuration Loading

### Loading Configuration

```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Load default configuration
config = load_config()

# Load environment-specific configuration
dev_config = load_config(environment="development")
prod_config = load_config(environment="production")
test_config = load_config(environment="test")

# Load from specific file
custom_config = load_config(config_path="/path/to/custom_config.json")

# Load with environment and custom overrides
config = load_config(
    config_path="/path/to/base_config.json",
    environment="production"
)
```

### Environment Variable Overrides

You can override any configuration value using environment variables with the pattern:
`FF_<SECTION>_<SETTING>`

```bash
# Override storage base path
export FF_STORAGE_BASE_PATH="/custom/data/path"

# Override runtime log level
export FF_RUNTIME_LOG_LEVEL="DEBUG"

# Override search settings
export FF_SEARCH_DEFAULT_SEARCH_LIMIT="200"
export FF_SEARCH_ENABLE_FUZZY_SEARCH="false"

# Override vector settings
export FF_VECTOR_SIMILARITY_THRESHOLD="0.8"
export FF_VECTOR_EMBEDDING_PROVIDER="openai"

# Override locking settings
export FF_LOCKING_ENABLE_FILE_LOCKING="true"
export FF_LOCKING_LOCK_TIMEOUT_SECONDS="45.0"
```

### Configuration Validation

The system validates configuration at startup:

```python
# Example of configuration validation
config = load_config(environment="production")

# Validation occurs automatically during loading
# Invalid configurations will raise ConfigurationError

# Manual validation
from ff_class_configs.ff_configuration_manager_config import validate_configuration

errors = validate_configuration(config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Performance Tuning

### Memory Management

```json
{
    "runtime": {
        "cache_size_limit": 1000,              // Adjust based on available memory
        "storage_default_message_limit": 500   // Lower limit for memory-constrained environments
    },
    "vector": {
        "vector_cache_size": 5000,             // Vector cache size
        "embedding_batch_size": 20             // Batch size for embedding generation
    },
    "search": {
        "indexing_batch_size": 1000            // Batch size for search indexing
    }
}
```

### I/O Performance

```json
{
    "storage": {
        "enable_file_locking": true,           // Enable for safety, disable for speed
        "backup_on_write": false               // Disable for better write performance
    },
    "locking": {
        "lock_timeout_seconds": 10.0,          // Shorter timeout for faster failures
        "max_concurrent_operations": 50        // Increase for better concurrency
    }
}
```

### Search Performance

```json
{
    "search": {
        "enable_automatic_indexing": true,     // Keep enabled for real-time search
        "indexing_batch_size": 1000,          // Larger batches for better throughput
        "search_timeout_seconds": 15,         // Reasonable timeout
        "index_rebuild_threshold": 5000       // Higher threshold reduces rebuilds
    }
}
```

## Security Configuration

### File System Security

```json
{
    "storage": {
        "base_path": "/secure/chatdb/data",    // Use secure directory with proper permissions
        "enable_file_locking": true,          // Essential for multi-process safety
        "backup_on_write": true               // Enable for data protection
    },
    "locking": {
        "enable_file_locking": true,          // Critical for data integrity
        "cleanup_stale_locks": true,          // Prevent lock file accumulation
        "stale_lock_timeout_seconds": 300     // Reasonable timeout for stale locks
    }
}
```

### Validation Security

```json
{
    "runtime": {
        "user_id_min_length": 3,              // Prevent short/guessable user IDs
        "user_id_max_length": 50,             // Prevent excessively long user IDs
        "message_content_max_length": 100000, // Prevent DoS via large messages
        "document_content_max_length": 10000000, // Limit document sizes
        "filename_max_length": 255            // Standard filesystem limit
    }
}
```

### Content Security

```json
{
    "document": {
        "allowed_extensions": [".txt", ".md", ".json"], // Restrict file types
        "max_extraction_size_bytes": 50000000,         // Limit processing size
        "store_file_metadata": true,                    // Track file origins
        "compute_content_hash": true                    // Detect content changes
    }
}
```

## Troubleshooting

### Common Configuration Issues

#### 1. Path Configuration Problems

```bash
# Problem: Storage path doesn't exist
Error: "Storage path '/var/lib/chatdb/data' does not exist"

# Solution: Create the directory or update configuration
mkdir -p /var/lib/chatdb/data
# OR
export FF_STORAGE_BASE_PATH="./existing_data"
```

#### 2. Permission Issues

```bash
# Problem: Permission denied errors
Error: "Permission denied when writing to '/var/lib/chatdb/data'"

# Solution: Fix permissions
sudo chown -R $USER:$USER /var/lib/chatdb/data
sudo chmod -R 755 /var/lib/chatdb/data
```

#### 3. File Locking Issues

```bash
# Problem: Lock timeout errors in development
Error: "Failed to acquire lock after 30 seconds"

# Solution: Disable locking for development
export FF_LOCKING_ENABLE_FILE_LOCKING="false"
# OR reduce timeout
export FF_LOCKING_LOCK_TIMEOUT_SECONDS="5.0"
```

#### 4. Memory Issues

```bash
# Problem: Out of memory errors
Error: "Memory allocation failed during vector operations"

# Solution: Reduce cache sizes
export FF_RUNTIME_CACHE_SIZE_LIMIT="100"
export FF_VECTOR_VECTOR_CACHE_SIZE="500"
export FF_SEARCH_INDEXING_BATCH_SIZE="100"
```

#### 5. Search Performance Issues

```bash
# Problem: Slow search operations
Error: "Search timeout after 30 seconds"

# Solution: Optimize search configuration
export FF_SEARCH_SEARCH_TIMEOUT_SECONDS="60"
export FF_SEARCH_INDEXING_BATCH_SIZE="500"
export FF_SEARCH_ENABLE_AUTOMATIC_INDEXING="false"  # Manual indexing
```

### Configuration Debugging

Enable detailed logging to debug configuration issues:

```json
{
    "runtime": {
        "log_level": "DEBUG",
        "enable_performance_logging": true
    },
    "locking": {
        "enable_lock_monitoring": true
    }
}
```

View configuration at runtime:

```python
from ff_class_configs.ff_configuration_manager_config import load_config

config = load_config()
print("Current configuration:")
print(f"Storage base path: {config.storage.base_path}")
print(f"Log level: {config.runtime.log_level}")
print(f"File locking enabled: {config.locking.enable_file_locking}")
print(f"Search timeout: {config.search.search_timeout_seconds}")
```

### Environment Variable Debugging

List all flatfile-related environment variables:

```bash
env | grep ^FF_ | sort
```

This configuration guide provides comprehensive information for setting up and tuning the flatfile chat database system for any environment or use case.