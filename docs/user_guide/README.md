# Flatfile Chat Database v2

A high-performance, file-based chat database system built for scalability, maintainability, and ease of use. This system provides a complete solution for storing and managing chat conversations, documents, user profiles, and context data without requiring a traditional database server.

## 🚀 Key Features

- **Zero Database Dependencies**: Pure file-based storage with JSON and JSONL formats
- **Modular Architecture**: Clean separation of concerns with specialized managers
- **Type-Safe**: Comprehensive type annotations throughout
- **Protocol-Based Design**: Interface-driven development for maximum testability
- **Dependency Injection**: Modern DI container with lifetime management
- **Configuration-Driven**: Externalized configuration with environment support
- **Streaming Support**: Memory-efficient handling of large datasets
- **Search & Vector Capabilities**: Full-text search with optional vector similarity search
- **Document Processing**: Built-in document ingestion and analysis
- **Compression Support**: Optional data compression for storage efficiency

## 🏗️ Architecture Overview

The system follows a **function-first, configuration-driven** approach with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                   Protocol Interfaces                      │
├─────────────────────────────────────────────────────────────┤
│                 Specialized Managers                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │    User     │ │   Session   │ │  Document   │   ...   │
│  │   Manager   │ │   Manager   │ │   Manager   │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│              Core Infrastructure                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │   Storage   │ │   Search    │ │   Vector    │   ...   │
│  │   Backend   │ │   Manager   │ │   Storage   │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                 Configuration System                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚦 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd flatfile_chat_database_v2

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO

# Initialize storage manager
storage = FFStorageManager()
await storage.initialize()

# Create a user
await storage.create_user("alice", {"name": "Alice Smith"})

# Create a session
session_id = await storage.create_session("alice", "My First Chat")

# Add a message
message = FFMessageDTO(
    role="user",
    content="Hello, how are you?",
    timestamp="2025-01-01T10:00:00"
)
await storage.add_message("alice", session_id, message)

# Retrieve messages
messages = await storage.get_all_messages("alice", session_id)
print(f"Found {len(messages)} messages")
```

### Using Dependency Injection

```python
from ff_dependency_injection_manager import ff_get_container
from ff_protocols import StorageProtocol

# Get service through dependency injection
container = ff_get_container()
storage = container.resolve(StorageProtocol)

# Use the service
await storage.initialize()
# ... rest of operations
```

## 📁 Project Structure

```
flatfile_chat_database_v2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── ff_storage_manager.py             # Main API interface
├── ff_protocols.py                   # Protocol definitions
├── ff_dependency_injection_manager.py # DI container
│
├── ff_*_manager.py                   # Specialized managers
│   ├── ff_user_manager.py            # User operations
│   ├── ff_session_manager.py         # Session & message operations
│   ├── ff_document_manager.py        # Document operations
│   ├── ff_context_manager.py         # Context operations
│   ├── ff_panel_manager.py           # Panel & persona operations
│   ├── ff_search_manager.py          # Search operations
│   ├── ff_vector_storage_manager.py  # Vector operations
│   ├── ff_streaming_manager.py       # Streaming operations
│   └── ff_compression_manager.py     # Compression operations
│
├── ff_class_configs/                 # Configuration classes
│   ├── ff_configuration_manager_config.py  # Main config
│   ├── ff_chat_entities_config.py          # Data models
│   └── ff_*_config.py                      # Specialized configs
│
├── ff_utils/                         # Utility modules
│   ├── ff_file_ops.py               # File operations
│   ├── ff_validation.py             # Input validation
│   ├── ff_logging.py                # Logging utilities
│   └── ff_*.py                      # Other utilities
│
├── backends/                         # Storage backends
│   └── ff_flatfile_storage_backend.py
│
├── tests/                           # Test suite
├── docs/                            # Documentation
└── demo/                            # Examples and demos
```

## 🔧 Core Components

### Storage Manager (`ff_storage_manager.py`)
The main API interface providing high-level operations for all storage needs.

### Specialized Managers
- **User Manager**: User profiles and user-specific operations
- **Session Manager**: Chat sessions and message management
- **Document Manager**: Document storage and retrieval
- **Context Manager**: Situational context management
- **Panel Manager**: Multi-persona panels and personas
- **Search Manager**: Full-text search across sessions
- **Vector Storage Manager**: Vector embeddings and similarity search
- **Streaming Manager**: Memory-efficient data streaming
- **Compression Manager**: Data compression utilities

### Configuration System
Domain-specific configuration classes with environment-aware loading:
- `StorageConfig`: File operations and paths
- `SearchConfig`: Search behavior and indexing
- `VectorConfig`: Embedding and vector search settings
- `DocumentConfig`: Document processing settings
- `RuntimeConfig`: Runtime behavior and validation

### Dependency Injection
Modern DI container with support for:
- **Transient**: New instance each time
- **Singleton**: Single instance for application lifetime
- **Scoped**: Single instance per scope

## 📚 Documentation

- [Architecture Guide](../upgrade_v2/ARCHITECTURE.md) - System architecture and design principles
- [Configuration Guide](../upgrade_v2/CONFIGURATION_GUIDE.md) - Configuration options and setup
- [API Reference](../upgrade_v2/API_REFERENCE.md) - Complete API documentation
- [Migration Guide](../upgrade_v2/MIGRATION_GUIDE.md) - Upgrading from previous versions
- [Data Models](../upgrade_v2/DATA_MODELS.md) - Data structures and schemas

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_core_functionality.py
python -m pytest tests/test_dependency_injection.py
python -m pytest tests/test_storage_integration.py
```

## 🎯 Key Improvements in v2

### ✅ Architectural Enhancements
- **Single Responsibility**: Each manager focuses on one domain
- **Type Safety**: Comprehensive type annotations throughout
- **Protocol-Based**: Interface-driven development
- **Configuration Externalization**: No hardcoded values
- **Error Handling**: Standardized error handling with proper logging

### ✅ Performance Improvements
- **Lazy Loading**: Services loaded on-demand
- **Streaming Support**: Handle large datasets efficiently
- **Compression**: Optional data compression
- **Caching**: Strategic caching for frequently accessed data

### ✅ Developer Experience
- **Clear APIs**: Intuitive and well-documented interfaces
- **Better Testing**: Easy mocking and service replacement
- **Rich Debugging**: Container introspection and detailed error messages
- **Configuration Validation**: Prevent configuration errors at startup

## 📈 Performance Characteristics

- **Storage**: Efficient JSON/JSONL file operations with optional compression
- **Memory**: Streaming support for large datasets, configurable cache limits
- **Search**: Fast full-text search with optional vector similarity
- **Scalability**: Handles thousands of sessions and messages per user
- **Concurrency**: Thread-safe operations with file locking support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing patterns
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Python's asyncio for high-performance async operations
- Uses modern Python features including type hints and dataclasses
- Inspired by dependency injection patterns from other languages
- Designed for maximum testability and maintainability