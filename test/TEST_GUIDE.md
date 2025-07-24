# Flatfile Chat Database - Testing Guide

## Quick Start

```bash
# Run the quick demo
python3 quick_test_demo.py

# Run basic tests
python3 test_flatfile.py

# Run advanced search tests
python3 test_advanced_search.py

# Run Phase 4 features test (streaming, compression, migration)
python3 test_phase4_features.py
```

## Test Coverage

### 1. **Basic Functionality Test** (`test_flatfile.py`)
Tests core CRUD operations:
- User creation and management
- Session lifecycle
- Message storage and retrieval
- Document handling
- Session listing
- Basic search

### 2. **Advanced Search Test** (`test_advanced_search.py`)
Tests sophisticated search features:
- Text search with relevance scoring
- Entity-based search (URLs, emails, languages)
- Time-range queries
- Role-based filtering
- Cross-session search
- Search index building

### 3. **Production Features Test** (`test_phase4_features.py`)
Tests production-ready features:
- **Streaming**: Efficient handling of large sessions
- **Compression**: Data compression for storage optimization
- **Migration**: Export to SQLite and JSON formats
- **Performance**: Benchmarks ensuring targets are met

### 4. **Complete System Test** (`test_complete_system.py`)
Comprehensive test suite covering:
- All basic operations
- Document management
- Context tracking
- Search capabilities
- Panel sessions
- Performance validation
- Error handling

### 5. **Quick Demo** (`quick_test_demo.py`)
Interactive demonstration showing:
- Real-world usage example
- All major features in action
- Storage structure visualization

## Manual Testing

### Testing Storage Operations

```python
import asyncio
from flatfile_chat_database import StorageManager, StorageConfig, Message

async def test_basic():
    # Initialize
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create user
    await storage.create_user("test_user", {"name": "Test User"})
    
    # Create session
    session_id = await storage.create_session("test_user", "Test Chat")
    
    # Add messages
    msg = Message(role="user", content="Hello!")
    await storage.add_message("test_user", session_id, msg)
    
    # Retrieve messages
    messages = await storage.get_messages("test_user", session_id)
    print(f"Retrieved {len(messages)} messages")

asyncio.run(test_basic())
```

### Testing Search Features

```python
from flatfile_chat_database import SearchQuery

async def test_search():
    # ... (setup storage and add data)
    
    # Basic search
    results = await storage.search_messages("user_id", "Python")
    
    # Advanced search with filters
    query = SearchQuery(
        query="programming",
        user_id="user_id",
        message_roles=["user"],
        min_relevance_score=0.5
    )
    results = await storage.advanced_search(query)
    
    # Entity extraction
    entities = await storage.extract_entities(
        "Check https://example.com and email test@example.com"
    )
    print(f"Extracted: {entities}")

asyncio.run(test_search())
```

### Testing Performance

```python
from flatfile_chat_database.benchmark import PerformanceBenchmark

async def test_performance():
    benchmark = PerformanceBenchmark(iterations=50)
    results = await benchmark.run_all_benchmarks(verbose=True)
    
    # Check if targets are met
    for op, target in results["performance_targets"].items():
        print(f"{op}: {'PASS' if target['passed'] else 'FAIL'}")

asyncio.run(test_performance())
```

## Integration Testing

### With Existing Chat System

1. **Replace Storage Layer**:
   ```python
   # Before: Scattered storage operations
   # After: Centralized storage
   storage = StorageManager(config)
   await storage.initialize()
   ```

2. **Migration from Existing Data**:
   ```python
   from flatfile_chat_database.migration import DatabaseImporter
   
   importer = DatabaseImporter(storage)
   stats = await importer.import_from_json("existing_data.json")
   print(f"Imported: {stats.total_messages} messages")
   ```

## Performance Benchmarks

Expected performance (based on tests):
- **Session creation**: ~1-2ms (target: <10ms) ✓
- **Message append**: ~1-2ms (target: <5ms) ✓
- **Message retrieval (100)**: <1ms (target: <50ms) ✓
- **Basic search**: ~10ms (target: <200ms) ✓
- **Document upload (1MB)**: ~3ms (target: <100ms) ✓

## Debugging

### Enable Verbose Logging

```python
# In your test code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Storage Structure

```bash
# View storage structure
tree ./demo_chat_data/ -L 3

# Check file contents
cat ./demo_chat_data/alice/*/messages.jsonl | jq .
```

### Common Issues

1. **Permission Errors**: Ensure write permissions on storage directory
2. **File Not Found**: Check if parent directories are created (config.create_parent_directories)
3. **Performance Issues**: Run benchmarks to identify bottlenecks

## Load Testing

For production load testing:

```python
async def load_test():
    # Create many users and sessions
    for i in range(100):
        user_id = f"user_{i}"
        await storage.create_user(user_id)
        
        # Multiple sessions per user
        for j in range(10):
            session_id = await storage.create_session(user_id)
            
            # Many messages per session
            for k in range(100):
                msg = Message(role="user", content=f"Message {k}")
                await storage.add_message(user_id, session_id, msg)
    
    # Test search performance with large dataset
    results = await storage.search_messages("user_50", "Message")
    print(f"Search returned {len(results)} results")
```

## Continuous Testing

For CI/CD integration:

```bash
#!/bin/bash
# run_tests.sh

echo "Running Flatfile Chat Database Tests..."

# Basic functionality
python3 test_flatfile.py || exit 1

# Advanced features
python3 test_advanced_search.py || exit 1

# Production features
python3 test_phase4_features.py || exit 1

echo "All tests passed!"
```