# Flatfile Chat Database - Testing Guide

## Quick Start Testing

### 1. Basic Test - Verify Installation
```bash
cd /home/markly2/claude_code

# Run the quick demo to see everything working
python3 flatfile_chat_database/quick_test_demo.py
```

### 2. Interactive CLI Demo
```bash
# Run the interactive chat interface
python3 interactive_chat_demo.py

# With options:
python3 interactive_chat_demo.py --theme colorful
python3 interactive_chat_demo.py --storage-path ./my_test_data
python3 interactive_chat_demo.py --no-ai  # Disable AI simulation
```

## Running Test Suites

### Navigate to the test directory first:
```bash
cd /home/markly2/claude_code
```

### 1. Basic Functionality Tests
```bash
# Test core CRUD operations
python3 flatfile_chat_database/tests/test_flatfile.py
```
Tests:
- User creation and profile management
- Session lifecycle (create, update, delete)
- Message storage and retrieval
- Document uploads
- Basic search

### 2. Advanced Search Tests
```bash
# Test sophisticated search features
python3 flatfile_chat_database/tests/test_advanced_search.py
```
Tests:
- Text search with relevance scoring
- Entity extraction (URLs, emails, code)
- Time-range queries
- Role-based filtering
- Cross-session search

### 3. Production Features Tests
```bash
# Test streaming, compression, migration
python3 flatfile_chat_database/tests/test_phase4_features.py
```
Tests:
- Message streaming for large sessions
- Compression (GZIP/ZLIB)
- Export to SQLite
- Import/Export JSON
- Performance benchmarks

### 4. Complete System Test
```bash
# Comprehensive test of all features
python3 flatfile_chat_database/tests/test_complete_system.py
```

## Manual Testing Examples

### Basic Usage Test
```python
import asyncio
from flatfile_chat_database import StorageManager, StorageConfig, Message

async def test_basic_usage():
    # Initialize storage
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create a user
    await storage.create_user("test_user", {
        "name": "Test User",
        "email": "test@example.com"
    })
    
    # Create a session
    session_id = await storage.create_session("test_user", "My Test Chat")
    print(f"Created session: {session_id}")
    
    # Add messages
    msg1 = Message(role="user", content="Hello, how are you?")
    msg2 = Message(role="assistant", content="I'm doing well, thank you!")
    
    await storage.add_message("test_user", session_id, msg1)
    await storage.add_message("test_user", session_id, msg2)
    
    # Retrieve messages
    messages = await storage.get_messages("test_user", session_id)
    print(f"Retrieved {len(messages)} messages")
    
    for msg in messages:
        print(f"[{msg.role}]: {msg.content}")

# Run the test
asyncio.run(test_basic_usage())
```

### Search Testing
```python
async def test_search():
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Add test data
    user_id = "search_test_user"
    await storage.create_user(user_id)
    
    # Create multiple sessions with different content
    sessions = [
        ("Python Tutorial", ["How to use lists in Python", "Python dictionaries"]),
        ("JavaScript Guide", ["JavaScript arrays", "React components"]),
        ("Data Science", ["Python pandas tutorial", "Machine learning basics"])
    ]
    
    for title, messages in sessions:
        session_id = await storage.create_session(user_id, title)
        for content in messages:
            msg = Message(role="user", content=content)
            await storage.add_message(user_id, session_id, msg)
    
    # Test search
    from flatfile_chat_database import SearchQuery
    
    # Search for "Python"
    query = SearchQuery(
        query="Python",
        user_id=user_id,
        include_context=True
    )
    results = await storage.advanced_search(query)
    
    print(f"Found {len(results)} results for 'Python':")
    for result in results:
        print(f"- {result.content} (score: {result.relevance_score:.2f})")

asyncio.run(test_search())
```

### Document Upload Testing
```python
async def test_documents():
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    user_id = "doc_test_user"
    await storage.create_user(user_id)
    session_id = await storage.create_session(user_id, "Document Test")
    
    # Upload a text document
    doc_content = b"# Test Document\n\nThis is a test document."
    doc_id = await storage.save_document(
        user_id, session_id, "test.md", doc_content,
        metadata={"type": "markdown", "size": len(doc_content)}
    )
    print(f"Uploaded document: {doc_id}")
    
    # List documents
    docs = await storage.list_documents(user_id, session_id)
    print(f"Found {len(docs)} documents:")
    for doc in docs:
        print(f"- {doc.original_name} ({doc.size} bytes)")
    
    # Retrieve document
    content, metadata = await storage.get_document(user_id, session_id, doc_id)
    print(f"Retrieved document: {len(content)} bytes")

asyncio.run(test_documents())
```

### Performance Testing
```python
async def test_performance():
    from flatfile_chat_database.benchmark import PerformanceBenchmark
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(iterations=100)
    results = await benchmark.run_all_benchmarks(verbose=True)
    
    # Check results
    print("\nPerformance Results:")
    print("-" * 40)
    for operation, metrics in results["metrics"].items():
        avg_time = metrics["average_ms"]
        target = results["performance_targets"][operation]["target_ms"]
        passed = "✓" if results["performance_targets"][operation]["passed"] else "✗"
        print(f"{operation}: {avg_time:.2f}ms (target: {target}ms) {passed}")

asyncio.run(test_performance())
```

## Interactive CLI Demo Commands

When running `interactive_chat_demo.py`, try these commands:

```bash
# Navigation
/sessions       # List all your sessions
/new           # Create a new session
/switch 2      # Switch to session #2

# Messages
/history       # Show recent messages
/history 20    # Show last 20 messages
/search Python # Search for "Python" in messages

# Data Management
/upload /path/to/document.pdf  # Upload a document
/docs          # List documents in session
/export        # Export current session to JSON
/stats         # Show user statistics

# Utility
/help          # Show all commands
/context       # Show conversation context
/clear         # Clear screen
/theme dark    # Change color theme
/exit          # Exit the demo
```

## Testing Scenarios

### 1. Multi-User Test
```python
async def test_multi_user():
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create multiple users
    users = ["alice", "bob", "charlie"]
    for user in users:
        await storage.create_user(user, {"name": user.title()})
        
        # Each user gets multiple sessions
        for i in range(3):
            session_id = await storage.create_session(user, f"Chat {i+1}")
            
            # Add some messages
            for j in range(5):
                msg = Message(role="user", content=f"Message {j+1} from {user}")
                await storage.add_message(user, session_id, msg)
    
    # Verify isolation
    alice_sessions = await storage.list_sessions("alice")
    bob_sessions = await storage.list_sessions("bob")
    
    print(f"Alice has {len(alice_sessions)} sessions")
    print(f"Bob has {len(bob_sessions)} sessions")

asyncio.run(test_multi_user())
```

### 2. Large Session Test
```python
async def test_large_session():
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    user_id = "stress_test"
    await storage.create_user(user_id)
    session_id = await storage.create_session(user_id, "Large Session")
    
    # Add many messages
    print("Adding 1000 messages...")
    for i in range(1000):
        msg = Message(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i+1}: " + "x" * 100
        )
        await storage.add_message(user_id, session_id, msg)
        
        if (i + 1) % 100 == 0:
            print(f"Added {i+1} messages...")
    
    # Test retrieval with pagination
    print("\nTesting pagination...")
    page1 = await storage.get_messages(user_id, session_id, limit=50, offset=0)
    page2 = await storage.get_messages(user_id, session_id, limit=50, offset=50)
    
    print(f"Page 1: {len(page1)} messages")
    print(f"Page 2: {len(page2)} messages")
    
    # Test streaming
    print("\nTesting streaming...")
    from flatfile_chat_database.streaming import MessageStreamer
    streamer = MessageStreamer(storage, batch_size=100)
    
    batch_count = 0
    async for batch in streamer.stream_messages(user_id, session_id):
        batch_count += 1
        print(f"Received batch {batch_count}: {len(batch)} messages")

asyncio.run(test_large_session())
```

### 3. Migration Test
```python
async def test_migration():
    config = StorageConfig(storage_base_path="./test_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create test data
    user_id = "migration_test"
    await storage.create_user(user_id)
    session_id = await storage.create_session(user_id, "Test Session")
    
    for i in range(10):
        msg = Message(role="user", content=f"Test message {i+1}")
        await storage.add_message(user_id, session_id, msg)
    
    # Export to SQLite
    from flatfile_chat_database.migration import FlatfileExporter, SQLiteAdapter
    
    exporter = FlatfileExporter(storage)
    db_path = "./test_export.db"
    
    adapter = SQLiteAdapter(db_path)
    await adapter.initialize()
    
    stats = await exporter.export_to_database(adapter)
    print(f"Exported to SQLite: {stats}")
    
    # Export to JSON
    json_stats = await exporter.export_to_json("./test_export.json")
    print(f"Exported to JSON: {json_stats}")

asyncio.run(test_migration())
```

## Debugging Tips

### 1. Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your tests to see detailed logs
```

### 2. Inspect Storage Structure
```bash
# View the storage directory structure
tree ./test_data -L 3

# Check specific user data
ls -la ./test_data/alice/

# View raw message data
cat ./test_data/alice/*/messages.jsonl | jq .
```

### 3. Check File Permissions
```bash
# Ensure write permissions
chmod -R u+w ./test_data

# Check ownership
ls -la ./test_data
```

## Common Issues and Solutions

### 1. Import Errors
```bash
# Always run from the parent directory
cd /home/markly2/claude_code
python3 flatfile_chat_database/tests/test_flatfile.py
```

### 2. Permission Denied
```python
# Set proper permissions in config
config = StorageConfig(
    storage_base_path="./test_data",
    file_permissions=0o644,
    directory_permissions=0o755
)
```

### 3. Storage Full
```python
# Check available space
import shutil
usage = shutil.disk_usage("./test_data")
print(f"Free space: {usage.free / (1024**3):.2f} GB")
```

## Continuous Testing Script

Create a script to run all tests:

```bash
#!/bin/bash
# run_all_tests.sh

echo "Running Flatfile Chat Database Tests..."
cd /home/markly2/claude_code

# Basic tests
echo -e "\n1. Running basic tests..."
python3 flatfile_chat_database/tests/test_flatfile.py || exit 1

# Advanced search
echo -e "\n2. Running search tests..."
python3 flatfile_chat_database/tests/test_advanced_search.py || exit 1

# Production features
echo -e "\n3. Running production tests..."
python3 flatfile_chat_database/tests/test_phase4_features.py || exit 1

# Performance
echo -e "\n4. Running benchmarks..."
python3 -c "
import asyncio
from flatfile_chat_database.benchmark import PerformanceBenchmark
async def run():
    b = PerformanceBenchmark(iterations=50)
    await b.run_all_benchmarks(verbose=False)
asyncio.run(run())
"

echo -e "\n✅ All tests passed!"
```

Make it executable:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```