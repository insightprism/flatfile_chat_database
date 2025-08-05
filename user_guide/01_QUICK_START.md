# Quick Start Guide

Get up and running with the Flatfile Chat Database in under 10 minutes!

## 🚀 1-Minute Setup

### Prerequisites Check
```bash
python --version  # Should be 3.8+
pip --version     # Should be available
```

### Installation
```bash
# Clone or download the project
git clone <repository-url>
cd flatfile_chat_database_v2

# Install dependencies
pip install -r requirements.txt
```

### Your First Chat Database
```python
import asyncio
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

async def hello_world():
    # 1. Load configuration
    config = load_config()
    
    # 2. Create and initialize storage
    storage = FFStorageManager(config)
    await storage.initialize()
    
    # 3. Create a user
    user_created = await storage.create_user("alice", {
        "name": "Alice Smith",
        "email": "alice@example.com"
    })
    print(f"User created: {user_created}")
    
    # 4. Create a chat session
    session_id = await storage.create_session("alice", "Getting Started Chat")
    print(f"Session created: {session_id}")
    
    # 5. Add messages to the conversation
    messages = [
        FFMessageDTO(role=MessageRole.USER, content="Hello! I'm new here."),
        FFMessageDTO(role=MessageRole.ASSISTANT, content="Welcome! I'm here to help."),
        FFMessageDTO(role=MessageRole.USER, content="How does this chat system work?"),
        FFMessageDTO(role=MessageRole.ASSISTANT, content="It stores conversations persistently and allows you to search through them!")
    ]
    
    for message in messages:
        await storage.add_message("alice", session_id, message)
        print(f"Added message: {message.content[:30]}...")
    
    # 6. Retrieve the conversation
    retrieved_messages = await storage.get_all_messages("alice", session_id)
    print(f"\nRetrieved {len(retrieved_messages)} messages:")
    
    for i, msg in enumerate(retrieved_messages, 1):
        print(f"{i}. [{msg.role.value}]: {msg.content}")
    
    print("\n🎉 Success! Your first chat database is working!")

# Run it!
if __name__ == "__main__":
    asyncio.run(hello_world())
```

Save this as `quick_start.py` and run:
```bash
python quick_start.py
```

## 🎯 Core Concepts (2 minutes)

### 1. **Users** 👤
Users represent individuals in your system. Each user has:
- Unique user ID
- Profile information (name, preferences, etc.)
- Multiple chat sessions

### 2. **Sessions** 💬
Sessions are individual conversations. Each session contains:
- Unique session ID
- Title/name for the conversation
- Messages in chronological order
- Optional metadata

### 3. **Messages** 📝
Messages are the basic communication units:
- Role (user, assistant, system)
- Content (the actual message text)
- Timestamp and metadata
- Unique message ID

### 4. **Storage Manager** 🗄️
The main interface for all operations:
- Manages users, sessions, and messages
- Handles file system operations
- Provides search and retrieval capabilities

## 🔧 Essential Operations

### Creating Users
```python
# Simple user creation
await storage.create_user("bob")

# User with profile information
await storage.create_user("charlie", {
    "name": "Charlie Brown",
    "role": "admin",
    "preferences": {"theme": "dark"}
})

# Check if user exists
exists = await storage.user_exists("alice")
```

### Managing Sessions
```python
# Create a new session
session_id = await storage.create_session("alice", "Project Discussion")

# Get session details
session = await storage.get_session("alice", session_id)
print(session.title, session.created_at, session.message_count)

# List all sessions for a user
sessions = await storage.list_sessions("alice")
```

### Working with Messages
```python
# Create different types of messages
user_msg = FFMessageDTO(
    role=MessageRole.USER,
    content="What's the weather like?"
)

assistant_msg = FFMessageDTO(
    role=MessageRole.ASSISTANT,
    content="I'd be happy to help with weather information!"
)

system_msg = FFMessageDTO(
    role=MessageRole.SYSTEM,
    content="Weather service connected successfully."
)

# Add messages to session
await storage.add_message("alice", session_id, user_msg)
await storage.add_message("alice", session_id, assistant_msg)
await storage.add_message("alice", session_id, system_msg)

# Retrieve messages
all_messages = await storage.get_all_messages("alice", session_id)
recent_messages = await storage.get_messages("alice", session_id, limit=5)
```

## 🔍 Basic Search

```python
# Search across all user's messages
results = await storage.search_messages("alice", "weather")

# Search with filters
results = await storage.search_messages(
    "alice", 
    "project discussion",
    session_ids=[session_id],
    limit=10
)

print(f"Found {len(results)} matching messages")
for result in results:
    print(f"- {result.content[:50]}...")
```

## 📁 File Structure (What Gets Created)

After running the quick start, you'll see:
```
your_project/
├── data/                          # Default storage location
│   └── users/
│       ├── alice/
│       │   ├── profile.json       # User profile
│       │   └── chat_session_*/    # Session directories
│       │       ├── session.json   # Session metadata
│       │       └── messages.jsonl # Message history
│       ├── bob/
│       └── charlie/
└── quick_start.py                 # Your test script
```

## 🎨 Configuration Basics

### Default Configuration
The system works out-of-the-box with defaults:
```python
config = load_config()  # Uses default settings
```

### Custom Storage Location
```python
config = load_config()
config.storage.base_path = "/my/custom/path"
storage = FFStorageManager(config)
```

### Environment-Specific Configs
```python
# Use different configurations
dev_config = load_config("development")
test_config = load_config("test")
prod_config = load_config("production")
```

## 🧪 Verify Your Setup

Run this verification script to ensure everything works:

```python
import asyncio
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config

async def verify_setup():
    """Verify that the system is working correctly."""
    try:
        config = load_config()
        storage = FFStorageManager(config)
        await storage.initialize()
        
        # Test user operations
        await storage.create_user("test_user")
        exists = await storage.user_exists("test_user")
        assert exists, "User creation failed"
        
        # Test session operations
        session_id = await storage.create_session("test_user", "Test Session")
        assert session_id, "Session creation failed"
        
        # Test message operations
        from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
        message = FFMessageDTO(role=MessageRole.USER, content="Test message")
        success = await storage.add_message("test_user", session_id, message)
        assert success, "Message addition failed"
        
        # Test retrieval
        messages = await storage.get_all_messages("test_user", session_id)
        assert len(messages) == 1, "Message retrieval failed"
        
        print("✅ All tests passed! Your setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(verify_setup())
```

## 🚨 Common Quick Start Issues

### Issue: Import Errors
```
ModuleNotFoundError: No module named 'ff_storage_manager'
```
**Solution**: Make sure you're in the project directory and have installed dependencies:
```bash
cd flatfile_chat_database_v2
pip install -r requirements.txt
```

### Issue: Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Ensure the storage directory is writable:
```bash
chmod 755 data/  # Or use a different directory
```

### Issue: Path Not Found
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: The system will create directories automatically, but ensure the parent path exists.

## 🎯 Next Steps

Now that you have the basics working:

1. **Explore More Features**: Check out [Basic Usage](04_BASIC_USAGE.md) for detailed operations
2. **Customize Configuration**: See [Configuration Guide](03_CONFIGURATION.md)
3. **Add Documents**: Learn about document handling in [Advanced Features](05_ADVANCED_FEATURES.md)
4. **Set Up Search**: Enable full-text search capabilities
5. **Performance Tuning**: Optimize for your use case in [Performance Guide](08_PERFORMANCE.md)

## 💡 Quick Tips

- **Start Simple**: Begin with basic user/session/message operations
- **Use Async**: All operations are async - don't forget `await`
- **Check Returns**: Most operations return success/failure indicators
- **Explore Examples**: Check the `demo/` directory for more examples
- **Read Errors**: Error messages are designed to be helpful

## 🎉 Congratulations!

You've successfully set up and used the Flatfile Chat Database! The system is now ready for integration into your applications.

**Ready for more?** Continue to [Installation & Setup](02_INSTALLATION.md) for production deployment guidance, or jump to [Basic Usage](04_BASIC_USAGE.md) to explore more features.