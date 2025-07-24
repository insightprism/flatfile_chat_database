# Quick Testing Checklist

## 🚀 Fastest Way to Test

```bash
cd /home/markly2/claude_code
python3 test_runner.py
# Select option 1 (Quick Demo)
```

## 📋 Testing Checklist

### 1. First Time Setup Test
```bash
# Run the quick demo - shows all features
python3 flatfile_chat_database/quick_test_demo.py
```

### 2. Interactive Testing
```bash
# Start the chat interface
python3 interactive_chat_demo.py

# In the chat, try:
# - Create a user (enter any username)
# - Send some messages
# - Type /help to see commands
# - Type /search <word> to search
# - Type /stats to see statistics
# - Type /exit to quit
```

### 3. Unit Tests (if you want to verify everything works)
```bash
# From /home/markly2/claude_code directory:

# Test basic operations
python3 flatfile_chat_database/tests/test_flatfile.py

# Test search features  
python3 flatfile_chat_database/tests/test_advanced_search.py

# Test advanced features
python3 flatfile_chat_database/tests/test_phase4_features.py
```

### 4. Quick Manual Test
```python
# Save this as test_manual.py and run it
import asyncio
from flatfile_chat_database import StorageManager, StorageConfig, Message

async def quick_test():
    # Setup
    storage = StorageManager(StorageConfig(storage_base_path="./test_data"))
    await storage.initialize()
    
    # Create user
    await storage.create_user("test_user")
    
    # Create session
    session_id = await storage.create_session("test_user", "Test Chat")
    
    # Add message
    msg = Message(role="user", content="Hello world!")
    await storage.add_message("test_user", session_id, msg)
    
    # Retrieve
    messages = await storage.get_messages("test_user", session_id)
    print(f"Success! Retrieved {len(messages)} messages")
    print(f"Message: {messages[0].content}")

asyncio.run(quick_test())
```

## 🎯 What to Look For

### ✅ Success Indicators:
- "All tests passed!" message
- No error messages
- Files created in `./demo_chat_data/` or `./test_data/`
- Can send and retrieve messages
- Search returns results

### ❌ Common Issues:
- **Import Error**: Run from `/home/markly2/claude_code` directory
- **Permission Denied**: Check folder permissions
- **File Not Found**: Storage directory will be created automatically

## 🔍 Check Results

After testing, you can inspect the created files:
```bash
# See what was created
ls -la ./demo_chat_data/

# View storage structure
tree ./demo_chat_data/ -L 3

# Check a user's data
ls -la ./demo_chat_data/alice/
```

## 💡 Quick Tips

1. **Fastest Test**: Run `test_runner.py` and choose option 1
2. **Most Interactive**: Run `interactive_chat_demo.py`  
3. **Most Thorough**: Run all unit tests
4. **Custom Storage Path**: Add `--storage-path ./my_data`

## 📝 Sample Test Commands for CLI

When in the interactive demo, try these:
```
/help              # Show all commands
/new               # Create new session  
/search Python     # Search for "Python"
/history 10        # Show last 10 messages
/stats             # Show your statistics
/sessions          # List all sessions
/export            # Export current session
/theme dark        # Change to dark theme
/exit              # Exit the demo
```