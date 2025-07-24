#!/usr/bin/env python3
"""
Test the flatfile chat database from outside the package.
"""

import asyncio
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the package
from flatfile_chat_database import StorageManager, StorageConfig, Message


async def test_basic_functionality():
    """Test basic functionality"""
    print("Testing Flatfile Chat Database Basic Functionality")
    print("=" * 50)
    
    # Create storage manager with temp directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    config = StorageConfig(storage_base_path=temp_dir)
    manager = StorageManager(config=config)
    
    try:
        # Initialize
        print("1. Initializing storage...", end=" ")
        result = await manager.initialize()
        assert result is True
        print("✓ PASSED")
        
        # Create user
        print("2. Creating user...", end=" ")
        user_id = "test_user"
        result = await manager.create_user(user_id, {
            "username": "Test User",
            "preferences": {"theme": "dark"}
        })
        assert result is True
        print("✓ PASSED")
        
        # Get user profile
        print("3. Getting user profile...", end=" ")
        profile = await manager.get_user_profile(user_id)
        assert profile is not None
        assert profile["username"] == "Test User"
        print("✓ PASSED")
        
        # Create session
        print("4. Creating session...", end=" ")
        session_id = await manager.create_session(user_id, "Test Session")
        assert session_id != ""
        assert session_id.startswith("chat_session_")
        print("✓ PASSED")
        
        # Add messages
        print("5. Adding messages...", end=" ")
        messages = [
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well, thank you!")
        ]
        
        for msg in messages:
            result = await manager.add_message(user_id, session_id, msg)
            assert result is True
        print("✓ PASSED")
        
        # Get messages
        print("6. Getting messages...", end=" ")
        retrieved = await manager.get_messages(user_id, session_id)
        assert len(retrieved) == 4
        assert retrieved[0].content == "Hello!"
        assert retrieved[3].content == "I'm doing well, thank you!"
        print("✓ PASSED")
        
        # Save document
        print("7. Saving document...", end=" ")
        doc_content = b"This is a test document"
        doc_id = await manager.save_document(
            user_id, session_id, "test.txt", doc_content
        )
        assert doc_id != ""
        print("✓ PASSED")
        
        # Get document
        print("8. Getting document...", end=" ")
        retrieved_doc = await manager.get_document(user_id, session_id, doc_id)
        assert retrieved_doc == doc_content
        print("✓ PASSED")
        
        # List sessions
        print("9. Listing sessions...", end=" ")
        sessions = await manager.list_sessions(user_id)
        assert len(sessions) >= 1
        assert sessions[0].title == "Test Session"
        print("✓ PASSED")
        
        # Search messages
        print("10. Searching messages...", end=" ")
        results = await manager.search_messages(user_id, "Hello")
        assert len(results) >= 1
        assert "Hello" in results[0].content
        print("✓ PASSED")
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\n✗ FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_basic_functionality())
    sys.exit(exit_code)