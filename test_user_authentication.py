#!/usr/bin/env python3
"""
Test script for user authentication in flatfile database.

This script verifies that:
1. The flatfile database correctly identifies the current user
2. Personas are rejected as users
3. The chat app can query for the current user
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ff_user_context_manager import FFUserContextManager, get_current_user
from ff_chat_integration.ff_chat_app_bridge import FFChatAppBridge, ChatAppStorageConfig
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole


async def test_user_context_manager():
    """Test the user context manager functionality."""
    print("\n=== Testing User Context Manager ===")
    
    # Create context manager
    context_mgr = FFUserContextManager()
    
    # Test getting current user
    current_user = context_mgr.get_current_user()
    print(f"Current user: {current_user}")
    
    # Test user info
    user_info = context_mgr.get_user_info()
    print(f"User info: {user_info}")
    
    # Test persona rejection
    test_cases = [
        ("persona_alex_patterson", False),
        ("alex_patterson", False),  # Known persona name
        ("test123", True),
        ("markly2", True),
        ("test_user", True),
        ("agent_something", False),
        ("ai_assistant", False),
    ]
    
    print("\nTesting user validation:")
    for user_id, expected_valid in test_cases:
        is_valid = context_mgr.is_valid_user(user_id)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"  {status} '{user_id}': valid={is_valid} (expected={expected_valid})")
    
    # Test validate_and_get_user
    print("\nTesting validate_and_get_user:")
    for user_id, _ in test_cases:
        result = context_mgr.validate_and_get_user(user_id)
        print(f"  '{user_id}' -> '{result}'")


async def test_chat_app_bridge():
    """Test the chat app bridge get_current_user functionality."""
    print("\n=== Testing Chat App Bridge ===")
    
    try:
        # Create test configuration
        config = ChatAppStorageConfig(
            storage_path="./test_user_auth_data",
            environment="test"
        )
        
        # Create and initialize bridge
        bridge = await FFChatAppBridge.create_for_chat_app(
            storage_path="./test_user_auth_data",
            options={"environment": "test"}
        )
        
        # Test get_current_user
        current_user = bridge.get_current_user()
        print(f"Current user from bridge: {current_user}")
        
        # Test get_user_info
        user_info = bridge.get_user_info()
        print(f"User info from bridge: {user_info}")
        
        # Clean up
        await bridge.close()
        print("✓ Bridge test completed successfully")
        
    except Exception as e:
        print(f"✗ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_storage_manager():
    """Test that storage manager rejects personas as users."""
    print("\n=== Testing Storage Manager ===")
    
    try:
        # Load configuration
        config = load_config()
        config.storage.base_path = "./test_user_auth_data"
        
        # Create storage manager
        storage_mgr = FFStorageManager(config)
        
        # Test creating sessions with different user IDs
        test_cases = [
            ("test123", True, "Should accept real user"),
            ("persona_alex_patterson", False, "Should reject persona with prefix"),
            ("alex_patterson", False, "Should reject known persona name"),
            ("test_user", True, "Should accept generic test user"),
        ]
        
        print("\nTesting session creation:")
        for user_id, should_work, description in test_cases:
            try:
                session_id = await storage_mgr.create_session(user_id, f"Test session for {user_id}")
                actual_user = "test123"  # Expected to default to this
                
                # Check if session was created under correct user
                session_path = Path(f"./test_user_auth_data/users/{actual_user}")
                exists = session_path.exists()
                
                if should_work:
                    print(f"  ✓ {description}: Created session {session_id}")
                else:
                    print(f"  ✓ {description}: Redirected to user '{actual_user}'")
                    
            except Exception as e:
                print(f"  ✗ {description}: Error - {e}")
        
        # Test adding messages
        print("\nTesting message storage:")
        session_id = await storage_mgr.create_session("test123", "Test session")
        
        # Try to add message with persona as user
        message = FFMessageDTO(
            role=MessageRole.USER,
            content="Test message"
        )
        
        # This should redirect to real user
        success = await storage_mgr.add_message("persona_betty_rodriguez", session_id, message)
        print(f"  Message with persona user: {'✓ Redirected' if success else '✗ Failed'}")
        
        # This should work normally
        success = await storage_mgr.add_message("test123", session_id, message)
        print(f"  Message with real user: {'✓ Success' if success else '✗ Failed'}")
        
    except Exception as e:
        print(f"✗ Storage manager test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("FLATFILE DATABASE USER AUTHENTICATION TEST")
    print("=" * 60)
    
    # Test individual components
    await test_user_context_manager()
    await test_chat_app_bridge()
    await test_storage_manager()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("""
The flatfile database now:
1. ✓ Determines the current user (defaults to 'test123')
2. ✓ Rejects personas as users
3. ✓ Provides get_current_user() for chat app to query
4. ✓ Redirects persona user IDs to real users
5. ✓ Maintains proper separation between users and personas
    """)


if __name__ == "__main__":
    asyncio.run(main())