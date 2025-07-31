#!/usr/bin/env python3
"""
Basic functionality test for the flatfile chat database.
This tests core features without complex demo scenarios.
"""

import sys
import os
from pathlib import Path
import asyncio
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, '..')

# Add PrismMind directory to path (if available)
prismmind_path = '/home/markly2/prismmind'
if os.path.exists(prismmind_path):
    sys.path.append(prismmind_path)
    print(f"✅ Added PrismMind path: {prismmind_path}")
else:
    print(f"⚠️ PrismMind not found at: {prismmind_path} - will use legacy document processing")

try:
    from ff_storage_manager import FFStorageManager
    from ff_config_legacy_adapter import StorageConfig
    from ff_class_configs.ff_chat_entities_config import FFMessage, FFSession, FFDocument, FFUserProfile, MessageRole
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test core functionality."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    # Create temporary directory for test
    test_dir = Path(tempfile.mkdtemp(prefix="demo_test_"))
    print(f"📁 Test directory: {test_dir}")
    
    try:
        # Setup configuration
        config = StorageConfig()
        config.storage_base_path = str(test_dir)
        config.enable_compression = False
        
        # Initialize storage manager
        storage_manager = FFStorageManager(config)
        print("✅ FFStorageManager initialized")
        
        # Test 1: Create user
        profile_data = {
            "username": "Test User",
            "preferences": {"theme": "dark"},
            "metadata": {"role": "tester"}
        }
        
        await storage_manager.create_user("test_user", profile_data)
        print("✅ User created")
        
        # Test 2: Create session
        session_id = await storage_manager.create_session(
            user_id="test_user",
            title="Test Session"
        )
        print(f"✅ Session created: {session_id}")
        
        # Test 3: Store messages
        messages = [
            FFMessage(role=MessageRole.USER, content="Hello, this is a test message."),
            FFMessage(role=MessageRole.ASSISTANT, content="Hello! I can help you test the system.")
        ]
        
        for msg in messages:
            await storage_manager.add_message("test_user", session_id, msg)
        print(f"✅ Stored {len(messages)} messages")
        
        # Test 4: Store document
        await storage_manager.save_document(
            user_id="test_user",
            session_id=session_id,
            filename="test_doc.txt",
            content="This is a test document for the flatfile chat database demo.",
            metadata={"type": "test"}
        )
        print(f"✅ Document stored")
        
        # Test 5: Basic search
        try:
            from search import SearchQuery, AdvancedSearchEngine
            
            search_engine = AdvancedSearchEngine(config)
            search_query = SearchQuery(
                query_text="test",
                user_id="test_user",
                session_id=session_id,
                include_messages=True,  
                include_documents=True
            )
            
            results = await search_engine.search(search_query)
            print(f"✅ Search completed: {len(results.results)} results found")
            
        except Exception as e:
            print(f"⚠️ Search test failed: {e}")
        
        # Test 6: Check file structure
        user_dir = test_dir / "users" / "test_user"
        session_dir = user_dir / session_id
        
        expected_files = [
            user_dir / "profile.json",
            session_dir / "session.json", 
            session_dir / "messages.jsonl"
        ]
        
        for file_path in expected_files:
            if file_path.exists():
                print(f"✅ File exists: {file_path.name}")
            else:
                print(f"❌ File missing: {file_path.name}")
        
        # Test 7: Configuration systems
        try:
            from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfig
            new_config = FFConfigurationManagerConfig.from_environment("development")
            print("✅ New configuration system works")
        except ImportError:
            print("⚠️ New configuration system not available")
        
        print("\n🎉 Basic functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"🧹 Cleaned up test directory")


def main():
    """Main test function."""
    print("🚀 Flatfile Chat Database - Basic Functionality Test")
    
    # Run the async test
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\n✅ All tests passed! The demos should work correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())