#!/usr/bin/env python3
"""
Test runner for integration tests.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the flatfile_chat_database package
from flatfile_chat_database.storage import StorageManager
from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.models import (
    Message, Session, SituationalContext, Document,
    UserProfile, Persona, PanelMessage, PanelInsight
)


async def create_test_storage():
    """Create storage manager for testing"""
    temp_dir = tempfile.mkdtemp()
    config = StorageConfig(
        storage_base_path=temp_dir,
        max_messages_per_session=1000,
        max_document_size_bytes=10_000_000  # 10MB for tests
    )
    
    manager = StorageManager(config=config)
    await manager.initialize()
    
    return manager, temp_dir


async def test_user_management():
    """Test user management functionality"""
    print("\n=== Testing User Management ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        # Test 1: Create and get user
        print("Test 1: Create and get user...", end=" ")
        user_id = "test_user"
        profile = {
            "username": "Test User",
            "preferences": {"theme": "dark"},
            "metadata": {"source": "test"}
        }
        
        result = await manager.create_user(user_id, profile)
        assert result is True
        
        retrieved = await manager.get_user_profile(user_id)
        assert retrieved is not None
        assert retrieved["user_id"] == user_id
        assert retrieved["username"] == "Test User"
        print("✓ PASSED")
        
        # Test 2: Update user profile
        print("Test 2: Update user profile...", end=" ")
        updates = {
            "username": "Updated Name",
            "preferences": {"language": "es"}
        }
        result = await manager.update_user_profile(user_id, updates)
        assert result is True
        
        profile = await manager.get_user_profile(user_id)
        assert profile["username"] == "Updated Name"
        assert profile["preferences"]["language"] == "es"
        print("✓ PASSED")
        
        # Test 3: List users
        print("Test 3: List users...", end=" ")
        users = ["alice", "bob", "charlie"]
        for uid in users:
            await manager.create_user(uid)
        
        user_list = await manager.list_users()
        assert len(user_list) >= 3
        for uid in users:
            assert uid in user_list
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_session_management():
    """Test session management functionality"""
    print("\n=== Testing Session Management ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        # Test 1: Create and get session
        print("Test 1: Create and get session...", end=" ")
        user_id = "session_test_user"
        await manager.create_user(user_id)
        
        session_id = await manager.create_session(user_id, "Test Session")
        assert session_id != ""
        assert session_id.startswith("chat_session_")
        
        session = await manager.get_session(user_id, session_id)
        assert session is not None
        assert session.id == session_id
        assert session.title == "Test Session"
        print("✓ PASSED")
        
        # Test 2: Update session
        print("Test 2: Update session...", end=" ")
        updates = {
            "title": "Updated Title",
            "metadata": {"tags": ["important", "work"]}
        }
        result = await manager.update_session(user_id, session_id, updates)
        assert result is True
        
        session = await manager.get_session(user_id, session_id)
        assert session.title == "Updated Title"
        assert session.metadata["tags"] == ["important", "work"]
        print("✓ PASSED")
        
        # Test 3: List sessions
        print("Test 3: List sessions...", end=" ")
        session_ids = []
        for i in range(5):
            sid = await manager.create_session(user_id, f"Session {i}")
            session_ids.append(sid)
            await asyncio.sleep(0.01)
        
        sessions = await manager.list_sessions(user_id)
        assert len(sessions) >= 5
        
        sessions = await manager.list_sessions(user_id, limit=3)
        assert len(sessions) == 3
        print("✓ PASSED")
        
        # Test 4: Delete session
        print("Test 4: Delete session...", end=" ")
        result = await manager.delete_session(user_id, session_id)
        assert result is True
        
        session = await manager.get_session(user_id, session_id)
        assert session is None
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_message_management():
    """Test message functionality"""
    print("\n=== Testing Message Management ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        # Test 1: Add and get messages
        print("Test 1: Add and get messages...", end=" ")
        user_id = "message_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="What can you help me with?"),
            Message(role="assistant", content="I can help with many things!")
        ]
        
        for msg in messages:
            result = await manager.add_message(user_id, session_id, msg)
            assert result is True
        
        retrieved = await manager.get_messages(user_id, session_id)
        assert len(retrieved) == 4
        
        for i, msg in enumerate(retrieved):
            assert msg.role == messages[i].role
            assert msg.content == messages[i].content
        print("✓ PASSED")
        
        # Test 2: Message pagination
        print("Test 2: Message pagination...", end=" ")
        user_id = "pagination_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        for i in range(20):
            msg = Message(role="user" if i % 2 == 0 else "assistant", 
                         content=f"Message {i}")
            await manager.add_message(user_id, session_id, msg)
        
        page1 = await manager.get_messages(user_id, session_id, limit=10)
        assert len(page1) == 10
        assert page1[0].content == "Message 0"
        
        page2 = await manager.get_messages(user_id, session_id, limit=10, offset=10)
        assert len(page2) == 10
        assert page2[0].content == "Message 10"
        print("✓ PASSED")
        
        # Test 3: Search messages
        print("Test 3: Search messages...", end=" ")
        user_id = "search_test_user"
        await manager.create_user(user_id)
        
        session1 = await manager.create_session(user_id)
        await manager.add_message(user_id, session1, 
                                Message(role="user", content="Tell me about Python"))
        await manager.add_message(user_id, session1,
                                Message(role="assistant", content="Python is a programming language"))
        
        session2 = await manager.create_session(user_id)
        await manager.add_message(user_id, session2,
                                Message(role="user", content="What is JavaScript?"))
        
        results = await manager.search_messages(user_id, "Python")
        assert len(results) == 2
        
        results = await manager.search_messages(user_id, "JavaScript", session_id=session2)
        assert len(results) == 1
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_document_management():
    """Test document handling"""
    print("\n=== Testing Document Management ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        # Test 1: Save and get document
        print("Test 1: Save and get document...", end=" ")
        user_id = "doc_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        content = b"This is a test document content"
        doc_id = await manager.save_document(
            user_id, session_id, "test.txt", content,
            metadata={"description": "Test file"}
        )
        assert doc_id != ""
        
        retrieved = await manager.get_document(user_id, session_id, doc_id)
        assert retrieved == content
        
        docs = await manager.list_documents(user_id, session_id)
        assert len(docs) == 1
        assert docs[0].original_name == "test.txt"
        assert docs[0].size == len(content)
        print("✓ PASSED")
        
        # Test 2: Document analysis
        print("Test 2: Document analysis...", end=" ")
        user_id = "analysis_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        doc_id = await manager.save_document(
            user_id, session_id, "analyze.pdf", b"PDF content"
        )
        
        analysis = {
            "summary": "This is a PDF document",
            "word_count": 100,
            "topics": ["test", "document"]
        }
        result = await manager.update_document_analysis(
            user_id, session_id, doc_id, analysis
        )
        assert result is True
        
        docs = await manager.list_documents(user_id, session_id)
        assert docs[0].analysis["analysis"]["results"]["summary"] == "This is a PDF document"
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_context_management():
    """Test situational context functionality"""
    print("\n=== Testing Context Management ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        # Test 1: Context operations
        print("Test 1: Context operations...", end=" ")
        user_id = "context_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        context = SituationalContext(
            summary="Discussing Python programming",
            key_points=["Functions", "Classes", "Modules"],
            entities={"languages": ["Python"], "concepts": ["OOP"]},
            confidence=0.9
        )
        
        result = await manager.update_context(user_id, session_id, context)
        assert result is True
        
        retrieved = await manager.get_context(user_id, session_id)
        assert retrieved is not None
        assert retrieved.summary == context.summary
        assert retrieved.confidence == 0.9
        print("✓ PASSED")
        
        # Test 2: Context history
        print("Test 2: Context history...", end=" ")
        user_id = "history_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        contexts = []
        for i in range(3):
            ctx = SituationalContext(
                summary=f"Context {i}",
                key_points=[f"Point {i}"],
                entities={"index": [str(i)]},
                confidence=0.5 + i * 0.1
            )
            contexts.append(ctx)
            await manager.save_context_snapshot(user_id, session_id, ctx)
            await asyncio.sleep(0.01)
        
        history = await manager.get_context_history(user_id, session_id)
        assert len(history) == 3
        assert history[0].summary == "Context 2"
        assert history[2].summary == "Context 0"
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_complete_workflow():
    """Test complete usage workflow"""
    print("\n=== Testing Complete Workflow ===")
    manager, temp_dir = await create_test_storage()
    
    try:
        print("Test: Full chat workflow...", end=" ")
        
        # 1. Create user
        user_id = "workflow_user"
        await manager.create_user(user_id, {
            "username": "Test User",
            "preferences": {"ai_model": "gpt-4"}
        })
        
        # 2. Create session
        session_id = await manager.create_session(user_id, "Python Help Session")
        
        # 3. Add messages
        await manager.add_message(user_id, session_id,
                                Message(role="user", content="How do I read a file in Python?"))
        await manager.add_message(user_id, session_id,
                                Message(role="assistant", content="You can use the open() function..."))
        
        # 4. Upload a document
        doc_id = await manager.save_document(
            user_id, session_id, "example.txt",
            b"with open('file.txt', 'r') as f:\n    content = f.read()"
        )
        
        # 5. Update context
        context = SituationalContext(
            summary="User learning file I/O in Python",
            key_points=["open() function", "context managers", "file modes"],
            entities={"topics": ["file I/O"], "language": ["Python"]},
            confidence=0.95
        )
        await manager.update_context(user_id, session_id, context)
        
        # 6. Verify everything
        session = await manager.get_session(user_id, session_id)
        assert session.message_count == 2
        
        messages = await manager.get_messages(user_id, session_id)
        assert len(messages) == 2
        
        docs = await manager.list_documents(user_id, session_id)
        assert len(docs) == 1
        
        current_context = await manager.get_context(user_id, session_id)
        assert current_context.confidence == 0.95
        
        print("✓ PASSED")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all tests"""
    print("Running Flatfile Chat Database Tests")
    print("=" * 50)
    
    tests = [
        test_user_management,
        test_session_management,
        test_message_management,
        test_document_management,
        test_context_management,
        test_complete_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {passed} test suites passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)