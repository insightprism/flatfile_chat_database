"""
Comprehensive integration tests for the FFStorageManager.

Tests all storage operations including user management, sessions, messages,
documents, contexts, and panels with real file system operations.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import sys

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFSituationalContextDTO,
    FFUserProfileDTO, MessageRole
)
from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend


class TestStorageManagerInitialization:
    """Test storage manager initialization and setup."""
    
    @pytest.mark.asyncio
    async def test_storage_manager_initialization(self, test_config, temp_dir):
        """Test storage manager initializes correctly."""
        backend = FFFlatfileStorageBackend(test_config)
        storage_manager = FFStorageManager(test_config, backend)
        
        # Should initialize successfully
        result = await storage_manager.initialize()
        assert result is True
        assert storage_manager._initialized is True
        
        # Should create base directories
        base_path = Path(test_config.storage.base_path)
        assert base_path.exists()
        assert base_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_storage_manager_double_initialization(self, storage_manager):
        """Test that double initialization is handled gracefully."""
        # First initialization
        result1 = await storage_manager.initialize()
        assert result1 is True
        
        # Second initialization should also succeed
        result2 = await storage_manager.initialize()
        assert result2 is True
    
    def test_storage_manager_configuration(self, storage_manager, test_config):
        """Test that storage manager uses provided configuration."""
        assert storage_manager.config == test_config
        assert str(storage_manager.base_path) == test_config.storage.base_path


class TestUserManagement:
    """Test user management operations."""
    
    @pytest.mark.asyncio
    async def test_create_user_basic(self, storage_manager):
        """Test basic user creation."""
        result = await storage_manager.create_user("test_user")
        assert result is True
        
        # User should exist
        exists = await storage_manager.user_exists("test_user")
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_create_user_with_profile(self, storage_manager):
        """Test user creation with profile data."""
        profile_data = {
            "username": "testuser123",
            "preferences": {"theme": "dark", "language": "en"},
            "metadata": {"created_via": "test", "test_flag": True}
        }
        
        result = await storage_manager.create_user("profile_user", profile_data)
        assert result is True
        
        # Retrieve and verify profile
        retrieved_profile = await storage_manager.get_user_profile("profile_user")
        assert retrieved_profile is not None
        assert retrieved_profile["username"] == "testuser123"
        assert retrieved_profile["preferences"]["theme"] == "dark"
        assert retrieved_profile["metadata"]["test_flag"] is True
    
    @pytest.mark.asyncio
    async def test_create_duplicate_user(self, storage_manager):
        """Test creating duplicate user fails gracefully."""
        # Create first user
        result1 = await storage_manager.create_user("duplicate_user")
        assert result1 is True
        
        # Attempt to create duplicate
        result2 = await storage_manager.create_user("duplicate_user")
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_update_user_profile(self, storage_manager):
        """Test updating user profile."""
        # Create user first
        await storage_manager.create_user("update_user", {"username": "original"})
        
        # Update profile
        updates = {
            "username": "updated",
            "preferences": {"new_pref": "value"},
            "metadata": {"updated": True}
        }
        
        result = await storage_manager.update_user_profile("update_user", updates)
        assert result is True
        
        # Verify updates
        profile = await storage_manager.get_user_profile("update_user")
        assert profile["username"] == "updated"
        assert profile["preferences"]["new_pref"] == "value"
        assert profile["metadata"]["updated"] is True
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_user(self, storage_manager):
        """Test updating nonexistent user fails."""
        result = await storage_manager.update_user_profile("nonexistent", {"username": "test"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_store_user_profile_dto(self, storage_manager):
        """Test storing user profile DTO directly."""
        profile = sample_user_profile("dto_user")
        
        result = await storage_manager.store_user_profile(profile)
        assert result is True
        
        # Verify stored profile
        retrieved = await storage_manager.get_user_profile("dto_user")
        assert retrieved is not None
        assert retrieved["user_id"] == "dto_user"
        assert retrieved["username"] == profile.username
    
    @pytest.mark.asyncio
    async def test_list_users(self, storage_manager, sample_users):
        """Test listing all users."""
        # Create multiple users
        for user in sample_users:
            await storage_manager.create_user(user.user_id, user.to_dict())
        
        # List users
        users = await storage_manager.list_users()
        
        # Should find all created users
        expected_user_ids = {user.user_id for user in sample_users}
        actual_user_ids = set(users)
        
        assert expected_user_ids.issubset(actual_user_ids)
    
    @pytest.mark.asyncio
    async def test_user_existence_check(self, storage_manager):
        """Test user existence checking."""
        # Non-existent user
        exists_before = await storage_manager.user_exists("existence_test")
        assert exists_before is False
        
        # Create user
        await storage_manager.create_user("existence_test")
        
        # Should exist now
        exists_after = await storage_manager.user_exists("existence_test")
        assert exists_after is True


class TestSessionManagement:
    """Test session management operations."""
    
    @pytest.mark.asyncio
    async def test_create_session_basic(self, storage_manager):
        """Test basic session creation."""
        # Create user first
        await storage_manager.create_user("session_user")
        
        session_id = await storage_manager.create_session("session_user", "Test Session")
        assert session_id != ""
        assert len(session_id) > 0
    
    @pytest.mark.asyncio
    async def test_create_session_auto_user(self, storage_manager):
        """Test session creation automatically creates user if needed."""
        session_id = await storage_manager.create_session("auto_user", "Auto Session")
        assert session_id != ""
        
        # User should have been created
        user_exists = await storage_manager.user_exists("auto_user")
        assert user_exists is True
    
    @pytest.mark.asyncio
    async def test_get_session_metadata(self, storage_manager):
        """Test retrieving session metadata."""
        await storage_manager.create_user("meta_user")
        session_id = await storage_manager.create_session("meta_user", "Metadata Session")
        
        session = await storage_manager.get_session("meta_user", session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.user_id == "meta_user"
        assert session.title == "Metadata Session"
        assert session.message_count == 0
    
    @pytest.mark.asyncio
    async def test_update_session_metadata(self, storage_manager):
        """Test updating session metadata."""
        await storage_manager.create_user("update_session_user")
        session_id = await storage_manager.create_session("update_session_user", "Original Title")
        
        # Update session
        updates = {
            "title": "Updated Title",
            "metadata": {"updated": True, "priority": "high"}
        }
        
        result = await storage_manager.update_session("update_session_user", session_id, updates)
        assert result is True
        
        # Verify updates
        session = await storage_manager.get_session("update_session_user", session_id)
        assert session.title == "Updated Title"
        assert session.metadata["updated"] is True
        assert session.metadata["priority"] == "high"
    
    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, storage_manager):
        """Test listing sessions with pagination."""
        await storage_manager.create_user("paginate_user")
        
        # Create multiple sessions
        session_ids = []
        for i in range(15):
            session_id = await storage_manager.create_session("paginate_user", f"Session {i}")
            session_ids.append(session_id)
        
        # Test pagination
        page1 = await storage_manager.list_sessions("paginate_user", limit=5, offset=0)
        assert len(page1) == 5
        
        page2 = await storage_manager.list_sessions("paginate_user", limit=5, offset=5)
        assert len(page2) == 5
        
        # Pages should be different
        page1_ids = {s.session_id for s in page1}
        page2_ids = {s.session_id for s in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_delete_session(self, storage_manager):
        """Test session deletion."""
        await storage_manager.create_user("delete_session_user")
        session_id = await storage_manager.create_session("delete_session_user", "Delete Me")
        
        # Verify session exists
        session = await storage_manager.get_session("delete_session_user", session_id)
        assert session is not None
        
        # Delete session
        result = await storage_manager.delete_session("delete_session_user", session_id)
        assert result is True
        
        # Verify session is gone
        session_after = await storage_manager.get_session("delete_session_user", session_id)
        assert session_after is None


class TestMessageManagement:
    """Test message management operations."""
    
    @pytest.mark.asyncio
    async def test_add_message_basic(self, storage_manager):
        """Test basic message addition."""
        await storage_manager.create_user("msg_user")
        session_id = await storage_manager.create_session("msg_user", "Message Session")
        
        message = sample_message(MessageRole.USER, "Hello, world!")
        result = await storage_manager.add_message("msg_user", session_id, message)
        assert result is True
        
        # Session message count should update
        session = await storage_manager.get_session("msg_user", session_id)
        assert session.message_count == 1
    
    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, storage_manager, sample_messages):
        """Test adding multiple messages."""
        await storage_manager.create_user("multi_msg_user")
        session_id = await storage_manager.create_session("multi_msg_user", "Multi Message Session")
        
        # Add all messages
        for message in sample_messages:
            result = await storage_manager.add_message("multi_msg_user", session_id, message)
            assert result is True
        
        # Verify message count
        session = await storage_manager.get_session("multi_msg_user", session_id)
        assert session.message_count == len(sample_messages)
    
    @pytest.mark.asyncio
    async def test_get_messages_pagination(self, storage_manager, test_data_generator):
        """Test retrieving messages with pagination."""
        await storage_manager.create_user("paginate_msg_user")
        session_id = await storage_manager.create_session("paginate_msg_user", "Paginated Messages")
        
        # Add many messages
        messages = test_data_generator.create_test_conversation(20)
        for message in messages:
            await storage_manager.add_message("paginate_msg_user", session_id, message)
        
        # Test pagination
        page1 = await storage_manager.get_messages("paginate_msg_user", session_id, limit=5, offset=0)
        assert len(page1) == 5
        
        page2 = await storage_manager.get_messages("paginate_msg_user", session_id, limit=5, offset=5) 
        assert len(page2) == 5
        
        # Should be different messages
        page1_ids = {m.id for m in page1}
        page2_ids = {m.id for m in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_get_all_messages(self, storage_manager, sample_messages):
        """Test retrieving all messages from session."""
        await storage_manager.create_user("all_msg_user")
        session_id = await storage_manager.create_session("all_msg_user", "All Messages")
        
        # Add messages
        for message in sample_messages:
            await storage_manager.add_message("all_msg_user", session_id, message)
        
        # Get all messages
        all_messages = await storage_manager.get_all_messages("all_msg_user", session_id)
        assert len(all_messages) == len(sample_messages)
        
        # Verify message content
        message_contents = {m.content for m in all_messages}
        expected_contents = {m.content for m in sample_messages}
        assert message_contents == expected_contents
    
    @pytest.mark.asyncio
    async def test_message_size_limit(self, storage_manager, test_data_generator):
        """Test message size limit enforcement."""
        await storage_manager.create_user("size_limit_user")
        session_id = await storage_manager.create_session("size_limit_user", "Size Limit Test")
        
        # Create message that exceeds size limit
        large_message = test_data_generator.create_large_message(
            size_bytes=storage_manager.config.storage.max_message_size_bytes + 1000
        )
        
        # Should fail to add
        result = await storage_manager.add_message("size_limit_user", session_id, large_message)
        assert result is False
        
        # Session message count should remain 0
        session = await storage_manager.get_session("size_limit_user", session_id)
        assert session.message_count == 0
    
    @pytest.mark.asyncio
    async def test_search_messages_single_session(self, storage_manager, sample_messages):
        """Test searching messages within a single session."""
        await storage_manager.create_user("search_user")
        session_id = await storage_manager.create_session("search_user", "Search Session")
        
        # Add messages with searchable content
        searchable_message = sample_message(MessageRole.USER, "Python programming is great")
        await storage_manager.add_message("search_user", session_id, searchable_message)
        
        other_message = sample_message(MessageRole.ASSISTANT, "Java is also nice")
        await storage_manager.add_message("search_user", session_id, other_message)
        
        # Search for Python
        results = await storage_manager.search_messages("search_user", "Python", session_id=session_id)
        
        assert len(results) == 1
        assert "Python" in results[0].content
    
    @pytest.mark.asyncio
    async def test_search_messages_multiple_sessions(self, storage_manager):
        """Test searching messages across multiple sessions."""
        await storage_manager.create_user("multi_search_user")
        
        # Create multiple sessions with searchable content
        session1_id = await storage_manager.create_session("multi_search_user", "Session 1")
        message1 = sample_message(MessageRole.USER, "Testing search functionality")
        await storage_manager.add_message("multi_search_user", session1_id, message1)
        
        session2_id = await storage_manager.create_session("multi_search_user", "Session 2")  
        message2 = sample_message(MessageRole.USER, "More testing of search")
        await storage_manager.add_message("multi_search_user", session2_id, message2)
        
        # Search across all sessions
        results = await storage_manager.search_messages("multi_search_user", "testing")
        
        assert len(results) == 2
        result_contents = {r.content for r in results}
        assert "Testing search functionality" in result_contents
        assert "More testing of search" in result_contents


class TestDocumentManagement:
    """Test document management operations."""
    
    @pytest.mark.asyncio
    async def test_save_document_basic(self, storage_manager):
        """Test basic document saving."""
        await storage_manager.create_user("doc_user")
        session_id = await storage_manager.create_session("doc_user", "Document Session")
        
        content = b"This is test document content"
        metadata = {"type": "test", "size": len(content)}
        
        doc_id = await storage_manager.save_document(
            "doc_user", session_id, "test.txt", content, metadata
        )
        
        assert doc_id != ""
        assert "test.txt" in doc_id
    
    @pytest.mark.asyncio
    async def test_retrieve_document(self, storage_manager):
        """Test document retrieval."""
        await storage_manager.create_user("retrieve_user")
        session_id = await storage_manager.create_session("retrieve_user", "Retrieve Session")
        
        original_content = b"Document content to retrieve"
        doc_id = await storage_manager.save_document(
            "retrieve_user", session_id, "retrieve.txt", original_content
        )
        
        # Retrieve document
        retrieved_content = await storage_manager.get_document("retrieve_user", session_id, doc_id)
        assert retrieved_content == original_content
    
    @pytest.mark.asyncio
    async def test_list_documents(self, storage_manager, sample_documents):
        """Test listing documents in session."""
        await storage_manager.create_user("list_doc_user")
        session_id = await storage_manager.create_session("list_doc_user", "List Documents")
        
        # Save multiple documents
        saved_docs = []
        for doc in sample_documents:
            content = f"Content for {doc.filename}".encode()
            doc_id = await storage_manager.save_document(
                "list_doc_user", session_id, doc.filename, content, doc.metadata
            )
            saved_docs.append(doc_id)
        
        # List documents
        documents = await storage_manager.list_documents("list_doc_user", session_id)
        
        assert len(documents) == len(sample_documents)
        
        # Verify document properties
        filenames = {doc.original_name for doc in documents}
        expected_filenames = {doc.filename for doc in sample_documents}
        assert filenames == expected_filenames
    
    @pytest.mark.asyncio
    async def test_document_size_limit(self, storage_manager):
        """Test document size limit enforcement."""
        await storage_manager.create_user("size_doc_user")
        session_id = await storage_manager.create_session("size_doc_user", "Size Test")
        
        # Create document that exceeds size limit
        large_content = b"X" * (storage_manager.config.storage.max_document_size_bytes + 1000)
        
        doc_id = await storage_manager.save_document(
            "size_doc_user", session_id, "large.txt", large_content
        )
        
        # Should fail
        assert doc_id == ""
    
    @pytest.mark.asyncio
    async def test_document_extension_validation(self, storage_manager):
        """Test document extension validation."""
        await storage_manager.create_user("ext_user")
        session_id = await storage_manager.create_session("ext_user", "Extension Test")
        
        content = b"test content"
        
        # Test disallowed extension
        disallowed_doc_id = await storage_manager.save_document(
            "ext_user", session_id, "test.exe", content
        )
        assert disallowed_doc_id == ""
        
        # Test allowed extension
        allowed_doc_id = await storage_manager.save_document(
            "ext_user", session_id, "test.txt", content
        )
        assert allowed_doc_id != ""
    
    @pytest.mark.asyncio
    async def test_update_document_analysis(self, storage_manager):
        """Test updating document analysis results."""
        await storage_manager.create_user("analysis_user")
        session_id = await storage_manager.create_session("analysis_user", "Analysis Session")
        
        content = b"Document for analysis"
        doc_id = await storage_manager.save_document(
            "analysis_user", session_id, "analyze.txt", content
        )
        
        # Update analysis
        analysis = {
            "word_count": 3,
            "language": "english",
            "sentiment": "neutral",
            "topics": ["document", "analysis"]
        }
        
        result = await storage_manager.update_document_analysis(
            "analysis_user", session_id, doc_id, analysis
        )
        assert result is True
        
        # Verify analysis was stored
        documents = await storage_manager.list_documents("analysis_user", session_id)
        doc = next(d for d in documents if d.filename == doc_id)
        assert "analysis" in doc.analysis
        assert doc.analysis["analysis"]["word_count"] == 3


class TestContextManagement:
    """Test situational context management."""
    
    @pytest.mark.asyncio
    async def test_update_context(self, storage_manager):
        """Test updating situational context."""
        await storage_manager.create_user("context_user")
        session_id = await storage_manager.create_session("context_user", "Context Session")
        
        context = sample_context()
        result = await storage_manager.update_context("context_user", session_id, context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_context(self, storage_manager):
        """Test retrieving situational context."""
        await storage_manager.create_user("get_context_user")
        session_id = await storage_manager.create_session("get_context_user", "Get Context")
        
        original_context = sample_context()
        await storage_manager.update_context("get_context_user", session_id, original_context)
        
        # Retrieve context
        retrieved_context = await storage_manager.get_context("get_context_user", session_id)
        
        assert retrieved_context is not None
        assert retrieved_context.summary == original_context.summary
        assert retrieved_context.key_points == original_context.key_points
        assert retrieved_context.confidence == original_context.confidence
    
    @pytest.mark.asyncio
    async def test_context_history_snapshots(self, storage_manager):
        """Test context history and snapshots."""
        await storage_manager.create_user("history_user")
        session_id = await storage_manager.create_session("history_user", "History Session")
        
        # Create multiple context snapshots
        contexts = []
        for i in range(3):
            context = FFSituationalContextDTO(
                summary=f"Context snapshot {i}",
                key_points=[f"Point {i}"],
                entities={"iteration": [str(i)]},
                confidence=0.8 + i * 0.05
            )
            contexts.append(context)
            
            # Save snapshot
            result = await storage_manager.save_context_snapshot("history_user", session_id, context)
            assert result is True
        
        # Get context history
        history = await storage_manager.get_context_history("history_user", session_id)
        
        assert len(history) == 3
        
        # Verify snapshots (should be in reverse chronological order)
        summaries = [ctx.summary for ctx in history]
        assert "Context snapshot 2" in summaries
        assert "Context snapshot 1" in summaries
        assert "Context snapshot 0" in summaries
    
    @pytest.mark.asyncio
    async def test_context_history_pagination(self, storage_manager):
        """Test context history with pagination."""
        await storage_manager.create_user("paginate_context_user")
        session_id = await storage_manager.create_session("paginate_context_user", "Paginate Context")
        
        # Create many context snapshots
        for i in range(10):
            context = FFSituationalContextDTO(
                summary=f"Snapshot {i}",
                key_points=[f"Point {i}"],
                entities={},
                confidence=0.8
            )
            await storage_manager.save_context_snapshot("paginate_context_user", session_id, context)
        
        # Get limited history
        limited_history = await storage_manager.get_context_history(
            "paginate_context_user", session_id, limit=5
        )
        
        assert len(limited_history) == 5


class TestSessionStatistics:
    """Test session statistics functionality."""
    
    @pytest.mark.asyncio
    async def test_get_session_stats_basic(self, storage_manager):
        """Test getting basic session statistics."""
        await storage_manager.create_user("stats_user")
        session_id = await storage_manager.create_session("stats_user", "Stats Session")
        
        # Get stats for empty session
        stats = await storage_manager.get_session_stats("stats_user", session_id)
        
        assert stats["session_id"] == session_id
        assert stats["user_id"] == "stats_user"
        assert stats["message_count"] == 0
        assert stats["document_count"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["has_context"] is False
    
    @pytest.mark.asyncio
    async def test_get_session_stats_with_data(self, storage_manager, sample_messages):
        """Test session statistics with actual data."""
        await storage_manager.create_user("data_stats_user")
        session_id = await storage_manager.create_session("data_stats_user", "Data Stats")
        
        # Add messages
        for message in sample_messages:
            await storage_manager.add_message("data_stats_user", session_id, message)
        
        # Add document
        doc_content = b"Test document for statistics"
        await storage_manager.save_document(
            "data_stats_user", session_id, "stats.txt", doc_content
        )
        
        # Add context
        context = sample_context()
        await storage_manager.update_context("data_stats_user", session_id, context)
        
        # Get stats
        stats = await storage_manager.get_session_stats("data_stats_user", session_id)
        
        assert stats["message_count"] == len(sample_messages)
        assert stats["document_count"] == 1
        assert stats["total_size_bytes"] > 0  # Should include message and document sizes
        assert stats["has_context"] is True
        assert stats["average_message_size"] > 0
    
    @pytest.mark.asyncio
    async def test_get_session_stats_nonexistent(self, storage_manager):
        """Test getting stats for nonexistent session."""
        with pytest.raises(ValueError, match="Session .* not found"):
            await storage_manager.get_session_stats("nonexistent_user", "nonexistent_session")


@pytest.mark.integration
class TestStorageIntegrationWorkflows:
    """Integration tests for complete storage workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_chat_workflow(self, storage_manager, test_data_generator):
        """Test complete chat session workflow."""
        # Create user
        await storage_manager.create_user("workflow_user", {
            "username": "WorkflowTester",
            "preferences": {"notifications": True}
        })
        
        # Create session
        session_id = await storage_manager.create_session("workflow_user", "Complete Workflow")
        
        # Add conversation
        session, messages = test_data_generator.create_test_session_with_messages(
            "workflow_user", session_id, 10
        )
        
        for message in messages:
            await storage_manager.add_message("workflow_user", session_id, message)
        
        # Add document
        doc_content = "# Workflow Documentation\n\nThis documents the complete workflow test."
        await storage_manager.save_document(
            "workflow_user", session_id, "workflow.md", doc_content.encode()
        )
        
        # Update context
        context = FFSituationalContextDTO(
            summary="User is testing complete workflow functionality",
            key_points=["Testing", "Workflow", "Integration"],
            entities={"topics": ["testing", "workflow"], "users": ["workflow_user"]},
            confidence=0.95
        )
        await storage_manager.update_context("workflow_user", session_id, context)
        
        # Verify complete workflow
        final_session = await storage_manager.get_session("workflow_user", session_id)
        assert final_session.message_count == 10
        
        all_messages = await storage_manager.get_all_messages("workflow_user", session_id)
        assert len(all_messages) == 10
        
        documents = await storage_manager.list_documents("workflow_user", session_id)
        assert len(documents) == 1
        assert documents[0].original_name == "workflow.md"
        
        retrieved_context = await storage_manager.get_context("workflow_user", session_id)
        assert retrieved_context.confidence == 0.95
        
        # Get comprehensive stats
        stats = await storage_manager.get_session_stats("workflow_user", session_id)
        assert stats["message_count"] == 10
        assert stats["document_count"] == 1
        assert stats["has_context"] is True
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, storage_manager):
        """Test that multiple users are properly isolated."""
        # Create two users with same session names
        await storage_manager.create_user("user1")
        await storage_manager.create_user("user2")
        
        session1_id = await storage_manager.create_session("user1", "Shared Title")
        session2_id = await storage_manager.create_session("user2", "Shared Title")
        
        # Add different messages to each
        msg1 = sample_message(MessageRole.USER, "Message from user1")
        msg2 = sample_message(MessageRole.USER, "Message from user2") 
        
        await storage_manager.add_message("user1", session1_id, msg1)
        await storage_manager.add_message("user2", session2_id, msg2)
        
        # Verify isolation
        user1_messages = await storage_manager.get_all_messages("user1", session1_id)
        user2_messages = await storage_manager.get_all_messages("user2", session2_id)
        
        assert len(user1_messages) == 1
        assert len(user2_messages) == 1
        assert user1_messages[0].content == "Message from user1"
        assert user2_messages[0].content == "Message from user2"
        
        # Cross-user access should fail
        cross_messages = await storage_manager.get_all_messages("user1", session2_id)
        assert len(cross_messages) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, storage_manager):
        """Test concurrent storage operations."""
        await storage_manager.create_user("concurrent_user")
        session_id = await storage_manager.create_session("concurrent_user", "Concurrent Test")
        
        # Create multiple concurrent message additions
        async def add_message_task(i):
            message = sample_message(MessageRole.USER, f"Concurrent message {i}")
            return await storage_manager.add_message("concurrent_user", session_id, message)
        
        # Run concurrent tasks
        tasks = [add_message_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # Verify all messages were added
        final_session = await storage_manager.get_session("concurrent_user", session_id)
        assert final_session.message_count == 10
        
        messages = await storage_manager.get_all_messages("concurrent_user", session_id)
        assert len(messages) == 10


# Helper functions (duplicated from conftest.py for standalone testing)
def sample_user_profile(user_id: str = "test_user") -> FFUserProfileDTO:
    """Create a sample user profile for testing."""
    return FFUserProfileDTO(
        user_id=user_id,
        username=f"user_{user_id}",
        preferences={"theme": "dark", "language": "en"},
        metadata={"created_via": "test", "test_flag": True}
    )


def sample_message(
    role: MessageRole = MessageRole.USER,
    content: str = "Test message content",
    message_id: str = "msg_test123"
) -> FFMessageDTO:
    """Create a sample message for testing."""
    return FFMessageDTO(
        id=message_id,
        role=role,
        content=content,
        timestamp=datetime.now().isoformat(),
        metadata={"test_message": True}
    )


def sample_context() -> FFSituationalContextDTO:
    """Create a sample situational context for testing."""
    return FFSituationalContextDTO(
        summary="Test conversation about testing",
        key_points=["Testing is important", "We're testing the system"],
        entities={"topics": ["testing", "system"], "users": ["test_user"]},
        confidence=0.85,
        metadata={"test_context": True}
    )