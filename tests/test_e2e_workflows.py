"""
End-to-end workflow tests for the flatfile chat database system.

Tests complete user workflows from system initialization through
complex multi-component interactions to verify the recent architecture
changes haven't broken any critical functionality.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ff_dependency_injection_manager import ff_create_application_container, ff_clear_global_container
from ff_class_configs.ff_configuration_manager_config import create_default_config
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFSituationalContextDTO,
    FFUserProfileDTO, MessageRole
)
from ff_protocols import StorageProtocol, BackendProtocol, SearchProtocol


class TestSystemBootstrap:
    """Test complete system bootstrap and initialization."""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self, temp_dir):
        """Test complete system initialization from scratch."""
        # Clear any global state
        ff_clear_global_container()
        
        # Create configuration
        config = create_default_config("test")
        config.storage.base_path = str(temp_dir)
        
        # Create application container
        container = ff_create_application_container()
        
        # Override config
        container._singletons[type(config)] = config
        
        # Resolve all core services
        storage = container.resolve(StorageProtocol)
        backend = container.resolve(BackendProtocol)
        
        # Initialize system
        backend_init = await backend.initialize()
        assert backend_init is True
        
        storage_init = await storage.initialize()
        assert storage_init is True
        
        # Verify system is ready
        assert storage is not None
        assert backend is not None
        
        # Verify directory structure
        base_path = Path(temp_dir)
        assert base_path.exists()
        assert base_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_system_initialization_with_existing_data(self, temp_dir):
        """Test system initialization with pre-existing data."""
        # First run - create some data
        config = create_default_config("test")
        config.storage.base_path = str(temp_dir)
        
        container1 = ff_create_application_container()
        container1._singletons[type(config)] = config
        
        storage1 = container1.resolve(StorageProtocol)
        await storage1.initialize()
        
        # Create test data
        await storage1.create_user("existing_user")
        session_id = await storage1.create_session("existing_user", "Existing Session")
        message = FFMessageDTO(role=MessageRole.USER, content="Pre-existing message")
        await storage1.add_message("existing_user", session_id, message)
        
        # Second run - reinitialize system
        ff_clear_global_container()
        container2 = ff_create_application_container()
        container2._singletons[type(config)] = config
        
        storage2 = container2.resolve(StorageProtocol)
        await storage2.initialize()
        
        # Verify existing data is accessible
        user_exists = await storage2.user_exists("existing_user")
        assert user_exists is True
        
        sessions = await storage2.list_sessions("existing_user")
        assert len(sessions) >= 1
        
        messages = await storage2.get_all_messages("existing_user", session_id)
        assert len(messages) >= 1
        assert messages[0].content == "Pre-existing message"


class TestCompleteUserWorkflow:
    """Test complete user workflow from registration to complex interactions."""
    
    @pytest.mark.asyncio
    async def test_new_user_complete_workflow(self, app_container):
        """Test complete workflow for a new user."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Step 1: User registration
        user_profile = {
            "username": "newuser123",
            "preferences": {
                "theme": "dark",
                "language": "en", 
                "notifications": True
            },
            "metadata": {
                "signup_date": datetime.now().isoformat(),
                "source": "web_app"
            }
        }
        
        user_created = await storage.create_user("newuser", user_profile)
        assert user_created is True
        
        # Verify user profile
        retrieved_profile = await storage.get_user_profile("newuser")
        assert retrieved_profile["username"] == "newuser123"
        assert retrieved_profile["preferences"]["theme"] == "dark"
        
        # Step 2: First chat session
        session_id = await storage.create_session("newuser", "My First Chat")
        assert session_id != ""
        
        # Step 3: Conversation with multiple messages
        conversation = [
            ("user", "Hello! I'm new to this system."),
            ("assistant", "Welcome! I'm happy to help you get started."),
            ("user", "Can you help me understand how this works?"),
            ("assistant", "Of course! This is a chat system where we can have conversations and I can help with various tasks."),
            ("user", "That's great! Can I upload documents too?"),
            ("assistant", "Yes, you can upload documents and I can analyze them for you.")
        ]
        
        for role, content in conversation:
            message = FFMessageDTO(
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content
            )
            message_added = await storage.add_message("newuser", session_id, message)
            assert message_added is True
        
        # Verify conversation
        messages = await storage.get_all_messages("newuser", session_id)
        assert len(messages) == 6
        assert messages[0].content == "Hello! I'm new to this system."
        assert messages[-1].content == "Yes, you can upload documents and I can analyze them for you."
        
        # Step 4: Document upload
        document_content = """
        # Getting Started Guide
        
        Welcome to the chat system! Here are some key features:
        
        1. Real-time conversations
        2. Document upload and analysis
        3. Context-aware responses
        4. Session management
        
        ## Tips for Success
        - Be specific in your questions
        - Upload relevant documents
        - Use clear, concise language
        """
        
        doc_id = await storage.save_document(
            "newuser", 
            session_id, 
            "getting_started.md", 
            document_content.encode(),
            {"type": "guide", "category": "onboarding"}
        )
        assert doc_id != ""
        
        # Verify document was stored
        documents = await storage.list_documents("newuser", session_id)
        assert len(documents) == 1
        assert documents[0].original_name == "getting_started.md"
        assert documents[0].mime_type == "text/markdown"
        
        # Step 5: Update situational context
        context = FFSituationalContextDTO(
            summary="New user onboarding conversation with document upload",
            key_points=[
                "User is new to the system",
                "Explained basic features",
                "User uploaded getting started guide",
                "User interested in document analysis"
            ],
            entities={
                "topics": ["onboarding", "features", "documents"],
                "user_type": ["new_user"],
                "intent": ["learning", "exploration"]
            },
            confidence=0.92
        )
        
        context_updated = await storage.update_context("newuser", session_id, context)
        assert context_updated is True
        
        # Verify context
        retrieved_context = await storage.get_context("newuser", session_id)
        assert retrieved_context.summary == context.summary
        assert len(retrieved_context.key_points) == 4
        assert retrieved_context.confidence == 0.92
        
        # Step 6: Get comprehensive session statistics
        stats = await storage.get_session_stats("newuser", session_id)
        assert stats["message_count"] == 6
        assert stats["document_count"] == 1
        assert stats["has_context"] is True
        assert stats["total_size_bytes"] > 0
        
        # Step 7: Search through conversation
        search_results = await storage.search_messages("newuser", "document")
        assert len(search_results) >= 2  # Should find mentions of documents
        
        # Step 8: Create second session for continued interaction
        session2_id = await storage.create_session("newuser", "Advanced Features")
        
        followup_message = FFMessageDTO(
            role=MessageRole.USER,
            content="I've read the guide. Now I want to try advanced features."
        )
        await storage.add_message("newuser", session2_id, followup_message)
        
        # Verify user now has multiple sessions
        all_sessions = await storage.list_sessions("newuser")
        assert len(all_sessions) == 2
        
        session_titles = {s.title for s in all_sessions}
        assert "My First Chat" in session_titles
        assert "Advanced Features" in session_titles
    
    @pytest.mark.asyncio
    async def test_power_user_workflow(self, app_container, test_data_generator):
        """Test workflow for power user with complex interactions."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Create power user
        power_user_profile = {
            "username": "poweruser",
            "preferences": {
                "advanced_features": True,
                "auto_context": True,
                "batch_operations": True
            },
            "metadata": {
                "user_type": "power_user",
                "experience_level": "expert"
            }
        }
        
        await storage.create_user("poweruser", power_user_profile)
        
        # Create multiple sessions for different projects
        project_sessions = {}
        projects = [
            ("AI Research", "Discussing machine learning and AI research topics"),
            ("Code Review", "Reviewing and analyzing code submissions"),
            ("Data Analysis", "Working with datasets and statistical analysis")
        ]
        
        for project_name, description in projects:
            session_id = await storage.create_session("poweruser", project_name)
            project_sessions[project_name] = session_id
            
            # Add initial context-setting message
            initial_msg = FFMessageDTO(
                role=MessageRole.USER,
                content=f"Starting work on {project_name}: {description}"
            )
            await storage.add_message("poweruser", session_id, initial_msg)
        
        # Simulate intensive work in AI Research session
        ai_session = project_sessions["AI Research"]
        
        # Add comprehensive conversation
        ai_conversation = test_data_generator.create_test_conversation(25)
        for message in ai_conversation:
            await storage.add_message("poweruser", ai_session, message)
        
        # Add multiple documents to AI Research session
        research_docs = [
            ("research_paper.pdf", b"PDF content simulating research paper"),
            ("dataset_description.txt", b"Description of the research dataset"),
            ("model_architecture.py", b"# Python code for model architecture\nclass NeuralNet:\n    pass"),
            ("results.json", b'{"accuracy": 0.95, "precision": 0.92, "recall": 0.88}')
        ]
        
        for filename, content in research_docs:
            doc_id = await storage.save_document("poweruser", ai_session, filename, content)
            assert doc_id != ""
        
        # Create context snapshots for different stages
        research_stages = [
            ("Initial Planning", "Setting up research objectives and methodology"),
            ("Data Collection", "Gathering and preprocessing research data"),
            ("Model Development", "Developing and training ML models"),
            ("Results Analysis", "Analyzing results and drawing conclusions")
        ]
        
        for stage_name, stage_desc in research_stages:
            stage_context = FFSituationalContextDTO(
                summary=f"AI Research: {stage_name} - {stage_desc}",
                key_points=[stage_name, "AI Research", "Progress tracking"],
                entities={"project": ["AI Research"], "stage": [stage_name]},
                confidence=0.88
            )
            await storage.save_context_snapshot("poweruser", ai_session, stage_context)
        
        # Update current context
        current_context = FFSituationalContextDTO(
            summary="AI Research project at analysis stage with comprehensive documentation",
            key_points=[
                "25+ messages of technical discussion",
                "4 research documents uploaded",
                "Multiple project stages tracked",
                "High engagement and detail level"
            ],
            entities={
                "topics": ["AI", "research", "machine learning", "analysis"],
                "documents": ["paper", "dataset", "code", "results"],
                "project_status": ["active", "advanced"]
            },
            confidence=0.94
        )
        await storage.update_context("poweruser", ai_session, current_context)
        
        # Verify power user workflow results
        ai_stats = await storage.get_session_stats("poweruser", ai_session)
        assert ai_stats["message_count"] >= 25
        assert ai_stats["document_count"] == 4
        assert ai_stats["has_context"] is True
        assert ai_stats["context_snapshots"] == 4
        
        # Test cross-session search
        all_search_results = await storage.search_messages("poweruser", "research")
        assert len(all_search_results) > 0
        
        # Test advanced features work across sessions
        all_sessions = await storage.list_sessions("poweruser")
        assert len(all_sessions) == 3
        
        total_messages = 0
        for session in all_sessions:
            session_messages = await storage.get_all_messages("poweruser", session.session_id)
            total_messages += len(session_messages)
        
        assert total_messages >= 28  # 25 + initial messages in each session


class TestMultiUserScenarios:
    """Test scenarios with multiple users interacting with the system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_operations(self, app_container):
        """Test multiple users operating concurrently."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Define multiple users
        users = [
            ("alice", "Alice Johnson", {"role": "researcher"}),
            ("bob", "Bob Smith", {"role": "developer"}), 
            ("charlie", "Charlie Brown", {"role": "analyst"})
        ]
        
        async def create_user_workflow(user_id, full_name, metadata):
            # Create user
            user_profile = {
                "username": full_name,
                "preferences": {"role": metadata["role"]},
                "metadata": metadata
            }
            await storage.create_user(user_id, user_profile)
            
            # Create session
            session_id = await storage.create_session(user_id, f"{full_name}'s Session")
            
            # Add messages
            messages = [
                f"Hello, I'm {full_name}",
                f"I work as a {metadata['role']}",
                "I'm testing the concurrent functionality"
            ]
            
            for content in messages:
                message = FFMessageDTO(role=MessageRole.USER, content=content)
                await storage.add_message(user_id, session_id, message)
            
            return user_id, session_id
        
        # Execute concurrent user workflows
        tasks = [create_user_workflow(uid, name, meta) for uid, name, meta in users]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        
        # Verify each user's data independently
        for user_id, session_id in results:
            # Verify user exists
            user_exists = await storage.user_exists(user_id)
            assert user_exists is True
            
            # Verify session
            session = await storage.get_session(user_id, session_id)
            assert session is not None
            assert session.message_count == 3
            
            # Verify messages
            messages = await storage.get_all_messages(user_id, session_id)
            assert len(messages) == 3
            
            # Verify isolation - user can't access other users' data
            for other_user_id, other_session_id in results:
                if other_user_id != user_id:
                    cross_access_messages = await storage.get_all_messages(user_id, other_session_id)
                    assert len(cross_access_messages) == 0
    
    @pytest.mark.asyncio
    async def test_user_data_isolation(self, app_container):
        """Test that user data is properly isolated."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Create two users with identical session names and content
        users_data = [
            ("user1", "Shared Session Title", "Secret message from user1"),
            ("user2", "Shared Session Title", "Secret message from user2")
        ]
        
        user_sessions = {}
        
        for user_id, session_title, secret_content in users_data:
            # Create user and session
            await storage.create_user(user_id)
            session_id = await storage.create_session(user_id, session_title)
            user_sessions[user_id] = session_id
            
            # Add secret message
            message = FFMessageDTO(role=MessageRole.USER, content=secret_content)
            await storage.add_message(user_id, session_id, message)
            
            # Add document with user-specific content
            doc_content = f"Document content for {user_id}".encode()
            await storage.save_document(user_id, session_id, "shared_name.txt", doc_content)
        
        # Verify isolation
        for user_id in ["user1", "user2"]:
            session_id = user_sessions[user_id]
            
            # Get messages
            messages = await storage.get_all_messages(user_id, session_id)
            assert len(messages) == 1
            
            # Verify correct content
            if user_id == "user1":
                assert messages[0].content == "Secret message from user1"
            else:
                assert messages[0].content == "Secret message from user2"
            
            # Get documents
            documents = await storage.list_documents(user_id, session_id)
            assert len(documents) == 1
            
            # Verify document content
            doc_content = await storage.get_document(user_id, session_id, documents[0].filename)
            expected_content = f"Document content for {user_id}".encode()
            assert doc_content == expected_content
        
        # Test cross-user access fails
        user1_session = user_sessions["user1"]
        user2_session = user_sessions["user2"]
        
        # User1 trying to access User2's data should get empty results
        cross_messages = await storage.get_all_messages("user1", user2_session)
        assert len(cross_messages) == 0
        
        cross_documents = await storage.list_documents("user1", user2_session)
        assert len(cross_documents) == 0


class TestComplexDocumentWorkflows:
    """Test complex document handling workflows."""
    
    @pytest.mark.asyncio
    async def test_document_intensive_workflow(self, app_container):
        """Test workflow with intensive document operations."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Create user for document testing
        await storage.create_user("doc_user", {
            "username": "DocumentExpert",
            "preferences": {"document_analysis": True}
        })
        
        session_id = await storage.create_session("doc_user", "Document Analysis Project")
        
        # Create various types of documents
        documents = [
            ("project_brief.txt", b"Project overview and objectives", "text/plain"),
            ("requirements.md", b"# Requirements\n\n1. Feature A\n2. Feature B", "text/markdown"),
            ("data.json", b'{"key": "value", "items": [1, 2, 3]}', "application/json"),
            ("config.txt", b"setting1=value1\nsetting2=value2", "text/plain"),
            ("readme.md", b"# Project README\n\nThis is the main documentation.", "text/markdown")
        ]
        
        doc_ids = []
        for filename, content, mime_type in documents:
            doc_id = await storage.save_document("doc_user", session_id, filename, content)
            assert doc_id != ""
            doc_ids.append(doc_id)
        
        # Verify all documents are stored
        stored_docs = await storage.list_documents("doc_user", session_id)
        assert len(stored_docs) == 5
        
        # Add analysis results for each document
        analysis_results = [
            {"word_count": 25, "type": "brief", "complexity": "low"},
            {"word_count": 15, "type": "requirements", "complexity": "medium", "sections": 2},
            {"type": "json", "keys": 2, "structure": "simple"},
            {"type": "config", "settings": 2, "format": "key_value"},
            {"word_count": 30, "type": "documentation", "format": "markdown"}
        ]
        
        for i, (doc_id, analysis) in enumerate(zip(doc_ids, analysis_results)):
            result = await storage.update_document_analysis("doc_user", session_id, doc_id, analysis)
            assert result is True
        
        # Create comprehensive context based on documents
        doc_context = FFSituationalContextDTO(
            summary="Document analysis project with 5 different file types processed",
            key_points=[
                "Project brief uploaded and analyzed",
                "Requirements document with 2 sections",
                "Configuration and data files processed",
                "Documentation files analyzed for content"
            ],
            entities={
                "document_types": ["brief", "requirements", "json", "config", "documentation"],
                "file_formats": ["txt", "md", "json"],
                "analysis_complete": ["true"]
            },
            confidence=0.89
        )
        
        await storage.update_context("doc_user", session_id, doc_context)
        
        # Add messages discussing the documents
        document_messages = [
            "I've uploaded all the project documents",
            "Please analyze the project brief first", 
            "The requirements document outlines our main features",
            "The JSON data file contains our test dataset",
            "All documents have been processed and analyzed"
        ]
        
        for content in document_messages:
            message = FFMessageDTO(role=MessageRole.USER, content=content)
            await storage.add_message("doc_user", session_id, message)
        
        # Verify final state
        final_stats = await storage.get_session_stats("doc_user", session_id)
        assert final_stats["document_count"] == 5
        assert final_stats["message_count"] == 5
        assert final_stats["has_context"] is True
        assert final_stats["total_size_bytes"] > 0  # Should include all document sizes
        
        # Test document search functionality
        search_results = await storage.search_messages("doc_user", "document")
        assert len(search_results) >= 3  # Should find multiple mentions
        
        # Test document retrieval
        for doc_id, (original_filename, original_content, _) in zip(doc_ids, documents):
            retrieved_content = await storage.get_document("doc_user", session_id, doc_id)
            assert retrieved_content == original_content


class TestSystemRecoveryAndConsistency:
    """Test system recovery and data consistency."""
    
    @pytest.mark.asyncio
    async def test_system_restart_data_persistence(self, temp_dir):
        """Test that data persists across system restarts."""
        # Phase 1: Create data with first system instance
        config1 = create_default_config("test")
        config1.storage.base_path = str(temp_dir)
        
        ff_clear_global_container()
        container1 = ff_create_application_container()
        container1._singletons[type(config1)] = config1
        
        storage1 = container1.resolve(StorageProtocol)
        await storage1.initialize()
        
        # Create comprehensive test data
        await storage1.create_user("persistent_user", {
            "username": "PersistentTester",
            "preferences": {"test": True}
        })
        
        session_id = await storage1.create_session("persistent_user", "Persistent Session")
        
        # Add messages
        test_messages = [
            "This message should persist across restarts",
            "System recovery test in progress",
            "Data consistency verification"
        ]
        
        for content in test_messages:
            message = FFMessageDTO(role=MessageRole.USER, content=content)
            await storage1.add_message("persistent_user", session_id, message)
        
        # Add document
        doc_content = b"This document should survive system restart"
        doc_id = await storage1.save_document(
            "persistent_user", session_id, "persistent.txt", doc_content
        )
        
        # Add context
        context = FFSituationalContextDTO(
            summary="Test data for system persistence verification",
            key_points=["Persistence test", "System restart", "Data integrity"],
            entities={"test_type": ["persistence", "recovery"]},
            confidence=0.95
        )
        await storage1.update_context("persistent_user", session_id, context)
        
        # Get original stats
        original_stats = await storage1.get_session_stats("persistent_user", session_id)
        
        # Phase 2: Restart system with new instance
        ff_clear_global_container()
        del storage1, container1
        
        config2 = create_default_config("test")
        config2.storage.base_path = str(temp_dir)
        
        container2 = ff_create_application_container()
        container2._singletons[type(config2)] = config2
        
        storage2 = container2.resolve(StorageProtocol)
        await storage2.initialize()
        
        # Phase 3: Verify all data persisted
        # Check user
        user_exists = await storage2.user_exists("persistent_user")
        assert user_exists is True
        
        profile = await storage2.get_user_profile("persistent_user")
        assert profile["username"] == "PersistentTester"
        assert profile["preferences"]["test"] is True
        
        # Check session
        session = await storage2.get_session("persistent_user", session_id)
        assert session is not None
        assert session.title == "Persistent Session"
        
        # Check messages
        messages = await storage2.get_all_messages("persistent_user", session_id)
        assert len(messages) == 3
        
        message_contents = {m.content for m in messages}
        expected_contents = set(test_messages)
        assert message_contents == expected_contents
        
        # Check document
        documents = await storage2.list_documents("persistent_user", session_id)
        assert len(documents) == 1
        assert documents[0].original_name == "persistent.txt"
        
        retrieved_content = await storage2.get_document("persistent_user", session_id, doc_id)
        assert retrieved_content == doc_content
        
        # Check context
        retrieved_context = await storage2.get_context("persistent_user", session_id)
        assert retrieved_context.summary == context.summary
        assert retrieved_context.confidence == 0.95
        
        # Check stats consistency
        new_stats = await storage2.get_session_stats("persistent_user", session_id)
        assert new_stats["message_count"] == original_stats["message_count"]
        assert new_stats["document_count"] == original_stats["document_count"]
        assert new_stats["has_context"] == original_stats["has_context"]
    
    @pytest.mark.asyncio
    async def test_partial_operation_recovery(self, app_container):
        """Test recovery from partial operations."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Create user and session
        await storage.create_user("recovery_user")
        session_id = await storage.create_session("recovery_user", "Recovery Test")
        
        # Simulate partial operation - add message successfully
        message1 = FFMessageDTO(role=MessageRole.USER, content="First message added successfully")
        result1 = await storage.add_message("recovery_user", session_id, message1)
        assert result1 is True
        
        # Verify system state is consistent
        session = await storage.get_session("recovery_user", session_id)
        assert session.message_count == 1
        
        messages = await storage.get_all_messages("recovery_user", session_id)
        assert len(messages) == 1
        
        # Continue operations after "recovery"
        message2 = FFMessageDTO(role=MessageRole.ASSISTANT, content="System recovered, continuing operation")
        result2 = await storage.add_message("recovery_user", session_id, message2)
        assert result2 is True
        
        # Verify final consistent state
        final_session = await storage.get_session("recovery_user", session_id)
        assert final_session.message_count == 2
        
        final_messages = await storage.get_all_messages("recovery_user", session_id)
        assert len(final_messages) == 2
        assert final_messages[0].content == "First message added successfully"
        assert final_messages[1].content == "System recovered, continuing operation"


@pytest.mark.integration
class TestArchitectureChangesValidation:
    """Test that recent architecture changes haven't broken functionality."""
    
    @pytest.mark.asyncio
    async def test_dependency_injection_integration(self, temp_dir):
        """Test that new DI system works with all components."""
        # Clear global state
        ff_clear_global_container()
        
        # Create configuration
        config = create_default_config("test")
        config.storage.base_path = str(temp_dir)
        
        # Test full DI container creation and resolution
        container = ff_create_application_container()
        container._singletons[type(config)] = config
        
        # Resolve all services through DI
        storage = container.resolve(StorageProtocol)
        backend = container.resolve(BackendProtocol)
        
        # Test that services are properly injected
        assert storage is not None
        assert backend is not None
        assert hasattr(storage, 'backend')
        
        # Test full workflow through DI-resolved services
        await backend.initialize()
        await storage.initialize()
        
        # Create user through DI-resolved storage
        user_created = await storage.create_user("di_test_user")
        assert user_created is True
        
        # Create session
        session_id = await storage.create_session("di_test_user", "DI Test Session")
        assert session_id != ""
        
        # Add message
        message = FFMessageDTO(role=MessageRole.USER, content="Testing DI integration")
        message_added = await storage.add_message("di_test_user", session_id, message)
        assert message_added is True
        
        # Verify through different service resolution
        storage2 = container.resolve(StorageProtocol)
        assert storage2 is storage  # Should be same singleton instance
        
        # Verify data through second resolution
        messages = await storage2.get_all_messages("di_test_user", session_id)
        assert len(messages) == 1
        assert messages[0].content == "Testing DI integration"
    
    @pytest.mark.asyncio
    async def test_configuration_system_integration(self, temp_dir):
        """Test that new configuration system works end-to-end."""
        # Test configuration loading and application
        config = create_default_config("test")
        config.storage.base_path = str(temp_dir)
        
        # Verify configuration is properly structured
        assert hasattr(config, 'storage')
        assert hasattr(config, 'search')
        assert hasattr(config, 'vector')
        assert hasattr(config, 'document')
        assert hasattr(config, 'locking')
        assert hasattr(config, 'panel')
        
        # Test configuration validation
        errors = config.validate_all()
        assert len(errors) == 0, f"Configuration validation errors: {errors}"
        
        # Test configuration with real components
        ff_clear_global_container()
        container = ff_create_application_container()
        container._singletons[type(config)] = config
        
        storage = container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Test that configuration values are used
        assert str(storage.base_path) == config.storage.base_path
        
        # Test configuration-driven behavior
        # Create large message to test size limits
        large_content = "X" * (config.storage.max_message_size_bytes + 100)
        large_message = FFMessageDTO(role=MessageRole.USER, content=large_content)
        
        await storage.create_user("config_test_user")
        session_id = await storage.create_session("config_test_user", "Config Test")
        
        # Should fail due to size limit
        result = await storage.add_message("config_test_user", session_id, large_message)
        assert result is False
        
        # Normal message should work
        normal_message = FFMessageDTO(role=MessageRole.USER, content="Normal message")
        normal_result = await storage.add_message("config_test_user", session_id, normal_message)
        assert normal_result is True
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, app_container):
        """Test that changes maintain backward compatibility."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Test that old-style entity creation still works
        from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO, FFUserProfileDTO
        
        # Create entities with both new and old patterns
        user_profile = FFUserProfileDTO(
            user_id="compat_user",
            username="CompatibilityTest"
        )
        
        # Store using new DTO
        profile_stored = await storage.store_user_profile(user_profile)
        assert profile_stored is True
        
        # Create session
        session_id = await storage.create_session("compat_user", "Compatibility Test")
        
        # Create message using DTO
        message = FFMessageDTO(
            role=MessageRole.USER,
            content="Testing backward compatibility"
        )
        
        message_added = await storage.add_message("compat_user", session_id, message)
        assert message_added is True
        
        # Verify all data is accessible
        retrieved_profile = await storage.get_user_profile("compat_user")
        assert retrieved_profile["username"] == "CompatibilityTest"
        
        retrieved_messages = await storage.get_all_messages("compat_user", session_id)
        assert len(retrieved_messages) == 1
        assert retrieved_messages[0].content == "Testing backward compatibility"
        
        # Test that search still works
        search_results = await storage.search_messages("compat_user", "compatibility")
        assert len(search_results) >= 1


# Helper functions
def sample_message(role: MessageRole, content: str) -> FFMessageDTO:
    """Create a sample message for testing."""
    return FFMessageDTO(role=role, content=content)