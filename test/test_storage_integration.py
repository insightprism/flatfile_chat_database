"""
Integration tests for StorageManager.

These tests verify the complete functionality of the storage system
with all components working together.
"""

import asyncio
import tempfile
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import StorageManager
from config import StorageConfig
from models import (
    Message, Session, SituationalContext, Document,
    UserProfile, Persona, PanelMessage, PanelInsight
)


async def storage_manager():
    """Create StorageManager with temp directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(
            storage_base_path=temp_dir,
            max_messages_per_session=1000,
            max_document_size_bytes=10_000_000  # 10MB for tests
        )
        
        manager = StorageManager(config=config)
        await manager.initialize()
        
        yield manager
        
        # Cleanup is automatic with temp directory


class TestUserManagement:
    """Test user management functionality"""
    
    # Test method - async
    async def test_create_and_get_user(self, storage_manager):
        """Test creating and retrieving user"""
        user_id = "test_user"
        profile = {
            "username": "Test User",
            "preferences": {"theme": "dark"},
            "metadata": {"source": "test"}
        }
        
        # Create user
        result = await storage_manager.create_user(user_id, profile)
        assert result is True
        
        # Get user profile
        retrieved = await storage_manager.get_user_profile(user_id)
        assert retrieved is not None
        assert retrieved["user_id"] == user_id
        assert retrieved["username"] == "Test User"
        assert retrieved["preferences"]["theme"] == "dark"
    
    # Test method - async
    async def test_update_user_profile(self, storage_manager):
        """Test updating user profile"""
        user_id = "update_test"
        
        # Create user
        await storage_manager.create_user(user_id)
        
        # Update profile
        updates = {
            "username": "Updated Name",
            "preferences": {"language": "es"}
        }
        result = await storage_manager.update_user_profile(user_id, updates)
        assert result is True
        
        # Verify updates
        profile = await storage_manager.get_user_profile(user_id)
        assert profile["username"] == "Updated Name"
        assert profile["preferences"]["language"] == "es"
    
    # Test method - async
    async def test_list_users(self, storage_manager):
        """Test listing all users"""
        # Create multiple users
        users = ["alice", "bob", "charlie"]
        for user_id in users:
            await storage_manager.create_user(user_id)
        
        # List users
        user_list = await storage_manager.list_users()
        assert len(user_list) >= 3
        for user_id in users:
            assert user_id in user_list


class TestSessionManagement:
    """Test session management functionality"""
    
    # Test method - async
    async def test_create_and_get_session(self, storage_manager):
        """Test creating and retrieving session"""
        user_id = "session_test_user"
        await storage_manager.create_user(user_id)
        
        # Create session
        session_id = await storage_manager.create_session(user_id, "Test Session")
        assert session_id != ""
        assert session_id.startswith("chat_session_")
        
        # Get session
        session = await storage_manager.get_session(user_id, session_id)
        assert session is not None
        assert session.id == session_id
        assert session.title == "Test Session"
        assert session.user_id == user_id
    
    # Test method - async
    async def test_update_session(self, storage_manager):
        """Test updating session metadata"""
        user_id = "session_update_user"
        await storage_manager.create_user(user_id)
        
        # Create session
        session_id = await storage_manager.create_session(user_id)
        
        # Update session
        updates = {
            "title": "Updated Title",
            "metadata": {"tags": ["important", "work"]}
        }
        result = await storage_manager.update_session(user_id, session_id, updates)
        assert result is True
        
        # Verify updates
        session = await storage_manager.get_session(user_id, session_id)
        assert session.title == "Updated Title"
        assert session.metadata["tags"] == ["important", "work"]
    
    # Test method - async
    async def test_list_sessions(self, storage_manager):
        """Test listing user sessions"""
        user_id = "session_list_user"
        await storage_manager.create_user(user_id)
        
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = await storage_manager.create_session(user_id, f"Session {i}")
            session_ids.append(session_id)
            await asyncio.sleep(0.01)  # Ensure different timestamps
        
        # List sessions
        sessions = await storage_manager.list_sessions(user_id)
        assert len(sessions) == 5
        
        # Test pagination
        sessions = await storage_manager.list_sessions(user_id, limit=3)
        assert len(sessions) == 3
        
        sessions = await storage_manager.list_sessions(user_id, limit=3, offset=3)
        assert len(sessions) == 2
    
    # Test method - async
    async def test_delete_session(self, storage_manager):
        """Test deleting session"""
        user_id = "session_delete_user"
        await storage_manager.create_user(user_id)
        
        # Create session
        session_id = await storage_manager.create_session(user_id)
        
        # Verify it exists
        session = await storage_manager.get_session(user_id, session_id)
        assert session is not None
        
        # Delete session
        result = await storage_manager.delete_session(user_id, session_id)
        assert result is True
        
        # Verify it's gone
        session = await storage_manager.get_session(user_id, session_id)
        assert session is None


class TestMessageManagement:
    """Test message functionality"""
    
    # Test method - async
    async def test_add_and_get_messages(self, storage_manager):
        """Test adding and retrieving messages"""
        user_id = "message_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Add messages
        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="What can you help me with?"),
            Message(role="assistant", content="I can help with many things!")
        ]
        
        for msg in messages:
            result = await storage_manager.add_message(user_id, session_id, msg)
            assert result is True
        
        # Get messages
        retrieved = await storage_manager.get_messages(user_id, session_id)
        assert len(retrieved) == 4
        
        # Verify content
        for i, msg in enumerate(retrieved):
            assert msg.role == messages[i].role
            assert msg.content == messages[i].content
    
    # Test method - async
    async def test_message_pagination(self, storage_manager):
        """Test message pagination"""
        user_id = "pagination_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Add many messages
        for i in range(20):
            msg = Message(role="user" if i % 2 == 0 else "assistant", 
                         content=f"Message {i}")
            await storage_manager.add_message(user_id, session_id, msg)
        
        # Test pagination
        page1 = await storage_manager.get_messages(user_id, session_id, limit=10)
        assert len(page1) == 10
        assert page1[0].content == "Message 0"
        
        page2 = await storage_manager.get_messages(user_id, session_id, limit=10, offset=10)
        assert len(page2) == 10
        assert page2[0].content == "Message 10"
    
    # Test method - async
    async def test_search_messages(self, storage_manager):
        """Test message search"""
        user_id = "search_test_user"
        await storage_manager.create_user(user_id)
        
        # Create sessions with messages
        session1 = await storage_manager.create_session(user_id)
        await storage_manager.add_message(user_id, session1, 
                                        Message(role="user", content="Tell me about Python"))
        await storage_manager.add_message(user_id, session1,
                                        Message(role="assistant", content="Python is a programming language"))
        
        session2 = await storage_manager.create_session(user_id)
        await storage_manager.add_message(user_id, session2,
                                        Message(role="user", content="What is JavaScript?"))
        
        # Search across sessions
        results = await storage_manager.search_messages(user_id, "Python")
        assert len(results) == 2  # Found in both user message and assistant response
        
        # Search in specific session
        results = await storage_manager.search_messages(user_id, "JavaScript", session_id=session2)
        assert len(results) == 1


class TestDocumentManagement:
    """Test document handling"""
    
    # Test method - async
    async def test_save_and_get_document(self, storage_manager):
        """Test document upload and retrieval"""
        user_id = "doc_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Save document
        content = b"This is a test document content"
        doc_id = await storage_manager.save_document(
            user_id, session_id, "test.txt", content,
            metadata={"description": "Test file"}
        )
        assert doc_id != ""
        
        # Get document
        retrieved = await storage_manager.get_document(user_id, session_id, doc_id)
        assert retrieved == content
        
        # List documents
        docs = await storage_manager.list_documents(user_id, session_id)
        assert len(docs) == 1
        assert docs[0].original_name == "test.txt"
        assert docs[0].size == len(content)
    
    # Test method - async
    async def test_document_analysis(self, storage_manager):
        """Test document analysis storage"""
        user_id = "analysis_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Save document
        doc_id = await storage_manager.save_document(
            user_id, session_id, "analyze.pdf", b"PDF content"
        )
        
        # Add analysis
        analysis = {
            "summary": "This is a PDF document",
            "word_count": 100,
            "topics": ["test", "document"]
        }
        result = await storage_manager.update_document_analysis(
            user_id, session_id, doc_id, analysis
        )
        assert result is True
        
        # Verify analysis stored
        docs = await storage_manager.list_documents(user_id, session_id)
        assert docs[0].analysis["analysis"]["results"]["summary"] == "This is a PDF document"


class TestContextManagement:
    """Test situational context functionality"""
    
    # Test method - async
    async def test_context_operations(self, storage_manager):
        """Test context save and retrieve"""
        user_id = "context_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Create context
        context = SituationalContext(
            summary="Discussing Python programming",
            key_points=["Functions", "Classes", "Modules"],
            entities={"languages": ["Python"], "concepts": ["OOP"]},
            confidence=0.9
        )
        
        # Update current context
        result = await storage_manager.update_context(user_id, session_id, context)
        assert result is True
        
        # Get current context
        retrieved = await storage_manager.get_context(user_id, session_id)
        assert retrieved is not None
        assert retrieved.summary == context.summary
        assert retrieved.confidence == 0.9
    
    # Test method - async
    async def test_context_history(self, storage_manager):
        """Test context history tracking"""
        user_id = "history_test_user"
        await storage_manager.create_user(user_id)
        session_id = await storage_manager.create_session(user_id)
        
        # Save multiple context snapshots
        contexts = []
        for i in range(3):
            context = SituationalContext(
                summary=f"Context {i}",
                key_points=[f"Point {i}"],
                entities={"index": [str(i)]},
                confidence=0.5 + i * 0.1
            )
            contexts.append(context)
            await storage_manager.save_context_snapshot(user_id, session_id, context)
            await asyncio.sleep(0.01)  # Ensure different timestamps
        
        # Get history
        history = await storage_manager.get_context_history(user_id, session_id)
        assert len(history) == 3
        
        # Should be in reverse chronological order
        assert history[0].summary == "Context 2"
        assert history[2].summary == "Context 0"


class TestPanelManagement:
    """Test panel session functionality"""
    
    # Test method - async
    async def test_create_panel(self, storage_manager):
        """Test panel creation"""
        # Create some personas first
        personas_data = [
            {"id": "analyst", "name": "Data Analyst", "expertise": ["data", "statistics"]},
            {"id": "engineer", "name": "Software Engineer", "expertise": ["coding", "architecture"]},
            {"id": "manager", "name": "Project Manager", "expertise": ["planning", "coordination"]}
        ]
        
        for persona in personas_data:
            await storage_manager.save_persona(persona["id"], persona)
        
        # Create panel
        panel_id = await storage_manager.create_panel(
            "expert_panel",
            ["analyst", "engineer", "manager"],
            config={"topic": "System Design"}
        )
        assert panel_id != ""
        assert panel_id.startswith("panel_")
    
    # Test method - async
    async def test_panel_messages(self, storage_manager):
        """Test panel message handling"""
        # Setup personas and panel
        await storage_manager.save_persona("persona1", {"name": "Expert 1"})
        await storage_manager.save_persona("persona2", {"name": "Expert 2"})
        
        panel_id = await storage_manager.create_panel("discussion", ["persona1", "persona2"])
        
        # Add messages
        msg1 = PanelMessage(
            role="persona1",
            content="I think we should use microservices",
            persona_id="persona1"
        )
        msg2 = PanelMessage(
            role="persona2",
            content="I agree, but we need to consider the complexity",
            persona_id="persona2",
            response_to=msg1.id
        )
        
        await storage_manager.add_panel_message(panel_id, msg1)
        await storage_manager.add_panel_message(panel_id, msg2)
        
        # Get messages
        messages = await storage_manager.get_panel_messages(panel_id)
        assert len(messages) == 2
        assert messages[1].response_to == msg1.id
    
    # Test method - async
    async def test_panel_insights(self, storage_manager):
        """Test panel insight storage"""
        # Create panel
        await storage_manager.save_persona("expert", {"name": "Domain Expert"})
        panel_id = await storage_manager.create_panel("analysis", ["expert"])
        
        # Save insight
        insight = PanelInsight(
            panel_id=panel_id,
            type="conclusion",
            content="The system should use event-driven architecture",
            consensus_level=0.85,
            supporting_messages=["msg1", "msg2"]
        )
        
        result = await storage_manager.save_panel_insight(panel_id, insight)
        assert result is True


class TestPersonaManagement:
    """Test persona functionality"""
    
    # Test method - async
    async def test_global_personas(self, storage_manager):
        """Test global persona management"""
        # Save global persona
        persona_data = {
            "id": "helpful_assistant",
            "name": "Helpful Assistant",
            "traits": ["friendly", "knowledgeable"],
            "communication_style": {"tone": "professional", "verbosity": "moderate"}
        }
        
        result = await storage_manager.save_persona("helpful_assistant", persona_data)
        assert result is True
        
        # Get persona
        retrieved = await storage_manager.get_persona("helpful_assistant")
        assert retrieved is not None
        assert retrieved["name"] == "Helpful Assistant"
        
        # List personas
        personas = await storage_manager.list_personas()
        assert len(personas) >= 1
    
    # Test method - async
    async def test_user_personas(self, storage_manager):
        """Test user-specific personas"""
        user_id = "persona_test_user"
        await storage_manager.create_user(user_id)
        
        # Save user persona
        persona_data = {
            "id": "custom_assistant",
            "name": "My Custom Assistant",
            "owner_id": user_id
        }
        
        result = await storage_manager.save_persona("custom_assistant", persona_data, user_id=user_id)
        assert result is True
        
        # Get user persona
        retrieved = await storage_manager.get_persona("custom_assistant", user_id=user_id)
        assert retrieved is not None
        assert retrieved["owner_id"] == user_id
        
        # List with user personas
        all_personas = await storage_manager.list_personas(user_id=user_id)
        user_personas = [p for p in all_personas if p.get("owner_id") == user_id]
        assert len(user_personas) >= 1


class TestCompleteWorkflow:
    """Test complete usage workflow"""
    
    # Test method - async
    async def test_full_chat_workflow(self, storage_manager):
        """Test a complete chat session workflow"""
        # 1. Create user
        user_id = "workflow_user"
        await storage_manager.create_user(user_id, {
            "username": "Test User",
            "preferences": {"ai_model": "gpt-4"}
        })
        
        # 2. Create session
        session_id = await storage_manager.create_session(user_id, "Python Help Session")
        
        # 3. Add messages
        await storage_manager.add_message(user_id, session_id,
                                        Message(role="user", content="How do I read a file in Python?"))
        await storage_manager.add_message(user_id, session_id,
                                        Message(role="assistant", content="You can use the open() function..."))
        
        # 4. Upload a document
        doc_id = await storage_manager.save_document(
            user_id, session_id, "example.py",
            b"with open('file.txt', 'r') as f:\n    content = f.read()"
        )
        
        # 5. Update context
        context = SituationalContext(
            summary="User learning file I/O in Python",
            key_points=["open() function", "context managers", "file modes"],
            entities={"topics": ["file I/O"], "language": ["Python"]},
            confidence=0.95
        )
        await storage_manager.update_context(user_id, session_id, context)
        
        # 6. Verify everything
        session = await storage_manager.get_session(user_id, session_id)
        assert session.message_count == 2
        
        messages = await storage_manager.get_messages(user_id, session_id)
        assert len(messages) == 2
        
        docs = await storage_manager.list_documents(user_id, session_id)
        assert len(docs) == 1
        
        current_context = await storage_manager.get_context(user_id, session_id)
        assert current_context.confidence == 0.95


if __name__ == "__main__":
    print("Please use run_tests.py to run these tests")