"""
Unit tests for FF Chat Session Manager following existing FF test patterns.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import FF chat components to test
from ff_chat_session_manager import FFChatSessionManager, FFChatSessionState
from ff_class_configs.ff_chat_session_config import FFChatSessionConfigDTO


@dataclass
class MockFFChatSession:
    """Mock chat session for testing"""
    session_id: str
    user_id: str
    use_case: str
    context: dict = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class TestFFChatSessionManager:
    """Test FF Chat Session Manager functionality"""
    
    @pytest.fixture
    def mock_ff_storage(self):
        """Mock FF storage manager for testing"""
        storage = AsyncMock()
        return storage
    
    @pytest.fixture
    def session_config(self):
        """Session configuration for testing"""
        return FFChatSessionConfigDTO(
            session_timeout=300,  # 5 minutes for testing
            cleanup_interval=60,  # 1 minute for testing
            max_processing_time=30
        )
    
    @pytest.fixture
    def session_manager(self, mock_ff_storage, session_config):
        """Session manager instance for testing"""
        return FFChatSessionManager(mock_ff_storage, session_config)
    
    @pytest.fixture
    def mock_chat_session(self):
        """Mock chat session for testing"""
        return MockFFChatSession(
            session_id="test_session_123",
            user_id="test_user",
            use_case="basic_chat",
            context={"test": "data"}
        )
    
    @pytest.mark.asyncio
    async def test_session_manager_initialization(self, session_manager):
        """Test session manager initializes correctly"""
        
        # Test initialization
        success = await session_manager.initialize()
        assert success
        assert session_manager._initialized
        
        # Verify cleanup task was started
        assert session_manager._cleanup_task is not None
        assert not session_manager._cleanup_task.done()
        
        # Cleanup
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_register_session(self, session_manager, mock_chat_session):
        """Test registering chat sessions"""
        
        await session_manager.initialize()
        
        # Register session
        success = await session_manager.register_session(mock_chat_session)
        assert success
        
        # Verify session was registered
        assert mock_chat_session.session_id in session_manager.active_sessions
        assert mock_chat_session.session_id in session_manager.session_locks
        
        # Verify session state
        session_state = session_manager.active_sessions[mock_chat_session.session_id]
        assert session_state.session_id == mock_chat_session.session_id
        assert session_state.user_id == mock_chat_session.user_id
        assert session_state.use_case == mock_chat_session.use_case
        assert session_state.message_count == 0
        assert not session_state.processing
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_unregister_session(self, session_manager, mock_chat_session):
        """Test unregistering chat sessions"""
        
        await session_manager.initialize()
        
        # Register session first
        await session_manager.register_session(mock_chat_session)
        assert mock_chat_session.session_id in session_manager.active_sessions
        
        # Unregister session
        success = await session_manager.unregister_session(mock_chat_session.session_id)
        assert success
        
        # Verify session was unregistered
        assert mock_chat_session.session_id not in session_manager.active_sessions
        assert mock_chat_session.session_id not in session_manager.session_locks
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_session_state(self, session_manager, mock_chat_session):
        """Test getting session state"""
        
        await session_manager.initialize()
        
        # Test non-existent session
        state = await session_manager.get_session_state("non_existent")
        assert state is None
        
        # Register session
        await session_manager.register_session(mock_chat_session)
        
        # Get session state
        state = await session_manager.get_session_state(mock_chat_session.session_id)
        assert state is not None
        assert state["session_id"] == mock_chat_session.session_id
        assert state["user_id"] == mock_chat_session.user_id
        assert state["use_case"] == mock_chat_session.use_case
        assert state["message_count"] == 0
        assert not state["processing"]
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_update_session_activity(self, session_manager, mock_chat_session):
        """Test updating session activity"""
        
        await session_manager.initialize()
        await session_manager.register_session(mock_chat_session)
        
        # Get initial activity time
        initial_state = session_manager.active_sessions[mock_chat_session.session_id]
        initial_time = initial_state.last_activity
        
        # Wait a bit and update activity
        await asyncio.sleep(0.1)
        await session_manager.update_session_activity(mock_chat_session.session_id)
        
        # Verify activity was updated
        updated_state = session_manager.active_sessions[mock_chat_session.session_id]
        assert updated_state.last_activity > initial_time
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_increment_message_count(self, session_manager, mock_chat_session):
        """Test incrementing message count"""
        
        await session_manager.initialize()
        await session_manager.register_session(mock_chat_session)
        
        # Initial count should be 0
        session_state = session_manager.active_sessions[mock_chat_session.session_id]
        assert session_state.message_count == 0
        
        # Increment message count
        await session_manager.increment_message_count(mock_chat_session.session_id)
        
        # Verify count was incremented
        assert session_state.message_count == 1
        
        # Increment again
        await session_manager.increment_message_count(mock_chat_session.session_id)
        assert session_state.message_count == 2
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_set_session_processing(self, session_manager, mock_chat_session):
        """Test setting session processing status"""
        
        await session_manager.initialize()
        await session_manager.register_session(mock_chat_session)
        
        session_state = session_manager.active_sessions[mock_chat_session.session_id]
        assert not session_state.processing
        
        # Set processing to True
        await session_manager.set_session_processing(mock_chat_session.session_id, True)
        assert session_state.processing
        
        # Set processing to False
        await session_manager.set_session_processing(mock_chat_session.session_id, False)
        assert not session_state.processing
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_session_lock(self, session_manager, mock_chat_session):
        """Test getting session locks"""
        
        await session_manager.initialize()
        
        # Test non-existent session
        lock = await session_manager.get_session_lock("non_existent")
        assert lock is None
        
        # Register session
        await session_manager.register_session(mock_chat_session)
        
        # Get session lock
        lock = await session_manager.get_session_lock(mock_chat_session.session_id)
        assert lock is not None
        assert isinstance(lock, asyncio.Lock)
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_active_sessions(self, session_manager):
        """Test getting list of active sessions"""
        
        await session_manager.initialize()
        
        # Initially no sessions
        sessions = session_manager.get_active_sessions()
        assert len(sessions) == 0
        
        # Register multiple sessions
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        
        # Get active sessions
        sessions = session_manager.get_active_sessions()
        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, session_manager):
        """Test getting sessions for specific users"""
        
        await session_manager.initialize()
        
        # Register sessions for different users
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user1", "rag_chat")
        session3 = MockFFChatSession("session3", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        await session_manager.register_session(session3)
        
        # Get sessions for user1
        user1_sessions = session_manager.get_user_sessions("user1")
        assert len(user1_sessions) == 2
        assert "session1" in user1_sessions
        assert "session2" in user1_sessions
        
        # Get sessions for user2
        user2_sessions = session_manager.get_user_sessions("user2")
        assert len(user2_sessions) == 1
        assert "session3" in user2_sessions
        
        # Get sessions for non-existent user
        user3_sessions = session_manager.get_user_sessions("user3")
        assert len(user3_sessions) == 0
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_sessions_by_use_case(self, session_manager):
        """Test getting sessions by use case"""
        
        await session_manager.initialize()
        
        # Register sessions with different use cases
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user2", "basic_chat")
        session3 = MockFFChatSession("session3", "user3", "rag_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        await session_manager.register_session(session3)
        
        # Get basic_chat sessions
        basic_sessions = session_manager.get_sessions_by_use_case("basic_chat")
        assert len(basic_sessions) == 2
        assert "session1" in basic_sessions
        assert "session2" in basic_sessions
        
        # Get rag_chat sessions
        rag_sessions = session_manager.get_sessions_by_use_case("rag_chat")
        assert len(rag_sessions) == 1
        assert "session3" in rag_sessions
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_processing_sessions(self, session_manager):
        """Test getting sessions that are processing"""
        
        await session_manager.initialize()
        
        # Register sessions
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        
        # Initially no sessions are processing
        processing = session_manager.get_processing_sessions()
        assert len(processing) == 0
        
        # Set one session as processing
        await session_manager.set_session_processing("session1", True)
        
        processing = session_manager.get_processing_sessions()
        assert len(processing) == 1
        assert "session1" in processing
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_session_statistics(self, session_manager):
        """Test getting session statistics"""
        
        await session_manager.initialize()
        
        # Register sessions with different properties
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user1", "rag_chat")
        session3 = MockFFChatSession("session3", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        await session_manager.register_session(session3)
        
        # Set one session as processing
        await session_manager.set_session_processing("session1", True)
        
        # Get statistics
        stats = await session_manager.get_session_statistics()
        
        assert stats["total_active_sessions"] == 3
        assert stats["processing_sessions"] == 1
        assert stats["unique_users"] == 2
        assert stats["use_case_distribution"]["basic_chat"] == 2
        assert stats["use_case_distribution"]["rag_chat"] == 1
        assert stats["average_sessions_per_user"] == 1.5
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_force_cleanup_user_sessions(self, session_manager):
        """Test force cleanup of user sessions"""
        
        await session_manager.initialize()
        
        # Register sessions for different users
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user1", "rag_chat")
        session3 = MockFFChatSession("session3", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        await session_manager.register_session(session3)
        
        assert len(session_manager.active_sessions) == 3
        
        # Force cleanup user1 sessions
        count = await session_manager.force_cleanup_user_sessions("user1")
        assert count == 2
        assert len(session_manager.active_sessions) == 1
        assert "session3" in session_manager.active_sessions
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_force_cleanup_processing_sessions(self, session_manager):
        """Test force cleanup of stuck processing sessions"""
        
        await session_manager.initialize()
        
        # Register sessions
        session1 = MockFFChatSession("session1", "user1", "basic_chat")
        session2 = MockFFChatSession("session2", "user2", "basic_chat")
        
        await session_manager.register_session(session1)
        await session_manager.register_session(session2)
        
        # Set sessions as processing and simulate old activity
        await session_manager.set_session_processing("session1", True)
        await session_manager.set_session_processing("session2", True)
        
        # Manually set old activity times to simulate stuck sessions
        old_time = datetime.now() - timedelta(seconds=session_manager.config.max_processing_time * 3)
        session_manager.active_sessions["session1"].last_activity = old_time
        session_manager.active_sessions["session2"].last_activity = old_time
        
        # Force cleanup processing sessions
        count = await session_manager.force_cleanup_processing_sessions()
        assert count == 2
        
        # Verify sessions are no longer processing
        assert not session_manager.active_sessions["session1"].processing
        assert not session_manager.active_sessions["session2"].processing
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions_integration(self, mock_ff_storage):
        """Test cleanup task integration (shorter timeouts for testing)"""
        
        # Use very short timeouts for testing
        config = FFChatSessionConfigDTO(
            session_timeout=1,  # 1 second
            cleanup_interval=1,  # 1 second
            max_processing_time=1
        )
        
        session_manager = FFChatSessionManager(mock_ff_storage, config)
        await session_manager.initialize()
        
        # Register a session
        session = MockFFChatSession("test_session", "user1", "basic_chat")
        await session_manager.register_session(session)
        
        assert len(session_manager.active_sessions) == 1
        
        # Wait for cleanup to occur (session should timeout)
        await asyncio.sleep(2.5)
        
        # Session should be cleaned up
        assert len(session_manager.active_sessions) == 0
        
        await session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, session_manager, mock_chat_session):
        """Test shutting down session manager"""
        
        await session_manager.initialize()
        
        # Register session
        await session_manager.register_session(mock_chat_session)
        assert len(session_manager.active_sessions) == 1
        
        # Shutdown
        await session_manager.shutdown()
        
        assert not session_manager._initialized
        assert len(session_manager.active_sessions) == 0
        assert len(session_manager.session_locks) == 0
        
        # Cleanup task should be cancelled
        if session_manager._cleanup_task:
            assert session_manager._cleanup_task.cancelled() or session_manager._cleanup_task.done()


class TestFFChatSessionState:
    """Test FF Chat Session State data class"""
    
    def test_session_state_creation(self):
        """Test creating session state"""
        
        state = FFChatSessionState(
            session_id="test_session",
            user_id="test_user",
            use_case="basic_chat",
            last_activity=datetime.now(),
            message_count=5,
            processing=True,
            metadata={"test": "data"}
        )
        
        assert state.session_id == "test_session"
        assert state.user_id == "test_user"
        assert state.use_case == "basic_chat"
        assert state.message_count == 5
        assert state.processing == True
        assert state.metadata["test"] == "data"
    
    def test_session_state_defaults(self):
        """Test session state with defaults"""
        
        state = FFChatSessionState(
            session_id="test_session",
            user_id="test_user",
            use_case="basic_chat",
            last_activity=datetime.now(),
            message_count=0
        )
        
        assert state.processing == False
        assert state.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])