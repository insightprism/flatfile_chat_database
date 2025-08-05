"""
Unit tests for FF Chat Application following existing FF test patterns.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
from datetime import datetime

# Import existing FF infrastructure for testing
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

# Import FF chat components to test
from ff_chat_application import FFChatApplication, create_ff_chat_app, FFChatSession
from ff_class_configs.ff_chat_application_config import FFChatApplicationConfigDTO


class TestFFChatApplication:
    """Test FF Chat Application functionality"""
    
    @pytest.fixture
    def mock_ff_storage(self):
        """Mock FF storage manager for testing"""
        storage = AsyncMock()
        storage.initialize.return_value = True
        storage.create_session.return_value = "ff_session_123"
        storage.add_message.return_value = True
        storage.get_messages.return_value = []
        storage.get_session.return_value = Mock()
        storage.search_messages.return_value = []
        return storage
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock chat session manager for testing"""
        manager = AsyncMock()
        manager.initialize.return_value = True
        manager.register_session.return_value = True
        manager.unregister_session.return_value = True
        manager.shutdown.return_value = None
        return manager
    
    @pytest.fixture
    def mock_use_case_manager(self):
        """Mock use case manager for testing"""
        manager = AsyncMock()
        manager.initialize.return_value = True
        manager.is_use_case_supported.return_value = True
        manager.get_use_case_config.return_value = {
            "description": "Test use case",
            "category": "test",
            "mode": "ff_storage"
        }
        manager.process_message.return_value = {
            "success": True,
            "response_content": "Test response",
            "processor": "test"
        }
        manager.list_use_cases.return_value = ["basic_chat", "test_case"]
        manager.get_use_case_info.return_value = {
            "use_case": "test_case",
            "description": "Test use case"
        }
        manager.shutdown.return_value = None
        return manager
    
    @pytest.fixture
    def ff_config(self):
        """FF configuration for testing"""
        return load_config()
    
    @pytest.fixture
    def chat_config(self):
        """Chat configuration for testing"""
        return FFChatApplicationConfigDTO()
    
    @pytest.mark.asyncio
    async def test_chat_application_initialization(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test FF chat application initializes correctly"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            # Create chat application
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            
            # Test initialization
            success = await chat_app.initialize()
            assert success
            assert chat_app._initialized
            
            # Verify managers were initialized
            mock_ff_storage.initialize.assert_called_once()
            mock_session_manager.initialize.assert_called_once()
            mock_use_case_manager.initialize.assert_called_once()
            
            # Cleanup
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_chat_session(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test creating chat sessions"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create chat session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat",
                title="Test Chat Session"
            )
            
            assert session_id is not None
            assert session_id.startswith("chat_")
            assert session_id in chat_app.active_sessions
            
            # Verify session properties
            session = chat_app.active_sessions[session_id]
            assert session.user_id == "test_user"
            assert session.use_case == "basic_chat"
            assert session.active == True
            
            # Verify FF storage was called
            mock_ff_storage.create_session.assert_called_once()
            mock_session_manager.register_session.assert_called_once()
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_session_unsupported_use_case(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test creating session with unsupported use case"""
        
        # Mock unsupported use case
        mock_use_case_manager.is_use_case_supported.return_value = False
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Attempt to create session with unsupported use case
            with pytest.raises(ValueError, match="Unsupported use case"):
                await chat_app.create_chat_session(
                    user_id="test_user",
                    use_case="unsupported_case"
                )
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test processing messages in chat sessions"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            # Process message
            result = await chat_app.process_message(session_id, "Hello, FF chat!")
            
            assert result["success"] == True
            assert "response_content" in result
            assert result["processor"] == "test"
            
            # Verify message was stored in FF storage
            assert mock_ff_storage.add_message.call_count >= 1
            mock_use_case_manager.process_message.assert_called_once()
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_invalid_session(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test processing message with invalid session"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Attempt to process message with invalid session
            with pytest.raises(ValueError, match="Invalid or inactive chat session"):
                await chat_app.process_message("invalid_session", "Hello")
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_session_messages(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test getting messages from chat session"""
        
        # Mock FF storage to return test messages
        test_messages = [
            FFMessageDTO(role=MessageRole.USER.value, content="Hello"),
            FFMessageDTO(role=MessageRole.ASSISTANT.value, content="Hi there!")
        ]
        mock_ff_storage.get_messages.return_value = test_messages
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            # Get messages
            messages = await chat_app.get_session_messages(session_id)
            
            assert len(messages) == 2
            assert messages[0].content == "Hello"
            assert messages[1].content == "Hi there!"
            
            # Verify FF storage was called correctly
            mock_ff_storage.get_messages.assert_called_with(
                user_id="test_user",
                session_id="ff_session_123",
                limit=None,
                offset=0
            )
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_session_info(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test getting session information"""
        
        # Mock FF session info
        mock_ff_session = Mock()
        mock_ff_session.to_dict.return_value = {"ff_info": "test"}
        mock_ff_storage.get_session.return_value = mock_ff_session
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            # Get session info
            info = await chat_app.get_session_info(session_id)
            
            assert info["chat_session_id"] == session_id
            assert info["user_id"] == "test_user"
            assert info["use_case"] == "basic_chat"
            assert info["active"] == True
            assert "ff_session_info" in info
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_close_session(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test closing chat sessions"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            assert session_id in chat_app.active_sessions
            
            # Close session
            await chat_app.close_session(session_id)
            
            assert session_id not in chat_app.active_sessions
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_list_use_cases(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test listing available use cases"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # List use cases
            use_cases = chat_app.list_use_cases()
            
            assert "basic_chat" in use_cases
            assert "test_case" in use_cases
            
            mock_use_case_manager.list_use_cases.assert_called_once()
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_use_case_info(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test getting use case information"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Get use case info
            info = await chat_app.get_use_case_info("test_case")
            
            assert info["use_case"] == "test_case"
            assert info["description"] == "Test use case"
            
            mock_use_case_manager.get_use_case_info.assert_called_once_with("test_case")
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_messages(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test searching messages across sessions"""
        
        # Mock search results
        mock_ff_storage.search_messages.return_value = [{"result": "test"}]
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            # Search messages
            results = await chat_app.search_messages(
                user_id="test_user",
                query="test query",
                session_ids=[session_id]
            )
            
            assert len(results) == 1
            assert results[0]["result"] == "test"
            
            # Verify FF storage search was called with correct parameters
            mock_ff_storage.search_messages.assert_called_once()
            
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, ff_config, chat_config, mock_ff_storage, mock_session_manager, mock_use_case_manager):
        """Test shutting down chat application"""
        
        with patch('ff_chat_application.FFCoreStorageManager', return_value=mock_ff_storage), \
             patch('ff_chat_application.FFChatSessionManager', return_value=mock_session_manager), \
             patch('ff_chat_application.FFChatUseCaseManager', return_value=mock_use_case_manager):
            
            chat_app = FFChatApplication(ff_config=ff_config, chat_config=chat_config)
            await chat_app.initialize()
            
            # Create some sessions
            session1 = await chat_app.create_chat_session("user1", "basic_chat")
            session2 = await chat_app.create_chat_session("user2", "basic_chat")
            
            assert len(chat_app.active_sessions) == 2
            
            # Shutdown
            await chat_app.shutdown()
            
            assert not chat_app._initialized
            assert len(chat_app.active_sessions) == 0
            
            # Verify managers were shut down
            mock_session_manager.shutdown.assert_called_once()
            mock_use_case_manager.shutdown.assert_called_once()


class TestFFChatApplicationConvenienceFunctions:
    """Test convenience functions for FF Chat Application"""
    
    @pytest.mark.asyncio
    async def test_create_ff_chat_app(self):
        """Test create_ff_chat_app convenience function"""
        
        with patch('ff_chat_application.FFChatApplication') as mock_app_class:
            mock_app = AsyncMock()
            mock_app.initialize.return_value = True
            mock_app_class.return_value = mock_app
            
            app = await create_ff_chat_app()
            
            assert app == mock_app
            mock_app.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_ff_chat_session_convenience(self):
        """Test create_ff_chat_session convenience function"""
        
        with patch('ff_chat_application.create_ff_chat_app') as mock_create_app:
            mock_app = AsyncMock()
            mock_app.create_chat_session.return_value = "session_123"
            mock_create_app.return_value = mock_app
            
            from ff_chat_application import create_ff_chat_session
            app, session_id = await create_ff_chat_session("test_case", "test_user")
            
            assert app == mock_app
            assert session_id == "session_123"
            mock_app.create_chat_session.assert_called_once_with("test_user", "test_case")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])