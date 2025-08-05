"""
Phase 1 FF Chat Integration Tests

Tests the complete integration of FF Chat system with existing FF infrastructure
following existing FF test patterns.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from datetime import datetime
import os

# Import existing FF infrastructure for testing
from ff_core_storage_manager import FFCoreStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config, FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

# Import FF chat components for integration testing
from ff_chat_application import FFChatApplication, create_ff_chat_app, ff_quick_chat
from ff_class_configs.ff_chat_application_config import FFChatApplicationConfigDTO
from ff_chat_session_manager import FFChatSessionManager
from ff_chat_use_case_manager import FFChatUseCaseManager


class TestFFChatPhase1Integration:
    """Test complete Phase 1 integration with existing FF infrastructure"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def ff_test_config(self, temp_data_dir):
        """FF configuration for integration testing"""
        config = load_config()
        # Use temporary directory for test data
        config.storage.base_path = temp_data_dir
        return config
    
    @pytest.fixture
    def chat_test_config(self):
        """Chat configuration for integration testing"""
        return FFChatApplicationConfigDTO(
            default_use_case="basic_chat",
            max_concurrent_sessions=10,
            enable_session_persistence=True
        )
    
    @pytest.mark.asyncio
    async def test_complete_ff_chat_integration(self, ff_test_config, chat_test_config):
        """Test complete FF chat system integration from start to finish"""
        
        # Create FF chat application with real FF backend
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_config=chat_test_config)
        
        try:
            # Test initialization
            success = await chat_app.initialize()
            assert success, "FF Chat Application should initialize successfully"
            
            # Verify FF storage is working
            assert chat_app.ff_storage is not None
            assert chat_app._initialized
            
            # Test use case listing
            use_cases = chat_app.list_use_cases()
            assert "basic_chat" in use_cases
            assert "multimodal_chat" in use_cases
            assert "rag_chat" in use_cases
            assert "memory_chat" in use_cases
            
            # Test use case info retrieval
            basic_info = await chat_app.get_use_case_info("basic_chat")
            assert basic_info["use_case"] == "basic_chat"
            assert basic_info["category"] == "basic"
            assert basic_info["mode"] == "ff_storage"
            
            # Test session creation
            session_id = await chat_app.create_chat_session(
                user_id="integration_test_user",
                use_case="basic_chat",
                title="Integration Test Session"
            )
            
            assert session_id is not None
            assert session_id.startswith("chat_")
            assert session_id in chat_app.active_sessions
            
            # Test session info retrieval
            session_info = await chat_app.get_session_info(session_id)
            assert session_info["user_id"] == "integration_test_user"
            assert session_info["use_case"] == "basic_chat"
            assert session_info["active"] == True
            assert session_info["ff_session_id"] is not None
            
            # Test message processing
            result = await chat_app.process_message(
                session_id, 
                "Hello, this is an integration test message!"
            )
            
            assert result["success"] == True
            assert "response_content" in result
            assert result["use_case"] == "basic_chat"
            assert result["processor"] == "ff_storage_basic"
            
            # Test message retrieval
            messages = await chat_app.get_session_messages(session_id)
            assert len(messages) >= 1
            
            # Find our test message
            user_messages = [msg for msg in messages if msg.role == MessageRole.USER.value]
            assert len(user_messages) >= 1
            assert any("integration test message" in msg.content for msg in user_messages)
            
            # Test multiple message exchange
            for i in range(3):
                result = await chat_app.process_message(
                    session_id, 
                    f"Test message {i + 2}"
                )
                assert result["success"] == True
            
            # Verify all messages are stored
            all_messages = await chat_app.get_session_messages(session_id)
            user_messages = [msg for msg in all_messages if msg.role == MessageRole.USER.value]
            assert len(user_messages) >= 4  # Original + 3 more
            
            # Test session closure
            await chat_app.close_session(session_id)
            assert session_id not in chat_app.active_sessions
            
        finally:
            # Cleanup
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, ff_test_config, chat_test_config):
        """Test handling multiple concurrent chat sessions"""
        
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_config=chat_test_config)
        
        try:
            await chat_app.initialize()
            
            # Create multiple sessions for different users and use cases
            sessions = []
            for i in range(5):
                use_case = ["basic_chat", "multimodal_chat", "rag_chat", "memory_chat"][i % 4]
                session_id = await chat_app.create_chat_session(
                    user_id=f"user_{i}",
                    use_case=use_case,
                    title=f"Concurrent Session {i}"
                )
                sessions.append(session_id)
            
            assert len(chat_app.active_sessions) == 5
            
            # Process messages concurrently in all sessions
            async def process_messages_in_session(session_id, user_num):
                for msg_num in range(3):
                    result = await chat_app.process_message(
                        session_id, 
                        f"Message {msg_num} from user {user_num}"
                    )
                    assert result["success"] == True
                return session_id
            
            # Run concurrent processing
            tasks = [
                process_messages_in_session(session_id, i)
                for i, session_id in enumerate(sessions)
            ]
            
            completed_sessions = await asyncio.gather(*tasks)
            assert len(completed_sessions) == 5
            
            # Verify all sessions still active
            active_sessions = chat_app.list_active_sessions()
            assert len(active_sessions) == 5
            
            # Test search across all sessions
            search_results = await chat_app.search_messages(
                user_id="user_0",
                query="Message"
            )
            # Should find results (exact behavior depends on FF search implementation)
            assert isinstance(search_results, list)
            
        finally:
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_multimodal_use_case_integration(self, ff_test_config, chat_test_config):
        """Test multimodal use case integration with FF document processing"""
        
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_config=chat_test_config)
        
        try:
            await chat_app.initialize()
            
            # Create multimodal session
            session_id = await chat_app.create_chat_session(
                user_id="multimodal_user",
                use_case="multimodal_chat",
                title="Multimodal Test"
            )
            
            # Test text message
            result = await chat_app.process_message(session_id, "Hello multimodal!")
            assert result["success"] == True
            assert result["use_case"] == "multimodal_chat"
            
            # Test message with attachments
            multimodal_message = {
                "content": "Here's a message with attachments",
                "attachments": [
                    {"type": "image", "name": "test.jpg", "size": 1024},
                    {"type": "pdf", "name": "document.pdf", "size": 2048}
                ]
            }
            
            result = await chat_app.process_message(session_id, multimodal_message)
            assert result["success"] == True
            assert "attachments" in result["response_content"].lower()
            
        finally:
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_rag_use_case_integration(self, ff_test_config, chat_test_config):
        """Test RAG use case integration with FF vector storage"""
        
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_config=chat_test_config)
        
        try:
            await chat_app.initialize()
            
            # Create RAG session
            session_id = await chat_app.create_chat_session(
                user_id="rag_user",
                use_case="rag_chat",
                title="RAG Test"
            )
            
            # Test RAG message processing
            result = await chat_app.process_message(
                session_id, 
                "What can you tell me about machine learning?"
            )
            
            assert result["success"] == True
            assert result["use_case"] == "rag_chat"
            assert result["processor"] == "ff_enhanced_basic"
            assert "rag" in result["response_content"].lower()
            
        finally:
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_use_case_integration(self, ff_test_config, chat_test_config):
        """Test memory use case integration with FF storage"""
        
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_test_config)
        
        try:
            await chat_app.initialize()
            
            # Create memory session
            session_id = await chat_app.create_chat_session(
                user_id="memory_user",
                use_case="memory_chat",
                title="Memory Test"
            )
            
            # Test memory message processing
            result = await chat_app.process_message(
                session_id, 
                "Please remember that I like Python programming."
            )
            
            assert result["success"] == True
            assert result["use_case"] == "memory_chat"
            assert "memory" in result["response_content"].lower()
            
            # Test follow-up message that should use memory
            result = await chat_app.process_message(
                session_id, 
                "What do I like?"
            )
            
            assert result["success"] == True
            
        finally:
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_session_manager_integration(self, ff_test_config):
        """Test session manager integration with FF storage"""
        
        # Test session manager independently
        ff_storage = FFCoreStorageManager(ff_test_config)
        await ff_storage.initialize()
        
        try:
            session_config = chat_test_config.chat_session
            session_manager = FFChatSessionManager(ff_storage, session_config)
            
            await session_manager.initialize()
            
            # Create mock session
            from ff_chat_application import FFChatSession
            test_session = FFChatSession(
                session_id="test_session_integration",
                user_id="test_user",
                use_case="basic_chat",
                ff_storage_session_id="ff_session_123",
                context={"test": "data"},
                created_at=datetime.now()
            )
            
            # Test session registration
            success = await session_manager.register_session(test_session)
            assert success
            
            # Test session state retrieval
            state = await session_manager.get_session_state(test_session.session_id)
            assert state is not None
            assert state["user_id"] == "test_user"
            
            # Test activity updates
            await session_manager.update_session_activity(test_session.session_id)
            await session_manager.increment_message_count(test_session.session_id)
            
            updated_state = await session_manager.get_session_state(test_session.session_id)
            assert updated_state["message_count"] == 1
            
            # Test statistics
            stats = await session_manager.get_session_statistics()
            assert stats["total_active_sessions"] == 1
            
            await session_manager.shutdown()
            
        finally:
            # Note: Don't shutdown FF storage as it might be used elsewhere
            pass
    
    @pytest.mark.asyncio
    async def test_use_case_manager_integration(self, ff_test_config):
        """Test use case manager integration with FF storage"""
        
        ff_storage = FFCoreStorageManager(ff_test_config)
        await ff_storage.initialize()
        
        try:
            from ff_class_configs.ff_chat_use_case_config import FFChatUseCaseConfigDTO
            use_case_config = FFChatUseCaseConfigDTO()
            use_case_manager = FFChatUseCaseManager(ff_storage, use_case_config)
            
            await use_case_manager.initialize()
            
            # Test use case support
            assert await use_case_manager.is_use_case_supported("basic_chat")
            assert not await use_case_manager.is_use_case_supported("nonexistent_case")
            
            # Test use case listing
            use_cases = use_case_manager.list_use_cases()
            assert len(use_cases) >= 4
            assert "basic_chat" in use_cases
            
            # Test use case configuration
            config = await use_case_manager.get_use_case_config("basic_chat")
            assert config["mode"] == "ff_storage"
            assert config["category"] == "basic"
            
            # Test processing statistics
            stats = use_case_manager.get_processing_statistics()
            assert "use_case_stats" in stats
            assert stats["total_use_cases"] >= 4
            
            await use_case_manager.shutdown()
            
        finally:
            pass
    
    @pytest.mark.asyncio
    async def test_convenience_functions_integration(self, ff_test_config):
        """Test convenience functions with real FF backend"""
        
        # Set environment for testing
        original_config = os.environ.get("CHATDB_BASE_PATH")
        os.environ["CHATDB_BASE_PATH"] = ff_test_config.storage.base_path
        
        try:
            # Test create_ff_chat_app
            app = await create_ff_chat_app()
            assert app._initialized
            await app.shutdown()
            
            # Test ff_quick_chat
            response = await ff_quick_chat(
                "Hello from quick chat test!",
                use_case="basic_chat",
                user_id="quick_test_user"
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            
        finally:
            # Restore environment
            if original_config:
                os.environ["CHATDB_BASE_PATH"] = original_config
            else:
                os.environ.pop("CHATDB_BASE_PATH", None)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, ff_test_config, chat_test_config):
        """Test error handling in integrated system"""
        
        chat_app = FFChatApplication(ff_config=ff_test_config, chat_config=chat_test_config)
        
        try:
            await chat_app.initialize()
            
            # Test invalid session operations
            with pytest.raises(ValueError, match="Invalid or inactive chat session"):
                await chat_app.process_message("invalid_session", "test")
            
            with pytest.raises(ValueError, match="Chat session not found"):
                await chat_app.get_session_messages("invalid_session")
            
            with pytest.raises(ValueError, match="Chat session not found"):
                await chat_app.get_session_info("invalid_session")
            
            # Test unsupported use case
            with pytest.raises(ValueError, match="Unsupported use case"):
                await chat_app.create_chat_session(
                    user_id="test_user",
                    use_case="nonexistent_use_case"
                )
            
            # Test graceful session closure of non-existent session (should not raise)
            await chat_app.close_session("nonexistent_session")
            
        finally:
            await chat_app.shutdown()
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, temp_data_dir):
        """Test configuration system integration"""
        
        # Test with custom configuration
        custom_ff_config = FFConfigurationManagerConfigDTO.from_environment("test")
        custom_ff_config.storage.base_path = temp_data_dir
        
        custom_chat_config = FFChatApplicationConfigDTO(
            default_use_case="basic_chat",
            max_concurrent_sessions=5,
            enable_session_persistence=True,
            session_id_length=16
        )
        
        chat_app = FFChatApplication(
            ff_config=custom_ff_config, 
            chat_config=custom_chat_config
        )
        
        try:
            await chat_app.initialize()
            
            # Verify custom configuration is used
            assert chat_app.chat_config.max_concurrent_sessions == 5
            assert chat_app.chat_config.session_id_length == 16
            
            # Test session creation with custom config
            session_id = await chat_app.create_chat_session(
                user_id="config_test_user",
                use_case="basic_chat"
            )
            
            # Session ID should have custom length (excluding "chat_" prefix)
            assert len(session_id) == len("chat_") + 16
            
            # Test configuration summary
            summary = chat_app.chat_config.get_chat_summary()
            assert "chat_config" in summary
            assert summary["chat_config"]["max_concurrent_sessions"] == 5
            
        finally:
            await chat_app.shutdown()


class TestFFChatBackwardCompatibility:
    """Test that existing FF functionality remains unchanged"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def ff_test_config(self, temp_data_dir):
        """FF configuration for backward compatibility testing"""
        config = load_config()
        config.storage.base_path = temp_data_dir
        return config
    
    @pytest.mark.asyncio
    async def test_ff_storage_manager_unchanged(self, ff_test_config):
        """Test that existing FF storage manager functionality is unchanged"""
        
        # Test existing FF storage manager directly
        ff_storage = FFCoreStorageManager(ff_test_config)
        await ff_storage.initialize()
        
        try:
            # Test existing FF operations work exactly as before
            
            # Create user
            await ff_storage.create_user("backward_compat_user")
            
            # Create session
            session_id = await ff_storage.create_session(
                "backward_compat_user", 
                "Backward Compatibility Test Session"
            )
            assert session_id is not None
            
            # Add messages
            message1 = FFMessageDTO(
                role=MessageRole.USER.value, 
                content="Test message 1"
            )
            message2 = FFMessageDTO(
                role=MessageRole.ASSISTANT.value, 
                content="Test response 1"
            )
            
            await ff_storage.add_message("backward_compat_user", session_id, message1)
            await ff_storage.add_message("backward_compat_user", session_id, message2)
            
            # Retrieve messages
            messages = await ff_storage.get_messages("backward_compat_user", session_id)
            assert len(messages) == 2
            assert messages[0].content == "Test message 1"
            assert messages[1].content == "Test response 1"
            
            # Get session info
            session_info = await ff_storage.get_session("backward_compat_user", session_id)
            assert session_info is not None
            
            # List sessions
            sessions = await ff_storage.list_sessions("backward_compat_user")
            assert len(sessions) >= 1
            
            # Search messages (if implemented)
            try:
                search_results = await ff_storage.search_messages(
                    "backward_compat_user", 
                    "Test message"
                )
                assert isinstance(search_results, list)
            except NotImplementedError:
                # Search might not be implemented yet
                pass
            
        finally:
            # Clean up - but don't shutdown as other tests might use it
            pass
    
    @pytest.mark.asyncio
    async def test_ff_configuration_compatibility(self, temp_data_dir):
        """Test that existing FF configuration system works unchanged"""
        
        # Test loading existing FF configuration
        ff_config = load_config()
        assert ff_config is not None
        assert hasattr(ff_config, 'storage')
        assert hasattr(ff_config, 'search')
        assert hasattr(ff_config, 'vector')
        
        # Test configuration modification works as before
        ff_config.storage.base_path = temp_data_dir
        assert ff_config.storage.base_path == temp_data_dir
        
        # Test configuration validation
        errors = ff_config.validate_all()
        assert isinstance(errors, list)
        
        # Test configuration summary
        summary = ff_config.get_summary()
        assert isinstance(summary, dict)
        assert "base_path" in summary
    
    @pytest.mark.asyncio
    async def test_ff_utilities_unchanged(self):
        """Test that FF utilities work unchanged"""
        
        # Test FF logging
        from ff_utils.ff_logging import get_logger
        logger = get_logger("test_backward_compat")
        assert logger is not None
        
        # Test that we can log without issues
        logger.info("Backward compatibility test log message")
        
        # Test FF validation (if available)
        try:
            from ff_utils.ff_validation import validate_user_id
            # This might not exist yet, but if it does, it should work
            result = validate_user_id("test_user")
            assert isinstance(result, bool)
        except ImportError:
            # Expected if validation utils don't exist yet
            pass
    
    @pytest.mark.asyncio 
    async def test_existing_ff_protocols_unchanged(self):
        """Test that existing FF protocols are unchanged"""
        
        # Test importing existing protocols
        from ff_protocols.ff_storage_protocol import StorageProtocol
        from ff_protocols.ff_search_protocol import SearchProtocol
        
        # These should import without issues
        assert StorageProtocol is not None
        assert SearchProtocol is not None
        
        # Test that new chat protocols don't interfere
        from ff_protocols.ff_chat_protocol import FFChatApplicationProtocol
        assert FFChatApplicationProtocol is not None
        
        # Both old and new protocols should coexist
        assert StorageProtocol != FFChatApplicationProtocol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])