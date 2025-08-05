"""
Unit tests for FF Chat Use Case Manager following existing FF test patterns.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from pathlib import Path
import yaml
import tempfile
from datetime import datetime

# Import FF chat components to test
from ff_chat_use_case_manager import FFChatUseCaseManager
from ff_class_configs.ff_chat_use_case_config import FFChatUseCaseConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole


@pytest.fixture
def mock_ff_storage():
    """Mock FF storage manager for testing"""
    storage = AsyncMock()
    storage.get_messages.return_value = []
    return storage


@pytest.fixture
def use_case_config():
    """Use case configuration for testing"""
    return FFChatUseCaseConfigDTO(
        config_file_path="test_config.yaml",
        validate_use_cases=True,
        enable_fallback_processing=True
    )


@pytest.fixture
def use_case_manager(mock_ff_storage, use_case_config):
    """Use case manager instance for testing"""
    return FFChatUseCaseManager(mock_ff_storage, use_case_config)


@pytest.fixture
def mock_chat_session():
    """Mock chat session for testing"""
    session = Mock()
    session.session_id = "test_session_123"
    session.user_id = "test_user"
    session.use_case = "basic_chat"
    session.ff_storage_session_id = "ff_session_123"
    session.context = {"test": "data"}
    return session


@pytest.fixture
def test_message():
    """Test message for processing"""
    return FFMessageDTO(
        role=MessageRole.USER.value,
        content="Hello, this is a test message",
        attachments=[]
    )


class TestFFChatUseCaseManager:
    """Test FF Chat Use Case Manager functionality"""
    
    @pytest.mark.asyncio
    async def test_use_case_manager_initialization(self, use_case_manager):
        """Test use case manager initializes correctly"""
        
        with patch.object(use_case_manager, '_load_use_case_definitions') as mock_load:
            mock_load.return_value = None
            
            # Test initialization
            success = await use_case_manager.initialize()
            assert success
            assert use_case_manager._initialized
            
            # Verify use case definitions were loaded
            mock_load.assert_called_once()
            
            # Verify processing stats were initialized
            assert isinstance(use_case_manager.processing_stats, dict)
            
            await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_load_default_use_case_definitions(self, use_case_manager):
        """Test loading default use case definitions"""
        
        # Test with non-existent config file
        with patch('pathlib.Path.exists', return_value=False):
            await use_case_manager._load_use_case_definitions()
        
        # Should have default use cases
        assert "basic_chat" in use_case_manager.use_case_definitions
        assert "multimodal_chat" in use_case_manager.use_case_definitions
        assert "rag_chat" in use_case_manager.use_case_definitions
        assert "memory_chat" in use_case_manager.use_case_definitions
        
        # Verify structure of basic_chat definition
        basic_chat = use_case_manager.use_case_definitions["basic_chat"]
        assert basic_chat["description"] is not None
        assert basic_chat["category"] == "basic"
        assert basic_chat["mode"] == "ff_storage"
        assert "ff_text_chat" in basic_chat["components"]
    
    @pytest.mark.asyncio
    async def test_load_use_case_definitions_from_file(self, use_case_manager):
        """Test loading use case definitions from YAML file"""
        
        test_yaml_content = """
use_cases:
  test_use_case:
    description: "Test use case from file"
    category: "test"
    components: ["ff_test"]
    mode: "ff_storage"
    enabled: true
    settings:
      test_setting: "value"
"""
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=test_yaml_content)):
            
            await use_case_manager._load_use_case_definitions()
        
        # Should have loaded test use case from file
        assert "test_use_case" in use_case_manager.use_case_definitions
        test_case = use_case_manager.use_case_definitions["test_use_case"]
        assert test_case["description"] == "Test use case from file"
        assert test_case["category"] == "test"
        assert test_case["mode"] == "ff_storage"
    
    @pytest.mark.asyncio
    async def test_validate_use_case_definitions(self, use_case_manager):
        """Test use case definition validation"""
        
        # Set up invalid use case definition
        use_case_manager.use_case_definitions = {
            "valid_case": {
                "description": "Valid case",
                "category": "test",
                "components": ["ff_test"],
                "mode": "ff_storage"
            },
            "invalid_case": {
                "description": "Missing required fields"
                # Missing category, components, mode
            }
        }
        
        # Should raise error for invalid definition when fallback is disabled
        use_case_manager.config.enable_fallback_processing = False
        
        with pytest.raises(ValueError):
            await use_case_manager._validate_use_case_definitions()
        
        # Should not raise error when fallback is enabled
        use_case_manager.config.enable_fallback_processing = True
        await use_case_manager._validate_use_case_definitions()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_is_use_case_supported(self, use_case_manager):
        """Test checking if use cases are supported"""
        
        await use_case_manager.initialize()
        
        # Test supported use case
        assert await use_case_manager.is_use_case_supported("basic_chat")
        
        # Test unsupported use case
        assert not await use_case_manager.is_use_case_supported("nonexistent_case")
        
        # Test disabled use case
        use_case_manager.use_case_definitions["disabled_case"] = {
            "enabled": False,
            "description": "Disabled case"
        }
        assert not await use_case_manager.is_use_case_supported("disabled_case")
        
        await use_case_manager.shutdown()
    
    def test_list_use_cases(self, use_case_manager):
        """Test listing available use cases"""
        
        use_case_manager.use_case_definitions = {
            "enabled_case1": {"enabled": True},
            "enabled_case2": {"enabled": True},
            "disabled_case": {"enabled": False},
            "default_case": {}  # Should default to enabled
        }
        
        use_cases = use_case_manager.list_use_cases()
        
        assert "enabled_case1" in use_cases
        assert "enabled_case2" in use_cases
        assert "default_case" in use_cases
        assert "disabled_case" not in use_cases
    
    def test_list_all_use_cases(self, use_case_manager):
        """Test listing all use cases including disabled ones"""
        
        use_case_manager.use_case_definitions = {
            "enabled_case": {"enabled": True},
            "disabled_case": {"enabled": False}
        }
        
        all_use_cases = use_case_manager.list_all_use_cases()
        
        assert "enabled_case" in all_use_cases
        assert "disabled_case" in all_use_cases
        assert len(all_use_cases) == 2
    
    @pytest.mark.asyncio
    async def test_get_use_case_config(self, use_case_manager):
        """Test getting use case configuration"""
        
        test_config = {
            "description": "Test case",
            "settings": {"test": "value"}
        }
        use_case_manager.use_case_definitions["test_case"] = test_config
        
        # Test existing use case
        config = await use_case_manager.get_use_case_config("test_case")
        assert config == test_config
        assert config is not test_config  # Should be a copy
        
        # Test non-existent use case
        with pytest.raises(ValueError, match="Unknown use case"):
            await use_case_manager.get_use_case_config("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_use_case_info(self, use_case_manager):
        """Test getting detailed use case information"""
        
        await use_case_manager.initialize()
        
        # Get info for basic_chat
        info = await use_case_manager.get_use_case_info("basic_chat")
        
        assert info["use_case"] == "basic_chat"
        assert "description" in info
        assert "category" in info
        assert "components" in info
        assert "mode" in info
        assert "statistics" in info
        
        await use_case_manager.shutdown()
    
    def test_get_use_cases_by_category(self, use_case_manager):
        """Test getting use cases by category"""
        
        use_case_manager.use_case_definitions = {
            "basic1": {"category": "basic", "enabled": True},
            "basic2": {"category": "basic", "enabled": True},
            "advanced1": {"category": "advanced", "enabled": True},
            "disabled_basic": {"category": "basic", "enabled": False}
        }
        
        basic_cases = use_case_manager.get_use_cases_by_category("basic")
        advanced_cases = use_case_manager.get_use_cases_by_category("advanced")
        
        assert len(basic_cases) == 2
        assert "basic1" in basic_cases
        assert "basic2" in basic_cases
        assert "disabled_basic" not in basic_cases
        
        assert len(advanced_cases) == 1
        assert "advanced1" in advanced_cases
    
    def test_get_use_cases_by_mode(self, use_case_manager):
        """Test getting use cases by processing mode"""
        
        use_case_manager.use_case_definitions = {
            "storage1": {"mode": "ff_storage", "enabled": True},
            "storage2": {"mode": "ff_storage", "enabled": True},
            "enhanced1": {"mode": "ff_enhanced", "enabled": True},
            "disabled_storage": {"mode": "ff_storage", "enabled": False}
        }
        
        storage_cases = use_case_manager.get_use_cases_by_mode("ff_storage")
        enhanced_cases = use_case_manager.get_use_cases_by_mode("ff_enhanced")
        
        assert len(storage_cases) == 2
        assert "storage1" in storage_cases
        assert "storage2" in storage_cases
        assert "disabled_storage" not in storage_cases
        
        assert len(enhanced_cases) == 1
        assert "enhanced1" in enhanced_cases
    
    @pytest.mark.asyncio
    async def test_process_message_ff_storage(self, use_case_manager, mock_chat_session, test_message):
        """Test processing message with FF storage mode"""
        
        await use_case_manager.initialize()
        
        # Mock get_messages to return some context
        use_case_manager.ff_storage.get_messages.return_value = [
            FFMessageDTO(role=MessageRole.USER.value, content="Previous message")
        ]
        
        # Process message
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == True
        assert "response_content" in result
        assert result["processor"] == "ff_storage_basic"
        assert result["use_case"] == "basic_chat"
        assert result["processing_mode"] == "ff_storage"
        assert result["message_count"] == 2  # 1 previous + 1 new
        
        # Verify FF storage was called
        use_case_manager.ff_storage.get_messages.assert_called_once()
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_ff_enhanced(self, use_case_manager, mock_chat_session, test_message):
        """Test processing message with FF enhanced mode"""
        
        await use_case_manager.initialize()
        
        # Set session to use enhanced mode
        mock_chat_session.use_case = "rag_chat"
        
        # Mock get_messages
        use_case_manager.ff_storage.get_messages.return_value = []
        
        # Process message
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == True
        assert result["processor"] == "ff_enhanced_basic"
        assert result["processing_mode"] == "ff_enhanced"
        assert "[RAG context would be retrieved here]" in result["response_content"]
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_multimodal(self, use_case_manager, mock_chat_session):
        """Test processing message with multimodal attachments"""
        
        await use_case_manager.initialize()
        
        # Set session to multimodal
        mock_chat_session.use_case = "multimodal_chat"
        
        # Create message with attachments
        message_with_attachments = FFMessageDTO(
            role=MessageRole.USER.value,
            content="Test with attachments",
            attachments=[{"type": "image", "name": "test.jpg"}, {"type": "pdf", "name": "doc.pdf"}]
        )
        
        # Mock get_messages
        use_case_manager.ff_storage.get_messages.return_value = []
        
        # Process message
        result = await use_case_manager.process_message(mock_chat_session, message_with_attachments)
        
        assert result["success"] == True
        assert "[Processed 2 attachments]" in result["response_content"]
        assert result["attachments_processed"] == 2
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_memory_chat(self, use_case_manager, mock_chat_session, test_message):
        """Test processing message with memory chat use case"""
        
        await use_case_manager.initialize()
        
        # Set session to memory chat
        mock_chat_session.use_case = "memory_chat"
        
        # Mock get_messages
        use_case_manager.ff_storage.get_messages.return_value = []
        
        # Process message
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == True
        assert "[Memory context would be retrieved here]" in result["response_content"]
        assert result["memory_context"] == "episodic"
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_unsupported_use_case(self, use_case_manager, mock_chat_session, test_message):
        """Test processing message with unsupported use case"""
        
        await use_case_manager.initialize()
        
        # Set unsupported use case
        mock_chat_session.use_case = "unsupported_case"
        
        # Process message
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == False
        assert "error" in result
        assert "not supported" in result["error"]
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_process_message_with_fallback(self, use_case_manager, mock_chat_session, test_message):
        """Test processing message with fallback when main processing fails"""
        
        await use_case_manager.initialize()
        
        # Mock FF storage to raise exception
        use_case_manager.ff_storage.get_messages.side_effect = Exception("Storage error")
        
        # Process message (should trigger fallback)
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == True
        assert result["processor"] == "ff_fallback"
        assert result["processing_mode"] == "fallback"
        assert "fallback" in result["response_content"].lower()
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_processing_statistics_tracking(self, use_case_manager, mock_chat_session, test_message):
        """Test that processing statistics are tracked correctly"""
        
        await use_case_manager.initialize()
        
        # Mock get_messages
        use_case_manager.ff_storage.get_messages.return_value = []
        
        # Process several messages
        for i in range(3):
            await use_case_manager.process_message(mock_chat_session, test_message)
        
        # Check statistics
        stats = use_case_manager.processing_stats["basic_chat"]
        assert stats["total_processed"] == 3
        assert stats["successful_processed"] == 3
        assert stats["failed_processed"] == 0
        assert stats["average_processing_time"] > 0
        assert stats["last_processed"] is not None
        
        await use_case_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_processing_statistics_with_failures(self, use_case_manager, mock_chat_session, test_message):
        """Test statistics tracking with failures"""
        
        await use_case_manager.initialize()
        
        # Disable fallback to force failures
        use_case_manager.config.enable_fallback_processing = False
        
        # Mock FF storage to fail
        use_case_manager.ff_storage.get_messages.side_effect = Exception("Storage error")
        
        # Process message (should fail)
        result = await use_case_manager.process_message(mock_chat_session, test_message)
        
        assert result["success"] == False
        
        # Check failure statistics
        stats = use_case_manager.processing_stats["basic_chat"]
        assert stats["total_processed"] == 1
        assert stats["successful_processed"] == 0
        assert stats["failed_processed"] == 1
        
        await use_case_manager.shutdown()
    
    def test_get_processing_statistics(self, use_case_manager):
        """Test getting overall processing statistics"""
        
        # Set up some processing stats
        use_case_manager.use_case_definitions = {
            "case1": {"enabled": True},
            "case2": {"enabled": True},
            "case3": {"enabled": False}
        }
        use_case_manager.processing_stats = {
            "case1": {"total_processed": 10},
            "case2": {"total_processed": 5}
        }
        
        stats = use_case_manager.get_processing_statistics()
        
        assert stats["total_use_cases"] == 3
        assert stats["enabled_use_cases"] == 2
        assert stats["disabled_use_cases"] == 1
        assert "use_case_stats" in stats
    
    @pytest.mark.asyncio
    async def test_reload_use_case_definitions(self, use_case_manager):
        """Test reloading use case definitions"""
        
        with patch.object(use_case_manager, '_load_use_case_definitions') as mock_load:
            mock_load.return_value = None
            
            success = await use_case_manager.reload_use_case_definitions()
            assert success
            
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enable_disable_use_case(self, use_case_manager):
        """Test enabling and disabling use cases"""
        
        use_case_manager.use_case_definitions = {
            "test_case": {"enabled": True}
        }
        
        # Test disabling
        success = await use_case_manager.disable_use_case("test_case")
        assert success
        assert not use_case_manager.use_case_definitions["test_case"]["enabled"]
        
        # Test enabling
        success = await use_case_manager.enable_use_case("test_case")
        assert success
        assert use_case_manager.use_case_definitions["test_case"]["enabled"]
        
        # Test non-existent use case
        success = await use_case_manager.enable_use_case("nonexistent")
        assert not success
    
    @pytest.mark.asyncio
    async def test_shutdown(self, use_case_manager):
        """Test shutting down use case manager"""
        
        await use_case_manager.initialize()
        
        # Add some data
        use_case_manager.use_case_definitions["test"] = {"test": "data"}
        use_case_manager.processing_stats["test"] = {"total": 1}
        
        # Shutdown
        await use_case_manager.shutdown()
        
        assert not use_case_manager._initialized
        assert len(use_case_manager.use_case_definitions) == 0
        assert len(use_case_manager.processing_stats) == 0


class TestFFChatUseCaseManagerEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_ff_storage, use_case_config):
        """Test handling initialization failures"""
        
        use_case_manager = FFChatUseCaseManager(mock_ff_storage, use_case_config)
        
        with patch.object(use_case_manager, '_load_use_case_definitions') as mock_load:
            mock_load.side_effect = Exception("Load error")
            
            success = await use_case_manager.initialize()
            assert not success
            assert not use_case_manager._initialized
    
    @pytest.mark.asyncio
    async def test_yaml_parsing_error(self, use_case_manager):
        """Test handling YAML parsing errors"""
        
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=invalid_yaml)):
            
            # Should fallback to defaults on YAML error
            await use_case_manager._load_use_case_definitions()
            
            # Should have default use cases
            assert "basic_chat" in use_case_manager.use_case_definitions
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, use_case_manager, test_message):
        """Test concurrent message processing"""
        
        await use_case_manager.initialize()
        
        # Mock FF storage
        use_case_manager.ff_storage.get_messages.return_value = []
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = Mock()
            session.session_id = f"session_{i}"
            session.user_id = f"user_{i}"
            session.use_case = "basic_chat"
            session.ff_storage_session_id = f"ff_session_{i}"
            sessions.append(session)
        
        # Process messages concurrently
        tasks = [
            use_case_manager.process_message(session, test_message)
            for session in sessions
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result["success"] == True
        
        # Statistics should be updated correctly
        stats = use_case_manager.processing_stats["basic_chat"]
        assert stats["total_processed"] == 3
        
        await use_case_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])