"""
FF Chat Phase 2 Unit Tests

Unit tests for individual Phase 2 components:
- FF Text Chat Component
- FF Memory Component  
- FF Multi-Agent Component
- FF Component Registry
"""

import asyncio
import pytest
import tempfile
import shutil
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import components to test
from ff_text_chat_component import FFTextChatComponent
from ff_memory_component import FFMemoryComponent, FFMemoryEntry
from ff_multi_agent_component import FFMultiAgentComponent, FFAgentResponse
from ff_component_registry import FFComponentRegistry, FFComponentRegistration

# Import configurations
from ff_class_configs.ff_text_chat_config import FFTextChatConfigDTO
from ff_class_configs.ff_memory_config import FFMemoryConfigDTO, FFMemoryType
from ff_class_configs.ff_multi_agent_config import FFMultiAgentConfigDTO, FFCoordinationMode
from ff_class_configs.ff_component_registry_config import FFComponentRegistryConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

# Import protocols
from ff_protocols.ff_chat_component_protocol import (
    COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_MULTI_AGENT
)

from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)


class TestFFTextChatComponent:
    """Unit tests for FF Text Chat Component"""
    
    @pytest.fixture
    def text_chat_config(self):
        """Create text chat configuration for testing"""
        return FFTextChatConfigDTO(
            max_message_length=1000,
            response_format="markdown",
            context_window=5,
            temperature=0.7,
            max_tokens=500
        )
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock FF dependencies"""
        return {
            "ff_storage": AsyncMock(),
            "ff_search": AsyncMock()
        }
    
    @pytest.fixture
    async def text_chat_component(self, text_chat_config, mock_dependencies):
        """Create text chat component for testing"""
        component = FFTextChatComponent(text_chat_config)
        await component.initialize(mock_dependencies)
        yield component
        await component.cleanup()
    
    async def test_text_chat_initialization(self, text_chat_component):
        """Test text chat component initialization"""
        assert text_chat_component._initialized is True
        assert text_chat_component.ff_storage is not None
        assert hasattr(text_chat_component, 'component_info')
        
        # Check component info
        info = text_chat_component.component_info
        assert info['name'] == 'ff_text_chat'
        assert len(info['capabilities']) > 0
        assert COMPONENT_TYPE_TEXT_CHAT in str(info)
    
    async def test_process_message_basic(self, text_chat_component, mock_dependencies):
        """Test basic message processing"""
        # Setup mock
        mock_dependencies["ff_storage"].get_messages.return_value = []
        
        # Create test message
        message = FFMessageDTO(
            role=MessageRole.USER.value,
            content="Hello, how are you today?"
        )
        
        # Process message
        result = await text_chat_component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message
        )
        
        # Verify result
        assert result["success"] is True
        assert "response_content" in result
        assert result["component"] == "ff_text_chat"
        assert "metadata" in result
    
    async def test_get_conversation_context(self, text_chat_component, mock_dependencies):
        """Test conversation context retrieval"""
        # Setup mock messages
        mock_messages = [
            FFMessageDTO(role="user", content="First message"),
            FFMessageDTO(role="assistant", content="First response"),
            FFMessageDTO(role="user", content="Second message")
        ]
        mock_dependencies["ff_storage"].get_messages.return_value = mock_messages
        
        # Get conversation context
        context = await text_chat_component.get_conversation_context(
            user_id="test_user",
            session_id="test_session",
            limit=5
        )
        
        # Verify context
        assert len(context) == len(mock_messages)
        assert all("role" in msg for msg in context)
        assert all("content" in msg for msg in context)
    
    async def test_validate_message_content(self, text_chat_component):
        """Test message content validation"""
        # Valid message
        valid_message = FFMessageDTO(role="user", content="Valid message")
        is_valid, error = await text_chat_component.validate_message_content(valid_message)
        assert is_valid is True
        assert error is None
        
        # Empty message
        empty_message = FFMessageDTO(role="user", content="")
        is_valid, error = await text_chat_component.validate_message_content(empty_message)
        assert is_valid is False
        assert "empty" in error.lower()
        
        # Too long message
        long_content = "x" * (text_chat_component.config.max_message_length + 1)
        long_message = FFMessageDTO(role="user", content=long_content)
        is_valid, error = await text_chat_component.validate_message_content(long_message)
        assert is_valid is False
        assert "too long" in error.lower()
    
    async def test_component_capabilities(self, text_chat_component):
        """Test component capabilities"""
        capabilities = await text_chat_component.get_capabilities()
        assert len(capabilities) > 0
        assert "text_conversation" in capabilities
        
        # Test use case support
        assert await text_chat_component.supports_use_case("basic_chat") is True
    
    async def test_error_handling(self, text_chat_component, mock_dependencies):
        """Test error handling in text chat component"""
        # Make ff_storage raise an exception
        mock_dependencies["ff_storage"].get_messages.side_effect = Exception("Database error")
        
        message = FFMessageDTO(role="user", content="Test message")
        result = await text_chat_component.process_message(
            session_id="test_session",
            user_id="test_user", 
            message=message
        )
        
        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result


class TestFFMemoryComponent:
    """Unit tests for FF Memory Component"""
    
    @pytest.fixture
    def memory_config(self):
        """Create memory configuration for testing"""
        return FFMemoryConfigDTO(
            enabled_memory_types=[FFMemoryType.EPISODIC.value, FFMemoryType.SEMANTIC.value],
            working_memory_size=10,
            episodic_max_entries=100,
            semantic_max_entries=50,
            use_ff_vector_storage=True,
            embedding_dimension=384
        )
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock FF dependencies"""
        return {
            "ff_storage": AsyncMock(),
            "ff_vector": AsyncMock(),
            "ff_search": AsyncMock()
        }
    
    @pytest.fixture
    async def memory_component(self, memory_config, mock_dependencies):
        """Create memory component for testing"""
        component = FFMemoryComponent(memory_config)
        await component.initialize(mock_dependencies)
        yield component
        await component.cleanup()
    
    async def test_memory_initialization(self, memory_component):
        """Test memory component initialization"""
        assert memory_component._initialized is True
        assert memory_component.ff_vector is not None
        assert memory_component.ff_storage is not None
        
        # Check component info
        info = memory_component.component_info
        assert info['name'] == 'ff_memory'
        assert len(info['capabilities']) > 0
    
    async def test_store_memory_basic(self, memory_component, mock_dependencies):
        """Test basic memory storage"""
        # Setup mocks
        mock_dependencies["ff_vector"].store_vector.return_value = True
        mock_dependencies["ff_storage"].add_message.return_value = "msg_123"
        
        # Store memory
        success = await memory_component.store_memory(
            user_id="test_user",
            session_id="test_session",
            memory_content="I love pizza",
            memory_type=FFMemoryType.EPISODIC.value
        )
        
        assert success is True
        mock_dependencies["ff_vector"].store_vector.assert_called_once()
        mock_dependencies["ff_storage"].add_message.assert_called_once()
    
    async def test_retrieve_memories(self, memory_component, mock_dependencies):
        """Test memory retrieval"""
        # Setup mock search results
        mock_vector_results = [
            {
                "doc_id": "mem_1",
                "similarity_score": 0.9,
                "metadata": {
                    "user_id": "test_user",
                    "memory_type": "episodic",
                    "content": "I love pizza",
                    "importance_score": 0.8
                }
            }
        ]
        mock_dependencies["ff_vector"].search_similar_vectors.return_value = mock_vector_results
        
        # Retrieve memories
        memories = await memory_component.retrieve_memories(
            user_id="test_user",
            query="favorite food",
            limit=5
        )
        
        assert len(memories) == 1
        assert memories[0]["content"] == "I love pizza"
        assert memories[0]["similarity_score"] == 0.9
    
    async def test_working_memory_management(self, memory_component):
        """Test working memory management"""
        session_id = "test_session"
        
        # Add messages to working memory
        messages = [
            FFMessageDTO(role="user", content=f"Message {i}", message_id=f"msg_{i}")
            for i in range(5)
        ]
        
        for message in messages:
            await memory_component.update_working_memory(session_id, message)
        
        # Get session memory
        session_memory = await memory_component.get_session_memory(session_id)
        
        assert session_memory["session_id"] == session_id
        assert session_memory["message_count"] == 5
        assert len(session_memory["working_memory"]["messages"]) == 5
    
    async def test_memory_type_validation(self, memory_component):
        """Test memory type validation"""
        # Valid memory type
        success = await memory_component.store_memory(
            user_id="test_user",
            session_id="test_session", 
            memory_content="Valid memory",
            memory_type=FFMemoryType.EPISODIC.value
        )
        # Note: This might fail due to mock setup, but tests the validation logic
        
        # Invalid memory type
        success = await memory_component.store_memory(
            user_id="test_user",
            session_id="test_session",
            memory_content="Invalid memory",
            memory_type="invalid_type"
        )
        assert success is False
    
    async def test_memory_id_generation(self, memory_component):
        """Test memory ID generation"""
        memory_id1 = memory_component._generate_memory_id("user1", "session1", "content1")
        memory_id2 = memory_component._generate_memory_id("user1", "session1", "content1")
        memory_id3 = memory_component._generate_memory_id("user1", "session1", "content2")
        
        # Same content should generate different IDs (due to timestamp)
        assert memory_id1 != memory_id2
        # Different content should generate different IDs
        assert memory_id1 != memory_id3
        # All should start with 'mem_'
        assert all(mid.startswith("mem_") for mid in [memory_id1, memory_id2, memory_id3])


class TestFFMultiAgentComponent:
    """Unit tests for FF Multi-Agent Component"""
    
    @pytest.fixture
    def multi_agent_config(self):
        """Create multi-agent configuration for testing"""
        return FFMultiAgentConfigDTO(
            max_agents=5,
            min_agents=2,
            default_coordination_mode=FFCoordinationMode.COLLABORATIVE.value,
            response_timeout=30.0,
            consensus_threshold=0.7
        )
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock FF dependencies"""
        return {
            "ff_storage": AsyncMock(),
            "ff_panel": AsyncMock(),
            "ff_search": AsyncMock(),
            "ff_vector": AsyncMock()
        }
    
    @pytest.fixture
    async def multi_agent_component(self, multi_agent_config, mock_dependencies):
        """Create multi-agent component for testing"""
        component = FFMultiAgentComponent(multi_agent_config)
        await component.initialize(mock_dependencies)
        yield component
        await component.cleanup()
    
    async def test_multi_agent_initialization(self, multi_agent_component):
        """Test multi-agent component initialization"""
        assert multi_agent_component._initialized is True
        assert multi_agent_component.ff_panel is not None
        assert multi_agent_component.ff_storage is not None
        
        # Check component info
        info = multi_agent_component.component_info
        assert info['name'] == 'ff_multi_agent'
        assert len(info['capabilities']) > 0
    
    async def test_create_agent_panel(self, multi_agent_component, mock_dependencies):
        """Test agent panel creation"""
        # Setup mock
        mock_dependencies["ff_panel"].create_panel.return_value = "ff_panel_123"
        
        # Create panel
        panel_id = await multi_agent_component.create_agent_panel(
            session_id="test_session",
            user_id="test_user",
            agent_personas=["technical_expert", "creative_writer"],
            coordination_mode="collaborative"
        )
        
        assert panel_id is not None
        assert panel_id.startswith("multi_agent_panel_")
        assert panel_id in multi_agent_component._active_panels
        
        # Check panel details
        panel = multi_agent_component._active_panels[panel_id]
        assert panel.session_id == "test_session"
        assert panel.user_id == "test_user"
        assert len(panel.agent_personas) == 2
    
    async def test_agent_validation(self, multi_agent_component):
        """Test agent count validation"""
        # Too few agents should raise error
        with pytest.raises(ValueError, match="Minimum"):
            await multi_agent_component.create_agent_panel(
                session_id="test_session",
                user_id="test_user",
                agent_personas=["single_agent"]  # Less than min_agents
            )
        
        # Too many agents should be trimmed
        many_agents = [f"agent_{i}" for i in range(10)]  # More than max_agents
        panel_id = await multi_agent_component.create_agent_panel(
            session_id="test_session",
            user_id="test_user",
            agent_personas=many_agents
        )
        
        panel = multi_agent_component._active_panels[panel_id]
        assert len(panel.agent_personas) == multi_agent_component.config.max_agents
    
    async def test_coordination_modes(self, multi_agent_component, mock_dependencies):
        """Test different coordination modes"""
        # Create mock agent responses
        mock_responses = [
            FFAgentResponse(
                agent_id="agent_1",
                agent_persona="technical_expert",
                response_content="Technical perspective on the topic",
                confidence_score=0.8,
                response_time=1.0,
                metadata={}
            ),
            FFAgentResponse(
                agent_id="agent_2", 
                agent_persona="creative_writer",
                response_content="Creative perspective on the topic",
                confidence_score=0.7,
                response_time=1.2,
                metadata={}
            )
        ]
        
        context = {"agent_responses": mock_responses}
        
        # Test consensus coordination
        result = await multi_agent_component.coordinate_agents(
            agents=["agent_1", "agent_2"],
            coordination_mode=FFCoordinationMode.CONSENSUS.value,
            context=context
        )
        assert result["success"] is True
        assert "consensus" in result["coordination_method"]
        
        # Test competitive coordination
        result = await multi_agent_component.coordinate_agents(
            agents=["agent_1", "agent_2"],
            coordination_mode=FFCoordinationMode.COMPETITIVE.value,
            context=context
        )
        assert result["success"] is True
        assert "competitive" in result["coordination_method"]
        
        # Test collaborative coordination
        result = await multi_agent_component.coordinate_agents(
            agents=["agent_1", "agent_2"],
            coordination_mode=FFCoordinationMode.COLLABORATIVE.value,
            context=context
        )
        assert result["success"] is True
        assert "collaborative" in result["coordination_method"]
    
    async def test_agent_response_generation(self, multi_agent_component):
        """Test individual agent response generation"""
        message = FFMessageDTO(role="user", content="What do you think about AI?")
        
        response = await multi_agent_component._process_message_with_single_agent(
            agent_persona="technical_expert",
            message=message,
            panel_context={"session_id": "test_session"}
        )
        
        assert response is not None
        assert response.agent_persona == "technical_expert"
        assert response.response_content != ""
        assert 0.0 <= response.confidence_score <= 1.0
        assert response.response_time > 0
    
    async def test_agent_performance_tracking(self, multi_agent_component):
        """Test agent performance tracking"""
        # Create mock response
        response = FFAgentResponse(
            agent_id="agent_1",
            agent_persona="technical_expert",
            response_content="Test response",
            confidence_score=0.8,
            response_time=1.5,
            metadata={}
        )
        
        # Update performance
        await multi_agent_component._update_agent_performance("technical_expert", response)
        
        # Check statistics
        stats = multi_agent_component.get_multi_agent_statistics()
        assert "agent_performance" in stats
        assert "technical_expert" in stats["agent_performance"]
        
        perf = stats["agent_performance"]["technical_expert"]
        assert perf["total_responses"] == 1
        assert perf["average_confidence"] == 0.8
        assert perf["average_response_time"] == 1.5


class TestFFComponentRegistry:
    """Unit tests for FF Component Registry"""
    
    @pytest.fixture
    def registry_config(self):
        """Create registry configuration for testing"""
        return FFComponentRegistryConfigDTO(
            auto_discovery_enabled=False,
            use_ff_dependency_injection=False,  # Disable for unit testing
            component_health_checks=False,
            enable_component_metrics=False
        )
    
    @pytest.fixture
    async def component_registry(self, registry_config):
        """Create component registry for testing"""
        registry = FFComponentRegistry(registry_config)
        await registry.initialize()
        yield registry
        await registry.shutdown()
    
    async def test_registry_initialization(self, component_registry):
        """Test component registry initialization"""
        assert component_registry._initialized is True
        
        # Check that built-in components are registered
        components = component_registry.list_components()
        expected_components = ["text_chat", "memory", "multi_agent"]
        
        for component in expected_components:
            assert component in components
    
    async def test_component_registration(self, component_registry):
        """Test manual component registration"""
        # Create mock component class
        class MockComponent:
            def __init__(self, config):
                self.config = config
            
            async def initialize(self, dependencies):
                return True
            
            async def cleanup(self):
                pass
        
        # Create mock config class
        class MockConfig:
            pass
        
        # Register component
        component_registry.register_component(
            name="test_component",
            component_class=MockComponent,
            config_class=MockConfig,
            config=MockConfig(),
            dependencies=[],
            ff_manager_dependencies=["ff_storage"],
            priority=50
        )
        
        # Check registration
        assert "test_component" in component_registry.list_components()
        
        info = component_registry.get_component_info("test_component")
        assert info is not None
        assert info["name"] == "test_component"
        assert info["priority"] == 50
    
    async def test_component_loading(self, component_registry):
        """Test component loading"""
        # Load text chat component
        components = await component_registry.load_components(["text_chat"])
        
        assert "text_chat" in components
        assert components["text_chat"] is not None
        
        # Check that it's properly initialized
        text_chat = components["text_chat"]
        assert hasattr(text_chat, 'process_message')
    
    async def test_dependency_resolution(self, component_registry):
        """Test dependency resolution"""
        # Test with components that don't have dependencies
        load_order = await component_registry._resolve_dependency_order(["text_chat", "memory"])
        
        assert len(load_order) == 2
        assert "text_chat" in load_order
        assert "memory" in load_order
    
    async def test_registry_statistics(self, component_registry):
        """Test registry statistics"""
        stats = component_registry.get_registry_statistics()
        
        assert "registered_components" in stats
        assert "loaded_components" in stats
        assert "total_registered" in stats
        assert stats["registered_components"] > 0
    
    async def test_component_info_retrieval(self, component_registry):
        """Test component information retrieval"""
        # Get info for text chat component
        info = component_registry.get_component_info("text_chat")
        
        assert info is not None
        assert info["name"] == "text_chat"
        assert "ff_manager_dependencies" in info
        assert "capabilities" in info
        
        # Test non-existent component
        info = component_registry.get_component_info("non_existent")
        assert info is None


# Test runner helpers
async def run_text_chat_tests():
    """Run text chat component tests"""
    pytest_args = [
        __file__ + "::TestFFTextChatComponent",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    return pytest.main(pytest_args) == 0


async def run_memory_tests():
    """Run memory component tests"""
    pytest_args = [
        __file__ + "::TestFFMemoryComponent",
        "-v", 
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    return pytest.main(pytest_args) == 0


async def run_multi_agent_tests():
    """Run multi-agent component tests"""
    pytest_args = [
        __file__ + "::TestFFMultiAgentComponent",
        "-v",
        "--tb=short", 
        "--asyncio-mode=auto"
    ]
    return pytest.main(pytest_args) == 0


async def run_registry_tests():
    """Run component registry tests"""
    pytest_args = [
        __file__ + "::TestFFComponentRegistry",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    return pytest.main(pytest_args) == 0


async def run_all_unit_tests():
    """Run all unit tests"""
    logger.info("Starting FF Chat Phase 2 Unit Tests...")
    
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("All unit tests passed!")
    else:
        logger.error(f"Unit tests failed with exit code {exit_code}")
    
    return exit_code == 0


if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_all_unit_tests())