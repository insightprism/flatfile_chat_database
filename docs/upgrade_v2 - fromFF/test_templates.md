# FF Chat System - Test Templates and Patterns

## Overview

This document provides comprehensive test templates and patterns for the FF Chat System. All tests follow existing FF testing patterns while ensuring proper integration with FF infrastructure and maintaining backward compatibility.

## 1. Test Structure and Organization

### Test Directory Structure

```
tests/
├── unit/
│   ├── test_ff_chat_components.py
│   ├── test_ff_chat_session_manager.py
│   ├── test_ff_chat_application_manager.py
│   └── test_ff_chat_configs.py
├── integration/
│   ├── test_ff_chat_storage_integration.py
│   ├── test_ff_chat_search_integration.py
│   ├── test_ff_chat_vector_integration.py
│   └── test_ff_chat_panel_integration.py
├── system/
│   ├── test_ff_chat_use_cases.py
│   ├── test_ff_chat_api.py
│   └── test_ff_chat_performance.py
├── fixtures/
│   ├── ff_chat_test_data.py
│   ├── ff_chat_mock_configs.py
│   └── ff_chat_test_utils.py
└── conftest.py
```

## 2. Base Test Templates

### Test Configuration and Fixtures

```python
# conftest.py - Test configuration and fixtures following FF patterns

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator

# Import existing FF test utilities
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_panel_manager import FFPanelManager
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO

# Import FF chat components
from ff_chat_application_manager import FFChatApplicationManager, FFChatApplicationConfigDTO
from ff_chat_session_manager import FFChatSessionManager
from ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO
from ff_memory_component import FFMemoryComponent, FFMemoryConfigDTO

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_storage_dir():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="ff_chat_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
async def ff_test_config(temp_storage_dir) -> FFChatApplicationConfigDTO:
    """Create test configuration following FF patterns"""
    
    config = FFChatApplicationConfigDTO()
    
    # Configure FF storage for testing
    config.ff_storage_config.enabled = True
    config.ff_storage_config.storage_base_path = temp_storage_dir
    config.ff_storage_config.session_id_prefix = "test_chat_"
    config.ff_storage_config.user_id_prefix = "test_user_"
    config.ff_storage_config.enable_file_locking = False  # Disable for tests
    
    # Configure FF search for testing
    config.ff_search_config.enabled = True
    config.ff_search_config.max_results = 10
    config.ff_search_config.relevance_threshold = 0.5
    
    # Configure FF vector storage for testing
    config.ff_vector_storage_config.enabled = True
    config.ff_vector_storage_config.dimension = 384
    config.ff_vector_storage_config.similarity_metric = "cosine"
    config.ff_vector_storage_config.max_vectors = 1000
    
    # Configure FF panel for testing
    config.ff_persona_panel_config.enabled = True
    config.ff_persona_panel_config.max_personas = 3
    
    # Configure chat settings for testing
    config.chat_enabled = True
    config.chat_session.max_messages_per_session = 100
    config.chat_session.session_timeout_minutes = 10
    config.chat_components.text_chat_enabled = True
    config.chat_components.memory_enabled = True
    config.chat_components.multi_agent_enabled = True
    
    return config

@pytest.fixture
async def ff_storage_manager(ff_test_config) -> AsyncGenerator[FFStorageManager, None]:
    """Create FF storage manager for testing"""
    
    storage = FFStorageManager(ff_test_config)
    await storage.initialize()
    
    yield storage
    
    await storage.cleanup()

@pytest.fixture
async def ff_search_manager(ff_test_config) -> AsyncGenerator[FFSearchManager, None]:
    """Create FF search manager for testing"""
    
    search = FFSearchManager(ff_test_config)
    await search.initialize()
    
    yield search
    
    await search.cleanup()

@pytest.fixture
async def ff_vector_manager(ff_test_config) -> AsyncGenerator[FFVectorStorageManager, None]:
    """Create FF vector storage manager for testing"""
    
    vector = FFVectorStorageManager(ff_test_config)
    await vector.initialize()
    
    yield vector
    
    await vector.cleanup()

@pytest.fixture
async def ff_panel_manager(ff_test_config) -> AsyncGenerator[FFPanelManager, None]:
    """Create FF panel manager for testing"""
    
    panel = FFPanelManager(ff_test_config)
    await panel.initialize()
    
    yield panel
    
    await panel.cleanup()

@pytest.fixture
async def ff_chat_app(ff_test_config) -> AsyncGenerator[FFChatApplicationManager, None]:
    """Create FF chat application for testing"""
    
    app = FFChatApplicationManager(ff_test_config)
    await app.initialize()
    
    yield app
    
    await app.cleanup()

@pytest.fixture
def sample_test_data():
    """Sample test data for FF chat tests"""
    return {
        "users": [
            {"user_id": "test_user_1", "name": "Test User 1"},
            {"user_id": "test_user_2", "name": "Test User 2"},
        ],
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            {"role": "user", "content": "Can you help me with a problem?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help. What's the problem?"},
        ],
        "use_cases": [
            "basic_chat",
            "memory_chat", 
            "rag_chat",
            "multi_ai_panel"
        ]
    }
```

## 3. Unit Test Templates

### FF Chat Component Unit Tests

```python
# test_ff_chat_components.py - Unit tests for FF chat components

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO
from ff_memory_component import FFMemoryComponent, FFMemoryConfigDTO
from ff_multi_agent_component import FFMultiAgentComponent, FFMultiAgentConfigDTO

class TestFFTextChatComponent:
    """Unit tests for FF Text Chat Component"""
    
    @pytest.fixture
    def text_chat_config(self):
        """Create text chat configuration for testing"""
        return FFTextChatConfigDTO(
            enabled=True,
            max_tokens=1000,
            temperature=0.7
        )
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock FF storage manager"""
        mock = AsyncMock()
        mock.add_message = AsyncMock(return_value="msg_123")
        mock.get_session_messages = AsyncMock(return_value=[])
        mock.create_session = AsyncMock(return_value="session_123")
        return mock
    
    @pytest.mark.asyncio
    async def test_component_initialization(self, text_chat_config, mock_storage_manager):
        """Test FF text chat component initialization"""
        
        component = FFTextChatComponent(text_chat_config)
        
        # Test initialization
        success = await component.initialize(storage=mock_storage_manager)
        
        assert success, "Component should initialize successfully"
        assert component.storage_manager == mock_storage_manager
        assert component._initialized, "Component should be marked as initialized"
    
    @pytest.mark.asyncio
    async def test_component_message_processing(self, text_chat_config, mock_storage_manager):
        """Test message processing uses FF storage"""
        
        component = FFTextChatComponent(text_chat_config)
        await component.initialize(storage=mock_storage_manager)
        
        # Test message processing
        session_id = "test_session_123"
        message = "Hello, this is a test message"
        
        result = await component.process_message(session_id, message)
        
        # Verify result structure
        assert result["success"], f"Processing should succeed: {result.get('error')}"
        assert result["component"] == "FFTextChatComponent"
        assert "response" in result
        
        # Verify FF storage was called
        mock_storage_manager.add_message.assert_called()
        call_args = mock_storage_manager.add_message.call_args
        assert call_args[1]["content"] == message
    
    @pytest.mark.asyncio
    async def test_component_cleanup(self, text_chat_config, mock_storage_manager):
        """Test component cleanup follows FF patterns"""
        
        component = FFTextChatComponent(text_chat_config)
        await component.initialize(storage=mock_storage_manager)
        
        # Test cleanup
        success = await component.cleanup()
        
        assert success, "Component cleanup should succeed"
        assert not component._initialized, "Component should be marked as not initialized"
    
    @pytest.mark.asyncio
    async def test_component_error_handling(self, text_chat_config):
        """Test component error handling"""
        
        component = FFTextChatComponent(text_chat_config)
        
        # Test processing without initialization
        result = await component.process_message("session", "message")
        
        assert not result["success"], "Processing should fail without initialization"
        assert "error" in result
        assert "not initialized" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_component_with_context(self, text_chat_config, mock_storage_manager):
        """Test component processing with context"""
        
        component = FFTextChatComponent(text_chat_config)
        await component.initialize(storage=mock_storage_manager)
        
        # Test with context
        context = {
            "user_preferences": {"style": "formal"},
            "session_metadata": {"topic": "business"}
        }
        
        result = await component.process_message(
            "session_123", 
            "Hello", 
            context=context
        )
        
        assert result["success"], "Processing with context should succeed"
        assert "metadata" in result

class TestFFMemoryComponent:
    """Unit tests for FF Memory Component"""
    
    @pytest.fixture
    def memory_config(self):
        """Create memory component configuration"""
        return FFMemoryConfigDTO(
            enabled=True,
            max_entries=1000,
            similarity_threshold=0.8
        )
    
    @pytest.fixture
    def mock_vector_manager(self):
        """Create mock FF vector storage manager"""
        mock = AsyncMock()
        mock.add_vector = AsyncMock(return_value="vec_123")
        mock.search_similar = AsyncMock(return_value=[])
        mock.get_vector = AsyncMock(return_value=None)
        return mock
    
    @pytest.mark.asyncio
    async def test_memory_component_initialization(self, memory_config, mock_vector_manager):
        """Test memory component initialization with FF vector storage"""
        
        component = FFMemoryComponent(memory_config)
        
        success = await component.initialize(
            storage=AsyncMock(),
            vector=mock_vector_manager
        )
        
        assert success, "Memory component should initialize successfully"
        assert component.vector_manager == mock_vector_manager
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self, memory_config, mock_vector_manager):
        """Test memory storage and retrieval using FF vector storage"""
        
        component = FFMemoryComponent(memory_config)
        await component.initialize(storage=AsyncMock(), vector=mock_vector_manager)
        
        # Test memory storage
        session_id = "memory_session_123"
        message = "Remember that I like Python programming"
        
        result = await component.process_message(session_id, message)
        
        assert result["success"], "Memory storage should succeed"
        
        # Verify vector storage was called
        mock_vector_manager.add_vector.assert_called()
        
        # Test memory retrieval
        mock_vector_manager.search_similar.return_value = [
            {"content": "I like Python programming", "similarity": 0.9}
        ]
        
        retrieval_result = await component.process_message(
            session_id, 
            "What do I like?",
            context={"retrieve_memory": True}
        )
        
        assert retrieval_result["success"], "Memory retrieval should succeed"
        mock_vector_manager.search_similar.assert_called()

class TestFFMultiAgentComponent:
    """Unit tests for FF Multi-Agent Component"""
    
    @pytest.fixture
    def multi_agent_config(self):
        """Create multi-agent component configuration"""
        return FFMultiAgentConfigDTO(
            enabled=True,
            max_agents=3,
            coordination_model="round_robin"
        )
    
    @pytest.fixture
    def mock_panel_manager(self):
        """Create mock FF panel manager"""
        mock = AsyncMock()
        mock.create_panel = AsyncMock(return_value="panel_123")
        mock.add_persona = AsyncMock(return_value="persona_123")
        mock.process_panel_message = AsyncMock(return_value={
            "success": True,
            "responses": [{"persona": "agent_1", "response": "Hello from agent 1"}]
        })
        return mock
    
    @pytest.mark.asyncio
    async def test_multi_agent_initialization(self, multi_agent_config, mock_panel_manager):
        """Test multi-agent component initialization with FF panel manager"""
        
        component = FFMultiAgentComponent(multi_agent_config)
        
        success = await component.initialize(
            storage=AsyncMock(),
            panel=mock_panel_manager
        )
        
        assert success, "Multi-agent component should initialize successfully"
        assert component.panel_manager == mock_panel_manager
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, multi_agent_config, mock_panel_manager):
        """Test multi-agent coordination using FF panel manager"""
        
        component = FFMultiAgentComponent(multi_agent_config)
        await component.initialize(storage=AsyncMock(), panel=mock_panel_manager)
        
        # Test multi-agent processing
        session_id = "multi_agent_session"
        message = "What are the pros and cons of renewable energy?"
        
        result = await component.process_message(session_id, message)
        
        assert result["success"], "Multi-agent processing should succeed"
        
        # Verify panel manager was used
        mock_panel_manager.process_panel_message.assert_called()
        
        # Check that multiple agent responses are handled
        assert "responses" in result.get("metadata", {})
```

## 4. Integration Test Templates

### FF Storage Integration Tests

```python
# test_ff_chat_storage_integration.py - Integration tests with FF storage

import pytest
import asyncio
from pathlib import Path

from ff_storage_manager import FFStorageManager
from ff_chat_session_manager import FFChatSessionManager
from ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO

class TestFFChatStorageIntegration:
    """Integration tests between FF chat components and FF storage"""
    
    @pytest.mark.asyncio
    async def test_chat_session_creation_with_ff_storage(self, ff_storage_manager, ff_test_config):
        """Test chat session creation integrates with FF storage"""
        
        session_manager = FFChatSessionManager(ff_test_config)
        await session_manager.initialize(ff_storage_manager)
        
        try:
            # Create chat session
            user_id = "integration_test_user"
            use_case = "basic_chat"
            
            session_id = await session_manager.create_session(
                user_id=user_id,
                use_case=use_case,
                config={"test": True}
            )
            
            assert session_id is not None, "Session should be created"
            assert session_id.startswith("test_chat_"), "Session should use FF prefix"
            
            # Verify session exists in FF storage
            session = await ff_storage_manager.get_session(user_id, session_id)
            assert session is not None, "Session should exist in FF storage"
            assert session.user_id == user_id
            
            # Verify session metadata includes use case
            metadata = await session_manager.get_session_metadata(user_id, session_id)
            assert metadata["use_case"] == use_case
            
        finally:
            await session_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_chat_message_persistence_in_ff_storage(self, ff_storage_manager, ff_test_config):
        """Test chat messages are properly stored in FF storage"""
        
        # Create text chat component
        config = FFTextChatConfigDTO()
        component = FFTextChatComponent(config)
        await component.initialize(storage=ff_storage_manager)
        
        try:
            session_id = "test_message_persistence"
            user_id = "test_user"
            
            # Create session in FF storage
            await ff_storage_manager.create_session(user_id, "Test Session")
            
            # Process messages through component
            messages = [
                "Hello, this is the first message",
                "This is the second message", 
                "And this is the third message"
            ]
            
            for message in messages:
                result = await component.process_message(session_id, message)
                assert result["success"], f"Message should be processed: {message}"
            
            # Verify messages are stored in FF storage
            stored_messages = await ff_storage_manager.get_session_messages(session_id)
            
            # Should have user messages + assistant responses
            user_messages = [msg for msg in stored_messages if msg.role == "user"]
            assert len(user_messages) >= len(messages), "All user messages should be stored"
            
            # Verify message content
            for i, user_msg in enumerate(user_messages[:len(messages)]):
                assert user_msg.content == messages[i], f"Message content should match: {messages[i]}"
            
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_chat_session_retrieval_from_ff_storage(self, ff_storage_manager, ff_test_config):
        """Test chat session retrieval uses FF storage correctly"""
        
        session_manager = FFChatSessionManager(ff_test_config)
        await session_manager.initialize(ff_storage_manager)
        
        try:
            user_id = "retrieval_test_user"
            
            # Create multiple sessions
            session_names = ["Session 1", "Session 2", "Session 3"]
            created_sessions = []
            
            for name in session_names:
                session_id = await session_manager.create_session(
                    user_id=user_id,
                    use_case="basic_chat",
                    config={"session_name": name}
                )
                created_sessions.append(session_id)
            
            # Retrieve sessions through FF storage
            ff_sessions = await ff_storage_manager.get_user_sessions(user_id)
            
            assert len(ff_sessions) >= len(session_names), "All sessions should be retrievable"
            
            # Verify session details
            for session in ff_sessions[-len(session_names):]:  # Get latest sessions
                assert session.user_id == user_id
                assert session.session_id in created_sessions
            
        finally:
            await session_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_chat_search_integration_with_ff_storage(self, ff_storage_manager, ff_search_manager, ff_test_config):
        """Test chat search functionality integrates with FF storage and search"""
        
        # Create chat component with search capabilities
        component = FFTextChatComponent(FFTextChatConfigDTO(search_enabled=True))
        await component.initialize(
            storage=ff_storage_manager,
            search=ff_search_manager
        )
        
        try:
            session_id = "search_integration_test"
            user_id = "search_test_user"
            
            # Create session
            await ff_storage_manager.create_session(user_id, "Search Test Session")
            
            # Add messages with searchable content
            search_messages = [
                "I love programming in Python",
                "JavaScript is also a great language",
                "Machine learning with TensorFlow is fascinating",
                "React makes frontend development easier"
            ]
            
            for message in search_messages:
                await component.process_message(session_id, message)
            
            # Test search through FF search manager
            search_results = await ff_search_manager.search("Python programming")
            
            # Should find relevant messages
            assert len(search_results) > 0, "Search should find relevant messages"
            
            # Verify search results contain expected content
            python_found = any("Python" in result.content for result in search_results)
            assert python_found, "Search should find Python-related content"
            
        finally:
            await component.cleanup()
```

### FF Vector Storage Integration Tests

```python
# test_ff_chat_vector_integration.py - Integration tests with FF vector storage

import pytest
import numpy as np

from ff_vector_storage_manager import FFVectorStorageManager
from ff_memory_component import FFMemoryComponent, FFMemoryConfigDTO

class TestFFChatVectorIntegration:
    """Integration tests between FF chat memory and FF vector storage"""
    
    @pytest.mark.asyncio
    async def test_memory_vector_storage_integration(self, ff_storage_manager, ff_vector_manager, ff_test_config):
        """Test memory component uses FF vector storage for embeddings"""
        
        config = FFMemoryConfigDTO(
            enabled=True,
            max_entries=1000,
            similarity_threshold=0.7
        )
        
        component = FFMemoryComponent(config)
        await component.initialize(
            storage=ff_storage_manager,
            vector=ff_vector_manager
        )
        
        try:
            session_id = "vector_memory_test"
            
            # Store memory items
            memory_items = [
                "I work as a software engineer",
                "My favorite programming language is Python", 
                "I enjoy hiking on weekends",
                "I have a cat named Whiskers"
            ]
            
            for item in memory_items:
                result = await component.process_message(
                    session_id, 
                    f"Remember: {item}",
                    context={"store_memory": True}
                )
                assert result["success"], f"Memory storage should succeed: {item}"
            
            # Test memory retrieval through vector similarity
            retrieval_queries = [
                ("What is my job?", "software engineer"),
                ("What language do I like?", "Python"),
                ("What do I do for fun?", "hiking"),
                ("Do I have pets?", "cat")
            ]
            
            for query, expected_content in retrieval_queries:
                result = await component.process_message(
                    session_id,
                    query,
                    context={"retrieve_memory": True}
                )
                
                assert result["success"], f"Memory retrieval should succeed: {query}"
                
                # Check if relevant memory was retrieved
                retrieved_memories = result.get("metadata", {}).get("retrieved_memories", [])
                assert len(retrieved_memories) > 0, f"Should retrieve memories for: {query}"
                
                # Verify content relevance
                relevant_found = any(
                    expected_content.lower() in memory["content"].lower() 
                    for memory in retrieved_memories
                )
                assert relevant_found, f"Should find relevant memory for: {query}"
        
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search_accuracy(self, ff_vector_manager):
        """Test FF vector storage similarity search accuracy"""
        
        # Test data with known similarities
        test_vectors = [
            ("Python programming language", [0.1, 0.9, 0.3, 0.2]),
            ("Java programming language", [0.2, 0.8, 0.4, 0.1]), 
            ("Cooking Italian pasta", [0.8, 0.1, 0.7, 0.9]),
            ("Baking chocolate cake", [0.7, 0.2, 0.8, 0.8])
        ]
        
        # Store test vectors
        vector_ids = []
        for content, vector in test_vectors:
            vector_id = await ff_vector_manager.add_vector(
                content=content,
                vector=np.array(vector, dtype=np.float32),
                metadata={"content": content}
            )
            vector_ids.append(vector_id)
        
        # Test similarity search
        query_vector = np.array([0.15, 0.85, 0.35, 0.15], dtype=np.float32)  # Similar to programming vectors
        
        results = await ff_vector_manager.search_similar(
            query_vector=query_vector,
            limit=2,
            similarity_threshold=0.5
        )
        
        assert len(results) >= 1, "Should find similar vectors"
        
        # Programming-related vectors should be more similar
        programming_results = [
            r for r in results 
            if "programming" in r["metadata"]["content"].lower()
        ]
        assert len(programming_results) > 0, "Should find programming-related vectors"
    
    @pytest.mark.asyncio
    async def test_memory_component_vector_cleanup(self, ff_storage_manager, ff_vector_manager, ff_test_config):
        """Test memory component properly manages vector storage cleanup"""
        
        config = FFMemoryConfigDTO(max_entries=5)  # Small limit for testing
        component = FFMemoryComponent(config)
        await component.initialize(
            storage=ff_storage_manager,
            vector=ff_vector_manager
        )
        
        try:
            session_id = "vector_cleanup_test"
            
            # Store more memories than the limit
            memories = [f"Memory item {i}" for i in range(10)]
            
            for memory in memories:
                await component.process_message(
                    session_id,
                    f"Remember: {memory}",
                    context={"store_memory": True}
                )
            
            # Check that vector storage respects limits (implementation dependent)
            # This would test the component's cleanup logic
            
            # Verify that retrieval still works after cleanup
            result = await component.process_message(
                session_id,
                "What do you remember?",
                context={"retrieve_memory": True}
            )
            
            assert result["success"], "Memory retrieval should work after cleanup"
            
        finally:
            await component.cleanup()
```

## 5. System Test Templates

### End-to-End Use Case Tests

```python
# test_ff_chat_use_cases.py - System tests for all 22 use cases

import pytest
import asyncio
from typing import List, Dict, Any

from ff_chat_application_manager import FFChatApplicationManager, FFChatApplicationConfigDTO

class TestFFChatUseCases:
    """System tests for all FF chat use cases"""
    
    @pytest.mark.asyncio
    async def test_basic_chat_use_case(self, ff_chat_app, sample_test_data):
        """Test basic 1:1 chat use case end-to-end"""
        
        user_id = "basic_chat_user"
        
        # Create basic chat session
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case="basic_chat"
        )
        
        # Test conversation flow
        conversation = [
            ("Hello, how are you today?", "greeting"),
            ("Can you help me with a math problem?", "question"),
            ("What is 15 + 27?", "calculation"),  
            ("Thank you for your help!", "gratitude")
        ]
        
        for message, message_type in conversation:
            result = await ff_chat_app.process_message(session_id, message)
            
            assert result["success"], f"Basic chat should handle {message_type}: {message}"
            assert result["response"], f"Should generate response for {message_type}"
            assert result["use_case"] == "basic_chat"
            
            # Verify only text_chat component was used
            components_used = list(result["component_results"].keys())
            assert "text_chat" in components_used
            assert len(components_used) == 1, "Basic chat should only use text_chat component"
    
    @pytest.mark.asyncio
    async def test_memory_chat_use_case(self, ff_chat_app):
        """Test memory chat use case with context retention"""
        
        user_id = "memory_chat_user"
        
        # Create memory chat session
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case="memory_chat"
        )
        
        # Test memory storage and retrieval
        memory_conversation = [
            ("My name is Alice and I'm a teacher", "personal_info"),
            ("I teach mathematics at a high school", "profession_detail"),
            ("I have been teaching for 10 years", "experience"),
            ("What do you remember about me?", "memory_query"),
            ("What subject do I teach?", "specific_recall")
        ]
        
        for i, (message, message_type) in enumerate(memory_conversation):
            result = await ff_chat_app.process_message(session_id, message)
            
            assert result["success"], f"Memory chat should handle {message_type}: {message}"
            assert result["use_case"] == "memory_chat"
            
            # Verify memory and text_chat components were used
            components_used = list(result["component_results"].keys())
            assert "text_chat" in components_used
            assert "memory" in components_used
            
            # For memory queries, check that relevant information is recalled
            if message_type in ["memory_query", "specific_recall"]:
                memory_result = result["component_results"]["memory"]
                assert memory_result["success"], "Memory component should succeed"
                
                # Response should reference stored information
                response = result["response"].lower()
                if message_type == "memory_query":
                    assert any(word in response for word in ["alice", "teacher", "mathematics"]), \
                        "Should recall stored personal information"
                elif message_type == "specific_recall":
                    assert "mathematics" in response or "math" in response, \
                        "Should recall specific subject taught"
    
    @pytest.mark.asyncio 
    async def test_rag_chat_use_case(self, ff_chat_app):
        """Test RAG chat use case with knowledge retrieval"""
        
        user_id = "rag_chat_user"
        
        # Create RAG chat session
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case="rag_chat"
        )
        
        # First, store some knowledge documents (would be done through document ingestion)
        knowledge_items = [
            "Python is a high-level programming language known for its simplicity",
            "Machine learning is a subset of artificial intelligence",
            "Django is a popular web framework for Python development"
        ]
        
        # Add knowledge items (this would typically be done during setup)
        for item in knowledge_items:
            await ff_chat_app.process_message(
                session_id, 
                f"Please remember this fact: {item}",
                context={"store_knowledge": True}
            )
        
        # Test knowledge retrieval queries
        rag_queries = [
            ("What do you know about Python?", "python"),
            ("Tell me about machine learning", "machine learning"),
            ("What web frameworks work with Python?", "django")
        ]
        
        for query, expected_topic in rag_queries:
            result = await ff_chat_app.process_message(session_id, query)
            
            assert result["success"], f"RAG chat should handle query: {query}"
            assert result["use_case"] == "rag_chat"
            
            # Verify all required components were used
            components_used = list(result["component_results"].keys())
            assert "text_chat" in components_used
            assert "memory" in components_used
            assert "search" in components_used
            
            # Response should incorporate retrieved knowledge
            response = result["response"].lower()
            assert expected_topic.lower() in response, \
                f"Response should mention {expected_topic} for query: {query}"
    
    @pytest.mark.asyncio
    async def test_multi_ai_panel_use_case(self, ff_chat_app):
        """Test multi-AI panel use case with agent coordination"""
        
        user_id = "multi_panel_user"
        
        # Create multi-AI panel session
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case="multi_ai_panel"
        )
        
        # Test complex problem that benefits from multiple perspectives
        complex_queries = [
            "What are the pros and cons of renewable energy?",
            "How should we address climate change?",
            "What are the ethical implications of AI development?"
        ]
        
        for query in complex_queries:
            result = await ff_chat_app.process_message(session_id, query)
            
            assert result["success"], f"Multi-AI panel should handle: {query}"
            assert result["use_case"] == "multi_ai_panel"
            
            # Verify multi-agent components were used
            components_used = list(result["component_results"].keys())
            assert "multi_agent" in components_used
            assert "memory" in components_used
            assert "persona" in components_used
            
            # Response should show multiple perspectives
            multi_agent_result = result["component_results"]["multi_agent"]
            assert multi_agent_result["success"], "Multi-agent processing should succeed"
            
            # Should have responses from multiple agents
            agent_responses = multi_agent_result.get("metadata", {}).get("agent_responses", [])
            assert len(agent_responses) > 1, "Should have multiple agent responses"
    
    @pytest.mark.parametrize("use_case,expected_components", [
        ("basic_chat", ["text_chat"]),
        ("memory_chat", ["text_chat", "memory"]),
        ("rag_chat", ["text_chat", "memory", "search"]),
        ("multimodal_chat", ["text_chat", "multimodal"]),
        ("translation_chat", ["text_chat", "multimodal", "tools", "memory", "persona"]),
        ("personal_assistant", ["text_chat", "tools", "memory", "persona"]),
        ("multi_ai_panel", ["multi_agent", "memory", "persona"]),
        ("ai_debate", ["multi_agent", "persona", "trace"]),
        ("prompt_sandbox", ["text_chat", "trace"])
    ])
    @pytest.mark.asyncio
    async def test_use_case_component_mapping(self, ff_chat_app, use_case, expected_components):
        """Test that each use case activates the correct components"""
        
        user_id = f"component_mapping_user_{use_case}"
        
        # Create session for specific use case
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case=use_case
        )
        
        # Send test message
        result = await ff_chat_app.process_message(
            session_id, 
            "This is a test message for component mapping"
        )
        
        assert result["success"], f"Use case {use_case} should process successfully"
        assert result["use_case"] == use_case
        
        # Verify expected components were activated
        components_used = list(result["component_results"].keys())
        
        for expected_component in expected_components:
            assert expected_component in components_used, \
                f"Use case {use_case} should activate component {expected_component}"
        
        # Verify all activated components succeeded
        for component_name, component_result in result["component_results"].items():
            assert component_result["success"], \
                f"Component {component_name} should succeed for use case {use_case}"

class TestFFChatPerformanceAndScalability:
    """Performance and scalability tests for FF chat system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, ff_chat_app):
        """Test handling multiple concurrent chat sessions"""
        
        num_concurrent_sessions = 10
        messages_per_session = 5
        
        async def create_and_test_session(session_index: int):
            """Create session and exchange messages"""
            user_id = f"concurrent_user_{session_index}"
            
            session_id = await ff_chat_app.create_chat_session(
                user_id=user_id,
                use_case="basic_chat"
            )
            
            # Exchange multiple messages
            for msg_index in range(messages_per_session):
                message = f"Message {msg_index} from session {session_index}"
                result = await ff_chat_app.process_message(session_id, message)
                assert result["success"], f"Concurrent message should succeed: {message}"
            
            return session_id
        
        # Run concurrent sessions
        tasks = [
            create_and_test_session(i) 
            for i in range(num_concurrent_sessions)
        ]
        
        session_ids = await asyncio.gather(*tasks)
        
        # Verify all sessions were created successfully
        assert len(session_ids) == num_concurrent_sessions
        assert len(set(session_ids)) == num_concurrent_sessions, "All session IDs should be unique"
    
    @pytest.mark.asyncio
    async def test_memory_performance_with_large_context(self, ff_chat_app):
        """Test memory component performance with large conversation context"""
        
        user_id = "memory_performance_user"
        
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case="memory_chat"
        )
        
        # Create large conversation history
        num_messages = 100
        
        import time
        start_time = time.time()
        
        for i in range(num_messages):
            message = f"This is message number {i} with some content to test memory performance"
            result = await ff_chat_app.process_message(session_id, message)
            assert result["success"], f"Message {i} should be processed successfully"
        
        processing_time = time.time() - start_time
        
        # Performance should be reasonable (adjust threshold as needed)
        avg_time_per_message = processing_time / num_messages
        assert avg_time_per_message < 1.0, f"Average processing time too high: {avg_time_per_message}s"
        
        # Test memory retrieval performance
        start_time = time.time()
        
        retrieval_result = await ff_chat_app.process_message(
            session_id,
            "What was discussed in our conversation?",
            context={"retrieve_memory": True}
        )
        
        retrieval_time = time.time() - start_time
        
        assert retrieval_result["success"], "Memory retrieval should succeed"
        assert retrieval_time < 2.0, f"Memory retrieval too slow: {retrieval_time}s"
    
    @pytest.mark.asyncio
    async def test_ff_storage_integrity_under_load(self, ff_chat_app):
        """Test FF storage integrity under concurrent load"""
        
        num_concurrent_users = 5
        messages_per_user = 20
        
        async def stress_test_user(user_index: int):
            """Stress test single user with rapid messages"""
            user_id = f"stress_user_{user_index}"
            
            session_id = await ff_chat_app.create_chat_session(
                user_id=user_id,
                use_case="memory_chat"
            )
            
            messages = []
            for i in range(messages_per_user):
                message = f"Stress test message {i} from user {user_index}"
                result = await ff_chat_app.process_message(session_id, message)
                assert result["success"], f"Stress message should succeed: {message}"
                messages.append(message)
            
            return user_id, session_id, messages
        
        # Run stress test
        tasks = [stress_test_user(i) for i in range(num_concurrent_users)]
        results = await asyncio.gather(*tasks)
        
        # Verify storage integrity - check that all messages were stored
        for user_id, session_id, sent_messages in results:
            stored_messages = await ff_chat_app.storage_manager.get_session_messages(session_id)
            
            user_messages = [msg for msg in stored_messages if msg.role == "user"]
            assert len(user_messages) >= len(sent_messages), \
                f"All messages should be stored for user {user_id}"
            
            # Verify message content integrity
            for i, stored_msg in enumerate(user_messages[:len(sent_messages)]):
                assert stored_msg.content == sent_messages[i], \
                    f"Message content should match for user {user_id}"
```

## 6. Test Execution and Reporting

### Test Configuration

```python
# pytest.ini - Test configuration
[tool:pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=ff_chat
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

markers =
    unit: Unit tests
    integration: Integration tests  
    system: System/end-to-end tests
    performance: Performance tests
    slow: Slow-running tests
```

### Test Execution Scripts

```bash
#!/bin/bash
# run_tests.sh - Test execution script

echo "Running FF Chat System Tests..."

# Run unit tests
echo "=== Unit Tests ==="
pytest tests/unit/ -v -m unit

# Run integration tests  
echo "=== Integration Tests ==="
pytest tests/integration/ -v -m integration

# Run system tests
echo "=== System Tests ==="
pytest tests/system/ -v -m system

# Run performance tests (optional)
if [ "$1" = "--performance" ]; then
    echo "=== Performance Tests ==="
    pytest tests/system/test_ff_chat_performance.py -v -m performance
fi

# Generate coverage report
echo "=== Coverage Report ==="
pytest --cov=ff_chat --cov-report=html --cov-report=term

echo "Tests completed. Coverage report available in htmlcov/"
```

This comprehensive test template library ensures thorough testing of the FF Chat System while maintaining integration with existing FF infrastructure and following established FF testing patterns.