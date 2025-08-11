# FF Chat System - Code Templates and Patterns

## Overview

This document provides code templates and patterns for implementing FF Chat components following existing FF architecture patterns. All templates demonstrate integration with existing FF managers and maintain backward compatibility.

## 1. FF Chat Component Template

### Basic FF Chat Component Structure

```python
"""
FF Chat Component Template - Follow this pattern for all chat components.

This template demonstrates:
- Integration with existing FF managers as backend services
- FF-style configuration and protocol patterns
- Proper async/await patterns matching existing FF code
- FF logging and error handling patterns
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Import existing FF infrastructure
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_dependency_injection_manager import ff_get_container

@dataclass
class FFChatComponentConfigDTO(FFBaseConfigDTO):
    """FF Chat component configuration following FF patterns"""
    enabled: bool = True
    priority: int = 100
    component_type: str = "base"
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        # Add component-specific validation
        if not isinstance(self.priority, int) or self.priority < 0:
            raise ValueError("Priority must be non-negative integer")

@runtime_checkable
class FFChatComponentProtocol(Protocol):
    """FF Chat component protocol following FF protocol patterns"""
    
    async def initialize(self, 
                        storage: FFStorageManager,
                        search: Optional[FFSearchManager] = None,
                        vector: Optional[FFVectorStorageManager] = None) -> bool:
        """Initialize component with FF managers"""
        ...
    
    async def process_message(self, 
                            session_id: str, 
                            message: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process chat message using FF backend services"""
        ...
    
    async def cleanup(self) -> bool:
        """Cleanup resources following FF patterns"""
        ...

class FFBaseChatComponent(FFChatComponentProtocol):
    """Base FF chat component implementation"""
    
    def __init__(self, config: FFChatComponentConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
        self.storage_manager: Optional[FFStorageManager] = None
        self.search_manager: Optional[FFSearchManager] = None
        self.vector_manager: Optional[FFVectorStorageManager] = None
        self._initialized = False
    
    async def initialize(self, 
                        storage: FFStorageManager,
                        search: Optional[FFSearchManager] = None,
                        vector: Optional[FFVectorStorageManager] = None) -> bool:
        """Initialize using existing FF managers"""
        try:
            self.storage_manager = storage
            self.search_manager = search
            self.vector_manager = vector
            
            # Perform component-specific initialization
            await self._component_initialize()
            
            self._initialized = True
            self.logger.info(f"Initialized FF chat component: {self.__class__.__name__}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF chat component: {e}")
            return False
    
    async def _component_initialize(self):
        """Override in subclasses for specific initialization"""
        pass
    
    async def process_message(self, 
                            session_id: str, 
                            message: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message using existing FF storage backend"""
        if not self._initialized:
            return {"success": False, "error": "Component not initialized"}
        
        try:
            # Template for FF backend integration
            result = await self._process_with_ff_backend(session_id, message, context or {})
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "component": self.__class__.__name__,
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            self.logger.error(f"FF chat component processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_with_ff_backend(self, 
                                     session_id: str, 
                                     message: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses - integrate with specific FF managers"""
        raise NotImplementedError("Subclasses must implement FF backend processing")
    
    async def cleanup(self) -> bool:
        """Cleanup resources following FF patterns"""
        try:
            # Perform component-specific cleanup
            await self._component_cleanup()
            
            self._initialized = False
            self.logger.info(f"Cleaned up FF chat component: {self.__class__.__name__}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup FF chat component: {e}")
            return False
    
    async def _component_cleanup(self):
        """Override in subclasses for specific cleanup"""
        pass
```

## 2. FF Chat Manager Template

### Chat Application Manager

```python
"""
FF Chat Application Manager Template

Demonstrates orchestration of multiple FF chat components
using existing FF managers as backend services.
"""

from typing import Dict, Any, List, Optional
import asyncio
from dataclasses import dataclass, field

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_panel_manager import FFPanelManager
from ff_dependency_injection_manager import ff_get_container
from ff_utils.ff_logging import get_logger

@dataclass
class FFChatApplicationConfigDTO(FFConfigurationManagerConfigDTO):
    """Enhanced FF configuration with chat capabilities"""
    
    # Chat-specific configurations
    chat_enabled: bool = True
    default_use_case: str = "basic_chat"
    max_concurrent_sessions: int = 100
    session_timeout_minutes: int = 60
    
    # Component configurations
    text_chat_enabled: bool = True
    memory_enabled: bool = True
    multi_agent_enabled: bool = False
    tools_enabled: bool = False
    
    # Use case mappings
    use_case_components: Dict[str, List[str]] = field(default_factory=lambda: {
        "basic_chat": ["text_chat"],
        "memory_chat": ["text_chat", "memory"],
        "multi_ai_panel": ["multi_agent", "memory"],
        "rag_chat": ["text_chat", "memory", "search"]
    })

class FFChatApplicationManager:
    """FF Chat Application Manager using existing FF infrastructure"""
    
    def __init__(self, config: FFChatApplicationConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Use existing FF managers
        self.storage_manager: Optional[FFStorageManager] = None
        self.search_manager: Optional[FFSearchManager] = None
        self.vector_manager: Optional[FFVectorStorageManager] = None
        self.panel_manager: Optional[FFPanelManager] = None
        
        # Chat-specific managers
        self.session_manager: Optional[FFChatSessionManager] = None
        self.component_registry: Dict[str, FFChatComponentProtocol] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize chat application with existing FF managers"""
        try:
            # Get dependency injection container
            container = ff_get_container()
            
            # Initialize existing FF managers
            self.storage_manager = FFStorageManager(self.config)
            await self.storage_manager.initialize()
            
            if self.config.ff_search_config.enabled:
                self.search_manager = FFSearchManager(self.config)
                await self.search_manager.initialize()
            
            if self.config.ff_vector_storage_config.enabled:
                self.vector_manager = FFVectorStorageManager(self.config)
                await self.vector_manager.initialize()
            
            if self.config.ff_persona_panel_config.enabled:
                self.panel_manager = FFPanelManager(self.config)
                await self.panel_manager.initialize()
            
            # Initialize chat-specific managers
            self.session_manager = FFChatSessionManager(self.config)
            await self.session_manager.initialize(self.storage_manager)
            
            # Register and initialize chat components
            await self._initialize_chat_components()
            
            self._initialized = True
            self.logger.info("FF Chat Application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Chat Application: {e}")
            return False
    
    async def _initialize_chat_components(self):
        """Initialize chat components based on configuration"""
        
        # Initialize text chat component if enabled
        if self.config.text_chat_enabled:
            text_chat = FFTextChatComponent(FFTextChatConfigDTO())
            await text_chat.initialize(
                storage=self.storage_manager,
                search=self.search_manager
            )
            self.component_registry["text_chat"] = text_chat
        
        # Initialize memory component if enabled
        if self.config.memory_enabled:
            memory = FFMemoryComponent(FFMemoryConfigDTO())
            await memory.initialize(
                storage=self.storage_manager,
                vector=self.vector_manager
            )
            self.component_registry["memory"] = memory
        
        # Initialize multi-agent component if enabled
        if self.config.multi_agent_enabled:
            multi_agent = FFMultiAgentComponent(FFMultiAgentConfigDTO())
            await multi_agent.initialize(
                storage=self.storage_manager,
                search=self.search_manager,
                vector=self.vector_manager
            )
            self.component_registry["multi_agent"] = multi_agent
    
    async def create_chat_session(self, 
                                user_id: str,
                                use_case: str = None,
                                session_config: Optional[Dict[str, Any]] = None) -> str:
        """Create new chat session with specified use case"""
        
        if not self._initialized:
            raise RuntimeError("Chat application not initialized")
        
        use_case = use_case or self.config.default_use_case
        
        # Create session using existing FF storage
        session_id = await self.session_manager.create_session(
            user_id=user_id,
            use_case=use_case,
            config=session_config or {}
        )
        
        # Track active session
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "use_case": use_case,
            "created_at": asyncio.get_event_loop().time(),
            "components": self.config.use_case_components.get(use_case, ["text_chat"])
        }
        
        self.logger.info(f"Created chat session {session_id} for use case {use_case}")
        return session_id
    
    async def process_message(self, 
                            session_id: str,
                            message: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message through appropriate components"""
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}
        
        session_info = self.active_sessions[session_id]
        required_components = session_info["components"]
        
        try:
            # Process through each required component
            results = {}
            final_response = ""
            
            for component_name in required_components:
                if component_name in self.component_registry:
                    component = self.component_registry[component_name]
                    result = await component.process_message(
                        session_id=session_id,
                        message=message,
                        context=context
                    )
                    results[component_name] = result
                    
                    if result.get("success") and result.get("response"):
                        final_response = result["response"]
            
            return {
                "success": True,
                "response": final_response,
                "session_id": session_id,
                "use_case": session_info["use_case"],
                "component_results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message in session {session_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """Cleanup all resources following FF patterns"""
        try:
            # Cleanup chat components
            for component in self.component_registry.values():
                await component.cleanup()
            
            # Cleanup chat managers
            if self.session_manager:
                await self.session_manager.cleanup()
            
            # Cleanup existing FF managers
            if self.panel_manager:
                await self.panel_manager.cleanup()
            
            if self.vector_manager:
                await self.vector_manager.cleanup()
            
            if self.search_manager:
                await self.search_manager.cleanup()
            
            if self.storage_manager:
                await self.storage_manager.cleanup()
            
            self._initialized = False
            self.logger.info("FF Chat Application cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during FF Chat Application cleanup: {e}")
            return False
```

## 3. FF Configuration Template

### Enhanced Configuration Classes

```python
"""
FF Enhanced Configuration Template

Demonstrates extending existing FF configuration classes
while maintaining backward compatibility.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Import existing FF configurations
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_base_config import FFBaseConfigDTO

@dataclass
class FFChatSessionConfigDTO(FFBaseConfigDTO):
    """FF Chat session configuration"""
    max_messages_per_session: int = 1000
    session_timeout_minutes: int = 60
    auto_save_interval_seconds: int = 30
    enable_message_threading: bool = True
    max_concurrent_sessions_per_user: int = 10

@dataclass
class FFChatComponentsConfigDTO(FFBaseConfigDTO):
    """FF Chat components configuration"""
    
    # Text chat settings
    text_chat_enabled: bool = True  
    text_chat_max_tokens: int = 4000
    text_chat_temperature: float = 0.7
    
    # Memory settings
    memory_enabled: bool = True
    memory_max_entries: int = 10000
    memory_similarity_threshold: float = 0.8
    
    # Multi-agent settings
    multi_agent_enabled: bool = False
    multi_agent_max_agents: int = 5
    multi_agent_coordination_model: str = "round_robin"
    
    # Tools settings
    tools_enabled: bool = False
    tools_sandbox_enabled: bool = True
    tools_max_execution_time: int = 30

@dataclass  
class FFChatUseCasesConfigDTO(FFBaseConfigDTO):
    """FF Chat use cases configuration"""
    
    # Default use case
    default_use_case: str = "basic_chat"
    
    # Use case to component mappings
    use_case_components: Dict[str, List[str]] = field(default_factory=lambda: {
        # Basic patterns (4 use cases)
        "basic_chat": ["text_chat"],
        "multimodal_chat": ["text_chat", "multimodal"],
        "rag_chat": ["text_chat", "memory", "search"],
        "multimodal_rag": ["text_chat", "multimodal", "memory", "search"],
        
        # Specialized modes (9 use cases)  
        "translation_chat": ["text_chat", "multimodal", "tools", "memory", "persona"],
        "personal_assistant": ["text_chat", "tools", "memory", "persona"],
        "interactive_tutor": ["text_chat", "persona"],
        "language_tutor": ["text_chat", "tools", "persona"],
        "exam_assistant": ["text_chat", "memory", "search"],
        "ai_notetaker": ["multimodal", "memory", "search", "tools"],
        "chatops_assistant": ["text_chat", "tools"],
        "cross_team_concierge": ["text_chat", "memory", "search", "tools"],
        "scene_critic": ["multimodal", "persona"],
        
        # Multi-participant (5 use cases)
        "multi_ai_panel": ["multi_agent", "memory", "persona"],
        "ai_debate": ["multi_agent", "persona", "trace"],
        "topic_delegation": ["text_chat", "memory", "search", "multi_agent", "router"],
        "ai_game_master": ["text_chat", "multi_agent", "memory"],
        "auto_task_agent": ["tools", "multi_agent", "memory"],
        
        # Context & memory (3 use cases)
        "memory_chat": ["text_chat", "memory"],
        "thought_partner": ["text_chat", "memory"],
        "story_world_chat": ["text_chat", "memory", "persona"],
        
        # Development (1 use case)
        "prompt_sandbox": ["text_chat", "trace"]
    })
    
    # Use case specific settings
    use_case_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class FFChatApplicationConfigDTO(FFConfigurationManagerConfigDTO):
    """
    Enhanced FF configuration with comprehensive chat capabilities.
    
    Extends existing FFConfigurationManagerConfigDTO to inherit all
    existing FF manager configurations while adding chat-specific settings.
    """
    
    # Chat application settings
    chat_enabled: bool = True
    chat_api_enabled: bool = False
    chat_websocket_enabled: bool = False
    
    # Chat-specific configurations
    chat_session: FFChatSessionConfigDTO = field(default_factory=FFChatSessionConfigDTO)
    chat_components: FFChatComponentsConfigDTO = field(default_factory=FFChatComponentsConfigDTO)  
    chat_use_cases: FFChatUseCasesConfigDTO = field(default_factory=FFChatUseCasesConfigDTO)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate chat configuration consistency
        self._validate_chat_config()
    
    def _validate_chat_config(self):
        """Validate chat configuration consistency"""
        
        # Ensure required FF managers are enabled for chat functionality
        if self.chat_enabled:
            # Storage manager is always required
            if not self.ff_storage_config.enabled:
                raise ValueError("FF Storage must be enabled for chat functionality")
            
            # Enable search if memory components are used
            if self.chat_components.memory_enabled and not self.ff_search_config.enabled:
                self.logger.warning("Enabling FF Search for memory component support")
                self.ff_search_config.enabled = True
            
            # Enable vector storage if memory with embeddings is used  
            if self.chat_components.memory_enabled and not self.ff_vector_storage_config.enabled:
                self.logger.warning("Enabling FF Vector Storage for memory component support")
                self.ff_vector_storage_config.enabled = True
            
            # Enable panel manager if multi-agent is used
            if self.chat_components.multi_agent_enabled and not self.ff_persona_panel_config.enabled:
                self.logger.warning("Enabling FF Panel Manager for multi-agent support")
                self.ff_persona_panel_config.enabled = True
    
    def get_use_case_config(self, use_case: str) -> Dict[str, Any]:
        """Get configuration for specific use case"""
        
        # Get required components
        components = self.chat_use_cases.use_case_components.get(use_case, ["text_chat"])
        
        # Get use case specific settings
        settings = self.chat_use_cases.use_case_settings.get(use_case, {})
        
        return {
            "use_case": use_case,
            "components": components,
            "settings": settings,
            "component_configs": {
                "text_chat": {
                    "enabled": "text_chat" in components,
                    "max_tokens": self.chat_components.text_chat_max_tokens,
                    "temperature": self.chat_components.text_chat_temperature
                },
                "memory": {
                    "enabled": "memory" in components,
                    "max_entries": self.chat_components.memory_max_entries,
                    "similarity_threshold": self.chat_components.memory_similarity_threshold
                },
                "multi_agent": {
                    "enabled": "multi_agent" in components,
                    "max_agents": self.chat_components.multi_agent_max_agents,
                    "coordination_model": self.chat_components.multi_agent_coordination_model
                },
                "tools": {
                    "enabled": "tools" in components,
                    "sandbox_enabled": self.chat_components.tools_sandbox_enabled,
                    "max_execution_time": self.chat_components.tools_max_execution_time
                }
            }
        }
```

## 4. FF Testing Template

### Comprehensive Test Patterns

```python
"""
FF Chat Testing Template

Demonstrates testing patterns for FF chat components
using existing FF infrastructure and test utilities.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Import existing FF test utilities
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager  
from ff_vector_storage_manager import FFVectorStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config

# Import FF chat components (these would be implemented)
from ff_chat_application_manager import FFChatApplicationManager, FFChatApplicationConfigDTO
from ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO

class TestFFChatIntegration:
    """Test FF chat system integration with existing FF infrastructure"""
    
    @pytest.fixture
    async def ff_test_config(self):
        """Create test configuration following FF patterns"""
        
        # Create temporary directory for test data
        test_dir = tempfile.mkdtemp()
        
        # Create test configuration using existing FF config loading
        config = FFChatApplicationConfigDTO()
        
        # Configure storage to use test directory
        config.ff_storage_config.storage_base_path = test_dir
        config.ff_storage_config.session_id_prefix = "test_chat_"
        
        # Enable required FF managers for testing
        config.ff_storage_config.enabled = True
        config.ff_search_config.enabled = True
        config.ff_vector_storage_config.enabled = True
        
        # Configure chat settings for testing
        config.chat_enabled = True
        config.chat_components.text_chat_enabled = True
        config.chat_components.memory_enabled = True
        
        yield config
        
        # Cleanup test directory
        shutil.rmtree(test_dir, ignore_errors=True)
    
    @pytest.fixture  
    async def ff_storage_manager(self, ff_test_config):
        """Create FF storage manager for testing"""
        
        storage = FFStorageManager(ff_test_config)
        await storage.initialize()
        
        yield storage
        
        await storage.cleanup()
    
    @pytest.fixture
    async def ff_chat_app(self, ff_test_config):
        """Create FF chat application for testing"""
        
        app = FFChatApplicationManager(ff_test_config)
        await app.initialize()
        
        yield app
        
        await app.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_chat_application_initialization(self, ff_test_config):
        """Test FF chat application initializes with existing FF managers"""
        
        app = FFChatApplicationManager(ff_test_config)
        
        # Test initialization
        success = await app.initialize()
        assert success, "FF chat application should initialize successfully"
        
        # Verify existing FF managers are initialized
        assert app.storage_manager is not None, "FF storage manager should be initialized"
        assert app.search_manager is not None, "FF search manager should be initialized"  
        assert app.vector_manager is not None, "FF vector manager should be initialized"
        
        # Verify chat-specific managers are initialized
        assert app.session_manager is not None, "FF chat session manager should be initialized"
        assert len(app.component_registry) > 0, "Chat components should be registered"
        
        # Test cleanup
        cleanup_success = await app.cleanup()
        assert cleanup_success, "FF chat application should cleanup successfully"
    
    @pytest.mark.asyncio
    async def test_ff_chat_session_creation(self, ff_chat_app):
        """Test chat session creation using FF storage backend"""
        
        user_id = "test_user_123"
        use_case = "basic_chat"
        
        # Create session
        session_id = await ff_chat_app.create_chat_session(
            user_id=user_id,
            use_case=use_case
        )
        
        assert session_id is not None, "Session ID should be generated"
        assert session_id.startswith("test_chat_"), "Session ID should use FF prefix"
        assert session_id in ff_chat_app.active_sessions, "Session should be tracked"
        
        # Verify session info
        session_info = ff_chat_app.active_sessions[session_id]
        assert session_info["user_id"] == user_id
        assert session_info["use_case"] == use_case
        assert "text_chat" in session_info["components"]
    
    @pytest.mark.asyncio
    async def test_ff_chat_message_processing(self, ff_chat_app):
        """Test message processing through FF chat components"""
        
        # Create session
        session_id = await ff_chat_app.create_chat_session(
            user_id="test_user",
            use_case="basic_chat"
        )
        
        # Process message
        message = "Hello, this is a test message"
        result = await ff_chat_app.process_message(
            session_id=session_id,
            message=message
        )
        
        # Verify processing results
        assert result["success"], f"Message processing should succeed: {result.get('error')}"
        assert result["response"], "Response should be generated"
        assert result["session_id"] == session_id
        assert result["use_case"] == "basic_chat"
        assert "component_results" in result
        
        # Verify component-specific results
        component_results = result["component_results"]
        assert "text_chat" in component_results, "Text chat component should process message"
        assert component_results["text_chat"]["success"], "Text chat should succeed"
    
    @pytest.mark.asyncio
    async def test_ff_chat_component_ff_storage_integration(self, ff_storage_manager):
        """Test FF chat component integration with existing FF storage"""
        
        # Create text chat component
        config = FFTextChatConfigDTO()
        component = FFTextChatComponent(config)
        
        # Initialize with existing FF storage manager
        success = await component.initialize(
            storage=ff_storage_manager
        )
        assert success, "Component should initialize with FF storage manager"
        
        # Test message processing uses FF storage
        session_id = "test_session_123"
        message = "Test message for FF storage integration"
        
        result = await component.process_message(
            session_id=session_id,
            message=message
        )
        
        assert result["success"], f"Component processing should succeed: {result.get('error')}"
        assert result["component"] == "FFTextChatComponent"
        
        # Verify message was stored using FF storage
        # (This would use existing FF storage manager methods)
        stored_messages = await ff_storage_manager.get_session_messages(
            session_id=session_id
        )
        assert len(stored_messages) > 0, "Message should be stored via FF storage"
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_chat_memory_component_integration(self, ff_test_config, ff_storage_manager):
        """Test FF memory component integration with existing FF vector storage"""
        
        # Create vector storage manager
        vector_manager = FFVectorStorageManager(ff_test_config)
        await vector_manager.initialize()
        
        try:
            # Create memory component
            config = FFMemoryConfigDTO()
            component = FFMemoryComponent(config)
            
            # Initialize with FF managers
            success = await component.initialize(
                storage=ff_storage_manager,
                vector=vector_manager
            )
            assert success, "Memory component should initialize with FF managers"
            
            # Test memory processing
            session_id = "test_memory_session"
            message = "Remember this important information for later"
            
            result = await component.process_message(
                session_id=session_id,
                message=message
            )
            
            assert result["success"], "Memory component should process successfully"
            
            # Test memory retrieval
            retrieval_result = await component.process_message(
                session_id=session_id,
                message="What should I remember?",
                context={"retrieve_memory": True}
            )
            
            assert retrieval_result["success"], "Memory retrieval should succeed"
            # Memory should be retrieved from FF vector storage
            
            await component.cleanup()
            
        finally:
            await vector_manager.cleanup()
    
    @pytest.mark.asyncio  
    async def test_ff_chat_use_case_routing(self, ff_chat_app):
        """Test use case routing uses correct FF component combinations"""
        
        # Test basic chat use case
        basic_session = await ff_chat_app.create_chat_session(
            user_id="test_user",
            use_case="basic_chat"
        )
        session_info = ff_chat_app.active_sessions[basic_session]
        assert session_info["components"] == ["text_chat"]
        
        # Test memory chat use case
        memory_session = await ff_chat_app.create_chat_session(
            user_id="test_user", 
            use_case="memory_chat"
        )
        session_info = ff_chat_app.active_sessions[memory_session]
        assert "text_chat" in session_info["components"]
        assert "memory" in session_info["components"]
        
        # Test RAG chat use case
        rag_session = await ff_chat_app.create_chat_session(
            user_id="test_user",
            use_case="rag_chat" 
        )
        session_info = ff_chat_app.active_sessions[rag_session]
        assert "text_chat" in session_info["components"]
        assert "memory" in session_info["components"]
        assert "search" in session_info["components"]
    
    @pytest.mark.asyncio
    async def test_ff_chat_backward_compatibility(self, ff_test_config):
        """Test FF chat system maintains backward compatibility"""
        
        # Test existing FF storage manager functionality unchanged
        storage = FFStorageManager(ff_test_config)
        await storage.initialize()
        
        try:
            # Test existing FF storage operations work unchanged
            user_id = "test_compatibility_user"
            session_name = "Test Compatibility Session"
            
            # Create session using existing FF storage methods
            session_id = await storage.create_session(user_id, session_name)
            assert session_id is not None, "Existing FF session creation should work"
            
            # Add message using existing FF storage methods
            message_id = await storage.add_message(
                user_id=user_id,
                session_id=session_id,
                role="user",
                content="Test compatibility message"
            )
            assert message_id is not None, "Existing FF message storage should work"
            
            # Retrieve messages using existing FF storage methods
            messages = await storage.get_session_messages(session_id)
            assert len(messages) == 1, "Existing FF message retrieval should work"
            assert messages[0].content == "Test compatibility message"
            
            # Test existing FF storage operations unchanged by chat system
            sessions = await storage.get_user_sessions(user_id)
            assert len(sessions) >= 1, "Existing FF session retrieval should work"
            
        finally:
            await storage.cleanup()

# Performance tests
class TestFFChatPerformance:
    """Test FF chat system performance with existing FF infrastructure"""
    
    @pytest.mark.asyncio
    async def test_ff_chat_session_creation_performance(self, ff_chat_app):
        """Test session creation performance doesn't degrade FF storage"""
        
        import time
        
        # Create multiple sessions and measure performance
        start_time = time.time()
        session_ids = []
        
        for i in range(10):
            session_id = await ff_chat_app.create_chat_session(
                user_id=f"perf_user_{i}",
                use_case="basic_chat"
            )
            session_ids.append(session_id)
        
        creation_time = time.time() - start_time
        
        # Performance should be reasonable (adjust threshold as needed)
        assert creation_time < 5.0, f"Session creation took too long: {creation_time}s"
        assert len(session_ids) == 10, "All sessions should be created"
        assert len(set(session_ids)) == 10, "All session IDs should be unique"
    
    @pytest.mark.asyncio
    async def test_ff_chat_concurrent_message_processing(self, ff_chat_app):
        """Test concurrent message processing doesn't affect FF storage integrity"""
        
        # Create session
        session_id = await ff_chat_app.create_chat_session(
            user_id="concurrent_test_user",
            use_case="basic_chat"
        )
        
        # Process multiple messages concurrently
        messages = [f"Concurrent message {i}" for i in range(5)]
        
        tasks = [
            ff_chat_app.process_message(session_id, message)
            for message in messages
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All messages should be processed successfully
        for i, result in enumerate(results):
            assert result["success"], f"Message {i} should be processed successfully"
            assert result["session_id"] == session_id
        
        # Verify FF storage integrity - all messages should be stored
        stored_messages = await ff_chat_app.storage_manager.get_session_messages(session_id)
        
        # Should have at least the user messages stored
        user_messages = [msg for msg in stored_messages if msg.role == "user"]
        assert len(user_messages) >= len(messages), "All user messages should be stored"
```

## 5. Usage Examples

### Basic FF Chat Application Usage

```python
"""
Example usage of FF Chat Application following FF patterns
"""

import asyncio
from ff_chat_application_manager import FFChatApplicationManager, FFChatApplicationConfigDTO

async def main():
    """Example FF chat application usage"""
    
    # Create configuration following FF patterns
    config = FFChatApplicationConfigDTO()
    
    # Configure FF storage (existing pattern)
    config.ff_storage_config.storage_base_path = "./chat_data"
    config.ff_storage_config.enabled = True
    
    # Configure chat capabilities
    config.chat_enabled = True
    config.chat_components.text_chat_enabled = True
    config.chat_components.memory_enabled = True
    
    # Initialize FF chat application
    app = FFChatApplicationManager(config)
    success = await app.initialize()
    
    if not success:
        print("Failed to initialize FF chat application")
        return
    
    try:
        # Create chat session
        session_id = await app.create_chat_session(
            user_id="example_user",
            use_case="memory_chat"
        )
        print(f"Created session: {session_id}")
        
        # Process messages
        messages = [
            "Hello, I'm testing the FF chat system",
            "Please remember that I like Python programming",
            "What did I say I liked?"
        ]
        
        for message in messages:
            result = await app.process_message(
                session_id=session_id,
                message=message
            )
            
            if result["success"]:
                print(f"User: {message}")
                print(f"Assistant: {result['response']}")
                print(f"Components used: {list(result['component_results'].keys())}")
                print("---")
            else:
                print(f"Error processing message: {result['error']}")
    
    finally:
        # Cleanup following FF patterns
        await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

This template library provides comprehensive patterns for implementing FF chat components that integrate seamlessly with existing FF infrastructure while maintaining all established patterns and backward compatibility.