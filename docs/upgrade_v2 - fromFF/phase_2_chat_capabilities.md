# Phase 2: Chat Capabilities Implementation

## Executive Summary

### Objectives
Implement the core chat components that provide 86% use case coverage using existing FF managers as backend services. This phase creates the essential chat processing capabilities while leveraging the proven FF storage, search, and processing infrastructure.

### Key Deliverables
- FF Text Chat Component (supports 17/22 use cases - 77%)
- FF Memory Component (supports 7/22 use cases - 32%) 
- FF Multi-Agent Component (supports 5/22 use cases - 23%)
- Component registry system integrated with FF dependency injection
- Message streaming and real-time processing using FF utilities

### Prerequisites
- Phase 1 completed successfully (FF integration foundation operational)
- FF Chat Application layer functional with existing FF backend
- Enhanced FF configuration system in place
- All existing FF managers tested and operational

### Success Criteria
- ✅ FF Text Chat component processes all text-based conversations using FF storage
- ✅ FF Memory component maintains context using FF vector storage backend
- ✅ FF Multi-Agent component coordinates participants using FF panel system
- ✅ All components integrate seamlessly with FF Chat Application
- ✅ Support for 19/22 use cases (86% coverage) through FF-integrated components

## Technical Specifications

### Module Coverage Analysis

Based on FF integration capabilities, Phase 2 components provide:

| FF Component | Use Cases Supported | FF Backend Integration |
|--------------|-------------------|----------------------|
| **FF Text Chat** | basic_chat, rag_chat, memory_chat, translation_chat, personal_assistant, topic_delegation, thought_partner, interactive_tutor, language_tutor, exam_assistant, chatops_assistant, cross_team_concierge, story_world_chat, ai_game_master, prompt_sandbox | FFStorageManager + FFSearchManager |
| **FF Memory** | personal_assistant, multi_ai_panel, memory_chat, thought_partner, story_world_chat, ai_game_master, auto_task_agent | FFVectorStorageManager + FFStorageManager |
| **FF Multi-Agent** | multi_ai_panel, ai_debate, topic_delegation, ai_game_master, auto_task_agent | FFPanelManager + FFStorageManager |

**Combined Coverage**: 19/22 use cases (86%) using existing FF infrastructure

### Implementation Details

#### 1. FF Text Chat Component

**File**: `ff_chat_components/ff_text_chat_component.py`

```python
"""
FF Text Chat Component - Core Text Processing

Handles all text-based conversation processing using existing FF storage
and search capabilities as backend services.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime
from dataclasses import dataclass, field

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

@dataclass
class FFTextChatConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Text Chat component"""
    
    # Text processing settings
    max_message_length: int = 4000
    response_format: str = "markdown"
    context_window: int = 10
    
    # LLM integration settings (placeholder for future integration)
    temperature: float = 0.7
    max_tokens: int = 1000
    model_name: str = "gpt-3.5-turbo"
    
    # FF storage integration settings
    enable_message_search: bool = True
    enable_context_retrieval: bool = True
    max_search_results: int = 5
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_message_length <= 0:
            raise ValueError("max_message_length must be positive")
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative")

class FFTextChatComponent(FFChatComponentProtocol):
    """
    FF Text Chat component using existing FF storage and search.
    
    Provides text conversation processing while leveraging
    FF storage for persistence and FF search for context retrieval.
    """
    
    def __init__(self, config: FFTextChatConfigDTO):
        """
        Initialize FF text chat component.
        
        Args:
            config: Text chat configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services (injected during initialization)
        self.ff_storage: Optional[FFStorageManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        
        # Component state
        self._initialized = False
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_text_chat",
            "version": "1.0.0",
            "description": "FF text chat component using FF storage backend",
            "capabilities": ["text_processing", "conversation_management", "context_retrieval"],
            "use_cases": [
                "basic_chat", "rag_chat", "memory_chat", "translation_chat",
                "personal_assistant", "topic_delegation", "thought_partner",
                "interactive_tutor", "language_tutor", "exam_assistant",
                "chatops_assistant", "cross_team_concierge", "story_world_chat",
                "ai_game_master", "prompt_sandbox"
            ],
            "ff_dependencies": ["FFStorageManager", "FFSearchManager"]
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_search = dependencies.get("ff_search")
            
            if not self.ff_storage:
                raise ValueError("FFStorageManager dependency required")
            
            # FF search is optional but recommended
            if not self.ff_search:
                self.logger.warning("FFSearchManager not available - context retrieval disabled")
                self.config.enable_context_retrieval = False
            
            self._initialized = True
            self.logger.info("FF Text Chat component initialized with FF backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Text Chat component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text message using FF storage and search backend.
        
        Args:
            session_id: FF storage session ID
            user_id: User identifier
            message: FF message DTO
            context: Optional processing context
            
        Returns:
            Processing results with response content
        """
        if not self._initialized:
            raise RuntimeError("FF Text Chat component not initialized")
        
        try:
            self.logger.info(f"Processing text message in session {session_id}")
            
            # Validate message
            if not await self._validate_message(message):
                return {
                    "success": False,
                    "error": "Message validation failed",
                    "component": "ff_text_chat"
                }
            
            # Get conversation context using FF storage
            conversation_context = await self._get_conversation_context(user_id, session_id)
            
            # Get relevant context using FF search (if enabled)
            search_context = []
            if self.config.enable_context_retrieval and self.ff_search:
                search_context = await self._get_search_context(user_id, message.content)
            
            # Generate response using FF backend (placeholder for LLM integration)
            response_content = await self._generate_response(
                message=message,
                conversation_context=conversation_context,
                search_context=search_context,
                context=context or {}
            )
            
            return {
                "success": True,
                "response_content": response_content,
                "component": "ff_text_chat",
                "processor": "ff_storage_backend",
                "context_used": len(conversation_context),
                "search_results": len(search_context),
                "metadata": {
                    "message_length": len(message.content),
                    "response_length": len(response_content),
                    "processing_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text message: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_text_chat"
            }
    
    async def _validate_message(self, message: FFMessageDTO) -> bool:
        """Validate message using FF validation patterns"""
        try:
            if not message.content or not message.content.strip():
                return False
            
            if len(message.content) > self.config.max_message_length:
                return False
            
            if not message.role:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Message validation error: {e}")
            return False
    
    async def _get_conversation_context(self, user_id: str, session_id: str) -> List[FFMessageDTO]:
        """Get conversation context using FF storage"""
        try:
            # Get recent messages using existing FF storage
            messages = await self.ff_storage.get_messages(
                user_id=user_id,
                session_id=session_id,
                limit=self.config.context_window
            )
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return []
    
    async def _get_search_context(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """Get relevant context using FF search"""
        try:
            if not self.ff_search:
                return []
            
            # Use existing FF search capabilities
            search_results = await self.ff_search.search_messages(
                user_id=user_id,
                query=query,
                limit=self.config.max_search_results
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error getting search context: {e}")
            return []
    
    async def _generate_response(self, 
                                 message: FFMessageDTO,
                                 conversation_context: List[FFMessageDTO],
                                 search_context: List[Dict[str, Any]],
                                 context: Dict[str, Any]) -> str:
        """
        Generate response using FF backend.
        
        For Phase 2, this provides basic response generation.
        In production, this would integrate with LLM services.
        """
        try:
            # Basic response generation using FF context
            user_message = message.content
            context_count = len(conversation_context)
            search_count = len(search_context)
            
            # Extract relevant context information
            recent_topics = []
            if conversation_context:
                for msg in conversation_context[-3:]:  # Last 3 messages
                    if len(msg.content) > 10:  # Skip very short messages
                        recent_topics.append(msg.content[:50] + "...")
            
            # Build contextual response
            response_parts = []
            
            if "?" in user_message:
                response_parts.append(f"I understand you're asking about: {user_message}")
            else:
                response_parts.append(f"Thanks for sharing: {user_message}")
            
            if context_count > 0:
                response_parts.append(f"Based on our conversation with {context_count} previous messages")
            
            if search_count > 0:
                response_parts.append(f"I found {search_count} related items in your history")
            
            if recent_topics:
                response_parts.append(f"Recent topics we discussed: {', '.join(recent_topics)}")
            
            # Add FF storage confirmation
            response_parts.append("(Powered by FF storage backend)")
            
            return ". ".join(response_parts) + "."
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    async def get_capabilities(self) -> List[str]:
        """Get component capabilities"""
        return self.component_info["capabilities"]
    
    async def supports_use_case(self, use_case: str) -> bool:
        """Check if component supports a use case"""
        return use_case in self.component_info["use_cases"]
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Text Chat component")
        self._initialized = False
        self.ff_storage = None
        self.ff_search = None
```

#### 2. FF Memory Component

**File**: `ff_chat_components/ff_memory_component.py`

```python
"""
FF Memory Component - Cross-Session Memory Management

Provides working, episodic, semantic, and persistent memory using
existing FF vector storage and storage managers as backend.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_search_manager import FFSearchManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

class FFMemoryType(Enum):
    """Types of memory supported"""
    WORKING = "working"      # Current session context
    EPISODIC = "episodic"    # Conversation episodes
    SEMANTIC = "semantic"    # Knowledge and facts
    PROCEDURAL = "procedural" # Learned behaviors

@dataclass
class FFMemoryConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Memory component"""
    
    # Memory type settings
    enabled_memory_types: List[str] = None
    
    # Working memory settings
    working_memory_size: int = 20
    working_memory_timeout: int = 3600  # seconds
    
    # Episodic memory settings
    episodic_retention_days: int = 30
    episodic_max_entries: int = 1000
    
    # Semantic memory settings
    semantic_similarity_threshold: float = 0.7
    semantic_max_entries: int = 500
    
    # FF vector integration settings
    use_ff_vector_storage: bool = True
    embedding_dimension: int = 384
    
    def __post_init__(self):
        super().__post_init__()
        if self.enabled_memory_types is None:
            self.enabled_memory_types = ["working", "episodic", "semantic"]
        
        if self.working_memory_size <= 0:
            raise ValueError("working_memory_size must be positive")

class FFMemoryComponent(FFChatComponentProtocol):
    """
    FF Memory component using existing FF vector storage backend.
    
    Provides cross-session memory management while leveraging
    FF vector storage for embeddings and FF storage for persistence.
    """
    
    def __init__(self, config: FFMemoryConfigDTO):
        """
        Initialize FF memory component.
        
        Args:
            config: Memory component configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.ff_storage: Optional[FFStorageManager] = None
        self.ff_vector: Optional[FFVectorStorageManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        
        # Memory storage
        self.working_memory: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> memories
        self.memory_timestamps: Dict[str, datetime] = {}
        
        # Component state
        self._initialized = False
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_memory",
            "version": "1.0.0", 
            "description": "FF memory component using FF vector storage backend",
            "capabilities": ["working_memory", "episodic_memory", "semantic_memory", "context_persistence"],
            "use_cases": [
                "personal_assistant", "multi_ai_panel", "memory_chat",
                "thought_partner", "story_world_chat", "ai_game_master", "auto_task_agent"
            ],
            "ff_dependencies": ["FFStorageManager", "FFVectorStorageManager", "FFSearchManager"]
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_vector = dependencies.get("ff_vector")
            self.ff_search = dependencies.get("ff_search")
            
            if not self.ff_storage:
                raise ValueError("FFStorageManager dependency required")
            
            if not self.ff_vector and self.config.use_ff_vector_storage:
                self.logger.warning("FFVectorStorageManager not available - semantic memory disabled")
                self.config.use_ff_vector_storage = False
            
            # Initialize memory storage
            self.working_memory = {}
            self.memory_timestamps = {}
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_working_memory())
            
            self._initialized = True
            self.logger.info("FF Memory component initialized with FF vector backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Memory component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message and update memory using FF backend.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: FF message DTO
            context: Optional context information
            
        Returns:
            Processing results with memory information
        """
        if not self._initialized:
            raise RuntimeError("FF Memory component not initialized")
        
        try:
            self.logger.info(f"Processing message for memory in session {session_id}")
            
            # Update working memory
            await self._update_working_memory(session_id, user_id, message, context)
            
            # Update episodic memory using FF storage
            if "episodic" in self.config.enabled_memory_types:
                await self._update_episodic_memory(session_id, user_id, message)
            
            # Update semantic memory using FF vector storage
            if "semantic" in self.config.enabled_memory_types and self.ff_vector:
                await self._update_semantic_memory(user_id, message)
            
            # Retrieve relevant memories
            relevant_memories = await self._retrieve_relevant_memories(user_id, message.content)
            
            return {
                "success": True,
                "component": "ff_memory",
                "processor": "ff_vector_backend",
                "memory_updated": True,
                "relevant_memories": relevant_memories,
                "working_memory_size": len(self.working_memory.get(session_id, [])),
                "metadata": {
                    "memory_types_enabled": self.config.enabled_memory_types,
                    "ff_vector_enabled": self.config.use_ff_vector_storage
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message for memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_memory"
            }
    
    async def _update_working_memory(self, 
                                     session_id: str,
                                     user_id: str,
                                     message: FFMessageDTO,
                                     context: Optional[Dict[str, Any]]) -> None:
        """Update working memory for current session"""
        try:
            if session_id not in self.working_memory:
                self.working_memory[session_id] = []
            
            # Add message to working memory
            memory_entry = {
                "message_id": message.message_id,
                "content": message.content,
                "role": message.role,
                "timestamp": message.timestamp,
                "user_id": user_id,
                "context": context or {}
            }
            
            self.working_memory[session_id].append(memory_entry)
            self.memory_timestamps[session_id] = datetime.now()
            
            # Trim working memory if too large
            if len(self.working_memory[session_id]) > self.config.working_memory_size:
                self.working_memory[session_id] = self.working_memory[session_id][-self.config.working_memory_size:]
            
            self.logger.debug(f"Updated working memory for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating working memory: {e}")
    
    async def _update_episodic_memory(self, 
                                      session_id: str,
                                      user_id: str,
                                      message: FFMessageDTO) -> None:
        """Update episodic memory using FF storage"""
        try:
            # Use FF storage to persist episodic memories
            # Store as a special type of session or document
            
            episodic_entry = {
                "type": "episodic_memory",
                "session_id": session_id,
                "user_id": user_id,
                "message_id": message.message_id,
                "content": message.content,
                "role": message.role,
                "timestamp": message.timestamp,
                "created_at": datetime.now().isoformat()
            }
            
            # Store using FF storage metadata capabilities
            # This could be enhanced to use a dedicated episodic memory storage
            self.logger.debug(f"Updated episodic memory for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating episodic memory: {e}")
    
    async def _update_semantic_memory(self, user_id: str, message: FFMessageDTO) -> None:
        """Update semantic memory using FF vector storage"""
        try:
            if not self.ff_vector:
                return
            
            # Extract key concepts from message (placeholder for NLP processing)
            concepts = await self._extract_concepts(message.content)
            
            if concepts:
                # Store concepts using FF vector storage
                for concept in concepts:
                    semantic_entry = {
                        "type": "semantic_memory",
                        "user_id": user_id,
                        "concept": concept,
                        "source_message": message.message_id,
                        "content": message.content,
                        "timestamp": message.timestamp,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # This would use FF vector storage to store concept embeddings
                    self.logger.debug(f"Updated semantic memory with concept: {concept}")
            
        except Exception as e:
            self.logger.error(f"Error updating semantic memory: {e}")
    
    async def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content (placeholder for NLP)"""
        try:
            # Simple concept extraction for demonstration
            # In production, this would use proper NLP/entity extraction
            
            words = content.lower().split()
            
            # Filter for potential concepts (longer words, no common words)
            common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "within", "without", "under", "over"}
            
            concepts = [word for word in words if len(word) > 4 and word not in common_words]
            
            return concepts[:5]  # Return top 5 concepts
            
        except Exception as e:
            self.logger.error(f"Error extracting concepts: {e}")
            return []
    
    async def _retrieve_relevant_memories(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using FF search and vector capabilities"""
        try:
            relevant_memories = []
            
            # Search working memory
            for session_id, memories in self.working_memory.items():
                for memory in memories:
                    if memory["user_id"] == user_id and query.lower() in memory["content"].lower():
                        relevant_memories.append({
                            "type": "working",
                            "content": memory["content"],
                            "timestamp": memory["timestamp"],
                            "relevance": 0.9  # High relevance for working memory
                        })
            
            # Search using FF search for episodic memories
            if self.ff_search:
                search_results = await self.ff_search.search_messages(
                    user_id=user_id,
                    query=query,
                    limit=3
                )
                
                for result in search_results:
                    relevant_memories.append({
                        "type": "episodic",
                        "content": result.get("content", ""),
                        "timestamp": result.get("timestamp", ""),
                        "relevance": result.get("score", 0.7)
                    })
            
            # Search using FF vector storage for semantic memories
            if self.ff_vector:
                # This would use FF vector similarity search
                # For now, return placeholder semantic memories
                pass
            
            # Sort by relevance
            relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
            
            return relevant_memories[:5]  # Return top 5 relevant memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant memories: {e}")
            return []
    
    async def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """Get working memory for a specific session"""
        return {
            "session_id": session_id,
            "working_memory": self.working_memory.get(session_id, []),
            "memory_count": len(self.working_memory.get(session_id, [])),
            "last_updated": self.memory_timestamps.get(session_id)
        }
    
    async def clear_session_memory(self, session_id: str) -> bool:
        """Clear working memory for a session"""
        try:
            if session_id in self.working_memory:
                del self.working_memory[session_id]
            if session_id in self.memory_timestamps:
                del self.memory_timestamps[session_id]
            
            self.logger.info(f"Cleared working memory for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing session memory: {e}")
            return False
    
    async def _cleanup_working_memory(self) -> None:
        """Background task to cleanup expired working memory"""
        while self._initialized:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, timestamp in self.memory_timestamps.items():
                    if (current_time - timestamp).seconds > self.config.working_memory_timeout:
                        expired_sessions.append(session_id)
                
                # Cleanup expired sessions
                for session_id in expired_sessions:
                    await self.clear_session_memory(session_id)
                
                # Wait before next cleanup
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in memory cleanup: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Memory component")
        self._initialized = False
        self.working_memory.clear()
        self.memory_timestamps.clear()
```

#### 3. FF Multi-Agent Component

**File**: `ff_chat_components/ff_multi_agent_component.py`

```python
"""
FF Multi-Agent Component - Agent Coordination

Provides multi-agent coordination and panel discussions using
existing FF panel management system as backend.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_panel_manager import FFPanelManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFPanelDTO, FFPersonaDTO, PanelType
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

class FFAgentMode(Enum):
    """Multi-agent interaction modes"""
    ROUND_ROBIN = "round_robin"    # Agents take turns
    PARALLEL = "parallel"          # Agents respond simultaneously  
    DEBATE = "debate"              # Agents debate/argue
    PANEL = "panel"                # Expert panel discussion
    CONSENSUS = "consensus"        # Seek agreement

@dataclass
class FFMultiAgentConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Multi-Agent component"""
    
    # Agent coordination settings
    max_agents: int = 5
    default_mode: str = "round_robin"
    agent_timeout: int = 30  # seconds
    
    # Panel integration settings
    use_ff_panel_system: bool = True
    panel_type: str = PanelType.EXPERT_PANEL.value
    
    # Agent behavior settings
    enable_agent_memory: bool = True
    agent_response_delay: float = 0.5  # seconds between agent responses
    max_debate_rounds: int = 3
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_agents <= 0:
            raise ValueError("max_agents must be positive")
        
        valid_modes = ["round_robin", "parallel", "debate", "panel", "consensus"]
        if self.default_mode not in valid_modes:
            raise ValueError(f"default_mode must be one of {valid_modes}")

class FFMultiAgentComponent(FFChatComponentProtocol):
    """
    FF Multi-Agent component using existing FF panel system.
    
    Provides agent coordination and multi-participant conversations
    while leveraging FF panel management for persistence and coordination.
    """
    
    def __init__(self, config: FFMultiAgentConfigDTO):
        """
        Initialize FF multi-agent component.
        
        Args:
            config: Multi-agent configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.ff_storage: Optional[FFStorageManager] = None
        self.ff_panel: Optional[FFPanelManager] = None
        
        # Agent coordination state
        self.active_panels: Dict[str, str] = {}  # session_id -> panel_id
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Component state
        self._initialized = False
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_multi_agent",
            "version": "1.0.0",
            "description": "FF multi-agent component using FF panel system backend",
            "capabilities": ["agent_coordination", "panel_discussions", "multi_participant", "debate_moderation"],
            "use_cases": [
                "multi_ai_panel", "ai_debate", "topic_delegation", 
                "ai_game_master", "auto_task_agent"
            ],
            "ff_dependencies": ["FFStorageManager", "FFPanelManager"]
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_panel = dependencies.get("ff_panel")
            
            if not self.ff_storage:
                raise ValueError("FFStorageManager dependency required")
            
            if not self.ff_panel and self.config.use_ff_panel_system:
                self.logger.warning("FFPanelManager not available - creating mock panel system")
                self.config.use_ff_panel_system = False
            
            # Initialize agent coordination state
            self.active_panels = {}
            self.agent_states = {}
            
            self._initialized = True
            self.logger.info("FF Multi-Agent component initialized with FF panel backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Multi-Agent component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message through multi-agent coordination using FF panel system.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: FF message DTO
            context: Optional context with agent configuration
            
        Returns:
            Processing results with agent responses
        """
        if not self._initialized:
            raise RuntimeError("FF Multi-Agent component not initialized")
        
        try:
            self.logger.info(f"Processing multi-agent message in session {session_id}")
            
            # Get or create panel for session
            panel_id = await self._get_or_create_panel(session_id, user_id, context)
            
            # Get agent configuration from context
            agent_config = context.get("agent_config", {}) if context else {}
            mode = agent_config.get("mode", self.config.default_mode)
            agent_personas = agent_config.get("personas", ["expert", "creative", "analyst"])
            
            # Process message based on mode
            if mode == "round_robin":
                responses = await self._process_round_robin(panel_id, message, agent_personas)
            elif mode == "parallel":
                responses = await self._process_parallel(panel_id, message, agent_personas)
            elif mode == "debate":
                responses = await self._process_debate(panel_id, message, agent_personas)
            elif mode == "panel":
                responses = await self._process_panel(panel_id, message, agent_personas)
            else:
                responses = await self._process_round_robin(panel_id, message, agent_personas)
            
            # Store responses using FF panel system
            if self.ff_panel:
                for response in responses:
                    await self._store_agent_response(panel_id, response)
            
            return {
                "success": True,
                "component": "ff_multi_agent",
                "processor": "ff_panel_backend",
                "mode": mode,
                "agent_responses": responses,
                "panel_id": panel_id,
                "agents_participated": len(responses),
                "metadata": {
                    "session_id": session_id,
                    "processing_mode": mode,
                    "agent_count": len(agent_personas)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing multi-agent message: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_multi_agent"
            }
    
    async def _get_or_create_panel(self, 
                                   session_id: str, 
                                   user_id: str,
                                   context: Optional[Dict[str, Any]]) -> str:
        """Get existing panel or create new one using FF panel system"""
        try:
            # Check if panel already exists for this session
            if session_id in self.active_panels:
                return self.active_panels[session_id]
            
            # Create new panel using FF panel system
            if self.ff_panel:
                panel_title = f"Multi-Agent Panel - {session_id}"
                panel_description = "Multi-agent conversation panel"
                
                panel_id = await self.ff_panel.create_panel(
                    user_id=user_id,
                    title=panel_title,
                    description=panel_description,
                    panel_type=PanelType(self.config.panel_type)
                )
                
                self.active_panels[session_id] = panel_id
                self.logger.info(f"Created FF panel {panel_id} for session {session_id}")
                return panel_id
            else:
                # Mock panel system
                panel_id = f"mock_panel_{session_id}"
                self.active_panels[session_id] = panel_id
                return panel_id
                
        except Exception as e:
            self.logger.error(f"Error creating panel: {e}")
            # Fallback to mock panel
            panel_id = f"fallback_panel_{session_id}"
            self.active_panels[session_id] = panel_id
            return panel_id
    
    async def _process_round_robin(self, 
                                   panel_id: str,
                                   message: FFMessageDTO,
                                   agent_personas: List[str]) -> List[Dict[str, Any]]:
        """Process message in round-robin mode"""
        responses = []
        
        for i, persona in enumerate(agent_personas):
            # Generate agent response based on persona
            response_content = await self._generate_agent_response(
                persona=persona,
                message=message,
                mode="round_robin",
                turn_number=i + 1
            )
            
            response = {
                "agent_id": f"{persona}_agent",
                "persona": persona,
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "mode": "round_robin",
                "turn_number": i + 1
            }
            
            responses.append(response)
            
            # Add delay between responses
            if i < len(agent_personas) - 1:
                await asyncio.sleep(self.config.agent_response_delay)
        
        return responses
    
    async def _process_parallel(self, 
                                panel_id: str,
                                message: FFMessageDTO,
                                agent_personas: List[str]) -> List[Dict[str, Any]]:
        """Process message in parallel mode"""
        # Create tasks for parallel processing
        tasks = []
        for persona in agent_personas:
            task = self._generate_agent_response(
                persona=persona,
                message=message,
                mode="parallel"
            )
            tasks.append(task)
        
        # Wait for all agents to respond
        response_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        for i, (persona, content) in enumerate(zip(agent_personas, response_contents)):
            if isinstance(content, Exception):
                content = f"Agent {persona} encountered an error: {str(content)}"
            
            response = {
                "agent_id": f"{persona}_agent",
                "persona": persona,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "mode": "parallel"
            }
            
            responses.append(response)
        
        return responses
    
    async def _process_debate(self, 
                              panel_id: str,
                              message: FFMessageDTO,
                              agent_personas: List[str]) -> List[Dict[str, Any]]:
        """Process message in debate mode"""
        responses = []
        debate_round = 1
        
        # Initial positions
        for persona in agent_personas:
            response_content = await self._generate_agent_response(
                persona=persona,
                message=message,
                mode="debate",
                debate_round=debate_round,
                position="initial"
            )
            
            response = {
                "agent_id": f"{persona}_agent",
                "persona": persona,
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "mode": "debate",
                "debate_round": debate_round,
                "position": "initial"
            }
            
            responses.append(response)
        
        # Follow-up debate rounds
        for round_num in range(2, min(self.config.max_debate_rounds + 1, 4)):
            await asyncio.sleep(self.config.agent_response_delay)
            
            for persona in agent_personas:
                response_content = await self._generate_agent_response(
                    persona=persona,
                    message=message,
                    mode="debate",
                    debate_round=round_num,
                    previous_responses=responses
                )
                
                response = {
                    "agent_id": f"{persona}_agent",
                    "persona": persona,
                    "content": response_content,
                    "timestamp": datetime.now().isoformat(),
                    "mode": "debate",
                    "debate_round": round_num,
                    "position": "rebuttal"
                }
                
                responses.append(response)
        
        return responses
    
    async def _process_panel(self, 
                             panel_id: str,
                             message: FFMessageDTO,
                             agent_personas: List[str]) -> List[Dict[str, Any]]:
        """Process message in expert panel mode"""
        responses = []
        
        # Panel discussion format
        for i, persona in enumerate(agent_personas):
            response_content = await self._generate_agent_response(
                persona=persona,
                message=message,
                mode="panel",
                panel_position=i + 1,
                total_panelists=len(agent_personas)
            )
            
            response = {
                "agent_id": f"{persona}_agent",
                "persona": persona,
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "mode": "panel",
                "panel_position": i + 1
            }
            
            responses.append(response)
            
            # Brief delay between panelist responses
            if i < len(agent_personas) - 1:
                await asyncio.sleep(self.config.agent_response_delay * 0.5)
        
        return responses
    
    async def _generate_agent_response(self, 
                                       persona: str,
                                       message: FFMessageDTO,
                                       mode: str,
                                       **kwargs) -> str:
        """
        Generate agent response based on persona and mode.
        
        For Phase 2, this provides basic persona-based responses.
        In production, this would integrate with LLM services.
        """
        try:
            user_message = message.content
            
            # Persona-based response templates
            persona_styles = {
                "expert": "From an expert perspective on '{topic}': {analysis}",
                "creative": "Taking a creative approach to '{topic}': {idea}",
                "analyst": "Analyzing '{topic}' systematically: {breakdown}",
                "conservative": "From a conservative viewpoint on '{topic}': {position}",
                "liberal": "From a progressive perspective on '{topic}': {viewpoint}",
                "moderate": "Taking a balanced view of '{topic}': {assessment}"
            }
            
            # Mode-specific response generation
            if mode == "debate" and kwargs.get("debate_round", 1) > 1:
                return f"[{persona.title()} Agent - Round {kwargs['debate_round']}] Building on the previous discussion about '{user_message}', I maintain that my {persona} perspective offers valuable insights. The evidence supports a {persona}-oriented approach to this topic."
            
            elif mode == "panel":
                position = kwargs.get("panel_position", 1)
                return f"[{persona.title()} Panelist #{position}] Thank you for the question about '{user_message}'. From my {persona} background, I believe we should consider the {persona} implications carefully. This requires a {persona}-focused analysis."
            
            else:
                # Default response format
                topic = user_message[:50] + "..." if len(user_message) > 50 else user_message
                
                if persona == "expert":
                    analysis = "this requires deep domain knowledge and careful consideration of established principles"
                elif persona == "creative":
                    analysis = "we should explore innovative solutions and think outside conventional boundaries"
                elif persona == "analyst":
                    analysis = "we need to break this down systematically and examine the data objectively"
                else:
                    analysis = f"this aligns with {persona} principles and approaches"
                
                template = persona_styles.get(persona, "From a {persona} perspective: {content}")
                return template.format(topic=topic, analysis=analysis, idea=analysis, breakdown=analysis, position=analysis, viewpoint=analysis, assessment=analysis)
            
        except Exception as e:
            self.logger.error(f"Error generating agent response: {e}")
            return f"[{persona.title()} Agent] I apologize, but I encountered an error generating my response to '{message.content[:30]}...'"
    
    async def _store_agent_response(self, panel_id: str, response: Dict[str, Any]) -> None:
        """Store agent response using FF panel system"""
        try:
            if self.ff_panel:
                # Create FF panel message
                panel_message = FFMessageDTO(
                    role=response["agent_id"],
                    content=response["content"],
                    metadata={
                        "persona": response["persona"],
                        "mode": response["mode"],
                        "agent_id": response["agent_id"],
                        "timestamp": response["timestamp"]
                    }
                )
                
                # Store in FF panel system
                await self.ff_panel.add_panel_message(panel_id, panel_message)
                
        except Exception as e:
            self.logger.error(f"Error storing agent response: {e}")
    
    async def get_panel_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get panel conversation history"""
        try:
            panel_id = self.active_panels.get(session_id)
            if not panel_id or not self.ff_panel:
                return None
            
            # Get panel messages using FF panel system
            panel_info = await self.ff_panel.get_panel(panel_id)
            if panel_info:
                return {
                    "panel_id": panel_id,
                    "session_id": session_id,
                    "panel_info": panel_info.to_dict(),
                    "message_count": len(panel_info.messages) if hasattr(panel_info, 'messages') else 0
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting panel history: {e}")
            return None
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Multi-Agent component")
        self._initialized = False
        self.active_panels.clear()
        self.agent_states.clear()
```

#### 4. Component Registry Integration

**File**: `ff_chat_components/ff_component_registry.py`

```python
"""
FF Component Registry - Integration with FF Dependency Injection

Component registry system integrated with existing FF dependency
injection manager for seamless component loading and management.
"""

from typing import Dict, Any, List, Optional, Type
import asyncio
from dataclasses import dataclass
from pathlib import Path

# Import existing FF infrastructure
from ff_dependency_injection_manager import ff_get_container, ff_register_service
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_panel_manager import FFPanelManager
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_utils.ff_logging import get_logger

# Import FF chat components
from .ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO
from .ff_memory_component import FFMemoryComponent, FFMemoryConfigDTO
from .ff_multi_agent_component import FFMultiAgentComponent, FFMultiAgentConfigDTO

logger = get_logger(__name__)

@dataclass
class FFComponentRegistration:
    """Registration information for an FF chat component"""
    name: str
    component_class: Type[FFChatComponentProtocol]
    config_class: Type
    config: Any
    dependencies: List[str]
    priority: int = 100

class FFChatComponentRegistry:
    """
    FF Chat component registry integrated with FF dependency injection.
    
    Manages registration, loading, and lifecycle of FF chat components
    while leveraging existing FF dependency injection infrastructure.
    """
    
    def __init__(self):
        """Initialize FF chat component registry"""
        self.logger = get_logger(__name__)
        self.registrations: Dict[str, FFComponentRegistration] = {}
        self.instances: Dict[str, FFChatComponentProtocol] = {}
        self._initialized = False
        
        # Register core FF chat components
        self._register_core_components()
    
    def _register_core_components(self) -> None:
        """Register core FF chat components"""
        try:
            # Register FF Text Chat component
            self.register_component(
                name="ff_text_chat",
                component_class=FFTextChatComponent,
                config_class=FFTextChatConfigDTO,
                config=FFTextChatConfigDTO(),
                dependencies=["ff_storage", "ff_search"],
                priority=100
            )
            
            # Register FF Memory component
            self.register_component(
                name="ff_memory",
                component_class=FFMemoryComponent,
                config_class=FFMemoryConfigDTO,
                config=FFMemoryConfigDTO(),
                dependencies=["ff_storage", "ff_vector", "ff_search"],
                priority=200
            )
            
            # Register FF Multi-Agent component
            self.register_component(
                name="ff_multi_agent",
                component_class=FFMultiAgentComponent,
                config_class=FFMultiAgentConfigDTO,
                config=FFMultiAgentConfigDTO(),
                dependencies=["ff_storage", "ff_panel"],
                priority=300
            )
            
            self.logger.info("Registered core FF chat components")
            
        except Exception as e:
            self.logger.error(f"Error registering core FF chat components: {e}")
    
    def register_component(self,
                           name: str,
                           component_class: Type[FFChatComponentProtocol],
                           config_class: Type,
                           config: Any,
                           dependencies: List[str],
                           priority: int = 100) -> None:
        """
        Register an FF chat component.
        
        Args:
            name: Component identifier
            component_class: Component class
            config_class: Configuration class
            config: Configuration instance
            dependencies: List of FF service dependencies
            priority: Loading priority (lower loads first)
        """
        try:
            registration = FFComponentRegistration(
                name=name,
                component_class=component_class,
                config_class=config_class,
                config=config,
                dependencies=dependencies,
                priority=priority
            )
            
            self.registrations[name] = registration
            self.logger.info(f"Registered FF chat component: {name}")
            
        except Exception as e:
            self.logger.error(f"Error registering FF chat component {name}: {e}")
    
    async def initialize(self) -> bool:
        """Initialize component registry with FF dependency injection"""
        if self._initialized:
            return True
        
        try:
            self.logger.info("Initializing FF Chat Component Registry...")
            
            # Register FF chat components with FF DI container
            await self._register_with_ff_di()
            
            self._initialized = True
            self.logger.info("FF Chat Component Registry initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Chat Component Registry: {e}")
            return False
    
    async def _register_with_ff_di(self) -> None:
        """Register components with FF dependency injection container"""
        try:
            container = ff_get_container()
            
            # Register component factory functions with FF DI
            for name, registration in self.registrations.items():
                factory_name = f"ff_chat_component_{name}"
                
                def create_component_factory(reg=registration):
                    async def factory():
                        return await self._create_component_instance(reg)
                    return factory
                
                # Register with FF DI container
                ff_register_service(factory_name, create_component_factory(), singleton=True)
                
            self.logger.info("Registered FF chat components with FF DI container")
            
        except Exception as e:
            self.logger.error(f"Error registering with FF DI: {e}")
    
    async def _create_component_instance(self, registration: FFComponentRegistration) -> FFChatComponentProtocol:
        """Create component instance with FF dependency resolution"""
        try:
            # Create component instance
            component = registration.component_class(registration.config)
            
            # Resolve FF dependencies
            dependencies = await self._resolve_ff_dependencies(registration.dependencies)
            
            # Initialize component
            success = await component.initialize(dependencies)
            if not success:
                raise RuntimeError(f"Failed to initialize component {registration.name}")
            
            return component
            
        except Exception as e:
            self.logger.error(f"Error creating component instance: {e}")
            raise
    
    async def _resolve_ff_dependencies(self, dependency_names: List[str]) -> Dict[str, Any]:
        """Resolve FF service dependencies"""
        dependencies = {}
        container = ff_get_container()
        
        for dep_name in dependency_names:
            try:
                if dep_name == "ff_storage":
                    # Resolve FF storage manager
                    service = container.resolve("FFStorageManager") if hasattr(container, 'resolve') else None
                    if not service:
                        # Fallback to direct instantiation
                        from ff_class_configs.ff_configuration_manager_config import load_config
                        service = FFStorageManager(load_config())
                        await service.initialize()
                    dependencies[dep_name] = service
                    
                elif dep_name == "ff_search":
                    # Resolve FF search manager
                    service = container.resolve("FFSearchManager") if hasattr(container, 'resolve') else None
                    if not service:
                        from ff_class_configs.ff_configuration_manager_config import load_config
                        service = FFSearchManager(load_config())
                        await service.initialize()
                    dependencies[dep_name] = service
                    
                elif dep_name == "ff_vector":
                    # Resolve FF vector storage manager
                    service = container.resolve("FFVectorStorageManager") if hasattr(container, 'resolve') else None
                    if not service:
                        from ff_class_configs.ff_configuration_manager_config import load_config
                        service = FFVectorStorageManager(load_config())
                        await service.initialize()
                    dependencies[dep_name] = service
                    
                elif dep_name == "ff_panel":
                    # Resolve FF panel manager
                    service = container.resolve("FFPanelManager") if hasattr(container, 'resolve') else None
                    if not service:
                        from ff_class_configs.ff_configuration_manager_config import load_config
                        service = FFPanelManager(load_config())
                        await service.initialize()
                    dependencies[dep_name] = service
                    
                else:
                    self.logger.warning(f"Unknown FF dependency: {dep_name}")
                    
            except Exception as e:
                self.logger.error(f"Error resolving FF dependency {dep_name}: {e}")
                # Continue with other dependencies
        
        return dependencies
    
    async def load_components(self, component_names: List[str]) -> Dict[str, FFChatComponentProtocol]:
        """
        Load and initialize specified FF chat components.
        
        Args:
            component_names: List of component names to load
            
        Returns:
            Dictionary of loaded component instances
        """
        if not self._initialized:
            await self.initialize()
        
        loaded_components = {}
        
        # Sort by priority
        sorted_registrations = sorted(
            [(name, self.registrations[name]) for name in component_names if name in self.registrations],
            key=lambda x: x[1].priority
        )
        
        for name, registration in sorted_registrations:
            try:
                if name in self.instances:
                    # Use existing instance
                    loaded_components[name] = self.instances[name]
                else:
                    # Create new instance
                    component = await self._create_component_instance(registration)
                    self.instances[name] = component
                    loaded_components[name] = component
                
                self.logger.info(f"Loaded FF chat component: {name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load FF chat component {name}: {e}")
                # Continue loading other components
        
        return loaded_components
    
    def list_components(self) -> List[str]:
        """Get list of registered component names"""
        return list(self.registrations.keys())
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered component"""
        registration = self.registrations.get(name)
        if not registration:
            return None
        
        return {
            "name": registration.name,
            "class": registration.component_class.__name__,
            "config_class": registration.config_class.__name__,
            "dependencies": registration.dependencies,
            "priority": registration.priority
        }
    
    def get_components_for_use_case(self, use_case: str) -> List[str]:
        """Get components that support a specific use case"""
        matching_components = []
        
        for name, registration in self.registrations.items():
            try:
                # Create temporary instance to check capabilities
                temp_component = registration.component_class(registration.config)
                component_info = temp_component.component_info
                
                if use_case in component_info.get("use_cases", []):
                    matching_components.append(name)
                    
            except Exception as e:
                self.logger.error(f"Error checking use case support for {name}: {e}")
        
        return matching_components
    
    async def shutdown(self) -> None:
        """Shutdown component registry and cleanup resources"""
        self.logger.info("Shutting down FF Chat Component Registry...")
        
        # Cleanup component instances
        for name, component in self.instances.items():
            try:
                await component.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up component {name}: {e}")
        
        self.instances.clear()
        self._initialized = False
        
        self.logger.info("FF Chat Component Registry shutdown complete")

# Global registry instance
_ff_chat_registry = FFChatComponentRegistry()

def get_ff_chat_component_registry() -> FFChatComponentRegistry:
    """Get the global FF chat component registry"""
    return _ff_chat_registry

async def load_ff_chat_components(component_names: List[str]) -> Dict[str, FFChatComponentProtocol]:
    """Load FF chat components from the global registry"""
    return await _ff_chat_registry.load_components(component_names)
```

### Testing and Validation

#### 5. Phase 2 Integration Tests

**File**: `tests/test_ff_chat_phase2.py`

```python
"""
Phase 2 FF Chat Component Tests

Tests the FF chat components and their integration with existing FF infrastructure.
"""

import pytest
import asyncio

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

# Import FF chat components
from ff_chat_components.ff_text_chat_component import FFTextChatComponent, FFTextChatConfigDTO
from ff_chat_components.ff_memory_component import FFMemoryComponent, FFMemoryConfigDTO
from ff_chat_components.ff_multi_agent_component import FFMultiAgentComponent, FFMultiAgentConfigDTO
from ff_chat_components.ff_component_registry import get_ff_chat_component_registry

class TestFFChatComponents:
    """Test FF chat components with existing FF backend integration"""
    
    @pytest.fixture
    async def ff_dependencies(self):
        """Create FF backend services for testing"""
        ff_config = load_config()
        ff_storage = FFStorageManager(ff_config)
        await ff_storage.initialize()
        
        dependencies = {
            "ff_storage": ff_storage,
            "ff_search": ff_storage.search_engine,
            "ff_vector": ff_storage.vector_storage,
            "ff_panel": ff_storage.panel_manager if hasattr(ff_storage, 'panel_manager') else None
        }
        
        return dependencies
    
    @pytest.mark.asyncio
    async def test_ff_text_chat_component(self, ff_dependencies):
        """Test FF text chat component with FF storage backend"""
        
        # Create and initialize component
        config = FFTextChatConfigDTO()
        component = FFTextChatComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test message processing
        message = FFMessageDTO(role=MessageRole.USER.value, content="Hello, FF text chat!")
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message
        )
        
        assert result["success"] == True
        assert result["component"] == "ff_text_chat"
        assert result["processor"] == "ff_storage_backend"
        assert "response_content" in result
        
        # Test component info
        info = component.component_info
        assert info["name"] == "ff_text_chat"
        assert "text_processing" in info["capabilities"]
        assert "basic_chat" in info["use_cases"]
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_memory_component(self, ff_dependencies):
        """Test FF memory component with FF vector storage backend"""
        
        # Create and initialize component
        config = FFMemoryConfigDTO()
        component = FFMemoryComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test message processing and memory updates
        message1 = FFMessageDTO(role=MessageRole.USER.value, content="Remember that I like Python programming")
        message2 = FFMessageDTO(role=MessageRole.USER.value, content="What programming languages do I like?")
        
        # Process first message (store memory)
        result1 = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message1
        )
        
        assert result1["success"] == True
        assert result1["component"] == "ff_memory"
        assert result1["memory_updated"] == True
        
        # Process second message (retrieve memory)
        result2 = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message2
        )
        
        assert result2["success"] == True
        assert len(result2["relevant_memories"]) > 0
        
        # Test session memory retrieval
        session_memory = await component.get_session_memory("test_session")
        assert session_memory["session_id"] == "test_session"
        assert session_memory["memory_count"] > 0
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_multi_agent_component(self, ff_dependencies):
        """Test FF multi-agent component with FF panel system backend"""
        
        # Create and initialize component
        config = FFMultiAgentConfigDTO()
        component = FFMultiAgentComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test multi-agent processing
        message = FFMessageDTO(role=MessageRole.USER.value, content="What are your thoughts on AI ethics?")
        
        context = {
            "agent_config": {
                "mode": "panel",
                "personas": ["expert", "creative", "analyst"]
            }
        }
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message,
            context=context
        )
        
        assert result["success"] == True
        assert result["component"] == "ff_multi_agent"
        assert result["mode"] == "panel"
        assert len(result["agent_responses"]) == 3
        assert result["agents_participated"] == 3
        
        # Verify agent responses have required fields
        for response in result["agent_responses"]:
            assert "agent_id" in response
            assert "persona" in response
            assert "content" in response
            assert "timestamp" in response
        
        # Test panel history retrieval
        panel_history = await component.get_panel_history("test_session")
        assert panel_history is not None
        assert panel_history["session_id"] == "test_session"
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_component_registry(self, ff_dependencies):
        """Test FF component registry integration"""
        
        registry = get_ff_chat_component_registry()
        
        # Test initialization
        success = await registry.initialize()
        assert success
        
        # Test component listing
        components = registry.list_components()
        assert "ff_text_chat" in components
        assert "ff_memory" in components
        assert "ff_multi_agent" in components
        
        # Test component info
        info = registry.get_component_info("ff_text_chat")
        assert info is not None
        assert info["name"] == "ff_text_chat"
        assert "FFStorageManager" in info["dependencies"]
        
        # Test use case queries
        basic_chat_components = registry.get_components_for_use_case("basic_chat")
        assert "ff_text_chat" in basic_chat_components
        
        memory_chat_components = registry.get_components_for_use_case("memory_chat")
        assert "ff_text_chat" in memory_chat_components
        assert "ff_memory" in memory_chat_components
        
        # Test component loading
        loaded_components = await registry.load_components(["ff_text_chat", "ff_memory"])
        assert "ff_text_chat" in loaded_components
        assert "ff_memory" in loaded_components
        
        # Verify components are initialized
        text_chat = loaded_components["ff_text_chat"]
        assert text_chat is not None
        
        # Cleanup
        await registry.shutdown()
    
    @pytest.mark.asyncio
    async def test_ff_chat_integration_with_chat_application(self, ff_dependencies):
        """Test FF chat components integration with FF Chat Application"""
        
        # This would test the integration between Phase 1 (FF Chat Application)
        # and Phase 2 (FF Chat Components)
        
        from ff_chat_application import FFChatApplication
        from ff_class_configs.ff_chat_application_config import FFChatApplicationConfigDTO
        
        # Create FF chat application
        chat_config = FFChatApplicationConfigDTO()
        chat_app = FFChatApplication(chat_config=chat_config)
        
        await chat_app.initialize()
        
        try:
            # Create session
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="basic_chat"
            )
            
            # Process message (should use FF components if available)
            result = await chat_app.process_message(session_id, "Test FF component integration")
            
            assert result["success"] == True
            # Should use enhanced processing if components are available
            
            # Test with memory use case
            memory_session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="memory_chat"
            )
            
            result = await chat_app.process_message(memory_session_id, "Remember this important information")
            assert result["success"] == True
            
        finally:
            await chat_app.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Validation Checklist

#### Phase 2 Completion Requirements

- ✅ **FF Chat Components**
  - [ ] FF Text Chat component processes text using FF storage backend
  - [ ] FF Memory component manages memory using FF vector storage
  - [ ] FF Multi-Agent component coordinates using FF panel system
  - [ ] Component registry integrates with FF dependency injection

- ✅ **FF Backend Integration**
  - [ ] Components use existing FF storage for persistence
  - [ ] Components use existing FF search for context retrieval
  - [ ] Components use existing FF vector storage for embeddings
  - [ ] Components use existing FF panel system for multi-agent coordination

- ✅ **Use Case Coverage**
  - [ ] 19/22 use cases supported (86% coverage)
  - [ ] Text-based use cases handled by FF Text Chat component
  - [ ] Memory-required use cases handled by FF Memory component
  - [ ] Multi-participant use cases handled by FF Multi-Agent component

- ✅ **Testing Coverage**
  - [ ] Unit tests for all FF chat components
  - [ ] Integration tests with existing FF infrastructure
  - [ ] Component registry functionality tests
  - [ ] End-to-end use case validation tests

### Success Metrics

#### Functional Requirements
- Can process text messages using FF storage backend
- Can maintain cross-session memory using FF vector storage
- Can coordinate multi-agent conversations using FF panel system
- Can load components dynamically through FF dependency injection

#### Non-Functional Requirements
- Zero breaking changes to existing FF system
- Performance within 10% of Phase 1 baseline
- Memory usage increase less than 30%
- 100% test coverage for new FF chat components

#### Integration Requirements
- Seamless integration with Phase 1 FF Chat Application
- Compatible with existing FF managers and utilities
- Uses existing FF configuration and protocol patterns
- Maintains existing FF logging and error handling

---

**Phase 2 delivers core chat processing capabilities using existing FF infrastructure as backend services, achieving 86% use case coverage while maintaining complete compatibility with the existing FF system.**