"""
FF Chat Component Protocol - Phase 2 Component Interface Definitions

Defines the protocol interfaces that all FF chat components must implement
for integration with the FF Chat Application and existing FF managers.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Import existing FF infrastructure
from ff_class_configs.ff_chat_entities_config import FFMessageDTO


@runtime_checkable
class FFChatComponentProtocol(Protocol):
    """
    FF Chat component protocol following FF protocol patterns.
    
    All FF chat components must implement this protocol to ensure
    consistent integration with the FF Chat Application system.
    """
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """
        Component metadata and capabilities.
        
        Returns:
            Dictionary containing component information:
            - name: Component identifier
            - version: Component version
            - description: Human-readable description
            - capabilities: List of capabilities
            - use_cases: List of supported use cases
            - ff_dependencies: List of required FF managers
        """
        ...
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances:
                - ff_storage: FFStorageManager instance
                - ff_search: FFSearchManager instance (optional)
                - ff_vector: FFVectorStorageManager instance (optional)
                - ff_panel: FFPanelManager instance (optional)
                - ff_document: FFDocumentProcessingManager instance (optional)
                
        Returns:
            True if initialization successful, False otherwise
        """
        ...
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process chat message using FF backend services.
        
        Args:
            session_id: FF storage session identifier
            user_id: User identifier
            message: FF message DTO with content and metadata
            context: Optional processing context and parameters
            
        Returns:
            Processing results dictionary:
            - success: Boolean indicating success/failure
            - response_content: Generated response content (if applicable)
            - component: Component identifier
            - processor: Backend processor used
            - metadata: Additional processing metadata
            - error: Error message (if success=False)
        """
        ...
    
    async def get_capabilities(self) -> List[str]:
        """
        Get list of component capabilities.
        
        Returns:
            List of capability identifiers
        """
        ...
    
    async def supports_use_case(self, use_case: str) -> bool:
        """
        Check if component supports a specific use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if use case is supported
        """
        ...
    
    async def cleanup(self) -> None:
        """
        Cleanup component resources following FF patterns.
        
        Should clean up any resources, connections, or background tasks
        created during component operation.
        """
        ...


@runtime_checkable 
class FFTextChatComponentProtocol(FFChatComponentProtocol, Protocol):
    """
    Protocol for text chat components.
    
    Extends base component protocol with text-specific capabilities.
    """
    
    async def generate_response(self,
                                conversation_context: List[FFMessageDTO],
                                current_message: FFMessageDTO,
                                response_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text response based on conversation context.
        
        Args:
            conversation_context: Previous messages in conversation
            current_message: Current user message
            response_context: Optional context for response generation
            
        Returns:
            Generated response text
        """
        ...
    
    async def get_conversation_context(self, 
                                       user_id: str,
                                       session_id: str,
                                       limit: int = 10) -> List[FFMessageDTO]:
        """
        Get conversation context from FF storage.
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        ...


@runtime_checkable
class FFMemoryComponentProtocol(FFChatComponentProtocol, Protocol):
    """
    Protocol for memory components.
    
    Extends base component protocol with memory-specific capabilities.
    """
    
    async def store_memory(self,
                           user_id: str,
                           session_id: str,
                           memory_content: str,
                           memory_type: str = "episodic",
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store memory using FF vector storage.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            memory_content: Content to store in memory
            memory_type: Type of memory (episodic, semantic, procedural)
            metadata: Optional memory metadata
            
        Returns:
            True if memory stored successfully
        """
        ...
    
    async def retrieve_memories(self,
                                user_id: str,
                                query: str,
                                memory_types: Optional[List[str]] = None,
                                limit: int = 5,
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using FF vector search.
        
        Args:
            user_id: User identifier
            query: Query text for memory retrieval
            memory_types: Types of memory to search (optional)
            limit: Maximum memories to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant memory entries
        """
        ...
    
    async def update_working_memory(self,
                                    session_id: str,
                                    message: FFMessageDTO) -> None:
        """
        Update working memory for current session.
        
        Args:
            session_id: Session identifier
            message: Message to add to working memory
        """
        ...
    
    async def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """
        Get working memory for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session memory information
        """
        ...


@runtime_checkable
class FFMultiAgentComponentProtocol(FFChatComponentProtocol, Protocol):
    """
    Protocol for multi-agent components.
    
    Extends base component protocol with multi-agent coordination capabilities.
    """
    
    async def create_agent_panel(self,
                                 session_id: str,
                                 user_id: str,
                                 agent_personas: List[str],
                                 coordination_mode: str = "round_robin") -> str:
        """
        Create agent panel using FF panel manager.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            agent_personas: List of agent personas to create
            coordination_mode: Agent coordination mode
            
        Returns:
            Panel identifier
        """
        ...
    
    async def process_multi_agent_message(self,
                                          panel_id: str,
                                          message: FFMessageDTO,
                                          agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process message through multiple agents.
        
        Args:
            panel_id: Panel identifier
            message: Message to process
            agent_config: Agent configuration parameters
            
        Returns:
            List of agent responses
        """
        ...
    
    async def coordinate_agents(self,
                                agents: List[str],
                                coordination_mode: str,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple agents for response generation.
        
        Args:
            agents: List of agent identifiers
            coordination_mode: Mode of coordination
            context: Coordination context
            
        Returns:
            Coordination results
        """
        ...
    
    async def get_panel_history(self, panel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get panel conversation history using FF panel manager.
        
        Args:
            panel_id: Panel identifier
            
        Returns:
            Panel history information or None if not found
        """
        ...


@runtime_checkable
class FFComponentRegistryProtocol(Protocol):
    """
    Protocol for component registry.
    
    Defines interface for managing FF chat component lifecycle.
    """
    
    async def initialize(self) -> bool:
        """
        Initialize component registry with FF dependency injection.
        
        Returns:
            True if initialization successful
        """
        ...
    
    def register_component(self,
                           name: str,
                           component_class: type,
                           config_class: type,
                           config: Any,
                           dependencies: List[str],
                           priority: int = 100) -> None:
        """
        Register FF chat component.
        
        Args:
            name: Component identifier
            component_class: Component class
            config_class: Configuration class  
            config: Configuration instance
            dependencies: List of FF service dependencies
            priority: Loading priority
        """
        ...
    
    async def load_components(self, component_names: List[str]) -> Dict[str, FFChatComponentProtocol]:
        """
        Load and initialize specified components.
        
        Args:
            component_names: List of component names to load
            
        Returns:
            Dictionary of loaded component instances
        """
        ...
    
    def list_components(self) -> List[str]:
        """
        Get list of registered component names.
        
        Returns:
            List of component names
        """
        ...
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered component.
        
        Args:
            name: Component name
            
        Returns:
            Component information or None if not found
        """
        ...
    
    def get_components_for_use_case(self, use_case: str) -> List[str]:
        """
        Get components that support a specific use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            List of component names supporting the use case
        """
        ...
    
    async def shutdown(self) -> None:
        """
        Shutdown component registry and cleanup resources.
        """
        ...


@dataclass
class FFComponentCapability:
    """
    Represents a component capability.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    ff_dependencies: List[str]


@dataclass
class FFComponentInfo:
    """
    Component information structure.
    """
    name: str
    version: str
    description: str
    capabilities: List[FFComponentCapability]
    use_cases: List[str]
    ff_dependencies: List[str]
    priority: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "parameters": cap.parameters,
                    "ff_dependencies": cap.ff_dependencies
                }
                for cap in self.capabilities
            ],
            "use_cases": self.use_cases,
            "ff_dependencies": self.ff_dependencies,
            "priority": self.priority
        }


# Component type definitions for use case routing
COMPONENT_TYPE_TEXT_CHAT = "text_chat"
COMPONENT_TYPE_MEMORY = "memory"
COMPONENT_TYPE_MULTI_AGENT = "multi_agent"
COMPONENT_TYPE_TOOLS = "tools"
COMPONENT_TYPE_MULTIMODAL = "multimodal"
COMPONENT_TYPE_SEARCH = "search"
COMPONENT_TYPE_ROUTER = "router"
COMPONENT_TYPE_TRACE = "trace"
COMPONENT_TYPE_PERSONA = "persona"

# Use case to component mappings (from Phase 2 specifications)
USE_CASE_COMPONENT_MAPPINGS = {
    # Basic patterns (4 use cases)
    "basic_chat": [COMPONENT_TYPE_TEXT_CHAT],
    "multimodal_chat": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MULTIMODAL],
    "rag_chat": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH],
    "multimodal_rag": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MULTIMODAL, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH],
    
    # Specialized modes (9 use cases)
    "translation_chat": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MULTIMODAL, COMPONENT_TYPE_TOOLS, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_PERSONA],
    "personal_assistant": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_TOOLS, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_PERSONA],
    "interactive_tutor": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_PERSONA],
    "language_tutor": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_TOOLS, COMPONENT_TYPE_PERSONA],
    "exam_assistant": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH],
    "ai_notetaker": [COMPONENT_TYPE_MULTIMODAL, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH, COMPONENT_TYPE_TOOLS],
    "chatops_assistant": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_TOOLS],
    "cross_team_concierge": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH, COMPONENT_TYPE_TOOLS],
    "scene_critic": [COMPONENT_TYPE_MULTIMODAL, COMPONENT_TYPE_PERSONA],
    
    # Multi-participant (5 use cases)
    "multi_ai_panel": [COMPONENT_TYPE_MULTI_AGENT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_PERSONA],
    "ai_debate": [COMPONENT_TYPE_MULTI_AGENT, COMPONENT_TYPE_PERSONA, COMPONENT_TYPE_TRACE],
    "topic_delegation": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_SEARCH, COMPONENT_TYPE_MULTI_AGENT, COMPONENT_TYPE_ROUTER],
    "ai_game_master": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MULTI_AGENT, COMPONENT_TYPE_MEMORY],
    "auto_task_agent": [COMPONENT_TYPE_TOOLS, COMPONENT_TYPE_MULTI_AGENT, COMPONENT_TYPE_MEMORY],
    
    # Context & memory (3 use cases)
    "memory_chat": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY],
    "thought_partner": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY],
    "story_world_chat": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_PERSONA],
    
    # Development (1 use case)
    "prompt_sandbox": [COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_TRACE]
}


def get_required_components_for_use_case(use_case: str) -> List[str]:
    """
    Get list of required components for a use case.
    
    Args:
        use_case: Use case identifier
        
    Returns:
        List of required component types
    """
    return USE_CASE_COMPONENT_MAPPINGS.get(use_case, [COMPONENT_TYPE_TEXT_CHAT])


def get_use_cases_for_component(component_type: str) -> List[str]:
    """
    Get list of use cases that require a specific component.
    
    Args:
        component_type: Component type identifier
        
    Returns:
        List of use case identifiers
    """
    use_cases = []
    for use_case, components in USE_CASE_COMPONENT_MAPPINGS.items():
        if component_type in components:
            use_cases.append(use_case)
    return use_cases