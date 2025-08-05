"""
FF Chat Application - Main Orchestration Layer

Provides high-level interface for creating and managing chat applications
integrated with existing FF storage and processing systems.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import uuid

# Import existing FF infrastructure
from ff_core_storage_manager import FFCoreStorageManager
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO, MessageRole
from ff_utils.ff_logging import get_logger
from ff_dependency_injection_manager import ff_get_container

# Import new FF chat configurations
from ff_class_configs.ff_chat_application_config import FFChatApplicationConfigDTO
from ff_class_configs.ff_chat_session_config import FFChatSessionConfigDTO
from ff_class_configs.ff_chat_use_case_config import FFChatUseCaseConfigDTO

# Import new FF chat managers
from ff_chat_session_manager import FFChatSessionManager
from ff_chat_use_case_manager import FFChatUseCaseManager

# Import Phase 2 components
from ff_component_registry import FFComponentRegistry
from ff_class_configs.ff_component_registry_config import FFComponentRegistryConfigDTO

# Import protocols
from ff_protocols.ff_chat_protocol import FFChatApplicationProtocol
from ff_protocols.ff_chat_component_protocol import (
    get_required_components_for_use_case, COMPONENT_TYPE_TEXT_CHAT,
    COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_MULTI_AGENT
)

logger = get_logger(__name__)


class FFChatApplicationMode(Enum):
    """Chat application operation modes"""
    FF_STORAGE_ONLY = "ff_storage_only"    # Use existing FF storage only
    FF_CHAT_ENHANCED = "ff_chat_enhanced"  # Use FF storage + chat components
    FF_CHAT_FULL = "ff_chat_full"          # Full chat capabilities


@dataclass
class FFChatSession:
    """Represents an active FF chat session"""
    session_id: str
    user_id: str
    use_case: str
    ff_storage_session_id: str  # Underlying FF storage session ID
    context: Dict[str, Any]
    created_at: datetime
    active: bool = True


class FFChatApplication(FFChatApplicationProtocol):
    """
    Main orchestration layer for FF-integrated chat applications.
    
    Manages chat sessions, use cases, and components while using
    existing FF managers as backend services.
    """
    
    def __init__(self, 
                 ff_config: Optional[FFConfigurationManagerConfigDTO] = None,
                 chat_config: Optional[FFChatApplicationConfigDTO] = None):
        """
        Initialize FF chat application.
        
        Args:
            ff_config: Existing FF configuration (uses default if not provided)
            chat_config: Chat-specific configuration (uses default if not provided)
        """
        self.ff_config = ff_config or load_config()
        self.chat_config = chat_config or FFChatApplicationConfigDTO()
        
        # Initialize existing FF storage as backend
        self.ff_storage = FFCoreStorageManager(self.ff_config)
        
        # Initialize chat managers
        self.chat_session_manager = FFChatSessionManager(
            self.ff_storage, 
            self.chat_config.chat_session
        )
        self.use_case_manager = FFChatUseCaseManager(
            self.ff_storage,
            self.chat_config.chat_use_cases
        )
        
        # Initialize Phase 2 component registry
        registry_config = FFComponentRegistryConfigDTO()
        self.component_registry = FFComponentRegistry(registry_config)
        
        # Component management
        self._loaded_components: Dict[str, Any] = {}
        
        # State management
        self.active_sessions: Dict[str, FFChatSession] = {}
        self._initialized = False
        
        logger.info("FF Chat Application initialized with existing FF backend")
    
    async def initialize(self) -> bool:
        """Initialize the FF chat application and underlying FF systems"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing FF Chat Application with Phase 2 components...")
            
            # Initialize existing FF storage backend
            await self.ff_storage.initialize()
            
            # Initialize Phase 2 component registry
            await self.component_registry.initialize()
            
            # Initialize chat managers
            await self.chat_session_manager.initialize()
            await self.use_case_manager.initialize()
            
            # Pre-load essential components for better performance
            await self._preload_essential_components()
            
            self._initialized = True
            logger.info("FF Chat Application with Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FF Chat Application: {e}")
            return False
    
    async def _preload_essential_components(self) -> None:
        """Pre-load essential components for common use cases"""
        try:
            # Always load text chat component as it's used by most use cases
            essential_components = [COMPONENT_TYPE_TEXT_CHAT]
            
            # Load memory component if enabled in config
            if hasattr(self.chat_config, 'memory_enabled') and self.chat_config.memory_enabled:
                essential_components.append(COMPONENT_TYPE_MEMORY)
            
            # Load components
            self._loaded_components = await self.component_registry.load_components(essential_components)
            
            logger.info(f"Pre-loaded {len(self._loaded_components)} essential components")
            
        except Exception as e:
            logger.warning(f"Failed to pre-load essential components: {e}")
            # Continue initialization even if pre-loading fails
    
    async def create_chat_session(self, 
                                  user_id: str, 
                                  use_case: str,
                                  title: Optional[str] = None,
                                  custom_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session using FF storage backend.
        
        Args:
            user_id: User identifier
            use_case: Use case identifier
            title: Optional session title
            custom_config: Optional configuration overrides
            
        Returns:
            Chat session ID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate use case - check against our supported list
            supported_use_cases = list(self.list_use_cases().keys())
            if use_case not in supported_use_cases:
                # Check for case where use case manager fails with different error
                try:
                    is_supported = await self.use_case_manager.is_use_case_supported(use_case)
                    if not is_supported:
                        raise ValueError(f"Unsupported use case: {use_case}")
                except Exception as e:
                    if "Unknown use case" in str(e):
                        raise ValueError(f"Unsupported use case: {use_case}")
                    raise ValueError(f"Unsupported use case: {use_case}")
            
            # Create underlying FF storage session
            ff_session_id = await self.ff_storage.create_session(
                user_id=user_id,
                session_name=title or f"Chat Session - {use_case}"
            )
            
            # Generate chat session ID
            chat_session_id = f"chat_{uuid.uuid4().hex[:self.chat_config.session_id_length]}"
            
            # Create chat session wrapper
            chat_session = FFChatSession(
                session_id=chat_session_id,
                user_id=user_id,
                use_case=use_case,
                ff_storage_session_id=ff_session_id,
                context={
                    "use_case_config": await self.use_case_manager.get_use_case_config(use_case),
                    "custom_config": custom_config or {},
                    "message_count": 0,
                    "created_at": datetime.now().isoformat()
                },
                created_at=datetime.now()
            )
            
            # Register session
            self.active_sessions[chat_session_id] = chat_session
            
            # Initialize session in chat session manager
            await self.chat_session_manager.register_session(chat_session)
            
            logger.info(f"Created FF chat session {chat_session_id} for use case {use_case}")
            return chat_session_id
            
        except Exception as e:
            logger.error(f"Failed to create FF chat session: {e}")
            raise
    
    
    async def process_message(self, 
                              session_id: str, 
                              message: Union[str, Dict[str, Any]],
                              role: str = MessageRole.USER.value,
                              **kwargs) -> Dict[str, Any]:
        """
        Process a message in the context of a chat session using Phase 2 components.
        
        Args:
            session_id: Chat session identifier
            message: Message content
            role: Message role (user, assistant, system)
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results with response content
        """
        session = self.active_sessions.get(session_id)
        if not session or not session.active:
            raise ValueError(f"Invalid or inactive chat session: {session_id}")
        
        try:
            # Create FF message DTO
            if isinstance(message, str):
                ff_message = FFMessageDTO(role=role, content=message)
            else:
                ff_message = FFMessageDTO(
                    role=role,
                    content=message.get("content", ""),
                    attachments=message.get("attachments", []),
                    metadata=message.get("metadata", {})
                )
            
            # Store user message using existing FF storage
            await self.ff_storage.add_message(
                user_id=session.user_id,
                session_id=session.ff_storage_session_id,
                message=ff_message
            )
            
            # Get required components for this use case
            required_components = get_required_components_for_use_case(session.use_case)
            
            # Load components if not already loaded
            await self._ensure_components_loaded(required_components)
            
            # Process through Phase 2 components
            result = await self._process_with_components(
                session=session,
                ff_message=ff_message,
                required_components=required_components,
                **kwargs
            )
            
            # Store response using existing FF storage if generated
            if result.get("response_content"):
                response_message = FFMessageDTO(
                    role=MessageRole.ASSISTANT.value,
                    content=result["response_content"],
                    metadata={
                        "processor": result.get("processor", "ff_chat_phase2"),
                        "components_used": result.get("components_used", []),
                        "use_case": session.use_case
                    }
                )
                
                await self.ff_storage.add_message(
                    user_id=session.user_id,
                    session_id=session.ff_storage_session_id,
                    message=response_message
                )
            
            # Update session context
            session.context["message_count"] += 1
            if result.get("response_content"):
                session.context["message_count"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message in FF chat session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response_content": "I apologize, but I encountered an error processing your message."
            }
    
    async def _ensure_components_loaded(self, required_components: List[str]) -> None:
        """Ensure required components are loaded"""
        try:
            components_to_load = []
            for component_type in required_components:
                if component_type not in self._loaded_components:
                    components_to_load.append(component_type)
            
            if components_to_load:
                logger.debug(f"Loading additional components: {components_to_load}")
                new_components = await self.component_registry.load_components(components_to_load)
                self._loaded_components.update(new_components)
                
        except Exception as e:
            logger.error(f"Failed to load required components: {e}")
            # Continue with available components
    
    async def _process_with_components(self, 
                                       session: FFChatSession,
                                       ff_message: FFMessageDTO,
                                       required_components: List[str],
                                       **kwargs) -> Dict[str, Any]:
        """Process message through Phase 2 components"""
        try:
            component_results = {}
            final_response_content = ""
            
            # Process through each required component in priority order
            # Text chat component handles basic conversation
            if COMPONENT_TYPE_TEXT_CHAT in required_components and COMPONENT_TYPE_TEXT_CHAT in self._loaded_components:
                text_chat_component = self._loaded_components[COMPONENT_TYPE_TEXT_CHAT]
                
                text_result = await text_chat_component.process_message(
                    session_id=session.ff_storage_session_id,
                    user_id=session.user_id,
                    message=ff_message,
                    context={
                        "use_case": session.use_case,
                        **kwargs
                    }
                )
                
                component_results[COMPONENT_TYPE_TEXT_CHAT] = text_result
                if text_result.get("success") and text_result.get("response_content"):
                    final_response_content = text_result["response_content"]
            
            # Memory component enhances with context
            if COMPONENT_TYPE_MEMORY in required_components and COMPONENT_TYPE_MEMORY in self._loaded_components:
                memory_component = self._loaded_components[COMPONENT_TYPE_MEMORY]
                
                memory_result = await memory_component.process_message(
                    session_id=session.ff_storage_session_id,
                    user_id=session.user_id,
                    message=ff_message,
                    context={
                        "use_case": session.use_case,
                        "include_memory_context": True,
                        **kwargs
                    }
                )
                
                component_results[COMPONENT_TYPE_MEMORY] = memory_result
                
                # Enhance response with memory context if available
                if memory_result.get("success") and memory_result.get("metadata", {}).get("memory_context"):
                    memory_context = memory_result["metadata"]["memory_context"]
                    if memory_context and final_response_content:
                        final_response_content = f"Based on our previous conversations: {memory_context}\n\n{final_response_content}"
            
            # Multi-agent component for collaborative responses
            if COMPONENT_TYPE_MULTI_AGENT in required_components and COMPONENT_TYPE_MULTI_AGENT in self._loaded_components:
                multi_agent_component = self._loaded_components[COMPONENT_TYPE_MULTI_AGENT]
                
                agent_result = await multi_agent_component.process_message(
                    session_id=session.ff_storage_session_id,
                    user_id=session.user_id,
                    message=ff_message,
                    context={
                        "use_case": session.use_case,
                        "agent_personas": kwargs.get("agent_personas", ["general_assistant", "specialist"]),
                        "coordination_mode": kwargs.get("coordination_mode", "collaborative"),
                        **kwargs
                    }
                )
                
                component_results[COMPONENT_TYPE_MULTI_AGENT] = agent_result
                if agent_result.get("success") and agent_result.get("response_content"):
                    final_response_content = agent_result["response_content"]
            
            # Fallback to use case manager if no components produced response
            if not final_response_content:
                logger.warning("No Phase 2 components produced response, falling back to Phase 1")
                fallback_result = await self.use_case_manager.process_message(
                    session=session,
                    message=ff_message,
                    **kwargs
                )
                
                component_results["fallback"] = fallback_result
                final_response_content = fallback_result.get("response_content", "I'm sorry, I couldn't generate a response.")
            
            # Build comprehensive result
            result = {
                "success": True,
                "response_content": final_response_content,
                "processor": "ff_chat_phase2_components",
                "components_used": list(component_results.keys()),
                "use_case": session.use_case,
                "component_results": component_results,
                "session_id": session.session_id,
                "ff_session_id": session.ff_storage_session_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing with Phase 2 components: {e}")
            # Fallback to Phase 1 processing
            return await self.use_case_manager.process_message(
                session=session,
                message=ff_message,
                **kwargs
            )
    
    
    async def get_session_messages(self, 
                                   session_id: str,
                                   limit: Optional[int] = None,
                                   offset: int = 0) -> List[FFMessageDTO]:
        """
        Get messages from a chat session using FF storage.
        
        Args:
            session_id: Chat session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of messages
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Chat session not found: {session_id}")
        
        return await self.ff_storage.get_messages(
            user_id=session.user_id,
            session_id=session.ff_storage_session_id,
            limit=limit,
            offset=offset
        )
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a chat session"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Chat session not found: {session_id}")
        
        # Get FF session info
        ff_session_info = await self.ff_storage.get_session(
            user_id=session.user_id,
            session_id=session.ff_storage_session_id
        )
        
        return {
            "chat_session_id": session.session_id,
            "ff_session_id": session.ff_storage_session_id,
            "user_id": session.user_id,
            "use_case": session.use_case,
            "active": session.active,
            "created_at": session.created_at.isoformat(),
            "context": session.context,
            "ff_session_info": ff_session_info.to_dict() if ff_session_info else None
        }
    
    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a chat session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            # Mark session as inactive
            session.active = False
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Closed FF chat session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error closing FF chat session {session_id}: {e}")
    
    def list_use_cases(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available use cases"""
        return {
            "basic_chat": {
                "description": "Basic text-based chat functionality",
                "components": ["text_chat"],
                "capabilities": ["conversation", "basic_responses"]
            },
            "memory_chat": {
                "description": "Chat with conversation memory",
                "components": ["text_chat", "memory"],
                "capabilities": ["conversation", "context_memory", "personal_history"]
            },
            "rag_chat": {
                "description": "RAG-enhanced chat with document search",
                "components": ["text_chat", "memory", "search"],
                "capabilities": ["conversation", "document_search", "knowledge_retrieval"]
            },
            "multimodal_chat": {
                "description": "Multimodal chat supporting text, images, and documents",
                "components": ["text_chat", "multimodal"],
                "capabilities": ["conversation", "image_analysis", "document_processing"]
            },
            "multi_ai_panel": {
                "description": "Multiple AI agents working together",
                "components": ["text_chat", "multi_agent"],
                "capabilities": ["conversation", "multi_agent_coordination", "specialized_responses"]
            },
            "personal_assistant": {
                "description": "Personal assistant with memory and tools",
                "components": ["text_chat", "memory", "tools"],
                "capabilities": ["conversation", "task_execution", "tool_usage", "personal_context"]
            },
            "translation_chat": {
                "description": "Multi-language translation chat",
                "components": ["text_chat", "translation"],
                "capabilities": ["conversation", "language_translation", "multilingual_support"]
            },
            "scene_critic": {
                "description": "Enhanced multimodal analysis with expert personas",
                "components": ["text_chat", "multimodal", "multi_agent"],
                "capabilities": ["conversation", "scene_analysis", "expert_critique", "multimodal_processing"]
            }
        }
    
    async def get_use_case_info(self, use_case: str) -> Dict[str, Any]:
        """Get information about a specific use case"""
        return await self.use_case_manager.get_use_case_info(use_case)
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active chat session IDs"""
        return [sid for sid, session in self.active_sessions.items() if session.active]
    
    async def search_messages(self, 
                              user_id: str,
                              query: str,
                              session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search messages across sessions using existing FF search capabilities.
        
        Args:
            user_id: User identifier
            query: Search query
            session_ids: Optional list of specific chat session IDs
            
        Returns:
            Search results
        """
        # Convert chat session IDs to FF session IDs if provided
        ff_session_ids = None
        if session_ids:
            ff_session_ids = []
            for chat_session_id in session_ids:
                session = self.active_sessions.get(chat_session_id)
                if session:
                    ff_session_ids.append(session.ff_storage_session_id)
        
        # Use existing FF search capabilities
        return await self.ff_storage.search_messages(
            user_id=user_id,
            query=query,
            session_ids=ff_session_ids
        )
    
    async def shutdown(self) -> None:
        """Shutdown the FF chat application and cleanup resources"""
        logger.info("Shutting down FF Chat Application...")
        
        try:
            # Close all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.close_session(session_id)
            
            # Shutdown Phase 2 component registry
            await self.component_registry.shutdown()
            
            # Clear loaded components
            self._loaded_components.clear()
            
            # Shutdown chat managers
            await self.chat_session_manager.shutdown()
            await self.use_case_manager.shutdown()
            
            # FF storage cleanup (if needed)
            # Note: Don't shutdown FF storage as it might be used by other systems
            
            self._initialized = False
            logger.info("FF Chat Application with Phase 2 components shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during FF Chat Application shutdown: {e}")
    
    def get_loaded_components(self) -> List[str]:
        """Get list of currently loaded components"""
        return list(self._loaded_components.keys())
    
    def get_component_registry_statistics(self) -> Dict[str, Any]:
        """Get component registry statistics"""
        return self.component_registry.get_registry_statistics()
        
    async def get_component_info(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific component"""
        return self.component_registry.get_component_info(component_name)
    
    async def get_components_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all components"""
        try:
            components_info = {}
            for component_name in self._loaded_components.keys():
                info = await self.get_component_info(component_name)
                if info:
                    components_info[component_name] = info
            return components_info
        except Exception as e:
            logger.error(f"Error getting components info: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Alias for shutdown method for compatibility"""
        await self.shutdown()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            return {
                "active_sessions": len(self.active_sessions),
                "loaded_components": len(self._loaded_components),
                "initialized": self._initialized,
                "components": await self.get_components_info()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}


# Convenience functions for common usage patterns

async def create_ff_chat_app(ff_config: Optional[FFConfigurationManagerConfigDTO] = None) -> FFChatApplication:
    """Create and initialize an FF chat application"""
    app = FFChatApplication(ff_config=ff_config)
    await app.initialize()
    return app


async def create_ff_chat_session(use_case: str = "basic_chat", 
                                 user_id: str = "default_user") -> tuple[FFChatApplication, str]:
    """Create FF chat application and session in one call"""
    app = await create_ff_chat_app()
    session_id = await app.create_chat_session(user_id, use_case)
    return app, session_id


async def ff_quick_chat(message: str, 
                        use_case: str = "basic_chat",
                        user_id: str = "default_user") -> str:
    """Quick one-off chat interaction using FF backend"""
    app, session_id = await create_ff_chat_session(use_case, user_id)
    try:
        result = await app.process_message(session_id, message)
        return result.get("response_content", "")
    finally:
        await app.close_session(session_id)
        await app.shutdown()