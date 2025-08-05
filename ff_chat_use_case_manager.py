"""
FF Chat Use Case Manager - Use Case Routing and Configuration

Manages use case definitions, routing, and processing using
existing FF storage and processing capabilities.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
import asyncio
from datetime import datetime

from ff_core_storage_manager import FFCoreStorageManager
from ff_class_configs.ff_chat_use_case_config import FFChatUseCaseConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_protocol import FFChatUseCaseProtocol
from ff_protocols.ff_chat_component_protocol import (
    get_required_components_for_use_case, get_use_cases_for_component,
    COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_MULTI_AGENT
)

logger = get_logger(__name__)


class FFChatUseCaseManager(FFChatUseCaseProtocol):
    """
    Manages use case definitions and processing using FF backend.
    
    Provides use case routing, configuration management,
    and basic processing using existing FF capabilities.
    """
    
    def __init__(self, 
                 ff_storage: FFCoreStorageManager,
                 config: FFChatUseCaseConfigDTO):
        """
        Initialize FF use case manager.
        
        Args:
            ff_storage: FF storage manager instance
            config: Use case configuration
        """
        self.ff_storage = ff_storage
        self.config = config
        self.logger = get_logger(__name__)
        
        # Use case definitions
        self.use_case_definitions: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
        # Processing statistics
        self.processing_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the use case manager"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing FF Chat Use Case Manager...")
            
            # Load use case definitions
            await self._load_use_case_definitions()
            
            # Initialize processing statistics
            self._initialize_processing_stats()
            
            self._initialized = True
            logger.info(f"FF Chat Use Case Manager initialized with {len(self.use_case_definitions)} use cases")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FF Chat Use Case Manager: {e}")
            return False
    
    async def _load_use_case_definitions(self) -> None:
        """Load use case definitions from configuration files"""
        try:
            # Try to load from YAML file
            config_file = Path(self.config.config_file_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.use_case_definitions = data.get("use_cases", {})
                logger.info(f"Loaded use case definitions from {config_file}")
            else:
                # Use default definitions
                self.use_case_definitions = self._get_default_use_case_definitions()
                logger.info("Using default use case definitions")
            
            # Validate use case definitions if enabled
            if self.config.validate_use_cases:
                await self._validate_use_case_definitions()
            
            logger.info(f"Loaded {len(self.use_case_definitions)} use case definitions")
            
        except Exception as e:
            logger.error(f"Failed to load use case definitions: {e}")
            # Fallback to defaults
            self.use_case_definitions = self._get_default_use_case_definitions()
    
    def _get_default_use_case_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get default use case definitions for Phase 1"""
        return {
            "basic_chat": {
                "description": "Simple 1:1 text conversation using FF storage",
                "category": "basic",
                "components": ["ff_text_chat"],
                "mode": "ff_storage",
                "priority": 100,
                "enabled": True,
                "settings": {
                    "max_history": 10,
                    "response_style": "conversational",
                    "timeout_seconds": 30
                },
                "requirements": {
                    "min_ff_version": "2.0.0",
                    "required_managers": ["storage"]
                }
            },
            "multimodal_chat": {
                "description": "1:1 conversation with multimedia support using FF document processing",
                "category": "basic", 
                "components": ["ff_text_chat", "ff_multimodal"],
                "mode": "ff_enhanced",
                "priority": 90,
                "enabled": True,
                "settings": {
                    "supported_formats": ["text", "image", "pdf", "audio"],
                    "max_file_size": "50MB",
                    "max_attachments": 5,
                    "timeout_seconds": 60
                },
                "requirements": {
                    "min_ff_version": "2.0.0",
                    "required_managers": ["storage", "document_processing"]
                }
            },
            "rag_chat": {
                "description": "1:1 conversation with knowledge injection using FF vector search",
                "category": "basic",
                "components": ["ff_text_chat", "ff_rag"],
                "mode": "ff_enhanced",
                "priority": 85,
                "enabled": True,
                "settings": {
                    "max_context_docs": 5,
                    "similarity_threshold": 0.7,
                    "search_timeout_seconds": 10,
                    "max_context_length": 2000
                },
                "requirements": {
                    "min_ff_version": "2.0.0",
                    "required_managers": ["storage", "search", "vector_storage"]
                }
            },
            "memory_chat": {
                "description": "Long-term memory across sessions using FF storage",
                "category": "context_memory",
                "components": ["ff_text_chat", "ff_memory"],
                "mode": "ff_enhanced",
                "priority": 80,
                "enabled": True,
                "settings": {
                    "memory_type": "episodic",
                    "retention_days": 30,
                    "max_memories": 100,
                    "memory_similarity_threshold": 0.6
                },
                "requirements": {
                    "min_ff_version": "2.0.0",
                    "required_managers": ["storage", "vector_storage"]
                }
            }
        }
    
    async def _validate_use_case_definitions(self) -> None:
        """Validate use case definitions"""
        for use_case_id, definition in self.use_case_definitions.items():
            try:
                # Validate required fields
                required_fields = ["description", "category", "components", "mode"]
                for field in required_fields:
                    if field not in definition:
                        raise ValueError(f"Missing required field '{field}' in use case '{use_case_id}'")
                
                # Validate mode
                valid_modes = ["ff_storage", "ff_enhanced", "ff_full"]
                if definition.get("mode") not in valid_modes:
                    raise ValueError(f"Invalid mode '{definition.get('mode')}' in use case '{use_case_id}'")
                
                # Validate components list
                if not isinstance(definition.get("components"), list):
                    raise ValueError(f"Components must be a list in use case '{use_case_id}'")
                
                logger.debug(f"Validated use case definition: {use_case_id}")
                
            except Exception as e:
                logger.error(f"Invalid use case definition '{use_case_id}': {e}")
                if not self.config.enable_fallback_processing:
                    raise
    
    def _initialize_processing_stats(self) -> None:
        """Initialize processing statistics tracking"""
        for use_case_id in self.use_case_definitions.keys():
            self.processing_stats[use_case_id] = {
                "total_processed": 0,
                "successful_processed": 0,
                "failed_processed": 0,
                "average_processing_time": 0.0,
                "last_processed": None
            }
    
    async def is_use_case_supported(self, use_case: str) -> bool:
        """
        Check if a use case is supported.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if supported
        """
        definition = self.use_case_definitions.get(use_case)
        return definition is not None and definition.get("enabled", True)
    
    def list_use_cases(self) -> List[str]:
        """
        Get list of available use cases.
        
        Returns:
            List of use case identifiers
        """
        return [
            use_case_id for use_case_id, definition in self.use_case_definitions.items()
            if definition.get("enabled", True)
        ]
    
    def list_all_use_cases(self) -> List[str]:
        """
        Get list of all use cases (including disabled ones).
        
        Returns:
            List of all use case identifiers
        """
        return list(self.use_case_definitions.keys())
    
    async def get_use_case_config(self, use_case: str) -> Dict[str, Any]:
        """
        Get configuration for a specific use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Use case configuration
        """
        if use_case not in self.use_case_definitions:
            raise ValueError(f"Unknown use case: {use_case}")
        return self.use_case_definitions[use_case].copy()
    
    async def get_use_case_info(self, use_case: str) -> Dict[str, Any]:
        """
        Get detailed information about a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Use case information
        """
        config = await self.get_use_case_config(use_case)
        stats = self.processing_stats.get(use_case, {})
        
        return {
            "use_case": use_case,
            "description": config.get("description", ""),
            "category": config.get("category", ""),
            "components": config.get("components", []),
            "mode": config.get("mode", "ff_storage"),
            "priority": config.get("priority", 100),
            "enabled": config.get("enabled", True),
            "settings": config.get("settings", {}),
            "requirements": config.get("requirements", {}),
            "statistics": stats
        }
    
    def get_use_cases_by_category(self, category: str) -> List[str]:
        """
        Get use cases in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of use case identifiers in the category
        """
        return [
            use_case for use_case, config in self.use_case_definitions.items()
            if config.get("category") == category and config.get("enabled", True)
        ]
    
    def get_use_cases_by_mode(self, mode: str) -> List[str]:
        """
        Get use cases by processing mode.
        
        Args:
            mode: Processing mode
            
        Returns:
            List of use case identifiers using the mode
        """
        return [
            use_case for use_case, config in self.use_case_definitions.items()
            if config.get("mode") == mode and config.get("enabled", True)
        ]
    
    def get_required_components_for_use_case(self, use_case: str) -> List[str]:
        """
        Get required Phase 2 components for a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            List of required component types
        """
        return get_required_components_for_use_case(use_case)
    
    def get_use_cases_for_component(self, component_type: str) -> List[str]:
        """
        Get use cases that require a specific component.
        
        Args:
            component_type: Component type identifier
            
        Returns:
            List of use case identifiers
        """
        return get_use_cases_for_component(component_type)
    
    def is_phase2_use_case(self, use_case: str) -> bool:
        """
        Check if use case requires Phase 2 components.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if use case requires Phase 2 components
        """
        required_components = self.get_required_components_for_use_case(use_case)
        
        # Phase 2 use cases require components beyond basic text chat
        phase2_components = [COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_MULTI_AGENT]
        return any(component in required_components for component in phase2_components)
    
    def get_component_routing_info(self, use_case: str) -> Dict[str, Any]:
        """
        Get component routing information for a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Component routing information
        """
        required_components = self.get_required_components_for_use_case(use_case)
        use_case_config = self.use_case_definitions.get(use_case, {})
        
        return {
            "use_case": use_case,
            "required_components": required_components,
            "processing_mode": use_case_config.get("mode", "ff_storage"),
            "is_phase2": self.is_phase2_use_case(use_case),
            "priority": use_case_config.get("priority", 100),
            "component_coordination": {
                "primary_component": required_components[0] if required_components else COMPONENT_TYPE_TEXT_CHAT,
                "supporting_components": required_components[1:] if len(required_components) > 1 else [],
                "coordination_strategy": self._determine_coordination_strategy(required_components)
            }
        }
    
    def _determine_coordination_strategy(self, components: List[str]) -> str:
        """Determine how components should be coordinated"""
        if len(components) <= 1:
            return "single"
        elif COMPONENT_TYPE_MULTI_AGENT in components:
            return "agent_driven"
        elif COMPONENT_TYPE_MEMORY in components:
            return "memory_enhanced"
        else:
            return "sequential"
    
    async def process_message(self, 
                              session,  # FFChatSession
                              message: FFMessageDTO,
                              **kwargs) -> Dict[str, Any]:
        """
        Process a message according to use case configuration using Phase 2 component routing.
        
        Args:
            session: FF chat session
            message: FF message DTO
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results
        """
        import time
        start_time = time.time()
        
        try:
            if not await self.is_use_case_supported(session.use_case):
                raise ValueError(f"Use case '{session.use_case}' is not supported or disabled")
            
            use_case_config = await self.get_use_case_config(session.use_case)
            
            # Check if this is a Phase 2 use case requiring components
            if self.is_phase2_use_case(session.use_case):
                # Route to Phase 2 component processing
                result = await self._process_with_phase2_components(session, message, use_case_config, **kwargs)
            else:
                # Use Phase 1 processing for basic use cases
                processing_mode = use_case_config.get("mode", "ff_storage")
                
                if processing_mode == "ff_storage":
                    result = await self._process_with_ff_storage(session, message, use_case_config, **kwargs)
                elif processing_mode == "ff_enhanced":
                    result = await self._process_with_ff_enhanced(session, message, use_case_config, **kwargs)
                else:
                    # Fallback to basic processing
                    result = await self._process_with_ff_storage(session, message, use_case_config, **kwargs)
            
            # Update statistics
            processing_time = time.time() - start_time
            await self._update_processing_stats(session.use_case, True, processing_time)
            
            return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_processing_stats(session.use_case, False, processing_time)
            
            logger.error(f"Error processing message for use case {session.use_case}: {e}")
            
            if self.config.enable_fallback_processing:
                # Try fallback processing
                try:
                    return await self._process_with_fallback(session, message, **kwargs)
                except Exception as fallback_e:
                    logger.error(f"Fallback processing also failed: {fallback_e}")
            
            return {
                "success": False,
                "error": str(e),
                "response_content": "I apologize, but I encountered an error processing your message."
            }
    
    async def _process_with_phase2_components(self, 
                                              session,
                                              message: FFMessageDTO,
                                              use_case_config: Dict[str, Any],
                                              **kwargs) -> Dict[str, Any]:
        """
        Process message using Phase 2 components via FF Chat Application.
        
        Args:
            session: FF chat session
            message: FF message DTO
            use_case_config: Use case configuration
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results from Phase 2 components
        """
        try:
            # Get component routing information
            routing_info = self.get_component_routing_info(session.use_case)
            required_components = routing_info["required_components"]
            coordination_strategy = routing_info["component_coordination"]["coordination_strategy"]
            
            logger.debug(f"Processing {session.use_case} with Phase 2 components: {required_components}")
            
            result = {
                "success": True,
                "response_content": f"Phase 2 processing for {session.use_case} would use components: {', '.join(required_components)}",
                "processor": "ff_use_case_manager_phase2",
                "use_case": session.use_case,
                "processing_mode": "phase2_components",
                "routing_info": routing_info,
                "required_components": required_components,
                "coordination_strategy": coordination_strategy,
                "component_delegation_note": "This processing is delegated to FF Chat Application's _process_with_components method"
            }
            
            # Add use case specific enhancements
            if session.use_case == "memory_chat":
                result["response_content"] += " [Memory context would be retrieved and applied]"
                result["memory_enhanced"] = True
                
            elif session.use_case in ["multi_ai_panel", "agent_debate", "consensus_building"]:
                result["response_content"] += " [Multi-agent coordination would be applied]"
                result["multi_agent_coordination"] = coordination_strategy
                
            elif session.use_case in ["smart_search_chat", "semantic_search_expert"]:
                result["response_content"] += " [Enhanced search and memory retrieval would be applied]"
                result["enhanced_search"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Phase 2 component processing: {e}")
            # Fallback to Phase 1 processing
            return await self._process_with_ff_enhanced(session, message, use_case_config, **kwargs)
    
    async def _process_with_ff_storage(self, 
                                       session, 
                                       message: FFMessageDTO,
                                       use_case_config: Dict[str, Any],
                                       **kwargs) -> Dict[str, Any]:
        """Process using basic FF storage capabilities"""
        try:
            # Get recent messages for context
            max_history = use_case_config.get("settings", {}).get("max_history", 5)
            recent_messages = await self.ff_storage.get_messages(
                user_id=session.user_id,
                session_id=session.ff_storage_session_id,
                limit=max_history
            )
            
            # Generate basic response (to be replaced with LLM integration in Phase 2)
            response_content = f"FF Chat ({session.use_case}) response to: {message.content}"
            
            return {
                "success": True,
                "response_content": response_content,
                "processor": "ff_storage_basic",
                "use_case": session.use_case,
                "message_count": len(recent_messages) + 1,
                "context_used": len(recent_messages),
                "processing_mode": "ff_storage"
            }
            
        except Exception as e:
            logger.error(f"Error in FF storage processing: {e}")
            raise
    
    async def _process_with_ff_enhanced(self, 
                                        session, 
                                        message: FFMessageDTO,
                                        use_case_config: Dict[str, Any],
                                        **kwargs) -> Dict[str, Any]:
        """Process using enhanced FF capabilities"""
        try:
            # Get base processing result
            result = await self._process_with_ff_storage(session, message, use_case_config, **kwargs)
            
            # Add enhanced processing based on use case
            if session.use_case == "multimodal_chat" and message.attachments:
                result["response_content"] += f" [Processed {len(message.attachments)} attachments]"
                result["attachments_processed"] = len(message.attachments)
            
            elif session.use_case == "rag_chat":
                # Simulate RAG context retrieval (will be implemented in Phase 2)
                result["response_content"] += " [RAG context would be retrieved here]"
                result["rag_context_docs"] = use_case_config.get("settings", {}).get("max_context_docs", 5)
            
            elif session.use_case == "memory_chat":
                # Simulate memory retrieval (will be implemented in Phase 2)
                result["response_content"] += " [Memory context would be retrieved here]"
                result["memory_context"] = "episodic"
            
            result["processor"] = "ff_enhanced_basic"
            result["processing_mode"] = "ff_enhanced"
            return result
            
        except Exception as e:
            logger.error(f"Error in FF enhanced processing: {e}")
            raise
    
    async def _process_with_fallback(self, 
                                     session,
                                     message: FFMessageDTO,
                                     **kwargs) -> Dict[str, Any]:
        """Fallback processing when main processing fails"""
        try:
            logger.info(f"Using fallback processing for session {session.session_id}")
            
            return {
                "success": True,
                "response_content": f"Fallback response to: {message.content}",
                "processor": "ff_fallback",
                "use_case": session.use_case,
                "processing_mode": "fallback",
                "note": "This response was generated using fallback processing"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            raise
    
    async def _update_processing_stats(self, use_case: str, success: bool, processing_time: float) -> None:
        """Update processing statistics for a use case"""
        if use_case not in self.processing_stats:
            self.processing_stats[use_case] = {
                "total_processed": 0,
                "successful_processed": 0,
                "failed_processed": 0,
                "average_processing_time": 0.0,
                "last_processed": None
            }
        
        stats = self.processing_stats[use_case]
        stats["total_processed"] += 1
        stats["last_processed"] = datetime.now().isoformat()
        
        if success:
            stats["successful_processed"] += 1
        else:
            stats["failed_processed"] += 1
        
        # Update average processing time
        current_avg = stats["average_processing_time"]
        total_count = stats["total_processed"]
        stats["average_processing_time"] = ((current_avg * (total_count - 1)) + processing_time) / total_count
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get overall processing statistics.
        
        Returns:
            Processing statistics for all use cases
        """
        return {
            "use_case_stats": self.processing_stats.copy(),
            "total_use_cases": len(self.use_case_definitions),
            "enabled_use_cases": len(self.list_use_cases()),
            "disabled_use_cases": len(self.use_case_definitions) - len(self.list_use_cases())
        }
    
    async def reload_use_case_definitions(self) -> bool:
        """
        Reload use case definitions from configuration file.
        
        Returns:
            True if reload successful
        """
        try:
            logger.info("Reloading use case definitions...")
            await self._load_use_case_definitions()
            self._initialize_processing_stats()
            logger.info("Use case definitions reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload use case definitions: {e}")
            return False
    
    async def enable_use_case(self, use_case: str) -> bool:
        """
        Enable a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if enabled successfully
        """
        if use_case not in self.use_case_definitions:
            return False
        
        self.use_case_definitions[use_case]["enabled"] = True
        logger.info(f"Enabled use case: {use_case}")
        return True
    
    async def disable_use_case(self, use_case: str) -> bool:
        """
        Disable a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if disabled successfully
        """
        if use_case not in self.use_case_definitions:
            return False
        
        self.use_case_definitions[use_case]["enabled"] = False
        logger.info(f"Disabled use case: {use_case}")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the use case manager"""
        logger.info("Shutting down FF Chat Use Case Manager...")
        
        # Save processing statistics if needed
        # (This could be extended to persist stats to FF storage)
        
        self.use_case_definitions.clear()
        self.processing_stats.clear()
        self._initialized = False
        logger.info("FF Chat Use Case Manager shutdown complete")