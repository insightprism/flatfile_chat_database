"""
FF Text Chat Component - Phase 2 Implementation

Handles text-based chat interactions using existing FF storage manager
as backend service. Supports 17/22 use cases (77% coverage).
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import existing FF infrastructure
from ff_core_storage_manager import FFCoreStorageManager
from ff_search_manager import FFSearchManager
from ff_class_configs.ff_text_chat_config import FFTextChatConfigDTO, FFTextChatUseCaseConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO
from ff_protocols.ff_chat_component_protocol import (
    FFTextChatComponentProtocol, FFComponentInfo, FFComponentCapability,
    COMPONENT_TYPE_TEXT_CHAT, get_use_cases_for_component
)
from ff_utils.ff_logging import get_logger


class FFTextChatComponent(FFTextChatComponentProtocol):
    """
    FF Text Chat Component implementing text conversation capabilities.
    
    Uses existing FF storage manager for message persistence and supports
    17/22 use cases requiring text chat functionality.
    """
    
    def __init__(self, config: FFTextChatConfigDTO):
        """
        Initialize FF Text Chat Component.
        
        Args:
            config: Text chat configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services (initialized via dependencies)
        self.ff_storage: Optional[FFCoreStorageManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        
        # Component state
        self._initialized = False
        self._component_info = self._create_component_info()
        
        # Processing cache (if enabled)
        self._response_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Context management
        self._conversation_contexts: Dict[str, List[FFMessageDTO]] = {}
        self._context_last_access: Dict[str, float] = {}
        
        # Processing statistics
        self._processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Get component metadata and capabilities"""
        return self._component_info.to_dict()
    
    def _create_component_info(self) -> FFComponentInfo:
        """Create component information structure"""
        capabilities = [
            FFComponentCapability(
                name="text_conversation",
                description="Process text-based conversations with context awareness",
                parameters={
                    "max_message_length": self.config.max_message_length,
                    "context_window": self.config.context_window,
                    "response_formats": ["text", "markdown"]
                },
                ff_dependencies=["ff_storage"]
            ),
            FFComponentCapability(
                name="context_retrieval",
                description="Retrieve conversation context from FF storage",
                parameters={
                    "max_context_length": self.config.max_context_length,
                    "search_enabled": self.config.enable_message_search
                },
                ff_dependencies=["ff_storage", "ff_search"]
            ),
            FFComponentCapability(
                name="response_generation",
                description="Generate contextual responses using conversation history",
                parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "response_style": self.config.response_style
                },
                ff_dependencies=[]
            )
        ]
        
        supported_use_cases = get_use_cases_for_component(COMPONENT_TYPE_TEXT_CHAT)
        
        return FFComponentInfo(
            name="ff_text_chat",
            version="2.0.0",
            description="FF Text Chat Component for conversation processing using FF storage backend",
            capabilities=capabilities,
            use_cases=supported_use_cases,
            ff_dependencies=["ff_storage", "ff_search"],
            priority=100
        )
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing FF Text Chat Component...")
            
            # Extract FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_search = dependencies.get("ff_search")
            
            # Validate required dependencies
            if not self.ff_storage:
                raise ValueError("ff_storage dependency is required")
            
            # Test FF storage connection
            if not await self._test_ff_storage_connection():
                raise RuntimeError("Failed to connect to FF storage backend")
            
            # Initialize response cache if enabled
            if self.config.cache_responses:
                await self._initialize_response_cache()
            
            # Initialize context management
            await self._initialize_context_management()
            
            self._initialized = True
            self.logger.info("FF Text Chat Component initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Text Chat Component: {e}")
            return False
    
    async def _test_ff_storage_connection(self) -> bool:
        """Test connection to FF storage backend"""
        try:
            # Test basic FF storage functionality
            test_user_id = "ff_text_chat_test_user"
            test_session_name = "FF Text Chat Component Test"
            
            # This should work with existing FF storage methods
            session_id = await self.ff_storage.create_session(test_user_id, test_session_name)
            if session_id:
                self.logger.debug("FF storage connection test successful")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"FF storage connection test failed: {e}")
            return False
    
    async def _initialize_response_cache(self) -> None:
        """Initialize response cache if enabled"""
        self._response_cache = {}
        self._cache_timestamps = {}
        self.logger.debug("Response cache initialized")
    
    async def _initialize_context_management(self) -> None:
        """Initialize conversation context management"""
        self._conversation_contexts = {}
        self._context_last_access = {}
        
        # Start background context cleanup task
        if self.config.enable_conversation_summary:
            asyncio.create_task(self._context_cleanup_task())
        
        self.logger.debug("Context management initialized")
    
    async def _context_cleanup_task(self) -> None:
        """Background task to clean up old conversation contexts"""
        while self._initialized:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, last_access in self._context_last_access.items():
                    if current_time - last_access > 3600:  # 1 hour timeout
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    self._conversation_contexts.pop(session_id, None)
                    self._context_last_access.pop(session_id, None)
                
                if expired_sessions:
                    self.logger.debug(f"Cleaned up {len(expired_sessions)} expired conversation contexts")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in context cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
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
            Processing results dictionary
        """
        if not self._initialized:
            return {
                "success": False,
                "error": "Component not initialized",
                "component": "ff_text_chat"
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.debug(f"Processing message in session {session_id} for user {user_id}")
            
            # Validate message
            validation_result = await self._validate_message(message)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Message validation failed: {validation_result['error']}",
                    "component": "ff_text_chat"
                }
            
            # Check response cache if enabled
            if self.config.cache_responses:
                cached_response = await self._check_response_cache(message.content, session_id)
                if cached_response:
                    self._processing_stats["cache_hits"] += 1
                    return cached_response
                else:
                    self._processing_stats["cache_misses"] += 1
            
            # Store user message in FF storage
            await self._store_message_in_ff_storage(session_id, user_id, message)
            
            # Get conversation context
            conversation_context = await self.get_conversation_context(user_id, session_id, self.config.context_window)
            
            # Generate response using FF backend
            response_content = await self.generate_response(
                conversation_context=conversation_context,
                current_message=message,
                response_context=context
            )
            
            # Create assistant response message
            assistant_message = FFMessageDTO(
                role="assistant",
                content=response_content,
                timestamp=datetime.now().isoformat(),
                message_id=f"ff_text_chat_resp_{int(time.time() * 1000)}"
            )
            
            # Store assistant response in FF storage
            await self._store_message_in_ff_storage(session_id, user_id, assistant_message)
            
            # Update processing statistics
            processing_time = time.time() - start_time
            await self._update_processing_stats(True, processing_time)
            
            # Cache response if enabled
            if self.config.cache_responses:
                await self._cache_response(message.content, session_id, response_content)
            
            result = {
                "success": True,
                "response_content": response_content,
                "component": "ff_text_chat",
                "processor": "ff_storage_backend",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "message_id": assistant_message.message_id,
                    "processing_time": processing_time,
                    "cache_hit": False,
                    "context_messages_used": len(conversation_context),
                    "response_format": self.config.response_format
                }
            }
            
            self.logger.debug(f"Successfully processed message in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_processing_stats(False, processing_time)
            
            self.logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_text_chat",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": processing_time
                }
            }
    
    async def _validate_message(self, message: FFMessageDTO) -> Dict[str, Any]:
        """Validate message according to configuration"""
        try:
            # Check message length
            if len(message.content) > self.config.max_message_length:
                return {
                    "valid": False,
                    "error": f"Message too long: {len(message.content)} > {self.config.max_message_length}"
                }
            
            if len(message.content) < self.config.min_message_length:
                return {
                    "valid": False,
                    "error": f"Message too short: {len(message.content)} < {self.config.min_message_length}"
                }
            
            # Check message type
            message_type = getattr(message, 'message_type', 'text')
            if message_type not in self.config.allowed_message_types:
                return {
                    "valid": False,
                    "error": f"Message type '{message_type}' not allowed"
                }
            
            # Content filtering if enabled
            if self.config.enable_content_filtering:
                content_check = await self._check_content_filter(message.content)
                if not content_check["allowed"]:
                    return {
                        "valid": False,
                        "error": f"Content filtered: {content_check['reason']}"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    async def _check_content_filter(self, content: str) -> Dict[str, Any]:
        """Basic content filtering (placeholder for more sophisticated filtering)"""
        # This is a placeholder - in a real implementation you might integrate
        # with content moderation services or implement custom filtering
        
        # Basic checks for demonstration
        prohibited_terms = ["spam", "inappropriate"]  # Very basic example
        
        for term in prohibited_terms:
            if term.lower() in content.lower():
                return {
                    "allowed": False,
                    "reason": f"Contains prohibited term: {term}"
                }
        
        return {"allowed": True}
    
    async def _store_message_in_ff_storage(self, session_id: str, user_id: str, message: FFMessageDTO) -> bool:
        """Store message using FF storage backend"""
        try:
            # Use existing FF storage methods to persist message
            message_id = await self.ff_storage.add_message(
                user_id=user_id,
                session_id=session_id,
                role=message.role,
                content=message.content,
                metadata=getattr(message, 'metadata', {})
            )
            
            if message_id:
                self.logger.debug(f"Stored message {message_id} in FF storage")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to store message in FF storage: {e}")
            return False
    
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
        try:
            # Check local context cache first
            cache_key = f"{user_id}:{session_id}"
            if cache_key in self._conversation_contexts:
                self._context_last_access[cache_key] = time.time()
                cached_context = self._conversation_contexts[cache_key]
                if len(cached_context) <= limit:
                    return cached_context
                else:
                    return cached_context[-limit:]  # Return most recent messages
            
            # Retrieve from FF storage
            messages = await self.ff_storage.get_messages(
                user_id=user_id,
                session_id=session_id,
                limit=limit
            )
            
            # Convert to FFMessageDTO format
            context_messages = []
            for msg in messages:
                context_msg = FFMessageDTO(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    message_id=getattr(msg, 'message_id', f"msg_{int(time.time() * 1000)}")
                )
                context_messages.append(context_msg)
            
            # Cache context
            self._conversation_contexts[cache_key] = context_messages
            self._context_last_access[cache_key] = time.time()
            
            return context_messages
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return []
    
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
        try:
            # This is a placeholder for LLM integration in Phase 2
            # In a full implementation, this would integrate with an LLM service
            
            response_context = response_context or {}
            
            # Build context string from conversation history
            context_str = ""
            for msg in conversation_context[-5:]:  # Use last 5 messages for context
                context_str += f"{msg.role}: {msg.content}\n"
            
            # Get use case specific customization
            use_case = response_context.get("use_case", "basic_chat")
            response_style = response_context.get("response_style", self.config.response_style)
            
            # Generate response based on style and context
            if response_style == "conversational":
                response = await self._generate_conversational_response(current_message.content, context_str, use_case)
            elif response_style == "formal":
                response = await self._generate_formal_response(current_message.content, context_str, use_case)
            elif response_style == "technical":
                response = await self._generate_technical_response(current_message.content, context_str, use_case)
            else:
                response = await self._generate_default_response(current_message.content, context_str, use_case)
            
            # Apply response formatting
            if self.config.response_format == "markdown":
                response = self._format_as_markdown(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error processing your message. Error: {str(e)}"
    
    async def _generate_conversational_response(self, message: str, context: str, use_case: str) -> str:
        """Generate conversational style response (placeholder)"""
        # This is a Phase 2 placeholder - would integrate with LLM in full implementation
        return f"Thank you for your message: '{message}'. I understand you're using the {use_case} mode. How can I help you further?"
    
    async def _generate_formal_response(self, message: str, context: str, use_case: str) -> str:
        """Generate formal style response (placeholder)"""
        return f"I have received your inquiry regarding '{message}' in the {use_case} context. I shall provide assistance accordingly."
    
    async def _generate_technical_response(self, message: str, context: str, use_case: str) -> str:
        """Generate technical style response (placeholder)"""
        return f"Processing query: '{message}' | Use case: {use_case} | Response generated using FF text chat component with FF storage backend."
    
    async def _generate_default_response(self, message: str, context: str, use_case: str) -> str:
        """Generate default response (placeholder)"""
        return f"I received your message about '{message}'. This is a response from the FF text chat component in {use_case} mode."
    
    def _format_as_markdown(self, response: str) -> str:
        """Format response as markdown"""
        # Basic markdown formatting
        return response
    
    async def _check_response_cache(self, message_content: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if response is cached"""
        if not self.config.cache_responses:
            return None
        
        cache_key = f"{session_id}:{hash(message_content)}"
        
        if cache_key in self._response_cache:
            cache_timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_timestamp < self.config.cache_ttl_seconds:
                cached_response = self._response_cache[cache_key]
                cached_response["metadata"]["cache_hit"] = True
                return cached_response
            else:
                # Cache expired
                self._response_cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)
        
        return None
    
    async def _cache_response(self, message_content: str, session_id: str, response_content: str) -> None:
        """Cache response for future use"""
        if not self.config.cache_responses:
            return
        
        cache_key = f"{session_id}:{hash(message_content)}"
        
        cached_response = {
            "success": True,
            "response_content": response_content,
            "component": "ff_text_chat",
            "processor": "ff_storage_backend",
            "metadata": {
                "cache_hit": True,
                "cached_at": datetime.now().isoformat()
            }
        }
        
        self._response_cache[cache_key] = cached_response
        self._cache_timestamps[cache_key] = time.time()
    
    async def _update_processing_stats(self, success: bool, processing_time: float) -> None:
        """Update processing statistics"""
        self._processing_stats["total_processed"] += 1
        
        if success:
            self._processing_stats["successful_processed"] += 1
        else:
            self._processing_stats["failed_processed"] += 1
        
        # Update average processing time
        current_avg = self._processing_stats["average_response_time"]
        total_count = self._processing_stats["total_processed"]
        self._processing_stats["average_response_time"] = ((current_avg * (total_count - 1)) + processing_time) / total_count
    
    async def get_capabilities(self) -> List[str]:
        """Get list of component capabilities"""
        return [cap.name for cap in self._component_info.capabilities]
    
    async def supports_use_case(self, use_case: str) -> bool:
        """Check if component supports a specific use case"""
        return use_case in self._component_info.use_cases
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self._processing_stats.copy()
    
    async def cleanup(self) -> None:
        """Cleanup component resources following FF patterns"""
        try:
            self.logger.info("Cleaning up FF Text Chat Component...")
            
            # Clear caches
            self._response_cache.clear()
            self._cache_timestamps.clear()
            self._conversation_contexts.clear()
            self._context_last_access.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("FF Text Chat Component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during FF Text Chat Component cleanup: {e}")
            raise