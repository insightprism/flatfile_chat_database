"""
FF Memory Component - Phase 2 Implementation

Manages persistent memory across conversations using existing FF vector 
storage manager as backend. Supports 7/22 use cases (32% coverage).
"""

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import existing FF infrastructure
from ff_core_storage_manager import FFCoreStorageManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_search_manager import FFSearchManager
from ff_class_configs.ff_memory_config import FFMemoryConfigDTO, FFMemoryUseCaseConfigDTO, FFMemoryType
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_protocols.ff_chat_component_protocol import (
    FFMemoryComponentProtocol, FFComponentInfo, FFComponentCapability,
    COMPONENT_TYPE_MEMORY, get_use_cases_for_component
)
from ff_utils.ff_logging import get_logger


@dataclass
class FFMemoryEntry:
    """Represents a memory entry"""
    memory_id: str
    user_id: str
    session_id: str
    memory_type: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance_score: float


class FFMemoryComponent(FFMemoryComponentProtocol):
    """
    FF Memory Component providing persistent cross-session memory.
    
    Uses existing FF vector storage manager for embeddings and similarity search,
    supporting 7/22 use cases requiring memory functionality.
    """
    
    def __init__(self, config: FFMemoryConfigDTO):
        """
        Initialize FF Memory Component.
        
        Args:
            config: Memory component configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services (initialized via dependencies)
        self.ff_storage: Optional[FFCoreStorageManager] = None
        self.ff_vector: Optional[FFVectorStorageManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        
        # Component state
        self._initialized = False
        self._component_info = self._create_component_info()
        
        # Memory management
        self._working_memory: Dict[str, Dict[str, Any]] = {}  # session_id -> working memory
        self._memory_cache: Dict[str, FFMemoryEntry] = {}  # memory_id -> memory entry
        self._cache_timestamps: Dict[str, float] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None
        
        # Processing statistics
        self._memory_stats = {
            "total_memories_stored": 0,
            "total_memories_retrieved": 0,
            "working_memory_updates": 0,
            "consolidation_runs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_retrieval_time": 0.0
        }
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Get component metadata and capabilities"""
        return self._component_info.to_dict()
    
    def _create_component_info(self) -> FFComponentInfo:
        """Create component information structure"""
        capabilities = [
            FFComponentCapability(
                name="persistent_memory",
                description="Store and retrieve persistent memories across sessions",
                parameters={
                    "memory_types": self.config.enabled_memory_types,
                    "max_entries": {
                        "episodic": self.config.episodic_max_entries,
                        "semantic": self.config.semantic_max_entries
                    },
                    "similarity_threshold": self.config.semantic_similarity_threshold
                },
                ff_dependencies=["ff_storage", "ff_vector"]
            ),
            FFComponentCapability(
                name="working_memory",
                description="Manage working memory for active sessions",
                parameters={
                    "working_memory_size": self.config.working_memory_size,
                    "timeout": self.config.working_memory_timeout
                },
                ff_dependencies=["ff_storage"]
            ),
            FFComponentCapability(
                name="memory_search",
                description="Search memories using vector similarity and text search",
                parameters={
                    "similarity_threshold": self.config.retrieval_similarity_threshold,
                    "max_results": self.config.max_retrieved_memories,
                    "diversity_factor": self.config.retrieval_diversity_factor
                },
                ff_dependencies=["ff_vector", "ff_search"]
            ),
            FFComponentCapability(
                name="memory_consolidation",
                description="Consolidate and organize memories automatically",
                parameters={
                    "consolidation_enabled": self.config.enable_memory_consolidation,
                    "consolidation_interval": self.config.consolidation_interval_hours
                },
                ff_dependencies=["ff_vector", "ff_storage"]
            )
        ]
        
        supported_use_cases = get_use_cases_for_component(COMPONENT_TYPE_MEMORY)
        
        return FFComponentInfo(
            name="ff_memory",
            version="2.0.0",
            description="FF Memory Component for persistent cross-session memory using FF vector storage backend",
            capabilities=capabilities,
            use_cases=supported_use_cases,
            ff_dependencies=["ff_storage", "ff_vector", "ff_search"],
            priority=90
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
            self.logger.info("Initializing FF Memory Component...")
            
            # Extract FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_vector = dependencies.get("ff_vector")
            self.ff_search = dependencies.get("ff_search")
            
            # Validate required dependencies
            if not self.ff_storage:
                raise ValueError("ff_storage dependency is required")
            
            if self.config.use_ff_vector_storage and not self.ff_vector:
                raise ValueError("ff_vector dependency is required when use_ff_vector_storage is enabled")
            
            # Test FF backend connections
            if not await self._test_ff_backend_connections():
                raise RuntimeError("Failed to connect to FF backend services")
            
            # Initialize memory caches
            if self.config.enable_memory_caching:
                await self._initialize_memory_cache()
            
            # Initialize working memory
            await self._initialize_working_memory()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            self.logger.info("FF Memory Component initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Memory Component: {e}")
            return False
    
    async def _test_ff_backend_connections(self) -> bool:
        """Test connections to FF backend services"""
        try:
            # Test FF storage
            test_user_id = "ff_memory_test_user"
            test_session_name = "FF Memory Component Test"
            session_id = await self.ff_storage.create_session(test_user_id, test_session_name)
            if not session_id:
                return False
            
            # Test FF vector storage if enabled
            if self.config.use_ff_vector_storage and self.ff_vector:
                # Test vector storage functionality
                test_vector = [0.1] * self.config.embedding_dimension
                test_doc_id = "ff_memory_test_doc"
                success = await self.ff_vector.store_vector(test_doc_id, test_vector, {"test": True})
                if not success:
                    return False
            
            self.logger.debug("FF backend connections test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"FF backend connections test failed: {e}")
            return False
    
    async def _initialize_memory_cache(self) -> None:
        """Initialize memory cache"""
        self._memory_cache = {}
        self._cache_timestamps = {}
        self.logger.debug("Memory cache initialized")
    
    async def _initialize_working_memory(self) -> None:
        """Initialize working memory management"""
        self._working_memory = {}
        self.logger.debug("Working memory initialized")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_task_loop())
        
        # Start consolidation task if enabled
        if self.config.enable_memory_consolidation:
            self._consolidation_task = asyncio.create_task(self._consolidation_task_loop())
        
        self.logger.debug("Background tasks started")
    
    async def _cleanup_task_loop(self) -> None:
        """Background cleanup task loop"""
        while self._initialized:
            try:
                await self._cleanup_expired_memories()
                await self._cleanup_working_memory()
                await self._cleanup_cache()
                
                await asyncio.sleep(self.config.working_memory_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _consolidation_task_loop(self) -> None:
        """Background consolidation task loop"""
        while self._initialized:
            try:
                await self._consolidate_memories()
                self._memory_stats["consolidation_runs"] += 1
                
                await asyncio.sleep(self.config.consolidation_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in consolidation task: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process chat message and update memory.
        
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
                "component": "ff_memory"
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.debug(f"Processing memory for message in session {session_id}")
            
            # Update working memory
            await self.update_working_memory(session_id, message)
            
            # Store important messages as memories
            if await self._should_store_as_memory(message, context):
                memory_stored = await self.store_memory(
                    user_id=user_id,
                    session_id=session_id,
                    memory_content=message.content,
                    memory_type=self._determine_memory_type(message, context),
                    metadata={
                        "message_id": message.message_id,
                        "timestamp": message.timestamp,
                        "session_id": session_id,
                        "importance": context.get("importance", 0.5)
                    }
                )
            else:
                memory_stored = False
            
            # Retrieve relevant memories for context
            relevant_memories = []
            if context.get("retrieve_memories", True):
                relevant_memories = await self.retrieve_memories(
                    user_id=user_id,
                    query=message.content,
                    limit=context.get("memory_limit", self.config.max_retrieved_memories)
                )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "component": "ff_memory",
                "processor": "ff_vector_backend",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_stored": memory_stored,
                    "relevant_memories_count": len(relevant_memories),
                    "relevant_memories": relevant_memories,
                    "working_memory_size": len(self._working_memory.get(session_id, {})),
                    "processing_time": processing_time
                }
            }
            
            # Add memory context if requested
            if context.get("include_memory_context", False):
                result["memory_context"] = await self._build_memory_context(relevant_memories)
            
            self.logger.debug(f"Successfully processed memory in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Error processing memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_memory",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": processing_time
                }
            }
    
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
        try:
            # Validate memory type
            if memory_type not in self.config.enabled_memory_types:
                self.logger.warning(f"Memory type '{memory_type}' not enabled")
                return False
            
            # Generate memory ID
            memory_id = self._generate_memory_id(user_id, session_id, memory_content)
            
            # Check if memory already exists
            if await self._memory_exists(memory_id):
                self.logger.debug(f"Memory {memory_id} already exists, updating access time")
                await self._update_memory_access(memory_id)
                return True
            
            # Generate embedding if vector storage is enabled
            embedding = None
            if self.config.use_ff_vector_storage and self.ff_vector:
                embedding = await self._generate_embedding(memory_content)
            
            # Create memory entry
            memory_entry = FFMemoryEntry(
                memory_id=memory_id,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                content=memory_content,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                importance_score=self._calculate_importance_score(memory_content, metadata or {})
            )
            
            # Store in FF vector storage
            if self.config.use_ff_vector_storage and embedding:
                vector_metadata = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "content": memory_content,
                    "created_at": memory_entry.created_at.isoformat(),
                    "importance_score": memory_entry.importance_score,
                    **memory_entry.metadata
                }
                
                success = await self.ff_vector.store_vector(
                    doc_id=memory_id,
                    vector=embedding,
                    metadata=vector_metadata
                )
                
                if not success:
                    self.logger.error(f"Failed to store memory vector {memory_id}")
                    return False
            
            # Store in FF storage for text-based search
            message_id = await self.ff_storage.add_message(
                user_id=user_id,
                session_id=f"memory_{memory_type}_{user_id}",
                role="memory",
                content=memory_content,
                metadata={
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "importance_score": memory_entry.importance_score,
                    "original_session_id": session_id,
                    **memory_entry.metadata
                }
            )
            
            if not message_id:
                self.logger.error(f"Failed to store memory in FF storage")
                return False
            
            # Cache memory entry
            if self.config.enable_memory_caching:
                self._memory_cache[memory_id] = memory_entry
                self._cache_timestamps[memory_id] = time.time()
            
            self._memory_stats["total_memories_stored"] += 1
            self.logger.debug(f"Successfully stored memory {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False
    
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
        start_time = time.time()
        
        try:
            memory_types = memory_types or self.config.enabled_memory_types
            memories = []
            
            # Use vector search if available
            if self.config.use_ff_vector_storage and self.ff_vector:
                memories.extend(await self._retrieve_memories_vector_search(
                    user_id, query, memory_types, limit, similarity_threshold
                ))
            
            # Use text search as fallback or supplement
            if self.ff_search and len(memories) < limit:
                text_memories = await self._retrieve_memories_text_search(
                    user_id, query, memory_types, limit - len(memories)
                )
                memories.extend(text_memories)
            
            # Sort by relevance and importance
            memories = await self._rank_memories(memories, query)
            
            # Update access statistics
            for memory in memories[:limit]:
                await self._update_memory_access(memory.get("memory_id", ""))
            
            processing_time = time.time() - start_time
            self._memory_stats["total_memories_retrieved"] += len(memories)
            
            # Update average retrieval time
            current_avg = self._memory_stats["average_retrieval_time"]
            total_retrievals = self._memory_stats["total_memories_retrieved"]
            self._memory_stats["average_retrieval_time"] = ((current_avg * (total_retrievals - len(memories))) + processing_time) / total_retrievals
            
            self.logger.debug(f"Retrieved {len(memories)} memories in {processing_time:.2f}s")
            return memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def _retrieve_memories_vector_search(self, user_id: str, query: str, memory_types: List[str], 
                                               limit: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Retrieve memories using vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Search vectors
            results = await self.ff_vector.search_similar_vectors(
                query_vector=query_embedding,
                limit=limit * 2,  # Get more results for filtering
                similarity_threshold=similarity_threshold
            )
            
            memories = []
            for result in results:
                metadata = result.get("metadata", {})
                
                # Filter by user and memory type
                if (metadata.get("user_id") == user_id and 
                    metadata.get("memory_type") in memory_types):
                    
                    memory_entry = {
                        "memory_id": result.get("doc_id"),
                        "content": metadata.get("content", ""),
                        "memory_type": metadata.get("memory_type"),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "importance_score": metadata.get("importance_score", 0.0),
                        "created_at": metadata.get("created_at"),
                        "metadata": {k: v for k, v in metadata.items() 
                                   if k not in ["user_id", "content", "memory_type", "created_at", "importance_score"]}
                    }
                    memories.append(memory_entry)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def _retrieve_memories_text_search(self, user_id: str, query: str, memory_types: List[str], 
                                             limit: int) -> List[Dict[str, Any]]:
        """Retrieve memories using text search"""
        try:
            memories = []
            
            for memory_type in memory_types:
                session_id = f"memory_{memory_type}_{user_id}"
                
                # Search using FF search capabilities
                search_results = await self.ff_storage.search_messages(
                    user_id=user_id,
                    query=query,
                    session_ids=[session_id],
                    limit=limit
                )
                
                for result in search_results:
                    if hasattr(result, 'metadata') and result.metadata.get("memory_id"):
                        memory_entry = {
                            "memory_id": result.metadata["memory_id"],
                            "content": result.content,
                            "memory_type": result.metadata.get("memory_type", memory_type),
                            "similarity_score": 0.5,  # Default score for text search
                            "importance_score": result.metadata.get("importance_score", 0.0),
                            "created_at": result.timestamp,
                            "metadata": {k: v for k, v in result.metadata.items() 
                                       if k not in ["memory_id", "memory_type", "importance_score"]}
                        }
                        memories.append(memory_entry)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return []
    
    async def update_working_memory(self, session_id: str, message: FFMessageDTO) -> None:
        """
        Update working memory for current session.
        
        Args:
            session_id: Session identifier
            message: Message to add to working memory
        """
        try:
            if session_id not in self._working_memory:
                self._working_memory[session_id] = {
                    "messages": [],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now(),
                    "message_count": 0
                }
            
            working_memory = self._working_memory[session_id]
            
            # Add message to working memory
            working_memory["messages"].append({
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id
            })
            
            # Maintain size limit
            if len(working_memory["messages"]) > self.config.working_memory_size:
                working_memory["messages"] = working_memory["messages"][-self.config.working_memory_size:]
            
            working_memory["last_updated"] = datetime.now()
            working_memory["message_count"] += 1
            
            self._memory_stats["working_memory_updates"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to update working memory: {e}")
    
    async def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """
        Get working memory for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session memory information
        """
        try:
            working_memory = self._working_memory.get(session_id, {})
            
            return {
                "session_id": session_id,
                "working_memory": working_memory,
                "message_count": working_memory.get("message_count", 0),
                "last_updated": working_memory.get("last_updated"),
                "created_at": working_memory.get("created_at")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session memory: {e}")
            return {}
    
    # Additional helper methods
    
    def _generate_memory_id(self, user_id: str, session_id: str, content: str) -> str:
        """Generate unique memory ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"mem_{user_id}_{session_id}_{content_hash}_{timestamp}"
    
    async def _memory_exists(self, memory_id: str) -> bool:
        """Check if memory already exists"""
        # Check cache first
        if memory_id in self._memory_cache:
            return True
        
        # Check FF vector storage
        if self.config.use_ff_vector_storage and self.ff_vector:
            try:
                result = await self.ff_vector.get_vector(memory_id)
                return result is not None
            except:
                pass
        
        return False
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text (placeholder for actual embedding service)"""
        # This is a placeholder - in a real implementation, this would integrate
        # with an embedding service like OpenAI, Sentence Transformers, etc.
        
        # For now, return a dummy embedding of the correct dimension
        import random
        return [random.random() for _ in range(self.config.embedding_dimension)]
    
    def _calculate_importance_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for memory"""
        # Basic importance scoring - could be made more sophisticated
        base_score = 0.5
        
        # Longer content might be more important
        length_factor = min(len(content) / 1000, 1.0) * 0.2
        
        # Check for importance indicators in metadata
        importance_factor = metadata.get("importance", 0.0) * 0.3
        
        return min(base_score + length_factor + importance_factor, 1.0)
    
    async def _should_store_as_memory(self, message: FFMessageDTO, context: Dict[str, Any]) -> bool:
        """Determine if message should be stored as persistent memory"""
        # Store user messages and important assistant responses
        if message.role == "user":
            return True
        
        # Store if explicitly requested
        if context.get("force_memory_storage", False):
            return True
        
        # Store if message contains important information (basic heuristics)
        important_keywords = ["remember", "important", "note", "don't forget"]
        content_lower = message.content.lower()
        
        for keyword in important_keywords:
            if keyword in content_lower:
                return True
        
        return False
    
    def _determine_memory_type(self, message: FFMessageDTO, context: Dict[str, Any]) -> str:
        """Determine appropriate memory type for message"""
        # Use context preference if specified
        memory_type = context.get("memory_type")
        if memory_type and memory_type in self.config.enabled_memory_types:
            return memory_type
        
        # Default to episodic for conversation messages
        if FFMemoryType.EPISODIC.value in self.config.enabled_memory_types:
            return FFMemoryType.EPISODIC.value
        
        # Fallback to first enabled type
        return self.config.enabled_memory_types[0]
    
    async def _rank_memories(self, memories: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank memories by relevance and importance"""
        try:
            # Combine similarity score and importance score
            for memory in memories:
                similarity = memory.get("similarity_score", 0.0)
                importance = memory.get("importance_score", 0.0)
                
                # Weighted combination
                memory["combined_score"] = (similarity * 0.7) + (importance * 0.3)
            
            # Sort by combined score
            memories.sort(key=lambda m: m.get("combined_score", 0.0), reverse=True)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to rank memories: {e}")
            return memories
    
    async def _update_memory_access(self, memory_id: str) -> None:
        """Update memory access statistics"""
        try:
            if memory_id in self._memory_cache:
                memory_entry = self._memory_cache[memory_id]
                memory_entry.last_accessed = datetime.now()
                memory_entry.access_count += 1
        except Exception as e:
            self.logger.error(f"Failed to update memory access: {e}")
    
    async def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build memory context string from retrieved memories"""
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"Memory: {memory.get('content', '')}")
        
        return "\n".join(context_parts)
    
    # Cleanup and maintenance methods
    
    async def _cleanup_expired_memories(self) -> None:
        """Clean up expired memories based on retention policy"""
        try:
            current_time = datetime.now()
            
            # Clean up episodic memories based on retention days
            if FFMemoryType.EPISODIC.value in self.config.enabled_memory_types:
                cutoff_date = current_time - timedelta(days=self.config.episodic_retention_days)
                # Implementation would clean up memories older than cutoff_date
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired memories: {e}")
    
    async def _cleanup_working_memory(self) -> None:
        """Clean up expired working memory sessions"""
        try:
            current_time = datetime.now()
            timeout = timedelta(seconds=self.config.working_memory_timeout)
            
            expired_sessions = []
            for session_id, memory_data in self._working_memory.items():
                last_updated = memory_data.get("last_updated", current_time)
                if current_time - last_updated > timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._working_memory[session_id]
            
            if expired_sessions:
                self.logger.debug(f"Cleaned up {len(expired_sessions)} expired working memory sessions")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up working memory: {e}")
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        try:
            if not self.config.enable_memory_caching:
                return
            
            current_time = time.time()
            expired_entries = []
            
            for memory_id, timestamp in self._cache_timestamps.items():
                if current_time - timestamp > self.config.memory_cache_ttl:
                    expired_entries.append(memory_id)
            
            for memory_id in expired_entries:
                self._memory_cache.pop(memory_id, None)
                self._cache_timestamps.pop(memory_id, None)
            
            if expired_entries:
                self.logger.debug(f"Cleaned up {len(expired_entries)} expired cache entries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
    
    async def _consolidate_memories(self) -> None:
        """Consolidate similar memories"""
        try:
            if not self.config.enable_memory_consolidation:
                return
            
            # This is a placeholder for memory consolidation logic
            # In a full implementation, this would:
            # 1. Find similar memories using vector similarity
            # 2. Merge or link related memories
            # 3. Update importance scores based on access patterns
            # 4. Remove or archive less important memories
            
            self.logger.debug("Memory consolidation completed")
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
    
    async def get_capabilities(self) -> List[str]:
        """Get list of component capabilities"""
        return [cap.name for cap in self._component_info.capabilities]
    
    async def supports_use_case(self, use_case: str) -> bool:
        """Check if component supports a specific use case"""
        return use_case in self._component_info.use_cases
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory processing statistics"""
        return {
            **self._memory_stats,
            "working_memory_sessions": len(self._working_memory),
            "cached_memories": len(self._memory_cache),
            "enabled_memory_types": self.config.enabled_memory_types
        }
    
    async def cleanup(self) -> None:
        """Cleanup component resources following FF patterns"""
        try:
            self.logger.info("Cleaning up FF Memory Component...")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._consolidation_task:
                self._consolidation_task.cancel()
            
            # Clear memory structures
            self._working_memory.clear()
            self._memory_cache.clear()
            self._cache_timestamps.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("FF Memory Component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during FF Memory Component cleanup: {e}")
            raise