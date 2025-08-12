"""
Main FFStorageManager class - the primary API interface for the flatfile chat database.

This class provides all storage operations for chat sessions, messages, documents,
and other data types.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
from backends import StorageBackend, FlatfileBackend
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFPanelDTO, FFSituationalContextDTO, FFDocumentDTO,
    FFUserProfileDTO, FFPersonaDTO, FFPanelMessageDTO, FFPanelInsightDTO
)
from ff_utils import (
    ff_write_json, ff_read_json, ff_append_jsonl, ff_read_jsonl, ff_read_jsonl_paginated,
    ff_get_user_path, ff_get_session_path, ff_get_panel_path, ff_get_global_personas_path,
    ff_get_user_personas_path, ff_get_documents_path, ff_get_context_history_path,
    ff_generate_session_id, ff_generate_panel_id, ff_generate_context_snapshot_id,
    ff_sanitize_filename, ff_build_file_paths, ff_build_panel_file_paths,
    # New centralized key functions
    ff_get_user_key, ff_get_session_key, ff_get_profile_key, ff_get_messages_key,
    ff_get_session_metadata_key
)
from ff_utils.ff_logging import get_logger
from ff_search_manager import FFSearchManager, FFSearchQueryDTO, FFSearchResultDTO
from ff_vector_storage_manager import FFVectorStorageManager
from ff_chunking_manager import FFChunkingManager
from ff_embedding_manager import FFEmbeddingManager

# Import PrismMind integration
try:
    from prismmind_integration import (
        FlatfileDocumentProcessor,
        FlatfilePrismMindConfig,
        FlatfilePrismMindConfigLoader
    )
    PRISMMIND_INTEGRATION_AVAILABLE = True
except ImportError:
    FlatfileDocumentProcessor = None
    FlatfilePrismMindConfig = None
    FlatfilePrismMindConfigLoader = None
    PRISMMIND_INTEGRATION_AVAILABLE = False


class FFStorageManager:
    """
    Main API interface for all storage operations.
    
    This class provides a high-level interface for managing chat data,
    abstracting away the underlying storage backend.
    """
    
    def __init__(self, config: Optional[FFConfigurationManagerConfigDTO] = None, 
                 backend: Optional[StorageBackend] = None,
                 enable_prismmind: bool = True):
        """
        Initialize storage manager.
        
        Args:
            config: Storage configuration (uses default if not provided)
            backend: Storage backend (uses FlatfileBackend if not provided)
            enable_prismmind: Whether to enable PrismMind integration
        """
        self.config = config or load_config()
        self.backend = backend or FlatfileBackend(self.config)
        self.base_path = Path(self.config.storage.base_path)
        self._initialized = False
        self.logger = get_logger(__name__)
        
        # Lazy-loaded components
        self._search_engine = None
        self._vector_storage = None
        self._chunking_engine = None
        self._embedding_engine = None
        self._document_processor = None
        
        # PrismMind configuration
        self._prismmind_processor = None
        self._prismmind_available = PRISMMIND_INTEGRATION_AVAILABLE and enable_prismmind
        self._enable_prismmind = enable_prismmind
    
    @property
    def search_engine(self) -> FFSearchManager:
        """Lazy-load search engine."""
        if self._search_engine is None:
            from ff_dependency_injection_manager import ff_get_container
            from ff_protocols import SearchProtocol
            try:
                self._search_engine = ff_get_container().resolve(SearchProtocol)
            except:
                # Fallback to direct instantiation
                self._search_engine = FFSearchManager(self.config)
        return self._search_engine
    
    @property
    def vector_storage(self) -> FFVectorStorageManager:
        """Lazy-load vector storage."""
        if self._vector_storage is None:
            from ff_dependency_injection_manager import ff_get_container
            from ff_protocols import VectorStoreProtocol
            try:
                self._vector_storage = ff_get_container().resolve(VectorStoreProtocol)
            except:
                # Fallback to direct instantiation
                self._vector_storage = FFVectorStorageManager(self.config)
        return self._vector_storage
    
    @property
    def chunking_engine(self) -> FFChunkingManager:
        """Lazy-load chunking engine."""
        if self._chunking_engine is None:
            self._chunking_engine = FFChunkingManager(self.config)
        return self._chunking_engine
    
    @property
    def embedding_engine(self) -> FFEmbeddingManager:
        """Lazy-load embedding engine."""
        if self._embedding_engine is None:
            self._embedding_engine = FFEmbeddingManager(self.config)
        return self._embedding_engine
    
    @property
    def document_processor(self) -> 'FFDocumentProcessingManager':
        """Lazy-load document processor."""
        if self._document_processor is None:
            from ff_dependency_injection_manager import ff_get_container
            from ff_protocols import DocumentProcessorProtocol
            try:
                self._document_processor = ff_get_container().resolve(DocumentProcessorProtocol)
            except:
                # Fallback to direct instantiation
                from ff_document_processing_manager import FFDocumentProcessingManager
                self._document_processor = FFDocumentProcessingManager(self.config)
        return self._document_processor
    
    @property
    def prismmind_processor(self) -> Optional[Any]:
        """Lazy-load PrismMind processor."""
        if self._prismmind_processor is None and self._prismmind_available:
            try:
                # Create PrismMind configuration from flatfile config
                prismmind_config = FlatfilePrismMindConfig(flatfile_config=self.config)
                self._prismmind_processor = FlatfileDocumentProcessor(prismmind_config)
            except Exception as e:
                from ff_utils.ff_logging import get_logger
                logger = get_logger(__name__)
                logger.error(f"Failed to initialize PrismMind integration: {e}", exc_info=True)
                self._prismmind_available = False
        return self._prismmind_processor
    
    @property
    def prismmind_available(self) -> bool:
        """Check if PrismMind is available."""
        return self._prismmind_available
    
    async def initialize(self) -> bool:
        """
        Initialize the storage system.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        success = await self.backend.initialize()
        if success:
            self._initialized = True
        
        return success
    
    # === User Management ===
    
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create new user with optional profile.
        
        Args:
            user_id: User identifier
            profile: Optional profile data
            
        Returns:
            True if successful
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Check if user already exists
        user_key = ff_get_user_key(self.base_path, user_id, self.config)
        if await self.backend.exists(user_key):
            self.logger.info(f"User {user_id} already exists")
            return False
        
        # Create user profile
        user_profile = FFUserProfileDTO(
            user_id=user_id,
            username=profile.get("username", "") if profile else "",
            preferences=profile.get("preferences", {}) if profile else {},
            metadata=profile.get("metadata", {}) if profile else {}
        )
        
        # Save profile
        profile_key = ff_get_profile_key(self.base_path, user_id, self.config)
        profile_data = user_profile.to_dict()
        
        return await self._write_json(profile_key, profile_data)
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile data or None
        """
        profile_key = ff_get_profile_key(self.base_path, user_id, self.config)
        return await self._read_json(profile_key)
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile.
        
        Args:
            user_id: User identifier
            updates: Profile updates to apply
            
        Returns:
            True if successful
        """
        # Get existing profile
        profile_data = await self.get_user_profile(user_id)
        if not profile_data:
            return False
        
        # Update profile
        profile = FFUserProfileDTO.from_dict(profile_data)
        
        # Update fields
        if "username" in updates:
            profile.username = updates["username"]
        if "preferences" in updates:
            profile.preferences.update(updates["preferences"])
        if "metadata" in updates:
            profile.metadata.update(updates["metadata"])
        
        profile.updated_at = datetime.now().isoformat()
        
        # Save updated profile
        profile_key = ff_get_profile_key(self.base_path, user_id, self.config)
        return await self._write_json(profile_key, profile.to_dict())
    
    async def store_user_profile(self, profile: FFUserProfileDTO) -> bool:
        """
        Store a user profile object directly.
        
        Args:
            profile: FFUserProfileDTO object to store
            
        Returns:
            True if successful
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Check if user already exists
        user_key = ff_get_user_key(self.base_path, profile.user_id, self.config)
        user_exists = await self.backend.exists(user_key)
        
        if user_exists:
            # Update existing profile
            return await self.update_user_profile(
                profile.user_id,
                {
                    "username": profile.username,
                    "preferences": profile.preferences,
                    "metadata": profile.metadata
                }
            )
        else:
            # Create new user with profile
            return await self.create_user(
                profile.user_id,
                profile.to_dict()
            )
    
    async def user_exists(self, user_id: str) -> bool:
        """
        Check if user exists.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user exists
        """
        user_key = ff_get_user_key(self.base_path, user_id, self.config)
        return await self.backend.exists(user_key)
    
    async def list_users(self) -> List[str]:
        """
        List all users in the system.
        
        Returns:
            List of user IDs
        """
        # List all directories in the user data directory
        user_data_dir = self.config.user_data_directory_name
        pattern = f"{user_data_dir}/*/"
        keys = await self.backend.list_keys("", pattern=pattern)
        
        users = []
        for key in keys:
            # Extract user ID from path: "users/user_id/"
            parts = key.rstrip('/').split('/')
            if len(parts) >= 2:
                user_id = parts[1]  # Second part is the user ID
                if user_id:
                    users.append(user_id)
        
        return sorted(list(set(users)))
    
    # === Session Management ===
    
    async def create_session(self, user_id: str, title: Optional[str] = None, session_type: str = "chat") -> str:
        """
        Create new chat session.
        
        Args:
            user_id: User identifier
            title: Optional session title
            session_type: Type of session (chat, panel, debate, problem_solving)
            
        Returns:
            Session ID
        """
        # Validate and get proper user (reject personas)
        from ff_user_context_manager import get_user_context
        user_context = get_user_context()
        validated_user_id = user_context.validate_and_get_user(user_id)
        
        # Log if user was changed
        if validated_user_id != user_id:
            self.logger.info(f"User ID '{user_id}' was invalid (likely a persona), using '{validated_user_id}' instead")
        
        # Use the validated user ID
        user_id = validated_user_id
        
        # Ensure user exists
        if not await self.user_exists(user_id):
            await self.create_user(user_id)
        
        # Generate session ID based on type
        session_id = ff_generate_session_id(self.config, session_type)
        
        # Set appropriate title based on session type
        if not title:
            title_map = {
                "chat": "New Chat",
                "panel": "Panel Session",
                "debate": "Debate Session",
                "problem_solving": "Problem Solving Session"
            }
            title = title_map.get(session_type, "New Session")
        
        # Create session object
        session = FFSessionDTO(
            session_id=session_id,
            user_id=user_id,
            title=title,
            session_type=session_type
        )
        
        # Save session metadata
        session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
        
        if await self._write_json(session_key, session.to_dict()):
            return session_id
        
        return ""
    
    async def get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]:
        """
        Get session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session object or None
        """
        session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
        session_data = await self._read_json(session_key)
        
        if session_data:
            return FFSessionDTO.from_dict(session_data)
        
        return None
    
    async def update_session(self, user_id: str, session_id: str, 
                           updates: Dict[str, Any]) -> bool:
        """
        Update session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            updates: Updates to apply
            
        Returns:
            True if successful
        """
        # Get existing session
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        
        # Update fields
        if "title" in updates:
            session.title = updates["title"]
        if "metadata" in updates:
            session.metadata.update(updates["metadata"])
        if "message_count" in updates:
            session.message_count = updates["message_count"]
        if "updated_at" in updates:
            session.updated_at = updates["updated_at"]
        else:
            session.update_timestamp()
        
        # Save updated session
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        paths = ff_build_file_paths(session_path, self.config)
        
        session_key = str(paths["session_metadata"].relative_to(self.base_path))
        
        return await self._write_json(session_key, session.to_dict())
    
    async def update_session_metadata(self, user_id: str, session_id: str, 
                                    metadata: Dict[str, Any]) -> bool:
        """
        Update session metadata - compatibility method for chat interface.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            metadata: Metadata updates to apply
            
        Returns:
            True if successful
        """
        return await self.update_session(user_id, session_id, {"metadata": metadata})
    
    async def list_sessions(self, user_id: str, limit: Optional[int] = None, 
                          offset: int = 0) -> List[FFSessionDTO]:
        """
        List user's sessions with pagination.
        
        Args:
            user_id: User identifier
            limit: Maximum sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of FFSessionDTO objects
        """
        user_key = ff_get_user_key(self.base_path, user_id, self.config)
        
        # List all session directories
        # Use just the filename for recursive search
        pattern = self.config.storage.session_metadata_filename
        session_keys = await self.backend.list_keys(user_key, pattern=pattern)
        
        # Sort by session ID (which includes timestamp)
        session_keys.sort(reverse=True)
        
        # Apply pagination
        limit = limit or self.config.search.default_page_size
        paginated_keys = session_keys[offset:offset + limit]
        
        # Load session metadata
        sessions = []
        for key in paginated_keys:
            session_data = await self._read_json(key)
            if session_data:
                sessions.append(FFSessionDTO.from_dict(session_data))
        
        return sessions
    
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete session and all associated data.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        session_key = str(session_path.relative_to(self.base_path))
        
        return await self.backend.delete(session_key)
    
    # === Message Management ===
    
    async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool:
        """
        Add message to session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message: Message to add
            
        Returns:
            True if successful
        """
        # Validate and get proper user (reject personas)
        from ff_user_context_manager import get_user_context
        user_context = get_user_context()
        validated_user_id = user_context.validate_and_get_user(user_id)
        
        # Log if user was changed
        if validated_user_id != user_id:
            self.logger.info(f"User ID '{user_id}' was invalid (likely a persona), using '{validated_user_id}' instead")
        
        # Use the validated user ID
        user_id = validated_user_id
        
        # Check message size
        message_json = message.to_dict()
        message_size = len(str(message_json).encode('utf-8'))
        
        if message_size > self.config.storage.max_message_size_bytes:
            self.logger.warning(f"Message size {message_size} exceeds limit {self.config.storage.max_message_size_bytes}")
            return False
        
        # Get messages file path
        messages_key = ff_get_messages_key(self.base_path, user_id, session_id, self.config)
        
        # Append message
        success = await self._append_jsonl(messages_key, message_json)
        
        if success:
            # Update session message count
            session = await self.get_session(user_id, session_id)
            if session:
                session.message_count += 1
                session.update_timestamp()
                await self.update_session(user_id, session_id, {
                    "message_count": session.message_count,
                    "updated_at": session.updated_at
                })
        
        return success
    
    async def get_messages(self, user_id: str, session_id: str, 
                         limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]:
        """
        Get messages with pagination.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum messages to return
            offset: Number of messages to skip
            
        Returns:
            List of FFMessageDTO objects
        """
        messages_key = ff_get_messages_key(self.base_path, user_id, session_id, self.config)
        messages_path = self.base_path / messages_key
        
        # Read messages with pagination
        limit = limit or self.config.search.default_page_size
        messages_data = await ff_read_jsonl(messages_path, self.config, limit=limit, offset=offset)
        
        # Convert to FFMessageDTO objects
        messages = []
        for data in messages_data:
            try:
                messages.append(FFMessageDTO.from_dict(data))
            except Exception as e:
                self.logger.error(f"Failed to parse message: {e}", exc_info=True)
                continue
        
        return messages
    
    async def get_all_messages(self, user_id: str, session_id: str) -> List[FFMessageDTO]:
        """
        Get all messages in a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of all FFMessageDTO objects
        """
        return await self.get_messages(user_id, session_id, limit=None)
    
    async def search_messages(self, user_id: str, query: str, 
                            session_id: Optional[str] = None,
                            session_ids: Optional[List[str]] = None) -> List[FFMessageDTO]:
        """
        Search messages across sessions.
        
        Args:
            user_id: User identifier
            query: Search query
            session_id: Optional single session to limit search to
            session_ids: Optional list of sessions to limit search to
            
        Returns:
            List of matching messages
        """
        # This is a simple implementation - could be optimized with indexing
        matching_messages = []
        
        # Determine which sessions to search
        if session_id:
            # Search single session (backward compatibility)
            search_session_ids = [session_id]
        elif session_ids:
            # Search specific sessions
            search_session_ids = session_ids
        else:
            # Search all sessions
            sessions = await self.list_sessions(user_id, limit=None)
            search_session_ids = [session.session_id for session in sessions]
        
        # Search through the determined sessions
        for sid in search_session_ids:
            messages = await self.get_all_messages(user_id, sid)
            for msg in messages:
                if query.lower() in msg.content.lower():
                    matching_messages.append(msg)
        
        # Limit results
        limit = self.config.runtime.default_search_limit
        return matching_messages[:limit]
    
    # === Advanced Search Methods ===
    
    async def advanced_search(self, query: FFSearchQueryDTO) -> List[FFSearchResultDTO]:
        """
        Execute advanced search with multiple filters and ranking.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of search results sorted by relevance
        """
        return await self.search_engine.search(query)
    
    async def search_by_entities(self, entities: Dict[str, List[str]], 
                                user_id: Optional[str] = None,
                                limit: Optional[int] = None) -> List[FFSearchResultDTO]:
        """
        Search for content containing specific entities.
        
        Args:
            entities: Entity types and values (e.g., {"languages": ["Python"], "urls": ["github.com"]})
            user_id: Optional user scope
            limit: Maximum results (uses config default if None)
            
        Returns:
            List of search results
        """
        limit = limit or self.config.runtime.default_search_limit
        return await self.search_engine.search_by_entities(entities, user_id, limit)
    
    async def search_by_time_range(self, start_date: datetime, end_date: datetime,
                                  user_id: Optional[str] = None,
                                  query_text: Optional[str] = None,
                                  limit: Optional[int] = None) -> List[FFSearchResultDTO]:
        """
        Search within a specific time range.
        
        Args:
            start_date: Start of time range
            end_date: End of time range  
            user_id: Optional user scope
            query_text: Optional text to search for
            limit: Maximum results (uses config default if None)
            
        Returns:
            List of search results
        """
        limit = limit or self.config.runtime.default_search_limit
        return await self.search_engine.search_by_time_range(
            start_date, end_date, user_id, query_text, limit
        )
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text for analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types to values
        """
        return await self.search_engine.extract_entities(text)
    
    async def build_search_index(self, user_id: str) -> Dict[str, Any]:
        """
        Build search index for a user (for optimization).
        
        Args:
            user_id: User to index
            
        Returns:
            Index metadata
        """
        return await self.search_engine.build_search_index(user_id)
    
    # === Session Statistics ===
    
    async def get_session_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session statistics and analytics.
        
        Provides detailed information about a chat session including message counts,
        document statistics, storage usage, vector information, and timestamps.
        Useful for monitoring, debugging, analytics, and user experience features.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary containing session statistics:
            - session_id: Session identifier
            - user_id: User identifier  
            - message_count: Total number of messages
            - document_count: Number of uploaded documents
            - total_size_bytes: Total storage size in bytes
            - average_message_size: Average message size in bytes
            - created_at: Session creation timestamp
            - updated_at: Last update timestamp
            - last_activity: Timestamp of most recent activity
            - vector_count: Number of stored vectors (if available)
            - storage_path: Relative path to session storage
            - has_context: Whether situational context exists
            - context_snapshots: Number of context history snapshots
            
        Raises:
            ValueError: If session doesn't exist
            
        Usage Examples:
            # Get basic session statistics
            stats = await storage.get_session_stats("alice", "session_123")
            print(f"Messages: {stats['message_count']}")
            
            # Monitor storage usage
            if (stats['total_size_bytes'] > self.config.runtime.large_session_threshold_bytes and 
                self.config.runtime.enable_large_session_warnings):
                self.logger.info("Large session detected", extra={
                    'session_size_bytes': stats['total_size_bytes'],
                    'threshold_bytes': self.config.runtime.large_session_threshold_bytes
                })
            
            # Check session activity
            if (stats['message_count'] == 0 and 
                self.config.runtime.enable_empty_session_notifications):
                self.logger.info("Empty session detected", extra={
                    'session_id': session_id,
                    'user_id': user_id
                })
        """
        # Validate session exists
        session = await self.get_session(user_id, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found for user {user_id}")
        
        # Initialize statistics
        stats = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': session.created_at,
            'updated_at': session.updated_at,
            'storage_path': f"users/{user_id}/chat_session_{session_id}",
            'message_count': 0,
            'document_count': 0,
            'total_size_bytes': 0,
            'average_message_size': 0.0,
            'vector_count': 0,
            'has_context': False,
            'context_snapshots': 0,
            'last_activity': session.updated_at or session.created_at
        }
        
        try:
            # Get message statistics
            messages = await self.get_all_messages(user_id, session_id)
            stats['message_count'] = len(messages)
            
            if messages:
                # Calculate message size statistics
                message_sizes = []
                for msg in messages:
                    msg_size = len(msg.content.encode('utf-8'))
                    message_sizes.append(msg_size)
                    # Update last activity from message timestamps
                    if msg.timestamp > stats['last_activity']:
                        stats['last_activity'] = msg.timestamp
                
                stats['total_size_bytes'] = sum(message_sizes)
                stats['average_message_size'] = round(sum(message_sizes) / len(message_sizes), 2)
            
        except Exception as e:
            self.logger.warning(f"Could not get message stats: {e}", exc_info=True)
        
        try:
            # Get document statistics
            documents = await self.list_documents(user_id, session_id)
            stats['document_count'] = len(documents)
            
            # Add document sizes to total
            for doc in documents:
                if hasattr(doc, 'size') and doc.size:
                    stats['total_size_bytes'] += doc.size
                    
        except Exception as e:
            self.logger.warning(f"Could not get document stats: {e}", exc_info=True)
        
        try:
            # Get vector storage statistics
            vector_stats = await self.get_vector_stats(user_id, session_id)
            if vector_stats:
                stats['vector_count'] = vector_stats.get('total_vectors', 0)
                # Add vector storage size if available
                if 'storage_size_bytes' in vector_stats:
                    stats['total_size_bytes'] += vector_stats['storage_size_bytes']
                    
        except Exception as e:
            self.logger.warning(f"Could not get vector stats: {e}", exc_info=True)
        
        try:
            # Check for situational context
            current_context = await self.get_context(user_id, session_id)
            stats['has_context'] = current_context is not None
            
            # Get context history count
            context_history = await self.get_context_history(user_id, session_id, limit=None)
            stats['context_snapshots'] = len(context_history) if context_history else 0
            
        except Exception as e:
            self.logger.warning(f"Could not get context stats: {e}", exc_info=True)
        
        return stats
    
    # === Helper Methods ===
    
    async def _write_json(self, key: str, data: Dict[str, Any]) -> bool:
        """Write JSON data using backend"""
        json_path = self.base_path / key
        return await ff_write_json(json_path, data, self.config)
    
    async def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON data using backend"""
        json_path = self.base_path / key
        return await ff_read_json(json_path, self.config)
    
    async def _append_jsonl(self, key: str, entry: Dict[str, Any]) -> bool:
        """Append to JSONL file using backend"""
        jsonl_path = self.base_path / key
        return await ff_append_jsonl(jsonl_path, entry, self.config)
    
    # === Document Management ===
    
    async def save_document(self, user_id: str, session_id: str, filename: str, 
                          content: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save document and return storage path.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            filename: Original filename
            content: File content as bytes
            metadata: Optional metadata
            
        Returns:
            Document ID or empty string if failed
        """
        # Check document size
        if len(content) > self.config.storage.max_document_size_bytes:
            self.logger.warning(f"Document size {len(content)} exceeds limit {self.config.storage.max_document_size_bytes}")
            return ""
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in self.config.document.allowed_extensions:
            self.logger.warning(f"File extension {ext} not allowed")
            return ""
        
        # Generate safe filename
        safe_filename = ff_sanitize_filename(filename)
        doc_id = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
        
        # Get document path
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        doc_dir = ff_get_documents_path(session_path, self.config)
        doc_path = doc_dir / doc_id
        
        # Save document
        doc_key = str(doc_path.relative_to(self.base_path))
        if not await self.backend.write(doc_key, content):
            return ""
        
        # Create document metadata
        document = FFDocumentDTO(
            filename=doc_id,
            original_name=filename,
            path=doc_key,
            mime_type=self._guess_mime_type(filename),
            size=len(content),
            uploaded_by=user_id,
            metadata=metadata or {}
        )
        
        # Update document metadata file
        metadata_path = doc_dir / self.config.document.metadata_filename
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        # Load existing metadata
        docs_metadata = await self._read_json(metadata_key) or {}
        docs_metadata[doc_id] = document.to_dict()
        
        # Save updated metadata
        if await self._write_json(metadata_key, docs_metadata):
            return doc_id
        
        return ""
    
    async def get_document(self, user_id: str, session_id: str, 
                         document_id: str) -> Optional[bytes]:
        """
        Retrieve document content.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            Document content or None
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        doc_dir = ff_get_documents_path(session_path, self.config)
        doc_path = doc_dir / document_id
        
        doc_key = str(doc_path.relative_to(self.base_path))
        return await self.backend.read(doc_key)
    
    async def list_documents(self, user_id: str, session_id: str) -> List[FFDocumentDTO]:
        """
        List all documents in session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of FFDocumentDTO objects
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        doc_dir = ff_get_documents_path(session_path, self.config)
        metadata_path = doc_dir / self.config.document.metadata_filename
        
        metadata_key = str(metadata_path.relative_to(self.base_path))
        docs_metadata = await self._read_json(metadata_key) or {}
        
        documents = []
        for doc_data in docs_metadata.values():
            try:
                documents.append(FFDocumentDTO.from_dict(doc_data))
            except Exception as e:
                self.logger.error(f"Failed to parse document metadata: {e}", exc_info=True)
                continue
        
        return documents
    
    async def update_document_analysis(self, user_id: str, session_id: str,
                                     document_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Store document analysis results.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            analysis: Analysis results
            
        Returns:
            True if successful
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        doc_dir = ff_get_documents_path(session_path, self.config)
        metadata_path = doc_dir / self.config.document.metadata_filename
        
        metadata_key = str(metadata_path.relative_to(self.base_path))
        docs_metadata = await self._read_json(metadata_key) or {}
        
        if document_id not in docs_metadata:
            return False
        
        # Update document analysis
        doc = FFDocumentDTO.from_dict(docs_metadata[document_id])
        doc.add_analysis("analysis", analysis)
        docs_metadata[document_id] = doc.to_dict()
        
        return await self._write_json(metadata_key, docs_metadata)
    
    # === Vector Storage and Search ===
    
    async def store_document_with_vectors(
        self,
        user_id: str,
        session_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = None,
        embedding_provider: str = None,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Store document with automatic chunking and embedding generation.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            content: Document text content
            metadata: Optional document metadata
            chunking_strategy: Strategy for chunking (default: "optimized_summary")
            embedding_provider: Provider for embeddings (default: "nomic-ai")
            api_key: API key if required by provider
            
        Returns:
            True if successful
        """
        # Store the document normally first
        # Add a .txt extension if document_id doesn't have one
        filename = document_id if '.' in document_id else f"{document_id}.txt"
        doc_stored = await self.save_document(
            user_id, session_id, filename,
            content.encode('utf-8'), metadata
        )
        
        if not doc_stored:
            return False
        
        try:
            # Chunk the content
            chunks = await self.chunking_engine.ff_chunk_text(
                content, 
                strategy=chunking_strategy
            )
            
            # Generate embeddings
            embedding_results = await self.embedding_engine.generate_embeddings(
                chunks,
                provider=embedding_provider,
                api_key=api_key
            )
            
            # Extract vectors
            vectors = [r["embedding_vector"] for r in embedding_results]
            
            # Store vectors
            vector_metadata = {
                "provider": embedding_provider or self.config.vector.default_embedding_provider,
                "chunking_strategy": chunking_strategy or self.config.vector.default_chunking_strategy,
                "document_id": document_id,
                "chunk_count": len(chunks)
            }
            
            success = await self.vector_storage.store_vectors(
                session_id=session_id,
                document_id=document_id,
                chunks=chunks,
                vectors=vectors,
                metadata=vector_metadata
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating vectors: {e}", exc_info=True)
            return False
    
    async def vector_search(
        self,
        user_id: str,
        query: str,
        session_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        embedding_provider: str = None,
        api_key: Optional[str] = None
    ) -> List[FFSearchResultDTO]:
        """
        Perform vector similarity search across sessions.
        
        Args:
            user_id: User identifier
            query: Search query text
            session_ids: Optional list of sessions to search (None = all)
            top_k: Number of results to return (uses config default if None)
            threshold: Minimum similarity threshold (uses config default if None)
            embedding_provider: Provider for query embedding (default: "nomic-ai")
            api_key: API key if required
            
        Returns:
            List of search results sorted by relevance
        """
        # Use config defaults if not provided
        top_k = top_k or self.config.runtime.vector_search_top_k
        threshold = threshold or self.config.runtime.similarity_threshold
        # Generate query embedding
        embedding_results = await self.embedding_engine.generate_embeddings(
            [query],
            provider=embedding_provider,
            api_key=api_key
        )
        
        query_vector = embedding_results[0]["embedding_vector"]
        
        # Get sessions to search
        if not session_ids:
            sessions = await self.list_sessions(user_id)
            session_ids = [s.session_id for s in sessions]
        
        # Search across sessions
        all_results = []
        
        for session_id in session_ids:
            try:
                results = await self.vector_storage.search_similar(
                    session_id=session_id,
                    query_vector=query_vector,
                    top_k=top_k,
                    threshold=threshold
                )
                
                # Convert to FFSearchResult format
                for r in results:
                    search_result = FFSearchResultDTO(
                        id=r.chunk_id,
                        type="vector_chunk",
                        content=r.chunk_text,
                        session_id=r.session_id,
                        user_id=user_id,
                        timestamp=datetime.now().isoformat(),
                        relevance_score=r.similarity_score,
                        highlights=[],
                        metadata={
                            "document_id": r.document_id,
                            "search_type": "vector",
                            **r.metadata
                        }
                    )
                    all_results.append(search_result)
                    
            except Exception as e:
                self.logger.warning(f"Error searching session {session_id}: {e}", exc_info=True)
                continue
        
        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_results[:top_k]
    
    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        session_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        **kwargs
    ) -> List[FFSearchResultDTO]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            user_id: User identifier
            query: Search query
            session_ids: Sessions to search
            top_k: Number of results (uses config default if None)
            vector_weight: Weight for vector search (0-1, uses config default if None)
            **kwargs: Additional arguments for searches
            
        Returns:
            Combined and re-ranked results
        """
        # Use config defaults if not provided
        top_k = top_k or self.config.runtime.vector_search_top_k
        vector_weight = vector_weight or self.config.runtime.hybrid_search_vector_weight
        # Perform text search
        search_query = FFSearchQueryDTO(
            query=query,
            user_id=user_id,
            session_ids=session_ids,
            max_results=top_k
        )
        text_results = await self.search_engine.search(search_query)
        
        # Perform vector search
        vector_results = await self.vector_search(
            user_id, query, session_ids, top_k=top_k, **kwargs
        )
        
        # Combine and re-rank
        combined = {}
        
        # Add text results
        for r in text_results:
            key = f"{r.session_id}_{r.id}"
            combined[key] = {
                "result": r,
                "text_score": r.relevance_score,
                "vector_score": 0.0
            }
        
        # Add/update with vector results
        for r in vector_results:
            key = f"{r.session_id}_{r.id}"
            if key in combined:
                combined[key]["vector_score"] = r.relevance_score
            else:
                combined[key] = {
                    "result": r,
                    "text_score": 0.0,
                    "vector_score": r.relevance_score
                }
        
        # Calculate combined scores
        for key, data in combined.items():
            text_weight = 1.0 - vector_weight
            data["combined_score"] = (
                text_weight * data["text_score"] +
                vector_weight * data["vector_score"]
            )
            data["result"].relevance_score = data["combined_score"]
        
        # Sort and return
        results = [data["result"] for data in combined.values()]
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:top_k]
    
    async def delete_document_vectors(
        self, 
        user_id: str,
        session_id: str, 
        document_id: str
    ) -> bool:
        """
        Delete all vectors associated with a document.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        return await self.vector_storage.delete_document_vectors(session_id, document_id)
    
    async def get_vector_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about vectors in a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with vector statistics
        """
        return await self.vector_storage.get_vector_stats(session_id)
    
    # === Context Management ===
    
    async def update_context(self, user_id: str, session_id: str, 
                           context: FFSituationalContextDTO) -> bool:
        """
        Update current situational context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            context: New context
            
        Returns:
            True if successful
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        paths = ff_build_file_paths(session_path, self.config)
        
        context_key = str(paths["situational_context"].relative_to(self.base_path))
        return await self._write_json(context_key, context.to_dict())
    
    async def get_context(self, user_id: str, session_id: str) -> Optional[FFSituationalContextDTO]:
        """
        Get current context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            FFSituationalContextDTO or None
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        paths = ff_build_file_paths(session_path, self.config)
        
        context_key = str(paths["situational_context"].relative_to(self.base_path))
        context_data = await self._read_json(context_key)
        
        if context_data:
            return FFSituationalContextDTO.from_dict(context_data)
        
        return None
    
    async def save_context_snapshot(self, user_id: str, session_id: str,
                                  context: FFSituationalContextDTO) -> bool:
        """
        Save context to history.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Context to save
            
        Returns:
            True if successful
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        history_dir = ff_get_context_history_path(session_path, self.config)
        
        # Generate snapshot filename
        snapshot_id = ff_generate_context_snapshot_id(self.config)
        snapshot_path = history_dir / f"{snapshot_id}.json"
        
        snapshot_key = str(snapshot_path.relative_to(self.base_path))
        return await self._write_json(snapshot_key, context.to_dict())
    
    async def get_context_history(self, user_id: str, session_id: str,
                                limit: Optional[int] = None) -> List[FFSituationalContextDTO]:
        """
        Get context evolution history.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of snapshots
            
        Returns:
            List of context snapshots
        """
        safe_user_id = ff_sanitize_filename(user_id)
        session_path = ff_get_session_path(self.base_path, safe_user_id, session_id)
        history_dir = ff_get_context_history_path(session_path, self.config)
        history_key = str(history_dir.relative_to(self.base_path))
        
        # List snapshot files
        snapshot_keys = await self.backend.list_keys(history_key, pattern="*.json")
        
        # Sort by filename (timestamp)
        snapshot_keys.sort(reverse=True)
        
        # Apply limit
        if limit:
            snapshot_keys = snapshot_keys[:limit]
        
        # Load snapshots
        contexts = []
        for key in snapshot_keys:
            context_data = await self._read_json(key)
            if context_data:
                try:
                    contexts.append(FFSituationalContextDTO.from_dict(context_data))
                except Exception as e:
                    self.logger.error(f"Failed to parse context snapshot: {e}", exc_info=True)
                    continue
        
        return contexts
    
    # === Panel Management ===
    
    async def create_panel(self, panel_type: str, personas: List[str],
                         config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create multi-persona panel session.
        
        Args:
            panel_type: Type of panel
            personas: List of persona IDs
            config: Optional panel configuration
            
        Returns:
            Panel ID or empty string if failed
        """
        # Validate personas count
        if len(personas) > self.config.panel.max_personas_per_panel:
            self.logger.warning(f"Too many personas: {len(personas)} > {self.config.panel.max_personas_per_panel}")
            return ""
        
        # Generate panel ID
        panel_id = ff_generate_panel_id(self.config)
        
        # Create panel object
        panel = FFPanelDTO(
            id=panel_id,
            type=panel_type,
            personas=personas,
            config=config or {},
            max_personas=self.config.panel.user_persona_limit
        )
        
        # Get panel path
        panel_path = ff_get_panel_path(self.base_path, panel_id, self.config)
        paths = ff_build_panel_file_paths(panel_path, self.config)
        
        # Save panel metadata
        panel_key = str(paths["panel_metadata"].relative_to(self.base_path))
        
        if await self._write_json(panel_key, panel.to_dict()):
            # Save persona snapshots
            for persona_id in personas:
                persona_data = await self.get_persona(persona_id)
                if persona_data:
                    persona_key = str((paths["personas"] / f"{persona_id}.json").relative_to(self.base_path))
                    await self._write_json(persona_key, persona_data)
            
            return panel_id
        
        return ""
    
    async def add_panel_message(self, panel_id: str, message: FFPanelMessageDTO) -> bool:
        """
        Add message to panel.
        
        Args:
            panel_id: Panel identifier
            message: Panel message
            
        Returns:
            True if successful
        """
        panel_path = ff_get_panel_path(self.base_path, panel_id, self.config)
        paths = ff_build_panel_file_paths(panel_path, self.config)
        
        messages_key = str(paths["messages"].relative_to(self.base_path))
        return await self._append_jsonl(messages_key, message.to_dict())
    
    async def get_panel_messages(self, panel_id: str, 
                               limit: Optional[int] = None) -> List[FFPanelMessageDTO]:
        """
        Get panel conversation.
        
        Args:
            panel_id: Panel identifier
            limit: Maximum messages to return
            
        Returns:
            List of panel messages
        """
        panel_path = ff_get_panel_path(self.base_path, panel_id, self.config)
        paths = ff_build_panel_file_paths(panel_path, self.config)
        
        messages_key = str(paths["messages"].relative_to(self.base_path))
        messages_path = self.base_path / messages_key
        
        limit = limit or self.config.search.default_page_size
        messages_data = await ff_read_jsonl(messages_path, self.config, limit=limit)
        
        messages = []
        for data in messages_data:
            try:
                messages.append(FFPanelMessageDTO.from_dict(data))
            except Exception as e:
                self.logger.error(f"Failed to parse panel message: {e}", exc_info=True)
                continue
        
        return messages
    
    async def save_panel_insight(self, panel_id: str, insight: FFPanelInsightDTO) -> bool:
        """
        Save panel analysis or conclusion.
        
        Args:
            panel_id: Panel identifier
            insight: Panel insight
            
        Returns:
            True if successful
        """
        panel_path = ff_get_panel_path(self.base_path, panel_id, self.config)
        insights_dir = panel_path / self.config.panel.insights_subdirectory
        
        insight_path = insights_dir / f"{insight.id}.json"
        insight_key = str(insight_path.relative_to(self.base_path))
        
        return await self._write_json(insight_key, insight.to_dict())
    
    # === Persona Management ===
    
    async def save_persona(self, persona_id: str, data: Dict[str, Any], 
                         user_id: Optional[str] = None) -> bool:
        """
        Save global or user persona.
        
        Args:
            persona_id: Persona identifier
            data: Persona data
            user_id: Optional user ID for user-specific persona
            
        Returns:
            True if successful
        """
        if user_id:
            # User-specific persona
            personas_path = ff_get_user_personas_path(self.base_path, user_id, self.config)
        else:
            # Global persona
            personas_path = ff_get_global_personas_path(self.base_path, self.config)
        
        persona_file = personas_path / f"{ff_sanitize_filename(persona_id)}.json"
        persona_key = str(persona_file.relative_to(self.base_path))
        
        return await self._write_json(persona_key, data)
    
    async def get_persona(self, persona_id: str, 
                        user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get persona definition.
        
        Args:
            persona_id: Persona identifier
            user_id: Optional user ID for user-specific lookup
            
        Returns:
            Persona data or None
        """
        safe_persona_id = ff_sanitize_filename(persona_id)
        
        # Try user-specific first if user_id provided
        if user_id:
            personas_path = ff_get_user_personas_path(self.base_path, user_id, self.config)
            persona_file = personas_path / f"{safe_persona_id}.json"
            persona_key = str(persona_file.relative_to(self.base_path))
            
            data = await self._read_json(persona_key)
            if data:
                return data
        
        # Try global personas
        personas_path = ff_get_global_personas_path(self.base_path, self.config)
        persona_file = personas_path / f"{safe_persona_id}.json"
        persona_key = str(persona_file.relative_to(self.base_path))
        
        return await self._read_json(persona_key)
    
    async def list_personas(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available personas.
        
        Args:
            user_id: Optional user ID to include user-specific personas
            
        Returns:
            List of persona data
        """
        personas = []
        
        # Get global personas
        global_path = ff_get_global_personas_path(self.base_path, self.config)
        global_key = str(global_path.relative_to(self.base_path))
        
        global_keys = await self.backend.list_keys(global_key, pattern="*.json")
        for key in global_keys:
            persona_data = await self._read_json(key)
            if persona_data:
                personas.append(persona_data)
        
        # Get user-specific personas if requested
        if user_id:
            user_path = ff_get_user_personas_path(self.base_path, user_id, self.config)
            user_key = str(user_path.relative_to(self.base_path))
            
            user_keys = await self.backend.list_keys(user_key, pattern="*.json")
            for key in user_keys:
                persona_data = await self._read_json(key)
                if persona_data:
                    personas.append(persona_data)
        
        return personas
    
    # === PrismMind Integration ===
    
    async def process_document_with_prismmind(
        self,
        document_path: str,
        user_id: str,
        session_id: str,
        document_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document using PrismMind engines.
        
        This method provides direct access to PrismMind processing capabilities,
        offering universal file support and advanced processing chains.
        
        Args:
            document_path: Path to document file
            user_id: User identifier  
            session_id: Session identifier
            document_id: Optional document ID (auto-generated if not provided)
            chunking_strategy: Override default chunking strategy
            embedding_provider: Override default embedding provider
            api_key: API key if required for embedding
            metadata: Additional document metadata
            
        Returns:
            Processing result dictionary with success status and details
        """
        if not self.prismmind_available or not self.prismmind_processor:
            return {
                "success": False,
                "error": "PrismMind integration not available",
                "document_id": document_id or "",
                "chunk_count": 0,
                "vector_count": 0,
                "processing_time": 0
            }
        
        try:
            result = await self.prismmind_processor.process_document(
                document_path=document_path,
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                chunking_strategy=chunking_strategy,
                embedding_provider=embedding_provider,
                api_key=api_key,
                metadata=metadata
            )
            
            # Convert ProcessingResult to dict if needed
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            elif hasattr(result, '__dict__'):
                return result.__dict__
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"PrismMind processing failed: {str(e)}",
                "document_id": document_id or "",
                "chunk_count": 0,
                "vector_count": 0,
                "processing_time": 0
            }
    
    def get_prismmind_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current PrismMind configuration.
        
        Returns:
            Configuration dictionary or None if not available
        """
        if not self.prismmind_available or not self.prismmind_processor:
            return None
            
        return self.prismmind_processor.config.to_dict() if hasattr(self.prismmind_processor.config, 'to_dict') else None
    
    def is_prismmind_available(self) -> bool:
        """
        Check if PrismMind integration is available and properly initialized.
        
        Returns:
            True if PrismMind integration is available
        """
        return self.prismmind_available and self.prismmind_processor is not None
    
    # === Helper Methods ===
    
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename"""
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'application/octet-stream')