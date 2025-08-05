"""
Session management functionality extracted from FFStorageManager.

Handles session creation, message storage, and session-specific operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO
from backends import StorageBackend
from ff_utils import (
    ff_get_session_key, ff_get_messages_key, ff_get_session_metadata_key,
    ff_generate_session_id, ff_append_jsonl, ff_read_jsonl, ff_write_json, ff_read_json
)
from ff_utils.ff_logging import get_logger
from ff_utils.ff_validation import validate_user_id, validate_session_name, validate_message_content


class FFSessionManager:
    """
    Manages chat sessions and messages.
    
    Single responsibility: Session and message data management.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        """
        Initialize session manager.
        
        Args:
            config: Storage configuration
            backend: Storage backend for data operations
        """
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def create_session(self, user_id: str, session_name: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create new chat session.
        
        Args:
            user_id: User identifier
            session_name: Name for the session
            metadata: Optional session metadata
            
        Returns:
            Session ID or empty string if failed
        """
        # Validate inputs
        user_errors = validate_user_id(user_id, self.config)
        session_errors = validate_session_name(session_name, self.config)
        
        if user_errors or session_errors:
            all_errors = user_errors + session_errors
            self.logger.warning(f"Invalid session creation inputs: {'; '.join(all_errors)}")
            return ""
        
        # Generate unique session ID
        session_id = ff_generate_session_id(self.config)
        
        # Create session object
        session = FFSessionDTO(
            session_id=session_id,
            user_id=user_id,
            title=session_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Save session metadata
        session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
        session_data = session.to_dict()
        
        if await self._write_json(session_key, session_data):
            return session_id
        else:
            return ""
    
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
        # Validate message content
        content_errors = validate_message_content(message.content, self.config)
        if content_errors:
            self.logger.warning(f"Invalid message content: {'; '.join(content_errors)}")
            return False
        
        # Check message size
        message_json = message.to_dict()
        message_size = len(str(message_json).encode('utf-8'))
        
        if message_size > self.config.storage.max_message_size_bytes:
            self.logger.warning(f"Message size {message_size} exceeds limit {self.config.storage.max_message_size_bytes}")
            return False
        
        # Get messages file path
        messages_key = ff_get_messages_key(self.base_path, user_id, session_id, self.config)
        messages_path = self.base_path / messages_key
        
        # Append message to JSONL file
        success = await ff_append_jsonl(messages_path, message_json, self.config)
        
        if success:
            # Update session timestamp
            await self._update_session_timestamp(user_id, session_id)
        
        return success
    
    async def get_messages(self, user_id: str, session_id: str, 
                          limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]:
        """
        Retrieve messages from session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum messages to return
            offset: Number of messages to skip
            
        Returns:
            List of messages
        """
        # Use config default if limit not specified
        limit = limit or self.config.runtime.storage_default_message_limit
        
        messages_key = ff_get_messages_key(self.base_path, user_id, session_id, self.config)
        messages_path = self.base_path / messages_key
        
        # Read JSONL data
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
        Retrieve all messages from session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of all messages
        """
        return await self.get_messages(user_id, session_id, limit=None)
    
    async def get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]:
        """
        Retrieve session metadata.
        
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
    
    async def list_sessions(self, user_id: str, limit: Optional[int] = None, 
                           offset: int = 0) -> List[FFSessionDTO]:
        """
        List user sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of sessions
        """
        # Use config default if limit not specified  
        limit = limit or self.config.runtime.storage_default_session_limit
        
        # Get session directory pattern
        session_pattern = f"{self.config.storage.user_data_directory_name}/{user_id}/*/session.json"
        session_keys = await self.backend.list_keys("", pattern=session_pattern)
        
        sessions = []
        for session_key in session_keys[offset:offset+limit]:
            session_data = await self._read_json(session_key)
            if session_data:
                try:
                    session = FFSessionDTO.from_dict(session_data)
                    sessions.append(session)
                except Exception as e:
                    self.logger.error(f"Failed to parse session: {e}", exc_info=True)
                    continue
        
        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda s: s.created_at or "", reverse=True)
        return sessions
    
    async def update_session_metadata(self, user_id: str, session_id: str, 
                                    updates: Dict[str, Any]) -> bool:
        """
        Update session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            updates: Metadata updates to apply
            
        Returns:
            True if successful
        """
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        
        # Apply updates
        if "title" in updates:
            session.title = updates["title"]
        if "metadata" in updates:
            session.metadata.update(updates["metadata"])
        
        session.updated_at = datetime.now().isoformat()
        
        # Save updated session
        session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
        return await self._write_json(session_key, session.to_dict())
    
    async def get_session_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        session = await self.get_session(user_id, session_id)
        if not session:
            return {}
        
        stats = {
            'session_id': session_id,
            'title': session.title,
            'created_at': session.created_at,
            'updated_at': session.updated_at,
            'message_count': 0,
            'total_size_bytes': 0,
            'average_message_size': 0.0,
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
        
        return stats
    
    async def session_exists(self, user_id: str, session_id: str) -> bool:
        """
        Check if session exists.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
        return await self.backend.exists(session_key)
    
    # === Helper Methods ===
    
    async def _update_session_timestamp(self, user_id: str, session_id: str) -> None:
        """Update session's last updated timestamp"""
        session = await self.get_session(user_id, session_id)
        if session:
            session.updated_at = datetime.now().isoformat()
            session_key = ff_get_session_metadata_key(self.base_path, user_id, session_id, self.config)
            await self._write_json(session_key, session.to_dict())
    
    async def _write_json(self, key: str, data: Dict[str, Any]) -> bool:
        """Write JSON data using backend"""
        json_path = self.base_path / key
        return await ff_write_json(json_path, data, self.config)
    
    async def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON data using backend"""
        json_path = self.base_path / key
        return await ff_read_json(json_path, self.config)