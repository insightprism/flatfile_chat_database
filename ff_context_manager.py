"""
Context management functionality extracted from FFStorageManager.

Handles situational context storage, retrieval, and context-specific operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFSituationalContextDTO
from backends import StorageBackend
from ff_utils import (
    ff_get_context_history_path, ff_generate_context_snapshot_id, 
    ff_write_json, ff_read_json
)
from ff_utils.ff_logging import get_logger


class FFContextManager:
    """
    Manages situational context and context history.
    
    Single responsibility: Context data management and operations.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        """
        Initialize context manager.
        
        Args:
            config: Storage configuration
            backend: Storage backend for data operations
        """
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def store_context(self, user_id: str, session_id: str, 
                          context: FFSituationalContextDTO) -> str:
        """
        Store situational context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Context to store
            
        Returns:
            Context snapshot ID or empty string if failed
        """
        # Generate snapshot ID if not provided
        if not context.snapshot_id:
            context.snapshot_id = ff_generate_context_snapshot_id(self.config)
        
        # Set timestamp
        context.timestamp = context.timestamp or datetime.now().isoformat()
        
        # Get context file path
        context_path = ff_get_context_history_path(self.base_path, user_id, session_id, self.config)
        context_extension = self.config.runtime.context_file_extension
        context_file = context_path / f"{context.snapshot_id}{context_extension}"
        context_key = str(context_file.relative_to(self.base_path))
        
        # Store context
        if await self._write_json(context_key, context.to_dict()):
            return context.snapshot_id
        return ""
    
    async def get_context(self, user_id: str, session_id: str, 
                         snapshot_id: Optional[str] = None) -> Optional[FFSituationalContextDTO]:
        """
        Retrieve context by snapshot ID or latest.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            snapshot_id: Specific snapshot ID (None for latest)
            
        Returns:
            Context object or None
        """
        if snapshot_id:
            # Get specific context
            context_path = ff_get_context_history_path(self.base_path, user_id, session_id, self.config)
            context_extension = self.config.runtime.context_file_extension
            context_file = context_path / f"{snapshot_id}{context_extension}"
            context_key = str(context_file.relative_to(self.base_path))
            
            context_data = await self._read_json(context_key)
            if context_data:
                return FFSituationalContextDTO.from_dict(context_data)
        else:
            # Get latest context
            contexts = await self.get_context_history(user_id, session_id, limit=1)
            if contexts:
                return contexts[0]
        
        return None
    
    async def get_context_history(self, user_id: str, session_id: str, 
                                limit: Optional[int] = None) -> List[FFSituationalContextDTO]:
        """
        Get context history for session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum contexts to return (None for all)
            
        Returns:
            List of context snapshots, newest first
        """
        # Get context directory
        context_path = ff_get_context_history_path(self.base_path, user_id, session_id, self.config)
        context_extension = self.config.runtime.context_file_extension
        context_pattern = str(context_path.relative_to(self.base_path) / f"*{context_extension}")
        
        # List context files
        snapshot_keys = await self.backend.list_keys("", pattern=context_pattern)
        
        # Sort by filename (which includes timestamp)
        snapshot_keys.sort(reverse=True)  # Newest first
        
        # Apply limit
        if limit:
            snapshot_keys = snapshot_keys[:limit]
        
        # Load contexts
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
    
    async def update_context(self, user_id: str, session_id: str, snapshot_id: str,
                           updates: Dict[str, Any]) -> bool:
        """
        Update existing context snapshot.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            snapshot_id: Context snapshot ID
            updates: Updates to apply
            
        Returns:
            True if successful
        """
        # Get existing context
        context = await self.get_context(user_id, session_id, snapshot_id)
        if not context:
            return False
        
        # Apply updates
        if "context_type" in updates:
            context.context_type = updates["context_type"]
        if "active_goals" in updates:
            context.active_goals = updates["active_goals"]
        if "conversation_flow" in updates:
            context.conversation_flow = updates["conversation_flow"]
        if "user_preferences" in updates:
            context.user_preferences.update(updates["user_preferences"])
        if "environmental_factors" in updates:
            context.environmental_factors.update(updates["environmental_factors"])
        if "metadata" in updates:
            context.metadata.update(updates["metadata"])
        
        context.updated_at = datetime.now().isoformat()
        
        # Save updated context
        context_path = ff_get_context_history_path(self.base_path, user_id, session_id, self.config)
        context_extension = self.config.runtime.context_file_extension
        context_file = context_path / f"{snapshot_id}{context_extension}"
        context_key = str(context_file.relative_to(self.base_path))
        
        return await self._write_json(context_key, context.to_dict())
    
    async def delete_context(self, user_id: str, session_id: str, snapshot_id: str) -> bool:
        """
        Delete context snapshot.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            snapshot_id: Context snapshot ID
            
        Returns:
            True if successful
        """
        context_path = ff_get_context_history_path(self.base_path, user_id, session_id, self.config)
        context_extension = self.config.runtime.context_file_extension
        context_file = context_path / f"{snapshot_id}{context_extension}"
        context_key = str(context_file.relative_to(self.base_path))
        
        return await self.backend.delete(context_key)
    
    async def search_contexts(self, user_id: str, query: str, 
                            session_ids: Optional[List[str]] = None) -> List[FFSituationalContextDTO]:
        """
        Search contexts by text content.
        
        Args:
            user_id: User identifier
            query: Search query
            session_ids: Optional list of sessions to search
            
        Returns:
            List of matching contexts
        """
        matching_contexts = []
        
        # Get sessions to search
        if session_ids is None:
            # Search all user sessions - would need session manager for this
            # For now, return empty list
            self.logger.warning("Context search across all sessions not implemented without session_ids")
            return []
        
        # Search specified sessions
        for session_id in session_ids:
            try:
                contexts = await self.get_context_history(user_id, session_id)
                
                for context in contexts:
                    # Simple text search in context fields
                    searchable_text = " ".join([
                        context.context_type or "",
                        " ".join(context.active_goals or []),
                        context.conversation_flow or "",
                        str(context.user_preferences),
                        str(context.environmental_factors),
                        str(context.metadata)
                    ]).lower()
                    
                    if query.lower() in searchable_text:
                        matching_contexts.append(context)
                        
            except Exception as e:
                self.logger.warning(f"Error searching contexts in session {session_id}: {e}", exc_info=True)
                continue
        
        # Sort by timestamp, newest first
        matching_contexts.sort(key=lambda c: c.timestamp or "", reverse=True)
        
        # Limit results
        limit = self.config.runtime.default_search_limit
        return matching_contexts[:limit]
    
    async def get_context_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get context statistics for session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Statistics dictionary
        """
        contexts = await self.get_context_history(user_id, session_id)
        
        stats = {
            'context_count': len(contexts),
            'context_types': {},
            'latest_snapshot': None,
            'oldest_snapshot': None
        }
        
        if contexts:
            # Count context types
            for context in contexts:
                if context.context_type:
                    context_type = context.context_type
                    stats['context_types'][context_type] = stats['context_types'].get(context_type, 0) + 1
            
            # Get latest and oldest snapshots
            stats['latest_snapshot'] = contexts[0].snapshot_id  # Already sorted newest first
            stats['oldest_snapshot'] = contexts[-1].snapshot_id
        
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