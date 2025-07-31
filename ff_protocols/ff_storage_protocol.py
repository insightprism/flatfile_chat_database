"""
Storage protocol interface.

Defines the contract for storage operations that all storage implementations
must follow.
"""

from typing import Protocol, Optional, List, Dict, Any, AsyncIterator
from datetime import datetime


class FFStorageProtocol(Protocol):
    """
    Storage interface for high-level storage operations.
    
    This protocol defines the contract that all storage managers must implement.
    """
    
    async def initialize(self) -> bool:
        """
        Initialize the storage system.
        
        Returns:
            True if initialization successful
        """
        ...
    
    # User operations
    async def create_user(self, user_id: str, profile_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new user with optional profile data.
        
        Args:
            user_id: Unique user identifier
            profile_data: Optional user profile information
            
        Returns:
            True if successful
        """
        ...
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Profile data or None if not found
        """
        ...
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Update user profile data.
        
        Args:
            user_id: User identifier
            profile_data: Updated profile data
            
        Returns:
            True if successful
        """
        ...
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user and all associated data.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        ...
    
    # Session operations
    async def create_session(self, user_id: str, session_id: Optional[str] = None,
                           title: str = "New Chat") -> str:
        """
        Create a new chat session.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID (generated if not provided)
            title: Session title
            
        Returns:
            Session ID
        """
        ...
    
    async def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session data or None
        """
        ...
    
    async def update_session(self, user_id: str, session_id: str, 
                           updates: Dict[str, Any]) -> bool:
        """
        Update session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            updates: Fields to update
            
        Returns:
            True if successful
        """
        ...
    
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete a session and all messages.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        ...
    
    async def list_sessions(self, user_id: str, limit: int = 50, 
                          offset: int = 0) -> List[Dict[str, Any]]:
        """
        List user's sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum results
            offset: Skip this many results
            
        Returns:
            List of session metadata
        """
        ...
    
    # Message operations
    async def add_message(self, user_id: str, session_id: str,
                         message: Dict[str, Any]) -> str:
        """
        Add a message to a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message: Message data
            
        Returns:
            Message ID
        """
        ...
    
    async def get_messages(self, user_id: str, session_id: str,
                          limit: int = 100, offset: int = 0,
                          reverse: bool = False) -> List[Dict[str, Any]]:
        """
        Get messages from a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum messages
            offset: Skip this many messages
            reverse: Get newest first if True
            
        Returns:
            List of messages
        """
        ...
    
    async def stream_messages(self, user_id: str, session_id: str,
                            start_after: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream messages from a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            start_after: Start after this message ID
            
        Yields:
            Message dictionaries
        """
        ...
    
    async def delete_message(self, user_id: str, session_id: str,
                           message_id: str) -> bool:
        """
        Delete a specific message.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message_id: Message identifier
            
        Returns:
            True if successful
        """
        ...
    
    # Document operations
    async def store_document(self, user_id: str, session_id: str,
                           document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a document.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_path: Path to document
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        ...
    
    async def get_document(self, user_id: str, session_id: str,
                         document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            Document data or None
        """
        ...
    
    async def delete_document(self, user_id: str, session_id: str,
                            document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        ...
    
    # Context operations
    async def save_context(self, user_id: str, session_id: str,
                         context: Dict[str, Any]) -> bool:
        """
        Save situational context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Context data
            
        Returns:
            True if successful
        """
        ...
    
    async def get_context(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current situational context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Context data or None
        """
        ...
    
    # Panel operations
    async def create_panel(self, user_id: str, panel_type: str,
                         personas: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a panel session.
        
        Args:
            user_id: User identifier
            panel_type: Type of panel
            personas: List of persona IDs
            metadata: Optional metadata
            
        Returns:
            Panel ID
        """
        ...
    
    async def get_panel(self, user_id: str, panel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get panel data.
        
        Args:
            user_id: User identifier
            panel_id: Panel identifier
            
        Returns:
            Panel data or None
        """
        ...
    
    # Utility operations
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Statistics dictionary
        """
        ...
    
    async def cleanup_old_data(self, days_to_keep: int) -> Dict[str, int]:
        """
        Clean up data older than specified days.
        
        Args:
            days_to_keep: Keep data from this many days
            
        Returns:
            Cleanup statistics
        """
        ...