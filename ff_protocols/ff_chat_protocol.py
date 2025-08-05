"""
FF Chat Protocols

Extended protocols for FF chat functionality following
existing FF protocol patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from ff_class_configs.ff_chat_entities_config import FFMessageDTO


class FFChatApplicationProtocol(ABC):
    """Protocol for FF chat application implementations"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the chat application.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def create_chat_session(self, 
                                  user_id: str, 
                                  use_case: str,
                                  title: Optional[str] = None,
                                  custom_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session.
        
        Args:
            user_id: User identifier
            use_case: Use case identifier
            title: Optional session title
            custom_config: Optional configuration overrides
            
        Returns:
            Chat session ID
        """
        pass
    
    @abstractmethod
    async def process_message(self, 
                              session_id: str, 
                              message: Union[str, Dict[str, Any]],
                              role: str = "user",
                              **kwargs) -> Dict[str, Any]:
        """
        Process a message in a chat session.
        
        Args:
            session_id: Chat session identifier
            message: Message content (string or dict)
            role: Message role (user, assistant, system)
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results with response content
        """
        pass
    
    @abstractmethod
    async def get_session_messages(self, 
                                   session_id: str,
                                   limit: Optional[int] = None,
                                   offset: int = 0) -> List[FFMessageDTO]:
        """
        Get messages from a chat session.
        
        Args:
            session_id: Chat session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of messages
        """
        pass
    
    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """
        Close a chat session.
        
        Args:
            session_id: Chat session identifier
        """
        pass
    
    @abstractmethod
    def list_use_cases(self) -> List[str]:
        """
        Get list of available use cases.
        
        Returns:
            List of use case identifiers
        """
        pass
    
    @abstractmethod
    async def search_messages(self, 
                              user_id: str,
                              query: str,
                              session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search messages across sessions.
        
        Args:
            user_id: User identifier
            query: Search query
            session_ids: Optional list of session IDs to search
            
        Returns:
            Search results
        """
        pass


class FFChatSessionProtocol(ABC):
    """Protocol for FF chat session management"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the session manager.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def register_session(self, session) -> bool:
        """
        Register a chat session for management.
        
        Args:
            session: Chat session object
            
        Returns:
            True if registration successful
        """
        pass
    
    @abstractmethod
    async def unregister_session(self, session_id: str) -> bool:
        """
        Unregister a chat session.
        
        Args:
            session_id: Chat session identifier
            
        Returns:
            True if unregistration successful
        """
        pass
    
    @abstractmethod
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session state.
        
        Args:
            session_id: Chat session identifier
            
        Returns:
            Session state or None if not found
        """
        pass
    
    @abstractmethod
    async def update_session_activity(self, session_id: str) -> None:
        """
        Update session activity timestamp.
        
        Args:
            session_id: Chat session identifier
        """
        pass
    
    @abstractmethod
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        pass
    
    @abstractmethod
    def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get active session IDs for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session IDs for the user
        """
        pass


class FFChatUseCaseProtocol(ABC):
    """Protocol for FF use case management"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the use case manager.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def is_use_case_supported(self, use_case: str) -> bool:
        """
        Check if use case is supported.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            True if supported
        """
        pass
    
    @abstractmethod
    async def process_message(self, 
                              session,
                              message: FFMessageDTO, 
                              **kwargs) -> Dict[str, Any]:
        """
        Process message according to use case.
        
        Args:
            session: Chat session object
            message: Message to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results
        """
        pass
    
    @abstractmethod
    def list_use_cases(self) -> List[str]:
        """
        List available use cases.
        
        Returns:
            List of use case identifiers
        """
        pass
    
    @abstractmethod
    async def get_use_case_config(self, use_case: str) -> Dict[str, Any]:
        """
        Get configuration for a specific use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Use case configuration
        """
        pass
    
    @abstractmethod
    async def get_use_case_info(self, use_case: str) -> Dict[str, Any]:
        """
        Get detailed information about a use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Use case information
        """
        pass