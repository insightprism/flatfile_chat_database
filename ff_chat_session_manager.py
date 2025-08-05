"""
FF Chat Session Manager - Real-time Chat Session Management

Provides real-time session management and coordination using
existing FF storage as backend persistence.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass

from ff_core_storage_manager import FFCoreStorageManager
from ff_class_configs.ff_chat_session_config import FFChatSessionConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_protocol import FFChatSessionProtocol

logger = get_logger(__name__)


@dataclass
class FFChatSessionState:
    """Real-time state for an active chat session"""
    session_id: str
    user_id: str
    use_case: str
    last_activity: datetime
    message_count: int
    processing: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FFChatSessionManager(FFChatSessionProtocol):
    """
    Manages real-time chat sessions using FF storage as backend.
    
    Provides session lifecycle management, activity tracking,
    and real-time coordination while persisting to FF storage.
    """
    
    def __init__(self, 
                 ff_storage: FFCoreStorageManager,
                 config: FFChatSessionConfigDTO):
        """
        Initialize FF chat session manager.
        
        Args:
            ff_storage: FF storage manager instance
            config: Chat session configuration
        """
        self.ff_storage = ff_storage
        self.config = config
        self.logger = get_logger(__name__)
        
        # Active session tracking
        self.active_sessions: Dict[str, FFChatSessionState] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the chat session manager"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing FF Chat Session Manager...")
            
            # Start background cleanup task
            if self.config.cleanup_interval > 0:
                self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
            
            self._initialized = True
            logger.info("FF Chat Session Manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FF Chat Session Manager: {e}")
            return False
    
    async def register_session(self, chat_session) -> bool:
        """
        Register a new chat session for real-time management.
        
        Args:
            chat_session: FFChatSession instance
            
        Returns:
            True if registered successfully
        """
        try:
            session_state = FFChatSessionState(
                session_id=chat_session.session_id,
                user_id=chat_session.user_id,
                use_case=chat_session.use_case,
                last_activity=datetime.now(),
                message_count=0,
                metadata=chat_session.context.copy() if chat_session.context else {}
            )
            
            self.active_sessions[chat_session.session_id] = session_state
            self.session_locks[chat_session.session_id] = asyncio.Lock()
            
            logger.info(f"Registered FF chat session: {chat_session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register FF chat session: {e}")
            return False
    
    async def unregister_session(self, session_id: str) -> bool:
        """
        Unregister a chat session from real-time management.
        
        Args:
            session_id: Chat session identifier
            
        Returns:
            True if unregistered successfully
        """
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.session_locks:
                del self.session_locks[session_id]
            
            logger.info(f"Unregistered FF chat session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister FF chat session: {e}")
            return False
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session state.
        
        Args:
            session_id: Chat session identifier
            
        Returns:
            Session state dictionary or None if not found
        """
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            return None
        
        return {
            "session_id": session_state.session_id,
            "user_id": session_state.user_id,
            "use_case": session_state.use_case,
            "last_activity": session_state.last_activity.isoformat(),
            "message_count": session_state.message_count,
            "processing": session_state.processing,
            "metadata": session_state.metadata
        }
    
    async def update_session_activity(self, session_id: str) -> None:
        """
        Update last activity timestamp for a session.
        
        Args:
            session_id: Chat session identifier
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.now()
    
    async def increment_message_count(self, session_id: str) -> None:
        """
        Increment message count for a session.
        
        Args:
            session_id: Chat session identifier
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].message_count += 1
            await self.update_session_activity(session_id)
    
    async def set_session_processing(self, session_id: str, processing: bool) -> None:
        """
        Set processing status for a session.
        
        Args:
            session_id: Chat session identifier
            processing: Processing status
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].processing = processing
            await self.update_session_activity(session_id)
    
    async def get_session_lock(self, session_id: str) -> Optional[asyncio.Lock]:
        """
        Get lock for a session to prevent concurrent processing.
        
        Args:
            session_id: Chat session identifier
            
        Returns:
            Asyncio lock or None if session not found
        """
        return self.session_locks.get(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get active session IDs for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session IDs for the user
        """
        return [
            session_id for session_id, state in self.active_sessions.items()
            if state.user_id == user_id
        ]
    
    def get_sessions_by_use_case(self, use_case: str) -> List[str]:
        """
        Get active session IDs for a specific use case.
        
        Args:
            use_case: Use case identifier
            
        Returns:
            List of session IDs for the use case
        """
        return [
            session_id for session_id, state in self.active_sessions.items()
            if state.use_case == use_case
        ]
    
    def get_processing_sessions(self) -> List[str]:
        """
        Get session IDs that are currently processing.
        
        Returns:
            List of session IDs currently processing
        """
        return [
            session_id for session_id, state in self.active_sessions.items()
            if state.processing
        ]
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get overall session statistics.
        
        Returns:
            Statistics dictionary
        """
        current_time = datetime.now()
        total_sessions = len(self.active_sessions)
        processing_sessions = len(self.get_processing_sessions())
        
        # Calculate activity statistics
        recent_activity_threshold = current_time - timedelta(minutes=5)
        recent_sessions = sum(
            1 for state in self.active_sessions.values()
            if state.last_activity > recent_activity_threshold
        )
        
        # Calculate use case distribution
        use_case_counts = {}
        for state in self.active_sessions.values():
            use_case_counts[state.use_case] = use_case_counts.get(state.use_case, 0) + 1
        
        # Calculate user distribution
        user_counts = {}
        for state in self.active_sessions.values():
            user_counts[state.user_id] = user_counts.get(state.user_id, 0) + 1
        
        return {
            "total_active_sessions": total_sessions,
            "processing_sessions": processing_sessions,
            "recent_activity_sessions": recent_sessions,
            "use_case_distribution": use_case_counts,
            "unique_users": len(user_counts),
            "average_sessions_per_user": total_sessions / len(user_counts) if user_counts else 0,
            "cleanup_interval": self.config.cleanup_interval,
            "session_timeout": self.config.session_timeout
        }
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to cleanup inactive sessions"""
        while self._initialized:
            try:
                current_time = datetime.now()
                inactive_sessions = []
                
                for session_id, state in self.active_sessions.items():
                    time_since_activity = current_time - state.last_activity
                    if time_since_activity > timedelta(seconds=self.config.session_timeout):
                        inactive_sessions.append(session_id)
                
                # Cleanup inactive sessions
                for session_id in inactive_sessions:
                    await self.unregister_session(session_id)
                    logger.info(f"Cleaned up inactive FF chat session: {session_id}")
                
                # Wait before next cleanup cycle
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                logger.error(f"Error in FF chat session cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def force_cleanup_user_sessions(self, user_id: str) -> int:
        """
        Force cleanup of all sessions for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of sessions cleaned up
        """
        user_sessions = self.get_user_sessions(user_id)
        count = 0
        
        for session_id in user_sessions:
            await self.unregister_session(session_id)
            count += 1
        
        logger.info(f"Force cleaned up {count} sessions for user {user_id}")
        return count
    
    async def force_cleanup_processing_sessions(self) -> int:
        """
        Force cleanup of sessions stuck in processing state.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        stuck_sessions = []
        
        for session_id, state in self.active_sessions.items():
            if state.processing:
                time_since_activity = current_time - state.last_activity
                if time_since_activity > timedelta(seconds=self.config.max_processing_time * 2):
                    stuck_sessions.append(session_id)
        
        count = 0
        for session_id in stuck_sessions:
            await self.set_session_processing(session_id, False)
            logger.warning(f"Reset stuck processing session: {session_id}")
            count += 1
        
        return count
    
    async def shutdown(self) -> None:
        """Shutdown the chat session manager"""
        logger.info("Shutting down FF Chat Session Manager...")
        
        self._initialized = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear active sessions
        self.active_sessions.clear()
        self.session_locks.clear()
        
        logger.info("FF Chat Session Manager shutdown complete")