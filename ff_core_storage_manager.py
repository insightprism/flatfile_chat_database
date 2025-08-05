"""
Core Storage Manager - Refactored architecture with single-responsibility managers.

This replaces the monolithic FFStorageManager with a composition of focused managers:
- FFUserManager: User profiles and operations
- FFSessionManager: Sessions and messages  
- FFDocumentManager: Document storage and retrieval
- FFContextManager: Situational context management
- FFPanelManager: Multi-persona panels and personas

Each manager has a single responsibility and can be tested/maintained independently.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, load_config
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFUserProfileDTO, FFDocumentDTO,
    FFSituationalContextDTO, FFPanelDTO, FFPersonaDTO, FFPanelMessageDTO, FFPanelInsightDTO
)
from backends import StorageBackend, FlatfileBackend
from ff_user_manager import FFUserManager
from ff_session_manager import FFSessionManager
from ff_document_manager import FFDocumentManager
from ff_context_manager import FFContextManager
from ff_panel_manager import FFPanelManager
from ff_utils.ff_logging import get_logger

# Import search and vector components (these would be refactored separately)
from ff_search_manager import FFSearchManager, FFSearchQueryDTO, FFSearchResultDTO
from ff_vector_storage_manager import FFVectorStorageManager
from ff_embedding_manager import FFEmbeddingManager


class FFCoreStorageManager:
    """
    Core storage manager that coordinates specialized managers.
    
    This class maintains the same external API as the original FFStorageManager
    but delegates operations to focused, single-responsibility managers.
    """
    
    def __init__(self, 
                 config: Optional[FFConfigurationManagerConfigDTO] = None,
                 backend: Optional[StorageBackend] = None):
        """
        Initialize core storage manager with specialized managers.
        
        Args:
            config: Storage configuration (uses default if not provided)
            backend: Storage backend (uses FlatfileBackend if not provided)
        """
        self.config = config or load_config()
        self.backend = backend or FlatfileBackend(self.config)
        self.base_path = Path(self.config.storage.base_path)
        self._initialized = False
        self.logger = get_logger(__name__)
        
        # Initialize specialized managers
        self.user_manager = FFUserManager(self.config, self.backend)
        self.session_manager = FFSessionManager(self.config, self.backend)
        self.document_manager = FFDocumentManager(self.config, self.backend)
        self.context_manager = FFContextManager(self.config, self.backend)
        self.panel_manager = FFPanelManager(self.config, self.backend)
        
        # Keep search and vector components (to be refactored separately)
        self._search_engine = None
        self._vector_storage = None
        self._embedding_engine = None
    
    async def initialize(self) -> bool:
        """
        Initialize storage manager and all components.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            # Initialize backend
            success = await self.backend.initialize()
            if not success:
                return False
            
            self._initialized = True
            self.logger.info("Core storage manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage manager: {e}", exc_info=True)
            return False
    
    # === User Management - Delegated to FFUserManager ===
    
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """Create new user - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.create_user(user_id, profile)
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.get_user_profile(user_id)
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.update_user_profile(user_id, updates)
    
    async def store_user_profile(self, profile: FFUserProfileDTO) -> bool:
        """Store user profile - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.store_user_profile(profile)
    
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.user_exists(user_id)
    
    async def list_users(self) -> List[str]:
        """List all users - delegated to user manager"""
        await self._ensure_initialized()
        return await self.user_manager.list_users()
    
    # === Session Management - Delegated to FFSessionManager ===
    
    async def create_session(self, user_id: str, session_name: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create session - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.create_session(user_id, session_name, metadata)
    
    async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool:
        """Add message - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.add_message(user_id, session_id, message)
    
    async def get_messages(self, user_id: str, session_id: str, 
                          limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]:
        """Get messages - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.get_messages(user_id, session_id, limit, offset)
    
    async def get_all_messages(self, user_id: str, session_id: str) -> List[FFMessageDTO]:
        """Get all messages - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.get_all_messages(user_id, session_id)
    
    async def get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]:
        """Get session - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.get_session(user_id, session_id)
    
    async def list_sessions(self, user_id: str, limit: Optional[int] = None, 
                           offset: int = 0) -> List[FFSessionDTO]:
        """List sessions - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.list_sessions(user_id, limit, offset)
    
    async def update_session_metadata(self, user_id: str, session_id: str, 
                                    updates: Dict[str, Any]) -> bool:
        """Update session metadata - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.update_session_metadata(user_id, session_id, updates)
    
    async def get_session_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get session statistics - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.get_session_statistics(user_id, session_id)
    
    async def session_exists(self, user_id: str, session_id: str) -> bool:
        """Check if session exists - delegated to session manager"""
        await self._ensure_initialized()
        return await self.session_manager.session_exists(user_id, session_id)
    
    # === Document Management - Delegated to FFDocumentManager ===
    
    async def store_document(self, user_id: str, session_id: str, filename: str, 
                           content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store document - delegated to document manager"""
        await self._ensure_initialized()
        return await self.document_manager.store_document(user_id, session_id, filename, content, metadata)
    
    async def get_document(self, user_id: str, session_id: str, doc_id: str) -> Optional[FFDocumentDTO]:
        """Get document - delegated to document manager"""
        await self._ensure_initialized()
        return await self.document_manager.get_document(user_id, session_id, doc_id)
    
    async def list_documents(self, user_id: str, session_id: str) -> List[FFDocumentDTO]:
        """List documents - delegated to document manager"""
        await self._ensure_initialized()
        return await self.document_manager.list_documents(user_id, session_id)
    
    async def update_document_analysis(self, user_id: str, session_id: str,
                                     doc_id: str, analysis: Dict[str, Any]) -> bool:
        """Update document analysis - delegated to document manager"""
        await self._ensure_initialized()
        return await self.document_manager.update_document_analysis(user_id, session_id, doc_id, analysis)
    
    async def delete_document(self, user_id: str, session_id: str, doc_id: str) -> bool:
        """Delete document - delegated to document manager"""
        await self._ensure_initialized()
        return await self.document_manager.delete_document(user_id, session_id, doc_id)
    
    # === Context Management - Delegated to FFContextManager ===
    
    async def store_context(self, user_id: str, session_id: str, 
                          context: FFSituationalContextDTO) -> str:
        """Store context - delegated to context manager"""
        await self._ensure_initialized()
        return await self.context_manager.store_context(user_id, session_id, context)
    
    async def get_context(self, user_id: str, session_id: str, 
                         snapshot_id: Optional[str] = None) -> Optional[FFSituationalContextDTO]:
        """Get context - delegated to context manager"""
        await self._ensure_initialized()
        return await self.context_manager.get_context(user_id, session_id, snapshot_id)
    
    async def get_context_history(self, user_id: str, session_id: str, 
                                limit: Optional[int] = None) -> List[FFSituationalContextDTO]:
        """Get context history - delegated to context manager"""
        await self._ensure_initialized()
        return await self.context_manager.get_context_history(user_id, session_id, limit)
    
    async def update_context(self, user_id: str, session_id: str, snapshot_id: str,
                           updates: Dict[str, Any]) -> bool:
        """Update context - delegated to context manager"""
        await self._ensure_initialized()
        return await self.context_manager.update_context(user_id, session_id, snapshot_id, updates)
    
    # === Panel Management - Delegated to FFPanelManager ===
    
    async def create_panel(self, user_id: str, personas: List[str], panel_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create panel - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.create_panel(user_id, personas, panel_name, metadata)
    
    async def add_panel_message(self, user_id: str, panel_id: str, 
                              message: FFPanelMessageDTO) -> bool:
        """Add panel message - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.add_panel_message(user_id, panel_id, message)
    
    async def get_panel_messages(self, user_id: str, panel_id: str, 
                               limit: Optional[int] = None, offset: int = 0) -> List[FFPanelMessageDTO]:
        """Get panel messages - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.get_panel_messages(user_id, panel_id, limit, offset)
    
    async def store_persona(self, persona: FFPersonaDTO, global_persona: bool = False) -> bool:
        """Store persona - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.store_persona(persona, global_persona)
    
    async def get_persona(self, persona_id: str, user_id: Optional[str] = None) -> Optional[FFPersonaDTO]:
        """Get persona - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.get_persona(persona_id, user_id)
    
    async def list_personas(self, user_id: Optional[str] = None, 
                          include_global: bool = True) -> List[FFPersonaDTO]:
        """List personas - delegated to panel manager"""
        await self._ensure_initialized()
        return await self.panel_manager.list_personas(user_id, include_global)
    
    # === Search and Vector Operations (Kept for backward compatibility) ===
    # These would be moved to separate search/vector managers in a future refactoring
    
    @property
    def search_engine(self) -> FFSearchManager:
        """Lazy-loaded search engine"""
        if self._search_engine is None:
            self._search_engine = FFSearchManager(self.config)
        return self._search_engine
    
    @property
    def vector_storage(self) -> FFVectorStorageManager:
        """Lazy-loaded vector storage"""
        if self._vector_storage is None:
            self._vector_storage = FFVectorStorageManager(self.config)
        return self._vector_storage
    
    @property
    def embedding_engine(self) -> FFEmbeddingManager:
        """Lazy-loaded embedding engine"""
        if self._embedding_engine is None:
            self._embedding_engine = FFEmbeddingManager(self.config)
        return self._embedding_engine
    
    async def search_messages(self, user_id: str, query: str, 
                            session_ids: Optional[List[str]] = None) -> List[FFSearchResultDTO]:
        """Search messages - uses search engine"""
        await self._ensure_initialized()
        # Implementation would delegate to search engine
        # This maintains API compatibility while delegating
        pass
    
    # === Helper Methods ===
    
    async def _ensure_initialized(self) -> None:
        """Ensure manager is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def close(self) -> None:
        """Close all resources"""
        if self.backend:
            await self.backend.close()
        
        self._initialized = False
        self.logger.info("Core storage manager closed")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"FFCoreStorageManager(base_path={self.base_path}, initialized={self._initialized})"


# Backward compatibility - alias to maintain existing imports
FFStorageManager = FFCoreStorageManager