"""
Panel management functionality extracted from FFStorageManager.

Handles multi-persona panel creation, message storage, and panel-specific operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import (
    FFPanelDTO, FFPersonaDTO, FFPanelMessageDTO, FFPanelInsightDTO
)
from backends import StorageBackend
from ff_utils import (
    ff_get_panel_path, ff_get_global_personas_path, ff_get_user_personas_path,
    ff_generate_panel_id, ff_append_jsonl, ff_read_jsonl, ff_write_json, ff_read_json,
    ff_build_panel_file_paths
)
from ff_utils.ff_logging import get_logger
from ff_utils.ff_validation import validate_user_id, validate_panel_name, validate_persona_name


class FFPanelManager:
    """
    Manages multi-persona panels and panel-specific operations.
    
    Single responsibility: Panel and persona data management.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        """
        Initialize panel manager.
        
        Args:
            config: Storage configuration
            backend: Storage backend for data operations
        """
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def create_panel(self, user_id: str, personas: List[str], panel_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create multi-persona panel.
        
        Args:
            user_id: User identifier
            personas: List of persona IDs
            panel_name: Name for the panel
            metadata: Optional panel metadata
            
        Returns:
            Panel ID or empty string if failed
        """
        # Validate inputs
        user_errors = validate_user_id(user_id, self.config)
        panel_errors = validate_panel_name(panel_name, self.config)
        
        if user_errors or panel_errors:
            all_errors = user_errors + panel_errors
            self.logger.warning(f"Invalid panel creation inputs: {'; '.join(all_errors)}")
            return ""
        
        # Validate personas count
        if len(personas) > self.config.panel.max_personas_per_panel:
            self.logger.warning(f"Too many personas: {len(personas)} > {self.config.panel.max_personas_per_panel}")
            return ""
        
        # Generate panel ID
        panel_id = ff_generate_panel_id(self.config)
        
        # Create panel object
        panel = FFPanelDTO(
            panel_id=panel_id,
            user_id=user_id,
            personas=personas,
            title=panel_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Get panel paths
        panel_files = await ff_build_panel_file_paths(
            self.base_path, user_id, panel_id, self.config
        )
        
        # Save panel metadata
        panel_key = str(panel_files["panel_metadata"].relative_to(self.base_path))
        if await self._write_json(panel_key, panel.to_dict()):
            return panel_id
        return ""
    
    async def add_panel_message(self, user_id: str, panel_id: str, 
                              message: FFPanelMessageDTO) -> bool:
        """
        Add message to panel.
        
        Args:
            user_id: User identifier
            panel_id: Panel identifier
            message: Panel message to add
            
        Returns:
            True if successful
        """
        # Get panel message file path
        panel_files = await ff_build_panel_file_paths(
            self.base_path, user_id, panel_id, self.config
        )
        
        messages_path = panel_files["messages"]
        
        # Add message to JSONL file
        success = await ff_append_jsonl(messages_path, message.to_dict(), self.config)
        
        if success:
            # Update panel timestamp
            await self._update_panel_timestamp(user_id, panel_id)
        
        return success
    
    async def get_panel_messages(self, user_id: str, panel_id: str, 
                               limit: Optional[int] = None, offset: int = 0) -> List[FFPanelMessageDTO]:
        """
        Retrieve messages from panel.
        
        Args:
            user_id: User identifier
            panel_id: Panel identifier
            limit: Maximum messages to return
            offset: Number of messages to skip
            
        Returns:
            List of panel messages
        """
        # Get panel messages file path
        panel_files = await ff_build_panel_file_paths(
            self.base_path, user_id, panel_id, self.config
        )
        
        messages_path = panel_files["messages"]
        
        # Read JSONL data
        messages_data = await ff_read_jsonl(messages_path, self.config, limit=limit, offset=offset)
        
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
        Save panel insight.
        
        Args:
            panel_id: Panel identifier
            insight: Insight to save
            
        Returns:
            True if successful
        """
        # This would need more context about where insights are stored
        # For now, implement basic functionality
        try:
            insight_extension = self.config.runtime.insight_file_extension
            insight_key = f"panels/{panel_id}/insights/{insight.insight_id}{insight_extension}"
            return await self._write_json(insight_key, insight.to_dict())
        except Exception as e:
            self.logger.error(f"Failed to save panel insight: {e}", exc_info=True)
            return False
    
    async def get_panel(self, user_id: str, panel_id: str) -> Optional[FFPanelDTO]:
        """
        Retrieve panel metadata.
        
        Args:
            user_id: User identifier
            panel_id: Panel identifier
            
        Returns:
            Panel object or None
        """
        panel_files = await ff_build_panel_file_paths(
            self.base_path, user_id, panel_id, self.config
        )
        
        panel_key = str(panel_files["panel_metadata"].relative_to(self.base_path))
        panel_data = await self._read_json(panel_key)
        
        if panel_data:
            return FFPanelDTO.from_dict(panel_data)
        return None
    
    async def list_panels(self, user_id: str) -> List[FFPanelDTO]:
        """
        List user panels.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of panels
        """
        # Get panel directory pattern
        panel_pattern = f"{self.config.panel.panel_sessions_directory}/{user_id}/*/panel.json"
        panel_keys = await self.backend.list_keys("", pattern=panel_pattern)
        
        panels = []
        for panel_key in panel_keys:
            panel_data = await self._read_json(panel_key)
            if panel_data:
                try:
                    panel = FFPanelDTO.from_dict(panel_data)
                    panels.append(panel)
                except Exception as e:
                    self.logger.error(f"Failed to parse panel: {e}", exc_info=True)
                    continue
        
        # Sort by created_at descending
        panels.sort(key=lambda p: p.created_at or "", reverse=True)
        return panels
    
    async def store_persona(self, persona: FFPersonaDTO, global_persona: bool = False) -> bool:
        """
        Store persona (global or user-specific).
        
        Args:
            persona: Persona to store
            global_persona: Whether this is a global persona
            
        Returns:
            True if successful
        """
        if global_persona:
            personas_path = ff_get_global_personas_path(self.base_path, self.config)
        else:
            if not persona.user_id:
                self.logger.error("User ID required for user-specific persona")
                return False
            personas_path = ff_get_user_personas_path(self.base_path, persona.user_id, self.config)
        
        persona_extension = self.config.runtime.persona_file_extension
        persona_file = personas_path / f"{persona.persona_id}{persona_extension}"
        persona_key = str(persona_file.relative_to(self.base_path))
        
        return await self._write_json(persona_key, persona.to_dict())
    
    async def get_persona(self, persona_id: str, user_id: Optional[str] = None) -> Optional[FFPersonaDTO]:
        """
        Retrieve persona by ID.
        
        Args:
            persona_id: Persona identifier
            user_id: User identifier for user-specific personas
            
        Returns:
            Persona object or None
        """
        # Try user-specific first if user_id provided
        if user_id:
            user_personas_path = ff_get_user_personas_path(self.base_path, user_id, self.config)
            persona_extension = self.config.runtime.persona_file_extension
            user_persona_file = user_personas_path / f"{persona_id}{persona_extension}"
            user_persona_key = str(user_persona_file.relative_to(self.base_path))
            
            persona_data = await self._read_json(user_persona_key)
            if persona_data:
                return FFPersonaDTO.from_dict(persona_data)
        
        # Try global personas
        global_personas_path = ff_get_global_personas_path(self.base_path, self.config)
        persona_extension = self.config.runtime.persona_file_extension
        global_persona_file = global_personas_path / f"{persona_id}{persona_extension}"
        global_persona_key = str(global_persona_file.relative_to(self.base_path))
        
        persona_data = await self._read_json(global_persona_key)
        if persona_data:
            return FFPersonaDTO.from_dict(persona_data)
        
        return None
    
    async def list_personas(self, user_id: Optional[str] = None, 
                          include_global: bool = True) -> List[FFPersonaDTO]:
        """
        List available personas.
        
        Args:
            user_id: User identifier for user-specific personas
            include_global: Whether to include global personas
            
        Returns:
            List of personas
        """
        personas = []
        
        # Get user-specific personas
        if user_id:
            user_personas_path = ff_get_user_personas_path(self.base_path, user_id, self.config)
            persona_extension = self.config.runtime.persona_file_extension
            user_pattern = str(user_personas_path.relative_to(self.base_path) / f"*{persona_extension}")
            
            user_persona_keys = await self.backend.list_keys("", pattern=user_pattern)
            for key in user_persona_keys:
                persona_data = await self._read_json(key)
                if persona_data:
                    try:
                        personas.append(FFPersonaDTO.from_dict(persona_data))
                    except Exception as e:
                        self.logger.error(f"Failed to parse user persona: {e}", exc_info=True)
                        continue
        
        # Get global personas
        if include_global:
            global_personas_path = ff_get_global_personas_path(self.base_path, self.config)
            persona_extension = self.config.runtime.persona_file_extension
            global_pattern = str(global_personas_path.relative_to(self.base_path) / f"*{persona_extension}")
            
            global_persona_keys = await self.backend.list_keys("", pattern=global_pattern)
            for key in global_persona_keys:
                persona_data = await self._read_json(key)
                if persona_data:
                    try:
                        personas.append(FFPersonaDTO.from_dict(persona_data))
                    except Exception as e:
                        self.logger.error(f"Failed to parse global persona: {e}", exc_info=True)
                        continue
        
        return personas
    
    async def panel_exists(self, user_id: str, panel_id: str) -> bool:
        """
        Check if panel exists.
        
        Args:
            user_id: User identifier
            panel_id: Panel identifier
            
        Returns:
            True if panel exists
        """
        panel_files = await ff_build_panel_file_paths(
            self.base_path, user_id, panel_id, self.config
        )
        
        panel_key = str(panel_files["panel_metadata"].relative_to(self.base_path))
        return await self.backend.exists(panel_key)
    
    # === Helper Methods ===
    
    async def _update_panel_timestamp(self, user_id: str, panel_id: str) -> None:
        """Update panel's last updated timestamp"""
        panel = await self.get_panel(user_id, panel_id)
        if panel:
            panel.updated_at = datetime.now().isoformat()
            
            panel_files = await ff_build_panel_file_paths(
                self.base_path, user_id, panel_id, self.config
            )
            
            panel_key = str(panel_files["panel_metadata"].relative_to(self.base_path))
            await self._write_json(panel_key, panel.to_dict())
    
    async def _write_json(self, key: str, data: Dict[str, Any]) -> bool:
        """Write JSON data using backend"""
        json_path = self.base_path / key
        return await ff_write_json(json_path, data, self.config)
    
    async def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON data using backend"""
        json_path = self.base_path / key
        return await ff_read_json(json_path, self.config)