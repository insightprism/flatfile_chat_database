"""
User management functionality extracted from FFStorageManager.

Handles user profiles, creation, updates, and user-specific operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFUserProfileDTO
from backends import StorageBackend
from ff_utils import (
    ff_get_user_key, ff_get_profile_key, ff_sanitize_filename
)
from ff_utils.ff_logging import get_logger
from ff_utils.ff_validation import validate_user_id


class FFUserManager:
    """
    Manages user profiles and user-specific operations.
    
    Single responsibility: User data management and operations.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        """
        Initialize user manager.
        
        Args:
            config: Storage configuration
            backend: Storage backend for data operations
        """
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create new user with optional profile.
        
        Args:
            user_id: User identifier
            profile: Optional profile data
            
        Returns:
            True if successful
        """
        # Validate user ID
        validation_errors = validate_user_id(user_id, self.config)
        if validation_errors:
            self.logger.warning(f"Invalid user ID: {'; '.join(validation_errors)}")
            return False
        
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
        user_data_dir = self.config.storage.user_data_directory_name
        pattern = f"{user_data_dir}/*/"
        keys = await self.backend.list_keys("", pattern=pattern)
        
        users = []
        for key in keys:
            # Extract user ID from path: "users/user_id/"
            parts = key.rstrip('/').split('/')
            if len(parts) >= self.config.runtime.path_component_min_length:
                user_id = parts[self.config.runtime.user_id_path_index]  # Configurable index
                if user_id:
                    users.append(user_id)
        
        return sorted(list(set(users)))
    
    # === Helper Methods ===
    
    async def _write_json(self, key: str, data: Dict[str, Any]) -> bool:
        """Write JSON data using backend"""
        from ff_utils import ff_write_json
        json_path = self.base_path / key
        return await ff_write_json(json_path, data, self.config)
    
    async def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON data using backend"""
        from ff_utils import ff_read_json
        json_path = self.base_path / key
        return await ff_read_json(json_path, self.config)