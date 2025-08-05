"""
Flatfile backend implementation for storage operations.

This backend stores data as files on the local filesystem, implementing
the FFStorageBackendBase interface.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime

from backends.ff_storage_backend_base import FFStorageBackendBase
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_utils import (
    ff_atomic_write, ff_atomic_append, ff_safe_read, ff_ensure_directory, ff_safe_delete,
    ff_file_exists, ff_directory_exists, ff_list_files, ff_get_file_size
)
from ff_utils.ff_logging import get_logger


class FFFlatfileStorageBackend(FFStorageBackendBase):
    """
    Flatfile storage backend implementation.
    
    Stores data as files on the filesystem with the key as the file path.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """
        Initialize flatfile backend.
        
        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def initialize(self) -> bool:
        """
        Initialize the storage backend by creating base directory.
        
        Returns:
            True if initialization successful
        """
        try:
            await ff_ensure_directory(self.base_path)
            
            # Create standard subdirectories from config
            subdirs = [
                self.config.panel.global_personas_directory,
                self.config.panel.panel_sessions_directory,
                self.config.storage.system_config_directory
            ]
            
            for subdir in subdirs:
                await ff_ensure_directory(self.base_path / subdir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize flatfile backend: {e}", exc_info=True)
            return False
    
    async def read(self, key: str) -> Optional[bytes]:
        """
        Read data by key (file path) with file locking.
        
        Args:
            key: Storage key (relative path from base)
            
        Returns:
            Data as bytes or None if not found
        """
        path = self._get_absolute_path(key)
        return await ff_safe_read(path, mode='rb', config=self.config)
    
    async def write(self, key: str, data: bytes) -> bool:
        """
        Write data with key (file path).
        
        Args:
            key: Storage key (relative path from base)
            data: Data to store as bytes
            
        Returns:
            True if successful
        """
        path = self._get_absolute_path(key)
        return await ff_atomic_write(path, data, self.config, mode='wb')
    
    async def append(self, key: str, data: bytes) -> bool:
        """
        Append data to existing key (file) with file locking.
        
        Args:
            key: Storage key (relative path from base)
            data: Data to append as bytes
            
        Returns:
            True if successful
        """
        path = self._get_absolute_path(key)
        
        # Use atomic append with locking
        return await ff_atomic_append(path, data, self.config)
    
    async def delete(self, key: str) -> bool:
        """
        Delete data by key (file or directory).
        
        Args:
            key: Storage key (relative path from base)
            
        Returns:
            True if successful
        """
        path = self._get_absolute_path(key)
        
        # Backup callback if needed
        backup_callback = None
        if self.config.backup_before_delete:
            backup_callback = self._create_backup
        
        return await ff_safe_delete(path, self.config, backup_callback)
    
    async def exists(self, key: str) -> bool:
        """
        Check if key (file or directory) exists.
        
        Args:
            key: Storage key (relative path from base)
            
        Returns:
            True if key exists
        """
        path = self._get_absolute_path(key)
        return path.exists()
    
    async def list_keys(self, prefix: str, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys (files) with prefix and optional pattern.
        
        Args:
            prefix: Key prefix to filter by (directory path)
            pattern: Optional glob pattern (e.g., "*.json")
            
        Returns:
            List of matching keys (relative paths)
        """
        prefix_path = self._get_absolute_path(prefix)
        
        if not prefix_path.exists():
            return []
        
        # Use pattern or default to all files
        glob_pattern = pattern or "*"
        
        # Get all matching files
        files = await ff_list_files(prefix_path, glob_pattern, recursive=True)
        
        # Convert to relative keys
        keys = []
        for file_path in files:
            try:
                # Get relative path from base
                relative = file_path.relative_to(self.base_path)
                keys.append(str(relative))
            except ValueError:
                # Skip files outside base path
                continue
        
        return sorted(keys)
    
    async def get_size(self, key: str) -> int:
        """
        Get size of data at key.
        
        Args:
            key: Storage key (relative path from base)
            
        Returns:
            Size in bytes or -1 if not found
        """
        path = self._get_absolute_path(key)
        return await ff_get_file_size(path)
    
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the stored data.
        
        Args:
            key: Storage key (relative path from base)
            
        Returns:
            Metadata dictionary or None if not found
        """
        path = self._get_absolute_path(key)
        
        if not path.exists():
            return None
        
        try:
            stat = path.stat()
            
            return {
                "key": key,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "permissions": oct(stat.st_mode)[-3:]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {key}: {e}", exc_info=True)
            return None
    
    async def move(self, source_key: str, dest_key: str) -> bool:
        """
        Move data from one key to another.
        
        Overrides base implementation for efficiency using OS rename.
        
        Args:
            source_key: Source storage key
            dest_key: Destination storage key
            
        Returns:
            True if successful
        """
        source_path = self._get_absolute_path(source_key)
        dest_path = self._get_absolute_path(dest_key)
        
        if not source_path.exists():
            return False
        
        try:
            # Ensure destination directory exists
            if self.config.create_parent_directories:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try atomic rename first
            source_path.rename(dest_path)
            return True
            
        except OSError:
            # Fall back to copy and delete if rename fails (e.g., across filesystems)
            return await super().move(source_key, dest_key)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the backend.
        
        Returns:
            Dictionary with health status information
        """
        base_health = await super().health_check()
        
        # Add flatfile-specific checks
        try:
            # Check if base directory is accessible
            can_read = os.access(self.base_path, os.R_OK)
            can_write = os.access(self.base_path, os.W_OK)
            
            # Get disk usage
            stat = os.statvfs(self.base_path)
            free_space = stat.f_bavail * stat.f_frsize
            total_space = stat.f_blocks * stat.f_frsize
            used_percentage = ((total_space - free_space) / total_space) * 100
            
            base_health.update({
                "base_path_exists": self.base_path.exists(),
                "can_read": can_read,
                "can_write": can_write,
                "disk_usage": {
                    "free_bytes": free_space,
                    "total_bytes": total_space,
                    "used_percentage": round(used_percentage, 2)
                }
            })
            
            # Consider unhealthy if disk is too full
            if used_percentage > 90:
                base_health["healthy"] = False
                base_health["warnings"] = base_health.get("warnings", []) + ["Disk usage above 90%"]
            
        except Exception as e:
            base_health["healthy"] = False
            base_health["errors"] = base_health.get("errors", []) + [str(e)]
        
        return base_health
    
    def _get_absolute_path(self, key: str) -> Path:
        """
        Convert storage key to absolute path.
        
        Args:
            key: Storage key (relative path)
            
        Returns:
            Absolute path
        """
        # Ensure key doesn't escape base directory
        clean_key = str(Path(key))
        if clean_key.startswith('..'):
            raise ValueError(f"Invalid key: {key}")
        
        return self.base_path / clean_key
    
    def _create_backup(self, path: Path) -> None:
        """
        Create backup of file or directory before deletion.
        
        Args:
            path: Path to backup
        """
        if not path.exists():
            return
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.name}.backup_{timestamp}"
        backup_path = path.parent / backup_name
        
        try:
            if path.is_file():
                import shutil
                shutil.copy2(path, backup_path)
            else:
                import shutil
                shutil.copytree(path, backup_path)
                
            self.logger.info(f"Created backup: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create backup of {path}: {e}", exc_info=True)