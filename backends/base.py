"""
Abstract backend interface for storage operations.

This allows the system to support different storage backends (flatfile, database, etc.)
while maintaining the same API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import StorageConfig


class StorageBackend(ABC):
    """
    Abstract backend interface for storage operations.
    
    All storage backends must implement this interface to ensure
    compatibility with the StorageManager.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize backend with configuration.
        
        Args:
            config: Storage configuration
        """
        self.config = config
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the storage backend.
        
        This might create directories, establish database connections, etc.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def read(self, key: str) -> Optional[bytes]:
        """
        Read data by key.
        
        Args:
            key: Storage key (e.g., file path for flatfile backend)
            
        Returns:
            Data as bytes or None if not found
        """
        pass
    
    @abstractmethod
    async def write(self, key: str, data: bytes) -> bool:
        """
        Write data with key.
        
        Args:
            key: Storage key
            data: Data to store as bytes
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def append(self, key: str, data: bytes) -> bool:
        """
        Append data to existing key.
        
        Args:
            key: Storage key
            data: Data to append as bytes
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists
        """
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys with prefix and optional pattern.
        
        Args:
            prefix: Key prefix to filter by
            pattern: Optional pattern to match (e.g., "*.json")
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    async def get_size(self, key: str) -> int:
        """
        Get size of data at key.
        
        Args:
            key: Storage key
            
        Returns:
            Size in bytes or -1 if not found
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the stored data.
        
        Args:
            key: Storage key
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    async def move(self, source_key: str, dest_key: str) -> bool:
        """
        Move data from one key to another.
        
        Default implementation uses read/write/delete.
        Backends can override for efficiency.
        
        Args:
            source_key: Source storage key
            dest_key: Destination storage key
            
        Returns:
            True if successful
        """
        # Read source
        data = await self.read(source_key)
        if data is None:
            return False
        
        # Write to destination
        if not await self.write(dest_key, data):
            return False
        
        # Delete source
        return await self.delete(source_key)
    
    async def copy(self, source_key: str, dest_key: str) -> bool:
        """
        Copy data from one key to another.
        
        Default implementation uses read/write.
        Backends can override for efficiency.
        
        Args:
            source_key: Source storage key
            dest_key: Destination storage key
            
        Returns:
            True if successful
        """
        # Read source
        data = await self.read(source_key)
        if data is None:
            return False
        
        # Write to destination
        return await self.write(dest_key, data)
    
    async def batch_read(self, keys: List[str]) -> Dict[str, Optional[bytes]]:
        """
        Read multiple keys at once.
        
        Default implementation reads sequentially.
        Backends can override for efficiency.
        
        Args:
            keys: List of storage keys
            
        Returns:
            Dictionary mapping keys to data (or None if not found)
        """
        results = {}
        for key in keys:
            results[key] = await self.read(key)
        return results
    
    async def batch_write(self, items: Dict[str, bytes]) -> Dict[str, bool]:
        """
        Write multiple key-value pairs at once.
        
        Default implementation writes sequentially.
        Backends can override for efficiency.
        
        Args:
            items: Dictionary of key-value pairs
            
        Returns:
            Dictionary mapping keys to success status
        """
        results = {}
        for key, data in items.items():
            results[key] = await self.write(key, data)
        return results
    
    async def batch_delete(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple keys at once.
        
        Default implementation deletes sequentially.
        Backends can override for efficiency.
        
        Args:
            keys: List of storage keys
            
        Returns:
            Dictionary mapping keys to success status
        """
        results = {}
        for key in keys:
            results[key] = await self.delete(key)
        return results
    
    async def close(self) -> None:
        """
        Close any resources held by the backend.
        
        This might close database connections, flush buffers, etc.
        Default implementation does nothing.
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the backend.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "healthy": True,
            "backend_type": self.__class__.__name__,
            "config": {
                "base_path": self.config.storage_base_path,
                "max_document_size": self.config.max_document_size_bytes
            }
        }
    
    def __repr__(self) -> str:
        """String representation of backend"""
        return f"{self.__class__.__name__}(base_path={self.config.storage_base_path})"