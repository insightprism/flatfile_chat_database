"""
Backend protocol interface.

Defines the low-level storage backend operations that all backend
implementations must provide.
"""

from typing import Protocol, Optional, List, Dict, Any
from pathlib import Path


class BackendProtocol(Protocol):
    """
    Low-level storage backend interface.
    
    All storage backends (flatfile, database, etc.) must implement this protocol.
    """
    
    async def initialize(self) -> bool:
        """
        Initialize the storage backend.
        
        This might create directories, establish database connections, etc.
        
        Returns:
            True if initialization successful
        """
        ...
    
    async def read(self, key: str) -> Optional[bytes]:
        """
        Read data by key.
        
        Args:
            key: Storage key (e.g., file path for flatfile backend)
            
        Returns:
            Data as bytes or None if not found
        """
        ...
    
    async def write(self, key: str, data: bytes) -> bool:
        """
        Write data with key.
        
        Args:
            key: Storage key
            data: Data to store as bytes
            
        Returns:
            True if successful
        """
        ...
    
    async def append(self, key: str, data: bytes) -> bool:
        """
        Append data to existing key.
        
        Args:
            key: Storage key
            data: Data to append as bytes
            
        Returns:
            True if successful
        """
        ...
    
    async def delete(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful
        """
        ...
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists
        """
        ...
    
    async def list_keys(self, prefix: str, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys with prefix and optional pattern.
        
        Args:
            prefix: Key prefix to filter by
            pattern: Optional pattern to match (e.g., "*.json")
            
        Returns:
            List of matching keys
        """
        ...
    
    async def get_size(self, key: str) -> int:
        """
        Get size of data at key.
        
        Args:
            key: Storage key
            
        Returns:
            Size in bytes or -1 if not found
        """
        ...
    
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the stored data.
        
        Args:
            key: Storage key
            
        Returns:
            Metadata dictionary or None if not found
        """
        ...
    
    async def move(self, source_key: str, dest_key: str) -> bool:
        """
        Move data from one key to another.
        
        Args:
            source_key: Source storage key
            dest_key: Destination storage key
            
        Returns:
            True if successful
        """
        ...
    
    async def copy(self, source_key: str, dest_key: str) -> bool:
        """
        Copy data from one key to another.
        
        Args:
            source_key: Source storage key
            dest_key: Destination storage key
            
        Returns:
            True if successful
        """
        ...
    
    async def batch_read(self, keys: List[str]) -> Dict[str, Optional[bytes]]:
        """
        Read multiple keys at once.
        
        Args:
            keys: List of storage keys
            
        Returns:
            Dictionary mapping keys to data (or None if not found)
        """
        ...
    
    async def batch_write(self, items: Dict[str, bytes]) -> Dict[str, bool]:
        """
        Write multiple key-value pairs at once.
        
        Args:
            items: Dictionary of key-value pairs
            
        Returns:
            Dictionary mapping keys to success status
        """
        ...
    
    async def batch_delete(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple keys at once.
        
        Args:
            keys: List of storage keys
            
        Returns:
            Dictionary mapping keys to success status
        """
        ...
    
    async def close(self) -> None:
        """
        Close any resources held by the backend.
        
        This might close database connections, flush buffers, etc.
        """
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the backend.
        
        Returns:
            Dictionary with health status information
        """
        ...