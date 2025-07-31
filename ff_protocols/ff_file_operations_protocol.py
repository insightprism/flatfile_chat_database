"""
File operations protocol interface.

Defines the contract for file operations with locking and atomic writes.
"""

from typing import Protocol, Optional, Union, List, Dict, Any, Callable
from pathlib import Path


class FFFileOperationsProtocol(Protocol):
    """
    File operations interface.
    
    All file operation implementations must follow this protocol.
    """
    
    async def read(self, path: Path, mode: str = 'rb',
                  encoding: str = 'utf-8') -> Optional[Union[str, bytes]]:
        """
        Read file contents with optional locking.
        
        Args:
            path: File path
            mode: Read mode ('r' for text, 'rb' for binary)
            encoding: Text encoding (used only for text mode)
            
        Returns:
            File contents or None if file doesn't exist
        """
        ...
    
    async def write(self, path: Path, data: Union[str, bytes],
                   mode: str = 'wb') -> bool:
        """
        Write file atomically with locking.
        
        Args:
            path: File path
            data: Data to write
            mode: Write mode ('w' for text, 'wb' for binary)
            
        Returns:
            True if successful
        """
        ...
    
    async def append(self, path: Path, data: Union[str, bytes]) -> bool:
        """
        Append to file with locking.
        
        Args:
            path: File path
            data: Data to append
            
        Returns:
            True if successful
        """
        ...
    
    async def delete(self, path: Path,
                    backup_callback: Optional[Callable[[Path], None]] = None) -> bool:
        """
        Delete file or directory with optional backup.
        
        Args:
            path: Path to delete
            backup_callback: Optional callback to backup before deletion
            
        Returns:
            True if successful
        """
        ...
    
    async def exists(self, path: Path) -> bool:
        """
        Check if path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if exists
        """
        ...
    
    async def is_file(self, path: Path) -> bool:
        """
        Check if path is a file.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a file
        """
        ...
    
    async def is_directory(self, path: Path) -> bool:
        """
        Check if path is a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a directory
        """
        ...
    
    async def create_directory(self, path: Path, parents: bool = True) -> bool:
        """
        Create directory.
        
        Args:
            path: Directory path
            parents: Create parent directories if needed
            
        Returns:
            True if successful
        """
        ...
    
    async def list_directory(self, path: Path, pattern: str = "*",
                           recursive: bool = False) -> List[Path]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: Search recursively
            
        Returns:
            List of matching paths
        """
        ...
    
    async def copy(self, source: Path, destination: Path) -> bool:
        """
        Copy file atomically.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful
        """
        ...
    
    async def move(self, source: Path, destination: Path) -> bool:
        """
        Move file atomically.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful
        """
        ...
    
    async def get_size(self, path: Path) -> int:
        """
        Get file or directory size.
        
        Args:
            path: Path to measure
            
        Returns:
            Size in bytes or -1 if error
        """
        ...
    
    async def get_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            path: File path
            
        Returns:
            Metadata dictionary including:
            - size: File size in bytes
            - created: Creation timestamp
            - modified: Modification timestamp
            - accessed: Access timestamp
            - is_file: Whether it's a file
            - is_directory: Whether it's a directory
            - permissions: File permissions
        """
        ...
    
    async def lock_file(self, path: Path, exclusive: bool = True,
                       timeout: float = 30.0) -> Any:
        """
        Acquire file lock.
        
        Args:
            path: File to lock
            exclusive: Exclusive lock if True, shared if False
            timeout: Maximum time to wait for lock
            
        Returns:
            Lock object (context manager)
        """
        ...
    
    async def cleanup_empty_directories(self, path: Path) -> None:
        """
        Remove empty directories up the tree.
        
        Args:
            path: Starting directory path
        """
        ...
    
    async def atomic_write_json(self, path: Path, data: Dict[str, Any],
                              indent: int = 2) -> bool:
        """
        Write JSON data atomically.
        
        Args:
            path: File path
            data: Data to write
            indent: JSON indentation
            
        Returns:
            True if successful
        """
        ...
    
    async def atomic_append_jsonl(self, path: Path, data: Dict[str, Any]) -> bool:
        """
        Append to JSONL file atomically.
        
        Args:
            path: File path
            data: Data to append
            
        Returns:
            True if successful
        """
        ...