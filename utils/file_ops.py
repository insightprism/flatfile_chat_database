"""
Core file operations with atomic writes and safety features.

This module provides the foundation for all file I/O operations in the
flatfile chat database system.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Union, Optional, Callable, List, Dict, Any
import tempfile
import platform
import time
from datetime import datetime

from ..config import StorageConfig


class FileOperationError(Exception):
    """Base exception for file operations"""
    pass


class AtomicWriteError(FileOperationError):
    """Error during atomic write operation"""
    pass


class LockTimeoutError(FileOperationError):
    """Error when unable to acquire lock within timeout"""
    pass


# Platform-specific imports for file locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl


class FileLock:
    """Cross-platform file lock implementation"""
    
    def __init__(self, path: Path, timeout: float = 30.0):
        """
        Initialize file lock.
        
        Args:
            path: Path to lock (will use .lock suffix)
            timeout: Maximum time to wait for lock in seconds
        """
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + '.lock')
        self.timeout = timeout
        self.lock_file = None
        self.is_windows = platform.system() == 'Windows'
        self.acquired_at = None
        
    async def acquire(self, exclusive: bool = True):
        """
        Acquire file lock with exponential backoff.
        
        Args:
            exclusive: True for exclusive lock, False for shared lock
            
        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout
        """
        start_time = time.time()
        retry_delay = 0.01  # Start with 10ms
        
        while True:
            try:
                # Ensure lock directory exists
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Try to acquire lock
                self.lock_file = open(self.lock_path, 'a+b')
                
                if self.is_windows:
                    # Windows file locking
                    msvcrt.locking(self.lock_file.fileno(),
                                  msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
                else:
                    # Unix/Linux file locking
                    fcntl.flock(self.lock_file.fileno(),
                               fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH | fcntl.LOCK_NB)
                
                self.acquired_at = datetime.now()
                return True
                
            except (IOError, OSError):
                # Lock is held by another process
                if self.lock_file:
                    self.lock_file.close()
                    self.lock_file = None
                
                # Check timeout
                if time.time() - start_time > self.timeout:
                    raise LockTimeoutError(f"Could not acquire lock on {self.path} within {self.timeout}s")
                
                # Exponential backoff
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 1.0)  # Cap at 1 second
    
    async def release(self):
        """Release file lock"""
        if self.lock_file:
            try:
                if self.is_windows:
                    # Windows unlock
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    # Unix/Linux unlock
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                
                self.lock_file.close()
                self.lock_file = None
                
                # Try to remove lock file (may fail if another process is waiting)
                try:
                    self.lock_path.unlink()
                except:
                    pass
                    
            except Exception:
                # Ignore errors during release
                pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()


class FileOperationManager:
    """
    Centralized manager for all file operations with locking support.
    
    This class provides thread-safe file operations with optional locking
    to prevent concurrent access issues.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize file operation manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self._locks: Dict[Path, FileLock] = {}
        self._lock_pool = asyncio.Lock()
        
    async def _get_lock(self, path: Path) -> FileLock:
        """
        Get or create a lock for a path.
        
        Args:
            path: File path to lock
            
        Returns:
            FileLock instance
        """
        # Normalize path to avoid duplicate locks
        canonical_path = path.resolve()
        
        async with self._lock_pool:
            if canonical_path not in self._locks:
                lock_timeout = getattr(self.config, 'lock_timeout_seconds', 30.0)
                self._locks[canonical_path] = FileLock(canonical_path, timeout=lock_timeout)
            
            return self._locks[canonical_path]
    
    async def execute(self, 
                     operation: str,
                     path: Path,
                     data: Optional[Union[str, bytes]] = None,
                     mode: str = 'exclusive',
                     **kwargs) -> Any:
        """
        Execute a file operation with optional locking.
        
        Args:
            operation: Operation type ('read', 'write', 'append', 'delete')
            path: File path
            data: Data for write operations
            mode: Lock mode ('exclusive', 'shared', 'none')
            **kwargs: Additional arguments for specific operations
            
        Returns:
            Operation result
        """
        # Check if locking is enabled
        if not getattr(self.config, 'enable_file_locking', True) or mode == 'none':
            # Execute without locking
            return await self._execute_unlocked(operation, path, data, **kwargs)
        
        # Get lock for this path
        lock = await self._get_lock(path)
        
        # Acquire appropriate lock
        exclusive = (mode == 'exclusive' or operation in ['write', 'append', 'delete'])
        await lock.acquire(exclusive=exclusive)
        
        try:
            return await self._execute_unlocked(operation, path, data, **kwargs)
        finally:
            await lock.release()
    
    async def _execute_unlocked(self,
                               operation: str,
                               path: Path,
                               data: Optional[Union[str, bytes]] = None,
                               **kwargs) -> Any:
        """
        Execute operation without locking.
        
        This method contains the actual file operation logic.
        """
        if operation == 'read':
            return await self._read_internal(path, **kwargs)
        elif operation == 'write':
            return await self._write_internal(path, data, **kwargs)
        elif operation == 'append':
            return await self._append_internal(path, data, **kwargs)
        elif operation == 'delete':
            return await self._delete_internal(path, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _read_internal(self, path: Path, read_mode: str = 'rb', encoding: str = 'utf-8') -> Optional[Union[str, bytes]]:
        """Internal read implementation"""
        if not path.exists():
            return None
            
        try:
            def read_data():
                with open(path, mode=read_mode, encoding=encoding if 'b' not in read_mode else None) as f:
                    return f.read()
            
            return await asyncio.get_event_loop().run_in_executor(None, read_data)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
    
    async def _write_internal(self, path: Path, data: Union[str, bytes], write_mode: str = 'wb') -> bool:
        """Internal write implementation with atomic operation"""
        # Ensure parent directory exists
        if self.config.create_parent_directories:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in same directory for atomic rename with unique name
        import uuid
        temp_suffix = f"{self.config.atomic_write_temp_suffix}.{uuid.uuid4().hex[:8]}"
        temp_path = path.with_suffix(path.suffix + temp_suffix)
        
        try:
            # Convert string to bytes if needed
            if write_mode == 'wb' and isinstance(data, str):
                data = data.encode('utf-8')
            elif write_mode == 'w' and isinstance(data, bytes):
                data = data.decode('utf-8')
            
            # Write to temp file
            def write_data():
                with open(temp_path, mode=write_mode) as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            
            await asyncio.get_event_loop().run_in_executor(None, write_data)
            
            # Atomic rename with retry for concurrent access
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    temp_path.rename(path)
                    return True
                except FileNotFoundError:
                    # Another process may have already renamed it
                    if not temp_path.exists() and path.exists():
                        # The file was successfully written by another process
                        return True
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.01 * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
        except Exception as e:
            # Clean up temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            raise AtomicWriteError(f"Failed to write {path}: {str(e)}")
    
    async def _append_internal(self, path: Path, data: Union[str, bytes]) -> bool:
        """Internal append implementation"""
        try:
            # Ensure parent directory exists
            if self.config.create_parent_directories:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Append to file
            def append_data():
                with open(path, 'ab') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            
            await asyncio.get_event_loop().run_in_executor(None, append_data)
            return True
            
        except Exception as e:
            print(f"Failed to append to {path}: {e}")
            return False
    
    async def _delete_internal(self, path: Path) -> bool:
        """Internal delete implementation"""
        if not path.exists():
            return True
            
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
            return False


# Global instance management
_manager_instance: Optional[FileOperationManager] = None
_manager_lock = asyncio.Lock()


async def get_file_operation_manager(config: StorageConfig) -> FileOperationManager:
    """
    Get or create the global file operation manager.
    
    Args:
        config: Storage configuration
        
    Returns:
        FileOperationManager instance
    """
    global _manager_instance
    
    async with _manager_lock:
        if _manager_instance is None:
            _manager_instance = FileOperationManager(config)
        return _manager_instance


async def atomic_write(
    path: Path,
    data: Union[str, bytes],
    config: StorageConfig,
    mode: str = 'wb'
) -> bool:
    """
    Perform atomic write operation using temp file and rename.
    
    This ensures that the file is either completely written or not written at all,
    preventing partial writes that could corrupt data. Now with file locking support.
    
    Args:
        path: Target file path
        data: Data to write (string or bytes)
        config: Storage configuration
        mode: File open mode ('w' for text, 'wb' for binary)
        
    Returns:
        True if successful
        
    Raises:
        AtomicWriteError: If write operation fails
    """
    manager = await get_file_operation_manager(config)
    return await manager.execute('write', path, data, mode='exclusive', write_mode=mode)


async def safe_read(
    path: Path,
    mode: str = 'rb',
    encoding: str = 'utf-8',
    config: Optional[StorageConfig] = None
) -> Optional[Union[str, bytes]]:
    """
    Safely read file contents with optional locking.
    
    Args:
        path: File path to read
        mode: Read mode ('r' for text, 'rb' for binary)
        encoding: Text encoding (used only for text mode)
        config: Storage configuration (if None, reads without locking)
        
    Returns:
        File contents or None if file doesn't exist
    """
    if config is None:
        # Backward compatibility - read without locking
        if not path.exists():
            return None
            
        try:
            def read_data():
                with open(path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
                    return f.read()
            
            return await asyncio.get_event_loop().run_in_executor(None, read_data)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
    
    # Use file operation manager with shared lock for reads
    manager = await get_file_operation_manager(config)
    return await manager.execute('read', path, mode='shared', read_mode=mode, encoding=encoding)


async def ensure_directory(path: Path) -> bool:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")
        return False


async def safe_delete(
    path: Path,
    config: StorageConfig,
    backup_callback: Optional[Callable[[Path], None]] = None
) -> bool:
    """
    Safely delete file or directory with optional backup.
    
    Args:
        path: Path to delete
        config: Storage configuration
        backup_callback: Optional callback to backup before deletion
        
    Returns:
        True if successful
    """
    if not path.exists():
        return True
        
    try:
        # Backup if configured
        if config.backup_before_delete and backup_callback:
            await asyncio.get_event_loop().run_in_executor(
                None, backup_callback, path
            )
        
        # Delete file or directory
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
            
        # Clean up empty parent directories if configured
        if config.auto_cleanup_empty_directories:
            await cleanup_empty_directories(path.parent)
            
        return True
        
    except Exception as e:
        print(f"Failed to delete {path}: {e}")
        return False


async def cleanup_empty_directories(path: Path) -> None:
    """
    Remove empty directories up the tree.
    
    Args:
        path: Starting directory path
    """
    try:
        current = path
        while current and current.exists():
            # Check if directory is empty
            if not any(current.iterdir()):
                current.rmdir()
                current = current.parent
            else:
                break
    except:
        # Stop on any error (like permission issues)
        pass


async def copy_file(
    source: Path,
    destination: Path,
    config: StorageConfig
) -> bool:
    """
    Copy file with atomic operation.
    
    Args:
        source: Source file path
        destination: Destination file path
        config: Storage configuration
        
    Returns:
        True if successful
    """
    if not source.exists():
        return False
        
    try:
        # Read source
        data = await safe_read(source, mode='rb')
        if data is None:
            return False
            
        # Write to destination atomically
        return await atomic_write(destination, data, config, mode='wb')
        
    except Exception as e:
        print(f"Failed to copy {source} to {destination}: {e}")
        return False


async def move_file(
    source: Path,
    destination: Path,
    config: StorageConfig
) -> bool:
    """
    Move file atomically.
    
    Args:
        source: Source file path
        destination: Destination file path
        config: Storage configuration
        
    Returns:
        True if successful
    """
    # First try simple rename if on same filesystem
    try:
        if config.create_parent_directories:
            destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return True
    except:
        # Fall back to copy and delete
        if await copy_file(source, destination, config):
            return await safe_delete(source, config)
        return False


async def get_file_size(path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size or -1 if error
    """
    try:
        return path.stat().st_size
    except:
        return -1


async def file_exists(path: Path) -> bool:
    """
    Check if file exists (async wrapper for consistency).
    
    Args:
        path: File path
        
    Returns:
        True if file exists
    """
    return path.exists() and path.is_file()


async def directory_exists(path: Path) -> bool:
    """
    Check if directory exists (async wrapper for consistency).
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists
    """
    return path.exists() and path.is_dir()


async def list_files(
    directory: Path,
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (default: "*")
        recursive: Search recursively
        
    Returns:
        List of matching file paths
    """
    if not directory.exists():
        return []
        
    try:
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    except:
        return []


async def get_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory contents.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for path in directory.rglob('*'):
            if path.is_file():
                total += path.stat().st_size
    except:
        pass
    return total


async def atomic_append(
    path: Path,
    data: Union[str, bytes],
    config: StorageConfig
) -> bool:
    """
    Perform atomic append operation with file locking.
    
    This ensures that concurrent appends don't interleave and corrupt the file.
    Particularly important for JSONL files where each line must be complete.
    
    Args:
        path: Target file path
        data: Data to append (string or bytes)
        config: Storage configuration
        
    Returns:
        True if successful
    """
    manager = await get_file_operation_manager(config)
    return await manager.execute('append', path, data, mode='exclusive')