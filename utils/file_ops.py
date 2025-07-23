"""
Core file operations with atomic writes and safety features.

This module provides the foundation for all file I/O operations in the
flatfile chat database system.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Union, Optional, Callable, List
import tempfile

from ..config import StorageConfig


class FileOperationError(Exception):
    """Base exception for file operations"""
    pass


class AtomicWriteError(FileOperationError):
    """Error during atomic write operation"""
    pass


async def atomic_write(
    path: Path,
    data: Union[str, bytes],
    config: StorageConfig,
    mode: str = 'wb'
) -> bool:
    """
    Perform atomic write operation using temp file and rename.
    
    This ensures that the file is either completely written or not written at all,
    preventing partial writes that could corrupt data.
    
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
    # Ensure parent directory exists
    if config.create_parent_directories:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    temp_path = path.with_suffix(path.suffix + config.atomic_write_temp_suffix)
    
    try:
        # Convert string to bytes if needed
        if mode == 'wb' and isinstance(data, str):
            data = data.encode('utf-8')
        elif mode == 'w' and isinstance(data, bytes):
            data = data.decode('utf-8')
        
        # Write to temp file
        def write_data():
            with open(temp_path, mode=mode) as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        
        await asyncio.get_event_loop().run_in_executor(None, write_data)
        
        # Atomic rename (on POSIX systems)
        temp_path.rename(path)
        return True
        
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if temp_path.exists():
                temp_path.unlink()
        except:
            pass
        raise AtomicWriteError(f"Failed to write {path}: {str(e)}")


async def safe_read(
    path: Path,
    mode: str = 'rb',
    encoding: str = 'utf-8'
) -> Optional[Union[str, bytes]]:
    """
    Safely read file contents.
    
    Args:
        path: File path to read
        mode: Read mode ('r' for text, 'rb' for binary)
        encoding: Text encoding (used only for text mode)
        
    Returns:
        File contents or None if file doesn't exist
    """
    if not path.exists():
        return None
        
    try:
        def read_data():
            with open(path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
                return f.read()
        
        return await asyncio.get_event_loop().run_in_executor(None, read_data)
    except Exception as e:
        # Log error but don't crash
        print(f"Error reading {path}: {e}")
        return None


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