"""
JSON and JSONL utilities for the flatfile chat database.

Provides functions for reading and writing JSON files and handling
JSONL (JSON Lines) format for streaming data like messages.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator
import aiofiles

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_utils.ff_file_ops import ff_atomic_write, ff_atomic_append, ff_safe_read


class FFJSONError(Exception):
    """JSON operation error"""
    pass


async def ff_write_json(
    path: Path,
    data: Dict[str, Any],
    config: FFConfigurationManagerConfigDTO,
    pretty: bool = True
) -> bool:
    """
    Write dictionary to JSON file atomically.
    
    Args:
        path: Target file path
        data: Dictionary to write
        config: Storage configuration
        pretty: Use pretty printing with indentation
        
    Returns:
        True if successful
        
    Raises:
        FFJSONError: If serialization fails
    """
    try:
        # Serialize to JSON
        if pretty:
            json_str = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
        else:
            json_str = json.dumps(data, ensure_ascii=False)
        
        # Write atomically
        return await ff_atomic_write(path, json_str, config, mode='w')
        
    except (TypeError, ValueError) as e:
        raise FFJSONError(f"Failed to serialize data: {str(e)}")


async def ff_read_json(
    path: Path,
    config: FFConfigurationManagerConfigDTO,
    default: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Read JSON file safely with file locking.
    
    Args:
        path: JSON file path
        config: Storage configuration
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed dictionary or default value
    """
    # Read file content with shared lock
    content = await ff_safe_read(path, mode='r', config=config)
    if content is None:
        return default
    
    try:
        data = json.loads(content)
        
        # Validate if configured
        if config.storage.validate_json_on_read and not isinstance(data, dict):
            print(f"Warning: Expected dict in {path}, got {type(data)}")
            return default
            
        return data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {path}: {e}")
        return default


async def ff_append_jsonl(
    path: Path,
    entry: Dict[str, Any],
    config: FFConfigurationManagerConfigDTO
) -> bool:
    """
    Append entry to JSONL file with file locking.
    
    JSONL format has one JSON object per line, which is perfect for
    append-only operations like message history. Now uses atomic append
    to prevent concurrent write corruption.
    
    Args:
        path: JSONL file path
        entry: Dictionary to append
        config: Storage configuration
        
    Returns:
        True if successful
    """
    try:
        # Serialize entry
        json_line = json.dumps(entry, ensure_ascii=False) + '\n'
        
        # Use atomic append with locking
        return await ff_atomic_append(path, json_line, config)
        
    except Exception as e:
        print(f"Failed to append to JSONL {path}: {e}")
        return False


async def ff_read_jsonl(
    path: Path,
    config: FFConfigurationManagerConfigDTO,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Read all entries from JSONL file.
    
    Args:
        path: JSONL file path
        config: Storage configuration
        limit: Maximum number of entries to read
        offset: Number of entries to skip
        
    Returns:
        List of parsed dictionaries
    """
    if not path.exists():
        return []
    
    entries = []
    
    try:
        async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
            line_num = 0
            async for line in f:
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Skip offset
                if line_num < offset:
                    line_num += 1
                    continue
                
                # Check limit
                if limit and len(entries) >= limit:
                    break
                
                try:
                    entry = json.loads(line)
                    if config.storage.validate_json_on_read and not isinstance(entry, dict):
                        print(f"Warning: Invalid JSONL entry at line {line_num + 1}")
                        continue
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSONL line {line_num + 1}: {e}")
                    continue
                
                line_num += 1
                
    except Exception as e:
        print(f"Error reading JSONL file {path}: {e}")
    
    return entries


async def ff_read_jsonl_paginated(
    path: Path,
    config: FFConfigurationManagerConfigDTO,
    page_size: Optional[int] = None,
    page: int = 0
) -> Dict[str, Any]:
    """
    Read JSONL file with pagination support.
    
    Args:
        path: JSONL file path
        config: Storage configuration
        page_size: Number of entries per page (default from config)
        page: Page number (0-based)
        
    Returns:
        Dictionary with entries and pagination metadata
    """
    page_size = page_size or config.message_pagination_default_limit
    offset = page * page_size
    
    # Get entries for current page
    entries = await ff_read_jsonl(path, config, limit=page_size, offset=offset)
    
    # Count total entries for pagination metadata
    total = await ff_count_jsonl_entries(path)
    
    return {
        "entries": entries,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_entries": total,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0,
            "has_next": offset + len(entries) < total,
            "has_previous": page > 0
        }
    }


async def ff_stream_jsonl(
    path: Path,
    config: FFConfigurationManagerConfigDTO,
    chunk_size: int = 100
) -> AsyncIterator[List[Dict[str, Any]]]:
    """
    Stream JSONL file in chunks for memory efficiency.
    
    Args:
        path: JSONL file path
        config: Storage configuration
        chunk_size: Number of entries per chunk
        
    Yields:
        Chunks of parsed entries
    """
    if not path.exists():
        return
    
    chunk = []
    
    try:
        async with aiofiles.open(path, mode='r', encoding='utf-8', 
                               buffering=config.storage.jsonl_read_buffer_size) as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    chunk.append(entry)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                        
                except json.JSONDecodeError:
                    continue
        
        # Yield remaining entries
        if chunk:
            yield chunk
            
    except Exception as e:
        print(f"Error streaming JSONL file {path}: {e}")


async def ff_count_jsonl_entries(path: Path) -> int:
    """
    Count entries in JSONL file efficiently.
    
    Args:
        path: JSONL file path
        
    Returns:
        Number of valid JSON entries
    """
    if not path.exists():
        return 0
    
    count = 0
    
    try:
        async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
            async for line in f:
                if line.strip():
                    count += 1
    except:
        pass
    
    return count


async def ff_update_jsonl_entry(
    path: Path,
    config: FFConfigurationManagerConfigDTO,
    entry_id: str,
    id_field: str,
    updates: Dict[str, Any]
) -> bool:
    """
    Update specific entry in JSONL file.
    
    Note: This requires rewriting the entire file, so use sparingly.
    
    Args:
        path: JSONL file path
        config: Storage configuration
        entry_id: ID of entry to update
        id_field: Name of the ID field in entries
        updates: Fields to update
        
    Returns:
        True if entry was found and updated
    """
    if not path.exists():
        return False
    
    # Read all entries
    entries = await ff_read_jsonl(path, config)
    
    # Find and update entry
    updated = False
    for entry in entries:
        if entry.get(id_field) == entry_id:
            entry.update(updates)
            updated = True
            break
    
    if not updated:
        return False
    
    # Write back all entries
    temp_path = path.with_suffix('.tmp')
    
    try:
        async with aiofiles.open(temp_path, mode='w', encoding='utf-8') as f:
            for entry in entries:
                await f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Atomic rename
        temp_path.rename(path)
        return True
        
    except Exception as e:
        print(f"Failed to update JSONL entry: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


async def ff_merge_json_files(
    files: List[Path],
    output_path: Path,
    config: FFConfigurationManagerConfigDTO
) -> bool:
    """
    Merge multiple JSON files into one.
    
    Args:
        files: List of JSON files to merge
        output_path: Output file path
        config: Storage configuration
        
    Returns:
        True if successful
    """
    merged = {}
    
    for file_path in files:
        data = await ff_read_json(file_path, config)
        if data:
            merged.update(data)
    
    return await ff_write_json(output_path, merged, config)