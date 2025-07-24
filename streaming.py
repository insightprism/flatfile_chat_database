"""
Streaming utilities for efficient handling of large data sets.

Provides streaming capabilities for messages, search results, and exports
to handle large sessions without loading everything into memory.
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass

from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.models import Message, Session, Document
from flatfile_chat_database.utils import read_jsonl_paginated, get_user_path


@dataclass
class StreamConfig:
    """Configuration for streaming operations"""
    chunk_size: int = 100
    buffer_size: int = 8192
    max_concurrent_streams: int = 5
    stream_timeout_seconds: int = 300


class MessageStreamer:
    """
    Streams messages from sessions for efficient memory usage.
    
    Useful for large sessions with thousands of messages.
    """
    
    def __init__(self, config: StorageConfig, stream_config: Optional[StreamConfig] = None):
        """
        Initialize message streamer.
        
        Args:
            config: Storage configuration
            stream_config: Streaming configuration
        """
        self.config = config
        self.stream_config = stream_config or StreamConfig()
        self.base_path = Path(config.storage_base_path)
    
    async def stream_messages(self, user_id: str, session_id: str,
                            start_offset: int = 0,
                            max_messages: Optional[int] = None) -> AsyncIterator[List[Message]]:
        """
        Stream messages in chunks from a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            start_offset: Starting message offset
            max_messages: Maximum messages to stream
            
        Yields:
            Chunks of Message objects
        """
        messages_path = (self.base_path / user_id / session_id / 
                        self.config.messages_filename)
        
        if not messages_path.exists():
            return
        
        current_offset = start_offset
        messages_yielded = 0
        
        while True:
            # Calculate chunk size
            chunk_size = self.stream_config.chunk_size
            if max_messages:
                remaining = max_messages - messages_yielded
                chunk_size = min(chunk_size, remaining)
            
            if chunk_size <= 0:
                break
            
            # Read next chunk
            page_data = await read_jsonl_paginated(
                messages_path, self.config,
                page_size=chunk_size,
                page=current_offset // chunk_size
            )
            
            if not page_data["entries"]:
                break
            
            # Convert to Message objects
            messages = []
            for entry in page_data["entries"]:
                try:
                    messages.append(Message.from_dict(entry))
                except Exception as e:
                    print(f"Error parsing message: {e}")
                    continue
            
            if messages:
                yield messages
                messages_yielded += len(messages)
            
            # Check if we've reached the end
            if not page_data["pagination"]["has_next"]:
                break
            
            current_offset += chunk_size
    
    async def stream_messages_reverse(self, user_id: str, session_id: str,
                                    limit: Optional[int] = None) -> AsyncIterator[List[Message]]:
        """
        Stream messages in reverse order (most recent first).
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum messages to stream
            
        Yields:
            Chunks of Message objects in reverse order
        """
        messages_path = (self.base_path / user_id / session_id / 
                        self.config.messages_filename)
        
        if not messages_path.exists():
            return
        
        # Count total messages first
        total_messages = await self._count_messages(messages_path)
        if total_messages == 0:
            return
        
        # Calculate starting position
        chunk_size = self.stream_config.chunk_size
        current_position = total_messages
        messages_yielded = 0
        
        while current_position > 0:
            # Calculate offset and size for this chunk
            offset = max(0, current_position - chunk_size)
            actual_chunk_size = current_position - offset
            
            if limit and messages_yielded + actual_chunk_size > limit:
                actual_chunk_size = limit - messages_yielded
                offset = current_position - actual_chunk_size
            
            # Read chunk
            page_data = await read_jsonl_paginated(
                messages_path, self.config,
                page_size=actual_chunk_size,
                page=offset // self.stream_config.chunk_size
            )
            
            if page_data["entries"]:
                # Convert and reverse
                messages = []
                for entry in reversed(page_data["entries"]):
                    try:
                        messages.append(Message.from_dict(entry))
                    except Exception:
                        continue
                
                if messages:
                    yield messages
                    messages_yielded += len(messages)
            
            current_position = offset
            
            if limit and messages_yielded >= limit:
                break
    
    async def parallel_stream_sessions(self, user_id: str, 
                                     session_ids: List[str]) -> AsyncIterator[Tuple[str, List[Message]]]:
        """
        Stream messages from multiple sessions in parallel.
        
        Args:
            user_id: User identifier
            session_ids: List of session IDs to stream
            
        Yields:
            Tuples of (session_id, message_chunk)
        """
        # Create streaming tasks for each session
        async def stream_session(session_id: str):
            async for chunk in self.stream_messages(user_id, session_id):
                yield (session_id, chunk)
        
        # Limit concurrent streams
        semaphore = asyncio.Semaphore(self.stream_config.max_concurrent_streams)
        
        async def bounded_stream(session_id: str):
            async with semaphore:
                async for item in stream_session(session_id):
                    yield item
        
        # Merge streams
        streams = [bounded_stream(sid) for sid in session_ids]
        
        # Use asyncio to merge the streams
        async def merge_streams():
            tasks = []
            for stream in streams:
                task = asyncio.create_task(self._consume_stream(stream))
                tasks.append(task)
            
            # Yield results as they come
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    result = await task
                    if result:
                        yield result
                    tasks.remove(task)
        
        async for item in merge_streams():
            yield item
    
    async def _count_messages(self, messages_path: Path) -> int:
        """Count total messages in a file"""
        count = 0
        try:
            with open(messages_path, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
        return count
    
    async def _consume_stream(self, stream):
        """Consume a single stream"""
        async for item in stream:
            return item
        return None


class ExportStreamer:
    """
    Streams data for export operations.
    
    Handles efficient export of large datasets without memory issues.
    """
    
    def __init__(self, config: StorageConfig, stream_config: Optional[StreamConfig] = None):
        """
        Initialize export streamer.
        
        Args:
            config: Storage configuration
            stream_config: Streaming configuration
        """
        self.config = config
        self.stream_config = stream_config or StreamConfig()
        self.base_path = Path(config.storage_base_path)
        self.message_streamer = MessageStreamer(config, stream_config)
    
    async def stream_session_export(self, user_id: str, session_id: str,
                                  include_documents: bool = True,
                                  include_context: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream session data for export.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            include_documents: Include document metadata
            include_context: Include context information
            
        Yields:
            Export data chunks
        """
        # First yield session metadata
        session_path = self.base_path / user_id / session_id
        session_meta_path = session_path / self.config.session_metadata_filename
        
        if session_meta_path.exists():
            with open(session_meta_path, 'r') as f:
                session_data = json.load(f)
                yield {
                    "type": "session_metadata",
                    "data": session_data
                }
        
        # Stream messages
        async for message_chunk in self.message_streamer.stream_messages(user_id, session_id):
            yield {
                "type": "messages",
                "data": [msg.to_dict() for msg in message_chunk]
            }
        
        # Include documents if requested
        if include_documents:
            docs_meta_path = (session_path / self.config.document_storage_subdirectory_name / 
                            self.config.document_metadata_filename)
            if docs_meta_path.exists():
                with open(docs_meta_path, 'r') as f:
                    docs_data = json.load(f)
                    yield {
                        "type": "documents",
                        "data": docs_data
                    }
        
        # Include context if requested
        if include_context:
            context_path = session_path / self.config.situational_context_filename
            if context_path.exists():
                with open(context_path, 'r') as f:
                    context_data = json.load(f)
                    yield {
                        "type": "context",
                        "data": context_data
                    }
            
            # Context history
            history_dir = session_path / self.config.context_history_subdirectory_name
            if history_dir.exists():
                history_files = sorted(history_dir.glob("*.json"))
                for history_file in history_files:
                    with open(history_file, 'r') as f:
                        yield {
                            "type": "context_history",
                            "data": {
                                "snapshot_id": history_file.stem,
                                "context": json.load(f)
                            }
                        }
    
    async def stream_user_export(self, user_id: str,
                               session_limit: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream all user data for export.
        
        Args:
            user_id: User identifier
            session_limit: Maximum number of sessions to export
            
        Yields:
            Export data chunks
        """
        user_path = get_user_path(self.base_path, user_id, self.config)
        
        # Yield user profile
        profile_path = user_path / self.config.user_profile_filename
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                yield {
                    "type": "user_profile",
                    "data": json.load(f)
                }
        
        # Get all sessions
        sessions = []
        for item in user_path.iterdir():
            if item.is_dir() and item.name.startswith(self.config.session_id_prefix):
                sessions.append(item.name)
        
        # Sort sessions by name (which includes timestamp)
        sessions.sort(reverse=True)
        
        # Apply limit if specified
        if session_limit:
            sessions = sessions[:session_limit]
        
        # Stream each session
        for session_id in sessions:
            yield {
                "type": "session_start",
                "data": {"session_id": session_id}
            }
            
            async for chunk in self.stream_session_export(user_id, session_id):
                yield chunk
            
            yield {
                "type": "session_end",
                "data": {"session_id": session_id}
            }


class LazyLoader:
    """
    Provides lazy loading capabilities for large data structures.
    
    Loads data on-demand to minimize memory usage.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize lazy loader.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.base_path = Path(config.storage_base_path)
        self._cache = {}
        self._cache_size_limit = 100  # Maximum cached items
    
    async def get_message_lazy(self, user_id: str, session_id: str, 
                             message_index: int) -> Optional[Message]:
        """
        Lazily load a single message by index.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message_index: Message index (0-based)
            
        Returns:
            Message object or None
        """
        cache_key = f"{user_id}:{session_id}:{message_index}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load message
        messages_path = (self.base_path / user_id / session_id / 
                        self.config.messages_filename)
        
        if not messages_path.exists():
            return None
        
        # Read specific line
        try:
            with open(messages_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == message_index:
                        if line.strip():
                            msg_data = json.loads(line)
                            msg = Message.from_dict(msg_data)
                            
                            # Cache it
                            self._add_to_cache(cache_key, msg)
                            
                            return msg
                        break
        except Exception as e:
            print(f"Error loading message: {e}")
        
        return None
    
    async def get_session_metadata_lazy(self, user_id: str, 
                                      session_id: str) -> Optional[Dict[str, Any]]:
        """
        Lazily load session metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session metadata or None
        """
        cache_key = f"session_meta:{user_id}:{session_id}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        session_path = self.base_path / user_id / session_id
        meta_path = session_path / self.config.session_metadata_filename
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self._add_to_cache(cache_key, data)
                return data
        
        return None
    
    def _add_to_cache(self, key: str, value: Any):
        """Add item to cache with size limit"""
        # Simple LRU-like behavior
        if len(self._cache) >= self._cache_size_limit:
            # Remove oldest item (first in dict)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = value
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()