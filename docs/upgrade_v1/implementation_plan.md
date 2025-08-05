# Flatfile Storage v2 - Implementation Plan

## Overview

This document provides a detailed implementation plan to upgrade the current flatfile storage codebase to support advanced chat capabilities. The plan is structured to maximize code reuse (~70%), minimize disruption, and deliver incremental value.

## Current State Analysis

### What We Keep (Minimal Changes)
1. **Core Infrastructure** (95% reuse)
   - `ff_dependency_injection_manager.py` - Excellent DI system
   - `ff_protocols/` - Clean protocol definitions
   - `ff_utils/` - Solid utility functions
   - `backends/ff_flatfile_storage_backend.py` - Storage abstraction

2. **Configuration System** (90% reuse)
   - `ff_class_configs/` - Well-structured configs
   - `ff_preset_configs/` - Configuration presets

3. **Search & Vector** (80% reuse)
   - `ff_search_manager.py` - Text search capabilities
   - `ff_vector_storage_manager.py` - Embedding storage
   - `ff_embedding_manager.py` - Embedding generation

### What Needs Major Refactoring
1. **Storage Manager** - Currently 1,682 lines doing too much
2. **Entity Models** - Need polymorphic message support
3. **Session Management** - Need flexible session types

### What's New
1. **Memory System** - Hierarchical memory with decay
2. **Multimodal Storage** - Image/audio/video handling
3. **Tool Execution Logs** - Tool usage tracking
4. **Trace System** - Comprehensive audit logs
5. **Agent Storage** - Agent configurations and routing

## Implementation Phases

### Phase 1: Data Models & Protocols (Week 1)
**Goal**: Update core data models to support new capabilities without breaking existing code

#### 1.1 Enhanced Entity Models (Day 1-2)

**File**: `ff_class_configs/ff_chat_entities_v2.py` (new file)

```python
# Keep imports from existing file
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid

# New enums for v2
class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"

class SenderType(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"

# Enhanced message model
@dataclass
class MessageV2DTO:
    """Polymorphic message supporting multiple content types"""
    id: str
    sender_type: SenderType
    sender_id: str
    sender_name: Optional[str] = None
    
    # Polymorphic content
    content_type: ContentType
    content_data: Dict[str, Any]
    
    # Support multiple content items
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Tool integration
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_responses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory references
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageV2DTO':
        # Handle enum conversions
        if 'sender_type' in data and isinstance(data['sender_type'], str):
            data['sender_type'] = SenderType(data['sender_type'])
        if 'content_type' in data and isinstance(data['content_type'], str):
            data['content_type'] = ContentType(data['content_type'])
        return cls(**data)
    
    @classmethod
    def from_legacy(cls, old_msg: 'FFMessageDTO') -> 'MessageV2DTO':
        """Convert from legacy format"""
        return cls(
            id=old_msg.message_id,
            sender_type=SenderType.USER if old_msg.role == "user" else SenderType.AGENT,
            sender_id="legacy",
            sender_name=old_msg.role,
            content_type=ContentType.TEXT,
            content_data={"text": old_msg.content},
            session_id=None,  # Will be set by context
            timestamp=old_msg.timestamp,
            metadata=old_msg.metadata
        )
```

**File**: `ff_class_configs/ff_memory_entities.py` (new file)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Specific conversations/events
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"        # Current context

@dataclass
class MemoryEntryDTO:
    """Storage model for memory entries"""
    id: str
    type: MemoryType
    content: str
    
    # Vector reference
    embedding_id: Optional[str] = None
    
    # Source tracking
    source_type: str = ""  # message, document, external
    source_id: str = ""
    
    # Temporal data
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    access_count: int = 0
    
    # Relevance
    importance: float = 0.5  # 0-1
    decay_factor: float = 1.0  # 0-1, decreases over time
    
    # Relationships
    related_memories: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntryDTO':
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = MemoryType(data['type'])
        return cls(**data)
    
    def access(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
        self.decay_factor = min(1.0, self.decay_factor + 0.1)
    
    def decay(self, rate: float = 0.01):
        """Apply time-based decay"""
        self.decay_factor = max(0.1, self.decay_factor - rate)
```

#### 1.2 Enhanced Protocols (Day 3)

**File**: `ff_protocols/ff_memory_protocol.py` (new file)

```python
from typing import Protocol, List, Optional, Dict, Any
from ff_class_configs.ff_memory_entities import MemoryEntryDTO, MemoryType

class MemoryStoreProtocol(Protocol):
    """Protocol for memory storage operations"""
    
    async def store_memory(self, memory: MemoryEntryDTO) -> bool:
        """Store a memory entry"""
        ...
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntryDTO]:
        """Retrieve a specific memory"""
        ...
    
    async def search_memories(
        self, 
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[MemoryEntryDTO]:
        """Search memories by query"""
        ...
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """Update memory access patterns"""
        ...
    
    async def decay_memories(self, user_id: str, decay_rate: float = 0.01) -> int:
        """Apply decay to memories"""
        ...
    
    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory storage statistics"""
        ...
```

**File**: `ff_protocols/ff_multimodal_protocol.py` (new file)

```python
from typing import Protocol, Optional, Dict, Any, BinaryIO
from pathlib import Path

class MultimodalStorageProtocol(Protocol):
    """Protocol for multimodal content storage"""
    
    async def store_image(
        self,
        session_id: str,
        image_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store image and return content ID"""
        ...
    
    async def store_audio(
        self,
        session_id: str,
        audio_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store audio and return content ID"""
        ...
    
    async def store_video(
        self,
        session_id: str,
        video_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store video and return content ID"""
        ...
    
    async def get_content(self, content_id: str) -> Optional[bytes]:
        """Retrieve multimodal content"""
        ...
    
    async def get_content_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content metadata"""
        ...
    
    async def stream_content(self, content_id: str) -> Optional[BinaryIO]:
        """Stream large content"""
        ...
```

#### 1.3 Configuration Updates (Day 4)

**File**: `ff_class_configs/ff_storage_config.py` (update existing)

```python
# Add to existing StorageConfigDTO
@dataclass
class StorageConfigDTO:
    # ... existing fields ...
    
    # New fields for v2
    multimodal_storage_path: str = "multimodal"
    memory_storage_path: str = "memory"
    trace_storage_path: str = "traces"
    
    # Multimodal limits
    max_image_size_mb: int = 10
    max_audio_size_mb: int = 50
    max_video_size_mb: int = 200
    allowed_image_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "gif", "webp"])
    allowed_audio_formats: List[str] = field(default_factory=lambda: ["mp3", "wav", "m4a", "ogg"])
    allowed_video_formats: List[str] = field(default_factory=lambda: ["mp4", "webm", "mov"])
    
    # Memory configuration
    memory_capacity_per_user: int = 10000
    memory_decay_rate: float = 0.01
    memory_compression_threshold: float = 0.9  # Compress when 90% full
    
    # Tool execution
    tool_execution_timeout_seconds: int = 30
    max_tool_executions_per_session: int = 100
```

### Phase 2: Core Managers Refactoring (Week 2)

#### 2.1 Split Storage Manager (Day 5-7)

Create focused managers instead of one monolithic storage manager:

**File**: `managers/ff_message_manager.py` (new file)

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO, ContentType
from ff_protocols import StorageProtocol, BackendProtocol
from ff_utils import ff_append_jsonl, ff_read_jsonl

class MessageManager:
    """Handles message storage and retrieval"""
    
    def __init__(self, config: Any, backend: BackendProtocol):
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
    
    async def store_message(self, message: MessageV2DTO) -> bool:
        """Store a message with support for multiple content types"""
        # Validate message size
        if not self._validate_message_size(message):
            return False
        
        # Get storage path
        messages_path = self._get_messages_path(message.session_id)
        
        # Handle multimodal content
        if message.content_type in [ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO]:
            content_id = await self._store_multimodal_content(message)
            if content_id:
                message.content_data["storage_ref"] = content_id
        
        # Store message
        message_data = message.to_dict()
        message_file = messages_path / f"{message.timestamp}_{message.id}.json"
        
        success = await self.backend.write(
            str(message_file.relative_to(self.base_path)),
            json.dumps(message_data).encode()
        )
        
        # Update index
        if success:
            await self._update_message_index(message.session_id, message)
        
        return success
    
    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        content_types: Optional[List[ContentType]] = None
    ) -> List[MessageV2DTO]:
        """Get messages with optional filtering"""
        messages_path = self._get_messages_path(session_id)
        
        # Get message files
        pattern = "*.json"
        message_keys = await self.backend.list_keys(
            str(messages_path.relative_to(self.base_path)),
            pattern=pattern
        )
        
        # Sort by timestamp (filename includes timestamp)
        message_keys.sort()
        
        # Apply pagination
        if limit:
            message_keys = message_keys[offset:offset + limit]
        
        # Load messages
        messages = []
        for key in message_keys:
            data = await self.backend.read(key)
            if data:
                msg_dict = json.loads(data.decode())
                msg = MessageV2DTO.from_dict(msg_dict)
                
                # Filter by content type if specified
                if content_types and msg.content_type not in content_types:
                    continue
                    
                messages.append(msg)
        
        return messages
    
    async def _store_multimodal_content(self, message: MessageV2DTO) -> Optional[str]:
        """Store multimodal content separately"""
        content_type = message.content_type.value
        content_data = message.content_data.get("data")
        
        if not content_data:
            return None
        
        # Generate content ID
        content_id = f"{content_type}_{uuid.uuid4().hex[:8]}"
        
        # Get storage path
        multimodal_path = self._get_multimodal_path(message.session_id, content_type)
        content_file = multimodal_path / f"{content_id}.{self._get_extension(message)}"
        
        # Store content
        success = await self.backend.write(
            str(content_file.relative_to(self.base_path)),
            base64.b64decode(content_data) if isinstance(content_data, str) else content_data
        )
        
        return content_id if success else None
    
    def _get_messages_path(self, session_id: str) -> Path:
        """Get messages storage path"""
        return self.base_path / "sessions" / session_id / "messages"
    
    def _get_multimodal_path(self, session_id: str, content_type: str) -> Path:
        """Get multimodal storage path"""
        return self.base_path / "sessions" / session_id / "multimodal" / content_type
```

**File**: `managers/ff_memory_manager.py` (new file)

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from ff_class_configs.ff_memory_entities import MemoryEntryDTO, MemoryType
from ff_protocols import BackendProtocol, VectorStoreProtocol

class MemoryManager:
    """Manages hierarchical memory storage with decay"""
    
    def __init__(self, config: Any, backend: BackendProtocol, vector_store: VectorStoreProtocol):
        self.config = config
        self.backend = backend
        self.vector_store = vector_store
        self.base_path = Path(config.storage.base_path)
    
    async def store_memory(
        self,
        memory: MemoryEntryDTO,
        scope: str = "user",  # user, agent, global
        scope_id: Optional[str] = None
    ) -> bool:
        """Store a memory entry"""
        # Get storage path
        memory_path = self._get_memory_path(scope, scope_id, memory.type)
        
        # Check capacity
        if scope == "user" and scope_id:
            if not await self._check_capacity(scope_id):
                await self._compress_memories(scope_id)
        
        # Generate embedding if needed
        if not memory.embedding_id:
            embedding = await self._generate_embedding(memory.content)
            if embedding:
                memory.embedding_id = await self.vector_store.store_vector(
                    f"memory_{memory.id}",
                    embedding,
                    {
                        "memory_id": memory.id,
                        "type": memory.type.value,
                        "scope": scope,
                        "scope_id": scope_id
                    }
                )
        
        # Store memory
        memory_file = memory_path / f"{memory.id}.json"
        success = await self.backend.write(
            str(memory_file.relative_to(self.base_path)),
            json.dumps(memory.to_dict()).encode()
        )
        
        # Update index
        if success:
            await self._update_memory_index(scope, scope_id, memory)
        
        return success
    
    async def search_memories(
        self,
        query: str,
        scope: str = "user",
        scope_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[MemoryEntryDTO]:
        """Search memories using vector similarity and metadata"""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search in vector store
        vector_results = await self.vector_store.search_similar(
            query_vector=query_embedding,
            filter_metadata={
                "scope": scope,
                "scope_id": scope_id
            },
            top_k=limit * 2,  # Get more for filtering
            threshold=threshold
        )
        
        # Load memory entries
        memories = []
        for result in vector_results:
            memory_id = result.metadata.get("memory_id")
            if memory_id:
                memory = await self.get_memory(memory_id, scope, scope_id)
                if memory:
                    # Filter by type if specified
                    if memory_types and memory.type not in memory_types:
                        continue
                    
                    # Apply relevance decay
                    if memory.decay_factor < threshold:
                        continue
                    
                    memories.append(memory)
        
        # Sort by relevance (similarity * importance * decay)
        memories.sort(
            key=lambda m: result.similarity_score * m.importance * m.decay_factor,
            reverse=True
        )
        
        return memories[:limit]
    
    async def decay_memories(
        self,
        scope: str = "user",
        scope_id: Optional[str] = None,
        decay_rate: float = None
    ) -> int:
        """Apply decay to memories"""
        decay_rate = decay_rate or self.config.storage.memory_decay_rate
        
        # Get all memories for scope
        memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
        count = 0
        
        for mem_type in memory_types:
            memory_path = self._get_memory_path(scope, scope_id, mem_type)
            pattern = "*.json"
            
            memory_keys = await self.backend.list_keys(
                str(memory_path.relative_to(self.base_path)),
                pattern=pattern
            )
            
            for key in memory_keys:
                # Load memory
                data = await self.backend.read(key)
                if data:
                    memory = MemoryEntryDTO.from_dict(json.loads(data.decode()))
                    
                    # Apply decay
                    old_decay = memory.decay_factor
                    memory.decay(decay_rate)
                    
                    # Save if changed
                    if old_decay != memory.decay_factor:
                        await self.backend.write(key, json.dumps(memory.to_dict()).encode())
                        count += 1
        
        return count
    
    async def _compress_memories(self, user_id: str):
        """Compress memories when approaching capacity"""
        # Get all memories
        all_memories = []
        for mem_type in MemoryType:
            memories = await self._get_memories_by_type(user_id, mem_type)
            all_memories.extend(memories)
        
        # Sort by relevance score
        all_memories.sort(
            key=lambda m: m.importance * m.decay_factor,
            reverse=True
        )
        
        # Keep top memories based on capacity
        capacity = self.config.storage.memory_capacity_per_user
        to_keep = int(capacity * 0.9)  # Keep 90%
        
        # Delete low-relevance memories
        for memory in all_memories[to_keep:]:
            await self._delete_memory(memory.id, "user", user_id)
    
    def _get_memory_path(self, scope: str, scope_id: Optional[str], memory_type: MemoryType) -> Path:
        """Get memory storage path"""
        if scope == "global":
            return self.base_path / "memory" / "global" / memory_type.value
        elif scope == "user" and scope_id:
            return self.base_path / "users" / scope_id / "memory" / memory_type.value
        elif scope == "agent" and scope_id:
            return self.base_path / "agents" / scope_id / "memory" / memory_type.value
        else:
            raise ValueError(f"Invalid scope/scope_id combination: {scope}/{scope_id}")
```

**File**: `managers/ff_session_manager.py` (new file)

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from ff_class_configs.ff_chat_entities_v2 import SessionV2DTO, SessionType
from ff_protocols import BackendProtocol

class SessionManager:
    """Manages flexible session types"""
    
    def __init__(self, config: Any, backend: BackendProtocol):
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
    
    async def create_session(self, session: SessionV2DTO) -> str:
        """Create a new session of any type"""
        # Validate session
        if not self._validate_session(session):
            raise ValueError("Invalid session configuration")
        
        # Get session path
        session_path = self._get_session_path(session.id)
        
        # Create session structure
        await self._create_session_structure(session_path, session.type)
        
        # Store session metadata
        metadata_file = session_path / "metadata.json"
        success = await self.backend.write(
            str(metadata_file.relative_to(self.base_path)),
            json.dumps(session.to_dict()).encode()
        )
        
        if success:
            # Initialize type-specific components
            await self._initialize_session_type(session)
            return session.id
        
        return ""
    
    async def update_session_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update session context"""
        # Load session
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Update context
        if not session.context:
            session.context = {}
        
        session.context.update(context_updates)
        session.updated_at = datetime.now().isoformat()
        
        # Save session
        return await self._save_session(session)
    
    async def add_participant(
        self,
        session_id: str,
        participant: Dict[str, Any]
    ) -> bool:
        """Add participant to session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Check if participant already exists
        existing_ids = [p.get("id") for p in session.participants]
        if participant.get("id") in existing_ids:
            return False
        
        # Add participant
        session.participants.append(participant)
        session.updated_at = datetime.now().isoformat()
        
        return await self._save_session(session)
    
    async def _create_session_structure(self, session_path: Path, session_type: SessionType):
        """Create directory structure based on session type"""
        # Common directories
        common_dirs = ["messages", "context", "traces"]
        
        # Type-specific directories
        type_dirs = {
            SessionType.CHAT: ["multimodal", "documents"],
            SessionType.PANEL: ["decisions", "insights"],
            SessionType.WORKFLOW: ["states", "executions"],
            SessionType.PLAYGROUND: ["experiments", "results"]
        }
        
        # Create directories
        all_dirs = common_dirs + type_dirs.get(session_type, [])
        
        for dir_name in all_dirs:
            dir_path = session_path / dir_name
            # Create directory (backend should handle this)
            await self.backend.write(
                str((dir_path / ".init").relative_to(self.base_path)),
                b""
            )
    
    async def _initialize_session_type(self, session: SessionV2DTO):
        """Initialize type-specific components"""
        if session.type == SessionType.PANEL:
            # Initialize panel-specific data
            await self._init_panel_session(session)
        elif session.type == SessionType.WORKFLOW:
            # Initialize workflow state
            await self._init_workflow_session(session)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get session storage path"""
        return self.base_path / "sessions" / session_id
```

#### 2.2 New Component Managers (Day 8-10)

**File**: `managers/ff_multimodal_manager.py` (new file)

```python
import hashlib
import mimetypes
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
from ff_protocols import BackendProtocol, MultimodalStorageProtocol

class MultimodalManager(MultimodalStorageProtocol):
    """Handles multimodal content storage and retrieval"""
    
    def __init__(self, config: Any, backend: BackendProtocol):
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
    
    async def store_image(
        self,
        session_id: str,
        image_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store image with deduplication"""
        # Validate
        if not self._validate_image(image_data, filename):
            raise ValueError("Invalid image")
        
        # Generate content ID with hash for deduplication
        content_hash = hashlib.sha256(image_data).hexdigest()[:16]
        content_id = f"img_{content_hash}"
        
        # Check if already exists
        existing = await self._check_duplicate(content_hash)
        if existing:
            return existing
        
        # Store image
        storage_path = self._get_storage_path(session_id, "images", content_id, filename)
        
        success = await self.backend.write(
            str(storage_path.relative_to(self.base_path)),
            image_data
        )
        
        if success:
            # Store metadata
            await self._store_metadata(
                session_id, "images", content_id,
                {
                    "filename": filename,
                    "size_bytes": len(image_data),
                    "hash": content_hash,
                    "mime_type": mimetypes.guess_type(filename)[0],
                    **(metadata or {})
                }
            )
            
            return content_id
        
        return ""
    
    async def store_audio(
        self,
        session_id: str,
        audio_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store audio file"""
        # Similar implementation to store_image
        # Add audio-specific validation and metadata
        pass
    
    async def get_content(self, content_id: str) -> Optional[bytes]:
        """Retrieve multimodal content"""
        # Find content file
        search_pattern = f"**/multimodal/**/{content_id}.*"
        
        results = await self.backend.list_keys("", pattern=search_pattern)
        if results:
            return await self.backend.read(results[0])
        
        return None
    
    async def stream_content(self, content_id: str) -> Optional[BinaryIO]:
        """Stream large content files"""
        # This would implement streaming for large files
        # For now, return regular content
        content = await self.get_content(content_id)
        if content:
            import io
            return io.BytesIO(content)
        return None
    
    def _validate_image(self, data: bytes, filename: str) -> bool:
        """Validate image constraints"""
        # Check size
        max_size = self.config.storage.max_image_size_mb * 1024 * 1024
        if len(data) > max_size:
            return False
        
        # Check format
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in self.config.storage.allowed_image_formats:
            return False
        
        # Could add actual image validation here
        return True
    
    def _get_storage_path(
        self,
        session_id: str,
        content_type: str,
        content_id: str,
        filename: str
    ) -> Path:
        """Get storage path for content"""
        ext = Path(filename).suffix
        return (
            self.base_path / "sessions" / session_id / 
            "multimodal" / content_type / f"{content_id}{ext}"
        )
```

**File**: `managers/ff_trace_manager.py` (new file)

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
from ff_protocols import BackendProtocol

@dataclass
class TraceEvent:
    id: str
    timestamp: str
    event_type: str
    component: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    message_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    parent_event_id: Optional[str] = None

class TraceManager:
    """Manages trace logging for debugging and auditing"""
    
    def __init__(self, config: Any, backend: BackendProtocol):
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self._trace_buffer: Dict[str, List[TraceEvent]] = {}
        self._flush_interval = 100  # Flush after 100 events
    
    async def log_event(self, event: TraceEvent):
        """Log a trace event"""
        session_id = event.session_id or "global"
        
        # Add to buffer
        if session_id not in self._trace_buffer:
            self._trace_buffer[session_id] = []
        
        self._trace_buffer[session_id].append(event)
        
        # Flush if needed
        if len(self._trace_buffer[session_id]) >= self._flush_interval:
            await self._flush_traces(session_id)
    
    async def start_span(
        self,
        name: str,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """Start a trace span"""
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        
        event = TraceEvent(
            id=span_id,
            timestamp=datetime.now().isoformat(),
            event_type="span_start",
            component=name,
            session_id=session_id,
            parent_event_id=parent_id,
            data={"span_name": name}
        )
        
        await self.log_event(event)
        return span_id
    
    async def end_span(self, span_id: str, session_id: Optional[str] = None):
        """End a trace span"""
        event = TraceEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            event_type="span_end",
            component="trace",
            session_id=session_id,
            parent_event_id=span_id,
            data={"span_id": span_id}
        )
        
        await self.log_event(event)
    
    async def get_session_trace(
        self,
        session_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_types: Optional[List[str]] = None
    ) -> List[TraceEvent]:
        """Get trace events for a session"""
        # Flush pending events
        await self._flush_traces(session_id)
        
        # Read trace files
        trace_path = self._get_trace_path(session_id)
        trace_file = trace_path / "events.jsonl"
        
        events = []
        data = await self.backend.read(str(trace_file.relative_to(self.base_path)))
        
        if data:
            for line in data.decode().strip().split('\n'):
                if line:
                    event_dict = json.loads(line)
                    event = TraceEvent(**event_dict)
                    
                    # Filter by time
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    
                    # Filter by type
                    if event_types and event.event_type not in event_types:
                        continue
                    
                    events.append(event)
        
        return events
    
    async def _flush_traces(self, session_id: str):
        """Flush trace buffer to storage"""
        if session_id not in self._trace_buffer:
            return
        
        events = self._trace_buffer[session_id]
        if not events:
            return
        
        # Get trace file
        trace_path = self._get_trace_path(session_id)
        trace_file = trace_path / "events.jsonl"
        
        # Append events
        for event in events:
            event_data = asdict(event)
            await self.backend.append(
                str(trace_file.relative_to(self.base_path)),
                (json.dumps(event_data) + '\n').encode()
            )
        
        # Clear buffer
        self._trace_buffer[session_id] = []
    
    def _get_trace_path(self, session_id: str) -> Path:
        """Get trace storage path"""
        if session_id == "global":
            return self.base_path / "system" / "traces" / datetime.now().strftime("%Y-%m-%d")
        return self.base_path / "sessions" / session_id / "traces"
```

### Phase 3: Integration Layer (Week 3)

#### 3.1 Updated Storage Manager Facade (Day 11-12)

**File**: `ff_storage_manager_v2.py` (new file)

```python
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO, SessionV2DTO
from ff_class_configs.ff_memory_entities import MemoryEntryDTO
from managers import (
    MessageManager, SessionManager, MemoryManager,
    MultimodalManager, TraceManager
)

class FFStorageManagerV2:
    """Facade for all storage operations - maintains similar API to v1"""
    
    def __init__(self, config: Optional[Any] = None, backend: Optional[Any] = None):
        self.config = config or load_config()
        self.backend = backend or FlatfileBackend(self.config)
        
        # Initialize managers
        self.messages = MessageManager(self.config, self.backend)
        self.sessions = SessionManager(self.config, self.backend)
        self.memory = MemoryManager(self.config, self.backend, self._get_vector_store())
        self.multimodal = MultimodalManager(self.config, self.backend)
        self.traces = TraceManager(self.config, self.backend)
        
        # Keep some original managers for unchanged functionality
        self.search_engine = FFSearchManager(self.config)
        self.vector_storage = FFVectorStorageManager(self.config)
    
    # Message operations - new API
    async def store_message(self, message: MessageV2DTO) -> bool:
        """Store a polymorphic message"""
        return await self.messages.store_message(message)
    
    # Legacy compatibility
    async def add_message(self, user_id: str, session_id: str, message: Any) -> bool:
        """Legacy method - converts old format to new"""
        if hasattr(message, 'message_id'):  # Old FFMessageDTO
            new_message = MessageV2DTO.from_legacy(message)
            new_message.session_id = session_id
            return await self.store_message(new_message)
        return await self.store_message(message)
    
    # Memory operations
    async def store_memory(
        self,
        memory: MemoryEntryDTO,
        user_id: Optional[str] = None
    ) -> bool:
        """Store a memory entry"""
        scope = "user" if user_id else "global"
        return await self.memory.store_memory(memory, scope, user_id)
    
    async def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemoryEntryDTO]:
        """Search memories"""
        scope = "user" if user_id else "global"
        return await self.memory.search_memories(
            query, scope, user_id, memory_types, limit
        )
    
    # Multimodal operations
    async def store_image(
        self,
        session_id: str,
        image_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an image"""
        return await self.multimodal.store_image(
            session_id, image_data, filename, metadata
        )
    
    # Trace operations
    async def log_trace_event(self, event: Dict[str, Any]):
        """Log a trace event"""
        trace_event = TraceEvent(**event)
        await self.traces.log_event(trace_event)
    
    # Keep all existing methods for compatibility
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """Existing method - unchanged"""
        # Implementation remains the same
        pass
```

#### 3.2 Migration Tools (Day 13)

**File**: `migration/migrate_to_v2.py` (new file)

```python
import asyncio
from pathlib import Path
from typing import Dict, Any
import json
from ff_storage_manager import FFStorageManager
from ff_storage_manager_v2 import FFStorageManagerV2
from ff_class_configs.ff_chat_entities import FFMessageDTO
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO

class MigrationTool:
    """Migrate from v1 to v2 storage format"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.v1_storage = FFStorageManager()
        self.v2_storage = FFStorageManagerV2()
        self.stats = {
            "messages_migrated": 0,
            "sessions_migrated": 0,
            "errors": []
        }
    
    async def migrate_all(self):
        """Run full migration"""
        print("Starting migration to v2...")
        
        # Migrate users and sessions
        users = await self.v1_storage.list_users()
        
        for user_id in users:
            print(f"Migrating user: {user_id}")
            await self.migrate_user(user_id)
        
        print(f"Migration complete: {self.stats}")
    
    async def migrate_user(self, user_id: str):
        """Migrate a single user's data"""
        # Migrate profile (unchanged)
        profile = await self.v1_storage.get_user_profile(user_id)
        if profile:
            await self.v2_storage.create_user(user_id, profile)
        
        # Migrate sessions
        sessions = await self.v1_storage.list_sessions(user_id, limit=None)
        
        for session in sessions:
            await self.migrate_session(user_id, session)
    
    async def migrate_session(self, user_id: str, session: Any):
        """Migrate a single session"""
        try:
            # Create v2 session
            session_v2 = SessionV2DTO(
                id=session.session_id,
                type="chat",  # Default type
                title=session.title,
                participants=[{
                    "id": user_id,
                    "type": "user",
                    "role": "member",
                    "name": user_id
                }],
                context={},
                capabilities=["text"],  # Basic capability
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=session.message_count,
                metadata=session.metadata
            )
            
            await self.v2_storage.sessions.create_session(session_v2)
            self.stats["sessions_migrated"] += 1
            
            # Migrate messages
            messages = await self.v1_storage.get_all_messages(user_id, session.session_id)
            
            for message in messages:
                await self.migrate_message(user_id, session.session_id, message)
                
        except Exception as e:
            self.stats["errors"].append({
                "type": "session",
                "id": session.session_id,
                "error": str(e)
            })
    
    async def migrate_message(self, user_id: str, session_id: str, message: FFMessageDTO):
        """Migrate a single message"""
        try:
            # Convert to v2 format
            message_v2 = MessageV2DTO.from_legacy(message)
            message_v2.session_id = session_id
            
            # Store in v2
            await self.v2_storage.store_message(message_v2)
            self.stats["messages_migrated"] += 1
            
        except Exception as e:
            self.stats["errors"].append({
                "type": "message",
                "id": message.message_id,
                "error": str(e)
            })

# Run migration
if __name__ == "__main__":
    async def main():
        migrator = MigrationTool(Path("./data"))
        await migrator.migrate_all()
    
    asyncio.run(main())
```

### Phase 4: Testing & Validation (Week 4)

#### 4.1 Unit Tests (Day 14-15)

**File**: `tests/test_v2_messages.py` (new file)

```python
import pytest
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO, ContentType, SenderType
from managers.ff_message_manager import MessageManager

class TestMessageV2:
    
    def test_text_message_creation(self):
        """Test creating a text message"""
        msg = MessageV2DTO(
            id="test_123",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.TEXT,
            content_data={"text": "Hello, world!"},
            session_id="session_123"
        )
        
        assert msg.id == "test_123"
        assert msg.content_type == ContentType.TEXT
        assert msg.content_data["text"] == "Hello, world!"
    
    def test_multimodal_message(self):
        """Test creating a multimodal message"""
        msg = MessageV2DTO(
            id="test_456",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.IMAGE,
            content_data={
                "storage_ref": "img_abc123",
                "mime_type": "image/jpeg",
                "caption": "Test image"
            },
            session_id="session_123"
        )
        
        assert msg.content_type == ContentType.IMAGE
        assert msg.content_data["storage_ref"] == "img_abc123"
    
    def test_tool_call_message(self):
        """Test message with tool calls"""
        msg = MessageV2DTO(
            id="test_789",
            sender_type=SenderType.AGENT,
            sender_id="agent_123",
            content_type=ContentType.TOOL_CALL,
            content_data={
                "message": "Let me search for that information"
            },
            tool_calls=[{
                "id": "call_001",
                "tool_name": "web_search",
                "arguments": {"query": "weather today"}
            }],
            session_id="session_123"
        )
        
        assert msg.content_type == ContentType.TOOL_CALL
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["tool_name"] == "web_search"
    
    @pytest.mark.asyncio
    async def test_message_storage(self, mock_backend):
        """Test storing messages"""
        manager = MessageManager(mock_config(), mock_backend)
        
        msg = MessageV2DTO(
            id="store_test",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.TEXT,
            content_data={"text": "Test storage"},
            session_id="session_123"
        )
        
        result = await manager.store_message(msg)
        assert result is True
        
        # Verify backend was called
        assert mock_backend.write.called
```

**File**: `tests/test_v2_memory.py` (new file)

```python
import pytest
from ff_class_configs.ff_memory_entities import MemoryEntryDTO, MemoryType
from managers.ff_memory_manager import MemoryManager

class TestMemorySystem:
    
    def test_memory_creation(self):
        """Test creating memory entries"""
        memory = MemoryEntryDTO(
            id="mem_123",
            type=MemoryType.EPISODIC,
            content="User discussed project timeline",
            source_type="message",
            source_id="msg_456",
            importance=0.8
        )
        
        assert memory.id == "mem_123"
        assert memory.type == MemoryType.EPISODIC
        assert memory.importance == 0.8
        assert memory.decay_factor == 1.0
    
    def test_memory_decay(self):
        """Test memory decay mechanism"""
        memory = MemoryEntryDTO(
            id="mem_decay",
            type=MemoryType.SEMANTIC,
            content="Test fact",
            decay_factor=1.0
        )
        
        # Apply decay
        memory.decay(0.1)
        assert memory.decay_factor == 0.9
        
        # Apply more decay
        memory.decay(0.5)
        assert memory.decay_factor == 0.4
        
        # Minimum decay
        memory.decay(0.5)
        assert memory.decay_factor == 0.1  # Minimum
    
    def test_memory_access(self):
        """Test memory access patterns"""
        memory = MemoryEntryDTO(
            id="mem_access",
            type=MemoryType.SEMANTIC,
            content="Important fact",
            access_count=0,
            decay_factor=0.5
        )
        
        # Access memory
        memory.access()
        
        assert memory.access_count == 1
        assert memory.last_accessed is not None
        assert memory.decay_factor == 0.6  # Boosted by 0.1
    
    @pytest.mark.asyncio
    async def test_memory_search(self, mock_backend, mock_vector_store):
        """Test searching memories"""
        manager = MemoryManager(mock_config(), mock_backend, mock_vector_store)
        
        # Mock vector search results
        mock_vector_store.search_similar.return_value = [
            MockSearchResult(
                metadata={"memory_id": "mem_1"},
                similarity_score=0.9
            )
        ]
        
        results = await manager.search_memories(
            "project timeline",
            scope="user",
            scope_id="user_123"
        )
        
        assert len(results) > 0
```

#### 4.2 Integration Tests (Day 16-17)

**File**: `tests/test_v2_integration.py` (new file)

```python
import pytest
from ff_storage_manager_v2 import FFStorageManagerV2
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO, ContentType

class TestV2Integration:
    
    @pytest.mark.asyncio
    async def test_multimodal_message_flow(self, temp_storage_dir):
        """Test complete multimodal message flow"""
        storage = FFStorageManagerV2(test_config(temp_storage_dir))
        
        # Create user and session
        await storage.create_user("test_user", {"name": "Test User"})
        session_id = await storage.create_session(SessionV2DTO(
            id="test_session",
            type="chat",
            title="Test Chat",
            participants=[{
                "id": "test_user",
                "type": "user",
                "role": "member"
            }],
            context={},
            capabilities=["text", "image"]
        ))
        
        # Store text message
        text_msg = MessageV2DTO(
            id="msg_1",
            sender_type=SenderType.USER,
            sender_id="test_user",
            content_type=ContentType.TEXT,
            content_data={"text": "Check this image"},
            session_id=session_id
        )
        assert await storage.store_message(text_msg)
        
        # Store image
        image_data = b"fake_image_data"
        image_id = await storage.store_image(
            session_id,
            image_data,
            "test.jpg",
            {"description": "Test image"}
        )
        assert image_id
        
        # Store image message
        image_msg = MessageV2DTO(
            id="msg_2",
            sender_type=SenderType.USER,
            sender_id="test_user",
            content_type=ContentType.IMAGE,
            content_data={
                "storage_ref": image_id,
                "caption": "My test image"
            },
            session_id=session_id
        )
        assert await storage.store_message(image_msg)
        
        # Retrieve messages
        messages = await storage.messages.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0].content_type == ContentType.TEXT
        assert messages[1].content_type == ContentType.IMAGE
        
        # Retrieve image
        retrieved_image = await storage.multimodal.get_content(image_id)
        assert retrieved_image == image_data
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, temp_storage_dir):
        """Test memory system integration"""
        storage = FFStorageManagerV2(test_config(temp_storage_dir))
        
        # Create user
        await storage.create_user("memory_user")
        
        # Store episodic memory
        memory = MemoryEntryDTO(
            id="mem_episode_1",
            type=MemoryType.EPISODIC,
            content="User asked about Python programming",
            source_type="message",
            source_id="msg_123",
            importance=0.7
        )
        
        assert await storage.store_memory(memory, "memory_user")
        
        # Search memories
        results = await storage.search_memories(
            "Python",
            user_id="memory_user"
        )
        
        assert len(results) > 0
        assert results[0].content == "User asked about Python programming"
        
        # Test decay
        decayed = await storage.memory.decay_memories(
            scope="user",
            scope_id="memory_user",
            decay_rate=0.1
        )
        assert decayed > 0
```

#### 4.3 Performance Tests (Day 18)

**File**: `tests/test_v2_performance.py` (new file)

```python
import pytest
import asyncio
import time
from ff_storage_manager_v2 import FFStorageManagerV2

class TestV2Performance:
    
    @pytest.mark.asyncio
    async def test_message_throughput(self, temp_storage_dir):
        """Test message storage throughput"""
        storage = FFStorageManagerV2(test_config(temp_storage_dir))
        
        # Setup
        await storage.create_user("perf_user")
        session_id = "perf_session"
        await storage.sessions.create_session(SessionV2DTO(
            id=session_id,
            type="chat",
            title="Performance Test"
        ))
        
        # Test throughput
        message_count = 1000
        start_time = time.time()
        
        tasks = []
        for i in range(message_count):
            msg = MessageV2DTO(
                id=f"perf_msg_{i}",
                sender_type=SenderType.USER,
                sender_id="perf_user",
                content_type=ContentType.TEXT,
                content_data={"text": f"Message {i}"},
                session_id=session_id
            )
            tasks.append(storage.store_message(msg))
        
        # Execute in batches
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            await asyncio.gather(*batch)
        
        duration = time.time() - start_time
        throughput = message_count / duration
        
        print(f"Message throughput: {throughput:.2f} messages/second")
        assert throughput > 500  # At least 500 messages/second
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, temp_storage_dir):
        """Test memory search performance"""
        storage = FFStorageManagerV2(test_config(temp_storage_dir))
        
        # Populate memories
        memory_count = 10000
        for i in range(memory_count):
            memory = MemoryEntryDTO(
                id=f"perf_mem_{i}",
                type=MemoryType.SEMANTIC,
                content=f"Fact number {i} about topic {i % 100}",
                importance=0.5
            )
            await storage.store_memory(memory, "perf_user")
        
        # Test search performance
        start_time = time.time()
        
        search_queries = [
            "topic 42",
            "fact number 1234",
            "topic 99"
        ]
        
        for query in search_queries:
            results = await storage.search_memories(
                query,
                user_id="perf_user",
                limit=10
            )
            assert len(results) > 0
        
        duration = time.time() - start_time
        avg_search_time = duration / len(search_queries)
        
        print(f"Average memory search time: {avg_search_time*1000:.2f}ms")
        assert avg_search_time < 0.5  # Less than 500ms per search
```

### Phase 5: Documentation & Deployment (Week 5)

#### 5.1 API Documentation (Day 19)

**File**: `docs/upgrade_v1/api_reference.md` (new file)

```markdown
# Storage V2 API Reference

## Message Operations

### store_message
Store a polymorphic message supporting text, images, audio, video, and tool calls.

```python
async def store_message(message: MessageV2DTO) -> bool
```

**Parameters:**
- `message`: MessageV2DTO object with content_type and content_data

**Example:**
```python
# Text message
text_msg = MessageV2DTO(
    id="msg_123",
    sender_type=SenderType.USER,
    sender_id="user_123",
    content_type=ContentType.TEXT,
    content_data={"text": "Hello, world!"},
    session_id="session_123"
)
await storage.store_message(text_msg)

# Image message
image_msg = MessageV2DTO(
    id="msg_124",
    sender_type=SenderType.USER,
    sender_id="user_123",
    content_type=ContentType.IMAGE,
    content_data={
        "storage_ref": "img_abc123",
        "caption": "Check this out!"
    },
    session_id="session_123"
)
await storage.store_message(image_msg)
```

## Memory Operations

### store_memory
Store a memory entry with automatic embedding generation.

```python
async def store_memory(
    memory: MemoryEntryDTO,
    user_id: Optional[str] = None
) -> bool
```

**Parameters:**
- `memory`: MemoryEntryDTO with type, content, and importance
- `user_id`: Optional user ID for user-scoped memory

**Example:**
```python
memory = MemoryEntryDTO(
    id="mem_123",
    type=MemoryType.EPISODIC,
    content="User discussed project timeline for Q1",
    source_type="message",
    source_id="msg_456",
    importance=0.8
)
await storage.store_memory(memory, "user_123")
```

### search_memories
Search memories using semantic similarity.

```python
async def search_memories(
    query: str,
    user_id: Optional[str] = None,
    memory_types: Optional[List[MemoryType]] = None,
    limit: int = 10
) -> List[MemoryEntryDTO]
```

## Multimodal Operations

### store_image
Store an image with automatic deduplication.

```python
async def store_image(
    session_id: str,
    image_data: bytes,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Returns:** Content ID for referencing in messages

## Trace Operations

### log_trace_event
Log events for debugging and auditing.

```python
async def log_trace_event(event: Dict[str, Any])
```

**Example:**
```python
await storage.log_trace_event({
    "event_type": "message_processed",
    "session_id": "session_123",
    "message_id": "msg_123",
    "duration_ms": 45,
    "data": {"tokens": 150}
})
```
```

#### 5.2 Migration Guide (Day 20)

**File**: `docs/upgrade_v1/migration_guide.md` (new file)

```markdown
# Migration Guide: V1 to V2

## Overview
This guide helps you migrate from Storage V1 to V2. The upgrade adds support for:
- Polymorphic messages (text, images, audio, video)
- Hierarchical memory system
- Tool execution tracking
- Comprehensive tracing
- Flexible session types

## Automatic Migration

Run the migration script to automatically convert existing data:

```bash
python migration/migrate_to_v2.py --data-dir ./data
```

## Code Changes

### Message Handling

**V1 (Old):**
```python
from ff_class_configs.ff_chat_entities import FFMessageDTO

message = FFMessageDTO(
    role="user",
    content="Hello",
    message_id="msg_123",
    timestamp=datetime.now().isoformat()
)
await storage.add_message("user_123", "session_123", message)
```

**V2 (New):**
```python
from ff_class_configs.ff_chat_entities_v2 import MessageV2DTO, ContentType, SenderType

message = MessageV2DTO(
    id="msg_123",
    sender_type=SenderType.USER,
    sender_id="user_123",
    content_type=ContentType.TEXT,
    content_data={"text": "Hello"},
    session_id="session_123"
)
await storage.store_message(message)
```

### Multimodal Content

**New in V2:**
```python
# Store image
image_id = await storage.store_image(
    session_id="session_123",
    image_data=image_bytes,
    filename="chart.png"
)

# Reference in message
image_msg = MessageV2DTO(
    id="msg_124",
    sender_type=SenderType.USER,
    sender_id="user_123",
    content_type=ContentType.IMAGE,
    content_data={
        "storage_ref": image_id,
        "caption": "Sales chart"
    },
    session_id="session_123"
)
```

### Memory System

**New in V2:**
```python
# Store memory
memory = MemoryEntryDTO(
    id="mem_123",
    type=MemoryType.EPISODIC,
    content="User prefers dark mode",
    importance=0.7
)
await storage.store_memory(memory, user_id="user_123")

# Search memories
memories = await storage.search_memories(
    "user preferences",
    user_id="user_123"
)
```

## Compatibility Mode

Use the legacy adapter for gradual migration:

```python
from ff_storage_manager_v2 import FFStorageManagerV2

storage = FFStorageManagerV2()

# Old API still works
await storage.add_message("user_123", "session_123", old_message)

# New API available
await storage.store_message(new_message)
```
```

### Implementation Timeline Summary

**Week 1: Foundation**
- Days 1-2: Enhanced entity models
- Day 3: New protocols
- Day 4: Configuration updates

**Week 2: Core Refactoring**
- Days 5-7: Split storage manager
- Days 8-10: New component managers

**Week 3: Integration**
- Days 11-12: Storage facade
- Day 13: Migration tools

**Week 4: Testing**
- Days 14-15: Unit tests
- Days 16-17: Integration tests
- Day 18: Performance tests

**Week 5: Documentation**
- Day 19: API documentation
- Day 20: Migration guide

## Success Metrics

1. **Code Reuse**: ~70% of existing code preserved
2. **Performance**: 
   - Message throughput > 500/second
   - Memory search < 500ms
   - Multimodal storage < 1s for 10MB files
3. **Test Coverage**: > 80% for new components
4. **Migration Success**: All existing data migrated without loss

## Next Steps

1. Review and approve implementation plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Weekly progress reviews
5. Incremental testing and validation