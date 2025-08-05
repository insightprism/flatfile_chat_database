# Phase 1: Data Models & Protocols Implementation

## Overview
This phase establishes the foundation for the storage v2 upgrade by implementing enhanced data models and protocols. All new models are designed to coexist with existing code, allowing for non-breaking migration.

## Objectives
1. Create enhanced entity models supporting polymorphic content
2. Define new protocols for memory, multimodal, and trace systems
3. Update configuration structures
4. Ensure backward compatibility with existing models

## File Structure
```
flatfile_chat_database_v2/
├── ff_class_configs/
│   ├── ff_chat_entities_v2.py (new)
│   ├── ff_memory_entities.py (new)
│   ├── ff_multimodal_entities.py (new)
│   ├── ff_trace_entities.py (new)
│   ├── ff_agent_entities.py (new)
│   └── ff_storage_config.py (update)
├── ff_protocols/
│   ├── ff_memory_protocol.py (new)
│   ├── ff_multimodal_protocol.py (new)
│   ├── ff_trace_protocol.py (new)
│   └── ff_agent_protocol.py (new)
└── ff_utils/
    └── ff_entity_converters.py (new)
```

## Implementation Details

### 1. Enhanced Message Model (`ff_class_configs/ff_chat_entities_v2.py`)

Create a new file that supports polymorphic messages while maintaining compatibility:

```python
"""
Enhanced chat entity models for v2 storage system.

These models support polymorphic content types, tool integration,
and memory references while maintaining backward compatibility.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid

# Import legacy models for conversion
from ff_class_configs.ff_chat_entities import FFMessageDTO


class ContentType(str, Enum):
    """Supported content types for messages"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM = "system"
    MEMORY_REFERENCE = "memory_reference"


class SenderType(str, Enum):
    """Types of message senders"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class SessionType(str, Enum):
    """Types of sessions supported"""
    CHAT = "chat"              # Standard 1:1 or group chat
    PANEL = "panel"            # Multi-agent panel discussion
    WORKFLOW = "workflow"      # Task-oriented workflow
    PLAYGROUND = "playground"  # Experimentation/testing
    INTERVIEW = "interview"    # Structured Q&A format


class ParticipantRole(str, Enum):
    """Roles that participants can have in a session"""
    HOST = "host"
    MEMBER = "member"
    OBSERVER = "observer"
    MODERATOR = "moderator"


@dataclass
class ContentData:
    """Base class for content data structures"""
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextContentData(ContentData):
    """Text content data"""
    text: str
    format: str = "plain"  # plain, markdown, html
    language: Optional[str] = None  # For code blocks


@dataclass
class ImageContentData(ContentData):
    """Image content data"""
    storage_ref: Optional[str] = None  # Reference to stored image
    url: Optional[str] = None         # External URL
    caption: Optional[str] = None
    mime_type: str = "image/jpeg"
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None


@dataclass
class AudioContentData(ContentData):
    """Audio content data"""
    storage_ref: Optional[str] = None
    url: Optional[str] = None
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None
    mime_type: str = "audio/mpeg"
    size_bytes: Optional[int] = None


@dataclass
class VideoContentData(ContentData):
    """Video content data"""
    storage_ref: Optional[str] = None
    url: Optional[str] = None
    thumbnail_ref: Optional[str] = None
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None
    mime_type: str = "video/mp4"
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None


@dataclass
class ToolCallData(ContentData):
    """Tool call content data"""
    tool_id: str
    function: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


@dataclass
class ToolResponseData(ContentData):
    """Tool response content data"""
    call_id: str
    tool_id: str
    result: Any
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


@dataclass
class MemoryReferenceData(ContentData):
    """Memory reference content data"""
    memory_id: str
    memory_type: str  # episodic, semantic, procedural
    relevance_score: float
    preview: Optional[str] = None  # Short preview of memory content


@dataclass
class MessageV2DTO:
    """
    Enhanced message model supporting polymorphic content.
    
    This model supports all content types needed for advanced chat applications
    while maintaining a clean, extensible structure.
    """
    # Core fields
    id: str
    sender_type: SenderType
    sender_id: str
    content_type: ContentType
    content_data: Dict[str, Any]  # Type-specific content
    
    # Optional sender info
    sender_name: Optional[str] = None
    sender_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Multiple content support
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Tool integration
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_responses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory integration
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    edited_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message after initialization"""
        # Convert string enums if needed
        if isinstance(self.sender_type, str):
            self.sender_type = SenderType(self.sender_type)
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert enums to strings
        data['sender_type'] = self.sender_type.value
        data['content_type'] = self.content_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageV2DTO':
        """Create from dictionary"""
        # Make a copy to avoid modifying original
        data = data.copy()
        
        # Convert string enums
        if 'sender_type' in data and isinstance(data['sender_type'], str):
            data['sender_type'] = SenderType(data['sender_type'])
        if 'content_type' in data and isinstance(data['content_type'], str):
            data['content_type'] = ContentType(data['content_type'])
        
        return cls(**data)
    
    @classmethod
    def from_legacy(cls, legacy_msg: FFMessageDTO) -> 'MessageV2DTO':
        """
        Convert from legacy FFMessageDTO format.
        
        This enables backward compatibility during migration.
        """
        # Determine sender type from role
        if legacy_msg.role == "user":
            sender_type = SenderType.USER
            sender_id = "legacy_user"
        elif legacy_msg.role == "assistant":
            sender_type = SenderType.AGENT
            sender_id = "legacy_assistant"
        elif legacy_msg.role == "system":
            sender_type = SenderType.SYSTEM
            sender_id = "system"
        else:
            sender_type = SenderType.AGENT
            sender_id = legacy_msg.role
        
        # Create v2 message
        return cls(
            id=legacy_msg.message_id,
            sender_type=sender_type,
            sender_id=sender_id,
            sender_name=legacy_msg.role,
            content_type=ContentType.TEXT,
            content_data={"text": legacy_msg.content, "format": "plain"},
            timestamp=legacy_msg.timestamp,
            attachments=[{"file": att} for att in legacy_msg.attachments],
            metadata=legacy_msg.metadata
        )
    
    def to_legacy(self) -> FFMessageDTO:
        """
        Convert to legacy FFMessageDTO format.
        
        This enables using v2 messages with legacy code.
        """
        # Map sender type to role
        if self.sender_type == SenderType.USER:
            role = "user"
        elif self.sender_type == SenderType.SYSTEM:
            role = "system"
        else:
            role = self.sender_name or "assistant"
        
        # Extract text content
        content = ""
        if self.content_type == ContentType.TEXT:
            content = self.content_data.get("text", "")
        else:
            # For non-text, create a description
            content = f"[{self.content_type.value}]"
        
        # Extract attachments
        attachments = [att.get("file", "") for att in self.attachments if "file" in att]
        
        return FFMessageDTO(
            role=role,
            content=content,
            message_id=self.id,
            timestamp=self.timestamp,
            attachments=attachments,
            metadata=self.metadata
        )


@dataclass
class SessionParticipant:
    """Participant in a session"""
    id: str
    type: SenderType
    role: ParticipantRole
    name: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    joined_at: str = field(default_factory=lambda: datetime.now().isoformat())
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['role'] = self.role.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionParticipant':
        data = data.copy()
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = SenderType(data['type'])
        if 'role' in data and isinstance(data['role'], str):
            data['role'] = ParticipantRole(data['role'])
        return cls(**data)


@dataclass
class SessionContext:
    """Context information for a session"""
    summary: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    memory_scope: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        return cls(**data)


@dataclass
class SessionV2DTO:
    """
    Enhanced session model supporting multiple session types.
    
    This model supports chat, panel, workflow, and other session types
    with flexible participant management and context tracking.
    """
    # Core fields
    id: str
    type: SessionType
    title: str
    
    # Participants
    participants: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context and configuration
    context: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # State
    active: bool = True
    message_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate session after initialization"""
        if isinstance(self.type, str):
            self.type = SessionType(self.type)
    
    def add_participant(self, participant: SessionParticipant) -> bool:
        """Add a participant to the session"""
        # Check if already exists
        existing_ids = [p.get("id") for p in self.participants]
        if participant.id in existing_ids:
            return False
        
        self.participants.append(participant.to_dict())
        self.updated_at = datetime.now().isoformat()
        return True
    
    def remove_participant(self, participant_id: str) -> bool:
        """Remove a participant from the session"""
        original_count = len(self.participants)
        self.participants = [p for p in self.participants if p.get("id") != participant_id]
        
        if len(self.participants) < original_count:
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def update_context(self, updates: Dict[str, Any]):
        """Update session context"""
        self.context.update(updates)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionV2DTO':
        """Create from dictionary"""
        data = data.copy()
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = SessionType(data['type'])
        return cls(**data)
    
    @classmethod
    def from_legacy(cls, legacy_session: Any) -> 'SessionV2DTO':
        """Convert from legacy session format"""
        return cls(
            id=legacy_session.session_id,
            type=SessionType.CHAT,
            title=legacy_session.title,
            participants=[{
                "id": legacy_session.user_id,
                "type": "user",
                "role": "member",
                "name": legacy_session.user_id
            }],
            context={},
            capabilities=["text"],
            created_at=legacy_session.created_at,
            updated_at=legacy_session.updated_at,
            message_count=legacy_session.message_count,
            metadata=legacy_session.metadata
        )


# Tool execution tracking
@dataclass
class ToolExecutionDTO:
    """Record of a tool execution"""
    id: str
    session_id: str
    message_id: str
    tool_id: str
    function: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Mark execution as complete"""
        self.completed_at = datetime.now().isoformat()
        self.result = result
        self.error = error
        
        # Calculate duration
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_ms = int((end - start).total_seconds() * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolExecutionDTO':
        return cls(**data)
```

### 2. Memory System Entities (`ff_class_configs/ff_memory_entities.py`)

Create a new file for memory system data models:

```python
"""
Memory system entities for hierarchical memory storage.

Supports episodic, semantic, procedural, and working memory
with decay mechanisms and relevance tracking.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import math


class MemoryType(str, Enum):
    """Types of memory supported"""
    EPISODIC = "episodic"      # Specific events/conversations
    SEMANTIC = "semantic"      # Facts and general knowledge
    PROCEDURAL = "procedural"  # How-to knowledge and procedures
    WORKING = "working"        # Current context/short-term


class MemoryScope(str, Enum):
    """Scope of memory storage"""
    USER = "user"       # User-specific memories
    AGENT = "agent"     # Agent-specific memories
    GLOBAL = "global"   # System-wide memories
    SESSION = "session" # Session-specific memories


@dataclass
class MemoryEntryDTO:
    """
    Storage model for memory entries with decay and relevance.
    
    Memories decay over time but can be reinforced through access.
    Important memories decay slower than less important ones.
    """
    # Identity
    id: str
    type: MemoryType
    scope: MemoryScope = MemoryScope.USER
    scope_id: Optional[str] = None  # User/agent/session ID
    
    # Content
    content: str
    summary: Optional[str] = None  # Brief summary for quick retrieval
    
    # Vector storage reference
    embedding_id: Optional[str] = None
    embedding_model: Optional[str] = None
    
    # Source tracking
    source_type: str = ""  # message, document, external, generated
    source_id: str = ""    # ID of source
    source_context: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal data
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    access_count: int = 0
    last_reinforced: Optional[str] = None
    
    # Relevance and importance
    importance: float = 0.5    # 0-1, how important is this memory
    confidence: float = 1.0    # 0-1, confidence in accuracy
    decay_factor: float = 1.0  # 0-1, current decay level
    base_decay_rate: float = 0.01  # How fast it decays
    
    # Relationships
    related_memories: List[str] = field(default_factory=list)
    parent_memory_id: Optional[str] = None  # For hierarchical memories
    child_memory_ids: List[str] = field(default_factory=list)
    
    # Categorization
    tags: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate memory after initialization"""
        if isinstance(self.type, str):
            self.type = MemoryType(self.type)
        if isinstance(self.scope, str):
            self.scope = MemoryScope(self.scope)
        
        # Validate ranges
        self.importance = max(0.0, min(1.0, self.importance))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.decay_factor = max(0.0, min(1.0, self.decay_factor))
    
    def access(self, boost: float = 0.1):
        """
        Record memory access and boost relevance.
        
        Each access reinforces the memory, slowing decay.
        """
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
        
        # Boost decay factor (make memory stronger)
        self.decay_factor = min(1.0, self.decay_factor + boost)
        
        # High access count reduces decay rate
        access_factor = math.log(self.access_count + 1) / 10
        self.base_decay_rate = max(0.001, self.base_decay_rate - access_factor * 0.001)
    
    def reinforce(self, strength: float = 0.2):
        """
        Reinforce memory with new evidence or usage.
        
        Stronger reinforcement for important memories.
        """
        self.last_reinforced = datetime.now().isoformat()
        boost = strength * self.importance
        self.decay_factor = min(1.0, self.decay_factor + boost)
        self.confidence = min(1.0, self.confidence + strength * 0.1)
    
    def decay(self, time_factor: float = 1.0):
        """
        Apply time-based decay to memory.
        
        Important memories decay slower.
        """
        # Calculate decay based on importance
        importance_modifier = 1.0 - (self.importance * 0.5)
        actual_decay = self.base_decay_rate * importance_modifier * time_factor
        
        # Apply decay
        self.decay_factor = max(0.1, self.decay_factor - actual_decay)
    
    def get_relevance_score(self) -> float:
        """
        Calculate current relevance score.
        
        Combines importance, confidence, and decay.
        """
        return self.importance * self.confidence * self.decay_factor
    
    def merge_with(self, other: 'MemoryEntryDTO'):
        """
        Merge with another related memory.
        
        Used for memory consolidation.
        """
        # Combine content
        if other.content not in self.content:
            self.content += f"\n\nRelated: {other.content}"
        
        # Update importance (take maximum)
        self.importance = max(self.importance, other.importance)
        
        # Average confidence
        self.confidence = (self.confidence + other.confidence) / 2
        
        # Boost decay factor
        self.decay_factor = min(1.0, self.decay_factor + 0.1)
        
        # Merge relationships
        for mem_id in other.related_memories:
            if mem_id not in self.related_memories and mem_id != self.id:
                self.related_memories.append(mem_id)
        
        # Merge tags and entities
        self.tags = list(set(self.tags + other.tags))
        for entity_type, values in other.entities.items():
            if entity_type not in self.entities:
                self.entities[entity_type] = []
            self.entities[entity_type].extend(values)
            self.entities[entity_type] = list(set(self.entities[entity_type]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['type'] = self.type.value
        data['scope'] = self.scope.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntryDTO':
        """Create from dictionary"""
        data = data.copy()
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = MemoryType(data['type'])
        if 'scope' in data and isinstance(data['scope'], str):
            data['scope'] = MemoryScope(data['scope'])
        return cls(**data)


@dataclass
class MemorySearchQuery:
    """Query parameters for memory search"""
    query: str
    memory_types: Optional[List[MemoryType]] = None
    scope: Optional[MemoryScope] = None
    scope_id: Optional[str] = None
    min_relevance: float = 0.5
    max_results: int = 10
    include_decayed: bool = False
    time_range: Optional[Dict[str, str]] = None  # start/end ISO timestamps
    tags: Optional[List[str]] = None
    entities: Optional[Dict[str, List[str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.memory_types:
            data['memory_types'] = [t.value for t in self.memory_types]
        if self.scope:
            data['scope'] = self.scope.value
        return data


@dataclass
class MemoryStats:
    """Statistics about memory storage"""
    total_memories: int = 0
    memories_by_type: Dict[str, int] = field(default_factory=dict)
    memories_by_scope: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    average_relevance: float = 0.0
    average_access_count: float = 0.0
    oldest_memory: Optional[str] = None
    newest_memory: Optional[str] = None
    most_accessed_memory_id: Optional[str] = None
    most_important_memory_id: Optional[str] = None
    decay_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 3. Multimodal Entities (`ff_class_configs/ff_multimodal_entities.py`)

Create entities for multimodal content handling:

```python
"""
Multimodal content entities for images, audio, video, and documents.

Handles metadata, processing results, and storage references.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class ContentStatus(str, Enum):
    """Status of multimodal content processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingType(str, Enum):
    """Types of content processing"""
    OCR = "ocr"                    # Optical character recognition
    TRANSCRIPTION = "transcription" # Audio/video to text
    TRANSLATION = "translation"     # Language translation
    ANALYSIS = "analysis"          # AI analysis
    THUMBNAIL = "thumbnail"        # Thumbnail generation
    COMPRESSION = "compression"    # Size optimization


@dataclass
class MultimodalContentDTO:
    """
    Base model for multimodal content storage.
    
    Tracks content through upload, processing, and retrieval.
    """
    # Identity
    id: str
    content_type: str  # image, audio, video, document
    
    # File information
    original_filename: str
    storage_path: str
    storage_ref: str  # Unique reference for retrieval
    
    # File properties
    mime_type: str
    size_bytes: int
    hash: str  # For deduplication
    
    # Processing status
    status: ContentStatus = ContentStatus.PENDING
    processing_results: Dict[ProcessingType, Dict[str, Any]] = field(default_factory=dict)
    
    # Relationships
    session_id: str = ""
    message_id: Optional[str] = None
    user_id: str = ""
    
    # Timestamps
    uploaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed_at: Optional[str] = None
    last_accessed: Optional[str] = None
    access_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ContentStatus(self.status)
    
    def add_processing_result(self, proc_type: ProcessingType, result: Dict[str, Any]):
        """Add processing result"""
        self.processing_results[proc_type] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.status = ContentStatus.COMPLETED
        self.processed_at = datetime.now().isoformat()
    
    def mark_failed(self, error: str):
        """Mark processing as failed"""
        self.status = ContentStatus.FAILED
        self.metadata["error"] = error
        self.processed_at = datetime.now().isoformat()
    
    def record_access(self):
        """Record content access"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        if self.processing_results:
            data['processing_results'] = {
                k.value: v for k, v in self.processing_results.items()
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultimodalContentDTO':
        data = data.copy()
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ContentStatus(data['status'])
        if 'processing_results' in data:
            # Convert string keys back to enums
            proc_results = {}
            for k, v in data['processing_results'].items():
                proc_results[ProcessingType(k)] = v
            data['processing_results'] = proc_results
        return cls(**data)


@dataclass
class ImageMetadata(MultimodalContentDTO):
    """Extended metadata for images"""
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None  # JPEG, PNG, etc.
    color_mode: Optional[str] = None  # RGB, CMYK, etc.
    has_transparency: bool = False
    exif_data: Dict[str, Any] = field(default_factory=dict)
    
    # AI analysis results
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    detected_text: List[Dict[str, Any]] = field(default_factory=list)
    detected_faces: List[Dict[str, Any]] = field(default_factory=list)
    scene_description: Optional[str] = None
    
    # Thumbnail
    thumbnail_ref: Optional[str] = None
    thumbnail_size: Optional[Dict[str, int]] = None  # width, height


@dataclass
class AudioMetadata(MultimodalContentDTO):
    """Extended metadata for audio files"""
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    
    # Transcription
    transcript: Optional[str] = None
    transcript_language: Optional[str] = None
    transcript_confidence: Optional[float] = None
    
    # Audio analysis
    detected_speakers: List[Dict[str, Any]] = field(default_factory=list)
    emotion_analysis: Dict[str, float] = field(default_factory=dict)
    background_noise_level: Optional[float] = None
    
    # Waveform data for visualization
    waveform_ref: Optional[str] = None


@dataclass
class VideoMetadata(MultimodalContentDTO):
    """Extended metadata for video files"""
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None
    
    # Thumbnails
    thumbnail_ref: Optional[str] = None
    preview_refs: List[str] = field(default_factory=list)  # Multiple preview images
    
    # Transcription
    transcript: Optional[str] = None
    transcript_language: Optional[str] = None
    captions: List[Dict[str, Any]] = field(default_factory=list)  # Timed captions
    
    # Video analysis
    scene_changes: List[float] = field(default_factory=list)  # Timestamps
    detected_objects_timeline: List[Dict[str, Any]] = field(default_factory=list)
    key_frames: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentMetadata(MultimodalContentDTO):
    """Extended metadata for documents (PDF, Word, etc.)"""
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    
    # Text extraction
    extracted_text: Optional[str] = None
    text_extraction_method: Optional[str] = None  # OCR, native, etc.
    
    # Document structure
    table_of_contents: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata extraction
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Preview
    preview_refs: List[str] = field(default_factory=list)  # Page previews


@dataclass
class ContentProcessingRequest:
    """Request to process multimodal content"""
    content_id: str
    processing_types: List[ProcessingType]
    priority: int = 5  # 1-10, higher is more urgent
    options: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['processing_types'] = [pt.value for pt in self.processing_types]
        return data
```

### 4. Trace System Entities (`ff_class_configs/ff_trace_entities.py`)

Create entities for comprehensive trace logging:

```python
"""
Trace system entities for debugging and audit logging.

Provides detailed tracking of system operations, performance metrics,
and decision paths for debugging and analysis.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class TraceLevel(str, Enum):
    """Levels of trace detail"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TraceEventType(str, Enum):
    """Types of trace events"""
    # Message flow
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    MESSAGE_STORED = "message_stored"
    MESSAGE_FAILED = "message_failed"
    
    # Agent operations
    AGENT_INVOKED = "agent_invoked"
    AGENT_RESPONDED = "agent_responded"
    AGENT_FAILED = "agent_failed"
    AGENT_ROUTING = "agent_routing"
    
    # Tool operations
    TOOL_CALLED = "tool_called"
    TOOL_EXECUTED = "tool_executed"
    TOOL_FAILED = "tool_failed"
    
    # Memory operations
    MEMORY_SEARCHED = "memory_searched"
    MEMORY_STORED = "memory_stored"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_DECAYED = "memory_decayed"
    
    # System operations
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    
    # Performance
    PERFORMANCE_METRIC = "performance_metric"
    SLOW_OPERATION = "slow_operation"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class TraceEvent:
    """
    Individual trace event with context and timing.
    
    Forms the basis of the trace logging system.
    """
    # Identity
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Event details
    event_type: TraceEventType
    level: TraceLevel = TraceLevel.INFO
    component: str = ""  # Component that generated the event
    operation: str = ""  # Specific operation being traced
    
    # Context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    message_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Performance
    duration_ms: Optional[int] = None
    
    # Relationships
    parent_event_id: Optional[str] = None
    correlation_id: Optional[str] = None  # For tracking across services
    
    # Error information
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = TraceEventType(self.event_type)
        if isinstance(self.level, str):
            self.level = TraceLevel(self.level)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceEvent':
        data = data.copy()
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = TraceEventType(data['event_type'])
        if 'level' in data and isinstance(data['level'], str):
            data['level'] = TraceLevel(data['level'])
        return cls(**data)


@dataclass
class TraceSpan:
    """
    Trace span representing a complete operation.
    
    Contains multiple events and timing information.
    """
    # Identity
    id: str = field(default_factory=lambda: f"span_{uuid.uuid4().hex[:8]}")
    name: str = ""
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[int] = None
    
    # Context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Relationships
    parent_span_id: Optional[str] = None
    
    # Data
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)  # Event IDs
    
    # Status
    status: str = "in_progress"  # in_progress, completed, failed
    error: Optional[str] = None
    
    def complete(self, error: Optional[str] = None):
        """Mark span as complete"""
        self.end_time = datetime.now().isoformat()
        
        # Calculate duration
        if self.start_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            self.duration_ms = int((end - start).total_seconds() * 1000)
        
        self.status = "failed" if error else "completed"
        self.error = error
    
    def add_event(self, event_id: str):
        """Add event to span"""
        if event_id not in self.events:
            self.events.append(event_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceSpan':
        return cls(**data)


@dataclass
class PerformanceMetric:
    """Performance metric for system monitoring"""
    name: str
    value: float
    unit: str  # ms, bytes, count, etc.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    component: str = ""
    operation: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceContext:
    """Context for distributed tracing"""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    flags: int = 0  # Trace flags (sampled, debug, etc.)
    baggage: Dict[str, str] = field(default_factory=dict)  # Propagated context
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceContext':
        return cls(**data)
    
    def create_child_context(self) -> 'TraceContext':
        """Create child context for nested operations"""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            span_id=uuid.uuid4().hex[:16],
            flags=self.flags,
            baggage=self.baggage.copy()
        )


@dataclass
class TraceSession:
    """Complete trace session for analysis"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    spans: List[TraceSpan] = field(default_factory=list)
    events: List[TraceEvent] = field(default_factory=list)
    metrics: List[PerformanceMetric] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: TraceSpan):
        """Add span to session"""
        self.spans.append(span)
    
    def add_event(self, event: TraceEvent):
        """Add event to session"""
        self.events.append(event)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add performance metric"""
        self.metrics.append(metric)
    
    def generate_summary(self):
        """Generate session summary"""
        self.summary = {
            "total_spans": len(self.spans),
            "total_events": len(self.events),
            "event_types": {},
            "error_count": 0,
            "warning_count": 0,
            "total_duration_ms": 0,
            "slowest_operations": []
        }
        
        # Count event types
        for event in self.events:
            event_type = event.event_type.value
            self.summary["event_types"][event_type] = \
                self.summary["event_types"].get(event_type, 0) + 1
            
            # Count errors and warnings
            if event.level == TraceLevel.ERROR:
                self.summary["error_count"] += 1
            elif event.level == TraceLevel.WARNING:
                self.summary["warning_count"] += 1
        
        # Calculate total duration and find slow operations
        slow_threshold = 1000  # 1 second
        for span in self.spans:
            if span.duration_ms:
                self.summary["total_duration_ms"] += span.duration_ms
                
                if span.duration_ms > slow_threshold:
                    self.summary["slowest_operations"].append({
                        "name": span.name,
                        "duration_ms": span.duration_ms
                    })
        
        # Sort slowest operations
        self.summary["slowest_operations"].sort(
            key=lambda x: x["duration_ms"],
            reverse=True
        )
        self.summary["slowest_operations"] = self.summary["slowest_operations"][:10]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert objects to dicts
        data['spans'] = [s.to_dict() for s in self.spans]
        data['events'] = [e.to_dict() for e in self.events]
        data['metrics'] = [m.to_dict() for m in self.metrics]
        return data
```

### 5. Agent System Entities (`ff_class_configs/ff_agent_entities.py`)

Create entities for agent configurations:

```python
"""
Agent system entities for multi-agent support.

Defines agent configurations, capabilities, and routing rules.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum


class AgentType(str, Enum):
    """Types of agents supported"""
    CONVERSATIONAL = "conversational"  # General conversation
    TASK_ORIENTED = "task_oriented"    # Specific task completion
    ANALYTICAL = "analytical"          # Data analysis and insights
    CREATIVE = "creative"              # Creative tasks
    SPECIALIZED = "specialized"        # Domain-specific expert
    ORCHESTRATOR = "orchestrator"      # Manages other agents


class AgentCapability(str, Enum):
    """Capabilities that agents can have"""
    TEXT_GENERATION = "text_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_ANALYSIS = "video_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    TOOL_USE = "tool_use"
    MEMORY_ACCESS = "memory_access"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    FACT_CHECKING = "fact_checking"


@dataclass
class AgentPromptTemplate:
    """Template for agent prompts"""
    system_prompt: str
    user_prompt_template: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[Dict[str, Any]] = None
    variables: List[str] = field(default_factory=list)  # Required variables
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def format(self, **kwargs) -> Dict[str, str]:
        """Format prompts with variables"""
        formatted = {
            "system": self.system_prompt.format(**kwargs)
        }
        
        if self.user_prompt_template:
            formatted["user"] = self.user_prompt_template.format(**kwargs)
        
        return formatted


@dataclass
class AgentConfiguration:
    """
    Complete configuration for an agent.
    
    Defines capabilities, behavior, and constraints.
    """
    # Identity
    id: str
    name: str
    type: AgentType
    description: str = ""
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Knowledge and expertise
    knowledge_domains: List[str] = field(default_factory=list)
    expertise_keywords: List[str] = field(default_factory=list)  # For routing
    
    # Prompting
    prompt_template: Optional[Dict[str, Any]] = None
    
    # Model configuration
    model_provider: str = "openai"  # openai, anthropic, local, etc.
    model_name: str = "gpt-4"
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Behavior
    personality_traits: Dict[str, float] = field(default_factory=dict)  # trait -> strength
    communication_style: Dict[str, Any] = field(default_factory=dict)
    response_format: Optional[str] = None  # json, markdown, plain
    
    # Access control
    allowed_tools: List[str] = field(default_factory=list)
    memory_access: Dict[str, Any] = field(default_factory=dict)  # Scope and permissions
    data_access: List[str] = field(default_factory=list)  # Data sources
    
    # Routing rules
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    confidence_threshold: float = 0.7  # Minimum confidence to handle request
    
    # Resource limits
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300
    rate_limit: Optional[Dict[str, int]] = None  # requests per time period
    
    # State
    is_active: bool = True
    is_available: bool = True
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string enums
        if isinstance(self.type, str):
            self.type = AgentType(self.type)
        
        if self.capabilities:
            self.capabilities = [
                AgentCapability(cap) if isinstance(cap, str) else cap
                for cap in self.capabilities
            ]
    
    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Calculate confidence score for handling a request.
        
        Returns confidence score 0-1.
        """
        score = 0.0
        
        # Check capabilities match
        required_capabilities = request.get("required_capabilities", [])
        for cap in required_capabilities:
            if cap in [c.value for c in self.capabilities]:
                score += 0.3
        
        # Check domain match
        topic = request.get("topic", "").lower()
        for domain in self.knowledge_domains:
            if domain.lower() in topic:
                score += 0.4
        
        # Check keyword match
        content = request.get("content", "").lower()
        keyword_matches = sum(
            1 for keyword in self.expertise_keywords
            if keyword.lower() in content
        )
        if keyword_matches > 0:
            score += min(0.3, keyword_matches * 0.1)
        
        return min(1.0, score)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['capabilities'] = [cap.value for cap in self.capabilities]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfiguration':
        data = data.copy()
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = AgentType(data['type'])
        if 'capabilities' in data:
            data['capabilities'] = [
                AgentCapability(cap) if isinstance(cap, str) else cap
                for cap in data['capabilities']
            ]
        return cls(**data)


@dataclass
class AgentRoutingDecision:
    """Decision made by routing system"""
    agent_id: str
    confidence: float
    reasoning: str
    context_segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentExecutionContext:
    """Context provided to agent for execution"""
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Content
    content: str = ""
    content_type: str = "text"
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    max_response_tokens: Optional[int] = None
    response_format: Optional[str] = None
    time_limit_seconds: Optional[int] = None
    
    # Tools
    available_tools: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_id: str
    request_id: str
    
    # Response content
    content: str
    content_type: str = "text"
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tool usage
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0
    reasoning: Optional[str] = None
    tokens_used: Optional[int] = None
    execution_time_ms: Optional[int] = None
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    # Follow-up
    suggested_next_agents: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentCollaborationRequest:
    """Request for multi-agent collaboration"""
    id: str
    initiator_agent_id: str
    target_agent_ids: List[str]
    
    # Task
    task_description: str
    task_type: str = "discussion"  # discussion, consensus, delegation
    
    # Context
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    max_rounds: int = 5
    consensus_threshold: float = 0.8
    timeout_seconds: int = 300
    
    # Results
    responses: List[AgentResponse] = field(default_factory=list)
    final_output: Optional[str] = None
    consensus_reached: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['responses'] = [r.to_dict() for r in self.responses]
        return data
```

### 6. Protocol Definitions

Create new protocol files for the storage system:

#### Memory Protocol (`ff_protocols/ff_memory_protocol.py`)

```python
"""
Memory storage protocol for hierarchical memory system.
"""

from typing import Protocol, List, Optional, Dict, Any
from ff_class_configs.ff_memory_entities import (
    MemoryEntryDTO, MemoryType, MemoryScope,
    MemorySearchQuery, MemoryStats
)


class MemoryStoreProtocol(Protocol):
    """Protocol for memory storage operations"""
    
    async def store_memory(
        self,
        memory: MemoryEntryDTO,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None
    ) -> bool:
        """
        Store a memory entry.
        
        Args:
            memory: Memory entry to store
            scope: Scope of memory (user, agent, global, session)
            scope_id: ID for scoped storage (user_id, agent_id, etc.)
            
        Returns:
            True if successful
        """
        ...
    
    async def get_memory(
        self,
        memory_id: str,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None
    ) -> Optional[MemoryEntryDTO]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: Memory identifier
            scope: Memory scope
            scope_id: Scope identifier
            
        Returns:
            Memory entry or None if not found
        """
        ...
    
    async def search_memories(
        self,
        query: MemorySearchQuery
    ) -> List[MemoryEntryDTO]:
        """
        Search memories using semantic similarity and filters.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of matching memories sorted by relevance
        """
        ...
    
    async def update_memory_access(
        self,
        memory_id: str,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None
    ) -> bool:
        """
        Update memory access patterns (count, timestamp, boost).
        
        Args:
            memory_id: Memory identifier
            scope: Memory scope
            scope_id: Scope identifier
            
        Returns:
            True if successful
        """
        ...
    
    async def decay_memories(
        self,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None,
        decay_rate: Optional[float] = None
    ) -> int:
        """
        Apply time-based decay to memories.
        
        Args:
            scope: Memory scope
            scope_id: Scope identifier
            decay_rate: Decay rate to apply (uses default if None)
            
        Returns:
            Number of memories decayed
        """
        ...
    
    async def consolidate_memories(
        self,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None,
        similarity_threshold: float = 0.8
    ) -> int:
        """
        Consolidate similar memories to prevent redundancy.
        
        Args:
            scope: Memory scope
            scope_id: Scope identifier
            similarity_threshold: Minimum similarity for consolidation
            
        Returns:
            Number of memories consolidated
        """
        ...
    
    async def get_memory_stats(
        self,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None
    ) -> MemoryStats:
        """
        Get statistics about memory storage.
        
        Args:
            scope: Memory scope
            scope_id: Scope identifier
            
        Returns:
            Memory statistics
        """
        ...
    
    async def export_memories(
        self,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None,
        format: str = "json"
    ) -> bytes:
        """
        Export memories for backup or analysis.
        
        Args:
            scope: Memory scope
            scope_id: Scope identifier
            format: Export format (json, csv, etc.)
            
        Returns:
            Exported data as bytes
        """
        ...
    
    async def import_memories(
        self,
        data: bytes,
        scope: MemoryScope = MemoryScope.USER,
        scope_id: Optional[str] = None,
        format: str = "json"
    ) -> int:
        """
        Import memories from backup.
        
        Args:
            data: Memory data to import
            scope: Memory scope
            scope_id: Scope identifier
            format: Data format
            
        Returns:
            Number of memories imported
        """
        ...
```

#### Multimodal Protocol (`ff_protocols/ff_multimodal_protocol.py`)

```python
"""
Multimodal content storage protocol.
"""

from typing import Protocol, Optional, Dict, Any, List, BinaryIO
from pathlib import Path
from ff_class_configs.ff_multimodal_entities import (
    MultimodalContentDTO, ContentStatus, ProcessingType,
    ImageMetadata, AudioMetadata, VideoMetadata, DocumentMetadata,
    ContentProcessingRequest
)


class MultimodalStorageProtocol(Protocol):
    """Protocol for multimodal content storage and retrieval"""
    
    async def store_content(
        self,
        content_type: str,  # image, audio, video, document
        data: bytes,
        filename: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store multimodal content with deduplication.
        
        Args:
            content_type: Type of content
            data: Content data
            filename: Original filename
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Content ID for retrieval
        """
        ...
    
    async def get_content(
        self,
        content_id: str
    ) -> Optional[bytes]:
        """
        Retrieve content by ID.
        
        Args:
            content_id: Content identifier
            
        Returns:
            Content data or None if not found
        """
        ...
    
    async def get_content_metadata(
        self,
        content_id: str
    ) -> Optional[MultimodalContentDTO]:
        """
        Get content metadata without retrieving the actual content.
        
        Args:
            content_id: Content identifier
            
        Returns:
            Content metadata or None
        """
        ...
    
    async def stream_content(
        self,
        content_id: str,
        chunk_size: int = 8192
    ) -> Optional[BinaryIO]:
        """
        Stream content for large files.
        
        Args:
            content_id: Content identifier
            chunk_size: Size of chunks to stream
            
        Returns:
            Binary stream or None
        """
        ...
    
    async def process_content(
        self,
        request: ContentProcessingRequest
    ) -> bool:
        """
        Submit content for processing (OCR, transcription, etc.).
        
        Args:
            request: Processing request
            
        Returns:
            True if processing started successfully
        """
        ...
    
    async def get_processing_status(
        self,
        content_id: str
    ) -> ContentStatus:
        """
        Get current processing status.
        
        Args:
            content_id: Content identifier
            
        Returns:
            Processing status
        """
        ...
    
    async def get_processing_results(
        self,
        content_id: str,
        processing_type: Optional[ProcessingType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get processing results.
        
        Args:
            content_id: Content identifier
            processing_type: Specific processing type or all
            
        Returns:
            Processing results or None
        """
        ...
    
    async def update_content_metadata(
        self,
        content_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update content metadata.
        
        Args:
            content_id: Content identifier
            metadata_updates: Metadata to update
            
        Returns:
            True if successful
        """
        ...
    
    async def delete_content(
        self,
        content_id: str
    ) -> bool:
        """
        Delete content and its metadata.
        
        Args:
            content_id: Content identifier
            
        Returns:
            True if successful
        """
        ...
    
    async def search_content(
        self,
        content_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None,
        limit: int = 100
    ) -> List[MultimodalContentDTO]:
        """
        Search for content by various criteria.
        
        Args:
            content_type: Filter by type
            session_id: Filter by session
            tags: Filter by tags
            date_range: Filter by upload date
            limit: Maximum results
            
        Returns:
            List of matching content metadata
        """
        ...
    
    async def get_storage_stats(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Storage statistics
        """
        ...
```

#### Trace Protocol (`ff_protocols/ff_trace_protocol.py`)

```python
"""
Trace logging protocol for debugging and auditing.
"""

from typing import Protocol, List, Optional, Dict, Any
from ff_class_configs.ff_trace_entities import (
    TraceEvent, TraceSpan, TraceContext,
    PerformanceMetric, TraceSession
)


class TraceLoggerProtocol(Protocol):
    """Protocol for trace logging operations"""
    
    async def log_event(
        self,
        event: TraceEvent
    ) -> bool:
        """
        Log a trace event.
        
        Args:
            event: Trace event to log
            
        Returns:
            True if successful
        """
        ...
    
    async def start_span(
        self,
        name: str,
        context: Optional[TraceContext] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """
        Start a new trace span.
        
        Args:
            name: Span name
            context: Trace context
            tags: Span tags
            
        Returns:
            New trace span
        """
        ...
    
    async def end_span(
        self,
        span: TraceSpan,
        error: Optional[str] = None
    ) -> bool:
        """
        End a trace span.
        
        Args:
            span: Span to end
            error: Optional error message
            
        Returns:
            True if successful
        """
        ...
    
    async def log_metric(
        self,
        metric: PerformanceMetric
    ) -> bool:
        """
        Log a performance metric.
        
        Args:
            metric: Performance metric
            
        Returns:
            True if successful
        """
        ...
    
    async def get_trace_events(
        self,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[TraceEvent]:
        """
        Retrieve trace events with filters.
        
        Args:
            session_id: Filter by session
            start_time: Start time filter
            end_time: End time filter
            event_types: Filter by event types
            limit: Maximum events to return
            
        Returns:
            List of trace events
        """
        ...
    
    async def get_trace_spans(
        self,
        session_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        min_duration_ms: Optional[int] = None
    ) -> List[TraceSpan]:
        """
        Retrieve trace spans.
        
        Args:
            session_id: Filter by session
            parent_span_id: Filter by parent
            min_duration_ms: Minimum duration filter
            
        Returns:
            List of trace spans
        """
        ...
    
    async def get_performance_metrics(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """
        Retrieve performance metrics.
        
        Args:
            component: Filter by component
            operation: Filter by operation
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of performance metrics
        """
        ...
    
    async def get_trace_session(
        self,
        session_id: str
    ) -> Optional[TraceSession]:
        """
        Get complete trace session for analysis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Trace session or None
        """
        ...
    
    async def export_traces(
        self,
        session_id: Optional[str] = None,
        format: str = "json"
    ) -> bytes:
        """
        Export traces for analysis.
        
        Args:
            session_id: Optional session filter
            format: Export format
            
        Returns:
            Exported trace data
        """
        ...
```

#### Agent Protocol (`ff_protocols/ff_agent_protocol.py`)

```python
"""
Agent management protocol for multi-agent systems.
"""

from typing import Protocol, List, Optional, Dict, Any
from ff_class_configs.ff_agent_entities import (
    AgentConfiguration, AgentExecutionContext,
    AgentResponse, AgentRoutingDecision,
    AgentCollaborationRequest
)


class AgentManagerProtocol(Protocol):
    """Protocol for agent management operations"""
    
    async def register_agent(
        self,
        config: AgentConfiguration
    ) -> bool:
        """
        Register a new agent.
        
        Args:
            config: Agent configuration
            
        Returns:
            True if successful
        """
        ...
    
    async def get_agent(
        self,
        agent_id: str
    ) -> Optional[AgentConfiguration]:
        """
        Get agent configuration.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent configuration or None
        """
        ...
    
    async def update_agent(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update agent configuration.
        
        Args:
            agent_id: Agent identifier
            updates: Configuration updates
            
        Returns:
            True if successful
        """
        ...
    
    async def list_agents(
        self,
        agent_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        is_active: Optional[bool] = True
    ) -> List[AgentConfiguration]:
        """
        List agents with filters.
        
        Args:
            agent_type: Filter by type
            capabilities: Required capabilities
            is_active: Filter by active status
            
        Returns:
            List of agent configurations
        """
        ...
    
    async def route_request(
        self,
        request: Dict[str, Any]
    ) -> List[AgentRoutingDecision]:
        """
        Route request to appropriate agents.
        
        Args:
            request: Request to route
            
        Returns:
            List of routing decisions
        """
        ...
    
    async def execute_agent(
        self,
        agent_id: str,
        context: AgentExecutionContext
    ) -> AgentResponse:
        """
        Execute agent with given context.
        
        Args:
            agent_id: Agent identifier
            context: Execution context
            
        Returns:
            Agent response
        """
        ...
    
    async def collaborate_agents(
        self,
        request: AgentCollaborationRequest
    ) -> Dict[str, Any]:
        """
        Coordinate multi-agent collaboration.
        
        Args:
            request: Collaboration request
            
        Returns:
            Collaboration results
        """
        ...
    
    async def get_agent_metrics(
        self,
        agent_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Agent metrics
        """
        ...
```

### 7. Configuration Updates (`ff_class_configs/ff_storage_config.py`)

Update the existing storage configuration to support new features:

```python
# Add these imports at the top
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add these new configuration sections to the existing StorageConfigDTO

@dataclass
class MultimodalConfigDTO:
    """Configuration for multimodal content handling"""
    enabled: bool = True
    storage_path: str = "multimodal"
    
    # Size limits (in MB)
    max_image_size_mb: int = 10
    max_audio_size_mb: int = 50
    max_video_size_mb: int = 200
    max_document_size_mb: int = 50
    
    # Allowed formats
    allowed_image_formats: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"
    ])
    allowed_audio_formats: List[str] = field(default_factory=lambda: [
        "mp3", "wav", "m4a", "ogg", "flac", "aac"
    ])
    allowed_video_formats: List[str] = field(default_factory=lambda: [
        "mp4", "webm", "mov", "avi", "mkv", "flv"
    ])
    allowed_document_formats: List[str] = field(default_factory=lambda: [
        "pdf", "doc", "docx", "txt", "rtf", "odt"
    ])
    
    # Processing
    enable_deduplication: bool = True
    enable_compression: bool = True
    compression_quality: int = 85  # For images
    
    # Thumbnails
    generate_thumbnails: bool = True
    thumbnail_size: Dict[str, int] = field(default_factory=lambda: {
        "width": 200,
        "height": 200
    })


@dataclass
class MemoryConfigDTO:
    """Configuration for memory system"""
    enabled: bool = True
    storage_path: str = "memory"
    
    # Capacity
    capacity_per_user: int = 10000
    capacity_per_agent: int = 5000
    global_capacity: int = 100000
    
    # Decay settings
    enable_decay: bool = True
    base_decay_rate: float = 0.01
    decay_check_interval_hours: int = 24
    min_decay_factor: float = 0.1
    
    # Consolidation
    enable_consolidation: bool = True
    consolidation_threshold: float = 0.8
    consolidation_interval_hours: int = 168  # Weekly
    
    # Memory types
    enabled_types: List[str] = field(default_factory=lambda: [
        "episodic", "semantic", "procedural", "working"
    ])
    
    # Compression
    compression_threshold: float = 0.9  # Compress when 90% full
    compression_ratio: float = 0.8     # Keep 80% of memories


@dataclass
class TraceConfigDTO:
    """Configuration for trace logging"""
    enabled: bool = True
    storage_path: str = "traces"
    
    # Logging levels
    log_level: str = "info"  # debug, info, warning, error
    log_performance_metrics: bool = True
    log_slow_operations: bool = True
    slow_operation_threshold_ms: int = 1000
    
    # Storage
    buffer_size: int = 100  # Events before flush
    flush_interval_seconds: int = 30
    retention_days: int = 30
    
    # Sampling
    enable_sampling: bool = False
    sampling_rate: float = 1.0  # 1.0 = 100%
    
    # Export
    export_format: str = "json"  # json, csv, otlp


@dataclass
class AgentConfigDTO:
    """Configuration for agent system"""
    enabled: bool = True
    storage_path: str = "agents"
    
    # Routing
    enable_topic_routing: bool = True
    routing_confidence_threshold: float = 0.7
    max_agents_per_request: int = 5
    
    # Execution
    default_timeout_seconds: int = 300
    max_concurrent_executions: int = 50
    enable_rate_limiting: bool = True
    
    # Collaboration
    max_collaboration_rounds: int = 10
    consensus_threshold: float = 0.8
    
    # Models
    default_model_provider: str = "openai"
    default_model_name: str = "gpt-4"
    fallback_model_name: str = "gpt-3.5-turbo"


@dataclass
class ToolConfigDTO:
    """Configuration for tool execution"""
    enabled: bool = True
    storage_path: str = "tools"
    
    # Execution
    default_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    
    # Limits
    max_executions_per_session: int = 100
    max_executions_per_message: int = 10
    
    # Security
    enable_sandboxing: bool = True
    allowed_tools: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)


# Update the main StorageConfigDTO to include new configs
@dataclass
class StorageConfigDTO:
    """Enhanced storage configuration for v2"""
    # ... existing fields ...
    
    # New configuration sections
    multimodal: MultimodalConfigDTO = field(default_factory=MultimodalConfigDTO)
    memory: MemoryConfigDTO = field(default_factory=MemoryConfigDTO)
    trace: TraceConfigDTO = field(default_factory=TraceConfigDTO)
    agents: AgentConfigDTO = field(default_factory=AgentConfigDTO)
    tools: ToolConfigDTO = field(default_factory=ToolConfigDTO)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "multimodal": True,
        "memory": True,
        "trace": True,
        "agents": True,
        "tools": True,
        "legacy_compatibility": True
    })
```

### 8. Entity Converters (`ff_utils/ff_entity_converters.py`)

Create utility functions for converting between v1 and v2 entities:

```python
"""
Entity conversion utilities for v1 to v2 migration.

Provides bidirectional conversion between legacy and new entity formats.
"""

from typing import Dict, Any, List, Optional
from ff_class_configs.ff_chat_entities import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO,
    FFUserProfileDTO, FFPanelDTO
)
from ff_class_configs.ff_chat_entities_v2 import (
    MessageV2DTO, SessionV2DTO, ContentType, SenderType,
    SessionType, ParticipantRole, SessionParticipant
)


class EntityConverter:
    """Handles conversion between v1 and v2 entities"""
    
    @staticmethod
    def message_v1_to_v2(v1_msg: FFMessageDTO) -> MessageV2DTO:
        """Convert v1 message to v2 format"""
        # Determine sender type
        sender_type = SenderType.USER if v1_msg.role == "user" else SenderType.AGENT
        if v1_msg.role == "system":
            sender_type = SenderType.SYSTEM
        
        # Extract any persona/agent ID from metadata
        sender_id = v1_msg.metadata.get("sender_id", v1_msg.role)
        
        # Convert attachments
        attachments = []
        for att in v1_msg.attachments:
            attachments.append({
                "type": "file",
                "path": att,
                "filename": att.split("/")[-1] if "/" in att else att
            })
        
        return MessageV2DTO(
            id=v1_msg.message_id,
            sender_type=sender_type,
            sender_id=sender_id,
            sender_name=v1_msg.role,
            content_type=ContentType.TEXT,
            content_data={
                "text": v1_msg.content,
                "format": "plain"
            },
            attachments=attachments,
            timestamp=v1_msg.timestamp,
            metadata=v1_msg.metadata
        )
    
    @staticmethod
    def message_v2_to_v1(v2_msg: MessageV2DTO) -> FFMessageDTO:
        """Convert v2 message to v1 format (lossy)"""
        # Map sender type to role
        role_map = {
            SenderType.USER: "user",
            SenderType.AGENT: "assistant",
            SenderType.SYSTEM: "system",
            SenderType.TOOL: "tool"
        }
        role = role_map.get(v2_msg.sender_type, "assistant")
        
        # Extract text content
        content = ""
        if v2_msg.content_type == ContentType.TEXT:
            content = v2_msg.content_data.get("text", "")
        else:
            # Create placeholder for non-text content
            content = f"[{v2_msg.content_type.value}: {v2_msg.id}]"
        
        # Extract file attachments
        attachments = []
        for att in v2_msg.attachments:
            if att.get("type") == "file" and "path" in att:
                attachments.append(att["path"])
        
        return FFMessageDTO(
            role=role,
            content=content,
            message_id=v2_msg.id,
            timestamp=v2_msg.timestamp,
            attachments=attachments,
            metadata=v2_msg.metadata
        )
    
    @staticmethod
    def session_v1_to_v2(v1_session: FFSessionDTO) -> SessionV2DTO:
        """Convert v1 session to v2 format"""
        # Create default participant
        participant = SessionParticipant(
            id=v1_session.user_id,
            type=SenderType.USER,
            role=ParticipantRole.MEMBER,
            name=v1_session.user_id,
            capabilities=["text"]
        )
        
        return SessionV2DTO(
            id=v1_session.session_id,
            type=SessionType.CHAT,
            title=v1_session.title,
            participants=[participant.to_dict()],
            context={},
            capabilities=["text"],
            created_at=v1_session.created_at,
            updated_at=v1_session.updated_at,
            message_count=v1_session.message_count,
            metadata=v1_session.metadata
        )
    
    @staticmethod
    def session_v2_to_v1(v2_session: SessionV2DTO) -> FFSessionDTO:
        """Convert v2 session to v1 format (lossy)"""
        # Extract primary user from participants
        user_id = "unknown"
        for participant in v2_session.participants:
            if participant.get("type") == "user":
                user_id = participant.get("id", "unknown")
                break
        
        return FFSessionDTO(
            session_id=v2_session.id,
            user_id=user_id,
            title=v2_session.title,
            created_at=v2_session.created_at,
            updated_at=v2_session.updated_at,
            message_count=v2_session.message_count,
            metadata=v2_session.metadata
        )
    
    @staticmethod
    def panel_to_v2_session(panel: FFPanelDTO) -> SessionV2DTO:
        """Convert panel to v2 session format"""
        # Create participants from personas
        participants = []
        for persona_id in panel.personas:
            participant = SessionParticipant(
                id=persona_id,
                type=SenderType.AGENT,
                role=ParticipantRole.MEMBER,
                name=persona_id,
                capabilities=["text"]
            )
            participants.append(participant.to_dict())
        
        return SessionV2DTO(
            id=panel.id,
            type=SessionType.PANEL,
            title=f"Panel: {panel.type}",
            participants=participants,
            context={
                "panel_type": panel.type,
                "config": panel.config
            },
            capabilities=["text", "multi_agent"],
            created_at=panel.created_at,
            updated_at=panel.updated_at,
            message_count=panel.message_count,
            metadata=panel.metadata
        )
    
    @staticmethod
    def extract_multimodal_content(v1_msg: FFMessageDTO) -> List[Dict[str, Any]]:
        """
        Extract potential multimodal content from v1 message.
        
        Looks for patterns in content and attachments that might
        indicate images, audio, etc.
        """
        multimodal_content = []
        
        # Check attachments
        for attachment in v1_msg.attachments:
            # Guess content type from extension
            ext = attachment.split(".")[-1].lower() if "." in attachment else ""
            
            if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
                multimodal_content.append({
                    "type": "image",
                    "path": attachment,
                    "filename": attachment.split("/")[-1]
                })
            elif ext in ["mp3", "wav", "m4a", "ogg"]:
                multimodal_content.append({
                    "type": "audio",
                    "path": attachment,
                    "filename": attachment.split("/")[-1]
                })
            elif ext in ["mp4", "webm", "mov"]:
                multimodal_content.append({
                    "type": "video",
                    "path": attachment,
                    "filename": attachment.split("/")[-1]
                })
            elif ext in ["pdf", "doc", "docx", "txt"]:
                multimodal_content.append({
                    "type": "document",
                    "path": attachment,
                    "filename": attachment.split("/")[-1]
                })
        
        # Check for embedded references in content
        # e.g., "[Image: path/to/image.jpg]"
        import re
        pattern = r'\[(Image|Audio|Video|Document):\s*([^\]]+)\]'
        matches = re.findall(pattern, v1_msg.content)
        
        for content_type, path in matches:
            multimodal_content.append({
                "type": content_type.lower(),
                "path": path.strip(),
                "embedded": True
            })
        
        return multimodal_content


def convert_message_batch(
    messages: List[FFMessageDTO],
    direction: str = "v1_to_v2"
) -> List[MessageV2DTO]:
    """
    Convert a batch of messages.
    
    Args:
        messages: List of messages to convert
        direction: "v1_to_v2" or "v2_to_v1"
        
    Returns:
        List of converted messages
    """
    converter = EntityConverter()
    converted = []
    
    for msg in messages:
        try:
            if direction == "v1_to_v2":
                converted.append(converter.message_v1_to_v2(msg))
            else:
                converted.append(converter.message_v2_to_v1(msg))
        except Exception as e:
            print(f"Failed to convert message {msg.message_id}: {e}")
            continue
    
    return converted


def migrate_session_data(
    v1_session: FFSessionDTO,
    v1_messages: List[FFMessageDTO]
) -> Dict[str, Any]:
    """
    Migrate complete session data from v1 to v2.
    
    Returns dict with 'session' and 'messages' keys.
    """
    converter = EntityConverter()
    
    # Convert session
    v2_session = converter.session_v1_to_v2(v1_session)
    
    # Convert messages
    v2_messages = []
    for msg in v1_messages:
        v2_msg = converter.message_v1_to_v2(msg)
        v2_msg.session_id = v2_session.id
        v2_messages.append(v2_msg)
    
    return {
        "session": v2_session,
        "messages": v2_messages
    }
```

## Testing Requirements

### Unit Tests to Create

1. **Test Message Models** (`tests/test_v2_entities.py`)
   - Test message creation with all content types
   - Test v1 to v2 conversion
   - Test validation

2. **Test Memory System** (`tests/test_memory_entities.py`)
   - Test memory creation and decay
   - Test relevance scoring
   - Test consolidation

3. **Test Multimodal Entities** (`tests/test_multimodal_entities.py`)
   - Test content metadata
   - Test processing status tracking

4. **Test Protocol Compliance** (`tests/test_v2_protocols.py`)
   - Ensure all protocols are properly defined
   - Test type hints

### Sample Test Implementation

```python
# tests/test_v2_entities.py
import pytest
from ff_class_configs.ff_chat_entities_v2 import (
    MessageV2DTO, ContentType, SenderType
)
from ff_utils.ff_entity_converters import EntityConverter

def test_text_message_creation():
    """Test creating a basic text message"""
    msg = MessageV2DTO(
        id="test_123",
        sender_type=SenderType.USER,
        sender_id="user_123",
        content_type=ContentType.TEXT,
        content_data={"text": "Hello, world!", "format": "plain"}
    )
    
    assert msg.id == "test_123"
    assert msg.sender_type == SenderType.USER
    assert msg.content_type == ContentType.TEXT
    assert msg.content_data["text"] == "Hello, world!"

def test_multimodal_message():
    """Test creating a message with image content"""
    msg = MessageV2DTO(
        id="test_456",
        sender_type=SenderType.USER,
        sender_id="user_123",
        content_type=ContentType.IMAGE,
        content_data={
            "storage_ref": "img_abc123",
            "caption": "My vacation photo",
            "mime_type": "image/jpeg"
        }
    )
    
    assert msg.content_type == ContentType.IMAGE
    assert "storage_ref" in msg.content_data

def test_legacy_conversion():
    """Test converting from v1 to v2 format"""
    from ff_class_configs.ff_chat_entities import FFMessageDTO
    
    v1_msg = FFMessageDTO(
        role="user",
        content="Hello from v1",
        message_id="legacy_123",
        timestamp="2024-01-01T10:00:00Z"
    )
    
    v2_msg = EntityConverter.message_v1_to_v2(v1_msg)
    
    assert v2_msg.id == "legacy_123"
    assert v2_msg.sender_type == SenderType.USER
    assert v2_msg.content_type == ContentType.TEXT
    assert v2_msg.content_data["text"] == "Hello from v1"
```

## Deliverables Checklist

- [ ] Enhanced message model (`ff_chat_entities_v2.py`)
- [ ] Memory system entities (`ff_memory_entities.py`)
- [ ] Multimodal entities (`ff_multimodal_entities.py`)
- [ ] Trace system entities (`ff_trace_entities.py`)
- [ ] Agent system entities (`ff_agent_entities.py`)
- [ ] Memory protocol (`ff_memory_protocol.py`)
- [ ] Multimodal protocol (`ff_multimodal_protocol.py`)
- [ ] Trace protocol (`ff_trace_protocol.py`)
- [ ] Agent protocol (`ff_agent_protocol.py`)
- [ ] Updated storage config
- [ ] Entity converters
- [ ] Unit tests
- [ ] Documentation

## Success Criteria

1. All new entities support serialization to/from dict
2. Backward compatibility maintained through converters
3. All enums properly defined and validated
4. Type hints complete and correct
5. No breaking changes to existing code
6. Unit tests pass with >90% coverage

## Next Phase Preview

Phase 2 will build on these models and protocols by implementing the actual managers that handle storage operations. The clean separation ensures that Phase 2 can proceed independently once these foundations are in place.