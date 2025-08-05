# Phase 1a: Enhanced Message and Session Models

## Overview
This sub-phase focuses on implementing the core message and session models that support polymorphic content and flexible session types. These models form the foundation for all chat interactions.

## Objectives
1. Create enhanced message model supporting multiple content types
2. Implement flexible session model for different interaction patterns
3. Ensure backward compatibility with existing v1 models
4. Provide conversion utilities between v1 and v2 formats

## Prerequisites
- Existing codebase with `ff_class_configs/ff_chat_entities.py`
- Understanding of current FFMessageDTO and FFSessionDTO structures

## Implementation Files

### 1. Enhanced Message Model (`ff_class_configs/ff_chat_entities_v2.py`)

Create a new file that supports polymorphic messages:

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

### 2. Message/Session Conversion Utilities (`ff_utils/ff_message_converters.py`)

Create conversion utilities specific to messages and sessions:

```python
"""
Message and session conversion utilities for v1 to v2 migration.

Provides bidirectional conversion between legacy and new formats.
"""

from typing import Dict, Any, List, Optional
from ff_class_configs.ff_chat_entities import (
    FFMessageDTO, FFSessionDTO, FFPanelDTO
)
from ff_class_configs.ff_chat_entities_v2 import (
    MessageV2DTO, SessionV2DTO, ContentType, SenderType,
    SessionType, ParticipantRole, SessionParticipant
)


class MessageConverter:
    """Handles conversion between v1 and v2 message formats"""
    
    @staticmethod
    def v1_to_v2(v1_msg: FFMessageDTO, session_id: Optional[str] = None) -> MessageV2DTO:
        """
        Convert v1 message to v2 format.
        
        Args:
            v1_msg: Legacy message
            session_id: Optional session ID to attach
            
        Returns:
            V2 message
        """
        # Use built-in conversion
        v2_msg = MessageV2DTO.from_legacy(v1_msg)
        
        # Add session ID if provided
        if session_id:
            v2_msg.session_id = session_id
        
        # Check for multimodal content in attachments
        if v1_msg.attachments:
            v2_msg = MessageConverter._process_attachments(v2_msg, v1_msg.attachments)
        
        return v2_msg
    
    @staticmethod
    def v2_to_v1(v2_msg: MessageV2DTO) -> FFMessageDTO:
        """
        Convert v2 message to v1 format (lossy).
        
        Args:
            v2_msg: V2 message
            
        Returns:
            Legacy message
        """
        return v2_msg.to_legacy()
    
    @staticmethod
    def _process_attachments(v2_msg: MessageV2DTO, attachments: List[str]) -> MessageV2DTO:
        """Process v1 attachments for potential multimodal content"""
        for attachment in attachments:
            # Guess content type from extension
            ext = attachment.split(".")[-1].lower() if "." in attachment else ""
            
            if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
                # Convert to image message if it's the primary content
                if v2_msg.content_type == ContentType.TEXT and not v2_msg.content_data.get("text"):
                    v2_msg.content_type = ContentType.IMAGE
                    v2_msg.content_data = {
                        "storage_ref": attachment,
                        "mime_type": f"image/{ext}"
                    }
                else:
                    # Add as attachment
                    v2_msg.attachments.append({
                        "type": "image",
                        "path": attachment,
                        "mime_type": f"image/{ext}"
                    })
            
            elif ext in ["mp3", "wav", "m4a", "ogg"]:
                if v2_msg.content_type == ContentType.TEXT and not v2_msg.content_data.get("text"):
                    v2_msg.content_type = ContentType.AUDIO
                    v2_msg.content_data = {
                        "storage_ref": attachment,
                        "mime_type": f"audio/{ext}"
                    }
                else:
                    v2_msg.attachments.append({
                        "type": "audio",
                        "path": attachment,
                        "mime_type": f"audio/{ext}"
                    })
            
            elif ext in ["mp4", "webm", "mov"]:
                if v2_msg.content_type == ContentType.TEXT and not v2_msg.content_data.get("text"):
                    v2_msg.content_type = ContentType.VIDEO
                    v2_msg.content_data = {
                        "storage_ref": attachment,
                        "mime_type": f"video/{ext}"
                    }
                else:
                    v2_msg.attachments.append({
                        "type": "video",
                        "path": attachment,
                        "mime_type": f"video/{ext}"
                    })
        
        return v2_msg
    
    @staticmethod
    def batch_convert(
        messages: List[FFMessageDTO],
        session_id: Optional[str] = None,
        direction: str = "v1_to_v2"
    ) -> List[MessageV2DTO]:
        """
        Convert a batch of messages.
        
        Args:
            messages: Messages to convert
            session_id: Optional session ID
            direction: "v1_to_v2" or "v2_to_v1"
            
        Returns:
            Converted messages
        """
        converted = []
        
        for msg in messages:
            try:
                if direction == "v1_to_v2":
                    converted.append(MessageConverter.v1_to_v2(msg, session_id))
                else:
                    converted.append(MessageConverter.v2_to_v1(msg))
            except Exception as e:
                print(f"Failed to convert message {getattr(msg, 'id', 'unknown')}: {e}")
                continue
        
        return converted


class SessionConverter:
    """Handles conversion between v1 and v2 session formats"""
    
    @staticmethod
    def v1_to_v2(v1_session: FFSessionDTO) -> SessionV2DTO:
        """Convert v1 session to v2 format"""
        return SessionV2DTO.from_legacy(v1_session)
    
    @staticmethod
    def v2_to_v1(v2_session: SessionV2DTO) -> FFSessionDTO:
        """
        Convert v2 session to v1 format (lossy).
        
        Only preserves primary user information.
        """
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
        """Convert v1 panel to v2 session format"""
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
    def migrate_complete_session(
        v1_session: FFSessionDTO,
        v1_messages: List[FFMessageDTO]
    ) -> Dict[str, Any]:
        """
        Migrate complete session with all messages.
        
        Returns:
            Dict with 'session' and 'messages' keys
        """
        # Convert session
        v2_session = SessionConverter.v1_to_v2(v1_session)
        
        # Convert messages
        v2_messages = MessageConverter.batch_convert(
            v1_messages,
            session_id=v2_session.id
        )
        
        return {
            "session": v2_session,
            "messages": v2_messages,
            "message_count": len(v2_messages)
        }
```

### 3. Message Model Tests (`tests/test_message_models_v2.py`)

Create comprehensive tests for the new message models:

```python
"""
Tests for v2 message and session models.
"""

import pytest
from datetime import datetime
from ff_class_configs.ff_chat_entities_v2 import (
    MessageV2DTO, SessionV2DTO, ContentType, SenderType,
    SessionType, ParticipantRole, SessionParticipant,
    TextContentData, ImageContentData, ToolCallData
)
from ff_class_configs.ff_chat_entities import FFMessageDTO, FFSessionDTO
from ff_utils.ff_message_converters import MessageConverter, SessionConverter


class TestMessageV2Models:
    """Test message v2 models"""
    
    def test_text_message_creation(self):
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
        assert msg.timestamp is not None
    
    def test_image_message_creation(self):
        """Test creating an image message"""
        msg = MessageV2DTO(
            id="img_msg_1",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.IMAGE,
            content_data={
                "storage_ref": "img_abc123",
                "caption": "My vacation photo",
                "mime_type": "image/jpeg",
                "width": 1920,
                "height": 1080
            }
        )
        
        assert msg.content_type == ContentType.IMAGE
        assert msg.content_data["storage_ref"] == "img_abc123"
        assert msg.content_data["width"] == 1920
    
    def test_tool_call_message(self):
        """Test creating a tool call message"""
        msg = MessageV2DTO(
            id="tool_msg_1",
            sender_type=SenderType.AGENT,
            sender_id="agent_123",
            content_type=ContentType.TOOL_CALL,
            content_data={
                "message": "Let me search for that information"
            },
            tool_calls=[{
                "tool_id": "web_search",
                "function": "search",
                "arguments": {"query": "weather today"},
                "call_id": "call_001"
            }]
        )
        
        assert msg.content_type == ContentType.TOOL_CALL
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["tool_id"] == "web_search"
    
    def test_message_serialization(self):
        """Test message to/from dict conversion"""
        original = MessageV2DTO(
            id="ser_test_1",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.TEXT,
            content_data={"text": "Test serialization"},
            metadata={"custom": "value"}
        )
        
        # Convert to dict
        msg_dict = original.to_dict()
        assert msg_dict["id"] == "ser_test_1"
        assert msg_dict["sender_type"] == "user"
        assert msg_dict["content_type"] == "text"
        
        # Convert back
        restored = MessageV2DTO.from_dict(msg_dict)
        assert restored.id == original.id
        assert restored.sender_type == original.sender_type
        assert restored.content_data == original.content_data
    
    def test_legacy_message_conversion(self):
        """Test converting from v1 to v2 format"""
        v1_msg = FFMessageDTO(
            role="user",
            content="Hello from v1",
            message_id="legacy_123",
            timestamp="2024-01-01T10:00:00Z",
            attachments=["file1.txt"],
            metadata={"source": "v1"}
        )
        
        # Convert to v2
        v2_msg = MessageV2DTO.from_legacy(v1_msg)
        
        assert v2_msg.id == "legacy_123"
        assert v2_msg.sender_type == SenderType.USER
        assert v2_msg.content_type == ContentType.TEXT
        assert v2_msg.content_data["text"] == "Hello from v1"
        assert len(v2_msg.attachments) == 1
        assert v2_msg.metadata["source"] == "v1"
        
        # Convert back to v1
        v1_restored = v2_msg.to_legacy()
        assert v1_restored.role == "user"
        assert v1_restored.content == "Hello from v1"
        assert v1_restored.message_id == "legacy_123"
    
    def test_multimodal_attachments(self):
        """Test message with multiple attachments"""
        msg = MessageV2DTO(
            id="multi_1",
            sender_type=SenderType.USER,
            sender_id="user_123",
            content_type=ContentType.TEXT,
            content_data={"text": "Check these files"},
            attachments=[
                {
                    "type": "image",
                    "path": "images/photo1.jpg",
                    "mime_type": "image/jpeg"
                },
                {
                    "type": "document",
                    "path": "docs/report.pdf",
                    "mime_type": "application/pdf"
                }
            ]
        )
        
        assert len(msg.attachments) == 2
        assert msg.attachments[0]["type"] == "image"
        assert msg.attachments[1]["type"] == "document"
    
    def test_memory_references(self):
        """Test message with memory references"""
        msg = MessageV2DTO(
            id="mem_ref_1",
            sender_type=SenderType.AGENT,
            sender_id="agent_123",
            content_type=ContentType.TEXT,
            content_data={"text": "Based on our previous discussion..."},
            memory_refs=[
                {
                    "memory_id": "mem_001",
                    "memory_type": "episodic",
                    "relevance_score": 0.95,
                    "preview": "User mentioned project deadline"
                },
                {
                    "memory_id": "mem_002",
                    "memory_type": "semantic",
                    "relevance_score": 0.85,
                    "preview": "Project management best practices"
                }
            ]
        )
        
        assert len(msg.memory_refs) == 2
        assert msg.memory_refs[0]["relevance_score"] == 0.95


class TestSessionV2Models:
    """Test session v2 models"""
    
    def test_chat_session_creation(self):
        """Test creating a basic chat session"""
        session = SessionV2DTO(
            id="session_123",
            type=SessionType.CHAT,
            title="Test Chat",
            participants=[{
                "id": "user_123",
                "type": "user",
                "role": "member",
                "name": "Alice"
            }],
            capabilities=["text", "image"]
        )
        
        assert session.id == "session_123"
        assert session.type == SessionType.CHAT
        assert len(session.participants) == 1
        assert session.active is True
    
    def test_panel_session_creation(self):
        """Test creating a panel session"""
        session = SessionV2DTO(
            id="panel_123",
            type=SessionType.PANEL,
            title="Expert Panel Discussion",
            participants=[
                {
                    "id": "moderator_1",
                    "type": "agent",
                    "role": "moderator",
                    "name": "Panel Moderator"
                },
                {
                    "id": "expert_1",
                    "type": "agent",
                    "role": "member",
                    "name": "Domain Expert 1"
                },
                {
                    "id": "expert_2",
                    "type": "agent",
                    "role": "member",
                    "name": "Domain Expert 2"
                }
            ],
            capabilities=["text", "multi_agent"],
            context={
                "topic": "AI Ethics",
                "format": "structured_discussion"
            }
        )
        
        assert session.type == SessionType.PANEL
        assert len(session.participants) == 3
        assert session.context["topic"] == "AI Ethics"
    
    def test_participant_management(self):
        """Test adding and removing participants"""
        session = SessionV2DTO(
            id="session_456",
            type=SessionType.CHAT,
            title="Dynamic Chat"
        )
        
        # Add participant
        participant = SessionParticipant(
            id="user_789",
            type=SenderType.USER,
            role=ParticipantRole.MEMBER,
            name="Bob",
            capabilities=["text", "voice"]
        )
        
        assert session.add_participant(participant) is True
        assert len(session.participants) == 1
        
        # Try adding same participant again
        assert session.add_participant(participant) is False
        assert len(session.participants) == 1
        
        # Remove participant
        assert session.remove_participant("user_789") is True
        assert len(session.participants) == 0
    
    def test_session_context_update(self):
        """Test updating session context"""
        session = SessionV2DTO(
            id="session_789",
            type=SessionType.WORKFLOW,
            title="Task Workflow",
            context={"stage": "initialization"}
        )
        
        # Update context
        session.update_context({
            "stage": "processing",
            "progress": 0.5,
            "current_task": "data_validation"
        })
        
        assert session.context["stage"] == "processing"
        assert session.context["progress"] == 0.5
        assert "current_task" in session.context
    
    def test_legacy_session_conversion(self):
        """Test converting v1 session to v2"""
        v1_session = FFSessionDTO(
            session_id="old_session_123",
            user_id="user_456",
            title="Legacy Chat",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T11:00:00Z",
            message_count=42,
            metadata={"client": "web"}
        )
        
        # Convert to v2
        v2_session = SessionV2DTO.from_legacy(v1_session)
        
        assert v2_session.id == "old_session_123"
        assert v2_session.type == SessionType.CHAT
        assert v2_session.title == "Legacy Chat"
        assert v2_session.message_count == 42
        assert len(v2_session.participants) == 1
        assert v2_session.participants[0]["id"] == "user_456"


class TestMessageConverters:
    """Test message conversion utilities"""
    
    def test_v1_to_v2_with_attachments(self):
        """Test converting v1 message with attachments"""
        v1_msg = FFMessageDTO(
            role="user",
            content="Check this image",
            message_id="att_test_1",
            timestamp="2024-01-01T10:00:00Z",
            attachments=["images/photo.jpg", "docs/file.pdf"]
        )
        
        v2_msg = MessageConverter.v1_to_v2(v1_msg, session_id="session_123")
        
        assert v2_msg.session_id == "session_123"
        assert len(v2_msg.attachments) == 2
        
        # Check attachment processing
        img_att = next(a for a in v2_msg.attachments if a["type"] == "image")
        assert img_att["mime_type"] == "image/jpg"
    
    def test_batch_conversion(self):
        """Test batch message conversion"""
        v1_messages = [
            FFMessageDTO(
                role="user",
                content=f"Message {i}",
                message_id=f"msg_{i}",
                timestamp=datetime.now().isoformat()
            )
            for i in range(5)
        ]
        
        v2_messages = MessageConverter.batch_convert(
            v1_messages,
            session_id="batch_session",
            direction="v1_to_v2"
        )
        
        assert len(v2_messages) == 5
        assert all(msg.session_id == "batch_session" for msg in v2_messages)
        assert v2_messages[0].content_data["text"] == "Message 0"
    
    def test_session_migration(self):
        """Test complete session migration"""
        v1_session = FFSessionDTO(
            session_id="migrate_test",
            user_id="user_123",
            title="Migration Test",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            message_count=2
        )
        
        v1_messages = [
            FFMessageDTO(
                role="user",
                content="Hello",
                message_id="msg_1",
                timestamp=datetime.now().isoformat()
            ),
            FFMessageDTO(
                role="assistant",
                content="Hi there!",
                message_id="msg_2",
                timestamp=datetime.now().isoformat()
            )
        ]
        
        result = SessionConverter.migrate_complete_session(v1_session, v1_messages)
        
        assert result["session"].id == "migrate_test"
        assert result["session"].type == SessionType.CHAT
        assert len(result["messages"]) == 2
        assert result["messages"][0].sender_type == SenderType.USER
        assert result["messages"][1].sender_type == SenderType.AGENT
```

## Implementation Steps

1. **Create the enhanced message model file**
   ```bash
   touch ff_class_configs/ff_chat_entities_v2.py
   # Copy the MessageV2DTO implementation
   ```

2. **Create the conversion utilities**
   ```bash
   touch ff_utils/ff_message_converters.py
   # Copy the converter implementations
   ```

3. **Create the test file**
   ```bash
   touch tests/test_message_models_v2.py
   # Copy the test implementations
   ```

4. **Run the tests**
   ```bash
   pytest tests/test_message_models_v2.py -v
   ```

## Validation Checklist

- [ ] MessageV2DTO supports all content types (text, image, audio, video, etc.)
- [ ] SessionV2DTO supports different session types (chat, panel, workflow, etc.)
- [ ] Backward compatibility maintained through from_legacy/to_legacy methods
- [ ] All models serialize/deserialize correctly
- [ ] Enum types properly handled in conversions
- [ ] Tool integration fields working
- [ ] Memory reference fields working
- [ ] Participant management in sessions working
- [ ] All tests passing

## Next Steps

Once Phase 1a is complete and tested:
1. Move to Phase 1b (Memory System Models)
2. No dependencies on Phase 1a implementation details
3. Can be developed in parallel by different developers

## Notes for Developers

1. **Enum Handling**: Always check if enum values are strings before creating enum instances
2. **Timestamp Format**: Use ISO format for all timestamps
3. **ID Generation**: Use UUID hex format for new IDs
4. **Metadata Fields**: Always initialize as empty dict, never None
5. **Legacy Compatibility**: Ensure all v2 models can convert to/from v1

This completes Phase 1a implementation specification.