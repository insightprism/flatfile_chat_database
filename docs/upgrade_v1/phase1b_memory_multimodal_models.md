# Phase 1b: Memory and Multimodal Models

## Overview
This sub-phase implements the memory system and multimodal content models. These models enable hierarchical memory storage with decay mechanisms and comprehensive multimodal content handling.

## Objectives
1. Create memory system entities with decay and relevance tracking
2. Implement multimodal content models for images, audio, video, and documents
3. Define processing and metadata structures
4. Ensure proper serialization and type safety

## Prerequisites
- Basic understanding of memory systems in AI
- Familiarity with multimodal content types
- No dependencies on Phase 1a (can be developed in parallel)

## Implementation Files

### 1. Memory System Entities (`ff_class_configs/ff_memory_entities.py`)

Create the memory system data models:

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryStats':
        return cls(**data)


@dataclass
class MemoryConsolidationResult:
    """Result of memory consolidation operation"""
    memories_consolidated: int = 0
    memories_removed: int = 0
    new_consolidated_memories: List[str] = field(default_factory=list)
    consolidation_groups: List[List[str]] = field(default_factory=list)
    time_taken_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 2. Multimodal Content Entities (`ff_class_configs/ff_multimodal_entities.py`)

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
                k.value if isinstance(k, ProcessingType) else k: v 
                for k, v in self.processing_results.items()
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultimodalContentDTO':
        data = data.copy()
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ContentStatus(data['status'])
        if 'processing_results' in data:
            # Convert string keys back to enums if needed
            proc_results = {}
            for k, v in data['processing_results'].items():
                try:
                    proc_results[ProcessingType(k)] = v
                except ValueError:
                    proc_results[k] = v  # Keep as string if not valid enum
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentProcessingRequest':
        data = data.copy()
        if 'processing_types' in data:
            data['processing_types'] = [
                ProcessingType(pt) if isinstance(pt, str) else pt
                for pt in data['processing_types']
            ]
        return cls(**data)


@dataclass
class ContentUploadRequest:
    """Request to upload multimodal content"""
    filename: str
    content_type: str  # image, audio, video, document
    mime_type: str
    size_bytes: int
    session_id: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    process_immediately: bool = True
    processing_types: List[ProcessingType] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate upload request, return list of errors"""
        errors = []
        
        if not self.filename:
            errors.append("Filename is required")
        
        if self.content_type not in ["image", "audio", "video", "document"]:
            errors.append(f"Invalid content type: {self.content_type}")
        
        if self.size_bytes <= 0:
            errors.append("Size must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.processing_types:
            data['processing_types'] = [pt.value for pt in self.processing_types]
        return data


@dataclass
class ContentSearchQuery:
    """Query for searching multimodal content"""
    content_type: Optional[str] = None  # Filter by type
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    mime_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None  # start/end dates
    min_size_bytes: Optional[int] = None
    max_size_bytes: Optional[int] = None
    has_processing: Optional[ProcessingType] = None
    tags: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.has_processing:
            data['has_processing'] = self.has_processing.value
        return data
```

### 3. Memory and Multimodal Tests (`tests/test_memory_multimodal_v2.py`)

Create comprehensive tests for memory and multimodal models:

```python
"""
Tests for memory and multimodal content models.
"""

import pytest
from datetime import datetime, timedelta
from ff_class_configs.ff_memory_entities import (
    MemoryEntryDTO, MemoryType, MemoryScope,
    MemorySearchQuery, MemoryStats
)
from ff_class_configs.ff_multimodal_entities import (
    MultimodalContentDTO, ImageMetadata, AudioMetadata,
    VideoMetadata, DocumentMetadata, ContentStatus,
    ProcessingType, ContentProcessingRequest,
    ContentUploadRequest
)


class TestMemoryModels:
    """Test memory system models"""
    
    def test_memory_creation(self):
        """Test creating a basic memory entry"""
        memory = MemoryEntryDTO(
            id="mem_123",
            type=MemoryType.EPISODIC,
            content="User discussed project timeline for Q1 2024",
            source_type="message",
            source_id="msg_456",
            importance=0.8
        )
        
        assert memory.id == "mem_123"
        assert memory.type == MemoryType.EPISODIC
        assert memory.importance == 0.8
        assert memory.decay_factor == 1.0  # No decay initially
        assert memory.scope == MemoryScope.USER  # Default scope
    
    def test_memory_decay_mechanism(self):
        """Test memory decay calculations"""
        memory = MemoryEntryDTO(
            id="mem_decay",
            type=MemoryType.SEMANTIC,
            content="Python is a programming language",
            importance=0.6,
            decay_factor=1.0
        )
        
        # Apply decay
        memory.decay(time_factor=1.0)
        
        # Important memories decay slower
        # With importance 0.6, modifier = 1.0 - (0.6 * 0.5) = 0.7
        # Decay = 0.01 * 0.7 * 1.0 = 0.007
        expected_decay = 1.0 - 0.007
        assert abs(memory.decay_factor - expected_decay) < 0.001
        
        # Apply more decay
        memory.decay(time_factor=10.0)
        assert memory.decay_factor < expected_decay
        
        # Test minimum decay
        memory.decay_factor = 0.15
        memory.decay(time_factor=10.0)
        assert memory.decay_factor >= 0.1  # Minimum threshold
    
    def test_memory_access_reinforcement(self):
        """Test memory reinforcement through access"""
        memory = MemoryEntryDTO(
            id="mem_access",
            type=MemoryType.EPISODIC,
            content="Important meeting notes",
            decay_factor=0.5,
            access_count=0
        )
        
        # Access memory
        memory.access(boost=0.1)
        
        assert memory.access_count == 1
        assert memory.last_accessed is not None
        assert memory.decay_factor == 0.6  # Boosted by 0.1
        
        # Multiple accesses
        for _ in range(5):
            memory.access(boost=0.1)
        
        assert memory.access_count == 6
        assert memory.decay_factor <= 1.0  # Capped at 1.0
    
    def test_memory_reinforcement(self):
        """Test explicit memory reinforcement"""
        memory = MemoryEntryDTO(
            id="mem_reinforce",
            type=MemoryType.SEMANTIC,
            content="Important fact",
            importance=0.8,
            confidence=0.7,
            decay_factor=0.6
        )
        
        # Reinforce memory
        memory.reinforce(strength=0.3)
        
        assert memory.last_reinforced is not None
        # Boost = 0.3 * 0.8 = 0.24
        assert abs(memory.decay_factor - 0.84) < 0.01
        # Confidence boost = 0.3 * 0.1 = 0.03
        assert abs(memory.confidence - 0.73) < 0.01
    
    def test_memory_relevance_score(self):
        """Test relevance score calculation"""
        memory = MemoryEntryDTO(
            id="mem_relevance",
            type=MemoryType.PROCEDURAL,
            content="How to use the API",
            importance=0.9,
            confidence=0.95,
            decay_factor=0.8
        )
        
        relevance = memory.get_relevance_score()
        expected = 0.9 * 0.95 * 0.8
        assert abs(relevance - expected) < 0.001
    
    def test_memory_merge(self):
        """Test merging related memories"""
        memory1 = MemoryEntryDTO(
            id="mem_1",
            type=MemoryType.SEMANTIC,
            content="Python is interpreted",
            importance=0.7,
            confidence=0.9,
            tags=["python", "programming"],
            entities={"languages": ["Python"]}
        )
        
        memory2 = MemoryEntryDTO(
            id="mem_2",
            type=MemoryType.SEMANTIC,
            content="Python is dynamically typed",
            importance=0.8,
            confidence=0.85,
            tags=["python", "types"],
            entities={"languages": ["Python"], "concepts": ["typing"]}
        )
        
        # Merge memory2 into memory1
        memory1.merge_with(memory2)
        
        assert "Python is dynamically typed" in memory1.content
        assert memory1.importance == 0.8  # Takes maximum
        assert abs(memory1.confidence - 0.875) < 0.001  # Average
        assert "types" in memory1.tags
        assert "concepts" in memory1.entities
        assert "typing" in memory1.entities["concepts"]
    
    def test_memory_serialization(self):
        """Test memory to/from dict conversion"""
        original = MemoryEntryDTO(
            id="mem_serial",
            type=MemoryType.EPISODIC,
            scope=MemoryScope.AGENT,
            scope_id="agent_123",
            content="Test serialization",
            importance=0.75,
            tags=["test", "serialization"],
            entities={"topics": ["testing"]}
        )
        
        # Convert to dict
        mem_dict = original.to_dict()
        assert mem_dict["type"] == "episodic"
        assert mem_dict["scope"] == "agent"
        
        # Convert back
        restored = MemoryEntryDTO.from_dict(mem_dict)
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.scope == original.scope
        assert restored.tags == original.tags
    
    def test_memory_search_query(self):
        """Test memory search query construction"""
        query = MemorySearchQuery(
            query="project timeline",
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            scope=MemoryScope.USER,
            scope_id="user_123",
            min_relevance=0.7,
            max_results=20,
            tags=["project", "planning"]
        )
        
        query_dict = query.to_dict()
        assert query_dict["query"] == "project timeline"
        assert "episodic" in query_dict["memory_types"]
        assert query_dict["scope"] == "user"


class TestMultimodalModels:
    """Test multimodal content models"""
    
    def test_multimodal_content_creation(self):
        """Test creating basic multimodal content"""
        content = MultimodalContentDTO(
            id="content_123",
            content_type="image",
            original_filename="photo.jpg",
            storage_path="multimodal/images/",
            storage_ref="img_abc123",
            mime_type="image/jpeg",
            size_bytes=1024000,
            hash="sha256_hash_here",
            session_id="session_123",
            user_id="user_123"
        )
        
        assert content.id == "content_123"
        assert content.content_type == "image"
        assert content.status == ContentStatus.PENDING
        assert content.uploaded_at is not None
    
    def test_content_processing_tracking(self):
        """Test tracking content processing"""
        content = MultimodalContentDTO(
            id="content_proc",
            content_type="image",
            original_filename="document.jpg",
            storage_path="multimodal/images/",
            storage_ref="img_proc_123",
            mime_type="image/jpeg",
            size_bytes=2048000,
            hash="hash_123"
        )
        
        # Add OCR processing result
        ocr_result = {
            "text": "Extracted text from image",
            "confidence": 0.95,
            "language": "en"
        }
        content.add_processing_result(ProcessingType.OCR, ocr_result)
        
        assert content.status == ContentStatus.COMPLETED
        assert content.processed_at is not None
        assert ProcessingType.OCR in content.processing_results
        assert content.processing_results[ProcessingType.OCR]["result"]["text"] == "Extracted text from image"
    
    def test_content_failure_handling(self):
        """Test handling processing failures"""
        content = MultimodalContentDTO(
            id="content_fail",
            content_type="audio",
            original_filename="audio.mp3",
            storage_path="multimodal/audio/",
            storage_ref="audio_fail_123",
            mime_type="audio/mpeg",
            size_bytes=5000000,
            hash="hash_fail"
        )
        
        # Mark as failed
        content.mark_failed("Transcription service unavailable")
        
        assert content.status == ContentStatus.FAILED
        assert content.processed_at is not None
        assert "error" in content.metadata
        assert "Transcription service unavailable" in content.metadata["error"]
    
    def test_content_access_tracking(self):
        """Test content access tracking"""
        content = MultimodalContentDTO(
            id="content_access",
            content_type="video",
            original_filename="video.mp4",
            storage_path="multimodal/video/",
            storage_ref="video_123",
            mime_type="video/mp4",
            size_bytes=50000000,
            hash="hash_video"
        )
        
        # Record multiple accesses
        for _ in range(3):
            content.record_access()
        
        assert content.access_count == 3
        assert content.last_accessed is not None
    
    def test_image_metadata(self):
        """Test image-specific metadata"""
        image = ImageMetadata(
            id="img_meta_123",
            content_type="image",
            original_filename="photo.jpg",
            storage_path="multimodal/images/",
            storage_ref="img_meta_ref",
            mime_type="image/jpeg",
            size_bytes=2048000,
            hash="img_hash",
            width=1920,
            height=1080,
            format="JPEG",
            color_mode="RGB",
            detected_objects=[
                {"object": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"object": "car", "confidence": 0.87, "bbox": [400, 200, 600, 400]}
            ],
            scene_description="Outdoor street scene with people and vehicles"
        )
        
        assert image.width == 1920
        assert image.height == 1080
        assert len(image.detected_objects) == 2
        assert image.scene_description is not None
    
    def test_audio_metadata(self):
        """Test audio-specific metadata"""
        audio = AudioMetadata(
            id="audio_meta_123",
            content_type="audio",
            original_filename="recording.mp3",
            storage_path="multimodal/audio/",
            storage_ref="audio_ref_123",
            mime_type="audio/mpeg",
            size_bytes=5000000,
            hash="audio_hash",
            duration_seconds=180.5,
            sample_rate=44100,
            channels=2,
            bitrate=320000,
            codec="mp3",
            transcript="This is the transcribed text",
            transcript_language="en",
            transcript_confidence=0.92,
            detected_speakers=[
                {"speaker_id": "speaker_1", "segments": [[0, 45.5], [120.0, 180.5]]},
                {"speaker_id": "speaker_2", "segments": [[45.5, 120.0]]}
            ]
        )
        
        assert audio.duration_seconds == 180.5
        assert audio.sample_rate == 44100
        assert audio.transcript is not None
        assert len(audio.detected_speakers) == 2
    
    def test_video_metadata(self):
        """Test video-specific metadata"""
        video = VideoMetadata(
            id="video_meta_123",
            content_type="video",
            original_filename="presentation.mp4",
            storage_path="multimodal/video/",
            storage_ref="video_ref_123",
            mime_type="video/mp4",
            size_bytes=100000000,
            hash="video_hash",
            duration_seconds=600.0,
            width=1280,
            height=720,
            fps=30.0,
            video_codec="h264",
            audio_codec="aac",
            scene_changes=[15.0, 45.5, 120.0, 300.0, 450.0],
            key_frames=[
                {"timestamp": 0.0, "frame_ref": "frame_0"},
                {"timestamp": 120.0, "frame_ref": "frame_120"}
            ]
        )
        
        assert video.duration_seconds == 600.0
        assert video.fps == 30.0
        assert len(video.scene_changes) == 5
        assert len(video.key_frames) == 2
    
    def test_document_metadata(self):
        """Test document-specific metadata"""
        document = DocumentMetadata(
            id="doc_meta_123",
            content_type="document",
            original_filename="report.pdf",
            storage_path="multimodal/documents/",
            storage_ref="doc_ref_123",
            mime_type="application/pdf",
            size_bytes=2500000,
            hash="doc_hash",
            page_count=25,
            word_count=5000,
            language="en",
            author="John Doe",
            title="Q4 2023 Report",
            extracted_text="Beginning of extracted text...",
            table_of_contents=[
                {"title": "Introduction", "page": 1},
                {"title": "Analysis", "page": 5},
                {"title": "Conclusion", "page": 20}
            ],
            tables=[
                {"page": 7, "rows": 10, "cols": 5, "title": "Financial Summary"}
            ]
        )
        
        assert document.page_count == 25
        assert document.word_count == 5000
        assert document.author == "John Doe"
        assert len(document.table_of_contents) == 3
        assert len(document.tables) == 1
    
    def test_content_processing_request(self):
        """Test content processing request"""
        request = ContentProcessingRequest(
            content_id="content_123",
            processing_types=[ProcessingType.OCR, ProcessingType.ANALYSIS],
            priority=8,
            options={
                "language": "en",
                "enhance": True
            }
        )
        
        request_dict = request.to_dict()
        assert request_dict["content_id"] == "content_123"
        assert "ocr" in request_dict["processing_types"]
        assert request_dict["priority"] == 8
        
        # Test reconstruction
        restored = ContentProcessingRequest.from_dict(request_dict)
        assert restored.content_id == request.content_id
        assert ProcessingType.OCR in restored.processing_types
    
    def test_content_upload_validation(self):
        """Test content upload request validation"""
        # Valid request
        valid_request = ContentUploadRequest(
            filename="test.jpg",
            content_type="image",
            mime_type="image/jpeg",
            size_bytes=1000000,
            session_id="session_123",
            user_id="user_123"
        )
        
        errors = valid_request.validate()
        assert len(errors) == 0
        
        # Invalid request
        invalid_request = ContentUploadRequest(
            filename="",
            content_type="invalid",
            mime_type="image/jpeg",
            size_bytes=-1,
            session_id="session_123",
            user_id="user_123"
        )
        
        errors = invalid_request.validate()
        assert len(errors) == 3
        assert any("Filename" in e for e in errors)
        assert any("content type" in e for e in errors)
        assert any("positive" in e for e in errors)


class TestMemoryMultimodalIntegration:
    """Test integration between memory and multimodal systems"""
    
    def test_memory_from_multimodal_content(self):
        """Test creating memory from multimodal content"""
        # Process an image with OCR
        image = ImageMetadata(
            id="img_ocr_123",
            content_type="image",
            original_filename="whiteboard.jpg",
            storage_path="multimodal/images/",
            storage_ref="img_ocr_ref",
            mime_type="image/jpeg",
            size_bytes=1500000,
            hash="ocr_hash"
        )
        
        # Add OCR result
        ocr_result = {
            "text": "Project deadline: March 15, 2024\nBudget: $50,000",
            "confidence": 0.93
        }
        image.add_processing_result(ProcessingType.OCR, ocr_result)
        
        # Create memory from extracted text
        memory = MemoryEntryDTO(
            id="mem_from_img",
            type=MemoryType.SEMANTIC,
            content=ocr_result["text"],
            source_type="multimodal",
            source_id=image.id,
            source_context={
                "content_type": "image",
                "filename": image.original_filename,
                "extraction_method": "OCR",
                "confidence": ocr_result["confidence"]
            },
            importance=0.8,
            tags=["project", "deadline", "budget"],
            entities={
                "dates": ["March 15, 2024"],
                "monetary": ["$50,000"]
            }
        )
        
        assert memory.source_type == "multimodal"
        assert memory.source_id == image.id
        assert "OCR" in memory.source_context["extraction_method"]
        assert "deadline" in memory.tags
```

## Implementation Steps

1. **Create the memory entities file**
   ```bash
   touch ff_class_configs/ff_memory_entities.py
   # Copy the MemoryEntryDTO implementation
   ```

2. **Create the multimodal entities file**
   ```bash
   touch ff_class_configs/ff_multimodal_entities.py
   # Copy the MultimodalContentDTO implementation
   ```

3. **Create the test file**
   ```bash
   touch tests/test_memory_multimodal_v2.py
   # Copy the test implementations
   ```

4. **Run the tests**
   ```bash
   pytest tests/test_memory_multimodal_v2.py -v
   ```

## Validation Checklist

- [ ] Memory entities support all memory types (episodic, semantic, procedural, working)
- [ ] Memory decay mechanism working correctly
- [ ] Memory reinforcement through access and explicit reinforcement
- [ ] Memory merge/consolidation functionality
- [ ] Multimodal content base model working
- [ ] Specialized metadata for images, audio, video, documents
- [ ] Processing status tracking
- [ ] Content access tracking
- [ ] All models serialize/deserialize correctly
- [ ] Enum types properly handled
- [ ] All tests passing

## Next Steps

Once Phase 1b is complete:
1. Move to Phase 1c (Trace and Agent Models)
2. These models can be used immediately by Phase 2 managers
3. No dependencies on other Phase 1 components

## Notes for Developers

1. **Memory Decay**: The decay mechanism uses importance to slow decay for critical memories
2. **Relevance Score**: Combines importance, confidence, and decay factor
3. **Processing Results**: Store results with timestamps for audit trails
4. **Content Deduplication**: Use hash field to detect duplicate uploads
5. **Access Patterns**: Track both access count and last accessed time

This completes Phase 1b implementation specification.