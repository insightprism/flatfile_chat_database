"""
Chat entity configuration models for the flatfile chat database system.

These entity configuration models provide type safety and validation for all 
data structures used in the storage system. As part of the configuration 
architecture, they define the structure and behavior of chat entities.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class MessageRole(str, Enum):
    """Valid message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    

class PanelType(str, Enum):
    """Types of panel sessions"""
    MULTI_PERSONA = "multi_persona"
    FOCUS_GROUP = "focus_group"
    EXPERT_PANEL = "expert_panel"
    BRAINSTORM = "brainstorm"


class SearchType(str, Enum):
    """Types of search operations"""
    TEXT = "text"
    VECTOR = "vector"
    HYBRID = "hybrid"


def generate_message_id(config: Optional['StorageConfig'] = None) -> str:
    """Generate unique message ID with configurable length"""
    from ff_config_legacy_adapter import StorageConfig
    if config is None:
        # Use default length if no config provided (for backward compatibility)
        length = 12
    else:
        length = config.message_id_length
    return f"msg_{uuid.uuid4().hex[:length]}"


def generate_insight_id(config: Optional['StorageConfig'] = None) -> str:
    """Generate unique insight ID with configurable length"""
    from ff_config_legacy_adapter import StorageConfig
    if config is None:
        # Use default length if no config provided (for backward compatibility)
        length = 12
    else:
        length = config.insight_id_length
    return f"insight_{uuid.uuid4().hex[:length]}"


def current_timestamp() -> str:
    """Get current ISO format timestamp"""
    return datetime.now().isoformat()


@dataclass
class FFMessage:
    """Individual chat message"""
    role: str  # Can be MessageRole value or persona_id for panels
    content: str
    message_id: str = field(default_factory=generate_message_id)
    timestamp: str = field(default_factory=current_timestamp)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message after initialization"""
        if not self.content:
            raise ValueError("Message content cannot be empty")
        if not self.role:
            raise ValueError("Message role cannot be empty")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFMessage':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFSession:
    """Chat session metadata"""
    session_id: str
    user_id: str
    title: str = "New Chat"
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate session after initialization"""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        if not self.user_id:
            raise ValueError("User ID cannot be empty")
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp"""
        self.updated_at = current_timestamp()
    
    @property
    def session_name(self) -> str:
        """Compatibility property - maps to title"""
        return self.title
    
    @session_name.setter
    def session_name(self, value: str) -> None:
        """Compatibility property setter - maps to title"""
        self.title = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFSession':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFPanel:
    """Multi-persona panel session"""
    id: str
    type: str  # PanelType value
    personas: List[str]
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    message_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_personas: int = 10  # Can be overridden from config
    
    def __post_init__(self):
        """Validate panel after initialization"""
        if not self.id:
            raise ValueError("Panel ID cannot be empty")
        if not self.personas:
            raise ValueError("Panel must have at least one persona")
        if len(self.personas) > self.max_personas:
            raise ValueError(f"Panel cannot have more than {self.max_personas} personas")
    
    def add_persona(self, persona_id: str) -> None:
        """Add a persona to the panel"""
        if persona_id not in self.personas:
            if len(self.personas) >= self.max_personas:
                raise ValueError(f"Panel cannot have more than {self.max_personas} personas")
            self.personas.append(persona_id)
            self.updated_at = current_timestamp()
            
    def remove_persona(self, persona_id: str) -> None:
        """Remove a persona from the panel"""
        if persona_id in self.personas:
            self.personas.remove(persona_id)
            self.updated_at = current_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Don't store config-related fields
        data.pop('max_personas', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFPanel':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFSituationalContext:
    """Conversation context snapshot"""
    summary: str
    key_points: List[str]
    entities: Dict[str, List[str]]
    timestamp: str = field(default_factory=current_timestamp)
    confidence: float = 0.0
    message_range: Optional[Dict[str, Any]] = None  # start/end message IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_key_points: int = 10  # Can be overridden from config
    
    def __post_init__(self):
        """Validate context after initialization"""
        if not self.summary:
            raise ValueError("Context summary cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if len(self.key_points) > self.max_key_points:
            raise ValueError(f"Too many key points (max {self.max_key_points})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Don't store config-related fields
        data.pop('max_key_points', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFSituationalContext':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFDocument:
    """Document metadata"""
    filename: str
    original_name: str
    path: str
    mime_type: str
    size: int  # bytes
    uploaded_at: str = field(default_factory=current_timestamp)
    uploaded_by: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate document after initialization"""
        if not self.filename:
            raise ValueError("Document filename cannot be empty")
        if not self.path:
            raise ValueError("Document path cannot be empty")
        if self.size < 0:
            raise ValueError("Document size cannot be negative")
    
    def add_analysis(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Add analysis results"""
        self.analysis[analysis_type] = {
            "results": results,
            "timestamp": current_timestamp()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFDocument':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFUserProfile:
    """User profile data"""
    user_id: str
    username: str = ""
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate user profile after initialization"""
        if not self.user_id:
            raise ValueError("User ID cannot be empty")
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference"""
        self.preferences[key] = value
        self.updated_at = current_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFUserProfile':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFPanelMessage(FFMessage):
    """Message specific to panel sessions"""
    persona_id: str = ""
    response_to: Optional[str] = None  # Message ID being responded to
    consensus_level: Optional[float] = None  # Agreement level (0-1)
    
    def __post_init__(self):
        """Validate panel message"""
        super().__post_init__()
        if not self.persona_id:
            self.persona_id = self.role  # Use role as persona_id if not set
        if self.consensus_level is not None:
            if self.consensus_level < 0 or self.consensus_level > 1:
                raise ValueError("Consensus level must be between 0 and 1")


@dataclass
class FFPersona:
    """AI Persona definition"""
    id: str
    name: str
    description: str = ""
    traits: List[str] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    is_global: bool = True  # True for global, False for user-specific
    owner_id: Optional[str] = None  # User ID if user-specific
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate persona after initialization"""
        if not self.id:
            raise ValueError("Persona ID cannot be empty")
        if not self.name:
            raise ValueError("Persona name cannot be empty")
        if not self.is_global and not self.owner_id:
            raise ValueError("User-specific personas must have an owner_id")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFPersona':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFPanelInsight:
    """Analysis or conclusion from a panel session"""
    id: str = field(default_factory=lambda: generate_insight_id())
    panel_id: str = ""
    type: str = "conclusion"  # conclusion, analysis, summary, recommendation
    content: str = ""
    supporting_messages: List[str] = field(default_factory=list)  # Message IDs
    consensus_level: float = 0.0
    created_at: str = field(default_factory=current_timestamp)
    created_by: str = "system"  # system or persona_id
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate panel insight"""
        if not self.panel_id:
            raise ValueError("Panel ID cannot be empty")
        if not self.content:
            raise ValueError("Insight content cannot be empty")
        if self.consensus_level < 0 or self.consensus_level > 1:
            raise ValueError("Consensus level must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFPanelInsight':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFVectorSearchResult:
    """Result from vector similarity search"""
    chunk_id: str
    chunk_text: str
    similarity_score: float
    document_id: str
    session_id: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate search result"""
        if self.similarity_score < 0 or self.similarity_score > 1:
            raise ValueError("Similarity score must be between 0 and 1")


@dataclass
class FFVectorMetadata:
    """Metadata for stored vectors"""
    chunk_id: str
    vector_index: int
    document_id: str
    session_id: str
    chunk_text: str
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFVectorMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FFProcessingResult:
    """Result from document processing pipeline"""
    success: bool
    document_id: str
    chunk_count: int
    vector_count: int
    processing_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)