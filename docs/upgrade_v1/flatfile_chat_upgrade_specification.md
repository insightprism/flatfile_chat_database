# Flatfile Chat Database v2 - Upgrade Specification

## Executive Summary

This specification outlines the comprehensive upgrade of the Flatfile Chat Database to support 25+ advanced chat application use cases. The upgrade transforms the current storage-focused system into a full-featured, modular chat platform capable of handling multimodal content, multi-agent orchestration, tool integrations, and intelligent routing.

### Key Objectives
- Support all identified use cases from simple 1:1 chat to complex multi-agent systems
- Maintain clean, modular architecture with clear separation of concerns
- Enable extensibility through capability-based design
- Preserve valuable existing code while redesigning for enhanced functionality
- Provide clear migration path for existing data

### Major New Capabilities
1. **Multimodal Support** - Text, images, audio, video, documents
2. **Tool Framework** - External integrations and function calling
3. **Memory System** - Long-term, cross-session memory
4. **Multi-Agent Orchestration** - Complex agent coordination
5. **Topic Routing** - Intelligent context extraction and delegation
6. **Real-time Features** - Streaming, presence, live updates
7. **Trace Logging** - Comprehensive debugging and auditing

## Architecture Overview

### Design Principles
1. **Single Responsibility** - Each module has one clear purpose
2. **Protocol-Based** - All components implement defined protocols
3. **Event-Driven** - Loose coupling through event bus
4. **Capability-Based** - Features as pluggable capabilities
5. **Context-First** - Context drives all behavior

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  REST API   │  │  WebSocket   │  │   GraphQL    │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Router    │  │  Coordinator │  │   Workflow   │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Capability Layer                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │ Text │ │Multi │ │ RAG  │ │Tools │ │Memory│ │Trace │   │
│  │      │ │modal │ │      │ │      │ │      │ │      │   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Core Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Models    │  │   Managers   │  │   Events     │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Flatfile   │  │   Database   │  │  Object Store│       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components Specification

### 1. Enhanced Message System

#### Current State
```python
@dataclass
class FFMessageDTO:
    role: str
    content: str
    message_id: str
    timestamp: str
    attachments: List[str]
    metadata: Dict[str, Any]
```

#### Upgraded Design
```python
from enum import Enum
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM = "system"
    MEMORY_REFERENCE = "memory_reference"

class SenderType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class MessageContent(ABC):
    """Base class for all message content types"""
    type: MessageType
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

@dataclass
class TextContent(MessageContent):
    type: MessageType = MessageType.TEXT
    text: str = ""
    format: str = "plain"  # plain, markdown, html
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text,
            "format": self.format
        }

@dataclass
class ImageContent(MessageContent):
    type: MessageType = MessageType.IMAGE
    url: Optional[str] = None
    data: Optional[bytes] = None
    mime_type: str = "image/jpeg"
    alt_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "url": self.url,
            "mime_type": self.mime_type,
            "alt_text": self.alt_text,
            "has_data": self.data is not None
        }

@dataclass
class ToolCall:
    tool_id: str
    function: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")

@dataclass
class ToolResponse:
    call_id: str
    tool_id: str
    result: Any
    error: Optional[str] = None

@dataclass
class MemoryReference:
    memory_id: str
    memory_type: str  # episodic, semantic, procedural
    relevance_score: float
    context: Optional[str] = None

@dataclass
class MessageSender:
    type: SenderType
    id: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Message:
    id: str
    sender: MessageSender
    content: Union[MessageContent, List[MessageContent]]
    timestamp: str = field(default_factory=current_timestamp)
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_responses: List[ToolResponse] = field(default_factory=list)
    memory_refs: List[MemoryReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        content_data = []
        if isinstance(self.content, list):
            content_data = [c.to_dict() for c in self.content]
        else:
            content_data = [self.content.to_dict()]
            
        return {
            "id": self.id,
            "sender": {
                "type": self.sender.type.value,
                "id": self.sender.id,
                "name": self.sender.name,
                "metadata": self.sender.metadata
            },
            "content": content_data,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "tool_responses": [asdict(tr) for tr in self.tool_responses],
            "memory_refs": [asdict(mr) for mr in self.memory_refs],
            "metadata": self.metadata
        }
```

### 2. Flexible Session Management

#### Current State
```python
@dataclass
class FFSessionDTO:
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    metadata: Dict[str, Any]
```

#### Upgraded Design
```python
class SessionType(Enum):
    CHAT = "chat"              # Standard 1:1 chat
    PANEL = "panel"            # Multi-agent panel
    WORKFLOW = "workflow"      # Task-oriented workflow
    PLAYGROUND = "playground"  # Experimentation
    INTERVIEW = "interview"    # Structured Q&A

class ParticipantRole(Enum):
    HOST = "host"
    MEMBER = "member"
    OBSERVER = "observer"
    MODERATOR = "moderator"

@dataclass
class Participant:
    id: str
    type: SenderType  # user, agent, system
    role: ParticipantRole
    name: str
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionContext:
    summary: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)

@dataclass
class MemoryScope:
    include_user_history: bool = False
    include_global_knowledge: bool = True
    session_specific: bool = True
    time_window: Optional[int] = None  # days
    relevance_threshold: float = 0.7

@dataclass
class Session:
    id: str
    type: SessionType
    title: str
    participants: List[Participant]
    context: SessionContext
    memory_scope: MemoryScope
    capabilities: Set[str]  # multimodal, tools, memory, etc.
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    message_count: int = 0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_participant(self, participant: Participant):
        if participant.id not in [p.id for p in self.participants]:
            self.participants.append(participant)
            self.updated_at = current_timestamp()
    
    def remove_participant(self, participant_id: str):
        self.participants = [p for p in self.participants if p.id != participant_id]
        self.updated_at = current_timestamp()
    
    def update_context(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        self.updated_at = current_timestamp()
```

### 3. Agent & Persona System

#### Current State
```python
@dataclass
class FFPersonaDTO:
    id: str
    name: str
    description: str
    traits: List[str]
    expertise: List[str]
    communication_style: Dict[str, Any]
    created_at: str
    updated_at: str
    is_global: bool
    owner_id: Optional[str]
    metadata: Dict[str, Any]
```

#### Upgraded Design
```python
class AgentType(Enum):
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task_oriented"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SPECIALIZED = "specialized"

class AgentCapability(Enum):
    TEXT_GENERATION = "text_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_PROCESSING = "audio_processing"
    TOOL_USE = "tool_use"
    MEMORY_ACCESS = "memory_access"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    DOCUMENT_ANALYSIS = "document_analysis"

@dataclass
class AgentPrompt:
    system: str
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[Dict[str, Any]] = None

@dataclass
class Agent:
    id: str
    name: str
    type: AgentType
    description: str
    capabilities: Set[AgentCapability]
    prompt: AgentPrompt
    knowledge_domains: List[str]
    personality_traits: Dict[str, float]  # trait -> strength (0-1)
    communication_style: Dict[str, Any]
    tools: List[str]  # tool IDs this agent can use
    memory_access: List[str]  # memory stores this agent can access
    routing_keywords: List[str]  # keywords for topic routing
    max_tokens: int = 4096
    temperature: float = 0.7
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    is_active: bool = True
    owner_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_handle_topic(self, topic: str) -> float:
        """Calculate confidence score for handling a topic"""
        score = 0.0
        topic_lower = topic.lower()
        
        # Check routing keywords
        for keyword in self.routing_keywords:
            if keyword.lower() in topic_lower:
                score += 0.3
        
        # Check knowledge domains
        for domain in self.knowledge_domains:
            if domain.lower() in topic_lower:
                score += 0.5
        
        return min(score, 1.0)
```

### 4. Memory Management System

#### New Component (Not in Current System)
```python
class MemoryType(Enum):
    EPISODIC = "episodic"      # Specific events/conversations
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"        # Current context

@dataclass
class MemoryEntry:
    id: str
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    source_id: Optional[str] = None  # message/session/document ID
    source_type: Optional[str] = None
    timestamp: str = field(default_factory=current_timestamp)
    access_count: int = 0
    last_accessed: Optional[str] = None
    relevance_decay: float = 1.0  # decreases over time
    importance: float = 0.5  # 0-1
    associations: List[str] = field(default_factory=list)  # related memory IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def decay(self, rate: float = 0.01):
        """Apply time-based decay to relevance"""
        self.relevance_decay = max(0.1, self.relevance_decay - rate)
    
    def access(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = current_timestamp()
        # Boost relevance on access
        self.relevance_decay = min(1.0, self.relevance_decay + 0.1)

@dataclass
class MemoryStore:
    user_id: str
    store_type: str  # user, agent, global
    memories: Dict[str, MemoryEntry] = field(default_factory=dict)
    index: Optional[Any] = None  # vector index for similarity search
    capacity: int = 10000
    compression_enabled: bool = True
    
    def add_memory(self, memory: MemoryEntry):
        if len(self.memories) >= self.capacity:
            # Remove least relevant memory
            self._compress_memories()
        self.memories[memory.id] = memory
    
    def _compress_memories(self):
        """Compress or remove low-relevance memories"""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.relevance_decay * m.importance
        )
        # Remove bottom 10%
        to_remove = int(len(sorted_memories) * 0.1)
        for memory in sorted_memories[:to_remove]:
            del self.memories[memory.id]
    
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memories by relevance"""
        # This would use vector similarity in practice
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
        return sorted(results, key=lambda m: m.relevance_decay, reverse=True)[:limit]
```

### 5. Context & Routing System

#### New Component
```python
@dataclass
class TopicSegment:
    text: str
    topic: str
    confidence: float
    entities: Dict[str, List[str]]
    start_pos: int
    end_pos: int

@dataclass
class RoutingDecision:
    agent_id: str
    topic_segments: List[TopicSegment]
    confidence: float
    reasoning: str

@dataclass
class ConversationContext:
    raw_text: str
    topics: List[str]
    entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    intent: Optional[str]
    topic_segments: List[TopicSegment]
    routing_decisions: List[RoutingDecision]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_agent_context(self, agent_id: str) -> str:
        """Get context specific to an agent"""
        segments = []
        for decision in self.routing_decisions:
            if decision.agent_id == agent_id:
                segments.extend(decision.topic_segments)
        return " ".join([s.text for s in segments])

class ContextRouter:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.topic_classifier = None  # ML model
    
    async def route_message(self, message: Message) -> List[RoutingDecision]:
        """Route message to appropriate agents"""
        # Extract context
        context = await self._extract_context(message)
        
        # Make routing decisions
        decisions = []
        for segment in context.topic_segments:
            best_agent = None
            best_score = 0.0
            
            for agent in self.agents:
                score = agent.can_handle_topic(segment.topic)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent and best_score > 0.5:
                decisions.append(RoutingDecision(
                    agent_id=best_agent.id,
                    topic_segments=[segment],
                    confidence=best_score,
                    reasoning=f"Agent {best_agent.name} specializes in {segment.topic}"
                ))
        
        return decisions
    
    async def _extract_context(self, message: Message) -> ConversationContext:
        """Extract context from message"""
        # This would use NLP models in practice
        # Simplified implementation
        text = ""
        if isinstance(message.content, TextContent):
            text = message.content.text
        
        return ConversationContext(
            raw_text=text,
            topics=self._extract_topics(text),
            entities=self._extract_entities(text),
            sentiment=self._analyze_sentiment(text),
            intent=self._classify_intent(text),
            topic_segments=self._segment_by_topic(text),
            routing_decisions=[]
        )
```

### 6. Tool Framework

#### New Component
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable

class ToolCategory(Enum):
    SEARCH = "search"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    FILE_OPERATION = "file_operation"
    DATA_ANALYSIS = "data_analysis"
    EXTERNAL_API = "external_api"

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None

@dataclass
class ToolSchema:
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)

class Tool(ABC):
    def __init__(self, schema: ToolSchema):
        self.schema = schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters against schema"""
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            if param.name in kwargs and param.enum:
                if kwargs[param.name] not in param.enum:
                    raise ValueError(f"Invalid value for {param.name}")
        
        return True

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {}
    
    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.schema.name] = tool
        
        category = tool.schema.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.schema.name)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolSchema]:
        """List available tools"""
        if category:
            tool_names = self.categories.get(category, [])
            return [self.tools[name].schema for name in tool_names]
        return [tool.schema for tool in self.tools.values()]

# Example Tool Implementation
class WebSearchTool(Tool):
    def __init__(self):
        schema = ToolSchema(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query"
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results to return",
                    required=False,
                    default=10
                )
            ],
            returns={"results": "List of search results"}
        )
        super().__init__(schema)
    
    async def execute(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        self.validate_parameters(query=query, max_results=max_results)
        
        # Actual web search implementation
        results = await self._search_web(query, max_results)
        
        return {
            "results": results,
            "query": query,
            "count": len(results)
        }
    
    async def _search_web(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        # Implementation would use actual search API
        return []
```

### 7. Multimodal Processing

#### New Component
```python
from abc import ABC, abstractmethod
import mimetypes

class ProcessorRegistry:
    def __init__(self):
        self.processors: Dict[MessageType, MessageProcessor] = {}
    
    def register(self, message_type: MessageType, processor: MessageProcessor):
        self.processors[message_type] = processor
    
    def get_processor(self, message_type: MessageType) -> Optional[MessageProcessor]:
        return self.processors.get(message_type)

class MessageProcessor(ABC):
    @abstractmethod
    async def process(self, content: MessageContent) -> Dict[str, Any]:
        """Process message content"""
        pass
    
    @abstractmethod
    async def validate(self, content: MessageContent) -> bool:
        """Validate content"""
        pass

class ImageProcessor(MessageProcessor):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get("max_image_size", 10 * 1024 * 1024)  # 10MB
        self.allowed_formats = config.get("allowed_formats", ["jpeg", "png", "gif", "webp"])
    
    async def process(self, content: ImageContent) -> Dict[str, Any]:
        if not await self.validate(content):
            raise ValueError("Invalid image content")
        
        # Process image
        result = {
            "type": "image",
            "format": content.mime_type,
            "analysis": {}
        }
        
        # Extract features
        if content.data:
            result["analysis"]["size"] = len(content.data)
            result["analysis"]["features"] = await self._extract_features(content.data)
        
        if content.url:
            result["url"] = content.url
        
        return result
    
    async def validate(self, content: ImageContent) -> bool:
        if content.data and len(content.data) > self.max_size:
            return False
        
        # Check format
        format = content.mime_type.split("/")[-1]
        if format not in self.allowed_formats:
            return False
        
        return True
    
    async def _extract_features(self, data: bytes) -> Dict[str, Any]:
        # Would use computer vision model
        return {"objects": [], "text": [], "colors": []}

class AudioProcessor(MessageProcessor):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_duration = config.get("max_audio_duration", 300)  # 5 minutes
    
    async def process(self, content: AudioContent) -> Dict[str, Any]:
        if not await self.validate(content):
            raise ValueError("Invalid audio content")
        
        result = {
            "type": "audio",
            "format": content.mime_type,
            "transcription": None,
            "analysis": {}
        }
        
        # Transcribe audio
        if content.data:
            result["transcription"] = await self._transcribe(content.data)
            result["analysis"] = await self._analyze_audio(content.data)
        
        return result
    
    async def validate(self, content: AudioContent) -> bool:
        # Validation logic
        return True
    
    async def _transcribe(self, data: bytes) -> str:
        # Would use speech-to-text model
        return ""
    
    async def _analyze_audio(self, data: bytes) -> Dict[str, Any]:
        # Audio analysis
        return {"duration": 0, "language": "en", "speakers": 1}
```

### 8. Trace Logging System

#### New Component
```python
@dataclass
class TraceEvent:
    id: str
    timestamp: str
    event_type: str
    component: str
    session_id: Optional[str]
    user_id: Optional[str]
    agent_id: Optional[str]
    data: Dict[str, Any]
    parent_id: Optional[str] = None
    duration_ms: Optional[int] = None
    
@dataclass
class TraceSpan:
    id: str
    name: str
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[int] = None
    parent_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)
    
    def end(self):
        self.end_time = current_timestamp()
        # Calculate duration

class TraceLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spans: Dict[str, TraceSpan] = {}
        self.events: List[TraceEvent] = []
        self.exporters: List[TraceExporter] = []
    
    def start_span(self, name: str, parent_id: Optional[str] = None) -> TraceSpan:
        span = TraceSpan(
            id=f"span_{uuid.uuid4().hex[:8]}",
            name=name,
            start_time=current_timestamp(),
            parent_id=parent_id
        )
        self.spans[span.id] = span
        return span
    
    def log_event(self, event: TraceEvent):
        self.events.append(event)
        
        # Add to current span if exists
        if event.parent_id and event.parent_id in self.spans:
            self.spans[event.parent_id].events.append(event)
    
    def export_traces(self):
        for exporter in self.exporters:
            exporter.export(self.spans.values(), self.events)

# Example usage in code
async def process_message_with_tracing(message: Message, tracer: TraceLogger):
    span = tracer.start_span("process_message")
    
    try:
        # Log message received
        tracer.log_event(TraceEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=current_timestamp(),
            event_type="message_received",
            component="message_handler",
            session_id=message.session_id,
            data={"message_id": message.id, "type": message.content.type},
            parent_id=span.id
        ))
        
        # Process message
        result = await process_message(message)
        
        # Log completion
        tracer.log_event(TraceEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=current_timestamp(),
            event_type="message_processed",
            component="message_handler",
            session_id=message.session_id,
            data={"message_id": message.id, "success": True},
            parent_id=span.id
        ))
        
        return result
        
    except Exception as e:
        # Log error
        tracer.log_event(TraceEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=current_timestamp(),
            event_type="error",
            component="message_handler",
            session_id=message.session_id,
            data={"message_id": message.id, "error": str(e)},
            parent_id=span.id
        ))
        raise
    
    finally:
        span.end()
```

## Storage Structure

### Current Structure
```
flatfile_chat_database_v2/
├── users/
│   └── {user_id}/
│       ├── profile.json
│       └── chat_session_{session_id}/
│           ├── session.json
│           ├── messages.jsonl
│           ├── documents/
│           └── context.json
├── panel_sessions/
│   └── {panel_id}/
│       ├── panel.json
│       ├── messages.jsonl
│       └── insights/
└── personas_global/
    └── {persona_id}.json
```

### Upgraded Structure
```
flatfile_chat_database_v3/
├── users/
│   └── {user_id}/
│       ├── profile.json
│       ├── memory/
│       │   ├── episodic/
│       │   ├── semantic/
│       │   └── index.json
│       └── sessions/
│           └── {session_id}/
│               ├── metadata.json
│               ├── messages/
│               │   ├── {timestamp}_{message_id}.json
│               │   └── index.jsonl
│               ├── multimodal/
│               │   ├── images/
│               │   ├── audio/
│               │   └── video/
│               ├── documents/
│               │   ├── {doc_id}/
│               │   └── metadata.json
│               ├── context/
│               │   ├── current.json
│               │   └── history/
│               ├── traces/
│               │   └── {trace_id}.jsonl
│               └── tools/
│                   └── executions/
├── agents/
│   ├── global/
│   │   └── {agent_id}/
│   │       ├── config.json
│   │       ├── prompts/
│   │       └── knowledge/
│   └── user/
│       └── {user_id}/
│           └── {agent_id}/
├── memory/
│   ├── global/
│   │   ├── semantic/
│   │   └── procedural/
│   └── indexes/
│       └── vectors/
├── panels/
│   └── {panel_id}/
│       ├── metadata.json
│       ├── participants.json
│       ├── messages/
│       ├── decisions/
│       └── insights/
├── workflows/
│   └── {workflow_id}/
│       ├── definition.json
│       ├── state.json
│       └── executions/
└── system/
    ├── tools/
    │   └── {tool_id}/
    │       ├── schema.json
    │       └── config.json
    ├── models/
    │   └── {model_id}/
    └── traces/
        └── {date}/
```

### Storage Schema Updates

#### Message Storage Format
```json
{
  "id": "msg_abc123",
  "sender": {
    "type": "user",
    "id": "user_123",
    "name": "Alice"
  },
  "content": [
    {
      "type": "text",
      "text": "Can you analyze this image?",
      "format": "plain"
    },
    {
      "type": "image",
      "url": "multimodal/images/img_xyz.jpg",
      "mime_type": "image/jpeg",
      "alt_text": "Sales chart"
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "session_abc",
  "tool_calls": [
    {
      "tool_id": "image_analyzer",
      "function": "analyze",
      "arguments": {"image_path": "multimodal/images/img_xyz.jpg"},
      "call_id": "call_123"
    }
  ],
  "memory_refs": [
    {
      "memory_id": "mem_456",
      "memory_type": "semantic",
      "relevance_score": 0.85,
      "context": "Previous sales analysis"
    }
  ]
}
```

#### Session Metadata Format
```json
{
  "id": "session_abc123",
  "type": "chat",
  "title": "Sales Analysis Discussion",
  "participants": [
    {
      "id": "user_123",
      "type": "user",
      "role": "member",
      "name": "Alice",
      "capabilities": ["text", "image_upload"]
    },
    {
      "id": "agent_sales",
      "type": "agent", 
      "role": "member",
      "name": "Sales Analyst",
      "capabilities": ["text", "image_understanding", "data_analysis"]
    }
  ],
  "context": {
    "summary": "Analyzing Q4 sales performance",
    "goals": ["Understand sales trends", "Identify opportunities"],
    "active_tools": ["image_analyzer", "data_visualizer"]
  },
  "memory_scope": {
    "include_user_history": true,
    "session_specific": true,
    "relevance_threshold": 0.7
  },
  "capabilities": ["multimodal", "tools", "memory"],
  "created_at": "2024-01-15T10:00:00Z",
  "message_count": 45
}
```

## Migration Strategy

### Phase 1: Data Model Migration (Week 1)

1. **Create Migration Scripts**
```python
# migrate_messages.py
async def migrate_messages(old_path: Path, new_path: Path):
    """Migrate messages to new format"""
    # Read old messages
    old_messages = await read_jsonl(old_path / "messages.jsonl")
    
    # Convert to new format
    for old_msg in old_messages:
        new_msg = Message(
            id=old_msg.get("message_id", generate_id()),
            sender=MessageSender(
                type=SenderType.USER if old_msg["role"] == "user" else SenderType.AGENT,
                id=old_msg.get("user_id", "unknown"),
                name=old_msg.get("role")
            ),
            content=TextContent(text=old_msg["content"]),
            timestamp=old_msg["timestamp"],
            metadata=old_msg.get("metadata", {})
        )
        
        # Save in new structure
        msg_path = new_path / "messages" / f"{new_msg.timestamp}_{new_msg.id}.json"
        await save_json(msg_path, new_msg.to_dict())
```

2. **Migrate User Data**
```python
async def migrate_user_data(old_base: Path, new_base: Path):
    """Migrate all user data"""
    users = list_directories(old_base / "users")
    
    for user_id in users:
        # Migrate profile
        old_profile = await read_json(old_base / "users" / user_id / "profile.json")
        await save_json(new_base / "users" / user_id / "profile.json", old_profile)
        
        # Migrate sessions
        sessions = list_directories(old_base / "users" / user_id)
        for session_dir in sessions:
            if session_dir.startswith("chat_session_"):
                session_id = session_dir.replace("chat_session_", "")
                await migrate_session(
                    old_base / "users" / user_id / session_dir,
                    new_base / "users" / user_id / "sessions" / session_id
                )
```

### Phase 2: Add New Components (Week 2-3)

1. **Initialize New Systems**
```python
async def initialize_new_components(base_path: Path):
    """Initialize new component structures"""
    # Create memory stores
    await create_memory_stores(base_path / "memory")
    
    # Set up agent registry
    await create_agent_registry(base_path / "agents")
    
    # Initialize tool registry
    await create_tool_registry(base_path / "system" / "tools")
```

2. **Backward Compatibility Layer**
```python
class LegacyAdapter:
    """Adapter for legacy code compatibility"""
    
    def __init__(self, storage_manager: StorageManagerV3):
        self.storage = storage_manager
    
    async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO):
        """Legacy method adapter"""
        # Convert old format to new
        new_message = self._convert_message(message)
        return await self.storage.add_message(user_id, session_id, new_message)
    
    def _convert_message(self, old_msg: FFMessageDTO) -> Message:
        """Convert legacy message format"""
        return Message(
            id=old_msg.message_id,
            sender=MessageSender(
                type=SenderType.USER if old_msg.role == "user" else SenderType.AGENT,
                id="legacy",
                name=old_msg.role
            ),
            content=TextContent(text=old_msg.content),
            timestamp=old_msg.timestamp,
            metadata=old_msg.metadata
        )
```

### Phase 3: Testing & Validation (Week 4)

1. **Data Integrity Tests**
```python
async def validate_migration():
    """Validate migrated data"""
    # Check message counts
    old_count = await count_messages_old()
    new_count = await count_messages_new()
    assert old_count == new_count
    
    # Validate content integrity
    sample_messages = await get_sample_messages()
    for msg in sample_messages:
        assert await verify_message_content(msg)
```

2. **Performance Tests**
```python
async def benchmark_operations():
    """Benchmark new vs old operations"""
    results = {}
    
    # Test message operations
    results["message_write"] = await benchmark_message_writes()
    results["message_read"] = await benchmark_message_reads()
    results["search"] = await benchmark_search()
    
    return results
```

## Implementation Roadmap

### Week 1-2: Core Models & Protocols
- [ ] Implement new message models
- [ ] Create session management system
- [ ] Define all protocols
- [ ] Set up event bus

### Week 3-4: Storage & Migration
- [ ] Implement new storage structure
- [ ] Create migration scripts
- [ ] Build backward compatibility
- [ ] Test data migration

### Week 5-6: Capabilities
- [ ] Multimodal processing
- [ ] Tool framework
- [ ] Memory system
- [ ] Basic routing

### Week 7-8: Advanced Features
- [ ] Multi-agent orchestration
- [ ] Topic routing
- [ ] Trace logging
- [ ] Real-time features

### Week 9-10: Integration & Testing
- [ ] API development
- [ ] Integration testing
- [ ] Performance optimization
- [ ] Documentation

## Code Examples

### Example 1: Multimodal Chat
```python
# Creating a multimodal message
message = Message(
    id=generate_message_id(),
    sender=MessageSender(type=SenderType.USER, id="user_123", name="Alice"),
    content=[
        TextContent(text="What's in this image?"),
        ImageContent(url="path/to/image.jpg", mime_type="image/jpeg")
    ],
    session_id="session_abc"
)

# Processing the message
async def handle_multimodal_message(message: Message):
    # Route to capable agent
    if any(isinstance(c, ImageContent) for c in message.content):
        agent = await get_agent_with_capability(AgentCapability.IMAGE_UNDERSTANDING)
        response = await agent.process(message)
    
    return response
```

### Example 2: Tool Usage
```python
# Define a calculation tool
class CalculatorTool(Tool):
    def __init__(self):
        schema = ToolSchema(
            name="calculator",
            description="Perform mathematical calculations",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(name="expression", type="string", 
                            description="Mathematical expression")
            ],
            returns={"result": "number"}
        )
        super().__init__(schema)
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        # Safe evaluation of math expression
        result = eval_math_expression(expression)
        return {"result": result, "expression": expression}

# Using the tool
tool_call = ToolCall(
    tool_id="calculator",
    function="execute",
    arguments={"expression": "2 + 2 * 3"}
)

result = await tool_registry.execute_tool(tool_call)
```

### Example 3: Multi-Agent Panel
```python
# Create a panel session
panel_session = Session(
    id=generate_session_id(),
    type=SessionType.PANEL,
    title="Product Strategy Discussion",
    participants=[
        Participant(id="user_123", type=SenderType.USER, role=ParticipantRole.MODERATOR),
        Participant(id="agent_product", type=SenderType.AGENT, role=ParticipantRole.MEMBER),
        Participant(id="agent_marketing", type=SenderType.AGENT, role=ParticipantRole.MEMBER),
        Participant(id="agent_engineering", type=SenderType.AGENT, role=ParticipantRole.MEMBER)
    ],
    context=SessionContext(
        goals=["Define Q1 product roadmap", "Align on priorities"],
        constraints=["Limited engineering resources", "Q4 commitments"]
    ),
    capabilities={"multimodal", "tools", "memory"},
    memory_scope=MemoryScope(include_user_history=True)
)

# Route message to panel members
async def handle_panel_message(message: Message, session: Session):
    # Extract context for routing
    context = await context_router.extract_context(message)
    
    # Get responses from relevant agents
    responses = []
    for decision in context.routing_decisions:
        agent = await get_agent(decision.agent_id)
        agent_context = context.get_agent_context(decision.agent_id)
        
        response = await agent.respond(agent_context)
        responses.append(response)
    
    # Synthesize responses
    final_response = await synthesize_panel_responses(responses)
    return final_response
```

### Example 4: Memory-Enhanced Conversation
```python
# Store conversation in memory
async def update_conversation_memory(session_id: str, message: Message, response: Message):
    memory_store = await get_memory_store(message.sender.id)
    
    # Create episodic memory
    episode = MemoryEntry(
        id=generate_memory_id(),
        type=MemoryType.EPISODIC,
        content=f"User asked: {message.content[0].text}\nAssistant responded: {response.content[0].text}",
        source_id=session_id,
        source_type="session",
        importance=0.7
    )
    
    # Extract semantic memories
    entities = await extract_entities(message.content[0].text)
    for entity_type, values in entities.items():
        for value in values:
            semantic = MemoryEntry(
                id=generate_memory_id(),
                type=MemoryType.SEMANTIC,
                content=f"{entity_type}: {value}",
                source_id=message.id,
                source_type="message",
                importance=0.5
            )
            memory_store.add_memory(semantic)
    
    memory_store.add_memory(episode)

# Retrieve relevant memories
async def get_relevant_memories(query: str, user_id: str) -> List[MemoryEntry]:
    memory_store = await get_memory_store(user_id)
    
    # Search memories
    memories = await memory_store.search(query, limit=5)
    
    # Update access patterns
    for memory in memories:
        memory.access()
    
    return memories
```

## Testing Strategy

### Unit Tests
```python
# Test message creation
def test_message_creation():
    message = Message(
        id="test_123",
        sender=MessageSender(type=SenderType.USER, id="user_1"),
        content=TextContent(text="Hello")
    )
    
    assert message.id == "test_123"
    assert isinstance(message.content, TextContent)
    assert message.content.text == "Hello"

# Test tool execution
async def test_tool_execution():
    tool = CalculatorTool()
    result = await tool.execute(expression="2 + 2")
    
    assert result["result"] == 4
    assert result["expression"] == "2 + 2"
```

### Integration Tests
```python
# Test multimodal message flow
async def test_multimodal_flow():
    # Create multimodal message
    message = create_multimodal_message()
    
    # Process through system
    response = await process_message(message)
    
    # Verify response
    assert response is not None
    assert len(response.content) > 0
```

### Performance Tests
```python
# Benchmark message throughput
async def test_message_throughput():
    start_time = time.time()
    message_count = 10000
    
    for i in range(message_count):
        message = create_test_message(i)
        await storage.add_message("user_1", "session_1", message)
    
    duration = time.time() - start_time
    throughput = message_count / duration
    
    assert throughput > 1000  # At least 1000 messages/second
```

## Conclusion

This comprehensive upgrade specification transforms the Flatfile Chat Database into a powerful, modular platform capable of supporting diverse chat applications. The design emphasizes:

1. **Flexibility** - Support for all 25+ use cases
2. **Modularity** - Clean separation of concerns
3. **Extensibility** - Easy to add new capabilities
4. **Performance** - Efficient storage and retrieval
5. **Maintainability** - Clear code organization

The phased implementation approach allows for incremental development while maintaining system stability. With approximately 70% code reuse from the existing system, the upgrade builds on a solid foundation while adding significant new capabilities.

Next steps:
1. Review and approve specification
2. Set up development environment
3. Begin Phase 1 implementation
4. Establish testing framework
5. Create detailed API documentation