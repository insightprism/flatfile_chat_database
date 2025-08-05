# Flatfile Storage System v2 - Upgrade Specification

## Executive Summary

This specification outlines the storage system upgrades required to support advanced chat applications. The Flatfile Chat Database is a **storage backend** that provides data persistence, retrieval, and search capabilities. Chat applications connect to this storage system via APIs.

### Scope Clarification
- **This specification**: Storage schema, data models, persistence layer, search/retrieval
- **NOT in scope**: Chat UI, message routing logic, LLM integration, real-time websockets

### Key Storage Upgrades
1. **Polymorphic Message Storage** - Store text, images, audio, video, tool calls
2. **Hierarchical Memory Storage** - User/agent/global memory with decay
3. **Multi-Agent Data Models** - Agent configurations, personas, capabilities
4. **Advanced Indexing** - Topic-based, temporal, vector-based retrieval
5. **Trace/Audit Storage** - Conversation flows, decision paths
6. **Tool Execution Logs** - Tool schemas, execution history
7. **Session Flexibility** - Multiple session types (chat, panel, workflow)

## Current Storage Capabilities

The existing system already provides:
- User profiles and session management
- Message storage (JSONL format)
- Document storage with metadata
- Vector embeddings and similarity search
- Basic panel/multi-persona support
- File-based backend with abstraction layer

## Required Storage Enhancements

### 1. Message Storage Enhancement

#### Current Schema
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

#### Enhanced Storage Schema
```python
@dataclass
class MessageDTO:
    """Storage model for polymorphic messages"""
    id: str
    sender_type: str  # user, agent, system, tool
    sender_id: str
    sender_name: Optional[str]
    
    # Polymorphic content storage
    content_type: str  # text, image, audio, video, tool_call, tool_response
    content_data: Dict[str, Any]  # Type-specific data
    
    # Multimodal support
    attachments: List[Dict[str, Any]]  # Multiple content items
    
    # Relationships
    session_id: str
    thread_id: Optional[str]
    parent_id: Optional[str]
    
    # Tool integration
    tool_calls: List[Dict[str, Any]]
    tool_responses: List[Dict[str, Any]]
    
    # Memory references
    memory_refs: List[Dict[str, Any]]
    
    # Metadata
    timestamp: str
    metadata: Dict[str, Any]
```

#### Storage Format Example
```json
{
  "id": "msg_abc123",
  "sender_type": "user",
  "sender_id": "user_123",
  "sender_name": "Alice",
  "content_type": "multimodal",
  "content_data": {
    "parts": [
      {
        "type": "text",
        "text": "Analyze this chart"
      },
      {
        "type": "image",
        "storage_path": "multimodal/images/img_xyz.jpg",
        "mime_type": "image/jpeg",
        "size_bytes": 245780
      }
    ]
  },
  "session_id": "session_abc",
  "tool_calls": [
    {
      "id": "call_123",
      "tool_name": "image_analyzer",
      "arguments": {"image_id": "img_xyz"}
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. Session Storage Enhancement

#### Current Schema
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

#### Enhanced Storage Schema
```python
@dataclass
class SessionDTO:
    """Flexible session storage supporting multiple types"""
    id: str
    type: str  # chat, panel, workflow, playground
    title: str
    
    # Participants
    participants: List[Dict[str, Any]]  # Users and agents
    
    # Context
    context: Dict[str, Any]  # Goals, constraints, shared knowledge
    
    # Configuration
    capabilities: List[str]  # Enabled features
    memory_config: Dict[str, Any]  # Memory access settings
    tool_config: Dict[str, Any]  # Available tools
    
    # Metadata
    created_at: str
    updated_at: str
    message_count: int
    active: bool
    metadata: Dict[str, Any]
```

### 3. Memory Storage System

#### New Storage Structure
```
memory/
├── user/
│   └── {user_id}/
│       ├── episodic/
│       │   └── {memory_id}.json
│       ├── semantic/
│       │   └── {memory_id}.json
│       ├── working/
│       │   └── {session_id}.json
│       └── index.json
├── agent/
│   └── {agent_id}/
│       └── [same structure]
└── global/
    ├── semantic/
    │   └── {memory_id}.json
    └── procedural/
        └── {memory_id}.json
```

#### Memory Entry Schema
```python
@dataclass
class MemoryEntryDTO:
    """Storage model for memory entries"""
    id: str
    type: str  # episodic, semantic, procedural, working
    content: str
    
    # Vector storage
    embedding_id: Optional[str]  # Reference to vector storage
    
    # Source tracking
    source_type: str  # message, document, external
    source_id: str
    
    # Temporal data
    created_at: str
    last_accessed: str
    access_count: int
    
    # Relevance
    importance: float  # 0-1
    decay_factor: float  # 0-1
    
    # Relationships
    related_memories: List[str]  # Memory IDs
    
    metadata: Dict[str, Any]
```

### 4. Agent/Persona Storage

#### Enhanced Agent Storage
```
agents/
├── global/
│   └── {agent_id}/
│       ├── config.json      # Agent configuration
│       ├── prompts.json     # System prompts
│       ├── capabilities.json # What the agent can do
│       ├── knowledge/       # Agent-specific knowledge
│       └── routing.json     # Topic routing rules
└── user/
    └── {user_id}/
        └── {agent_id}/
            └── [same structure]
```

#### Agent Configuration Schema
```python
@dataclass
class AgentConfigDTO:
    """Storage model for agent configurations"""
    id: str
    name: str
    type: str  # conversational, analytical, creative, etc.
    
    # Capabilities
    capabilities: List[str]  # text, image_understanding, tool_use, etc.
    
    # Knowledge domains
    expertise: List[str]
    routing_keywords: List[str]  # For topic-based routing
    
    # Configuration
    max_tokens: int
    temperature: float
    model_preferences: Dict[str, Any]
    
    # Access control
    memory_access: List[str]  # Memory store IDs
    tool_access: List[str]  # Tool IDs
    
    # Metadata
    created_at: str
    updated_at: str
    owner_id: Optional[str]
    active: bool
```

### 5. Tool Execution Storage

#### Tool Registry Storage
```
tools/
├── registry/
│   └── {tool_id}/
│       ├── schema.json     # Tool definition
│       └── config.json     # Tool configuration
└── executions/
    └── {session_id}/
        └── {execution_id}.json
```

#### Tool Execution Schema
```python
@dataclass
class ToolExecutionDTO:
    """Storage model for tool executions"""
    id: str
    session_id: str
    message_id: str
    tool_id: str
    
    # Execution details
    function: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    
    # Timing
    started_at: str
    completed_at: Optional[str]
    duration_ms: Optional[int]
    
    # Context
    user_id: str
    agent_id: Optional[str]
    
    metadata: Dict[str, Any]
```

### 6. Multimodal Content Storage

#### Storage Structure
```
sessions/
└── {session_id}/
    ├── multimodal/
    │   ├── images/
    │   │   ├── {image_id}.jpg
    │   │   └── {image_id}.metadata.json
    │   ├── audio/
    │   │   ├── {audio_id}.mp3
    │   │   └── {audio_id}.metadata.json
    │   ├── video/
    │   │   ├── {video_id}.mp4
    │   │   └── {video_id}.metadata.json
    │   └── documents/
    │       ├── {doc_id}.pdf
    │       └── {doc_id}.metadata.json
    └── processing/
        └── {content_id}_analysis.json
```

#### Multimodal Metadata Schema
```python
@dataclass
class MultimodalMetadataDTO:
    """Storage model for multimodal content metadata"""
    id: str
    type: str  # image, audio, video, document
    original_filename: str
    storage_path: str
    
    # File details
    mime_type: str
    size_bytes: int
    hash: str  # For deduplication
    
    # Processing results
    transcription: Optional[str]  # For audio/video
    extracted_text: Optional[str]  # For images/documents
    analysis: Dict[str, Any]  # AI analysis results
    
    # Relationships
    message_id: str
    session_id: str
    user_id: str
    
    # Metadata
    uploaded_at: str
    processed_at: Optional[str]
    metadata: Dict[str, Any]
```

### 7. Trace/Audit Storage

#### Trace Storage Structure
```
traces/
└── {session_id}/
    ├── conversation_flow.jsonl
    ├── routing_decisions.jsonl
    ├── tool_executions.jsonl
    └── performance_metrics.json
```

#### Trace Event Schema
```python
@dataclass
class TraceEventDTO:
    """Storage model for trace events"""
    id: str
    timestamp: str
    event_type: str  # message_received, agent_invoked, tool_executed, etc.
    
    # Context
    session_id: str
    user_id: Optional[str]
    agent_id: Optional[str]
    message_id: Optional[str]
    
    # Event data
    component: str  # Which system component
    action: str  # What happened
    data: Dict[str, Any]  # Event-specific data
    
    # Performance
    duration_ms: Optional[int]
    
    # Relationships
    parent_event_id: Optional[str]
    
    metadata: Dict[str, Any]
```

## Storage API Enhancements

### Current API (FFStorageManager)
```python
# Existing methods
async def create_session(user_id: str, title: str) -> str
async def add_message(user_id: str, session_id: str, message: FFMessageDTO) -> bool
async def get_messages(user_id: str, session_id: str, limit: int) -> List[FFMessageDTO]
```

### Enhanced Storage API
```python
class StorageManagerV2:
    # Message operations (polymorphic)
    async def store_message(self, message: MessageDTO) -> bool
    async def get_message(self, message_id: str) -> Optional[MessageDTO]
    async def query_messages(self, filters: MessageFilters) -> List[MessageDTO]
    
    # Multimodal content
    async def store_multimodal_content(self, content: MultimodalContentDTO) -> str
    async def get_multimodal_content(self, content_id: str) -> Optional[bytes]
    async def get_multimodal_metadata(self, content_id: str) -> Optional[MultimodalMetadataDTO]
    
    # Memory operations
    async def store_memory(self, memory: MemoryEntryDTO) -> bool
    async def query_memories(self, query: MemoryQuery) -> List[MemoryEntryDTO]
    async def update_memory_access(self, memory_id: str) -> bool
    async def decay_memories(self, strategy: DecayStrategy) -> int
    
    # Agent operations
    async def store_agent_config(self, agent: AgentConfigDTO) -> bool
    async def get_agent_config(self, agent_id: str) -> Optional[AgentConfigDTO]
    async def find_agents_for_topic(self, topic: str) -> List[AgentConfigDTO]
    
    # Tool operations
    async def log_tool_execution(self, execution: ToolExecutionDTO) -> bool
    async def get_tool_history(self, session_id: str) -> List[ToolExecutionDTO]
    
    # Trace operations
    async def log_trace_event(self, event: TraceEventDTO) -> bool
    async def get_session_trace(self, session_id: str) -> List[TraceEventDTO]
    
    # Session operations (enhanced)
    async def create_session(self, session: SessionDTO) -> str
    async def update_session_context(self, session_id: str, context: Dict) -> bool
    async def add_session_participant(self, session_id: str, participant: Dict) -> bool
```

## Index Enhancements

### Current Indexes
- Message timestamps (chronological retrieval)
- User/session relationships
- Basic text search

### Required New Indexes

1. **Message Content Type Index**
```json
{
  "index_name": "message_content_types",
  "fields": ["session_id", "content_type"],
  "purpose": "Quickly find all images/audio/video in a session"
}
```

2. **Tool Execution Index**
```json
{
  "index_name": "tool_executions",
  "fields": ["session_id", "tool_id", "timestamp"],
  "purpose": "Track tool usage patterns"
}
```

3. **Memory Relevance Index**
```json
{
  "index_name": "memory_relevance",
  "fields": ["user_id", "type", "importance", "decay_factor"],
  "purpose": "Efficient memory retrieval by relevance"
}
```

4. **Agent Routing Index**
```json
{
  "index_name": "agent_topics",
  "fields": ["expertise", "routing_keywords"],
  "purpose": "Quick agent selection for topics"
}
```

5. **Multimodal Content Index**
```json
{
  "index_name": "multimodal_content",
  "fields": ["session_id", "type", "size_bytes", "timestamp"],
  "purpose": "Manage storage quotas and retrieval"
}
```

## Migration Plan

### Phase 1: Schema Updates (Week 1)
1. Define new data models
2. Create storage directories
3. Update configuration system

### Phase 2: API Implementation (Week 2-3)
1. Implement new storage methods
2. Add multimodal file handling
3. Build memory management

### Phase 3: Index Creation (Week 4)
1. Build new indexes
2. Migrate existing data
3. Optimize query paths

### Phase 4: Testing & Optimization (Week 5)
1. Load testing
2. Query optimization
3. Storage efficiency

## Storage Efficiency Considerations

### 1. Message Storage
- Store large content (images, audio) separately from messages
- Use references in messages to point to multimodal storage
- Compress old messages after 30 days

### 2. Memory Management
- Implement automatic decay for low-importance memories
- Archive old episodic memories
- Use vector compression for embeddings

### 3. Multimodal Storage
- Implement content deduplication via hashing
- Use appropriate compression for each media type
- Set storage quotas per user/session

### 4. Trace Storage
- Use JSONL for append-only trace logs
- Archive old traces after sessions complete
- Aggregate metrics for long-term storage

## Performance Requirements

### Read Performance
- Message retrieval: < 50ms for recent 100 messages
- Memory search: < 200ms for relevance-based search
- Multimodal metadata: < 10ms lookup

### Write Performance
- Message storage: < 100ms including indexing
- Multimodal upload: Streaming support for large files
- Trace logging: < 5ms (async, non-blocking)

### Query Performance
- Complex message queries: < 500ms
- Cross-session memory search: < 1s
- Agent routing lookup: < 50ms

## Backward Compatibility

Since the codebase is not in production, we can make breaking changes. However, we should provide migration scripts for any existing test data:

```python
# Migration script example
async def migrate_v1_to_v2():
    # Migrate messages to new schema
    old_messages = await load_v1_messages()
    for msg in old_messages:
        new_msg = MessageDTO(
            id=msg.message_id,
            sender_type="user" if msg.role == "user" else "agent",
            sender_id=msg.user_id or "unknown",
            content_type="text",
            content_data={"text": msg.content},
            session_id=msg.session_id,
            timestamp=msg.timestamp,
            metadata=msg.metadata
        )
        await store_message(new_msg)
```

## Summary

This specification focuses purely on the **storage layer** enhancements needed to support advanced chat applications. The key changes are:

1. **Polymorphic message storage** for multimodal content
2. **Hierarchical memory system** with decay and relevance
3. **Flexible session types** supporting various interaction patterns
4. **Agent configuration storage** for multi-agent systems
5. **Tool execution logging** for reproducibility
6. **Comprehensive trace storage** for debugging
7. **Enhanced indexing** for efficient retrieval

The storage system remains agnostic to the chat application logic, providing a clean API for data persistence and retrieval. Chat applications can connect to this storage layer to build any of the 25+ use cases we identified.