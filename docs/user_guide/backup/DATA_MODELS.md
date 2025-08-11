# Data Models and Storage Schemas

## Overview

This document defines the data models and storage schemas for the modular chat platform. Each module has its own storage namespace and schema, allowing for independent scaling and optimization.

## Core Data Types

### Base Types

```python
# Timestamps
Timestamp = str  # ISO 8601 format: "2024-01-15T10:30:00Z"

# Identifiers
UserId = str     # Format: "user_{uuid}"
SessionId = str  # Format: "sess_{timestamp}_{random}"
MessageId = str  # Format: "msg_{timestamp}_{random}"
AgentId = str    # Format: "agent_{name}_{version}"
DocumentId = str # Format: "doc_{hash}"

# Common Enums
Role = Literal["user", "assistant", "system", "tool"]
MessageStatus = Literal["pending", "delivered", "read", "failed"]
ContentType = Literal["text", "image", "audio", "video", "document"]
```

### Core Models

```python
@dataclass
class Message:
    """Universal message format used across modules."""
    id: MessageId
    session_id: SessionId
    role: Role
    content: str
    timestamp: Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)
    status: MessageStatus = "pending"
    tokens: Optional[int] = None

@dataclass
class Session:
    """Chat session container."""
    id: SessionId
    user_id: UserId
    title: str
    created_at: Timestamp
    updated_at: Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    
@dataclass
class Attachment:
    """Media attachment for messages."""
    id: str
    type: ContentType
    url: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Module-Specific Models

### 1. Text Chat Module

```python
@dataclass
class ChatState:
    """Current state of a chat session."""
    session_id: SessionId
    last_message_id: Optional[MessageId]
    turn_count: int
    total_tokens: int
    active_since: Timestamp
    last_activity: Timestamp
    context_summary: Optional[str]

@dataclass
class StreamingResponse:
    """Streaming response chunk."""
    session_id: SessionId
    chunk_id: str
    content: str
    index: int
    is_final: bool
    timestamp: Timestamp
```

**Storage Structure:**
```
text_chat/
├── sessions/
│   └── {user_id}/
│       └── {session_id}/
│           ├── metadata.json
│           │   {
│           │     "id": "sess_20240115_103000_abc123",
│           │     "user_id": "user_123",
│           │     "title": "Morning Chat",
│           │     "created_at": "2024-01-15T10:30:00Z",
│           │     "updated_at": "2024-01-15T11:45:00Z",
│           │     "metadata": {
│           │       "client": "web",
│           │       "language": "en"
│           │     }
│           │   }
│           ├── messages.jsonl
│           │   {"id":"msg_001","role":"user","content":"Hello","timestamp":"2024-01-15T10:30:00Z"}
│           │   {"id":"msg_002","role":"assistant","content":"Hi there!","timestamp":"2024-01-15T10:30:02Z"}
│           └── state.json
│               {
│                 "session_id": "sess_20240115_103000_abc123",
│                 "last_message_id": "msg_002",
│                 "turn_count": 2,
│                 "total_tokens": 150,
│                 "active_since": "2024-01-15T10:30:00Z",
│                 "last_activity": "2024-01-15T10:30:02Z"
│               }
└── indices/
    └── user_sessions.json
        {
          "user_123": {
            "sessions": ["sess_20240115_103000_abc123", "sess_20240114_093000_def456"],
            "total_sessions": 2,
            "last_active": "2024-01-15T10:30:02Z"
          }
        }
```

### 2. Memory Module

```python
@dataclass
class MemoryItem:
    """Individual memory entry."""
    id: str
    key: str
    content: Any
    memory_type: Literal["short_term", "long_term", "working"]
    created_at: Timestamp
    accessed_at: Timestamp
    access_count: int
    importance_score: float
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextWindow:
    """Active context for a session."""
    session_id: SessionId
    messages: List[Message]
    facts: List[str]
    summary: Optional[str]
    token_count: int
    last_updated: Timestamp

@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""
    consolidated_count: int
    retained_count: int
    discarded_count: int
    new_insights: List[str]
    timestamp: Timestamp
```

**Storage Structure:**
```
memory/
├── short_term/
│   └── {session_id}/
│       ├── context.json
│       │   {
│       │     "session_id": "sess_123",
│       │     "messages": [...],
│       │     "facts": ["User likes coffee", "Meeting at 3pm"],
│       │     "summary": "Discussing daily schedule",
│       │     "token_count": 2500,
│       │     "last_updated": "2024-01-15T11:00:00Z"
│       │   }
│       ├── facts.jsonl
│       │   {"id":"fact_001","content":"User prefers morning meetings","importance":0.8}
│       │   {"id":"fact_002","content":"User has a dog named Max","importance":0.6}
│       └── working_memory.json
├── long_term/
│   └── {user_id}/
│       ├── memory_bank.jsonl
│       │   {"id":"mem_001","key":"preferences.meeting","content":"morning preferred","importance":0.9}
│       │   {"id":"mem_002","key":"personal.pets","content":"dog named Max","importance":0.7}
│       ├── relationships.json
│       │   {
│       │     "entities": {
│       │       "John": {"type": "colleague", "sentiment": 0.8},
│       │       "Alice": {"type": "manager", "sentiment": 0.9}
│       │     }
│       │   }
│       └── preferences.json
│           {
│             "communication": {"style": "formal", "length": "concise"},
│             "topics": {"interests": ["technology", "sports"], "avoid": ["politics"]}
│           }
└── consolidation/
    └── logs/
        └── {date}.jsonl
```

### 3. RAG Module

```python
@dataclass
class Document:
    """Document in knowledge base."""
    id: DocumentId
    title: str
    source: str
    content_type: str
    size_bytes: int
    created_at: Timestamp
    updated_at: Timestamp
    metadata: Dict[str, Any]
    chunk_count: int
    embedding_model: str

@dataclass
class Chunk:
    """Document chunk for retrieval."""
    id: str
    document_id: DocumentId
    content: str
    index: int
    tokens: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """RAG search result."""
    chunk_id: str
    document_id: DocumentId
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalContext:
    """Retrieved context for generation."""
    query: str
    results: List[SearchResult]
    total_tokens: int
    sources: List[str]
    timestamp: Timestamp
```

**Storage Structure:**
```
rag/
├── documents/
│   └── {document_id}/
│       ├── original.pdf
│       ├── metadata.json
│       │   {
│       │     "id": "doc_a1b2c3",
│       │     "title": "Product Manual",
│       │     "source": "uploads/manual.pdf",
│       │     "content_type": "application/pdf",
│       │     "size_bytes": 1048576,
│       │     "created_at": "2024-01-15T09:00:00Z",
│       │     "chunk_count": 42,
│       │     "embedding_model": "text-embedding-3-small"
│       │   }
│       └── chunks.jsonl
│           {"id":"chunk_001","content":"Introduction to...","index":0,"tokens":125}
│           {"id":"chunk_002","content":"Chapter 1...","index":1,"tokens":200}
├── embeddings/
│   └── {index_id}/
│       ├── vectors.npy         # NumPy array of embeddings
│       ├── index.faiss         # FAISS index file
│       └── mapping.json        # Chunk ID to vector index mapping
│           {
│             "chunk_001": 0,
│             "chunk_002": 1
│           }
└── indices/
    ├── document_registry.json
    │   {
    │     "documents": {
    │       "doc_a1b2c3": {
    │         "title": "Product Manual",
    │         "chunks": 42,
    │         "indexed": true
    │       }
    │     }
    │   }
    └── search_cache.json
```

### 4. Multi-Agent Module

```python
@dataclass
class Agent:
    """AI agent definition."""
    id: AgentId
    name: str
    type: str
    capabilities: List[str]
    persona_id: Optional[str]
    config: Dict[str, Any]
    status: Literal["active", "inactive", "busy"]
    created_at: Timestamp

@dataclass
class AgentResponse:
    """Response from an agent."""
    agent_id: AgentId
    conversation_id: str
    content: str
    confidence: float
    reasoning: Optional[str]
    timestamp: Timestamp

@dataclass
class Conversation:
    """Multi-agent conversation."""
    id: str
    topic: str
    participants: List[AgentId]
    max_turns: Optional[int]
    current_turn: int
    status: Literal["active", "completed", "terminated"]
    created_at: Timestamp
    
@dataclass
class ConsensusResult:
    """Result of agent consensus."""
    conversation_id: str
    decision: str
    agreement_score: float
    dissenting_agents: List[AgentId]
    reasoning: Dict[AgentId, str]
```

**Storage Structure:**
```
multi_agent/
├── agents/
│   └── {agent_id}/
│       ├── definition.json
│       │   {
│       │     "id": "agent_finance_v1",
│       │     "name": "Finance Expert",
│       │     "type": "specialist",
│       │     "capabilities": ["financial_analysis", "budgeting"],
│       │     "persona_id": "persona_professional",
│       │     "config": {"temperature": 0.7, "max_tokens": 1000}
│       │   }
│       ├── state.json
│       └── performance.jsonl
├── conversations/
│   └── {conversation_id}/
│       ├── metadata.json
│       ├── participants.json
│       ├── transcript.jsonl
│       │   {"turn":1,"agent":"agent_finance_v1","content":"From a financial perspective..."}
│       │   {"turn":2,"agent":"agent_legal_v1","content":"The legal implications are..."}
│       └── decisions.json
│           {
│             "final_decision": "Proceed with option A",
│             "consensus": {
│               "agreement_score": 0.85,
│               "votes": {"agent_finance_v1": "agree", "agent_legal_v1": "agree"}
│             }
│           }
└── coordination/
    └── active_tasks.json
```

### 5. Tool Use Module

```python
@dataclass
class Tool:
    """External tool definition."""
    id: str
    name: str
    description: str
    endpoint: str
    auth_type: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    rate_limits: Dict[str, int]
    cost_per_use: Optional[float]

@dataclass
class ToolExecution:
    """Tool execution record."""
    id: str
    tool_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    status: Literal["pending", "success", "failed"]
    error: Optional[str]
    duration_ms: int
    cost: Optional[float]
    timestamp: Timestamp

@dataclass
class ToolChain:
    """Sequence of tool executions."""
    id: str
    tools: List[str]
    strategy: str
    input_data: Dict[str, Any]
    executions: List[ToolExecution]
    final_output: Optional[Dict[str, Any]]
    total_duration_ms: int
    total_cost: float
```

**Storage Structure:**
```
tool_use/
├── tools/
│   └── {tool_id}/
│       ├── definition.json
│       │   {
│       │     "id": "tool_weather_api",
│       │     "name": "Weather API",
│       │     "endpoint": "https://api.weather.com/v1/current",
│       │     "auth_type": "api_key",
│       │     "input_schema": {
│       │       "type": "object",
│       │       "properties": {
│       │         "location": {"type": "string"},
│       │         "units": {"type": "string", "enum": ["metric", "imperial"]}
│       │       }
│       │     }
│       │   }
│       ├── usage_stats.json
│       └── rate_limit_state.json
├── executions/
│   └── {date}/
│       └── {execution_id}/
│           ├── request.json
│           ├── response.json
│           └── trace.json
└── chains/
    └── {chain_id}/
        ├── definition.json
        └── execution_log.jsonl
```

### 6. Persona Module

```python
@dataclass
class Persona:
    """AI personality definition."""
    id: str
    name: str
    description: str
    traits: Dict[str, Any]
    voice: Dict[str, str]
    knowledge_areas: List[str]
    behavioral_rules: List[str]
    example_interactions: List[Dict[str, str]]
    
@dataclass
class PersonaState:
    """Active persona state."""
    persona_id: str
    session_id: SessionId
    activation_time: Timestamp
    adaptation_score: float
    consistency_scores: List[float]
    
@dataclass
class TraitInjection:
    """Persona trait application."""
    message_id: MessageId
    persona_id: str
    original_content: str
    modified_content: str
    traits_applied: List[str]
    strength: float
```

**Storage Structure:**
```
persona/
├── definitions/
│   └── {persona_id}/
│       ├── profile.json
│       │   {
│       │     "id": "persona_teacher",
│       │     "name": "Patient Teacher",
│       │     "description": "Supportive and educational",
│       │     "traits": {
│       │       "personality": ["patient", "encouraging", "clear"],
│       │       "communication": ["simple_language", "use_examples"]
│       │     },
│       │     "voice": {
│       │       "tone": "warm",
│       │       "formality": "casual"
│       │     }
│       │   }
│       ├── examples.jsonl
│       └── rules.json
├── active/
│   └── {session_id}/
│       └── persona_state.json
└── analytics/
    ├── usage_stats.json
    └── consistency_scores.jsonl
```

### 7. Multimodal Module

```python
@dataclass
class MediaFile:
    """Uploaded media file."""
    id: str
    type: ContentType
    format: str
    size_bytes: int
    duration_seconds: Optional[float]
    dimensions: Optional[Dict[str, int]]
    metadata: Dict[str, Any]
    
@dataclass
class ProcessingJob:
    """Media processing job."""
    id: str
    media_id: str
    operation: str
    parameters: Dict[str, Any]
    status: Literal["queued", "processing", "completed", "failed"]
    progress: float
    result: Optional[Dict[str, Any]]
    
@dataclass
class TranscriptionResult:
    """Audio/video transcription."""
    media_id: str
    text: str
    language: str
    confidence: float
    timestamps: List[Dict[str, float]]
    speaker_labels: Optional[List[str]]
```

**Storage Structure:**
```
multimodal/
├── uploads/
│   └── {job_id}/
│       ├── original.mp4
│       ├── metadata.json
│       │   {
│       │     "id": "media_123",
│       │     "type": "video",
│       │     "format": "mp4",
│       │     "size_bytes": 10485760,
│       │     "duration_seconds": 120.5,
│       │     "dimensions": {"width": 1920, "height": 1080}
│       │   }
│       ├── processed/
│       │   ├── thumbnail.jpg
│       │   └── transcription.json
│       └── analysis.json
├── generated/
│   └── {generation_id}/
│       ├── output.png
│       ├── prompt.txt
│       └── settings.json
└── cache/
    └── {content_hash}/
        └── cached_result.json
```

### 8. Topic Router Module

```python
@dataclass
class TopicClassification:
    """Topic classification result."""
    content_id: str
    topics: List[str]
    confidence_scores: Dict[str, float]
    primary_topic: str
    context_boundary: Optional[int]
    
@dataclass
class RoutingDecision:
    """Specialist routing decision."""
    content_id: str
    classifications: List[TopicClassification]
    selected_specialists: List[str]
    routing_weights: Dict[str, float]
    fallback_specialist: Optional[str]
    
@dataclass
class SpecialistRegistration:
    """Specialist agent registration."""
    specialist_id: str
    topics: List[str]
    expertise_level: Dict[str, float]
    availability: str
    performance_metrics: Dict[str, float]
```

**Storage Structure:**
```
topic_router/
├── classifiers/
│   └── {model_id}/
│       ├── model.bin
│       ├── vocabulary.json
│       └── topic_hierarchy.json
├── specialists/
│   └── {specialist_id}/
│       ├── registration.json
│       │   {
│       │     "specialist_id": "spec_finance_001",
│       │     "topics": ["finance", "investing", "budgeting"],
│       │     "expertise_level": {"finance": 0.95, "investing": 0.85},
│       │     "availability": "online",
│       │     "performance_metrics": {"accuracy": 0.92, "response_time_ms": 450}
│       │   }
│       └── routing_history.jsonl
├── routing/
│   └── {session_id}/
│       ├── classifications.jsonl
│       └── routing_decisions.jsonl
└── analytics/
    └── topic_distribution.json
```

### 9. Trace Logger Module

```python
@dataclass
class LogEntry:
    """Structured log entry."""
    id: str
    timestamp: Timestamp
    level: str
    module: str
    message: str
    data: Optional[Dict[str, Any]]
    trace_id: Optional[str]
    span_id: Optional[str]
    
@dataclass
class TraceSpan:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: Timestamp
    end_time: Optional[Timestamp]
    duration_ms: Optional[int]
    tags: Dict[str, Any]
    
@dataclass
class AuditEntry:
    """Audit trail entry."""
    id: str
    timestamp: Timestamp
    user_id: UserId
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    result: str
```

**Storage Structure:**
```
trace_logger/
├── logs/
│   └── {date}/
│       └── {hour}/
│           └── events.jsonl
│               {"id":"log_001","timestamp":"2024-01-15T10:30:00Z","level":"INFO","module":"text_chat","message":"Session started"}
├── traces/
│   └── {trace_id}/
│       ├── spans.jsonl
│       │   {"trace_id":"trace_123","span_id":"span_001","operation":"chat.process","duration_ms":125}
│       └── metadata.json
├── audit/
│   └── {date}/
│       ├── actions.jsonl
│       │   {"id":"audit_001","user_id":"user_123","action":"delete_message","resource":"msg_456","result":"success"}
│       └── summary.json
└── analytics/
    ├── performance_metrics.json
    └── error_summary.json
```

## Cross-Module Data Relationships

### 1. Session-Based Relationships
```
Session (Text Chat) ──┬── Messages (Text Chat)
                      ├── Memory Context (Memory)
                      ├── Active Persona (Persona)
                      └── Routing History (Topic Router)
```

### 2. User-Based Relationships
```
User ──┬── Sessions (Text Chat)
       ├── Long-term Memory (Memory)
       ├── Preferences (Memory)
       └── Audit Trail (Trace Logger)
```

### 3. Content-Based Relationships
```
Message ──┬── Attachments (Multimodal)
          ├── RAG Context (RAG)
          ├── Tool Results (Tool Use)
          └── Classifications (Topic Router)
```

## Data Lifecycle Management

### 1. Retention Policies
```json
{
  "retention_policies": {
    "messages": {
      "active_sessions": "indefinite",
      "inactive_sessions": "90_days",
      "archived_sessions": "1_year"
    },
    "memory": {
      "short_term": "session_lifetime",
      "long_term": "indefinite",
      "consolidation_logs": "30_days"
    },
    "logs": {
      "debug": "7_days",
      "info": "30_days",
      "error": "90_days",
      "audit": "7_years"
    }
  }
}
```

### 2. Data Migration Strategies
- **Incremental Migration**: Module by module
- **Parallel Running**: Old and new systems side by side
- **Batch Processing**: Historical data conversion
- **Live Sync**: Real-time data synchronization

### 3. Backup and Recovery
```json
{
  "backup_strategy": {
    "frequency": "daily",
    "retention": "30_days",
    "locations": ["primary_storage", "secondary_storage", "cold_storage"],
    "encryption": "AES-256",
    "verification": "checksum"
  }
}
```

## Performance Considerations

### 1. Indexing Strategies
- **Primary Keys**: Optimized for direct lookups
- **Secondary Indices**: For common query patterns
- **Full-text Search**: For content discovery
- **Vector Indices**: For similarity search

### 2. Caching Layers
```json
{
  "caching": {
    "memory_cache": {
      "size_mb": 512,
      "ttl_seconds": 300,
      "eviction": "LRU"
    },
    "disk_cache": {
      "size_gb": 10,
      "ttl_hours": 24,
      "compression": true
    }
  }
}
```

### 3. Data Compression
- **Messages**: JSONL with gzip compression
- **Embeddings**: Binary format with quantization
- **Logs**: Time-series compression
- **Media**: Format-specific optimization

## Security and Privacy

### 1. Encryption at Rest
```json
{
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_rotation": "quarterly",
    "scope": ["user_data", "messages", "memory"]
  }
}
```

### 2. Access Control
```json
{
  "access_control": {
    "user_data": "owner_only",
    "shared_sessions": "participant_list",
    "system_logs": "admin_only",
    "audit_trail": "compliance_team"
  }
}
```

### 3. Data Anonymization
- **PII Removal**: Automatic detection and masking
- **Aggregated Analytics**: No individual tracking
- **Consent Management**: User control over data usage

## Conclusion

This data model provides a comprehensive foundation for the modular chat platform, with clear separation between modules while maintaining consistency through shared core types. The storage schemas are optimized for each module's specific needs while supporting cross-module integration where necessary.