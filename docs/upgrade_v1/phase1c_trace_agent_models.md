# Phase 1c: Trace and Agent Models

## Overview
This sub-phase implements the trace logging system for debugging/auditing and the agent configuration models for multi-agent support. These models enable comprehensive system observability and flexible agent management.

## Objectives
1. Create trace system entities for debugging and audit logging
2. Implement agent configuration models for multi-agent orchestration
3. Define routing and execution tracking structures
4. Ensure proper serialization and type safety

## Prerequisites
- Understanding of distributed tracing concepts
- Familiarity with multi-agent systems
- No dependencies on Phase 1a or 1b (can be developed in parallel)

## Implementation Files

### 1. Trace System Entities (`ff_class_configs/ff_trace_entities.py`)

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        return cls(**data)


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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceSession':
        data = data.copy()
        if 'spans' in data:
            data['spans'] = [TraceSpan.from_dict(s) for s in data['spans']]
        if 'events' in data:
            data['events'] = [TraceEvent.from_dict(e) for e in data['events']]
        if 'metrics' in data:
            data['metrics'] = [PerformanceMetric.from_dict(m) for m in data['metrics']]
        return cls(**data)


@dataclass
class TraceExportFormat:
    """Configuration for trace export"""
    format: str = "json"  # json, csv, otlp
    include_spans: bool = True
    include_events: bool = True
    include_metrics: bool = True
    time_range: Optional[Dict[str, str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
```

### 2. Agent System Entities (`ff_class_configs/ff_agent_entities.py`)

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRoutingDecision':
        return cls(**data)


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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentExecutionContext':
        return cls(**data)


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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        return cls(**data)


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
    
    def add_response(self, response: AgentResponse):
        """Add agent response"""
        self.responses.append(response)
    
    def check_consensus(self) -> bool:
        """Check if consensus is reached"""
        if len(self.responses) < len(self.target_agent_ids):
            return False
        
        # Simple consensus check - can be made more sophisticated
        agreements = sum(1 for r in self.responses if r.confidence >= self.consensus_threshold)
        self.consensus_reached = agreements / len(self.responses) >= self.consensus_threshold
        return self.consensus_reached
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['responses'] = [r.to_dict() for r in self.responses]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCollaborationRequest':
        data = data.copy()
        if 'responses' in data:
            data['responses'] = [AgentResponse.from_dict(r) for r in data['responses']]
        return cls(**data)


@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    agent_id: str
    period_start: str
    period_end: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Token usage
    total_tokens_used: int = 0
    average_tokens_per_request: float = 0.0
    
    # Quality metrics
    average_confidence: float = 0.0
    routing_accuracy: float = 0.0
    
    # Resource usage
    peak_concurrent_requests: int = 0
    memory_usage_mb: float = 0.0
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetrics':
        return cls(**data)
```

### 3. Trace and Agent Tests (`tests/test_trace_agent_v2.py`)

Create comprehensive tests for trace and agent models:

```python
"""
Tests for trace and agent system models.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from ff_class_configs.ff_trace_entities import (
    TraceEvent, TraceSpan, TraceLevel, TraceEventType,
    PerformanceMetric, TraceContext, TraceSession
)
from ff_class_configs.ff_agent_entities import (
    AgentConfiguration, AgentType, AgentCapability,
    AgentPromptTemplate, AgentRoutingDecision,
    AgentExecutionContext, AgentResponse,
    AgentCollaborationRequest
)


class TestTraceModels:
    """Test trace system models"""
    
    def test_trace_event_creation(self):
        """Test creating a basic trace event"""
        event = TraceEvent(
            event_type=TraceEventType.MESSAGE_RECEIVED,
            level=TraceLevel.INFO,
            component="message_handler",
            operation="receive_message",
            session_id="session_123",
            user_id="user_123",
            message_id="msg_456",
            data={"message_type": "text", "length": 150}
        )
        
        assert event.id.startswith("evt_")
        assert event.event_type == TraceEventType.MESSAGE_RECEIVED
        assert event.level == TraceLevel.INFO
        assert event.data["message_type"] == "text"
    
    def test_trace_event_error(self):
        """Test creating an error trace event"""
        event = TraceEvent(
            event_type=TraceEventType.MESSAGE_FAILED,
            level=TraceLevel.ERROR,
            component="message_processor",
            operation="process_message",
            error="Invalid message format",
            stack_trace="Traceback...",
            data={"message_id": "msg_789"}
        )
        
        assert event.level == TraceLevel.ERROR
        assert event.error == "Invalid message format"
        assert event.stack_trace is not None
    
    def test_trace_span_lifecycle(self):
        """Test trace span creation and completion"""
        span = TraceSpan(
            name="process_request",
            session_id="session_123",
            user_id="user_123"
        )
        
        # Add events
        span.add_event("evt_001")
        span.add_event("evt_002")
        
        # Complete span
        span.complete()
        
        assert span.status == "completed"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert len(span.events) == 2
        
        # Test failed span
        failed_span = TraceSpan(name="failed_operation")
        failed_span.complete(error="Operation failed")
        
        assert failed_span.status == "failed"
        assert failed_span.error == "Operation failed"
    
    def test_performance_metric(self):
        """Test performance metric creation"""
        metric = PerformanceMetric(
            name="database_query_time",
            value=45.3,
            unit="ms",
            component="storage_manager",
            operation="get_messages",
            tags={"query_type": "range", "index_used": "true"}
        )
        
        assert metric.name == "database_query_time"
        assert metric.value == 45.3
        assert metric.unit == "ms"
        assert metric.tags["query_type"] == "range"
    
    def test_trace_context_propagation(self):
        """Test trace context creation and propagation"""
        # Create root context
        root_context = TraceContext()
        
        assert root_context.trace_id is not None
        assert root_context.span_id is not None
        assert root_context.parent_span_id is None
        
        # Create child context
        child_context = root_context.create_child_context()
        
        assert child_context.trace_id == root_context.trace_id
        assert child_context.parent_span_id == root_context.span_id
        assert child_context.span_id != root_context.span_id
    
    def test_trace_session_summary(self):
        """Test trace session with summary generation"""
        session = TraceSession(
            session_id="session_123",
            start_time=datetime.now().isoformat()
        )
        
        # Add spans
        span1 = TraceSpan(name="operation_1")
        span1.duration_ms = 500
        span1.status = "completed"
        session.add_span(span1)
        
        span2 = TraceSpan(name="slow_operation")
        span2.duration_ms = 2500
        span2.status = "completed"
        session.add_span(span2)
        
        # Add events
        session.add_event(TraceEvent(
            event_type=TraceEventType.MESSAGE_RECEIVED,
            level=TraceLevel.INFO
        ))
        session.add_event(TraceEvent(
            event_type=TraceEventType.MESSAGE_PROCESSED,
            level=TraceLevel.INFO
        ))
        session.add_event(TraceEvent(
            event_type=TraceEventType.SLOW_OPERATION,
            level=TraceLevel.WARNING
        ))
        session.add_event(TraceEvent(
            event_type=TraceEventType.MESSAGE_FAILED,
            level=TraceLevel.ERROR
        ))
        
        # Generate summary
        session.generate_summary()
        
        assert session.summary["total_spans"] == 2
        assert session.summary["total_events"] == 4
        assert session.summary["error_count"] == 1
        assert session.summary["warning_count"] == 1
        assert session.summary["total_duration_ms"] == 3000
        assert len(session.summary["slowest_operations"]) == 1
        assert session.summary["slowest_operations"][0]["name"] == "slow_operation"
    
    def test_trace_serialization(self):
        """Test trace event serialization"""
        event = TraceEvent(
            event_type=TraceEventType.AGENT_INVOKED,
            level=TraceLevel.DEBUG,
            component="agent_manager",
            agent_id="agent_123",
            data={"request_id": "req_456"}
        )
        
        # Convert to dict
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "agent_invoked"
        assert event_dict["level"] == "debug"
        
        # Convert back
        restored = TraceEvent.from_dict(event_dict)
        assert restored.event_type == TraceEventType.AGENT_INVOKED
        assert restored.level == TraceLevel.DEBUG


class TestAgentModels:
    """Test agent system models"""
    
    def test_agent_configuration_creation(self):
        """Test creating agent configuration"""
        agent = AgentConfiguration(
            id="agent_123",
            name="Customer Support Agent",
            type=AgentType.TASK_ORIENTED,
            description="Handles customer inquiries",
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY_ACCESS
            ],
            knowledge_domains=["customer_service", "product_info"],
            expertise_keywords=["support", "help", "issue", "problem"],
            personality_traits={"helpful": 0.9, "patient": 0.8, "professional": 0.95}
        )
        
        assert agent.id == "agent_123"
        assert agent.type == AgentType.TASK_ORIENTED
        assert len(agent.capabilities) == 3
        assert AgentCapability.TOOL_USE in agent.capabilities
        assert agent.personality_traits["helpful"] == 0.9
    
    def test_agent_confidence_calculation(self):
        """Test agent confidence score calculation"""
        agent = AgentConfiguration(
            id="expert_agent",
            name="Finance Expert",
            type=AgentType.SPECIALIZED,
            capabilities=[AgentCapability.TEXT_GENERATION, AgentCapability.DATA_ANALYSIS],
            knowledge_domains=["finance", "accounting", "investing"],
            expertise_keywords=["budget", "roi", "investment", "profit", "loss"]
        )
        
        # Test high confidence request
        request = {
            "topic": "finance and accounting",
            "content": "What's the ROI on this investment?",
            "required_capabilities": ["text_generation", "data_analysis"]
        }
        
        confidence = agent.can_handle(request)
        assert confidence > 0.7  # Should be high confidence
        
        # Test low confidence request
        request2 = {
            "topic": "medical diagnosis",
            "content": "What are the symptoms of flu?",
            "required_capabilities": ["medical_knowledge"]
        }
        
        confidence2 = agent.can_handle(request2)
        assert confidence2 < 0.3  # Should be low confidence
    
    def test_agent_prompt_template(self):
        """Test agent prompt template formatting"""
        template = AgentPromptTemplate(
            system_prompt="You are a {role} assistant. Your expertise is in {domain}.",
            user_prompt_template="Please help with: {query}",
            examples=[
                {"user": "What's the weather?", "assistant": "I'll help you check the weather."}
            ],
            constraints=["Be concise", "Be helpful"],
            variables=["role", "domain", "query"]
        )
        
        formatted = template.format(
            role="customer support",
            domain="technical issues",
            query="My printer won't connect"
        )
        
        assert "customer support" in formatted["system"]
        assert "technical issues" in formatted["system"]
        assert "My printer won't connect" in formatted["user"]
    
    def test_agent_routing_decision(self):
        """Test agent routing decision"""
        decision = AgentRoutingDecision(
            agent_id="agent_123",
            confidence=0.85,
            reasoning="High keyword match and domain expertise",
            context_segments=[
                {"text": "budget planning", "relevance": 0.9},
                {"text": "financial forecast", "relevance": 0.8}
            ]
        )
        
        assert decision.confidence == 0.85
        assert len(decision.context_segments) == 2
        assert decision.context_segments[0]["relevance"] == 0.9
    
    def test_agent_execution_context(self):
        """Test agent execution context"""
        context = AgentExecutionContext(
            request_id="req_123",
            session_id="session_456",
            user_id="user_789",
            content="Help me analyze sales data",
            conversation_history=[
                {"role": "user", "content": "I need help with data"},
                {"role": "assistant", "content": "I can help analyze your data"}
            ],
            available_tools=["data_analyzer", "chart_generator"],
            max_response_tokens=1000
        )
        
        assert context.request_id == "req_123"
        assert len(context.conversation_history) == 2
        assert "data_analyzer" in context.available_tools
    
    def test_agent_response(self):
        """Test agent response creation"""
        response = AgentResponse(
            agent_id="agent_123",
            request_id="req_456",
            content="Here's the analysis of your sales data...",
            confidence=0.92,
            tokens_used=250,
            execution_time_ms=1500,
            tool_calls=[
                {"tool": "data_analyzer", "result": {"trend": "increasing"}}
            ],
            follow_up_questions=[
                "Would you like to see a breakdown by region?",
                "Should I compare this to last year's data?"
            ]
        )
        
        assert response.success is True
        assert response.confidence == 0.92
        assert len(response.tool_calls) == 1
        assert len(response.follow_up_questions) == 2
    
    def test_agent_collaboration_request(self):
        """Test multi-agent collaboration"""
        collab = AgentCollaborationRequest(
            id="collab_123",
            initiator_agent_id="orchestrator_1",
            target_agent_ids=["expert_1", "expert_2", "expert_3"],
            task_description="Analyze market trends and provide recommendations",
            task_type="consensus",
            consensus_threshold=0.8
        )
        
        # Add responses
        response1 = AgentResponse(
            agent_id="expert_1",
            request_id="collab_123",
            content="Market shows bullish trends",
            confidence=0.85
        )
        collab.add_response(response1)
        
        response2 = AgentResponse(
            agent_id="expert_2",
            request_id="collab_123",
            content="I agree with the bullish assessment",
            confidence=0.90
        )
        collab.add_response(response2)
        
        # Check consensus (not reached yet - need all agents)
        assert collab.check_consensus() is False
        
        response3 = AgentResponse(
            agent_id="expert_3",
            request_id="collab_123",
            content="Bullish trend confirmed",
            confidence=0.88
        )
        collab.add_response(response3)
        
        # Check consensus (should be reached now)
        assert collab.check_consensus() is True
        assert collab.consensus_reached is True
    
    def test_agent_serialization(self):
        """Test agent configuration serialization"""
        agent = AgentConfiguration(
            id="agent_serial",
            name="Test Agent",
            type=AgentType.CONVERSATIONAL,
            capabilities=[AgentCapability.TEXT_GENERATION],
            model_parameters={"top_p": 0.9, "frequency_penalty": 0.1}
        )
        
        # Convert to dict
        agent_dict = agent.to_dict()
        assert agent_dict["type"] == "conversational"
        assert agent_dict["capabilities"] == ["text_generation"]
        
        # Convert back
        restored = AgentConfiguration.from_dict(agent_dict)
        assert restored.id == agent.id
        assert restored.type == AgentType.CONVERSATIONAL
        assert AgentCapability.TEXT_GENERATION in restored.capabilities


class TestTraceAgentIntegration:
    """Test integration between trace and agent systems"""
    
    def test_agent_execution_tracing(self):
        """Test tracing agent execution"""
        # Create trace span for agent execution
        span = TraceSpan(
            name="agent_execution",
            tags={"agent_id": "agent_123", "request_id": "req_456"}
        )
        
        # Log agent invocation
        invoke_event = TraceEvent(
            event_type=TraceEventType.AGENT_INVOKED,
            component="agent_manager",
            agent_id="agent_123",
            data={"request_type": "analysis", "content_length": 500}
        )
        span.add_event(invoke_event.id)
        
        # Log tool usage
        tool_event = TraceEvent(
            event_type=TraceEventType.TOOL_CALLED,
            component="agent_123",
            data={"tool": "data_analyzer", "parameters": {"dataset": "sales_2024"}}
        )
        span.add_event(tool_event.id)
        
        # Log response
        response_event = TraceEvent(
            event_type=TraceEventType.AGENT_RESPONDED,
            component="agent_123",
            data={"tokens_used": 350, "confidence": 0.92}
        )
        span.add_event(response_event.id)
        
        # Complete span
        span.complete()
        
        assert span.status == "completed"
        assert len(span.events) == 3
        assert span.duration_ms is not None
    
    def test_multi_agent_collaboration_tracing(self):
        """Test tracing multi-agent collaboration"""
        # Create parent span
        parent_span = TraceSpan(
            name="multi_agent_collaboration",
            tags={"collaboration_id": "collab_123"}
        )
        
        # Create child spans for each agent
        agent_spans = []
        for agent_id in ["agent_1", "agent_2", "agent_3"]:
            agent_span = TraceSpan(
                name=f"agent_{agent_id}_execution",
                parent_span_id=parent_span.id,
                tags={"agent_id": agent_id}
            )
            agent_spans.append(agent_span)
            
            # Log agent work
            event = TraceEvent(
                event_type=TraceEventType.AGENT_INVOKED,
                component="collaboration_manager",
                agent_id=agent_id,
                parent_event_id=parent_span.id
            )
            agent_span.add_event(event.id)
        
        # Complete all spans
        for span in agent_spans:
            span.complete()
        parent_span.complete()
        
        assert all(s.status == "completed" for s in agent_spans)
        assert parent_span.status == "completed"
```

## Implementation Steps

1. **Create the trace entities file**
   ```bash
   touch ff_class_configs/ff_trace_entities.py
   # Copy the trace entity implementations
   ```

2. **Create the agent entities file**
   ```bash
   touch ff_class_configs/ff_agent_entities.py
   # Copy the agent entity implementations
   ```

3. **Create the test file**
   ```bash
   touch tests/test_trace_agent_v2.py
   # Copy the test implementations
   ```

4. **Run the tests**
   ```bash
   pytest tests/test_trace_agent_v2.py -v
   ```

## Validation Checklist

- [ ] Trace events support all event types and levels
- [ ] Trace spans track timing and relationships correctly
- [ ] Performance metrics capture system performance
- [ ] Trace context propagates correctly for distributed tracing
- [ ] Trace session generates accurate summaries
- [ ] Agent configurations support all agent types
- [ ] Agent confidence calculation working correctly
- [ ] Agent routing decisions track context segments
- [ ] Agent collaboration supports consensus checking
- [ ] All models serialize/deserialize correctly
- [ ] Enum types properly handled
- [ ] All tests passing

## Next Steps

Once Phase 1c is complete:
1. Move to Phase 1d (Protocols and Configuration)
2. All entity models are now available for Phase 2 managers
3. Can begin implementing storage managers in parallel

## Notes for Developers

1. **Trace IDs**: Use UUID hex format for unique identification
2. **Timing**: Always use ISO format timestamps
3. **Agent Confidence**: Score calculation can be customized per use case
4. **Collaboration**: Consensus checking is simplified, can be enhanced
5. **Performance**: Track duration in milliseconds for consistency

This completes Phase 1c implementation specification.