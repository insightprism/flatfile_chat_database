"""
FF Trace Logger Component - Advanced Logging and Conversation Tracing

Provides comprehensive logging, tracing, and analytics capabilities for the FF Chat System,
using FF logging infrastructure for advanced conversation analysis and monitoring.
"""

import asyncio
import time
import json
import traceback
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_trace_logger_config import FFTraceLoggerConfigDTO, FFTraceLevel, FFStorageBackend, FFTraceFormat
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_protocols.ff_message_dto import FFMessageDTO
from ff_managers.ff_storage_manager import FFStorageManager
from ff_managers.ff_document_manager import FFDocumentManager

logger = get_logger(__name__)


class FFTraceEventType(Enum):
    """Types of trace events"""
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    COMPONENT_INVOKED = "component_invoked"
    COMPONENT_COMPLETED = "component_completed"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"


class FFTraceStatus(Enum):
    """Status of trace events"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FFTraceEvent:
    """Individual trace event"""
    trace_id: str
    event_id: str
    event_type: FFTraceEventType
    timestamp: datetime
    session_id: str
    user_id: str
    component: str
    status: FFTraceStatus = FFTraceStatus.STARTED
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error_info: Optional[Dict[str, Any]] = None
    parent_event_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFTraceSpan:
    """Trace span for tracking operations"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: FFTraceStatus = FFTraceStatus.STARTED
    parent_span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFConversationTrace:
    """Complete conversation trace"""
    trace_id: str
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[FFTraceEvent] = field(default_factory=list)
    spans: List[FFTraceSpan] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFPerformanceMetrics:
    """Performance metrics collection"""
    response_time_ms: List[float] = field(default_factory=list)
    component_timing: Dict[str, List[float]] = field(default_factory=dict)
    error_rates: Dict[str, int] = field(default_factory=dict)
    throughput_per_minute: float = 0.0
    memory_usage_mb: List[float] = field(default_factory=list)
    concurrent_sessions: int = 0


class FFTraceLoggerComponent(FFChatComponentProtocol):
    """
    FF Trace Logger Component for advanced conversation tracing and logging.
    
    Provides comprehensive tracing, performance monitoring, and analytics
    for all chat system interactions and components.
    """
    
    def __init__(self, 
                 config: FFTraceLoggerConfigDTO,
                 storage_manager: FFStorageManager,
                 document_manager: FFDocumentManager):
        """
        Initialize FF Trace Logger Component.
        
        Args:
            config: Trace logger configuration
            storage_manager: FF storage manager for trace persistence
            document_manager: FF document manager for trace documents
        """
        self.config = config
        self.storage_manager = storage_manager
        self.document_manager = document_manager
        self.logger = get_logger(__name__)
        
        # Trace storage
        self.active_traces: Dict[str, FFConversationTrace] = {}
        self.active_spans: Dict[str, FFTraceSpan] = {}
        self.event_buffer: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_metrics = FFPerformanceMetrics()
        self.metrics_history: List[Tuple[datetime, FFPerformanceMetrics]] = []
        
        # Event handlers
        self.event_handlers: Dict[FFTraceEventType, List[Callable]] = defaultdict(list)
        
        # Background processing
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()
        
        # Analytics
        self.conversation_analytics: Dict[str, Any] = {
            "total_conversations": 0,
            "average_duration_minutes": 0.0,
            "message_counts": defaultdict(int),
            "error_patterns": defaultdict(int),
            "component_usage": defaultdict(int)
        }
        
        # Initialize background processing
        self._start_background_processing()
    
    async def process_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message: FFMessageDTO, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message and create trace events.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: Message to trace
            context: Additional context information
            
        Returns:
            Dict containing trace information and logging results
        """
        start_time = time.time()
        
        try:
            # Generate trace ID if not in context
            trace_id = (context or {}).get("trace_id", str(uuid.uuid4()))
            
            # Ensure trace exists
            await self._ensure_trace_exists(trace_id, session_id, user_id)
            
            # Create message received event
            event = await self._create_trace_event(
                trace_id=trace_id,
                event_type=FFTraceEventType.MESSAGE_RECEIVED,
                session_id=session_id,
                user_id=user_id,
                component="trace_logger",
                message=f"Message received: {getattr(message, 'content', str(message))[:100]}...",
                data={
                    "message_type": type(message).__name__,
                    "message_length": len(str(message)),
                    "context_keys": list(context.keys()) if context else []
                }
            )
            
            # Start span for message processing
            span = await self._start_span(
                trace_id=trace_id,
                operation_name="message_processing",
                tags={"component": "trace_logger", "session_id": session_id}
            )
            
            # Process tracing based on configuration
            trace_results = await self._process_message_tracing(
                trace_id, session_id, user_id, message, context
            )
            
            # Update performance metrics
            self._update_performance_metrics(start_time, "message_processing")
            
            # Complete span
            await self._finish_span(span.span_id, FFTraceStatus.COMPLETED)
            
            # Complete event
            await self._complete_trace_event(event.event_id, FFTraceStatus.COMPLETED)
            
            return {
                "success": True,
                "trace_id": trace_id,
                "event_id": event.event_id,
                "span_id": span.span_id,
                "trace_results": trace_results,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Trace logging failed: {e}")
            
            # Create error event
            try:
                await self._create_error_event(
                    trace_id=trace_id,
                    session_id=session_id,
                    user_id=user_id,
                    error=e,
                    context={"operation": "process_message"}
                )
            except Exception:
                pass  # Don't fail if error logging fails
            
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _ensure_trace_exists(self, trace_id: str, session_id: str, user_id: str) -> FFConversationTrace:
        """Ensure a conversation trace exists"""
        if trace_id not in self.active_traces:
            trace = FFConversationTrace(
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now()
            )
            self.active_traces[trace_id] = trace
            
            # Create conversation start event
            await self._create_trace_event(
                trace_id=trace_id,
                event_type=FFTraceEventType.CONVERSATION_START,
                session_id=session_id,
                user_id=user_id,
                component="trace_logger",
                message="Conversation trace started"
            )
            
            self.conversation_analytics["total_conversations"] += 1
        
        return self.active_traces[trace_id]
    
    async def _create_trace_event(self, 
                                trace_id: str,
                                event_type: FFTraceEventType,
                                session_id: str,
                                user_id: str,
                                component: str,
                                message: str = "",
                                data: Optional[Dict[str, Any]] = None,
                                parent_event_id: Optional[str] = None,
                                tags: Optional[List[str]] = None) -> FFTraceEvent:
        """Create a new trace event"""
        event = FFTraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            component=component,
            message=message,
            data=data or {},
            parent_event_id=parent_event_id,
            tags=tags or [],
            metadata={"created_by": "ff_trace_logger"}
        )
        
        # Add to trace
        if trace_id in self.active_traces:
            self.active_traces[trace_id].events.append(event)
        
        # Add to buffer for background processing
        self.event_buffer.append(event)
        
        # Update analytics
        self.conversation_analytics["component_usage"][component] += 1
        
        self.logger.debug(f"Created trace event {event.event_id} for trace {trace_id}")
        return event
    
    async def _start_span(self, 
                         trace_id: str,
                         operation_name: str,
                         parent_span_id: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None,
                         context: Optional[Dict[str, Any]] = None) -> FFTraceSpan:
        """Start a new trace span"""
        span = FFTraceSpan(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            parent_span_id=parent_span_id,
            tags=tags or {},
            context=context or {}
        )
        
        # Register active span
        self.active_spans[span.span_id] = span
        
        # Add to trace
        if trace_id in self.active_traces:
            self.active_traces[trace_id].spans.append(span)
        
        self.logger.debug(f"Started span {span.span_id} for operation {operation_name}")
        return span
    
    async def _finish_span(self, span_id: str, status: FFTraceStatus, error: Optional[Exception] = None) -> None:
        """Finish a trace span"""
        if span_id not in self.active_spans:
            self.logger.warning(f"Span {span_id} not found in active spans")
            return
        
        span = self.active_spans[span_id]
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if error:
            span.logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "error",
                "message": str(error),
                "error_type": type(error).__name__
            })
        
        # Remove from active spans
        del self.active_spans[span_id]
        
        # Update performance metrics
        if span.operation_name not in self.performance_metrics.component_timing:
            self.performance_metrics.component_timing[span.operation_name] = []
        self.performance_metrics.component_timing[span.operation_name].append(span.duration_ms)
        
        self.logger.debug(f"Finished span {span_id} in {span.duration_ms:.2f}ms with status {status}")
    
    async def _complete_trace_event(self, event_id: str, status: FFTraceStatus, error: Optional[Exception] = None) -> None:
        """Complete a trace event"""
        # Find event in active traces
        for trace in self.active_traces.values():
            for event in trace.events:
                if event.event_id == event_id:
                    event.status = status
                    if event.duration_ms is None:
                        event.duration_ms = (datetime.now() - event.timestamp).total_seconds() * 1000
                    
                    if error:
                        event.error_info = {
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "traceback": traceback.format_exc()
                        }
                        # Update error counts
                        trace.error_count += 1
                        self.performance_metrics.error_rates[event.component] = \
                            self.performance_metrics.error_rates.get(event.component, 0) + 1
                    
                    break
    
    async def _create_error_event(self, 
                                trace_id: str,
                                session_id: str,
                                user_id: str,
                                error: Exception,
                                context: Optional[Dict[str, Any]] = None) -> FFTraceEvent:
        """Create an error trace event"""
        return await self._create_trace_event(
            trace_id=trace_id,
            event_type=FFTraceEventType.ERROR_OCCURRED,
            session_id=session_id,
            user_id=user_id,
            component="system",
            message=f"Error: {str(error)}",
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context or {}
            },
            tags=["error", "exception"]
        )
    
    async def _process_message_tracing(self, 
                                     trace_id: str,
                                     session_id: str,
                                     user_id: str,
                                     message: FFMessageDTO,
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process detailed message tracing"""
        results = {}
        
        try:
            # Content analysis tracing
            if self.config.enable_content_analysis:
                content_span = await self._start_span(
                    trace_id=trace_id,
                    operation_name="content_analysis",
                    tags={"type": "analysis"}
                )
                
                content_analysis = await self._analyze_message_content(message, context)
                results["content_analysis"] = content_analysis
                
                await self._finish_span(content_span.span_id, FFTraceStatus.COMPLETED)
            
            # Performance tracing
            if self.config.enable_performance_tracing:
                perf_span = await self._start_span(
                    trace_id=trace_id,
                    operation_name="performance_tracking",
                    tags={"type": "performance"}
                )
                
                performance_data = await self._collect_performance_data()
                results["performance_data"] = performance_data
                
                await self._finish_span(perf_span.span_id, FFTraceStatus.COMPLETED)
            
            # Context tracing
            if context and self.config.trace_context:
                context_span = await self._start_span(
                    trace_id=trace_id,
                    operation_name="context_analysis",
                    tags={"type": "context"}
                )
                
                context_analysis = await self._analyze_context(context)
                results["context_analysis"] = context_analysis
                
                await self._finish_span(context_span.span_id, FFTraceStatus.COMPLETED)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Message tracing processing failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_message_content(self, message: FFMessageDTO, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze message content for tracing"""
        try:
            content = getattr(message, 'content', str(message))
            
            analysis = {
                "content_length": len(content),
                "word_count": len(content.split()),
                "character_count": len(content),
                "has_special_chars": bool([c for c in content if not c.isalnum() and not c.isspace()]),
                "language_detected": "en",  # Simplified - would use actual language detection
                "sentiment": "neutral",  # Simplified - would use actual sentiment analysis
                "topics": [],  # Would integrate with topic detection
                "complexity_score": min(1.0, len(content) / 1000),  # Simple complexity metric
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for patterns
            patterns = {
                "question": content.strip().endswith('?'),
                "exclamation": content.strip().endswith('!'),
                "command": content.strip().startswith('/'),
                "code": '```' in content or 'code' in content.lower(),
                "url": 'http' in content.lower(),
                "email": '@' in content and '.' in content
            }
            analysis["patterns"] = patterns
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {"error": str(e)}
    
    async def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect current performance data"""
        try:
            import psutil
            
            # System metrics
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections()),
                "active_processes": len(psutil.pids())
            }
            
            # Application metrics
            app_metrics = {
                "active_traces": len(self.active_traces),
                "active_spans": len(self.active_spans),
                "event_buffer_size": len(self.event_buffer),
                "total_conversations": self.conversation_analytics["total_conversations"],
                "avg_response_time": (
                    sum(self.performance_metrics.response_time_ms) / 
                    len(self.performance_metrics.response_time_ms)
                    if self.performance_metrics.response_time_ms else 0.0
                )
            }
            
            return {
                "system_metrics": system_metrics,
                "app_metrics": app_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance data collection failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context information"""
        try:
            analysis = {
                "context_size": len(context),
                "keys": list(context.keys()),
                "data_types": {k: type(v).__name__ for k, v in context.items()},
                "nested_objects": sum(1 for v in context.values() if isinstance(v, (dict, list))),
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for sensitive data patterns (simplified)
            sensitive_patterns = ["password", "token", "key", "secret", "auth"]
            sensitive_keys = [k for k in context.keys() if any(pattern in k.lower() for pattern in sensitive_patterns)]
            if sensitive_keys:
                analysis["potential_sensitive_data"] = sensitive_keys
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return {"error": str(e)}
    
    def _update_performance_metrics(self, start_time: float, operation: str) -> None:
        """Update performance metrics"""
        duration_ms = (time.time() - start_time) * 1000
        self.performance_metrics.response_time_ms.append(duration_ms)
        
        # Keep only recent metrics (last 1000)
        if len(self.performance_metrics.response_time_ms) > 1000:
            self.performance_metrics.response_time_ms = self.performance_metrics.response_time_ms[-1000:]
        
        # Update component timing
        if operation not in self.performance_metrics.component_timing:
            self.performance_metrics.component_timing[operation] = []
        self.performance_metrics.component_timing[operation].append(duration_ms)
        
        # Keep component timing manageable
        for component_times in self.performance_metrics.component_timing.values():
            if len(component_times) > 500:
                component_times[:] = component_times[-500:]
    
    def _start_background_processing(self) -> None:
        """Start background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._background_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            self.logger.info("Started background trace processing")
    
    def _background_processing_loop(self) -> None:
        """Background processing loop for trace data"""
        while not self.stop_processing.is_set():
            try:
                # Process buffered events
                self._process_event_buffer()
                
                # Clean up old traces
                self._cleanup_old_traces()
                
                # Update analytics
                self._update_analytics()
                
                # Store traces periodically
                if self.config.storage_backend != FFStorageBackend.MEMORY_ONLY:
                    asyncio.run(self._store_traces_periodically())
                
                # Sleep before next iteration
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                time.sleep(5)
    
    def _process_event_buffer(self) -> None:
        """Process events in the buffer"""
        try:
            # Process events based on configuration
            events_to_process = []
            
            # Drain buffer up to batch size
            batch_size = min(100, len(self.event_buffer))
            for _ in range(batch_size):
                if self.event_buffer:
                    events_to_process.append(self.event_buffer.popleft())
            
            if events_to_process:
                # Apply filters and transformations
                processed_events = self._filter_and_transform_events(events_to_process)
                
                # Update metrics
                for event in processed_events:
                    if event.event_type == FFTraceEventType.ERROR_OCCURRED:
                        pattern = f"{event.component}:{event.data.get('error_type', 'unknown')}"
                        self.conversation_analytics["error_patterns"][pattern] += 1
                
        except Exception as e:
            self.logger.error(f"Event buffer processing failed: {e}")
    
    def _filter_and_transform_events(self, events: List[FFTraceEvent]) -> List[FFTraceEvent]:
        """Filter and transform events based on configuration"""
        filtered_events = []
        
        for event in events:
            # Apply trace level filtering
            if self._should_include_event(event):
                # Apply transformations
                transformed_event = self._transform_event(event)
                filtered_events.append(transformed_event)
        
        return filtered_events
    
    def _should_include_event(self, event: FFTraceEvent) -> bool:
        """Check if event should be included based on trace level"""
        trace_level = self.config.trace_level
        
        if trace_level == FFTraceLevel.OFF:
            return False
        elif trace_level == FFTraceLevel.ERROR:
            return event.event_type == FFTraceEventType.ERROR_OCCURRED
        elif trace_level == FFTraceLevel.WARN:
            return event.event_type in [FFTraceEventType.ERROR_OCCURRED, FFTraceEventType.SYSTEM_EVENT]
        elif trace_level == FFTraceLevel.INFO:
            return event.event_type not in [FFTraceEventType.PERFORMANCE_METRIC]
        elif trace_level == FFTraceLevel.DEBUG:
            return True
        elif trace_level == FFTraceLevel.TRACE:
            return True
        
        return True
    
    def _transform_event(self, event: FFTraceEvent) -> FFTraceEvent:
        """Transform event based on configuration"""
        # Apply data scrubbing for sensitive information
        if self.config.scrub_sensitive_data:
            event = self._scrub_sensitive_data(event)
        
        # Apply format transformations
        if self.config.trace_format == FFTraceFormat.STRUCTURED:
            # Event is already structured
            pass
        elif self.config.trace_format == FFTraceFormat.JSON:
            # Convert to JSON-serializable format
            event.data = self._ensure_json_serializable(event.data)
        
        return event
    
    def _scrub_sensitive_data(self, event: FFTraceEvent) -> FFTraceEvent:
        """Remove sensitive data from event"""
        sensitive_patterns = ["password", "token", "key", "secret", "auth", "credential"]
        
        # Scrub data dictionary
        scrubbed_data = {}
        for key, value in event.data.items():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                scrubbed_data[key] = "[SCRUBBED]"
            elif isinstance(value, str) and len(value) > 50:
                # Check if it might be a token or key
                if any(pattern in value.lower() for pattern in ["bearer", "jwt", "token"]):
                    scrubbed_data[key] = "[SCRUBBED]"
                else:
                    scrubbed_data[key] = value
            else:
                scrubbed_data[key] = value
        
        event.data = scrubbed_data
        
        # Scrub message if it contains sensitive patterns
        if any(pattern in event.message.lower() for pattern in sensitive_patterns):
            event.message = "[Message contains sensitive data - scrubbed]"
        
        return event
    
    def _ensure_json_serializable(self, data: Any) -> Any:
        """Ensure data is JSON serializable"""
        if isinstance(data, dict):
            return {k: self._ensure_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return str(data)
    
    def _cleanup_old_traces(self) -> None:
        """Clean up old traces to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.trace_retention_hours)
        
        traces_to_remove = []
        for trace_id, trace in self.active_traces.items():
            if trace.start_time < cutoff_time:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            # Finalize trace before removal
            trace = self.active_traces[trace_id]
            if trace.end_time is None:
                trace.end_time = datetime.now()
                trace.status = "completed"
            
            del self.active_traces[trace_id]
        
        if traces_to_remove:
            self.logger.info(f"Cleaned up {len(traces_to_remove)} old traces")
    
    def _update_analytics(self) -> None:
        """Update conversation analytics"""
        try:
            # Calculate average conversation duration
            completed_traces = [
                trace for trace in self.active_traces.values() 
                if trace.end_time is not None
            ]
            
            if completed_traces:
                total_duration = sum(
                    (trace.end_time - trace.start_time).total_seconds() / 60
                    for trace in completed_traces
                )
                self.conversation_analytics["average_duration_minutes"] = \
                    total_duration / len(completed_traces)
            
            # Update message counts
            for trace in self.active_traces.values():
                message_events = [
                    event for event in trace.events 
                    if event.event_type in [FFTraceEventType.MESSAGE_RECEIVED, FFTraceEventType.MESSAGE_PROCESSED]
                ]
                self.conversation_analytics["message_counts"][trace.session_id] = len(message_events)
            
        except Exception as e:
            self.logger.error(f"Analytics update failed: {e}")
    
    async def _store_traces_periodically(self) -> None:
        """Store traces periodically based on configuration"""
        try:
            if self.config.storage_backend == FFStorageBackend.FF_STORAGE:
                await self._store_traces_to_ff_storage()
            elif self.config.storage_backend == FFStorageBackend.FILE_SYSTEM:
                await self._store_traces_to_filesystem()
            elif self.config.storage_backend == FFStorageBackend.DATABASE:
                await self._store_traces_to_database()
            
        except Exception as e:
            self.logger.error(f"Periodic trace storage failed: {e}")
    
    async def _store_traces_to_ff_storage(self) -> None:
        """Store traces to FF storage manager"""
        try:
            for trace_id, trace in list(self.active_traces.items()):
                if trace.end_time is not None:  # Only store completed traces
                    trace_data = asdict(trace)
                    
                    # Store as document
                    document_id = f"trace_{trace_id}"
                    await self.document_manager.store_document(
                        document_id=document_id,
                        content=json.dumps(trace_data),
                        metadata={
                            "type": "conversation_trace",
                            "session_id": trace.session_id,
                            "user_id": trace.user_id,
                            "start_time": trace.start_time.isoformat(),
                            "event_count": len(trace.events),
                            "span_count": len(trace.spans)
                        }
                    )
                    
                    self.logger.debug(f"Stored trace {trace_id} to FF storage")
            
        except Exception as e:
            self.logger.error(f"FF storage trace storage failed: {e}")
    
    async def _store_traces_to_filesystem(self) -> None:
        """Store traces to filesystem"""
        try:
            import os
            
            traces_dir = self.config.export.output_directory or "traces"
            os.makedirs(traces_dir, exist_ok=True)
            
            for trace_id, trace in list(self.active_traces.items()):
                if trace.end_time is not None:
                    trace_file = os.path.join(traces_dir, f"{trace_id}.json")
                    trace_data = asdict(trace)
                    
                    with open(trace_file, 'w') as f:
                        json.dump(trace_data, f, indent=2, default=str)
                    
                    self.logger.debug(f"Stored trace {trace_id} to filesystem")
            
        except Exception as e:
            self.logger.error(f"Filesystem trace storage failed: {e}")
    
    async def _store_traces_to_database(self) -> None:
        """Store traces to database (would integrate with actual database)"""
        # This would integrate with a database system
        # For now, we'll just log that we would store to database
        completed_traces = [
            trace for trace in self.active_traces.values() 
            if trace.end_time is not None
        ]
        
        if completed_traces:
            self.logger.info(f"Would store {len(completed_traces)} traces to database")
    
    # Public API methods
    
    async def start_conversation_trace(self, session_id: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new conversation trace"""
        trace_id = str(uuid.uuid4())
        trace = await self._ensure_trace_exists(trace_id, session_id, user_id)
        
        if metadata:
            trace.metadata.update(metadata)
        
        self.logger.info(f"Started conversation trace {trace_id} for session {session_id}")
        return trace_id
    
    async def end_conversation_trace(self, trace_id: str) -> bool:
        """End a conversation trace"""
        try:
            if trace_id in self.active_traces:
                trace = self.active_traces[trace_id]
                trace.end_time = datetime.now()
                trace.status = "completed"
                
                # Create conversation end event
                await self._create_trace_event(
                    trace_id=trace_id,
                    event_type=FFTraceEventType.CONVERSATION_END,
                    session_id=trace.session_id,
                    user_id=trace.user_id,
                    component="trace_logger",
                    message="Conversation trace ended",
                    data={
                        "duration_minutes": (trace.end_time - trace.start_time).total_seconds() / 60,
                        "event_count": len(trace.events),
                        "span_count": len(trace.spans),
                        "error_count": trace.error_count
                    }
                )
                
                self.logger.info(f"Ended conversation trace {trace_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to end conversation trace {trace_id}: {e}")
            return False
    
    async def log_component_invocation(self, 
                                     trace_id: str,
                                     component_name: str,
                                     operation: str,
                                     session_id: str,
                                     user_id: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log component invocation"""
        event = await self._create_trace_event(
            trace_id=trace_id,
            event_type=FFTraceEventType.COMPONENT_INVOKED,
            session_id=session_id,
            user_id=user_id,
            component=component_name,
            message=f"Component {component_name} invoked for operation: {operation}",
            data={
                "operation": operation,
                "metadata": metadata or {}
            },
            tags=["component", "invocation"]
        )
        
        return event.event_id
    
    async def log_performance_metric(self, 
                                   trace_id: str,
                                   metric_name: str,
                                   metric_value: Union[int, float],
                                   session_id: str,
                                   user_id: str,
                                   unit: str = "",
                                   tags: Optional[List[str]] = None) -> str:
        """Log performance metric"""
        event = await self._create_trace_event(
            trace_id=trace_id,
            event_type=FFTraceEventType.PERFORMANCE_METRIC,
            session_id=session_id,
            user_id=user_id,
            component="performance_monitor",
            message=f"Performance metric: {metric_name} = {metric_value} {unit}",
            data={
                "metric_name": metric_name,
                "metric_value": metric_value,
                "unit": unit,
                "timestamp": datetime.now().isoformat()
            },
            tags=(tags or []) + ["performance", "metric"]
        )
        
        return event.event_id
    
    def get_trace_info(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a trace"""
        if trace_id not in self.active_traces:
            return None
        
        trace = self.active_traces[trace_id]
        return {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "user_id": trace.user_id,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "status": trace.status,
            "event_count": len(trace.events),
            "span_count": len(trace.spans),
            "error_count": trace.error_count,
            "warning_count": trace.warning_count,
            "metadata": trace.metadata
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics"""
        return {
            "conversation_analytics": self.conversation_analytics.copy(),
            "performance_metrics": {
                "avg_response_time_ms": (
                    sum(self.performance_metrics.response_time_ms) / 
                    len(self.performance_metrics.response_time_ms)
                    if self.performance_metrics.response_time_ms else 0.0
                ),
                "component_timing": {
                    component: {
                        "avg_ms": sum(times) / len(times) if times else 0.0,
                        "count": len(times)
                    }
                    for component, times in self.performance_metrics.component_timing.items()
                },
                "error_rates": self.performance_metrics.error_rates.copy()
            },
            "system_stats": {
                "active_traces": len(self.active_traces),
                "active_spans": len(self.active_spans),
                "event_buffer_size": len(self.event_buffer)
            }
        }
    
    def add_event_handler(self, event_type: FFTraceEventType, handler: Callable) -> None:
        """Add event handler for specific event types"""
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Added event handler for {event_type}")
    
    def remove_event_handler(self, event_type: FFTraceEventType, handler: Callable) -> bool:
        """Remove event handler"""
        try:
            self.event_handlers[event_type].remove(handler)
            return True
        except ValueError:
            return False
    
    async def export_traces(self, 
                          output_format: FFTraceFormat = FFTraceFormat.JSON,
                          filter_criteria: Optional[Dict[str, Any]] = None) -> str:
        """Export traces in specified format"""
        try:
            # Filter traces based on criteria
            traces_to_export = []
            for trace in self.active_traces.values():
                if self._matches_filter_criteria(trace, filter_criteria):
                    traces_to_export.append(trace)
            
            # Convert to specified format
            if output_format == FFTraceFormat.JSON:
                export_data = [asdict(trace) for trace in traces_to_export]
                return json.dumps(export_data, indent=2, default=str)
            
            elif output_format == FFTraceFormat.CSV:
                # Simplified CSV export
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['trace_id', 'session_id', 'user_id', 'start_time', 'end_time', 'event_count', 'error_count'])
                
                # Write data
                for trace in traces_to_export:
                    writer.writerow([
                        trace.trace_id,
                        trace.session_id,
                        trace.user_id,
                        trace.start_time.isoformat(),
                        trace.end_time.isoformat() if trace.end_time else '',
                        len(trace.events),
                        trace.error_count
                    ])
                
                return output.getvalue()
            
            else:
                return json.dumps([asdict(trace) for trace in traces_to_export], indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Trace export failed: {e}")
            return f"Export failed: {str(e)}"
    
    def _matches_filter_criteria(self, trace: FFConversationTrace, criteria: Optional[Dict[str, Any]]) -> bool:
        """Check if trace matches filter criteria"""
        if not criteria:
            return True
        
        for key, value in criteria.items():
            if key == "session_id" and trace.session_id != value:
                return False
            elif key == "user_id" and trace.user_id != value:
                return False
            elif key == "start_after" and trace.start_time < datetime.fromisoformat(value):
                return False
            elif key == "start_before" and trace.start_time > datetime.fromisoformat(value):
                return False
            elif key == "has_errors" and (trace.error_count > 0) != value:
                return False
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown the trace logger component"""
        self.logger.info("Shutting down FF Trace Logger Component")
        
        # Stop background processing
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Finalize active traces
        for trace in self.active_traces.values():
            if trace.end_time is None:
                trace.end_time = datetime.now()
                trace.status = "shutdown"
        
        self.logger.info("FF Trace Logger Component shutdown complete")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "component_name": "FF Trace Logger Component",
            "version": "1.0.0",
            "config": {
                "trace_level": self.config.trace_level.value,
                "storage_backend": self.config.storage_backend.value,
                "enable_performance_tracing": self.config.enable_performance_tracing,
                "enable_content_analysis": self.config.enable_content_analysis,
                "trace_retention_hours": self.config.trace_retention_hours,
                "scrub_sensitive_data": self.config.scrub_sensitive_data
            },
            "status": "active" if not self.stop_processing.is_set() else "shutting_down",
            "analytics": self.get_analytics()
        }