"""
FF Trace Logger Component Configuration

Configuration classes for the FF Trace Logger component that provides
advanced conversation logging, tracing, and analysis using existing FF logging system.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ff_class_configs.ff_base_config import FFBaseConfigDTO


class FFTraceLevel(Enum):
    """Trace logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FFTraceEvent(Enum):
    """Types of trace events"""
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    COMPONENT_CALLED = "component_called"
    COMPONENT_RESPONSE = "component_response"
    COMPONENT_ERROR = "component_error"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ROUTING_DECISION = "routing_decision"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_ACCESS = "memory_access"
    SEARCH_QUERY = "search_query"
    CONTEXT_UPDATE = "context_update"


class FFTraceFormat(Enum):
    """Output formats for trace data"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"


class FFTraceStorage(Enum):
    """Storage backends for trace data"""
    MEMORY = "memory"
    FF_STORAGE = "ff_storage"
    FILE = "file"
    DATABASE = "database"
    EXTERNAL = "external"


@dataclass
class FFTraceEntry:
    """Individual trace log entry"""
    timestamp: str
    session_id: str
    user_id: str
    event_type: FFTraceEvent
    level: FFTraceLevel
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFTraceFilterConfigDTO:
    """Configuration for trace filtering"""
    
    # Level filtering
    min_level: str = "info"
    max_level: str = "critical"
    excluded_levels: List[str] = field(default_factory=list)
    
    # Event type filtering
    included_events: List[str] = field(default_factory=list)  # Empty = all
    excluded_events: List[str] = field(default_factory=list)
    
    # Component filtering
    included_components: List[str] = field(default_factory=list)  # Empty = all
    excluded_components: List[str] = field(default_factory=list)
    
    # Content filtering
    keyword_filters: List[str] = field(default_factory=list)
    regex_filters: List[str] = field(default_factory=list)
    exclude_sensitive_data: bool = True
    
    # Performance filtering
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    only_slow_operations: bool = False
    slow_operation_threshold_ms: float = 1000.0
    
    # User/session filtering
    included_users: List[str] = field(default_factory=list)
    excluded_users: List[str] = field(default_factory=list)
    included_sessions: List[str] = field(default_factory=list)
    excluded_sessions: List[str] = field(default_factory=list)


@dataclass
class FFTraceAnalysisConfigDTO:
    """Configuration for trace analysis features"""
    
    # Conversation analysis
    enable_conversation_analysis: bool = True
    conversation_window_size: int = 10
    track_conversation_flow: bool = True
    analyze_response_patterns: bool = True
    
    # Performance analysis
    enable_performance_analysis: bool = True
    performance_window_minutes: int = 60
    track_component_performance: bool = True
    enable_bottleneck_detection: bool = True
    
    # Error analysis
    enable_error_analysis: bool = True
    error_pattern_detection: bool = True
    error_correlation_analysis: bool = True
    track_error_recovery: bool = True
    
    # Usage analytics
    enable_usage_analytics: bool = True
    track_feature_usage: bool = True
    analyze_user_patterns: bool = True
    component_usage_tracking: bool = True
    
    # Real-time analysis
    enable_real_time_analysis: bool = False
    real_time_window_seconds: int = 60
    real_time_alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.1,
        "response_time_p95": 2000.0,
        "memory_usage_mb": 500.0
    })
    
    # Statistical analysis
    calculate_percentiles: bool = True
    percentiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    enable_trend_analysis: bool = True
    trend_window_hours: int = 24


@dataclass
class FFTraceExportConfigDTO:
    """Configuration for trace data export"""
    
    # Export formats
    supported_formats: List[str] = field(default_factory=lambda: ["json", "csv", "yaml"])
    default_format: str = "json"
    
    # Export scheduling
    enable_scheduled_export: bool = False
    export_interval_hours: int = 24
    export_retention_days: int = 30
    
    # Export destinations
    export_to_file: bool = True
    export_file_path: str = "./traces/exports"
    export_to_ff_storage: bool = True
    export_to_external: bool = False
    external_endpoint: Optional[str] = None
    
    # Export filtering
    use_export_filters: bool = True
    export_batch_size: int = 1000
    compress_exports: bool = True
    include_metadata: bool = True
    
    # Privacy and security
    anonymize_user_data: bool = False
    exclude_sensitive_fields: List[str] = field(default_factory=lambda: [
        "user_id", "session_id", "personal_data"
    ])
    encryption_enabled: bool = False


@dataclass
class FFTraceLoggerConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Trace Logger component"""
    
    # Core logging settings
    log_level: str = "info"
    enable_performance_tracing: bool = True
    enable_component_tracing: bool = True
    enable_error_tracing: bool = True
    enable_debug_tracing: bool = False
    
    # Storage settings
    storage_backend: str = "ff_storage"  # memory, ff_storage, file, database
    store_traces_in_ff: bool = True
    trace_retention_days: int = 7
    max_trace_entries: int = 10000
    enable_compression: bool = True
    
    # Memory management
    memory_buffer_size: int = 1000
    flush_interval_seconds: int = 60
    enable_circular_buffer: bool = True
    memory_cleanup_interval: int = 3600  # seconds
    
    # File storage settings (if using file backend)
    log_file_path: str = "./logs/ff_traces.log"
    max_log_file_size_mb: int = 100
    log_file_rotation: bool = True
    max_log_files: int = 10
    
    # FF integration settings
    use_ff_logging_system: bool = True
    ff_logger_name: str = "ff_trace_logger"
    integrate_with_ff_metrics: bool = True
    
    # Filtering configuration
    filters: FFTraceFilterConfigDTO = field(default_factory=FFTraceFilterConfigDTO)
    
    # Analysis configuration
    analysis: FFTraceAnalysisConfigDTO = field(default_factory=FFTraceAnalysisConfigDTO)
    
    # Export configuration
    export: FFTraceExportConfigDTO = field(default_factory=FFTraceExportConfigDTO)
    
    # Performance settings
    enable_async_logging: bool = True
    max_concurrent_traces: int = 100
    trace_queue_size: int = 1000
    enable_batch_processing: bool = True
    batch_size: int = 50
    
    # Context tracking
    enable_context_tracking: bool = True
    context_correlation: bool = True
    track_cross_component_calls: bool = True
    enable_distributed_tracing: bool = False
    
    # Security and privacy
    enable_data_sanitization: bool = True
    sanitize_user_input: bool = True
    exclude_sensitive_headers: bool = True
    pii_detection_enabled: bool = True
    pii_replacement_text: str = "[REDACTED]"
    
    # Alerting and monitoring
    enable_alerting: bool = False
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "error_rate_per_minute": 10,
        "slow_request_threshold_ms": 5000,
        "memory_usage_threshold_mb": 200
    })
    alert_channels: List[str] = field(default_factory=list)
    
    # Development and debugging features
    enable_debug_mode: bool = False
    debug_verbose_logging: bool = False
    enable_trace_sampling: bool = False
    sampling_rate: float = 1.0  # 1.0 = 100%, 0.1 = 10%
    
    # Use case specific settings
    prompt_sandbox_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_prompt_versioning": True,
        "track_prompt_performance": True,
        "enable_a_b_testing": True,
        "max_prompt_versions": 10
    })
    
    ai_debate_config: Dict[str, Any] = field(default_factory=lambda: {
        "track_argument_flow": True,
        "analyze_debate_quality": True,
        "measure_consensus_building": True,
        "track_participant_contributions": True
    })
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate log level
        valid_levels = [level.value for level in FFTraceLevel]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        
        # Validate storage backend
        valid_backends = [backend.value for backend in FFTraceStorage]
        if self.storage_backend not in valid_backends:
            raise ValueError(f"storage_backend must be one of {valid_backends}")
        
        # Validate retention settings
        if self.trace_retention_days <= 0:
            raise ValueError("trace_retention_days must be positive")
        if self.max_trace_entries <= 0:
            raise ValueError("max_trace_entries must be positive")
        
        # Validate performance settings
        if self.memory_buffer_size <= 0:
            raise ValueError("memory_buffer_size must be positive")
        if self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        
        # Validate sampling rate
        if not 0 <= self.sampling_rate <= 1:
            raise ValueError("sampling_rate must be between 0 and 1")
        
        # Initialize sub-configurations
        if not isinstance(self.filters, FFTraceFilterConfigDTO):
            self.filters = FFTraceFilterConfigDTO()
        if not isinstance(self.analysis, FFTraceAnalysisConfigDTO):
            self.analysis = FFTraceAnalysisConfigDTO()
        if not isinstance(self.export, FFTraceExportConfigDTO):
            self.export = FFTraceExportConfigDTO()
    
    def should_trace_level(self, level: FFTraceLevel) -> bool:
        """Check if a trace level should be logged"""
        level_priorities = {
            FFTraceLevel.DEBUG: 0,
            FFTraceLevel.INFO: 1,
            FFTraceLevel.WARNING: 2,
            FFTraceLevel.ERROR: 3,
            FFTraceLevel.CRITICAL: 4
        }
        
        min_priority = level_priorities.get(FFTraceLevel(self.log_level), 1)
        trace_priority = level_priorities.get(level, 1)
        
        return trace_priority >= min_priority
    
    def should_trace_event(self, event_type: FFTraceEvent) -> bool:
        """Check if an event type should be traced"""
        event_str = event_type.value
        
        # Check excluded events
        if event_str in self.filters.excluded_events:
            return False
        
        # Check included events (empty = all allowed)
        if self.filters.included_events and event_str not in self.filters.included_events:
            return False
        
        return True
    
    def should_trace_component(self, component: str) -> bool:
        """Check if a component should be traced"""
        # Check excluded components
        if component in self.filters.excluded_components:
            return False
        
        # Check included components (empty = all allowed)
        if self.filters.included_components and component not in self.filters.included_components:
            return False
        
        return True
    
    def get_trace_format_config(self, format_type: str) -> Dict[str, Any]:
        """Get configuration for a specific trace format"""
        format_configs = {
            "json": {
                "pretty_print": True,
                "include_metadata": True,
                "date_format": "iso"
            },
            "csv": {
                "delimiter": ",",
                "include_headers": True,
                "quote_strings": True
            },
            "yaml": {
                "default_flow_style": False,
                "indent": 2
            },
            "text": {
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
                "include_level": True,
                "include_component": True
            }
        }
        return format_configs.get(format_type, {})


@dataclass
class FFTraceMetrics:
    """Trace logger metrics and statistics"""
    total_traces: int = 0
    traces_by_level: Dict[str, int] = field(default_factory=dict)
    traces_by_component: Dict[str, int] = field(default_factory=dict)
    traces_by_event_type: Dict[str, int] = field(default_factory=dict)
    average_processing_time_ms: float = 0.0
    error_rate: float = 0.0
    storage_usage_mb: float = 0.0
    last_updated: str = ""


@dataclass
class FFPromptSandboxTrace:
    """Specialized trace for prompt sandbox use case"""
    prompt_id: str
    prompt_version: str
    prompt_text: str
    input_variables: Dict[str, Any]
    output_text: str
    performance_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFDebateTrace:
    """Specialized trace for AI debate use case"""
    debate_id: str
    participant_id: str
    argument_text: str
    argument_type: str  # opening, rebuttal, closing, etc.
    quality_score: float
    persuasiveness_score: float
    factual_accuracy_score: float
    response_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)