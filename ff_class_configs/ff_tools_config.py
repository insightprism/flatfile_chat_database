"""
FF Tools Component Configuration

Configuration classes for the FF Tools component that provides external
system integration using existing FF document processing as backend.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ff_class_configs.ff_base_config import FFBaseConfigDTO


class FFToolType(Enum):
    """Types of tools supported by FF Tools component"""
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    WEB_SEARCH = "web_search"
    EMAIL = "email"
    CALENDAR = "calendar"
    DOCUMENT_PROCESSING = "document_processing"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_INFO = "system_info"
    TRANSLATION = "translation"
    CODE_ANALYSIS = "code_analysis"


class FFToolSecurityLevel(Enum):
    """Security levels for tool execution"""
    READ_ONLY = "read_only"      # No modifications allowed
    RESTRICTED = "restricted"    # Limited safe operations
    SANDBOXED = "sandboxed"     # Isolated execution
    TRUSTED = "trusted"         # Full access (use with caution)


class FFToolExecutionMode(Enum):
    """Tool execution modes"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BACKGROUND = "background"
    STREAMING = "streaming"


@dataclass
class FFToolDefinition:
    """Definition of a tool available to the FF Tools component"""
    name: str
    tool_type: FFToolType
    description: str
    security_level: FFToolSecurityLevel
    parameters: Dict[str, Any]
    timeout: int = 30
    enabled: bool = True
    execution_mode: FFToolExecutionMode = FFToolExecutionMode.SYNCHRONOUS
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFToolsSecurityConfigDTO:
    """Security configuration for FF Tools component"""
    
    # Security levels
    default_security_level: str = "restricted"
    enable_sandboxing: bool = True
    sandbox_timeout: int = 30
    
    # Network restrictions
    allowed_domains: List[str] = field(default_factory=lambda: [
        "api.openai.com", 
        "httpbin.org",
        "jsonplaceholder.typicode.com"
    ])
    blocked_domains: List[str] = field(default_factory=lambda: [
        "localhost", 
        "127.0.0.1", 
        "0.0.0.0",
        "internal"
    ])
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443])
    
    # File system restrictions
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".json", ".csv", ".pdf", ".docx", ".xlsx"
    ])
    blocked_file_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/var", "/usr", "/sys", "/proc", "/root"
    ])
    max_file_size_mb: int = 10
    
    # Command restrictions
    allowed_commands: List[str] = field(default_factory=lambda: [
        "ls", "cat", "head", "tail", "grep", "find", "wc"
    ])
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm", "del", "format", "dd", "chmod", "chown", "sudo", "su"
    ])
    
    # Resource limits
    max_memory_mb: int = 100
    max_cpu_time: int = 10
    max_processes: int = 5


@dataclass
class FFToolsConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Tools component"""
    
    # Tool execution settings
    execution_timeout: int = 30
    max_concurrent_tools: int = 3
    enable_sandboxing: bool = True
    enable_tool_validation: bool = True
    
    # FF integration settings
    use_ff_document_processing: bool = True
    use_ff_storage_backend: bool = True
    store_tool_results: bool = True
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Tool management
    enabled_tools: List[str] = field(default_factory=lambda: [
        "web_search", "document_analysis", "file_info", 
        "api_call", "email_draft", "translation", "data_analysis"
    ])
    disabled_tools: List[str] = field(default_factory=list)
    
    # Security configuration
    security: FFToolsSecurityConfigDTO = field(default_factory=FFToolsSecurityConfigDTO)
    
    # Performance settings
    enable_parallel_execution: bool = True
    max_parallel_tools: int = 2
    tool_priority_levels: Dict[str, int] = field(default_factory=lambda: {
        "system_info": 1,
        "file_operation": 2,
        "document_processing": 3,
        "api_call": 4,
        "web_search": 5
    })
    
    # Monitoring and logging
    enable_tool_metrics: bool = True
    enable_execution_logging: bool = True
    log_tool_inputs: bool = False  # Security: don't log sensitive inputs
    log_tool_outputs: bool = True
    
    # Error handling
    enable_fallback_tools: bool = True
    max_retry_attempts: int = 2
    retry_delay_seconds: int = 1
    enable_graceful_degradation: bool = True
    
    # Tool-specific configurations
    web_search_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_results": 5,
        "timeout": 10,
        "user_agent": "FF-Chat-Tools/1.0"
    })
    
    document_processing_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_file_size_mb": 5,
        "supported_formats": ["pdf", "docx", "txt", "md", "csv"],
        "enable_ocr": False,
        "extract_metadata": True
    })
    
    api_call_config: Dict[str, Any] = field(default_factory=lambda: {
        "timeout": 15,
        "max_redirects": 3,
        "verify_ssl": True,
        "max_response_size_mb": 1
    })
    
    translation_config: Dict[str, Any] = field(default_factory=lambda: {
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"],
        "default_source_language": "auto",
        "default_target_language": "en"
    })
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate configuration
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
        if self.max_concurrent_tools <= 0:
            raise ValueError("max_concurrent_tools must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if self.max_parallel_tools > self.max_concurrent_tools:
            raise ValueError("max_parallel_tools cannot exceed max_concurrent_tools")
        
        # Validate security settings
        if not isinstance(self.security, FFToolsSecurityConfigDTO):
            self.security = FFToolsSecurityConfigDTO()
        
        # Remove disabled tools from enabled tools
        self.enabled_tools = [
            tool for tool in self.enabled_tools 
            if tool not in self.disabled_tools
        ]
        
        # Validate tool-specific configs
        self._validate_tool_configs()
    
    def _validate_tool_configs(self):
        """Validate tool-specific configurations"""
        # Web search config validation
        if self.web_search_config.get("max_results", 0) <= 0:
            self.web_search_config["max_results"] = 5
        if self.web_search_config.get("timeout", 0) <= 0:
            self.web_search_config["timeout"] = 10
        
        # Document processing config validation
        if self.document_processing_config.get("max_file_size_mb", 0) <= 0:
            self.document_processing_config["max_file_size_mb"] = 5
        
        # API call config validation
        if self.api_call_config.get("timeout", 0) <= 0:
            self.api_call_config["timeout"] = 15
        if self.api_call_config.get("max_response_size_mb", 0) <= 0:
            self.api_call_config["max_response_size_mb"] = 1
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled"""
        return (tool_name in self.enabled_tools and 
                tool_name not in self.disabled_tools)
    
    def get_tool_priority(self, tool_name: str) -> int:
        """Get priority level for a tool"""
        return self.tool_priority_levels.get(tool_name, 99)
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        config_map = {
            "web_search": self.web_search_config,
            "document_analysis": self.document_processing_config,
            "document_processing": self.document_processing_config,
            "api_call": self.api_call_config,
            "translation": self.translation_config
        }
        return config_map.get(tool_name, {})


@dataclass 
class FFToolExecutionResult:
    """Result of tool execution"""
    tool_name: str
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cached: bool = False
    security_level: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFToolExecutionContext:
    """Context for tool execution"""
    session_id: str
    user_id: str
    tool_name: str
    parameters: Dict[str, Any]
    security_level: FFToolSecurityLevel
    timeout: int
    execution_mode: FFToolExecutionMode
    metadata: Dict[str, Any] = field(default_factory=dict)


# Use case specific configurations
@dataclass
class FFPersonalAssistantToolsConfig:
    """Configuration for personal assistant use case tools"""
    enable_calendar_integration: bool = False
    enable_email_integration: bool = True
    enable_file_management: bool = True
    enable_web_search: bool = True
    enable_translation: bool = True
    
    calendar_provider: str = "generic"
    email_provider: str = "generic"
    default_file_formats: List[str] = field(default_factory=lambda: ["pdf", "docx", "txt"])


@dataclass
class FFChatOpsToolsConfig:
    """Configuration for ChatOps use case tools"""
    enable_system_monitoring: bool = True
    enable_deployment_tools: bool = False  # Disabled for security
    enable_log_analysis: bool = True
    enable_metric_collection: bool = True
    
    monitoring_endpoints: List[str] = field(default_factory=list)
    log_sources: List[str] = field(default_factory=list)
    metric_providers: List[str] = field(default_factory=lambda: ["generic"])


@dataclass
class FFAutoTaskAgentConfig:
    """Configuration for auto task agent use case"""
    enable_task_scheduling: bool = True
    enable_workflow_automation: bool = True
    enable_notification_system: bool = True
    
    max_automated_tasks: int = 5
    task_timeout_minutes: int = 30
    enable_human_approval: bool = True
    approval_timeout_minutes: int = 60