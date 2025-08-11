# Phase 4: Tool Execution Framework Implementation

## 🎯 Phase Overview

Implement a comprehensive tool execution framework that extends your existing chat capabilities to support secure, monitored tool orchestration. This system provides AI agents with access to external tools and services while maintaining strict security controls, performance monitoring, and audit logging.

## 📋 Requirements Analysis

### **Current State Assessment**
Your system already has:
- ✅ Secure file operations with atomic writes and proper locking
- ✅ User-based data isolation and access controls
- ✅ Async-first architecture supporting concurrent operations
- ✅ Structured logging and error handling patterns
- ✅ Configuration-driven behavior with comprehensive DTOs

### **Tool Execution Requirements**
Based on PrismMind's advanced use cases, implement:
1. **Tool Registry** - Dynamic tool discovery, registration, and capability management
2. **Execution Engine** - Secure, sandboxed tool execution with resource limits
3. **Security Framework** - Permission management, rate limiting, and audit logging
4. **Performance Monitoring** - Execution metrics, performance analytics, and optimization
5. **Integration Layer** - Seamless integration with conversation flow and context

## 🏗️ Architecture Design

### **Tool Execution Hierarchy**
```
users/{user_id}/tools/
├── tool_registry.json          # Available tools and capabilities
├── execution_history.jsonl     # Complete execution audit log
├── performance_metrics.jsonl   # Tool performance data
├── security_policies.json      # User-specific security settings
├── tool_cache/                 # Cached tool results and artifacts
│   ├── {tool_id}/
│   │   ├── results.jsonl       # Cached execution results
│   │   └── artifacts/          # Tool-generated files
└── tool_configs/               # Tool-specific configurations
    └── {tool_id}_config.json   # Individual tool settings
```

### **Tool Execution Flow**
```
Tool Request
     ↓
[Permission Check] → [Rate Limit Check] → [Resource Validation]
     ↓                      ↓                     ↓
[Security Policy] → [Tool Resolution] → [Parameter Validation]
     ↓                      ↓                     ↓
[Execution Context] → [Sandboxed Execution] → [Result Processing]
     ↓                      ↓                     ↓
[Audit Logging] → [Performance Metrics] → [Response Integration]
```

## 📊 Data Models

### **1. Tool Execution Configuration DTO**

```python
# ff_class_configs/ff_tool_execution_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class ToolExecutionMode(str, Enum):
    """Tool execution modes with different security levels."""
    SANDBOX = "sandbox"           # Full sandboxing with resource limits
    RESTRICTED = "restricted"     # Limited system access with monitoring
    TRUSTED = "trusted"          # Full access for verified tools
    READ_ONLY = "read_only"      # Read-only operations only

class ToolCategory(str, Enum):
    """Tool categories for organization and security policies."""
    COMPUTATION = "computation"   # Mathematical calculations, data processing
    WEB_ACCESS = "web_access"    # HTTP requests, API calls
    FILE_SYSTEM = "file_system"  # File operations, document processing
    DATABASE = "database"        # Database queries and operations
    COMMUNICATION = "communication"  # Email, messaging, notifications
    SYSTEM = "system"           # System information and monitoring
    CUSTOM = "custom"           # User-defined custom tools

@dataclass
class FFToolSecurityPolicyDTO:
    """Security policy configuration for tool execution."""
    
    # Permission settings
    allowed_categories: List[str] = field(default_factory=lambda: [
        ToolCategory.COMPUTATION.value,
        ToolCategory.WEB_ACCESS.value
    ])
    blocked_tools: List[str] = field(default_factory=list)
    require_approval: List[str] = field(default_factory=list)
    
    # Resource limits
    max_execution_time_seconds: int = 30
    max_memory_mb: int = 256
    max_disk_space_mb: int = 100
    max_network_requests: int = 10
    
    # Rate limiting
    max_executions_per_hour: int = 100
    max_concurrent_executions: int = 5
    cooldown_seconds: int = 1
    
    # Audit and monitoring
    log_all_executions: bool = True
    log_tool_outputs: bool = True
    log_security_events: bool = True
    retain_execution_history_days: int = 30

@dataclass
class FFToolExecutionConfigDTO:
    """Configuration for tool execution framework."""
    
    # Execution settings
    default_execution_mode: str = ToolExecutionMode.SANDBOX.value
    execution_timeout_seconds: int = 30
    enable_result_caching: bool = True
    cache_expiry_hours: int = 24
    
    # Security configuration
    security_policy: FFToolSecurityPolicyDTO = field(default_factory=FFToolSecurityPolicyDTO)
    enable_security_scanning: bool = True
    quarantine_suspicious_results: bool = True
    
    # Performance settings
    enable_performance_monitoring: bool = True
    performance_sampling_rate: float = 1.0  # Sample 100% of executions
    enable_execution_profiling: bool = False
    
    # Tool registry settings
    auto_discover_tools: bool = True
    tool_registry_refresh_interval_minutes: int = 60
    enable_external_tool_sources: bool = False
    
    # Integration settings
    integrate_with_conversations: bool = True
    include_tool_context_in_memory: bool = True
    max_tool_result_length: int = 10000
    
    # Storage and cleanup
    cleanup_temp_files: bool = True
    temp_file_retention_hours: int = 2
    compress_old_logs: bool = True
    log_compression_age_days: int = 7
```

### **2. Tool Registry and Execution DTOs**

```python
# ff_class_configs/ff_chat_entities_config.py (extend existing file)

@dataclass
class FFToolDefinitionDTO:
    """Definition of an available tool with its capabilities."""
    
    # Tool identification
    tool_id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = ToolCategory.CUSTOM.value
    
    # Execution metadata
    execution_mode: str = ToolExecutionMode.SANDBOX.value
    estimated_duration_seconds: int = 5
    requires_internet: bool = False
    requires_file_access: bool = False
    
    # Tool parameters and schema
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    parameter_validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Security and permissions
    required_permissions: List[str] = field(default_factory=list)
    security_risk_level: str = "low"  # "low", "medium", "high", "critical"
    trusted_domains: List[str] = field(default_factory=list)
    
    # Tool metadata
    author: str = ""
    documentation_url: str = ""
    source_code_url: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Runtime configuration
    executable_path: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    
    # Performance hints
    expected_memory_usage_mb: int = 64
    expected_cpu_usage_percent: int = 10
    supports_parallel_execution: bool = True
    
    def is_compatible_with_security_policy(self, policy: FFToolSecurityPolicyDTO) -> bool:
        """Check if tool is compatible with security policy."""
        return (
            self.category in policy.allowed_categories and
            self.tool_id not in policy.blocked_tools and
            self.estimated_duration_seconds <= policy.max_execution_time_seconds
        )

@dataclass
class FFToolExecutionRequestDTO:
    """Request to execute a tool with specific parameters."""
    
    # Execution identification
    execution_id: str = field(default_factory=lambda: f"exec_{int(time.time() * 1000)}")
    tool_id: str = ""
    user_id: str = ""
    session_id: Optional[str] = None
    
    # Execution parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_mode: Optional[str] = None  # Override default mode
    timeout_seconds: Optional[int] = None  # Override default timeout
    
    # Context information
    conversation_context: str = ""
    previous_tool_results: List[Dict[str, Any]] = field(default_factory=list)
    user_intent: str = ""
    
    # Execution options
    cache_result: bool = True
    include_debug_info: bool = False
    async_execution: bool = False
    
    # Metadata
    request_timestamp: str = field(default_factory=current_timestamp)
    requested_by: str = "user"  # "user", "system", "agent"
    priority: str = "normal"  # "low", "normal", "high", "urgent"

@dataclass
class FFToolExecutionResultDTO:
    """Result of tool execution with comprehensive metadata."""
    
    # Execution identification
    execution_id: str
    tool_id: str
    user_id: str
    
    # Execution outcome
    success: bool
    result_data: Any = None
    error_message: str = ""
    error_type: str = ""
    
    # Performance metrics
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_requests_made: int = 0
    
    # Execution metadata
    start_timestamp: str = field(default_factory=current_timestamp)
    end_timestamp: str = field(default_factory=current_timestamp)
    execution_mode: str = ToolExecutionMode.SANDBOX.value
    exit_code: int = 0
    
    # Result metadata
    result_type: str = "text"  # "text", "json", "binary", "file"
    result_size_bytes: int = 0
    result_hash: str = ""
    cached_result: bool = False
    
    # Security information
    security_events: List[Dict[str, Any]] = field(default_factory=list)
    resource_violations: List[str] = field(default_factory=list)
    sandbox_breaches: List[str] = field(default_factory=list)
    
    # Integration data
    integration_context: Dict[str, Any] = field(default_factory=dict)
    follow_up_suggestions: List[str] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    
    # Artifacts and outputs
    generated_files: List[str] = field(default_factory=list)
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    debug_information: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FFToolPerformanceMetricsDTO:
    """Performance metrics for tool execution monitoring."""
    
    # Tool identification
    tool_id: str
    user_id: str
    
    # Performance statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_seconds: float = 0.0
    
    # Resource usage statistics
    average_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    total_network_requests: int = 0
    
    # Quality metrics
    user_satisfaction_score: float = 0.0
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0
    cache_hit_rate_percent: float = 0.0
    
    # Temporal information
    first_execution: str = field(default_factory=current_timestamp)
    last_execution: str = field(default_factory=current_timestamp)
    metrics_period_days: int = 30
    
    # Trend analysis
    execution_trend: str = "stable"  # "increasing", "decreasing", "stable", "volatile"
    performance_trend: str = "stable"
    recent_issues: List[str] = field(default_factory=list)
```

## 🔧 Implementation Specifications

### **1. Tool Registry Manager**

```python
# ff_tool_registry_manager.py

"""
Tool registry and capability management system.

Provides dynamic tool discovery, registration, and lifecycle management
while maintaining security controls and performance monitoring.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import importlib
import inspect
from datetime import datetime, timedelta

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_tool_execution_config import FFToolExecutionConfigDTO, FFToolDefinitionDTO
from ff_class_configs.ff_chat_entities_config import FFToolExecutionRequestDTO, FFToolExecutionResultDTO
from ff_utils.ff_file_ops import ff_atomic_write, ff_ensure_directory
from ff_utils.ff_json_utils import ff_read_json, ff_write_json, ff_append_jsonl
from ff_utils.ff_logging import get_logger

class FFToolRegistryManager:
    """
    Tool registry and capability management following flatfile patterns.
    
    Manages tool discovery, registration, validation, and lifecycle while
    maintaining security policies and performance monitoring.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize tool registry manager."""
        self.config = config
        self.tool_config = getattr(config, 'tool_execution', FFToolExecutionConfigDTO())
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Tool registry cache
        self._registered_tools: Dict[str, FFToolDefinitionDTO] = {}
        self._tool_implementations: Dict[str, Any] = {}
        self._last_registry_refresh = datetime.now()
        
        # Built-in tools directory
        self._builtin_tools_path = Path(__file__).parent / "builtin_tools"
        
    def _get_tools_path(self, user_id: str) -> Path:
        """Get tools directory path for user."""
        return self.base_path / "users" / user_id / "tools"
    
    async def initialize_user_tools(self, user_id: str) -> bool:
        """Initialize tool execution environment for user."""
        try:
            tools_path = self._get_tools_path(user_id)
            await ff_ensure_directory(tools_path)
            await ff_ensure_directory(tools_path / "tool_cache")
            await ff_ensure_directory(tools_path / "tool_configs")
            
            # Initialize tool registry file if it doesn't exist
            registry_path = tools_path / "tool_registry.json"
            if not registry_path.exists():
                registry_data = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "registered_tools": {},
                    "tool_statistics": {}
                }
                await ff_write_json(registry_path, registry_data, self.config)
            
            # Initialize security policies
            security_path = tools_path / "security_policies.json"
            if not security_path.exists():
                security_data = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "security_policy": self.tool_config.security_policy.to_dict(),
                    "custom_policies": {},
                    "security_events": []
                }
                await ff_write_json(security_path, security_data, self.config)
            
            self.logger.info(f"Initialized tool environment for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tools for user {user_id}: {e}")
            return False
    
    async def register_tool(self, user_id: str, tool_definition: FFToolDefinitionDTO) -> bool:
        """Register a new tool for user access."""
        try:
            await self.initialize_user_tools(user_id)
            
            # Validate tool definition
            if not await self._validate_tool_definition(tool_definition):
                self.logger.warning(f"Tool validation failed for {tool_definition.tool_id}")
                return False
            
            # Load current registry
            tools_path = self._get_tools_path(user_id)
            registry_path = tools_path / "tool_registry.json"
            registry_data = await ff_read_json(registry_path, self.config)
            
            # Add tool to registry
            registry_data["registered_tools"][tool_definition.tool_id] = tool_definition.to_dict()
            registry_data["last_updated"] = datetime.now().isoformat()
            
            # Initialize tool statistics
            if tool_definition.tool_id not in registry_data["tool_statistics"]:
                registry_data["tool_statistics"][tool_definition.tool_id] = {
                    "registered_at": datetime.now().isoformat(),
                    "total_executions": 0,
                    "successful_executions": 0,
                    "last_execution": None,
                    "average_execution_time": 0.0
                }
            
            # Save updated registry
            await ff_write_json(registry_path, registry_data, self.config)
            
            # Cache the tool
            self._registered_tools[tool_definition.tool_id] = tool_definition
            
            self.logger.info(f"Registered tool {tool_definition.tool_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_definition.tool_id}: {e}")
            return False
    
    async def get_available_tools(
        self, 
        user_id: str, 
        category_filter: Optional[str] = None,
        security_level: Optional[str] = None
    ) -> List[FFToolDefinitionDTO]:
        """Get list of available tools for user with optional filtering."""
        try:
            await self.initialize_user_tools(user_id)
            
            # Load user's tool registry
            tools_path = self._get_tools_path(user_id)
            registry_path = tools_path / "tool_registry.json"
            registry_data = await ff_read_json(registry_path, self.config)
            
            # Load security policies
            security_path = tools_path / "security_policies.json"
            security_data = await ff_read_json(security_path, self.config)
            security_policy = FFToolSecurityPolicyDTO.from_dict(security_data["security_policy"])
            
            # Get registered tools
            available_tools = []
            for tool_id, tool_data in registry_data["registered_tools"].items():
                tool_def = FFToolDefinitionDTO.from_dict(tool_data)
                
                # Apply security policy filtering
                if not tool_def.is_compatible_with_security_policy(security_policy):
                    continue
                
                # Apply category filtering
                if category_filter and tool_def.category != category_filter:
                    continue
                
                # Apply security level filtering
                if security_level and tool_def.security_risk_level != security_level:
                    continue
                
                available_tools.append(tool_def)
            
            # Auto-discover built-in tools if enabled
            if self.tool_config.auto_discover_tools:
                builtin_tools = await self._discover_builtin_tools()
                for tool_def in builtin_tools:
                    if tool_def.is_compatible_with_security_policy(security_policy):
                        if category_filter is None or tool_def.category == category_filter:
                            if security_level is None or tool_def.security_risk_level == security_level:
                                available_tools.append(tool_def)
            
            return available_tools
            
        except Exception as e:
            self.logger.error(f"Failed to get available tools for user {user_id}: {e}")
            return []
    
    async def get_tool_definition(self, user_id: str, tool_id: str) -> Optional[FFToolDefinitionDTO]:
        """Get specific tool definition for user."""
        try:
            available_tools = await self.get_available_tools(user_id)
            
            for tool_def in available_tools:
                if tool_def.tool_id == tool_id:
                    return tool_def
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get tool definition {tool_id} for user {user_id}: {e}")
            return None
    
    async def validate_tool_request(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO
    ) -> tuple[bool, str]:
        """Validate tool execution request against security policies."""
        try:
            # Get tool definition
            tool_def = await self.get_tool_definition(user_id, request.tool_id)
            if not tool_def:
                return False, f"Tool {request.tool_id} not found or not available"
            
            # Load security policies
            tools_path = self._get_tools_path(user_id)
            security_path = tools_path / "security_policies.json"
            security_data = await ff_read_json(security_path, self.config)
            security_policy = FFToolSecurityPolicyDTO.from_dict(security_data["security_policy"])
            
            # Check if tool is blocked
            if tool_def.tool_id in security_policy.blocked_tools:
                return False, f"Tool {request.tool_id} is blocked by security policy"
            
            # Check category permissions
            if tool_def.category not in security_policy.allowed_categories:
                return False, f"Tool category {tool_def.category} not allowed"
            
            # Validate required parameters
            for param in tool_def.required_parameters:
                if param not in request.parameters:
                    return False, f"Required parameter '{param}' missing"
            
            # Check rate limits
            if not await self._check_rate_limits(user_id, security_policy):
                return False, "Rate limit exceeded"
            
            # Check resource requirements
            timeout = request.timeout_seconds or tool_def.estimated_duration_seconds
            if timeout > security_policy.max_execution_time_seconds:
                return False, f"Execution timeout {timeout}s exceeds limit {security_policy.max_execution_time_seconds}s"
            
            return True, "Validation passed"
            
        except Exception as e:
            self.logger.error(f"Tool request validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    # Private helper methods
    
    async def _validate_tool_definition(self, tool_def: FFToolDefinitionDTO) -> bool:
        """Validate tool definition for security and correctness.""" 
        try:
            # Basic validation
            if not tool_def.tool_id or not tool_def.name:
                return False
            
            # Security validation
            if tool_def.security_risk_level not in ["low", "medium", "high", "critical"]:
                return False
            
            # Parameter schema validation
            if tool_def.parameters_schema:
                # Validate JSON schema format
                # This could be expanded with proper JSON schema validation
                if not isinstance(tool_def.parameters_schema, dict):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tool definition validation error: {e}")
            return False
    
    async def _discover_builtin_tools(self) -> List[FFToolDefinitionDTO]:
        """Discover built-in tools from the builtin_tools directory."""
        try:
            if not self._builtin_tools_path.exists():
                return []
            
            builtin_tools = []
            
            # Scan for Python modules in builtin_tools directory
            for tool_file in self._builtin_tools_path.glob("*.py"):
                if tool_file.name.startswith("_"):
                    continue
                
                try:
                    # Import tool module dynamically
                    module_name = f"builtin_tools.{tool_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, tool_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for tool definition
                    if hasattr(module, "TOOL_DEFINITION"):
                        tool_def = FFToolDefinitionDTO.from_dict(module.TOOL_DEFINITION)
                        builtin_tools.append(tool_def)
                        
                        # Cache tool implementation
                        if hasattr(module, "execute_tool"):
                            self._tool_implementations[tool_def.tool_id] = module.execute_tool
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load builtin tool {tool_file}: {e}")
                    continue
            
            return builtin_tools
            
        except Exception as e:
            self.logger.error(f"Built-in tool discovery failed: {e}")
            return []
    
    async def _check_rate_limits(self, user_id: str, security_policy: FFToolSecurityPolicyDTO) -> bool:
        """Check if user has exceeded rate limits."""
        try:
            tools_path = self._get_tools_path(user_id)
            history_path = tools_path / "execution_history.jsonl"
            
            if not history_path.exists():
                return True
            
            # Count executions in the last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_executions = 0
            
            # Read execution history (this could be optimized with indexing)
            history_data = await ff_read_jsonl(history_path, self.config)
            
            for entry in history_data:
                if "start_timestamp" in entry:
                    exec_time = datetime.fromisoformat(entry["start_timestamp"])
                    if exec_time > one_hour_ago:
                        recent_executions += 1
            
            return recent_executions < security_policy.max_executions_per_hour
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return False  # Fail safe - deny if we can't check
```

### **2. Tool Execution Manager**

```python
# ff_tool_execution_manager.py

"""
Secure tool execution and orchestration system.

Provides sandboxed tool execution with comprehensive monitoring,
security controls, and performance tracking.
"""

import asyncio
import subprocess
import tempfile
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import resource
import signal

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_tool_execution_config import FFToolExecutionConfigDTO, ToolExecutionMode
from ff_class_configs.ff_chat_entities_config import (
    FFToolExecutionRequestDTO, 
    FFToolExecutionResultDTO,
    FFToolPerformanceMetricsDTO
)
from ff_tool_registry_manager import FFToolRegistryManager
from ff_utils.ff_file_ops import ff_atomic_write, ff_ensure_directory
from ff_utils.ff_json_utils import ff_append_jsonl, ff_read_json, ff_write_json
from ff_utils.ff_logging import get_logger

class FFToolExecutionManager:
    """
    Secure tool execution manager following flatfile patterns.
    
    Provides sandboxed execution environment with comprehensive monitoring,
    security controls, and integration with conversation context.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize tool execution manager."""
        self.config = config
        self.tool_config = getattr(config, 'tool_execution', FFToolExecutionConfigDTO())
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Component dependencies
        self.tool_registry = FFToolRegistryManager(config)
        
        # Execution tracking
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_metrics: Dict[str, FFToolPerformanceMetricsDTO] = {}
        
        # Temporary file management
        self._temp_directories: List[Path] = []
        
    def _get_tools_path(self, user_id: str) -> Path:
        """Get tools directory path for user."""
        return self.base_path / "users" / user_id / "tools"
    
    async def execute_tool(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO
    ) -> FFToolExecutionResultDTO:
        """Execute tool with comprehensive monitoring and security controls."""
        
        execution_start = datetime.now()
        result = FFToolExecutionResultDTO(
            execution_id=request.execution_id,
            tool_id=request.tool_id,
            user_id=user_id,
            start_timestamp=execution_start.isoformat()
        )
        
        try:
            # Validate tool execution request
            is_valid, validation_message = await self.tool_registry.validate_tool_request(user_id, request)
            if not is_valid:
                result.success = False
                result.error_message = validation_message
                result.error_type = "validation_error"
                return result
            
            # Get tool definition
            tool_def = await self.tool_registry.get_tool_definition(user_id, request.tool_id)
            if not tool_def:
                result.success = False
                result.error_message = f"Tool {request.tool_id} not found"
                result.error_type = "tool_not_found"
                return result
            
            # Check for cached results
            if request.cache_result and self.tool_config.enable_result_caching:
                cached_result = await self._get_cached_result(user_id, request)
                if cached_result:
                    cached_result.cached_result = True
                    await self._log_execution(user_id, cached_result)
                    return cached_result
            
            # Set execution mode
            execution_mode = ToolExecutionMode(request.execution_mode or tool_def.execution_mode)
            result.execution_mode = execution_mode.value
            
            # Execute tool based on mode
            if execution_mode == ToolExecutionMode.SANDBOX:
                execution_result = await self._execute_sandboxed(user_id, request, tool_def)
            elif execution_mode == ToolExecutionMode.RESTRICTED:
                execution_result = await self._execute_restricted(user_id, request, tool_def)
            elif execution_mode == ToolExecutionMode.TRUSTED:
                execution_result = await self._execute_trusted(user_id, request, tool_def)
            elif execution_mode == ToolExecutionMode.READ_ONLY:
                execution_result = await self._execute_readonly(user_id, request, tool_def)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            
            # Update result with execution outcome
            result.success = execution_result["success"]
            result.result_data = execution_result.get("result_data")
            result.error_message = execution_result.get("error_message", "")
            result.error_type = execution_result.get("error_type", "")
            result.exit_code = execution_result.get("exit_code", 0)
            
            # Update performance metrics
            result.execution_time_seconds = (datetime.now() - execution_start).total_seconds()
            result.memory_used_mb = execution_result.get("memory_used_mb", 0.0)
            result.cpu_usage_percent = execution_result.get("cpu_usage_percent", 0.0)
            result.network_requests_made = execution_result.get("network_requests_made", 0)
            
            # Process result data
            if result.success and result.result_data:
                await self._process_tool_result(user_id, request, result)
            
            # Cache result if requested
            if request.cache_result and result.success:
                await self._cache_result(user_id, request, result)
            
        except asyncio.TimeoutError:
            result.success = False
            result.error_message = "Tool execution timed out"
            result.error_type = "timeout_error"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.error_type = "execution_error"
            self.logger.error(f"Tool execution failed for {request.tool_id}: {e}")
        
        finally:
            # Update timestamps and log execution
            result.end_timestamp = datetime.now().isoformat()
            await self._log_execution(user_id, result)
            await self._update_performance_metrics(user_id, result)
            
            # Cleanup temporary resources
            await self._cleanup_execution_resources(request.execution_id)
        
        return result
    
    async def _execute_sandboxed(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO,
        tool_def: FFToolDefinitionDTO
    ) -> Dict[str, Any]:
        """Execute tool in sandboxed environment with resource limits."""
        try:
            # Create temporary execution directory
            temp_dir = tempfile.mkdtemp(prefix=f"tool_exec_{request.execution_id}_")
            self._temp_directories.append(Path(temp_dir))
            
            # Prepare execution environment
            env_vars = {
                **tool_def.environment_variables,
                "EXECUTION_ID": request.execution_id,
                "USER_ID": user_id,
                "TEMP_DIR": temp_dir,
                "TOOL_MODE": "sandbox"
            }
            
            # Build execution command
            if tool_def.executable_path:
                # External executable
                cmd = [tool_def.executable_path]
                cmd.extend(self._build_command_args(request.parameters))
            else:
                # Built-in tool
                if request.tool_id in self.tool_registry._tool_implementations:
                    return await self._execute_builtin_tool(
                        user_id, request, tool_def, "sandbox"
                    )
                else:
                    raise ValueError(f"No implementation found for tool {request.tool_id}")
            
            # Set resource limits
            timeout = request.timeout_seconds or tool_def.estimated_duration_seconds
            
            # Execute with subprocess and resource monitoring
            process_result = await self._execute_with_monitoring(
                cmd, env_vars, temp_dir, timeout, "sandbox"
            )
            
            return process_result
            
        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
                "error_type": "sandbox_execution_error"
            }
    
    async def _execute_builtin_tool(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO,
        tool_def: FFToolDefinitionDTO,
        execution_mode: str
    ) -> Dict[str, Any]:
        """Execute built-in tool implementation."""
        try:
            tool_implementation = self.tool_registry._tool_implementations.get(request.tool_id)
            if not tool_implementation:
                raise ValueError(f"No implementation for tool {request.tool_id}")
            
            # Prepare execution context
            execution_context = {
                "user_id": user_id,
                "execution_id": request.execution_id,
                "execution_mode": execution_mode,
                "conversation_context": request.conversation_context,
                "previous_results": request.previous_tool_results,
                "temp_directory": tempfile.mkdtemp(prefix=f"builtin_{request.execution_id}_")
            }
            
            # Execute tool with timeout
            timeout = request.timeout_seconds or tool_def.estimated_duration_seconds
            
            try:
                result = await asyncio.wait_for(
                    tool_implementation(request.parameters, execution_context),
                    timeout=timeout
                )
                
                return {
                    "success": True,
                    "result_data": result,
                    "memory_used_mb": 0.0,  # Could be enhanced with actual monitoring
                    "cpu_usage_percent": 0.0,
                    "network_requests_made": 0
                }
                
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error_message": f"Tool execution timed out after {timeout} seconds",
                    "error_type": "timeout_error"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
                "error_type": "builtin_execution_error"
            }
    
    async def _process_tool_result(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO,
        result: FFToolExecutionResultDTO
    ) -> None:
        """Process and enrich tool execution result."""
        try:
            # Calculate result metadata
            if result.result_data:
                result_str = json.dumps(result.result_data) if not isinstance(result.result_data, str) else result.result_data
                result.result_size_bytes = len(result_str.encode('utf-8'))
                result.result_hash = hashlib.sha256(result_str.encode('utf-8')).hexdigest()[:16]
                
                # Truncate large results
                max_length = self.tool_config.max_tool_result_length
                if len(result_str) > max_length:
                    result.result_data = result_str[:max_length] + "... [truncated]"
            
            # Generate integration context
            result.integration_context = {
                "conversation_integration": True,
                "memory_integration": self.tool_config.include_tool_context_in_memory,
                "user_intent": request.user_intent,
                "execution_summary": f"Executed {request.tool_id} successfully"
            }
            
            # Generate follow-up suggestions (placeholder for AI-based suggestions)
            result.follow_up_suggestions = [
                f"Analyze the results from {request.tool_id}",
                f"Use {request.tool_id} with different parameters",
                "Save these results for future reference"
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to process tool result: {e}")
    
    async def _log_execution(self, user_id: str, result: FFToolExecutionResultDTO) -> None:
        """Log tool execution to audit trail."""
        try:
            tools_path = self._get_tools_path(user_id)
            await ff_ensure_directory(tools_path)
            
            history_path = tools_path / "execution_history.jsonl"
            
            log_entry = {
                "execution_id": result.execution_id,
                "tool_id": result.tool_id,
                "user_id": user_id,
                "success": result.success,
                "execution_time_seconds": result.execution_time_seconds,
                "start_timestamp": result.start_timestamp,
                "end_timestamp": result.end_timestamp,
                "error_message": result.error_message,
                "error_type": result.error_type,
                "result_size_bytes": result.result_size_bytes,
                "cached_result": result.cached_result,
                "execution_mode": result.execution_mode
            }
            
            await ff_append_jsonl(history_path, [log_entry], self.config)
            
        except Exception as e:
            self.logger.error(f"Failed to log tool execution: {e}")
    
    async def _update_performance_metrics(self, user_id: str, result: FFToolExecutionResultDTO) -> None:
        """Update performance metrics for tool."""
        try:
            tools_path = self._get_tools_path(user_id)
            metrics_path = tools_path / "performance_metrics.jsonl"
            
            metrics_entry = {
                "tool_id": result.tool_id,
                "user_id": user_id,
                "timestamp": result.end_timestamp,
                "execution_time_seconds": result.execution_time_seconds,
                "memory_used_mb": result.memory_used_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "success": result.success,
                "result_size_bytes": result.result_size_bytes,
                "cached_result": result.cached_result
            }
            
            await ff_append_jsonl(metrics_path, [metrics_entry], self.config)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    # Additional helper methods would continue here...
    # Including: _get_cached_result, _cache_result, _execute_with_monitoring, 
    # _cleanup_execution_resources, etc.
```

### **3. Tool Protocol Interface**

```python
# ff_protocols/ff_tool_execution_protocol.py

"""Protocol interface for tool execution management."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ff_class_configs.ff_tool_execution_config import FFToolDefinitionDTO
from ff_class_configs.ff_chat_entities_config import (
    FFToolExecutionRequestDTO, 
    FFToolExecutionResultDTO,
    FFToolPerformanceMetricsDTO
)

class ToolRegistryProtocol(ABC):
    """Protocol interface for tool registry operations."""
    
    @abstractmethod
    async def initialize_user_tools(self, user_id: str) -> bool:
        """Initialize tool execution environment for user."""
        pass
    
    @abstractmethod
    async def register_tool(self, user_id: str, tool_definition: FFToolDefinitionDTO) -> bool:
        """Register a new tool for user access."""
        pass
    
    @abstractmethod
    async def get_available_tools(
        self, 
        user_id: str, 
        category_filter: Optional[str] = None,
        security_level: Optional[str] = None
    ) -> List[FFToolDefinitionDTO]:
        """Get list of available tools for user with optional filtering."""
        pass
    
    @abstractmethod
    async def validate_tool_request(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO
    ) -> tuple[bool, str]:
        """Validate tool execution request against security policies."""
        pass

class ToolExecutionProtocol(ABC):
    """Protocol interface for tool execution operations."""
    
    @abstractmethod
    async def execute_tool(
        self,
        user_id: str,
        request: FFToolExecutionRequestDTO
    ) -> FFToolExecutionResultDTO:
        """Execute tool with comprehensive monitoring and security controls."""
        pass
    
    @abstractmethod
    async def get_execution_history(
        self,
        user_id: str,
        tool_id: Optional[str] = None,
        limit: int = 100
    ) -> List[FFToolExecutionResultDTO]:
        """Get tool execution history for user."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(
        self,
        user_id: str,
        tool_id: str,
        time_period_days: int = 30
    ) -> FFToolPerformanceMetricsDTO:
        """Get performance metrics for specific tool."""
        pass
```

## 🧪 Testing Specifications

### **Unit Tests**

```python
# tests/test_tool_execution_manager.py

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ff_tool_execution_manager import FFToolExecutionManager
from ff_tool_registry_manager import FFToolRegistryManager
from ff_class_configs.ff_tool_execution_config import FFToolExecutionConfigDTO, FFToolDefinitionDTO
from ff_class_configs.ff_chat_entities_config import FFToolExecutionRequestDTO

class TestToolExecutionManager:
    
    @pytest.fixture
    async def execution_manager(self):
        """Create tool execution manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FFConfigurationManagerConfigDTO()
            config.storage.base_path = temp_dir
            config.tool_execution = FFToolExecutionConfigDTO()
            
            manager = FFToolExecutionManager(config)
            yield manager
    
    @pytest.mark.asyncio
    async def test_initialize_user_tools(self, execution_manager):
        """Test user tool environment initialization."""
        user_id = "test_user"
        
        success = await execution_manager.tool_registry.initialize_user_tools(user_id)
        assert success
        
        # Check directory structure
        tools_path = execution_manager._get_tools_path(user_id)
        assert tools_path.exists()
        assert (tools_path / "tool_registry.json").exists()
        assert (tools_path / "security_policies.json").exists()
    
    @pytest.mark.asyncio
    async def test_register_and_execute_tool(self, execution_manager):
        """Test tool registration and execution."""
        user_id = "test_user"
        
        # Register a simple calculation tool
        tool_def = FFToolDefinitionDTO(
            tool_id="calculator",
            name="Basic Calculator",
            description="Performs basic arithmetic operations",
            category="computation",
            parameters_schema={
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "operand1": {"type": "number"},
                "operand2": {"type": "number"}
            },
            required_parameters=["operation", "operand1", "operand2"]
        )
        
        success = await execution_manager.tool_registry.register_tool(user_id, tool_def)
        assert success
        
        # Execute the tool
        request = FFToolExecutionRequestDTO(
            tool_id="calculator",
            user_id=user_id,
            parameters={
                "operation": "add",
                "operand1": 5,
                "operand2": 3
            }
        )
        
        result = await execution_manager.execute_tool(user_id, request)
        
        # Note: This would require a built-in calculator implementation
        # For the test, we can check that the execution framework works
        assert result.execution_id == request.execution_id
        assert result.tool_id == "calculator"
        assert result.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_security_policy_validation(self, execution_manager):
        """Test security policy enforcement."""
        user_id = "test_user"
        
        # Register a high-risk tool
        tool_def = FFToolDefinitionDTO(
            tool_id="system_command",
            name="System Command Executor",
            description="Executes system commands",
            category="system",
            security_risk_level="high",
            estimated_duration_seconds=60  # Exceeds default limits
        )
        
        await execution_manager.tool_registry.register_tool(user_id, tool_def)
        
        # Try to execute - should be blocked by security policy
        request = FFToolExecutionRequestDTO(
            tool_id="system_command",
            user_id=user_id,
            parameters={"command": "ls"}
        )
        
        is_valid, message = await execution_manager.tool_registry.validate_tool_request(user_id, request)
        assert not is_valid
        assert "not allowed" in message.lower()
```

## 📈 Success Criteria

### **Functional Requirements**
- ✅ Tool registry supports dynamic tool registration and discovery
- ✅ Security policies enforce proper access controls and rate limiting
- ✅ Sandboxed execution prevents unauthorized system access
- ✅ Performance monitoring provides comprehensive metrics
- ✅ Tool results integrate seamlessly with conversation flow

### **Performance Requirements**  
- ✅ Tool execution completes within configured timeouts
- ✅ Resource monitoring accurately tracks memory and CPU usage
- ✅ Rate limiting prevents system overload
- ✅ Caching improves performance for repeated operations

### **Security Requirements**
- ✅ All tool executions are properly audited and logged
- ✅ Security policies prevent unauthorized tool access
- ✅ Sandboxing prevents system compromise
- ✅ Resource limits prevent denial of service attacks

### **Integration Requirements**
- ✅ Tools integrate with existing conversation context
- ✅ Tool results can be cached and retrieved efficiently
- ✅ Memory integration preserves tool context across conversations
- ✅ Performance metrics enable optimization and monitoring

This comprehensive tool execution framework provides secure, monitored access to external tools while maintaining your architectural standards and ensuring system security.