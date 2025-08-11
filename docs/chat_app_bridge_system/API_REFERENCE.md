# Chat Application Bridge System - API Reference

## Overview

This document provides comprehensive API reference for all components of the Chat Application Bridge System. All APIs are designed to be async/await compatible and follow standardized response patterns for consistency.

## Core Components

### FFChatAppBridge

The main bridge class that provides chat-optimized interface to Flatfile Database.

#### Class Methods

##### `FFChatAppBridge.__init__(config: ChatAppStorageConfig)`

Initialize bridge with configuration.

**Parameters:**
- `config` (ChatAppStorageConfig): Configuration object for the bridge

**Example:**
```python
from ff_chat_integration import FFChatAppBridge, ChatAppStorageConfig

config = ChatAppStorageConfig(
    storage_path="./chat_data",
    performance_mode="balanced"
)
bridge = FFChatAppBridge(config)
await bridge.initialize()
```

##### `FFChatAppBridge.create_for_chat_app(storage_path: str, options: Optional[Dict[str, Any]]) -> FFChatAppBridge`

Factory method for simple bridge creation.

**Parameters:**
- `storage_path` (str): Base storage path
- `options` (Dict[str, Any], optional): Configuration overrides

**Returns:**
- `FFChatAppBridge`: Fully initialized bridge instance

**Raises:**
- `ConfigurationError`: Invalid configuration provided
- `InitializationError`: Storage initialization failed

**Example:**
```python
# Simple setup
bridge = await FFChatAppBridge.create_for_chat_app("./chat_data")

# With custom options
bridge = await FFChatAppBridge.create_for_chat_app(
    "./chat_data",
    {
        "performance_mode": "speed",
        "cache_size_mb": 200,
        "enable_analytics": True
    }
)
```

##### `FFChatAppBridge.create_from_preset(preset_name: str, storage_path: str, overrides: Optional[Dict[str, Any]]) -> FFChatAppBridge`

Create bridge from configuration preset.

**Parameters:**
- `preset_name` (str): Name of preset ("development", "production", "high_performance", etc.)
- `storage_path` (str): Storage path
- `overrides` (Dict[str, Any], optional): Configuration overrides

**Returns:**
- `FFChatAppBridge`: Initialized bridge with preset configuration

**Available Presets:**
- `development`: Development-optimized settings
- `production`: Production-ready configuration
- `high_performance`: Maximum performance settings
- `feature_rich`: All features enabled
- `lightweight`: Minimal resource usage

**Example:**
```python
# Use production preset
bridge = await FFChatAppBridge.create_from_preset(
    "production",
    "/var/lib/chatapp/data"
)

# Production preset with overrides
bridge = await FFChatAppBridge.create_from_preset(
    "production",
    "./prod_data",
    {"cache_size_mb": 300}
)
```

##### `FFChatAppBridge.create_for_use_case(use_case: str, storage_path: str, **kwargs) -> FFChatAppBridge`

Create bridge optimized for specific use case.

**Parameters:**
- `use_case` (str): Type of chat application
- `storage_path` (str): Storage path
- `**kwargs`: Additional configuration options

**Supported Use Cases:**
- `simple_chat`: Basic chat functionality
- `ai_assistant`: AI/ML chat applications
- `high_volume_chat`: High-throughput chat systems
- `enterprise_chat`: Enterprise chat solutions
- `research_chat`: Research and analysis applications
- `gaming_chat`: Gaming chat systems

**Example:**
```python
# AI assistant setup
bridge = await FFChatAppBridge.create_for_use_case(
    "ai_assistant",
    "./ai_data",
    enable_vector_search=True,
    cache_size_mb=250
)
```

#### Instance Methods

##### `FFChatAppBridge.initialize() -> bool`

Initialize the bridge and underlying storage.

**Returns:**
- `bool`: True if initialization successful

**Raises:**
- `InitializationError`: If initialization fails

##### `FFChatAppBridge.get_standardized_config() -> Dict[str, Any]`

Get configuration in standardized format.

**Returns:**
- `Dict[str, Any]`: Standardized configuration dictionary

**Response Format:**
```python
{
    "storage_path": str,
    "capabilities": {
        "vector_search": bool,
        "streaming": bool,
        "analytics": bool,
        "compression": bool
    },
    "performance": {
        "mode": str,
        "cache_size_mb": int,
        "max_session_size_mb": int
    },
    "features": {
        "backup": bool,
        "compression": bool,
        "batch_size": int,
        "page_size": int
    },
    "environment": str,
    "initialized": bool,
    "initialization_time": Optional[float]
}
```

##### `FFChatAppBridge.health_check() -> Dict[str, Any]`

Comprehensive health check.

**Returns:**
- `Dict[str, Any]`: Health status information

**Response Format:**
```python
{
    "status": str,  # "healthy", "degraded", "error"
    "timestamp": str,
    "bridge_initialized": bool,
    "storage_accessible": bool,
    "write_permissions": bool,
    "disk_space_sufficient": bool,
    "performance_metrics": Dict[str, Any],
    "uptime_seconds": float,
    "configuration_valid": bool,
    "errors": List[str],
    "warnings": List[str]
}
```

##### `FFChatAppBridge.get_capabilities() -> Dict[str, Any]`

Discover available features and capabilities.

**Returns:**
- `Dict[str, Any]`: Available capabilities

**Response Format:**
```python
{
    "vector_search": bool,
    "streaming": bool,
    "analytics": bool,
    "compression": bool,
    "backup": bool,
    "max_file_size_mb": int,
    "supported_formats": List[str],
    "search_types": List[str],
    "storage_features": List[str]
}
```

##### `FFChatAppBridge.get_data_layer() -> FFChatDataLayer`

Get chat-optimized data access layer.

**Returns:**
- `FFChatDataLayer`: Data layer instance

##### `FFChatAppBridge.close() -> None`

Clean shutdown of all resources.

---

### ChatAppStorageConfig

Configuration class for chat applications.

#### Constructor

##### `ChatAppStorageConfig.__init__(**kwargs)`

**Parameters:**
- `storage_path` (str): Base storage path (required)
- `enable_vector_search` (bool): Enable vector search (default: True)
- `enable_streaming` (bool): Enable streaming operations (default: True)
- `enable_analytics` (bool): Enable analytics collection (default: True)
- `enable_compression` (bool): Enable data compression (default: False)
- `backup_enabled` (bool): Enable backups (default: False)
- `cache_size_mb` (int): Cache size in MB (default: 100)
- `performance_mode` (str): Performance mode (default: "balanced")
- `max_session_size_mb` (int): Max session size in MB (default: 50)
- `message_batch_size` (int): Message batch size (default: 100)
- `history_page_size` (int): History page size (default: 50)
- `search_result_limit` (int): Search result limit (default: 20)
- `environment` (str): Environment ("development", "production", "test")

#### Methods

##### `ChatAppStorageConfig.validate() -> List[str]`

Validate configuration settings.

**Returns:**
- `List[str]`: List of validation issues (empty if valid)

##### `ChatAppStorageConfig.to_dict() -> Dict[str, Any]`

Convert configuration to dictionary format.

**Returns:**
- `Dict[str, Any]`: Configuration as dictionary

---

### FFChatDataLayer

Chat-optimized data access layer with standardized responses.

#### Methods

##### `FFChatDataLayer.store_chat_message(user_id: str, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]`

Store a chat message with optimized handling.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `message` (Dict[str, Any]): Message data

**Message Format:**
```python
{
    "role": str,  # "user", "assistant", "system"
    "content": str,
    "metadata": Optional[Dict[str, Any]],
    "timestamp": Optional[str]
}
```

**Returns:**
- `Dict[str, Any]`: Standardized response

**Response Format:**
```python
{
    "success": bool,
    "data": {
        "message_id": str,
        "stored_at": str,
        "session_updated": bool
    },
    "metadata": {
        "operation": "store_chat_message",
        "operation_time_ms": float,
        "records_affected": int,
        "performance_metrics": {
            "cache_hit": bool,
            "validation_time_ms": float,
            "storage_time_ms": float
        }
    },
    "error": Optional[str],
    "warnings": List[str]
}
```

##### `FFChatDataLayer.get_chat_history(user_id: str, session_id: str, limit: Optional[int], offset: Optional[int]) -> Dict[str, Any]`

Retrieve chat history for a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `limit` (int, optional): Maximum messages to retrieve
- `offset` (int, optional): Offset for pagination

**Returns:**
- `Dict[str, Any]`: Standardized response with messages

**Response Format:**
```python
{
    "success": bool,
    "data": {
        "messages": List[Dict[str, Any]],
        "total_messages": int,
        "session_info": {
            "session_id": str,
            "session_name": str,
            "created_at": str,
            "updated_at": str
        },
        "pagination": {
            "limit": int,
            "offset": int,
            "has_more": bool
        }
    },
    "metadata": {
        "operation": "get_chat_history",
        "operation_time_ms": float,
        "records_retrieved": int,
        "performance_metrics": {
            "cache_hit": bool,
            "query_time_ms": float
        }
    },
    "error": Optional[str],
    "warnings": List[str]
}
```

##### `FFChatDataLayer.search_conversations(user_id: str, query: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]`

Search across conversations.

**Parameters:**
- `user_id` (str): User identifier
- `query` (str): Search query
- `options` (Dict[str, Any], optional): Search options

**Search Options:**
```python
{
    "search_type": str,  # "text", "vector", "hybrid"
    "session_ids": Optional[List[str]],
    "date_range": Optional[Dict[str, str]],
    "limit": Optional[int],
    "include_metadata": bool
}
```

**Returns:**
- `Dict[str, Any]`: Search results in standardized format

##### `FFChatDataLayer.stream_conversation(user_id: str, session_id: str, chunk_size: int) -> AsyncIterator[Dict[str, Any]]`

Stream conversation data in chunks.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `chunk_size` (int): Size of each chunk

**Yields:**
- `Dict[str, Any]`: Chunk data in standardized format

##### `FFChatDataLayer.get_analytics_summary(user_id: str, time_range: Optional[Dict[str, str]]) -> Dict[str, Any]`

Get analytics summary for user.

**Parameters:**
- `user_id` (str): User identifier
- `time_range` (Dict[str, str], optional): Time range for analytics

**Returns:**
- `Dict[str, Any]`: Analytics data in standardized format

##### `FFChatDataLayer.get_performance_metrics() -> Dict[str, Any]`

Get performance metrics for data layer operations.

**Returns:**
- `Dict[str, Any]`: Performance metrics

**Response Format:**
```python
{
    "operation_metrics": {
        "operation_name": {
            "total_operations": int,
            "average_ms": float,
            "recent_avg_ms": float,
            "min_ms": float,
            "max_ms": float,
            "error_rate": float
        }
    },
    "cache_stats": {
        "cache_size": int,
        "cache_hits": int,
        "cache_misses": int,
        "cache_hit_rate": float
    },
    "system_metrics": {
        "memory_usage_mb": float,
        "disk_usage_mb": float,
        "active_connections": int
    }
}
```

---

### FFChatConfigFactory

Factory for creating and managing configurations.

#### Methods

##### `FFChatConfigFactory.list_templates() -> Dict[str, Dict[str, Any]]`

List available configuration templates.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Template information

##### `FFChatConfigFactory.create_from_template(template_name: str, storage_path: str, overrides: Optional[Dict[str, Any]]) -> ChatAppStorageConfig`

Create configuration from template.

**Parameters:**
- `template_name` (str): Template name
- `storage_path` (str): Storage path
- `overrides` (Dict[str, Any], optional): Configuration overrides

**Returns:**
- `ChatAppStorageConfig`: Created configuration

##### `FFChatConfigFactory.create_for_environment(environment: str, storage_path: str, performance_level: str) -> ChatAppStorageConfig`

Create environment-specific configuration.

**Parameters:**
- `environment` (str): Target environment
- `storage_path` (str): Storage path
- `performance_level` (str): Performance level

**Returns:**
- `ChatAppStorageConfig`: Environment-optimized configuration

##### `FFChatConfigFactory.create_for_use_case(use_case: str, storage_path: str, **kwargs) -> ChatAppStorageConfig`

Create use-case optimized configuration.

**Parameters:**
- `use_case` (str): Use case type
- `storage_path` (str): Storage path
- `**kwargs`: Additional options

**Returns:**
- `ChatAppStorageConfig`: Use-case optimized configuration

##### `FFChatConfigFactory.validate_and_optimize(config: ChatAppStorageConfig) -> Dict[str, Any]`

Validate and provide optimization recommendations.

**Parameters:**
- `config` (ChatAppStorageConfig): Configuration to validate

**Returns:**
- `Dict[str, Any]`: Validation results and recommendations

##### `FFChatConfigFactory.migrate_from_wrapper_config(wrapper_config: Dict[str, Any]) -> ChatAppStorageConfig`

Migrate from wrapper-based configuration.

**Parameters:**
- `wrapper_config` (Dict[str, Any]): Old wrapper configuration

**Returns:**
- `ChatAppStorageConfig`: Migrated bridge configuration

---

### FFIntegrationHealthMonitor

Comprehensive health monitoring and diagnostics.

#### Constructor

##### `FFIntegrationHealthMonitor.__init__(bridge: FFChatAppBridge)`

**Parameters:**
- `bridge` (FFChatAppBridge): Bridge instance to monitor

#### Methods

##### `FFIntegrationHealthMonitor.start_monitoring(interval_seconds: int = 60) -> None`

Start background monitoring.

**Parameters:**
- `interval_seconds` (int): Monitoring interval (default: 60)

##### `FFIntegrationHealthMonitor.stop_monitoring() -> None`

Stop background monitoring.

##### `FFIntegrationHealthMonitor.comprehensive_health_check() -> Dict[str, Any]`

Perform comprehensive health check.

**Returns:**
- `Dict[str, Any]`: Complete health status

**Response Format:**
```python
{
    "overall_status": str,  # "healthy", "degraded", "error"
    "overall_score": int,   # 0-100
    "component_health": {
        "bridge": {
            "status": str,
            "score": int,
            "issues": List[str]
        },
        "storage": {
            "status": str,
            "score": int,
            "issues": List[str]
        },
        "data_layer": {
            "status": str,
            "score": int,
            "issues": List[str]
        }
    },
    "system_health": {
        "memory_usage": {
            "status": str,
            "usage_percent": float,
            "available_mb": float
        },
        "disk_space": {
            "status": str,
            "usage_percent": float,
            "available_gb": float
        },
        "cpu_usage": {
            "status": str,
            "usage_percent": float
        }
    },
    "performance_health": {
        "response_times": {
            "status": str,
            "average_ms": float,
            "recent_trend": str
        },
        "throughput": {
            "status": str,
            "operations_per_second": float
        }
    },
    "optimization_score": int,
    "recommendations": List[str],
    "timestamp": str
}
```

##### `FFIntegrationHealthMonitor.get_performance_analytics() -> Dict[str, Any]`

Get detailed performance analytics.

**Returns:**
- `Dict[str, Any]`: Performance analytics data

##### `FFIntegrationHealthMonitor.diagnose_issues() -> Dict[str, Any]`

Automated issue diagnosis.

**Returns:**
- `Dict[str, Any]`: Issue diagnosis and resolution plan

##### `FFIntegrationHealthMonitor.get_current_status() -> Dict[str, Any]`

Get current monitoring status.

**Returns:**
- `Dict[str, Any]`: Current status information

---

## Convenience Functions

### Configuration Functions

##### `create_chat_config_for_development(storage_path: str = "./dev_data", **overrides) -> ChatAppStorageConfig`

Create development-optimized configuration.

##### `create_chat_config_for_production(storage_path: str, performance_level: str = "balanced", **overrides) -> ChatAppStorageConfig`

Create production-optimized configuration.

##### `get_chat_app_presets() -> Dict[str, ChatAppStorageConfig]`

Get all predefined configuration presets.

##### `validate_chat_app_config(config: ChatAppStorageConfig) -> List[str]`

Validate configuration for chat app compatibility.

##### `optimize_chat_config(config: ChatAppStorageConfig) -> Dict[str, Any]`

Analyze and optimize configuration.

### Diagnostic Functions

##### `diagnose_bridge_issues(bridge: FFChatAppBridge) -> Dict[str, Any]`

Quick issue diagnosis for bridge instance.

##### `validate_bridge_system() -> bool`

Validate entire bridge system functionality.

---

## Exception Classes

### ChatIntegrationError

Base exception class for all bridge-related errors.

**Attributes:**
- `message` (str): Error message
- `context` (Dict[str, Any]): Error context
- `suggestions` (List[str]): Resolution suggestions
- `error_code` (str): Error code

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to dictionary format

### ConfigurationError

Configuration-related errors.

**Additional Attributes:**
- `config_field` (str): Problematic configuration field
- `config_value` (Any): Problematic value

### InitializationError

Initialization-related errors.

**Additional Attributes:**
- `component` (str): Component that failed to initialize
- `initialization_step` (str): Step where initialization failed

### StorageError

Storage operation errors.

**Additional Attributes:**
- `operation` (str): Storage operation that failed
- `storage_path` (str): Path involved in operation

---

## Response Format Standards

All API methods that perform operations return responses in this standardized format:

```python
{
    "success": bool,
    "data": Any,  # Operation-specific data
    "metadata": {
        "operation": str,
        "operation_time_ms": float,
        "records_affected": int,
        "performance_metrics": Dict[str, Any]
    },
    "error": Optional[str],
    "warnings": List[str]
}
```

This ensures consistent error handling and performance monitoring across all operations.

---

## Error Handling

All async methods may raise exceptions. Always wrap calls in try-catch blocks:

```python
try:
    result = await data_layer.store_chat_message(user_id, session_id, message)
    if not result["success"]:
        print(f"Operation failed: {result['error']}")
except ChatIntegrationError as e:
    print(f"Integration error: {e}")
    print(f"Suggestions: {e.suggestions}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Version Information

**Current Version:** 1.0.0  
**API Stability:** Stable  
**Python Compatibility:** 3.8+  
**Dependencies:** See requirements.txt

For migration guides and integration examples, see the accompanying documentation files.