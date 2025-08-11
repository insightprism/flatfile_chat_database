# Error Handling Standards - Chat Application Bridge System

## Overview

This document defines comprehensive error handling standards for the Chat Application Bridge System. These standards ensure consistent, informative, and actionable error handling across all components, enabling developers to quickly diagnose and resolve issues.

## Error Handling Philosophy

### Core Principles

1. **Fail Fast**: Detect errors early and fail quickly rather than propagating invalid states
2. **Informative Messages**: Provide clear, actionable error messages with sufficient context
3. **Graceful Degradation**: Continue operating when possible, with reduced functionality
4. **Consistent Structure**: Use standardized error formats across all components
5. **Recovery Guidance**: Include suggestions for error resolution
6. **Logging Integration**: Integrate with logging systems for debugging and monitoring

### Error Categories

#### 1. User Errors (4xx equivalent)
- Configuration errors
- Invalid input parameters
- Permission/authorization issues
- Resource not found errors

#### 2. System Errors (5xx equivalent)  
- Storage failures
- Network connectivity issues
- Resource exhaustion
- Internal system errors

#### 3. Integration Errors
- Third-party service failures
- Protocol errors
- Version compatibility issues

## Exception Hierarchy

### Base Exception Class

```python
class ChatIntegrationError(Exception):
    """
    Base exception for all Chat Application Bridge errors.
    
    Provides structured error information with context and recovery suggestions.
    """
    
    def __init__(self, 
                 message: str,
                 context: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None,
                 error_code: Optional[str] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize chat integration error.
        
        Args:
            message: Human-readable error message
            context: Additional context information
            suggestions: List of recovery suggestions
            error_code: Unique error code for programmatic handling
            cause: Original exception that caused this error
        """
        self.context = context or {}
        self.suggestions = suggestions or []
        self.error_code = error_code or self.__class__.__name__.upper()
        self.cause = cause
        
        # Format message with context
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            formatted_message = f"{message} (Context: {context_str})"
        else:
            formatted_message = message
        
        super().__init__(formatted_message)
        
        # Chain exceptions properly
        if cause:
            self.__cause__ = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for serialization."""
        return {
            "error": str(self),
            "error_code": self.error_code,
            "context": self.context,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
            "error_type": self.__class__.__name__
        }
    
    def add_context(self, key: str, value: Any) -> 'ChatIntegrationError':
        """Add additional context to the error."""
        self.context[key] = value
        return self
    
    def add_suggestion(self, suggestion: str) -> 'ChatIntegrationError':
        """Add a recovery suggestion to the error."""
        self.suggestions.append(suggestion)
        return self
```

### Specialized Exception Classes

#### Configuration Errors

```python
class ConfigurationError(ChatIntegrationError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, 
                 config_field: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 valid_values: Optional[List[Any]] = None,
                 **kwargs):
        
        context = kwargs.get("context", {})
        suggestions = kwargs.get("suggestions", [])
        
        if config_field:
            context["config_field"] = config_field
        if config_value is not None:
            context["config_value"] = config_value
        if valid_values:
            context["valid_values"] = valid_values
            suggestions.append(f"Use one of: {', '.join(map(str, valid_values))}")
        
        # Add standard configuration suggestions
        if not suggestions:
            suggestions.extend([
                "Check configuration parameter names and values",
                "Verify configuration against API documentation",
                "Use configuration presets for known working configurations"
            ])
        
        super().__init__(
            message,
            context=context,
            suggestions=suggestions,
            error_code="CONFIG_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
        )

def create_validation_error(field: str, value: Any, expected: str) -> ConfigurationError:
    """Create a standardized validation error."""
    return ConfigurationError(
        f"Invalid value for {field}",
        config_field=field,
        config_value=value,
        suggestions=[
            f"Expected {expected}",
            f"Current value '{value}' is not valid",
            "Check the API documentation for valid values"
        ]
    )
```

#### Initialization Errors

```python
class InitializationError(ChatIntegrationError):
    """Initialization and startup errors."""
    
    def __init__(self, message: str,
                 component: Optional[str] = None,
                 initialization_step: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get("context", {})
        suggestions = kwargs.get("suggestions", [])
        
        if component:
            context["component"] = component
        if initialization_step:
            context["initialization_step"] = initialization_step
        
        # Add standard initialization suggestions
        standard_suggestions = [
            "Verify storage path exists and is writable",
            "Check system resources (memory, disk space)",
            "Ensure all dependencies are properly installed",
            "Review configuration settings"
        ]
        
        suggestions.extend([s for s in standard_suggestions if s not in suggestions])
        
        super().__init__(
            message,
            context=context,
            suggestions=suggestions,
            error_code="INIT_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
        )
```

#### Storage Errors

```python
class StorageError(ChatIntegrationError):
    """Storage and persistence errors."""
    
    def __init__(self, message: str,
                 operation: Optional[str] = None,
                 storage_path: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get("context", {})
        suggestions = kwargs.get("suggestions", [])
        
        if operation:
            context["operation"] = operation
        if storage_path:
            context["storage_path"] = storage_path
        
        # Add operation-specific suggestions
        if operation == "read":
            suggestions.extend([
                "Verify file exists and is readable",
                "Check file permissions",
                "Ensure file is not corrupted"
            ])
        elif operation == "write":
            suggestions.extend([
                "Verify write permissions",
                "Check available disk space",
                "Ensure directory exists"
            ])
        elif operation == "delete":
            suggestions.extend([
                "Verify file exists",
                "Check delete permissions",
                "Ensure file is not locked by another process"
            ])
        
        super().__init__(
            message,
            context=context,
            suggestions=suggestions,
            error_code="STORAGE_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
        )
```

## Standardized Response Format

### Success Response Structure

```python
@dataclass
class StandardResponse:
    """Standardized response format for all operations."""
    
    success: bool
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata or {},
            "error": self.error,
            "warnings": self.warnings or []
        }

def create_success_response(data: Any = None, 
                          metadata: Optional[Dict[str, Any]] = None,
                          warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a standardized success response."""
    
    response = StandardResponse(
        success=True,
        data=data,
        metadata=metadata,
        warnings=warnings
    )
    
    return response.to_dict()

def create_error_response(error: Union[str, Exception],
                         context: Optional[Dict[str, Any]] = None,
                         warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    
    if isinstance(error, ChatIntegrationError):
        error_message = str(error)
        if context:
            # Merge context from error and additional context
            merged_context = {**error.context, **context}
        else:
            merged_context = error.context
    else:
        error_message = str(error)
        merged_context = context or {}
    
    response = StandardResponse(
        success=False,
        error=error_message,
        metadata={"context": merged_context},
        warnings=warnings
    )
    
    return response.to_dict()
```

### Error Response Examples

```python
# Example error responses

# Configuration error response
{
    "success": false,
    "data": null,
    "metadata": {
        "context": {
            "config_field": "performance_mode",
            "config_value": "invalid_mode",
            "valid_values": ["speed", "balanced", "quality"]
        }
    },
    "error": "Invalid value for performance_mode (Context: config_field=performance_mode, config_value=invalid_mode)",
    "warnings": []
}

# Storage error response
{
    "success": false,
    "data": null,
    "metadata": {
        "context": {
            "operation": "write",
            "storage_path": "/readonly/path",
            "attempted_file": "messages.json"
        }
    },
    "error": "Failed to write file: Permission denied",
    "warnings": ["Directory may be read-only", "Consider changing storage location"]
}

# Success response with warnings
{
    "success": true,
    "data": {
        "message_id": "msg_12345",
        "stored_at": "2024-01-15T10:30:00Z"
    },
    "metadata": {
        "operation": "store_chat_message",
        "operation_time_ms": 45.2,
        "records_affected": 1,
        "performance_metrics": {
            "cache_hit": false,
            "storage_time_ms": 42.1
        }
    },
    "error": null,
    "warnings": ["Message size is large (10MB), consider splitting into smaller messages"]
}
```

## Error Handling Patterns

### Try-Catch-Finally Pattern

```python
# Standard error handling pattern for bridge operations
async def safe_bridge_operation(operation_func, *args, **kwargs):
    """Execute bridge operation with standardized error handling."""
    
    operation_name = operation_func.__name__
    start_time = time.time()
    
    try:
        # Pre-operation validation
        await validate_operation_preconditions(operation_name, *args, **kwargs)
        
        # Execute operation
        result = await operation_func(*args, **kwargs)
        
        # Post-operation validation
        await validate_operation_result(result)
        
        operation_time = (time.time() - start_time) * 1000
        
        # Add performance metadata
        if isinstance(result, dict) and "metadata" in result:
            result["metadata"]["operation_time_ms"] = operation_time
        
        return result
        
    except ConfigurationError as e:
        logger.error(f"Configuration error in {operation_name}: {e}")
        return create_error_response(e, {"operation": operation_name})
        
    except InitializationError as e:
        logger.error(f"Initialization error in {operation_name}: {e}")
        return create_error_response(e, {"operation": operation_name})
        
    except StorageError as e:
        logger.error(f"Storage error in {operation_name}: {e}")
        return create_error_response(e, {"operation": operation_name})
        
    except ChatIntegrationError as e:
        logger.error(f"Integration error in {operation_name}: {e}")
        return create_error_response(e, {"operation": operation_name})
        
    except Exception as e:
        logger.exception(f"Unexpected error in {operation_name}: {e}")
        
        # Wrap unexpected errors
        integration_error = ChatIntegrationError(
            f"Unexpected error in {operation_name}: {e}",
            context={"operation": operation_name, "error_type": type(e).__name__},
            suggestions=[
                "This is an unexpected error - please report it",
                "Check system resources and configuration",
                "Try the operation again"
            ],
            cause=e
        )
        
        return create_error_response(integration_error)
    
    finally:
        # Cleanup operations
        await cleanup_operation_resources(operation_name)
```

### Retry Pattern with Exponential Backoff

```python
import asyncio
import random
from typing import Callable, Any, Type

async def retry_with_backoff(
    operation: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple = (StorageError, InitializationError),
    *args, **kwargs) -> Any:
    """
    Execute operation with exponential backoff retry logic.
    
    Args:
        operation: Async function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retry_on: Exception types to retry on
        *args, **kwargs: Arguments for the operation
    
    Returns:
        Operation result
        
    Raises:
        Last exception if all retries exhausted
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await operation(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"Operation succeeded after {attempt} retries")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if we should retry this exception
            if not isinstance(e, retry_on):
                logger.error(f"Non-retryable error: {e}")
                raise e
            
            # Don't retry on last attempt
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exhausted for operation")
                break
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(
                f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    # All retries exhausted
    if isinstance(last_exception, ChatIntegrationError):
        last_exception.add_context("retry_attempts", max_retries)
        last_exception.add_suggestion("All retry attempts exhausted - check system health")
    
    raise last_exception

# Usage example
async def reliable_message_storage(data_layer, user_id, session_id, message):
    """Store message with retry logic."""
    
    return await retry_with_backoff(
        data_layer.store_chat_message,
        max_retries=3,
        retry_on=(StorageError,),
        user_id, session_id, message
    )
```

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = StorageError):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, operation: Callable, *args, **kwargs):
        """Execute operation through circuit breaker."""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise ChatIntegrationError(
                    "Circuit breaker is OPEN - operation blocked",
                    context={
                        "failure_count": self.failure_count,
                        "last_failure": str(self.last_failure_time),
                        "state": self.state.value
                    },
                    suggestions=[
                        f"Circuit breaker will reset in {self.recovery_timeout} seconds",
                        "Check if underlying service has recovered",
                        "Consider using fallback mechanism"
                    ],
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = await operation(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker reset to CLOSED state")
            
            self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
            
        except Exception as e:
            # Unexpected exceptions don't trigger circuit breaker
            raise e
    
    def _record_failure(self):
        """Record a failure and update circuit breaker state."""
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will attempt reset in {self.recovery_timeout} seconds."
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

# Usage example
storage_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=StorageError
)

async def protected_storage_operation(operation, *args, **kwargs):
    """Execute storage operation with circuit breaker protection."""
    return await storage_circuit_breaker.call(operation, *args, **kwargs)
```

## Validation and Input Sanitization

### Input Validation Framework

```python
from typing import Any, Callable, Dict, List, Optional, Union
import re
from dataclasses import dataclass

@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    suggestion: Optional[str] = None

class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_standard_rules()
    
    def _setup_standard_rules(self):
        """Setup standard validation rules."""
        
        # User ID validation
        self.add_rule("user_id", ValidationRule(
            name="user_id_format",
            validator=lambda x: isinstance(x, str) and len(x) > 0 and len(x) <= 100,
            error_message="User ID must be a non-empty string with max 100 characters",
            suggestion="Use alphanumeric characters and underscores"
        ))
        
        self.add_rule("user_id", ValidationRule(
            name="user_id_pattern",
            validator=lambda x: re.match(r'^[a-zA-Z0-9_-]+$', str(x)) is not None,
            error_message="User ID contains invalid characters",
            suggestion="Use only letters, numbers, underscores, and hyphens"
        ))
        
        # Session ID validation
        self.add_rule("session_id", ValidationRule(
            name="session_id_format",
            validator=lambda x: isinstance(x, str) and len(x) > 0,
            error_message="Session ID must be a non-empty string",
            suggestion="Ensure session ID is provided and not empty"
        ))
        
        # Message content validation
        self.add_rule("message_content", ValidationRule(
            name="content_length",
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= 10000,
            error_message="Message content must be 1-10000 characters",
            suggestion="Split large messages into smaller chunks"
        ))
        
        # Role validation
        self.add_rule("message_role", ValidationRule(
            name="valid_role",
            validator=lambda x: x in ["user", "assistant", "system"],
            error_message="Message role must be 'user', 'assistant', or 'system'",
            suggestion="Use one of the valid role values"
        ))
        
        # Performance mode validation
        self.add_rule("performance_mode", ValidationRule(
            name="valid_performance_mode",
            validator=lambda x: x in ["speed", "balanced", "quality"],
            error_message="Performance mode must be 'speed', 'balanced', or 'quality'",
            suggestion="Choose appropriate performance mode for your use case"
        ))
        
        # Cache size validation
        self.add_rule("cache_size_mb", ValidationRule(
            name="cache_size_range",
            validator=lambda x: isinstance(x, (int, float)) and 10 <= x <= 2000,
            error_message="Cache size must be between 10 and 2000 MB",
            suggestion="Use 100-200 MB for typical applications"
        ))
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """Add validation rule for a field."""
        if field_name not in self.rules:
            self.rules[field_name] = []
        self.rules[field_name].append(rule)
    
    def validate_field(self, field_name: str, value: Any) -> List[str]:
        """Validate a single field and return error messages."""
        
        errors = []
        
        if field_name in self.rules:
            for rule in self.rules[field_name]:
                try:
                    if not rule.validator(value):
                        error_msg = f"{field_name}: {rule.error_message}"
                        if rule.suggestion:
                            error_msg += f" ({rule.suggestion})"
                        errors.append(error_msg)
                except Exception as e:
                    errors.append(f"{field_name}: Validation error - {e}")
        
        return errors
    
    def validate_message(self, message: Dict[str, Any]) -> List[str]:
        """Validate a chat message structure."""
        
        errors = []
        
        # Check required fields
        required_fields = ["role", "content"]
        for field in required_fields:
            if field not in message:
                errors.append(f"Missing required field: {field}")
            else:
                # Validate field value
                field_errors = self.validate_field(f"message_{field}", message[field])
                errors.extend(field_errors)
        
        # Validate optional fields
        if "timestamp" in message:
            if not isinstance(message["timestamp"], str):
                errors.append("Timestamp must be a string in ISO format")
        
        if "metadata" in message:
            if not isinstance(message["metadata"], dict):
                errors.append("Metadata must be a dictionary")
        
        return errors
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate bridge configuration."""
        
        errors = []
        
        # Validate individual configuration fields
        config_field_mappings = {
            "performance_mode": "performance_mode",
            "cache_size_mb": "cache_size_mb",
        }
        
        for config_key, validation_key in config_field_mappings.items():
            if config_key in config:
                field_errors = self.validate_field(validation_key, config[config_key])
                errors.extend(field_errors)
        
        # Cross-field validation
        if "cache_size_mb" in config and "performance_mode" in config:
            cache_size = config["cache_size_mb"]
            perf_mode = config["performance_mode"]
            
            if perf_mode == "speed" and cache_size < 100:
                errors.append("Speed mode typically requires cache size >= 100MB")
            elif perf_mode == "quality" and cache_size > 500:
                errors.append("Quality mode may not benefit from cache size > 500MB")
        
        return errors

# Global validator instance
validator = InputValidator()

def validate_input(field_name: str, value: Any):
    """Validate input and raise ConfigurationError if invalid."""
    
    errors = validator.validate_field(field_name, value)
    
    if errors:
        raise create_validation_error(field_name, value, "; ".join(errors))

# Usage in bridge operations
async def store_chat_message_with_validation(data_layer, user_id: str, session_id: str, message: Dict[str, Any]):
    """Store chat message with comprehensive validation."""
    
    try:
        # Validate inputs
        validate_input("user_id", user_id)
        validate_input("session_id", session_id)
        
        # Validate message structure
        message_errors = validator.validate_message(message)
        if message_errors:
            raise ConfigurationError(
                "Invalid message format",
                context={"message": message, "errors": message_errors},
                suggestions=["Fix message format according to API documentation"]
            )
        
        # Execute operation
        return await data_layer.store_chat_message(user_id, session_id, message)
        
    except ConfigurationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        # Wrap other errors
        raise StorageError(
            f"Failed to store message: {e}",
            operation="store_chat_message",
            context={"user_id": user_id, "session_id": session_id}
        ) from e
```

## Error Logging and Monitoring

### Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    """Structured logging for bridge system errors."""
    
    def __init__(self, logger_name: str = "ff_chat_bridge"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logging format."""
        
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_error(self, 
                  error: Exception,
                  operation: str,
                  context: Optional[Dict[str, Any]] = None,
                  user_id: Optional[str] = None):
        """Log error with structured format."""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "user_id": user_id
        }
        
        # Add bridge-specific error information
        if isinstance(error, ChatIntegrationError):
            log_entry.update({
                "error_code": error.error_code,
                "error_context": error.context,
                "error_suggestions": error.suggestions
            })
        
        self.logger.error(json.dumps(log_entry))
    
    def log_performance_warning(self,
                               operation: str,
                               duration_ms: float,
                               threshold_ms: float,
                               context: Optional[Dict[str, Any]] = None):
        """Log performance warnings."""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "WARNING",
            "type": "PERFORMANCE",
            "operation": operation,
            "duration_ms": duration_ms,
            "threshold_ms": threshold_ms,
            "context": context or {}
        }
        
        self.logger.warning(json.dumps(log_entry))

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logs."""
    
    def format(self, record):
        # If message is already JSON, return as-is
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Format as structured log
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            return json.dumps(log_entry)

# Global structured logger
structured_logger = StructuredLogger()

def log_bridge_error(error: Exception, operation: str, **kwargs):
    """Log bridge error with context."""
    structured_logger.log_error(error, operation, **kwargs)

def log_performance_issue(operation: str, duration_ms: float, **kwargs):
    """Log performance issues."""
    structured_logger.log_performance_warning(operation, duration_ms, 100, **kwargs)
```

### Error Metrics and Alerting

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio

class ErrorMetricsCollector:
    """Collect and analyze error metrics for alerting."""
    
    def __init__(self, window_minutes: int = 10):
        self.window_minutes = window_minutes
        self.error_counts = defaultdict(lambda: deque())
        self.error_rates = {}
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "error_count": 10,   # 10 errors in window
            "consecutive_errors": 5  # 5 consecutive errors
        }
    
    def record_error(self, error_type: str, operation: str):
        """Record an error occurrence."""
        
        key = f"{operation}:{error_type}"
        current_time = datetime.now()
        
        self.error_counts[key].append(current_time)
        self._cleanup_old_errors(key)
        
        # Check for alerts
        self._check_alerts(key, operation, error_type)
    
    def record_success(self, operation: str):
        """Record successful operation (for error rate calculation)."""
        
        success_key = f"{operation}:success"
        current_time = datetime.now()
        
        self.error_counts[success_key].append(current_time)
        self._cleanup_old_errors(success_key)
    
    def _cleanup_old_errors(self, key: str):
        """Remove errors outside the time window."""
        
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        
        while (self.error_counts[key] and 
               self.error_counts[key][0] < cutoff_time):
            self.error_counts[key].popleft()
    
    def _check_alerts(self, error_key: str, operation: str, error_type: str):
        """Check if any alert thresholds are exceeded."""
        
        current_error_count = len(self.error_counts[error_key])
        
        # Check error count threshold
        if current_error_count >= self.alert_thresholds["error_count"]:
            self._trigger_alert(
                "HIGH_ERROR_COUNT",
                f"High error count for {operation}: {current_error_count} {error_type} errors in {self.window_minutes} minutes"
            )
        
        # Check error rate
        success_key = f"{operation}:success"
        success_count = len(self.error_counts[success_key])
        total_operations = current_error_count + success_count
        
        if total_operations > 0:
            error_rate = current_error_count / total_operations
            
            if error_rate >= self.alert_thresholds["error_rate"]:
                self._trigger_alert(
                    "HIGH_ERROR_RATE",
                    f"High error rate for {operation}: {error_rate:.2%} ({current_error_count}/{total_operations})"
                )
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert (implement based on your alerting system)."""
        
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "message": message,
            "severity": "WARNING" if "RATE" in alert_type else "ERROR"
        }
        
        # Log the alert
        structured_logger.logger.warning(f"ALERT: {json.dumps(alert_data)}")
        
        # Here you would integrate with your alerting system
        # Examples: send to PagerDuty, Slack, email, etc.
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get current error summary."""
        
        summary = {
            "window_minutes": self.window_minutes,
            "timestamp": datetime.now().isoformat(),
            "operations": {}
        }
        
        # Group by operation
        operations = defaultdict(lambda: {"errors": {}, "successes": 0, "total": 0})
        
        for key, errors in self.error_counts.items():
            parts = key.split(":")
            operation = parts[0]
            error_type = parts[1]
            
            if error_type == "success":
                operations[operation]["successes"] = len(errors)
            else:
                operations[operation]["errors"][error_type] = len(errors)
            
            operations[operation]["total"] += len(errors)
        
        # Calculate error rates
        for operation, data in operations.items():
            total_errors = sum(data["errors"].values())
            total_ops = data["total"]
            
            if total_ops > 0:
                data["error_rate"] = total_errors / total_ops
            else:
                data["error_rate"] = 0
        
        summary["operations"] = dict(operations)
        return summary

# Global error metrics collector
error_metrics = ErrorMetricsCollector()

def record_operation_error(operation: str, error: Exception):
    """Record an operation error for metrics."""
    error_type = type(error).__name__
    error_metrics.record_error(error_type, operation)

def record_operation_success(operation: str):
    """Record successful operation for metrics."""
    error_metrics.record_success(operation)
```

## Testing Error Handling

### Error Handling Test Framework

```python
import pytest
from unittest.mock import AsyncMock, patch
from ff_chat_integration import (
    FFChatAppBridge, ConfigurationError, InitializationError, StorageError
)

class ErrorHandlingTestSuite:
    """Comprehensive error handling tests."""
    
    @pytest.mark.asyncio
    async def test_configuration_error_handling(self):
        """Test configuration error handling."""
        
        # Test invalid performance mode
        with pytest.raises(ConfigurationError) as exc_info:
            await FFChatAppBridge.create_for_chat_app(
                "./test_data",
                {"performance_mode": "invalid_mode"}
            )
        
        error = exc_info.value
        assert error.error_code == "CONFIG_ERROR"
        assert "performance_mode" in error.context
        assert len(error.suggestions) > 0
        assert "speed" in str(error.suggestions)  # Should suggest valid values
    
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test initialization error handling."""
        
        # Test with non-existent path
        with pytest.raises(InitializationError) as exc_info:
            await FFChatAppBridge.create_for_chat_app("/nonexistent/readonly/path")
        
        error = exc_info.value
        assert error.error_code == "INIT_ERROR"
        assert "storage_path" in error.context or "component" in error.context
        assert any("storage path" in s.lower() for s in error.suggestions)
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self):
        """Test storage error handling."""
        
        bridge = await FFChatAppBridge.create_for_chat_app("./test_data")
        data_layer = bridge.get_data_layer()
        
        # Mock storage failure
        with patch.object(data_layer.storage, 'add_message', side_effect=Exception("Storage failure")):
            result = await data_layer.store_chat_message(
                "test_user", "test_session",
                {"role": "user", "content": "test"}
            )
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"] is not None
        
        await bridge.close()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism for transient errors."""
        
        bridge = await FFChatAppBridge.create_for_chat_app("./test_data")
        data_layer = bridge.get_data_layer()
        
        # Setup user and session
        await data_layer.storage.create_user("retry_user", {"name": "Retry User"})
        session_id = await data_layer.storage.create_session("retry_user", "Retry Session")
        
        # Mock intermittent failures
        call_count = 0
        
        async def failing_operation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:  # Fail first 2 attempts
                raise StorageError("Transient storage failure", operation="test_operation")
            else:
                # Succeed on 3rd attempt
                return {"success": True, "data": {"message_id": "test_123"}}
        
        # Test retry with backoff
        result = await retry_with_backoff(
            failing_operation,
            max_retries=3,
            base_delay=0.1,  # Short delay for testing
            retry_on=(StorageError,)
        )
        
        assert result["success"] is True
        assert call_count == 3  # Should have retried 2 times
        
        await bridge.close()
    
    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """Test standardized error response format."""
        
        # Create a configuration error
        error = ConfigurationError(
            "Test configuration error",
            config_field="test_field",
            config_value="invalid_value"
        )
        
        response = create_error_response(error)
        
        # Verify response structure
        assert "success" in response
        assert "data" in response
        assert "metadata" in response
        assert "error" in response
        assert "warnings" in response
        
        assert response["success"] is False
        assert response["error"] is not None
        assert "context" in response["metadata"]
        assert "test_field" in response["metadata"]["context"]
    
    def test_error_context_and_suggestions(self):
        """Test error context and suggestions."""
        
        error = StorageError(
            "File write failed",
            operation="write",
            storage_path="/readonly/path"
        )
        
        error.add_context("attempted_file", "test.json")
        error.add_suggestion("Check file permissions")
        
        assert error.context["operation"] == "write"
        assert error.context["storage_path"] == "/readonly/path"
        assert error.context["attempted_file"] == "test.json"
        assert "Check file permissions" in error.suggestions
        
        # Test serialization
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "STORAGE_ERROR"
        assert error_dict["context"]["attempted_file"] == "test.json"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_error_handling(self):
        """Test circuit breaker error handling."""
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        failing_operation = AsyncMock(side_effect=StorageError("Service down"))
        
        # First failure
        with pytest.raises(StorageError):
            await breaker.call(failing_operation)
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(StorageError):
            await breaker.call(failing_operation)
        
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.OPEN
        
        # Third attempt - should be blocked by circuit breaker
        with pytest.raises(ChatIntegrationError) as exc_info:
            await breaker.call(failing_operation)
        
        error = exc_info.value
        assert error.error_code == "CIRCUIT_BREAKER_OPEN"
        assert "circuit breaker is OPEN" in str(error).lower()

# Run error handling tests
def run_error_handling_tests():
    """Run comprehensive error handling tests."""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_error_handling_tests()
```

## Error Handling Best Practices

### 1. Comprehensive Error Context

Always provide sufficient context for debugging:

```python
# ❌ Poor error handling
raise Exception("Operation failed")

# ✅ Good error handling  
raise StorageError(
    "Failed to write message to storage",
    operation="store_chat_message",
    storage_path="/path/to/storage",
    context={
        "user_id": user_id,
        "session_id": session_id,
        "message_size": len(message_content),
        "available_space_mb": available_space
    },
    suggestions=[
        "Check available disk space",
        "Verify write permissions",
        "Ensure storage path exists"
    ]
)
```

### 2. Graceful Degradation

Implement fallback mechanisms where possible:

```python
async def store_message_with_fallback(data_layer, user_id, session_id, message):
    """Store message with fallback mechanisms."""
    
    try:
        # Try primary storage
        return await data_layer.store_chat_message(user_id, session_id, message)
        
    except StorageError as e:
        logger.warning(f"Primary storage failed: {e}")
        
        try:
            # Try fallback storage (e.g., temporary file)
            return await store_to_fallback_storage(user_id, session_id, message)
            
        except Exception as fallback_error:
            logger.error(f"Fallback storage also failed: {fallback_error}")
            
            # Return error with both failures
            return create_error_response(
                ChatIntegrationError(
                    "Both primary and fallback storage failed",
                    context={
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error)
                    },
                    suggestions=[
                        "Check system health",
                        "Verify storage configuration",
                        "Contact system administrator"
                    ]
                )
            )
```

### 3. Proper Error Propagation

Handle errors at appropriate levels:

```python
# Data layer - wrap lower-level errors
async def store_chat_message(self, user_id, session_id, message):
    try:
        # Low-level storage operation
        result = await self._raw_storage_operation(user_id, session_id, message)
        return create_success_response(result)
        
    except FileNotFoundError as e:
        raise StorageError(
            "Storage file not found",
            operation="store_chat_message",
            storage_path=self.config.storage_path
        ) from e
    except PermissionError as e:
        raise StorageError(
            "Insufficient permissions for storage operation",
            operation="store_chat_message",
            storage_path=self.config.storage_path
        ) from e
    except Exception as e:
        # Unexpected errors
        raise StorageError(
            f"Unexpected error during message storage: {e}",
            operation="store_chat_message"
        ) from e

# Application layer - handle data layer errors
async def handle_user_message(bridge, user_id, session_id, message_content):
    try:
        data_layer = bridge.get_data_layer()
        message = {"role": "user", "content": message_content}
        
        result = await data_layer.store_chat_message(user_id, session_id, message)
        
        if result["success"]:
            return {"status": "success", "message": "Message stored successfully"}
        else:
            return {"status": "error", "message": result["error"]}
            
    except StorageError as e:
        # Log error and return user-friendly message
        log_bridge_error(e, "handle_user_message", user_id=user_id)
        
        return {
            "status": "error",
            "message": "Unable to store message at this time",
            "details": str(e),
            "suggestions": e.suggestions
        }
    except Exception as e:
        # Unexpected errors
        log_bridge_error(e, "handle_user_message", user_id=user_id)
        
        return {
            "status": "error", 
            "message": "An unexpected error occurred",
            "details": "Please try again or contact support"
        }
```

These error handling standards ensure the Chat Application Bridge System provides robust, informative, and actionable error handling that enables developers to quickly diagnose and resolve issues while maintaining system reliability and user experience.