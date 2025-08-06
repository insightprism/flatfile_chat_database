"""
Exception classes for Chat Application Bridge System.

Provides standardized error handling with context information and actionable
error messages for chat application developers.
"""

from typing import Dict, Any, Optional, List


class ChatIntegrationError(Exception):
    """
    Base exception for all chat integration issues.
    
    Provides structured error information with context for debugging
    and actionable guidance for developers.
    """
    
    def __init__(self, 
                 message: str,
                 context: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None,
                 error_code: Optional[str] = None):
        """
        Initialize chat integration error.
        
        Args:
            message: Human-readable error description
            context: Additional context information for debugging
            suggestions: List of actionable suggestions to resolve the issue
            error_code: Machine-readable error code for categorization
        """
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []
        self.error_code = error_code
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API responses."""
        return {
            "error": str(self),
            "error_code": self.error_code,
            "context": self.context,
            "suggestions": self.suggestions
        }
    
    def __str__(self) -> str:
        """Enhanced string representation with context."""
        base_message = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_message} (Context: {context_str})"
        return base_message


class ConfigurationError(ChatIntegrationError):
    """
    Configuration-related errors in chat integration.
    
    Raised when configuration validation fails or configuration
    settings are incompatible with chat application requirements.
    """
    
    def __init__(self, message: str, 
                 config_field: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            config_field: Name of problematic configuration field
            config_value: Value that caused the error
            **kwargs: Additional arguments passed to parent
        """
        context = kwargs.get('context', {})
        if config_field:
            context['config_field'] = config_field
        if config_value is not None:
            context['config_value'] = str(config_value)
            
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONFIG_ERROR')
        
        super().__init__(message, **kwargs)


class InitializationError(ChatIntegrationError):
    """
    Errors during chat integration initialization.
    
    Raised when the bridge system cannot be properly initialized,
    typically due to storage setup issues or dependency problems.
    """
    
    def __init__(self, message: str,
                 component: Optional[str] = None,
                 initialization_step: Optional[str] = None,
                 **kwargs):
        """
        Initialize initialization error.
        
        Args:
            message: Error description
            component: Name of component that failed to initialize
            initialization_step: Specific step where initialization failed
            **kwargs: Additional arguments passed to parent
        """
        context = kwargs.get('context', {})
        if component:
            context['component'] = component
        if initialization_step:
            context['initialization_step'] = initialization_step
            
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'INIT_ERROR')
        
        # Add common initialization suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Verify storage path exists and is writable",
            "Check that all required dependencies are available",
            "Ensure configuration values are valid"
        ])
        kwargs['suggestions'] = list(set(suggestions))  # Remove duplicates
        
        super().__init__(message, **kwargs)


class StorageError(ChatIntegrationError):
    """
    Storage operation failures in chat integration.
    
    Raised when storage operations fail, including file system issues,
    permission problems, or data corruption.
    """
    
    def __init__(self, message: str,
                 operation: Optional[str] = None,
                 storage_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize storage error.
        
        Args:
            message: Error description
            operation: Storage operation that failed (e.g., 'write', 'read')
            storage_path: File or directory path involved in the operation
            **kwargs: Additional arguments passed to parent
        """
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if storage_path:
            context['storage_path'] = storage_path
            
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'STORAGE_ERROR')
        
        # Add common storage suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check disk space availability",
            "Verify file system permissions",
            "Ensure storage path is accessible"
        ])
        kwargs['suggestions'] = list(set(suggestions))
        
        super().__init__(message, **kwargs)


class SearchError(ChatIntegrationError):
    """
    Search operation failures in chat integration.
    
    Raised when search operations fail, including index issues,
    query problems, or search service unavailability.
    """
    
    def __init__(self, message: str,
                 search_type: Optional[str] = None,
                 query: Optional[str] = None,
                 **kwargs):
        """
        Initialize search error.
        
        Args:
            message: Error description
            search_type: Type of search that failed (e.g., 'text', 'vector', 'hybrid')
            query: Search query that caused the error
            **kwargs: Additional arguments passed to parent
        """
        context = kwargs.get('context', {})
        if search_type:
            context['search_type'] = search_type
        if query:
            context['query'] = query[:100] + "..." if len(query) > 100 else query
            
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'SEARCH_ERROR')
        
        # Add common search suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Verify search indices are built and accessible",
            "Check if search service is running",
            "Validate search query format"
        ])
        kwargs['suggestions'] = list(set(suggestions))
        
        super().__init__(message, **kwargs)


class PerformanceError(ChatIntegrationError):
    """
    Performance-related errors in chat integration.
    
    Raised when operations exceed acceptable performance thresholds
    or when performance monitoring detects issues.
    """
    
    def __init__(self, message: str,
                 operation: Optional[str] = None,
                 response_time_ms: Optional[float] = None,
                 threshold_ms: Optional[float] = None,
                 **kwargs):
        """
        Initialize performance error.
        
        Args:
            message: Error description
            operation: Operation that exceeded performance threshold
            response_time_ms: Actual response time in milliseconds
            threshold_ms: Performance threshold that was exceeded
            **kwargs: Additional arguments passed to parent
        """
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if response_time_ms is not None:
            context['response_time_ms'] = response_time_ms
        if threshold_ms is not None:
            context['threshold_ms'] = threshold_ms
            
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'PERFORMANCE_ERROR')
        
        # Add performance optimization suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Consider enabling caching for frequently accessed data",
            "Review system resource usage (CPU, memory, disk)",
            "Optimize configuration for better performance"
        ])
        kwargs['suggestions'] = list(set(suggestions))
        
        super().__init__(message, **kwargs)


# Utility functions for error handling

def create_validation_error(field_name: str, field_value: Any, 
                          expected: str) -> ConfigurationError:
    """
    Create a standardized validation error.
    
    Args:
        field_name: Name of the field that failed validation
        field_value: Value that failed validation
        expected: Description of expected value format
        
    Returns:
        ConfigurationError with standardized formatting
    """
    return ConfigurationError(
        f"Invalid value for {field_name}: expected {expected}",
        config_field=field_name,
        config_value=field_value,
        suggestions=[
            f"Update {field_name} to match expected format: {expected}",
            "Review configuration documentation for correct values"
        ]
    )


def create_missing_dependency_error(dependency: str, 
                                  component: str) -> InitializationError:
    """
    Create a standardized missing dependency error.
    
    Args:
        dependency: Name of missing dependency
        component: Component that requires the dependency
        
    Returns:
        InitializationError with dependency information
    """
    return InitializationError(
        f"Missing required dependency '{dependency}' for {component}",
        component=component,
        initialization_step="dependency_check",
        suggestions=[
            f"Install required dependency: {dependency}",
            "Check system requirements documentation",
            "Verify all optional components are properly configured"
        ]
    )


def wrap_storage_operation_error(original_error: Exception, 
                               operation: str,
                               storage_path: str) -> StorageError:
    """
    Wrap a generic storage error with chat integration context.
    
    Args:
        original_error: Original exception that occurred
        operation: Storage operation that was being performed
        storage_path: Path where the operation failed
        
    Returns:
        StorageError with enhanced context
    """
    return StorageError(
        f"Storage operation '{operation}' failed: {str(original_error)}",
        operation=operation,
        storage_path=storage_path,
        context={
            "original_error_type": type(original_error).__name__,
            "original_error_message": str(original_error)
        }
    )