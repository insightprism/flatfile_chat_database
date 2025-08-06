"""
Chat Application Bridge System for Flatfile Database.

This module provides a simplified, chat-optimized interface for integrating
chat applications with Flatfile Database, eliminating the need for complex
configuration wrappers and providing specialized operations for chat use cases.

Key Components:
- FFChatAppBridge: Main bridge class with factory methods
- FFChatDataLayer: Chat-optimized data access operations
- ChatAppStorageConfig: Standardized configuration for chat applications
- Health monitoring and diagnostic capabilities

Example Usage:
    # Simple setup - no wrappers needed
    bridge = await FFChatAppBridge.create_for_chat_app(
        storage_path="./chat_data",
        options={"performance_mode": "balanced"}
    )
    
    # Get chat-optimized data layer
    data_layer = bridge.get_data_layer()
    
    # Store messages with optimized format
    result = await data_layer.store_chat_message(
        user_id="user123",
        session_id="session456", 
        message={
            "role": "user",
            "content": "Hello, how can you help me?",
            "timestamp": "2025-01-01T12:00:00Z"
        }
    )
"""

# Version information - Phase 6 Complete
__version__ = "1.0.0"
__author__ = "Flatfile Database Team"
__description__ = "Chat Application Bridge System for Flatfile Database - Production Ready"

# Public API exports will be added as components are implemented

# Exception classes - available immediately
from .ff_integration_exceptions import (
    # Base exception
    ChatIntegrationError,
    
    # Specific exception types
    ConfigurationError,
    InitializationError,
    StorageError,
    SearchError,
    PerformanceError,
    
    # Utility functions
    create_validation_error,
    create_missing_dependency_error,
    wrap_storage_operation_error
)

# Phase 2 exports - Bridge implementation
from .ff_chat_app_bridge import FFChatAppBridge, ChatAppStorageConfig

# Phase 3 exports - Data layer implementation
from .ff_chat_data_layer import FFChatDataLayer

# Phase 4 exports - Configuration factory and presets
from .ff_chat_config_factory import (
    FFChatConfigFactory,
    ChatConfigTemplate,
    create_chat_config_for_development,
    create_chat_config_for_production,
    create_chat_config_for_testing,
    get_chat_app_presets,
    validate_chat_app_config,
    optimize_chat_config,
    get_recommended_config_for_use_case
)

# Phase 5 exports - Health monitoring and diagnostics
from .ff_integration_health_monitor import (
    FFIntegrationHealthMonitor,
    HealthCheckResult,
    PerformanceMetric,
    create_health_monitor,
    quick_health_check,
    diagnose_bridge_issues
)

# Phase 6 exports - Testing utilities (for external validation)
try:
    from .tests import BridgeTestHelper, PerformanceTester
    _test_utilities_available = True
except ImportError:
    _test_utilities_available = False

# Module-level exports for Phases 1, 2, 3, 4, and 5
__all__ = [
    # Exception classes
    "ChatIntegrationError",
    "ConfigurationError", 
    "InitializationError",
    "StorageError",
    "SearchError",
    "PerformanceError",
    
    # Utility functions
    "create_validation_error",
    "create_missing_dependency_error",
    "wrap_storage_operation_error",
    
    # Phase 2: Bridge components
    "FFChatAppBridge",
    "ChatAppStorageConfig",
    
    # Phase 3: Data layer components
    "FFChatDataLayer",
    
    # Phase 4: Configuration factory and presets
    "FFChatConfigFactory",
    "ChatConfigTemplate",
    "create_chat_config_for_development",
    "create_chat_config_for_production",
    "create_chat_config_for_testing",
    "get_chat_app_presets",
    "validate_chat_app_config",
    "optimize_chat_config",
    "get_recommended_config_for_use_case",
    
    # Phase 5: Health monitoring and diagnostics
    "FFIntegrationHealthMonitor",
    "HealthCheckResult",
    "PerformanceMetric",
    "create_health_monitor",
    "quick_health_check",
    "diagnose_bridge_issues"
]

# Add Phase 6 test utilities if available
if _test_utilities_available:
    __all__.extend([
        "BridgeTestHelper",
        "PerformanceTester"
    ])

# Always add validation function
__all__.append("validate_bridge_system")

# Placeholder for future components (will be uncommented as they're implemented)
# from .ff_chat_data_layer import FFChatDataLayer
# from .ff_chat_config_factory import create_chat_app_config, get_chat_app_presets
# from .ff_integration_health_monitor import FFIntegrationHealthMonitor

# Future exports (will be added in subsequent phases)
# __all__.extend([
#     "FFChatAppBridge",
#     "ChatAppStorageConfig", 
#     "FFChatDataLayer",
#     "create_chat_app_config",
#     "get_chat_app_presets",
#     "FFIntegrationHealthMonitor"
# ])


def get_version() -> str:
    """Get the current version of the chat integration bridge."""
    return __version__


def get_module_info() -> dict:
    """Get comprehensive module information."""
    return {
        "name": "ff_chat_integration",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components_available": len(__all__),
        "test_utilities_available": _test_utilities_available,
        "exception_classes": [
            "ChatIntegrationError",
            "ConfigurationError", 
            "InitializationError",
            "StorageError",
            "SearchError",
            "PerformanceError"
        ],
        "phase_6_complete": True,
        "production_ready": True
    }


# Phase 6 convenience function for system validation
async def validate_bridge_system() -> bool:
    """
    Validate the entire bridge system is working correctly.
    
    Returns:
        True if all validation tests pass
    """
    try:
        if _test_utilities_available:
            from .tests.test_final_validation import run_all_validation_tests
            return await run_all_validation_tests()
        else:
            # Fallback basic validation
            bridge = await FFChatAppBridge.create_for_chat_app("./validation_test")
            success = bridge._initialized
            await bridge.close()
            return success
    except Exception:
        return False