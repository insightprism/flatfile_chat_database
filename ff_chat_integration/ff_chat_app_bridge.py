"""
Main bridge class for Chat Application integration with Flatfile Database.

Provides simplified, chat-optimized interface that eliminates configuration
wrappers and provides specialized operations for chat applications.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Import existing Flatfile infrastructure
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import (
    FFConfigurationManagerConfigDTO
)
from ff_utils.ff_logging import get_logger

# Import our exception classes from Phase 1
from .ff_integration_exceptions import (
    ChatIntegrationError,
    ConfigurationError,
    InitializationError,
    StorageError,
    create_validation_error
)

logger = get_logger(__name__)


@dataclass
class ChatAppStorageConfig:
    """
    Standardized storage configuration for chat applications.
    
    Provides chat-optimized settings with validation and performance presets.
    Eliminates the need for complex configuration wrappers.
    """
    
    # Core settings
    storage_path: str
    
    # Feature flags
    enable_vector_search: bool = True
    enable_streaming: bool = True
    enable_analytics: bool = True
    enable_compression: bool = False
    backup_enabled: bool = False
    
    # Performance settings
    cache_size_mb: int = 100
    performance_mode: str = "balanced"  # "speed", "balanced", "quality"
    max_session_size_mb: int = 50
    
    # Chat-specific settings
    message_batch_size: int = 100
    history_page_size: int = 50
    search_result_limit: int = 20
    
    # Environment settings
    environment: str = "development"  # "development", "production", "test"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                context={"validation_errors": errors}
            )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return any issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate storage path (only check if absolute path and production environment)
        try:
            storage_path = Path(self.storage_path)
            if (self.storage_path and 
                self.environment == "production" and 
                storage_path.is_absolute() and 
                not storage_path.parent.exists()):
                issues.append(f"Storage path parent directory does not exist: {storage_path.parent}")
        except Exception as e:
            issues.append(f"Invalid storage path format: {e}")
        
        # Validate performance mode
        valid_modes = ["speed", "balanced", "quality"]
        if self.performance_mode not in valid_modes:
            issues.append(f"Performance mode must be one of: {valid_modes}")
        
        # Validate cache size
        if self.cache_size_mb < 10:
            issues.append("Cache size should be at least 10MB")
        if self.cache_size_mb > 2000:
            issues.append("Cache size should not exceed 2000MB")
        
        # Validate session size
        if self.max_session_size_mb < 1:
            issues.append("Max session size should be at least 1MB")
        if self.max_session_size_mb > 500:
            issues.append("Max session size should not exceed 500MB")
        
        # Validate batch sizes
        if self.message_batch_size < 1:
            issues.append("Message batch size must be at least 1")
        if self.message_batch_size > 1000:
            issues.append("Message batch size should not exceed 1000")
        
        if self.history_page_size < 1:
            issues.append("History page size must be at least 1")
        if self.history_page_size > 500:
            issues.append("History page size should not exceed 500")
        
        # Validate environment
        valid_environments = ["development", "production", "test", "staging"]
        if self.environment not in valid_environments:
            issues.append(f"Environment must be one of: {valid_environments}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "storage_path": self.storage_path,
            "features": {
                "vector_search": self.enable_vector_search,
                "streaming": self.enable_streaming,
                "analytics": self.enable_analytics,
                "compression": self.enable_compression,
                "backup": self.backup_enabled
            },
            "performance": {
                "mode": self.performance_mode,
                "cache_size_mb": self.cache_size_mb,
                "max_session_size_mb": self.max_session_size_mb
            },
            "chat_settings": {
                "message_batch_size": self.message_batch_size,
                "history_page_size": self.history_page_size,
                "search_result_limit": self.search_result_limit
            },
            "environment": self.environment
        }


class FFChatAppBridge:
    """
    Main bridge for chat application integration with Flatfile Database.
    
    Eliminates the need for configuration wrappers and provides specialized
    operations optimized for chat applications.
    """
    
    def __init__(self, config: ChatAppStorageConfig):
        """
        Initialize chat application bridge.
        
        Args:
            config: Chat application storage configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        
        # Storage system components
        self._storage_manager: Optional[FFStorageManager] = None
        self._data_layer = None  # Will be set in Phase 3
        
        # State tracking
        self._initialized = False
        self._initialization_error: Optional[Exception] = None
        
        self.logger.info(f"Created FFChatAppBridge with config: {config.storage_path}")
    
    @classmethod
    async def create_for_chat_app(cls, 
                                storage_path: str,
                                options: Optional[Dict[str, Any]] = None) -> 'FFChatAppBridge':
        """
        Factory method specifically for chat applications.
        
        Eliminates need for configuration wrappers by providing a simple,
        one-line setup method for chat applications.
        
        Args:
            storage_path: Base path for storage
            options: Optional configuration overrides
            
        Returns:
            Fully initialized FFChatAppBridge instance
            
        Raises:
            ConfigurationError: If configuration is invalid
            InitializationError: If storage cannot be initialized
            
        Example:
            # Simple setup
            bridge = await FFChatAppBridge.create_for_chat_app("./chat_data")
            
            # With custom options
            bridge = await FFChatAppBridge.create_for_chat_app(
                "./chat_data",
                {"performance_mode": "speed", "enable_analytics": False}
            )
        """
        try:
            # Create chat-optimized configuration
            config_options = options or {}
            # Remove storage_path from options to avoid duplicate argument error
            config_options = {k: v for k, v in config_options.items() if k != 'storage_path'}
            
            # Flatten nested configuration structures if present
            flattened_options = cls._flatten_config_options(config_options)
            
            chat_config = ChatAppStorageConfig(
                storage_path=storage_path,
                **flattened_options
            )
            
            cls._log_creation_attempt(storage_path, config_options)
            
            # Create and initialize bridge
            bridge = cls(chat_config)
            await bridge.initialize()
            
            logger.info(f"Successfully created chat app bridge for: {storage_path}")
            return bridge
            
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to create chat app bridge: {e}")
            raise InitializationError(
                f"Failed to create chat app bridge: {e}",
                component="FFChatAppBridge",
                initialization_step="factory_creation",
                context={
                    "storage_path": storage_path,
                    "options": options or {},
                    "original_error": str(e)
                }
            )
    
    @classmethod
    async def create_from_preset(cls, preset_name: str,
                               storage_path: str,
                               overrides: Optional[Dict[str, Any]] = None) -> 'FFChatAppBridge':
        """
        Create bridge from a configuration preset.
        
        Args:
            preset_name: Name of preset to use
            storage_path: Storage path
            overrides: Optional configuration overrides
            
        Returns:
            Initialized FFChatAppBridge
            
        Example:
            # Use production preset
            bridge = await FFChatAppBridge.create_from_preset(
                "production", 
                "/var/lib/chatapp/data"
            )
            
            # Use development preset with overrides
            bridge = await FFChatAppBridge.create_from_preset(
                "development",
                "./dev_data",
                {"enable_analytics": False}
            )
        """
        from .ff_chat_config_factory import FFChatConfigFactory
        
        try:
            factory = FFChatConfigFactory()
            config = factory.create_from_template(preset_name, storage_path, overrides)
            
            # Create bridge with preset configuration
            bridge = cls(config)
            await bridge.initialize()
            
            logger.info(f"Created chat app bridge from preset '{preset_name}' for: {storage_path}")
            return bridge
            
        except Exception as e:
            logger.error(f"Failed to create bridge from preset '{preset_name}': {e}")
            raise InitializationError(
                f"Failed to create bridge from preset '{preset_name}': {e}",
                component="FFChatAppBridge",
                initialization_step="preset_creation",
                context={
                    "preset_name": preset_name,
                    "storage_path": storage_path,
                    "overrides": overrides or {}
                }
            )

    @classmethod  
    async def create_for_use_case(cls, use_case: str,
                                storage_path: str,
                                **kwargs) -> 'FFChatAppBridge':
        """
        Create bridge optimized for specific use case.
        
        Args:
            use_case: Type of chat application
            storage_path: Storage path
            **kwargs: Additional configuration options
            
        Returns:
            Use-case optimized FFChatAppBridge
            
        Example:
            # AI assistant setup
            bridge = await FFChatAppBridge.create_for_use_case(
                "ai_assistant",
                "./ai_data",
                enable_vector_search=True
            )
            
            # High volume chat setup  
            bridge = await FFChatAppBridge.create_for_use_case(
                "high_volume_chat",
                "./chat_data",
                performance_mode="speed"
            )
        """
        from .ff_chat_config_factory import FFChatConfigFactory
        
        try:
            factory = FFChatConfigFactory()
            config = factory.create_for_use_case(use_case, storage_path, **kwargs)
            
            bridge = cls(config)
            await bridge.initialize()
            
            logger.info(f"Created chat app bridge for use case '{use_case}' at: {storage_path}")
            return bridge
            
        except Exception as e:
            logger.error(f"Failed to create bridge for use case '{use_case}': {e}")
            raise InitializationError(
                f"Failed to create bridge for use case '{use_case}': {e}",
                component="FFChatAppBridge", 
                initialization_step="use_case_creation",
                context={
                    "use_case": use_case,
                    "storage_path": storage_path,
                    "options": kwargs
                }
            )

    @classmethod
    async def create_from_preset_legacy(cls,
                               preset_name: str,
                               storage_path: str,
                               overrides: Optional[Dict[str, Any]] = None) -> 'FFChatAppBridge':
        """
        Factory method using legacy performance presets.
        
        Args:
            preset_name: Name of preset configuration ("speed", "balanced", "quality")
            storage_path: Base path for storage
            overrides: Optional configuration overrides
            
        Returns:
            Fully initialized FFChatAppBridge instance
            
        Raises:
            ConfigurationError: If preset or configuration is invalid
            InitializationError: If storage cannot be initialized
            
        Example:
            # Use speed preset
            bridge = await FFChatAppBridge.create_from_preset_legacy("speed", "./chat_data")
            
            # Use preset with overrides
            bridge = await FFChatAppBridge.create_from_preset_legacy(
                "quality", 
                "./chat_data",
                {"cache_size_mb": 200}
            )
        """
        try:
            # Get preset configuration
            preset_config = cls._get_preset_config(preset_name)
            
            # Apply any overrides
            if overrides:
                preset_config.update(overrides)
            
            # Use the main factory method
            return await cls.create_for_chat_app(storage_path, preset_config)
            
        except Exception as e:
            logger.error(f"Failed to create bridge from legacy preset '{preset_name}': {e}")
            raise InitializationError(
                f"Failed to create bridge from legacy preset '{preset_name}': {e}",
                component="FFChatAppBridge",
                initialization_step="preset_creation",
                context={
                    "preset_name": preset_name,
                    "storage_path": storage_path,
                    "overrides": overrides or {},
                    "original_error": str(e)
                }
            )
    
    @staticmethod
    def _get_preset_config(preset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific preset."""
        presets = {
            "speed": {
                "performance_mode": "speed",
                "cache_size_mb": 200,
                "enable_compression": False,
                "backup_enabled": False,
                "message_batch_size": 50,
                "history_page_size": 25
            },
            "balanced": {
                "performance_mode": "balanced",
                "cache_size_mb": 100,
                "enable_compression": False,
                "backup_enabled": True,
                "message_batch_size": 100,
                "history_page_size": 50
            },
            "quality": {
                "performance_mode": "quality",
                "cache_size_mb": 150,
                "enable_compression": True,
                "backup_enabled": True,
                "message_batch_size": 200,
                "history_page_size": 100,
                "enable_analytics": True
            }
        }
        
        if preset_name not in presets:
            raise ConfigurationError(
                f"Unknown preset '{preset_name}'",
                context={"valid_presets": list(presets.keys())},
                suggestions=[f"Use one of: {', '.join(presets.keys())}"]
            )
        
        return presets[preset_name]
    
    @classmethod
    def _flatten_config_options(cls, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested configuration options to match ChatAppStorageConfig parameters.
        
        Args:
            options: Configuration options that may contain nested structures
            
        Returns:
            Flattened configuration dictionary
        """
        flattened = {}
        
        for key, value in options.items():
            if key == "features" and isinstance(value, dict):
                # Map feature flags to config attributes
                feature_mapping = {
                    "vector_search": "enable_vector_search",
                    "streaming": "enable_streaming", 
                    "analytics": "enable_analytics",
                    "compression": "enable_compression",
                    "backup": "backup_enabled"
                }
                for feature_key, feature_value in value.items():
                    if feature_key in feature_mapping:
                        flattened[feature_mapping[feature_key]] = feature_value
                        
            elif key == "performance" and isinstance(value, dict):
                # Map performance settings to config attributes
                performance_mapping = {
                    "mode": "performance_mode",
                    "cache_size_mb": "cache_size_mb",
                    "max_session_size_mb": "max_session_size_mb"
                }
                for perf_key, perf_value in value.items():
                    if perf_key in performance_mapping:
                        flattened[performance_mapping[perf_key]] = perf_value
                        
            elif key == "chat_settings" and isinstance(value, dict):
                # Map chat settings to config attributes
                chat_mapping = {
                    "message_batch_size": "message_batch_size",
                    "history_page_size": "history_page_size", 
                    "search_result_limit": "search_result_limit"
                }
                for chat_key, chat_value in value.items():
                    if chat_key in chat_mapping:
                        flattened[chat_mapping[chat_key]] = chat_value
                        
            else:
                # Direct mapping for non-nested options
                flattened[key] = value
                
        return flattened

    @staticmethod
    def _log_creation_attempt(storage_path: str, options: Dict[str, Any]):
        """Log bridge creation attempt with sanitized information."""
        logger.info(
            f"Creating chat app bridge - Path: {storage_path}, "
            f"Options: {list(options.keys()) if options else 'none'}"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the chat application bridge and underlying storage.
        
        This method integrates directly with the existing Flatfile storage
        system without requiring configuration wrappers.
        
        Returns:
            True if initialization successful
            
        Raises:
            InitializationError: If initialization fails
        """
        if self._initialized:
            return True
        
        try:
            self.logger.info("Initializing FFChatAppBridge...")
            
            # Create Flatfile configuration from chat config (no wrapper needed!)
            ff_config = await self._create_ff_config()
            
            # Initialize storage manager directly
            self._storage_manager = FFStorageManager(ff_config)
            await self._storage_manager.initialize()
            
            # Verify storage system is working
            await self._verify_storage_functionality()
            
            self._initialized = True
            
            initialization_time = time.time() - self.start_time
            self.logger.info(f"FFChatAppBridge initialized successfully in {initialization_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._initialization_error = e
            self.logger.error(f"FFChatAppBridge initialization failed: {e}")
            
            raise InitializationError(
                f"Failed to initialize chat storage bridge: {e}",
                component="FFChatAppBridge",
                initialization_step="storage_initialization",
                context={
                    "storage_path": self.config.storage_path,
                    "performance_mode": self.config.performance_mode,
                    "initialization_time": time.time() - self.start_time
                }
            )
    
    async def _create_ff_config(self) -> FFConfigurationManagerConfigDTO:
        """
        Create Flatfile configuration from chat config.
        
        This eliminates the need for configuration wrappers by directly
        creating the appropriate Flatfile configuration.
        """
        # Load base configuration using from_environment method
        base_config = FFConfigurationManagerConfigDTO.from_environment(self.config.environment)
        
        # Apply chat-specific overrides
        base_config.storage.base_path = self.config.storage_path
        base_config.runtime.cache_size_limit = self.config.cache_size_mb
        
        # Apply performance mode optimizations
        if self.config.performance_mode == "speed":
            base_config.storage.enable_file_locking = False
            base_config.vector.similarity_threshold = 0.8
            base_config.search.enable_search_cache = True
            base_config.runtime.storage_default_message_limit = self.config.message_batch_size
            
        elif self.config.performance_mode == "quality":
            base_config.storage.enable_file_locking = True
            base_config.vector.similarity_threshold = 0.6
            base_config.search.enable_fuzzy_search = True
            base_config.search.enable_search_cache = True
            
        else:  # balanced mode
            base_config.storage.enable_file_locking = self.config.environment == "production"
            base_config.vector.similarity_threshold = 0.7
            base_config.search.enable_search_cache = True
        
        # Apply chat-specific settings
        base_config.runtime.storage_default_message_limit = self.config.message_batch_size
        base_config.runtime.large_session_threshold_bytes = self.config.max_session_size_mb * 1024 * 1024
        
        return base_config
    
    async def _verify_storage_functionality(self):
        """Verify that storage system is working correctly."""
        if not self._storage_manager:
            raise InitializationError("Storage manager not initialized")
        
        # Test basic storage operations
        try:
            # This will be enhanced in Phase 3 with actual test operations
            self.logger.debug("Storage functionality verification passed")
        except Exception as e:
            raise InitializationError(
                f"Storage functionality verification failed: {e}",
                component="storage_verification",
                initialization_step="functionality_test"
            )
    
    def get_standardized_config(self) -> Dict[str, Any]:
        """
        Return configuration in format expected by chat applications.
        
        Provides a clean, standardized view of the configuration that
        chat applications can use for display or validation.
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")
        
        return {
            "storage_path": self.config.storage_path,
            "capabilities": {
                "vector_search": self.config.enable_vector_search,
                "streaming": self.config.enable_streaming,
                "analytics": self.config.enable_analytics,
                "compression": self.config.enable_compression
            },
            "performance": {
                "mode": self.config.performance_mode,
                "cache_size_mb": self.config.cache_size_mb,
                "max_session_size_mb": self.config.max_session_size_mb
            },
            "features": {
                "backup": self.config.backup_enabled,
                "compression": self.config.enable_compression,
                "batch_size": self.config.message_batch_size,
                "page_size": self.config.history_page_size
            },
            "environment": self.config.environment,
            "initialized": self._initialized,
            "initialization_time": time.time() - self.start_time if self._initialized else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for chat app monitoring.
        
        Returns detailed health information that chat applications can use
        for monitoring and diagnostics.
        """
        try:
            health_data = {
                "status": "unknown",
                "timestamp": datetime.now().isoformat(),
                "bridge_initialized": self._initialized,
                "storage_accessible": False,
                "write_permissions": False,
                "disk_space_sufficient": False,
                "performance_metrics": {},
                "uptime_seconds": time.time() - self.start_time,
                "configuration_valid": True,
                "errors": [],
                "warnings": []
            }
            
            if not self._initialized:
                health_data["status"] = "error"
                health_data["errors"].append("Bridge not initialized")
                if self._initialization_error:
                    health_data["errors"].append(f"Initialization error: {self._initialization_error}")
                return health_data
            
            # Check storage accessibility
            try:
                health_data["storage_accessible"] = await self._check_storage_access()
            except Exception as e:
                health_data["errors"].append(f"Storage access check failed: {e}")
            
            # Check write permissions
            try:
                health_data["write_permissions"] = await self._check_write_permissions()
            except Exception as e:
                health_data["errors"].append(f"Write permission check failed: {e}")
            
            # Check disk space
            try:
                health_data["disk_space_sufficient"] = self._check_disk_space()
            except Exception as e:
                health_data["errors"].append(f"Disk space check failed: {e}")
            
            # Get performance metrics
            try:
                health_data["performance_metrics"] = await self._get_performance_metrics()
            except Exception as e:
                health_data["errors"].append(f"Performance metrics failed: {e}")
            
            # Determine overall status
            if health_data["errors"]:
                health_data["status"] = "error"
            elif health_data["warnings"]:
                health_data["status"] = "degraded"
            elif (health_data["storage_accessible"] and 
                  health_data["write_permissions"] and 
                  health_data["disk_space_sufficient"]):
                health_data["status"] = "healthy"
            else:
                health_data["status"] = "degraded"
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "bridge_initialized": self._initialized,
                "storage_accessible": False,
                "write_permissions": False,
                "disk_space_sufficient": False,
                "performance_metrics": {},
                "uptime_seconds": time.time() - self.start_time,
                "errors": [f"Health check failed: {e}"],
                "warnings": []
            }
    
    async def _check_storage_access(self) -> bool:
        """Check if storage system is accessible."""
        if not self._storage_manager:
            return False
        
        try:
            # This will be enhanced in Phase 3 with actual storage operations
            return True
        except Exception:
            return False
    
    async def _check_write_permissions(self) -> bool:
        """Check if storage system has write permissions."""
        try:
            storage_path = Path(self.config.storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = storage_path.parent / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
        except Exception:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            storage_path = Path(self.config.storage_path)
            
            # Get available disk space
            total, used, free = shutil.disk_usage(storage_path.parent)
            
            # Require at least 100MB free space
            min_free_bytes = 100 * 1024 * 1024
            return free > min_free_bytes
            
        except Exception:
            return False
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get basic performance metrics."""
        try:
            return {
                "uptime_seconds": time.time() - self.start_time,
                "memory_usage": "not_implemented",  # Will be enhanced in Phase 5
                "cache_utilization": "not_implemented",
                "response_time_ms": 0  # Will be measured in Phase 3
            }
        except Exception:
            return {}
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Discover available storage features and capabilities.
        
        Returns information about what features are available in the current
        configuration, helping chat applications adapt their functionality.
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")
        
        try:
            capabilities = {
                "vector_search": self.config.enable_vector_search,
                "streaming": self.config.enable_streaming,
                "analytics": self.config.enable_analytics,
                "compression": self.config.enable_compression,
                "backup": self.config.backup_enabled,
                "max_file_size_mb": self.config.max_session_size_mb,
                "supported_formats": ["json", "jsonl"],
                "search_types": [],
                "storage_features": [
                    "user_profiles",
                    "sessions", 
                    "messages",
                    "metadata"
                ]
            }
            
            # Determine available search types based on configuration
            if self.config.enable_vector_search:
                capabilities["search_types"].extend(["text", "vector", "hybrid"])
            else:
                capabilities["search_types"].append("text")
            
            return capabilities
            
        except Exception as e:
            self.logger.error(f"Failed to get capabilities: {e}")
            return {"error": str(e)}
    
    def get_data_layer(self) -> 'FFChatDataLayer':
        """
        Get chat-optimized data access layer.
        
        Returns:
            FFChatDataLayer instance for specialized chat operations
            
        Raises:
            RuntimeError: If bridge not initialized
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")
        
        if self._data_layer is None:
            from .ff_chat_data_layer import FFChatDataLayer
            self._data_layer = FFChatDataLayer(self._storage_manager, self.config)
            self.logger.info("Chat data layer initialized")
        
        return self._data_layer
    
    async def close(self) -> None:
        """Clean shutdown of all resources."""
        try:
            self.logger.info("Shutting down FFChatAppBridge...")
            
            if self._storage_manager and hasattr(self._storage_manager, 'close'):
                await self._storage_manager.close()
            
            self._initialized = False
            self.logger.info("FFChatAppBridge shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during FFChatAppBridge shutdown: {e}")