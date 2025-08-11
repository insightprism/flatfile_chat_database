# Phase 4: Configuration Factory and Presets

## Overview

Phase 4 implements configuration factory methods and preset management that simplify setup for chat applications. This phase provides environment-specific presets, performance optimization templates, and configuration validation utilities that make it even easier for developers to get optimal configurations for their specific use cases.

**Estimated Time**: 3-4 days  
**Dependencies**: Phases 1-3 completed  
**Risk Level**: Low (primarily configuration management)

## Objectives

1. **Create Configuration Factory**: Simple factory methods for common configurations
2. **Implement Environment Presets**: Pre-configured settings for dev/staging/production
3. **Add Performance Templates**: Optimization presets for different performance needs
4. **Provide Configuration Validation**: Enhanced validation with recommendations
5. **Enable Migration Utilities**: Help migrate from wrapper-based configurations

## Current Configuration Context

### Existing Flatfile Configuration System

The current system we're building upon:

```python
# From ff_class_configs/ff_configuration_manager_config.py
def load_config(environment: Optional[str] = None) -> FFConfigurationManagerConfigDTO:
    """Load configuration with environment-specific overrides"""

# Current environments supported
environments = ["development", "production", "test"]
```

### Current Chat Application Setup (Complex)

What chat applications currently need to do:

```python
# Current complex setup that we're simplifying
config = load_config("production")
config.storage.base_path = "/path/to/chat/data"
config.runtime.cache_size_limit = 200
config.storage.enable_file_locking = True
config.vector.similarity_threshold = 0.7
# ... many more manual configurations

# Then create wrapper
wrapper = ConfigWrapper(config)  # 18+ lines of wrapper code
storage = FFStorageManager(wrapper)
```

### Target Simple Setup (After Phase 4)

What we want to enable:

```python
# Simple preset-based setup
bridge = await FFChatAppBridge.create_from_preset(
    "production_optimized",
    storage_path="/path/to/chat/data"
)

# Or custom factory
config = create_chat_config_for_production(
    storage_path="/path/to/chat/data",
    performance_level="high"
)
bridge = await FFChatAppBridge.create_for_chat_app(
    config.storage_path,
    config.to_options_dict()
)
```

## Implementation Details

### Step 1: Create Configuration Factory

Create `ff_chat_integration/ff_chat_config_factory.py`:

```python
"""
Configuration factory and preset management for Chat Application Bridge.

Provides simplified configuration creation with environment-specific presets,
performance optimization templates, and migration utilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import os

# Import existing Flatfile configuration
from ff_class_configs.ff_configuration_manager_config import (
    FFConfigurationManagerConfigDTO, load_config
)
from ff_utils.ff_logging import get_logger

# Import our bridge components
from .ff_chat_app_bridge import ChatAppStorageConfig
from .ff_integration_exceptions import (
    ConfigurationError, create_validation_error
)

logger = get_logger(__name__)


@dataclass
class ChatConfigTemplate:
    """Template for generating chat configurations."""
    name: str
    description: str
    base_config: ChatAppStorageConfig
    performance_optimizations: Dict[str, Any]
    use_cases: List[str]
    recommended_for: List[str]


class FFChatConfigFactory:
    """
    Factory for creating optimized chat application configurations.
    
    Provides preset configurations, performance templates, and validation
    utilities to simplify chat application setup.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._templates: Dict[str, ChatConfigTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in configuration templates."""
        
        # Development template
        self._templates["development"] = ChatConfigTemplate(
            name="development",
            description="Optimized for development with debugging features",
            base_config=ChatAppStorageConfig(
                storage_path="./dev_data",
                enable_vector_search=True,
                enable_streaming=False,  # Simpler for debugging
                enable_analytics=True,
                cache_size_mb=50,
                performance_mode="balanced",
                environment="development"
            ),
            performance_optimizations={
                "enable_file_locking": False,
                "validation_enabled": True,
                "debug_logging": True,
                "cache_ttl": 60  # Short for development
            },
            use_cases=["basic_chat", "testing", "development"],
            recommended_for=["local_development", "debugging", "testing"]
        )
        
        # Production template
        self._templates["production"] = ChatConfigTemplate(
            name="production",
            description="Production-ready with full features and optimization",
            base_config=ChatAppStorageConfig(
                storage_path="/var/lib/chatapp/data",
                enable_vector_search=True,
                enable_streaming=True,
                enable_analytics=True,
                enable_compression=True,
                backup_enabled=True,
                cache_size_mb=200,
                performance_mode="balanced",
                max_session_size_mb=100,
                environment="production"
            ),
            performance_optimizations={
                "enable_file_locking": True,
                "validation_enabled": False,
                "cache_ttl": 300,
                "batch_write_enabled": True
            },
            use_cases=["all_chat_types", "high_volume", "enterprise"],
            recommended_for=["production_deployment", "high_traffic", "enterprise"]
        )
        
        # High performance template
        self._templates["high_performance"] = ChatConfigTemplate(
            name="high_performance",
            description="Maximum performance with minimal features",
            base_config=ChatAppStorageConfig(
                storage_path="./data",
                enable_vector_search=False,
                enable_streaming=True,
                enable_analytics=False,
                cache_size_mb=500,
                performance_mode="speed",
                message_batch_size=200,
                history_page_size=100,
                environment="production"
            ),
            performance_optimizations={
                "enable_file_locking": False,
                "validation_enabled": False,
                "cache_ttl": 600,
                "memory_optimization": True
            },
            use_cases=["high_throughput", "real_time_chat", "gaming"],
            recommended_for=["high_volume_chat", "real_time_applications", "performance_critical"]
        )
        
        # Feature-rich template
        self._templates["feature_rich"] = ChatConfigTemplate(
            name="feature_rich",
            description="All features enabled for full functionality",
            base_config=ChatAppStorageConfig(
                storage_path="./data",
                enable_vector_search=True,
                enable_streaming=True,
                enable_analytics=True,
                enable_compression=True,
                backup_enabled=True,
                cache_size_mb=150,
                performance_mode="quality",
                environment="production"
            ),
            performance_optimizations={
                "enable_file_locking": True,
                "validation_enabled": True,
                "advanced_search": True,
                "full_indexing": True
            },
            use_cases=["ai_assistant", "knowledge_base", "research_chat"],
            recommended_for=["ai_applications", "research_tools", "advanced_chat_features"]
        )
        
        # Lightweight template
        self._templates["lightweight"] = ChatConfigTemplate(
            name="lightweight",
            description="Minimal resource usage for simple applications",
            base_config=ChatAppStorageConfig(
                storage_path="./data",
                enable_vector_search=False,
                enable_streaming=False,
                enable_analytics=False,
                cache_size_mb=25,
                performance_mode="speed",
                max_session_size_mb=10,
                message_batch_size=50,
                environment="development"
            ),
            performance_optimizations={
                "enable_file_locking": False,
                "minimal_metadata": True,
                "reduced_logging": True
            },
            use_cases=["simple_chat", "embedded_systems", "low_resource"],
            recommended_for=["simple_applications", "resource_constrained", "embedded_systems"]
        )
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available configuration templates.
        
        Returns:
            Dictionary of template information
        """
        return {
            name: {
                "description": template.description,
                "performance_mode": template.base_config.performance_mode,
                "features": {
                    "vector_search": template.base_config.enable_vector_search,
                    "streaming": template.base_config.enable_streaming,
                    "analytics": template.base_config.enable_analytics,
                    "compression": template.base_config.enable_compression
                },
                "use_cases": template.use_cases,
                "recommended_for": template.recommended_for
            }
            for name, template in self._templates.items()
        }
    
    def create_from_template(self, template_name: str, 
                           storage_path: str,
                           overrides: Optional[Dict[str, Any]] = None) -> ChatAppStorageConfig:
        """
        Create configuration from a template.
        
        Args:
            template_name: Name of template to use
            storage_path: Storage path for the configuration
            overrides: Optional overrides for template settings
            
        Returns:
            ChatAppStorageConfig based on template
            
        Raises:
            ConfigurationError: If template not found or invalid overrides
        """
        if template_name not in self._templates:
            available_templates = list(self._templates.keys())
            raise ConfigurationError(
                f"Template '{template_name}' not found",
                context={"available_templates": available_templates},
                suggestions=[f"Use one of: {', '.join(available_templates)}"]
            )
        
        template = self._templates[template_name]
        
        # Start with template base config
        config_dict = {
            "storage_path": storage_path,
            "enable_vector_search": template.base_config.enable_vector_search,
            "enable_streaming": template.base_config.enable_streaming,
            "enable_analytics": template.base_config.enable_analytics,
            "enable_compression": template.base_config.enable_compression,
            "backup_enabled": template.base_config.backup_enabled,
            "cache_size_mb": template.base_config.cache_size_mb,
            "performance_mode": template.base_config.performance_mode,
            "max_session_size_mb": template.base_config.max_session_size_mb,
            "message_batch_size": template.base_config.message_batch_size,
            "history_page_size": template.base_config.history_page_size,
            "search_result_limit": template.base_config.search_result_limit,
            "environment": template.base_config.environment
        }
        
        # Apply overrides
        if overrides:
            config_dict.update(overrides)
        
        try:
            config = ChatAppStorageConfig(**config_dict)
            self.logger.info(f"Created configuration from template '{template_name}' for {storage_path}")
            return config
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create configuration from template '{template_name}': {e}",
                context={"template": template_name, "overrides": overrides}
            )
    
    def create_for_environment(self, environment: str,
                             storage_path: str,
                             performance_level: str = "balanced") -> ChatAppStorageConfig:
        """
        Create configuration optimized for specific environment.
        
        Args:
            environment: Target environment ("development", "staging", "production")
            storage_path: Storage path
            performance_level: Performance optimization level ("speed", "balanced", "quality")
            
        Returns:
            Environment-optimized ChatAppStorageConfig
        """
        # Map environments to templates
        template_mapping = {
            "development": "development",
            "staging": "production",  # Use production template with modifications
            "production": "production"
        }
        
        base_template = template_mapping.get(environment, "development")
        
        # Environment-specific overrides
        overrides = {
            "environment": environment,
            "performance_mode": performance_level
        }
        
        if environment == "development":
            overrides.update({
                "cache_size_mb": 50,
                "enable_compression": False,
                "backup_enabled": False
            })
        elif environment == "staging":
            overrides.update({
                "cache_size_mb": 100,
                "enable_compression": True,
                "backup_enabled": True
            })
        elif environment == "production":
            overrides.update({
                "cache_size_mb": 200,
                "enable_compression": True,
                "backup_enabled": True,
                "max_session_size_mb": 100
            })
        
        return self.create_from_template(base_template, storage_path, overrides)
    
    def create_for_use_case(self, use_case: str,
                          storage_path: str,
                          **kwargs) -> ChatAppStorageConfig:
        """
        Create configuration optimized for specific use case.
        
        Args:
            use_case: Type of chat application
            storage_path: Storage path
            **kwargs: Additional configuration options
            
        Returns:
            Use-case optimized ChatAppStorageConfig
        """
        use_case_templates = {
            "simple_chat": "lightweight",
            "ai_assistant": "feature_rich",
            "high_volume_chat": "high_performance",
            "enterprise_chat": "production",
            "development_chat": "development",
            "research_chat": "feature_rich",
            "gaming_chat": "high_performance",
            "support_chat": "production"
        }
        
        template = use_case_templates.get(use_case, "development")
        
        # Use case specific overrides
        overrides = kwargs.copy()
        
        if use_case in ["ai_assistant", "research_chat"]:
            overrides.setdefault("enable_vector_search", True)
            overrides.setdefault("enable_analytics", True)
        elif use_case in ["high_volume_chat", "gaming_chat"]:
            overrides.setdefault("performance_mode", "speed")
            overrides.setdefault("enable_analytics", False)
        elif use_case == "simple_chat":
            overrides.setdefault("enable_vector_search", False)
            overrides.setdefault("enable_streaming", False)
        
        return self.create_from_template(template, storage_path, overrides)
    
    def validate_and_optimize(self, config: ChatAppStorageConfig) -> Dict[str, Any]:
        """
        Validate configuration and provide optimization recommendations.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results with recommendations
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "optimization_score": 0,
            "estimated_performance": "unknown"
        }
        
        # Basic validation (uses existing config validation)
        validation_errors = config.validate()
        if validation_errors:
            results["valid"] = False
            results["errors"] = validation_errors
        
        # Performance analysis
        performance_score = 0
        
        # Cache size optimization
        if config.cache_size_mb < 50:
            results["warnings"].append("Cache size is quite small - may impact performance")
            results["recommendations"].append("Consider increasing cache_size_mb to at least 50MB")
        elif config.cache_size_mb > 500:
            results["warnings"].append("Cache size is very large - may use excessive memory")
            results["recommendations"].append("Consider reducing cache_size_mb unless you have high memory availability")
        else:
            performance_score += 20
        
        # Performance mode analysis
        if config.performance_mode == "speed":
            performance_score += 30
            if config.enable_vector_search:
                results["warnings"].append("Vector search enabled in speed mode - may reduce performance")
                results["recommendations"].append("Consider disabling vector search for maximum speed")
        elif config.performance_mode == "quality":
            performance_score += 10
            if not config.enable_vector_search:
                results["recommendations"].append("Consider enabling vector search for better quality results")
        else:  # balanced
            performance_score += 20
        
        # Feature analysis
        if config.enable_compression and config.performance_mode == "speed":
            results["warnings"].append("Compression enabled in speed mode - adds processing overhead")
            results["recommendations"].append("Consider disabling compression for maximum speed")
        
        if not config.enable_analytics and config.environment == "production":
            results["recommendations"].append("Consider enabling analytics for production monitoring")
        
        # Storage path analysis
        storage_path = Path(config.storage_path)
        if not storage_path.is_absolute() and config.environment == "production":
            results["warnings"].append("Relative storage path in production environment")
            results["recommendations"].append("Use absolute paths for production deployments")
        
        # Environment consistency
        if config.environment == "production":
            if not config.backup_enabled:
                results["recommendations"].append("Enable backups for production environment")
            if config.cache_size_mb < 100:
                results["recommendations"].append("Increase cache size for production workloads")
            performance_score += 10
        
        results["optimization_score"] = min(performance_score, 100)
        
        # Estimate performance category
        if performance_score >= 80:
            results["estimated_performance"] = "high"
        elif performance_score >= 60:
            results["estimated_performance"] = "good"
        elif performance_score >= 40:
            results["estimated_performance"] = "moderate"
        else:
            results["estimated_performance"] = "needs_optimization"
        
        return results
    
    def migrate_from_wrapper_config(self, wrapper_config: Dict[str, Any]) -> ChatAppStorageConfig:
        """
        Migrate from old wrapper-based configuration to bridge configuration.
        
        Args:
            wrapper_config: Dictionary containing wrapper configuration
            
        Returns:
            Equivalent ChatAppStorageConfig
        """
        try:
            # Extract storage path
            storage_path = wrapper_config.get("base_path", "./data")
            
            # Map wrapper config to bridge config
            bridge_config = ChatAppStorageConfig(
                storage_path=storage_path,
                enable_vector_search=wrapper_config.get("enable_vector_search", True),
                enable_streaming=wrapper_config.get("enable_streaming", True),
                enable_analytics=wrapper_config.get("enable_analytics", True),
                enable_compression=wrapper_config.get("enable_compression", False),
                backup_enabled=wrapper_config.get("backup_enabled", False),
                cache_size_mb=wrapper_config.get("cache_size_limit", 100),
                performance_mode=wrapper_config.get("performance_mode", "balanced"),
                max_session_size_mb=wrapper_config.get("max_session_size_mb", 50),
                environment=wrapper_config.get("environment", "development")
            )
            
            self.logger.info("Successfully migrated wrapper configuration to bridge configuration")
            return bridge_config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to migrate wrapper configuration: {e}",
                context={"wrapper_config": wrapper_config}
            )


# Convenience factory functions

def create_chat_config_for_development(storage_path: str = "./dev_data",
                                     **overrides) -> ChatAppStorageConfig:
    """
    Create development-optimized chat configuration.
    
    Args:
        storage_path: Storage path for development
        **overrides: Configuration overrides
        
    Returns:
        Development-optimized configuration
    """
    factory = FFChatConfigFactory()
    return factory.create_for_environment("development", storage_path, **overrides)


def create_chat_config_for_production(storage_path: str,
                                    performance_level: str = "balanced",
                                    **overrides) -> ChatAppStorageConfig:
    """
    Create production-optimized chat configuration.
    
    Args:
        storage_path: Storage path for production
        performance_level: Performance optimization level
        **overrides: Configuration overrides
        
    Returns:
        Production-optimized configuration
    """
    factory = FFChatConfigFactory()
    config = factory.create_for_environment("production", storage_path, performance_level)
    
    # Apply any overrides
    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        # Reconstruct with overrides
        return ChatAppStorageConfig(
            storage_path=config_dict.get("storage_path", storage_path),
            **{k: v for k, v in config_dict.items() if k != "storage_path"}
        )
    
    return config


def get_chat_app_presets() -> Dict[str, ChatAppStorageConfig]:
    """
    Get predefined chat app configuration presets.
    
    Returns:
        Dictionary of preset configurations
    """
    factory = FFChatConfigFactory()
    
    presets = {}
    for template_name in ["development", "production", "high_performance", "feature_rich", "lightweight"]:
        try:
            # Use a placeholder path - will be overridden when used
            presets[template_name] = factory.create_from_template(template_name, "./data")
        except Exception as e:
            logger.warning(f"Failed to create preset {template_name}: {e}")
    
    return presets


def validate_chat_app_config(config: ChatAppStorageConfig) -> List[str]:
    """
    Validate configuration for chat app compatibility.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation issues
    """
    factory = FFChatConfigFactory()
    results = factory.validate_and_optimize(config)
    return results["errors"] + results["warnings"]


def optimize_chat_config(config: ChatAppStorageConfig) -> Dict[str, Any]:
    """
    Analyze and optimize chat configuration.
    
    Args:
        config: Configuration to optimize
        
    Returns:
        Optimization analysis and recommendations
    """
    factory = FFChatConfigFactory()
    return factory.validate_and_optimize(config)
```

### Step 2: Add Preset Support to Bridge

Update `ff_chat_integration/ff_chat_app_bridge.py` to add preset support:

```python
# Add to FFChatAppBridge class

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
```

### Step 3: Update Module Exports

Update `ff_chat_integration/__init__.py`:

```python
# Add Phase 4 exports
from .ff_chat_config_factory import (
    FFChatConfigFactory,
    create_chat_config_for_development,
    create_chat_config_for_production, 
    get_chat_app_presets,
    validate_chat_app_config,
    optimize_chat_config
)

# Update __all__
__all__.extend([
    "FFChatConfigFactory",
    "create_chat_config_for_development",
    "create_chat_config_for_production",
    "get_chat_app_presets", 
    "validate_chat_app_config",
    "optimize_chat_config"
])
```

## Validation and Testing

### Step 4: Create Phase 4 Validation Script

Create validation script for configuration factory:

```python
# Save as: test_phase4_validation.py in the project root
"""
Phase 4 validation script for Chat Application Bridge System.

Validates configuration factory, presets, and template management.
"""

import asyncio
import sys
import tempfile
import traceback
from pathlib import Path

async def test_factory_creation():
    """Test configuration factory creation."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        templates = factory.list_templates()
        
        assert len(templates) >= 5  # Should have at least 5 templates
        assert "development" in templates
        assert "production" in templates
        assert "high_performance" in templates
        print(f"✓ Factory created with {len(templates)} templates")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_template_creation():
    """Test creating configurations from templates."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Test development template
        dev_config = factory.create_from_template("development", "./test_data")
        assert dev_config.storage_path == "./test_data"
        assert dev_config.environment == "development"
        assert dev_config.performance_mode == "balanced"
        print("✓ Development template creation successful")
        
        # Test production template with overrides
        prod_config = factory.create_from_template(
            "production", 
            "/prod/data",
            {"cache_size_mb": 300, "enable_compression": False}
        )
        assert prod_config.cache_size_mb == 300
        assert prod_config.enable_compression is False
        print("✓ Production template with overrides successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Template creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_convenience_functions():
    """Test convenience factory functions."""
    try:
        from ff_chat_integration import (
            create_chat_config_for_development,
            create_chat_config_for_production,
            get_chat_app_presets
        )
        
        # Test development config creation
        dev_config = create_chat_config_for_development("./dev_test")
        assert dev_config.environment == "development"
        assert dev_config.storage_path == "./dev_test"
        print("✓ Development config convenience function works")
        
        # Test production config creation  
        prod_config = create_chat_config_for_production(
            "/prod/test",
            performance_level="speed"
        )
        assert prod_config.performance_mode == "speed"
        assert prod_config.environment == "production"
        print("✓ Production config convenience function works")
        
        # Test presets
        presets = get_chat_app_presets()
        assert len(presets) >= 3
        assert "development" in presets
        print(f"✓ Presets function returned {len(presets)} presets")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        traceback.print_exc()
        return False

async def test_validation_and_optimization():
    """Test configuration validation and optimization."""
    try:
        from ff_chat_integration import FFChatConfigFactory, ChatAppStorageConfig
        
        factory = FFChatConfigFactory()
        
        # Test good configuration
        good_config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced",
            cache_size_mb=100
        )
        
        results = factory.validate_and_optimize(good_config)
        assert results["valid"] is True
        assert "optimization_score" in results
        print("✓ Good configuration validation successful")
        
        # Test problematic configuration
        bad_config = ChatAppStorageConfig(
            storage_path="",  # Empty path
            performance_mode="invalid_mode",  # Invalid mode
            cache_size_mb=5  # Too small
        )
        
        try:
            # This should raise ConfigurationError due to validation
            factory.validate_and_optimize(bad_config)
            print("✗ Should have raised ConfigurationError")
            return False
        except Exception:
            print("✓ Bad configuration properly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        traceback.print_exc()
        return False

async def test_bridge_preset_integration():
    """Test bridge integration with presets."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "preset_test")
            
            # Test preset creation
            bridge = await FFChatAppBridge.create_from_preset(
                "development", 
                storage_path
            )
            
            assert bridge._initialized
            config = bridge.get_standardized_config()
            assert config["environment"] == "development"
            print("✓ Bridge preset integration successful")
            
            await bridge.close()
            
            # Test use case creation
            bridge2 = await FFChatAppBridge.create_for_use_case(
                "simple_chat",
                storage_path
            )
            
            assert bridge2._initialized
            print("✓ Bridge use case integration successful")
            
            await bridge2.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge preset integration test failed: {e}")
        traceback.print_exc()
        return False

async def test_migration_utility():
    """Test configuration migration from wrapper format."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Simulate old wrapper configuration
        wrapper_config = {
            "base_path": "./old_wrapper_data",
            "cache_size_limit": 150,
            "enable_vector_search": True,
            "enable_compression": True,
            "performance_mode": "speed",
            "environment": "production"
        }
        
        # Migrate to bridge configuration
        bridge_config = factory.migrate_from_wrapper_config(wrapper_config)
        
        assert bridge_config.storage_path == "./old_wrapper_data"
        assert bridge_config.cache_size_mb == 150
        assert bridge_config.enable_vector_search is True
        assert bridge_config.enable_compression is True
        assert bridge_config.performance_mode == "speed"
        print("✓ Configuration migration successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration utility test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 4 validation tests."""
    print("Phase 4 Validation - Configuration Factory and Presets")
    print("=" * 55)
    
    tests = [
        ("Factory Creation", test_factory_creation),
        ("Template Creation", test_template_creation),
        ("Convenience Functions", test_convenience_functions),
        ("Validation & Optimization", test_validation_and_optimization),
        ("Bridge Preset Integration", test_bridge_preset_integration),
        ("Migration Utility", test_migration_utility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if await test_func():
                passed += 1
            else:
                print(f"Test {test_name} failed!")
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
    
    print(f"\n" + "=" * 55)
    print(f"Phase 4 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 4 implementation is ready for Phase 5!")
        return True
    else:
        print("✗ Phase 4 needs fixes before proceeding to Phase 5") 
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
```

## Integration Points

### With Existing Configuration System
- **Compatible**: Works with existing `load_config()` and `FFConfigurationManagerConfigDTO`
- **Non-Breaking**: Existing configuration methods continue to work unchanged
- **Enhanced**: Provides simplified interface while leveraging existing infrastructure

### With Phases 1-3
- **Exception Handling**: Uses standardized exceptions from Phase 1
- **Bridge Integration**: Seamlessly integrates with `FFChatAppBridge` from Phase 2
- **Data Layer**: Optimizes configurations for data layer performance from Phase 3

### For Phase 5
- **Template Foundation**: Provides template system that health monitoring can analyze
- **Optimization Framework**: Establishes optimization scoring that can be enhanced

## Success Criteria

### Technical Validation
1. **Template System**: Multiple configuration templates available and working
2. **Factory Methods**: Simple factory methods create valid configurations
3. **Preset Integration**: Bridge can be created from presets and use cases
4. **Validation System**: Configuration validation provides actionable feedback
5. **Migration Support**: Can migrate from wrapper-based configurations

### Developer Experience Validation
1. **Simplified Setup**: One-line preset-based setup works
2. **Clear Templates**: Template descriptions help developers choose appropriate configuration
3. **Optimization Guidance**: Validation provides helpful optimization recommendations
4. **Migration Path**: Easy migration from existing wrapper configurations

## Phase Completion Checklist

- [ ] `FFChatConfigFactory` implemented with template system
- [ ] Environment-specific and use-case-specific factory methods
- [ ] Bridge integration with preset and use-case creation methods
- [ ] Configuration validation and optimization analysis
- [ ] Migration utilities for wrapper-based configurations
- [ ] Convenience functions for common configuration patterns
- [ ] Module exports updated for Phase 4 components
- [ ] Validation script passes all tests
- [ ] Templates cover common chat application scenarios

## Next Steps

After Phase 4 completion:
1. **Template Testing**: Validate templates work well for different chat application types
2. **Performance Validation**: Ensure optimized configurations deliver expected performance
3. **Migration Testing**: Test migration utilities with real wrapper configurations
4. **Proceed to Phase 5**: Health monitoring and comprehensive diagnostics

This phase significantly simplifies the configuration experience for chat applications, providing preset-based setup that eliminates the need for complex manual configuration while still allowing customization for specific needs.