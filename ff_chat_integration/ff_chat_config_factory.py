"""
Configuration factory and preset management for Chat Application Bridge.

Provides simplified configuration creation with environment-specific presets,
performance optimization templates, and migration utilities.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

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
        self._template_dir = Path(__file__).parent / "templates"
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
                storage_path="./prod_data",  # Use relative path for template
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

        # Testing template
        self._templates["testing"] = ChatConfigTemplate(
            name="testing",
            description="Optimized for test environments with fast cleanup",
            base_config=ChatAppStorageConfig(
                storage_path="./test_data",
                enable_vector_search=True,
                enable_streaming=False,
                enable_analytics=False,
                cache_size_mb=30,
                performance_mode="balanced",
                max_session_size_mb=5,
                message_batch_size=25,
                environment="test"
            ),
            performance_optimizations={
                "enable_file_locking": False,
                "validation_enabled": True,
                "fast_cleanup": True,
                "cache_ttl": 30
            },
            use_cases=["unit_testing", "integration_testing", "ci_cd"],
            recommended_for=["test_environments", "automated_testing", "development_ci"]
        )
        
        # Create template JSON files if they don't exist
        self._create_template_files()
        self.logger.info(f"Initialized {len(self._templates)} configuration templates")
    
    def _create_template_files(self):
        """Create JSON template files for easy customization."""
        self._template_dir.mkdir(exist_ok=True)
        
        for name, template in self._templates.items():
            template_file = self._template_dir / f"{name}.json"
            if not template_file.exists():
                template_data = {
                    "name": template.name,
                    "description": template.description,
                    "configuration": template.base_config.to_dict(),
                    "performance_optimizations": template.performance_optimizations,
                    "use_cases": template.use_cases,
                    "recommended_for": template.recommended_for,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
                
                with open(template_file, 'w') as f:
                    json.dump(template_data, f, indent=2)
                
                self.logger.debug(f"Created template file: {template_file}")
    
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
                "recommended_for": template.recommended_for,
                "performance_optimizations": template.performance_optimizations
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
            environment: Target environment ("development", "staging", "production", "test")
            storage_path: Storage path
            performance_level: Performance optimization level ("speed", "balanced", "quality")
            
        Returns:
            Environment-optimized ChatAppStorageConfig
        """
        # Map environments to templates
        template_mapping = {
            "development": "development",
            "staging": "production",  # Use production template with modifications
            "production": "production",
            "test": "testing"
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
        elif environment == "test":
            overrides.update({
                "cache_size_mb": 30,
                "enable_compression": False,
                "backup_enabled": False,
                "max_session_size_mb": 5
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
            "support_chat": "production",
            "testing_chat": "testing"
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
        elif use_case == "testing_chat":
            overrides.setdefault("environment", "test")
            overrides.setdefault("cache_size_mb", 30)
        
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
        
        # Test environment specific recommendations
        if config.environment == "test":
            if config.cache_size_mb > 50:
                results["recommendations"].append("Consider smaller cache size for test environment")
            if config.enable_compression:
                results["recommendations"].append("Disable compression for faster test execution")
            performance_score += 15
        
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
                environment=wrapper_config.get("environment", "development"),
                message_batch_size=wrapper_config.get("message_batch_size", 100),
                history_page_size=wrapper_config.get("history_page_size", 50),
                search_result_limit=wrapper_config.get("search_result_limit", 20)
            )
            
            self.logger.info("Successfully migrated wrapper configuration to bridge configuration")
            return bridge_config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to migrate wrapper configuration: {e}",
                context={"wrapper_config": wrapper_config}
            )

    def load_template_from_file(self, file_path: Union[str, Path]) -> ChatConfigTemplate:
        """
        Load configuration template from JSON file.
        
        Args:
            file_path: Path to template JSON file
            
        Returns:
            ChatConfigTemplate loaded from file
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            template_path = Path(file_path)
            if not template_path.exists():
                raise ConfigurationError(f"Template file not found: {file_path}")
            
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Create configuration from template data
            config_dict = template_data["configuration"].copy()
            
            # Remove any keys that aren't valid for ChatAppStorageConfig
            valid_keys = {
                "storage_path", "enable_vector_search", "enable_streaming", 
                "enable_analytics", "enable_compression", "backup_enabled",
                "cache_size_mb", "performance_mode", "max_session_size_mb",
                "message_batch_size", "history_page_size", "search_result_limit",
                "environment"
            }
            
            config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            base_config = ChatAppStorageConfig(**config_dict)
            
            # Create template object
            template = ChatConfigTemplate(
                name=template_data["name"],
                description=template_data["description"],
                base_config=base_config,
                performance_optimizations=template_data.get("performance_optimizations", {}),
                use_cases=template_data.get("use_cases", []),
                recommended_for=template_data.get("recommended_for", [])
            )
            
            self.logger.info(f"Loaded template '{template.name}' from {file_path}")
            return template
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load template from file {file_path}: {e}",
                context={"file_path": str(file_path)}
            )
    
    def export_template(self, template_name: str, output_path: Union[str, Path]) -> bool:
        """
        Export template to JSON file.
        
        Args:
            template_name: Name of template to export
            output_path: Path where to save the template
            
        Returns:
            True if export successful
            
        Raises:
            ConfigurationError: If template not found or export fails
        """
        if template_name not in self._templates:
            raise ConfigurationError(f"Template '{template_name}' not found")
        
        try:
            template = self._templates[template_name]
            output_file = Path(output_path)
            
            template_data = {
                "name": template.name,
                "description": template.description,
                "configuration": template.base_config.to_dict(),
                "performance_optimizations": template.performance_optimizations,
                "use_cases": template.use_cases,
                "recommended_for": template.recommended_for,
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            self.logger.info(f"Exported template '{template_name}' to {output_path}")
            return True
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to export template '{template_name}': {e}",
                context={"template_name": template_name, "output_path": str(output_path)}
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


def create_chat_config_for_testing(storage_path: str = "./test_data",
                                  **overrides) -> ChatAppStorageConfig:
    """
    Create testing-optimized chat configuration.
    
    Args:
        storage_path: Storage path for testing
        **overrides: Configuration overrides
        
    Returns:
        Testing-optimized configuration
    """
    factory = FFChatConfigFactory()
    return factory.create_for_environment("test", storage_path, **overrides)


def get_chat_app_presets() -> Dict[str, ChatAppStorageConfig]:
    """
    Get predefined chat app configuration presets.
    
    Returns:
        Dictionary of preset configurations
    """
    factory = FFChatConfigFactory()
    
    presets = {}
    for template_name in ["development", "production", "high_performance", "feature_rich", "lightweight", "testing"]:
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


def get_recommended_config_for_use_case(use_case: str) -> Dict[str, Any]:
    """
    Get recommended configuration settings for a specific use case.
    
    Args:
        use_case: Type of chat application
        
    Returns:
        Dictionary with recommended settings and explanation
    """
    factory = FFChatConfigFactory()
    templates = factory.list_templates()
    
    use_case_mapping = {
        "simple_chat": "lightweight",
        "ai_assistant": "feature_rich", 
        "high_volume_chat": "high_performance",
        "enterprise_chat": "production",
        "development_chat": "development",
        "research_chat": "feature_rich",
        "gaming_chat": "high_performance",
        "support_chat": "production",
        "testing_chat": "testing"
    }
    
    recommended_template = use_case_mapping.get(use_case, "development")
    
    if recommended_template in templates:
        return {
            "use_case": use_case,
            "recommended_template": recommended_template,
            "template_info": templates[recommended_template],
            "setup_example": f"""
# Recommended setup for {use_case}:
from ff_chat_integration import FFChatAppBridge

bridge = await FFChatAppBridge.create_for_use_case(
    "{use_case}", 
    "./your_data_path"
)
"""
        }
    else:
        return {
            "use_case": use_case,
            "error": f"No template found for use case: {use_case}",
            "available_use_cases": list(use_case_mapping.keys())
        }