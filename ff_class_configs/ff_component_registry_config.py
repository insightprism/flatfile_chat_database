"""
FF Component Registry Configuration

Configuration for FF component registry following existing FF configuration 
patterns and integrating with FF dependency injection manager.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from .ff_base_config import FFBaseConfigDTO, validate_positive


class FFComponentLoadingStrategy(str, Enum):
    """Component loading strategies"""
    LAZY = "lazy"                     # Load components on demand
    EAGER = "eager"                   # Load all components at startup
    PRIORITY_BASED = "priority"       # Load by priority order
    DEPENDENCY_BASED = "dependency"   # Load based on dependency graph


class FFComponentLifecycle(str, Enum):
    """Component lifecycle management"""
    SINGLETON = "singleton"           # One instance per application
    SESSION_SCOPED = "session"        # One instance per session
    REQUEST_SCOPED = "request"        # New instance per request
    PROTOTYPE = "prototype"           # New instance on each access


@dataclass
class FFComponentRegistryConfigDTO(FFBaseConfigDTO):
    """FF Component Registry configuration following FF patterns"""
    
    # Registry management
    auto_discovery_enabled: bool = True
    component_scan_packages: List[str] = field(default_factory=lambda: ["ff_components"])
    registry_cache_enabled: bool = True
    
    # Component loading settings
    loading_strategy: str = FFComponentLoadingStrategy.PRIORITY_BASED.value
    parallel_loading: bool = True
    max_loading_threads: int = 4
    component_timeout: int = 30  # seconds
    
    # Dependency management (integration with FF DI)
    use_ff_dependency_injection: bool = True
    enable_circular_dependency_detection: bool = True
    dependency_resolution_timeout: int = 60  # seconds
    
    # Component lifecycle
    default_lifecycle: str = FFComponentLifecycle.SINGLETON.value
    enable_component_hot_reload: bool = False
    component_health_checks: bool = True
    health_check_interval: int = 300  # seconds
    
    # Performance and monitoring
    enable_component_metrics: bool = True
    metrics_collection_interval: int = 60  # seconds
    enable_load_balancing: bool = False
    
    # Component validation
    enable_component_validation: bool = True
    validate_component_interfaces: bool = True
    validate_component_dependencies: bool = True
    
    # Error handling and resilience
    enable_component_fallbacks: bool = True
    max_component_retry_attempts: int = 3
    component_failure_threshold: float = 0.8  # failure rate threshold
    circuit_breaker_enabled: bool = False
    
    # Security and isolation
    enable_component_sandboxing: bool = False
    component_resource_limits: Dict[str, Any] = field(default_factory=dict)
    enable_component_audit_logging: bool = True
    
    # Configuration management
    component_config_hot_reload: bool = False
    config_validation_strict: bool = True
    enable_config_versioning: bool = False
    
    def validate(self) -> List[str]:
        """Validate component registry configuration"""
        errors = []
        
        # Validate loading strategy
        valid_strategies = [strategy.value for strategy in FFComponentLoadingStrategy]
        if self.loading_strategy not in valid_strategies:
            errors.append(f"loading_strategy must be one of {valid_strategies}")
        
        # Validate parallel loading settings
        if self.parallel_loading:
            error = validate_positive(self.max_loading_threads, "max_loading_threads")
            if error:
                errors.append(error)
        
        # Validate timeout settings
        error = validate_positive(self.component_timeout, "component_timeout")
        if error:
            errors.append(error)
            
        error = validate_positive(self.dependency_resolution_timeout, "dependency_resolution_timeout")
        if error:
            errors.append(error)
        
        # Validate lifecycle
        valid_lifecycles = [lifecycle.value for lifecycle in FFComponentLifecycle]
        if self.default_lifecycle not in valid_lifecycles:
            errors.append(f"default_lifecycle must be one of {valid_lifecycles}")
        
        # Validate health check settings
        if self.component_health_checks:
            error = validate_positive(self.health_check_interval, "health_check_interval")
            if error:
                errors.append(error)
        
        # Validate metrics settings
        if self.enable_component_metrics:
            error = validate_positive(self.metrics_collection_interval, "metrics_collection_interval")
            if error:
                errors.append(error)
        
        # Validate error handling settings
        error = validate_positive(self.max_component_retry_attempts, "max_component_retry_attempts")
        if error:
            errors.append(error)
            
        if self.component_failure_threshold < 0.0 or self.component_failure_threshold > 1.0:
            errors.append("component_failure_threshold must be between 0.0 and 1.0")
        
        # Validate scan packages
        if self.auto_discovery_enabled and not self.component_scan_packages:
            errors.append("component_scan_packages cannot be empty when auto_discovery_enabled is True")
        
        return errors
    
    def get_loading_config(self) -> Dict[str, Any]:
        """Get component loading configuration"""
        return {
            "strategy": self.loading_strategy,
            "parallel_loading": self.parallel_loading,
            "max_threads": self.max_loading_threads,
            "timeout": self.component_timeout,
            "auto_discovery": self.auto_discovery_enabled,
            "scan_packages": self.component_scan_packages
        }
    
    def get_dependency_config(self) -> Dict[str, Any]:
        """Get dependency management configuration"""
        return {
            "use_ff_di": self.use_ff_dependency_injection,
            "circular_detection": self.enable_circular_dependency_detection,
            "resolution_timeout": self.dependency_resolution_timeout,
            "validate_dependencies": self.validate_component_dependencies
        }
    
    def get_lifecycle_config(self) -> Dict[str, Any]:
        """Get component lifecycle configuration"""
        return {
            "default_lifecycle": self.default_lifecycle,
            "hot_reload": self.enable_component_hot_reload,
            "health_checks": self.component_health_checks,
            "health_check_interval": self.health_check_interval
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance and monitoring configuration"""
        return {
            "metrics_enabled": self.enable_component_metrics,
            "metrics_interval": self.metrics_collection_interval,
            "load_balancing": self.enable_load_balancing,
            "registry_cache": self.registry_cache_enabled
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get component validation configuration"""
        return {
            "validation_enabled": self.enable_component_validation,
            "interface_validation": self.validate_component_interfaces,
            "dependency_validation": self.validate_component_dependencies,
            "config_validation_strict": self.config_validation_strict
        }
    
    def get_resilience_config(self) -> Dict[str, Any]:
        """Get error handling and resilience configuration"""
        return {
            "fallbacks_enabled": self.enable_component_fallbacks,
            "max_retry_attempts": self.max_component_retry_attempts,
            "failure_threshold": self.component_failure_threshold,
            "circuit_breaker": self.circuit_breaker_enabled
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security and isolation configuration"""
        return {
            "sandboxing_enabled": self.enable_component_sandboxing,
            "resource_limits": self.component_resource_limits,
            "audit_logging": self.enable_component_audit_logging
        }
    
    def get_config_management_config(self) -> Dict[str, Any]:
        """Get configuration management settings"""
        return {
            "config_hot_reload": self.component_config_hot_reload,
            "config_versioning": self.enable_config_versioning,
            "validation_strict": self.config_validation_strict
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()
        
        # Additional post-initialization validation
        if hasattr(self, '_post_init_done'):
            return
        self._post_init_done = True
        
        # Ensure FF DI integration for proper component management
        if not self.use_ff_dependency_injection:
            # Force enable for proper FF integration
            self.use_ff_dependency_injection = True
        
        # Disable advanced features that may not be fully implemented
        if self.enable_component_hot_reload:
            # Hot reload requires careful implementation
            self.component_config_hot_reload = False
        
        # Set reasonable resource limits if sandboxing is enabled
        if self.enable_component_sandboxing and not self.component_resource_limits:
            self.component_resource_limits = {
                "max_memory_mb": 512,
                "max_cpu_percent": 50,
                "max_execution_time": 30
            }


@dataclass
class FFComponentRegistrationConfigDTO(FFBaseConfigDTO):
    """Configuration for individual component registration"""
    
    # Component identification
    component_name: str = ""
    component_version: str = "1.0.0"
    component_description: str = ""
    
    # Component class information
    component_class_path: str = ""
    config_class_path: str = ""
    
    # Dependency specification
    required_dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    ff_manager_dependencies: List[str] = field(default_factory=list)
    
    # Loading and lifecycle
    loading_priority: int = 100
    lifecycle: str = FFComponentLifecycle.SINGLETON.value
    lazy_loading: bool = False
    
    # Component capabilities
    supported_use_cases: List[str] = field(default_factory=list)
    component_capabilities: List[str] = field(default_factory=list)
    
    # Configuration
    component_config: Dict[str, Any] = field(default_factory=dict)
    environment_specific_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Validation and health
    enable_health_checks: bool = True
    health_check_endpoint: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate component registration configuration"""
        errors = []
        
        # Validate required fields
        if not self.component_name or not self.component_name.strip():
            errors.append("component_name cannot be empty")
        
        if not self.component_class_path or not self.component_class_path.strip():
            errors.append("component_class_path cannot be empty")
        
        if not self.config_class_path or not self.config_class_path.strip():
            errors.append("config_class_path cannot be empty")
        
        # Validate version format (basic check)
        if not self.component_version or not self.component_version.strip():
            errors.append("component_version cannot be empty")
        
        # Validate lifecycle
        valid_lifecycles = [lifecycle.value for lifecycle in FFComponentLifecycle]
        if self.lifecycle not in valid_lifecycles:
            errors.append(f"lifecycle must be one of {valid_lifecycles}")
        
        # Validate priority
        if self.loading_priority < 0:
            errors.append("loading_priority must be non-negative")
        
        # Validate FF manager dependencies
        valid_ff_managers = [
            "ff_storage", "ff_search", "ff_vector", "ff_panel", 
            "ff_document", "ff_dependency_injection"
        ]
        for dependency in self.ff_manager_dependencies:
            if dependency not in valid_ff_managers:
                errors.append(f"Invalid FF manager dependency '{dependency}'. Must be one of {valid_ff_managers}")
        
        return errors
    
    def get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency information"""
        return {
            "required": self.required_dependencies,
            "optional": self.optional_dependencies,
            "ff_managers": self.ff_manager_dependencies
        }
    
    def get_component_metadata(self) -> Dict[str, Any]:
        """Get component metadata"""
        return {
            "name": self.component_name,
            "version": self.component_version,
            "description": self.component_description,
            "class_path": self.component_class_path,
            "config_class_path": self.config_class_path,
            "supported_use_cases": self.supported_use_cases,
            "capabilities": self.component_capabilities
        }
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration"""
        return {
            "priority": self.loading_priority,
            "lifecycle": self.lifecycle,
            "lazy_loading": self.lazy_loading,
            "health_checks": self.enable_health_checks,
            "health_endpoint": self.health_check_endpoint
        }


@dataclass
class FFComponentEnvironmentConfigDTO(FFBaseConfigDTO):
    """Environment-specific component configuration"""
    
    # Environment identification
    environment_name: str = "default"
    environment_type: str = "development"  # development, testing, staging, production
    
    # Component overrides for this environment
    component_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Environment-specific settings
    enable_debug_mode: bool = False
    enable_verbose_logging: bool = False
    performance_monitoring_level: str = "basic"  # none, basic, detailed, full
    
    # Resource constraints for the environment
    max_concurrent_components: int = 10
    component_memory_limit_mb: int = 1024
    component_timeout_multiplier: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate environment configuration"""
        errors = []
        
        # Validate environment name
        if not self.environment_name or not self.environment_name.strip():
            errors.append("environment_name cannot be empty")
        
        # Validate environment type
        valid_types = ["development", "testing", "staging", "production"]
        if self.environment_type not in valid_types:
            errors.append(f"environment_type must be one of {valid_types}")
        
        # Validate performance monitoring level
        valid_levels = ["none", "basic", "detailed", "full"]
        if self.performance_monitoring_level not in valid_levels:
            errors.append(f"performance_monitoring_level must be one of {valid_levels}")
        
        # Validate resource constraints
        error = validate_positive(self.max_concurrent_components, "max_concurrent_components")
        if error:
            errors.append(error)
            
        error = validate_positive(self.component_memory_limit_mb, "component_memory_limit_mb")
        if error:
            errors.append(error)
            
        if self.component_timeout_multiplier <= 0.0:
            errors.append("component_timeout_multiplier must be positive")
        
        return errors
    
    def get_environment_overrides(self, component_name: str) -> Dict[str, Any]:
        """Get environment-specific overrides for a component"""
        return self.component_overrides.get(component_name, {})
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for this environment"""
        return {
            "max_concurrent_components": self.max_concurrent_components,
            "memory_limit_mb": self.component_memory_limit_mb,
            "timeout_multiplier": self.component_timeout_multiplier
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration for this environment"""
        return {
            "debug_mode": self.enable_debug_mode,
            "verbose_logging": self.enable_verbose_logging,
            "performance_level": self.performance_monitoring_level
        }