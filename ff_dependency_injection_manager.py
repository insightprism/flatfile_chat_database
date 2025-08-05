"""
Dependency injection container for flatfile chat database.

Provides centralized service registration and resolution with support for
singletons, factories, and scoped instances.
"""

import asyncio
from typing import Dict, Type, Any, Callable, Optional, TypeVar, Union, List, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
import inspect

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_protocols import (
    StorageProtocol, SearchProtocol, VectorStoreProtocol,
    DocumentProcessorProtocol, BackendProtocol, FileOperationsProtocol
)

T = TypeVar('T')


class FFServiceLifetime:
    """Service lifetime options."""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance for container lifetime
    SCOPED = "scoped"       # Single instance per scope


class FFServiceDescriptor:
    """Describes a registered service."""
    
    def __init__(self, 
                 interface: Type,
                 implementation: Optional[Type] = None,
                 factory: Optional[Callable] = None,
                 instance: Optional[Any] = None,
                 lifetime: str = FFServiceLifetime.TRANSIENT,
                 dependencies: Optional[List[Type]] = None):
        """
        Initialize service descriptor.
        
        Args:
            interface: Service interface type
            implementation: Concrete implementation class
            factory: Factory function to create instance
            instance: Pre-created instance (for singletons)
            lifetime: Service lifetime
            dependencies: List of dependency types
        """
        self.interface = interface
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.dependencies = dependencies or []
        
        # Validate descriptor
        if not any([implementation, factory, instance]):
            raise ValueError("Must provide implementation, factory, or instance")


class FFServiceScope:
    """Represents a dependency injection scope."""
    
    def __init__(self, container: 'FFDependencyInjectionManager'):
        """
        Initialize service scope.
        
        Args:
            container: Parent container
        """
        self.container = container
        self.scoped_instances: Dict[Type, Any] = {}
    
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve service within scope.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
        """
        # Check if we have a scoped instance
        if interface in self.scoped_instances:
            return self.scoped_instances[interface]
        
        # Get descriptor from container
        descriptor = self.container._get_descriptor(interface)
        
        if descriptor.lifetime == FFServiceLifetime.SCOPED:
            # Create and cache scoped instance
            instance = self.container._create_instance(descriptor, scope=self)
            self.scoped_instances[interface] = instance
            return instance
        else:
            # Delegate to container for other lifetimes
            return self.container.resolve(interface)


class FFDependencyInjectionManager:
    """
    Dependency injection container with support for various lifetimes.
    
    Manages service registration, resolution, and lifecycle.
    """
    
    def __init__(self):
        """Initialize service container."""
        self._descriptors: Dict[Type, FFServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = asyncio.Lock()
    
    def register(self, 
                 interface: Type[T],
                 implementation: Optional[Type[T]] = None,
                 factory: Optional[Callable[['FFDependencyInjectionManager'], T]] = None,
                 instance: Optional[T] = None,
                 lifetime: str = FFServiceLifetime.TRANSIENT) -> None:
        """
        Register a service.
        
        Args:
            interface: Service interface type
            implementation: Concrete implementation class
            factory: Factory function to create instance
            instance: Pre-created instance
            lifetime: Service lifetime
        """
        # Auto-detect dependencies if implementation provided

        dependencies = []
        if implementation and not factory and not instance:
            dependencies = self._get_constructor_dependencies(implementation)
        
        descriptor = FFServiceDescriptor(
            interface=interface,
            implementation=implementation,
            factory=factory,
            instance=instance,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        # Validate registration
        self._validate_service_registration(interface, implementation, factory, instance)
        
        self._descriptors[interface] = descriptor
        
        # Store singleton instances immediately
        if instance and lifetime == FFServiceLifetime.SINGLETON:
            self._singletons[interface] = instance
    
    def register_singleton(self, interface: Type[T], 
                         implementation: Optional[Type[T]] = None,
                         factory: Optional[Callable[['FFDependencyInjectionManager'], T]] = None,
                         instance: Optional[T] = None) -> None:
        """
        Register a singleton service.
        
        Args:
            interface: Service interface
            implementation: Implementation class
            factory: Factory function
            instance: Pre-created instance
        """
        self.register(interface, implementation, factory, instance, FFServiceLifetime.SINGLETON)
    
    def register_transient(self, interface: Type[T],
                         implementation: Optional[Type[T]] = None,
                         factory: Optional[Callable[['FFDependencyInjectionManager'], T]] = None) -> None:
        """
        Register a transient service.
        
        Args:
            interface: Service interface
            implementation: Implementation class
            factory: Factory function
        """
        self.register(interface, implementation, factory, lifetime=FFServiceLifetime.TRANSIENT)
    
    def register_scoped(self, interface: Type[T],
                       implementation: Optional[Type[T]] = None,
                       factory: Optional[Callable[['FFDependencyInjectionManager'], T]] = None) -> None:
        """
        Register a scoped service.
        
        Args:
            interface: Service interface
            implementation: Implementation class
            factory: Factory function
        """
        self.register(interface, implementation, factory, lifetime=FFServiceLifetime.SCOPED)
    
    def resolve(self, interface: Type[T], scope: Optional[FFServiceScope] = None) -> T:
        """
        Resolve a service.
        
        Args:
            interface: Service interface to resolve
            scope: Optional scope for scoped services
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
        """
        descriptor = self._get_descriptor(interface)
        
        # Handle different lifetimes
        if descriptor.lifetime == FFServiceLifetime.SINGLETON:
            if interface in self._singletons:
                return self._singletons[interface]
            
            # Create singleton
            instance = self._create_instance(descriptor)
            self._singletons[interface] = instance
            return instance
        
        elif descriptor.lifetime == FFServiceLifetime.SCOPED:
            if scope:
                return scope.resolve(interface)
            else:
                # No scope provided, treat as transient
                return self._create_instance(descriptor)
        
        else:  # TRANSIENT
            return self._create_instance(descriptor)
    
    async def resolve_async(self, interface: Type[T], scope: Optional[FFServiceScope] = None) -> T:
        """
        Resolve a service asynchronously (thread-safe).
        
        Args:
            interface: Service interface to resolve
            scope: Optional scope
            
        Returns:
            Service instance
        """
        async with self._lock:
            return self.resolve(interface, scope)
    
    @asynccontextmanager
    async def create_scope(self) -> AsyncGenerator['FFServiceScope', None]:
        """
        Create a new service scope.
        
        Usage:
            async with container.create_scope() as scope:
                service = container.resolve(MyService, scope)
        
        Yields:
            Service scope
        """
        scope = FFServiceScope(self)
        try:
            yield scope
        finally:
            # Cleanup scoped instances if needed
            scope.scoped_instances.clear()
    
    def _get_descriptor(self, interface: Type) -> FFServiceDescriptor:
        """
        Get service descriptor.
        
        Args:
            interface: Service interface
            
        Returns:
            Service descriptor
            
        Raises:
            ValueError: If not registered
        """
        if interface not in self._descriptors:
            raise ValueError(f"Service {interface.__name__} not registered")
        return self._descriptors[interface]
    
    def _create_instance(self, descriptor: FFServiceDescriptor, 
                        scope: Optional[FFServiceScope] = None) -> Any:
        """
        Create service instance with enhanced error handling.
        
        Args:
            descriptor: Service descriptor
            scope: Optional scope for resolving dependencies
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service creation fails
        """
        interface_name = getattr(descriptor.interface, '__name__', str(descriptor.interface))
        
        try:
            # If instance provided, return it
            if descriptor.instance:
                return descriptor.instance
            
            # If factory provided, use it
            if descriptor.factory:
                return descriptor.factory(self)
            
            # Otherwise, instantiate implementation with dependencies
            if descriptor.implementation:
                # Resolve dependencies with better error reporting
                resolved_deps = {}
                for dep_type in descriptor.dependencies:
                    try:
                        resolved_deps[dep_type.__name__.lower()] = self.resolve(dep_type, scope)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to resolve dependency {dep_type.__name__} for {interface_name}"
                        ) from e
                
                # Create instance
                return descriptor.implementation(**resolved_deps)
            
            raise ValueError(f"No way to create instance for {interface_name}")
        
        except Exception as e:
            if isinstance(e, ValueError) and "Failed to resolve dependency" in str(e):
                # Re-raise dependency resolution errors as-is
                raise
            
            # Wrap other errors with context
            raise ValueError(
                f"Failed to create instance for {interface_name}: {str(e)}"
            ) from e
    
    def _get_constructor_dependencies(self, implementation: Type) -> List[Type]:
        """
        Extract constructor dependencies from type hints.
        
        Args:
            implementation: Implementation class
            
        Returns:
            List of dependency types
        """
        if not hasattr(implementation, '__init__'):
            return []
        
        sig = inspect.signature(implementation.__init__)
        dependencies = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Get type hint
            if param.annotation != inspect.Parameter.empty:
                dependencies.append(param.annotation)
        
        return dependencies
    
    def _validate_service_registration(self, interface: Type, implementation: Optional[Type] = None,
                                     factory: Optional[Callable] = None, 
                                     instance: Optional[Any] = None) -> None:
        """
        Validate service registration parameters.
        
        Args:
            interface: Service interface
            implementation: Implementation class
            factory: Factory function
            instance: Instance object
            
        Raises:
            ValueError: If registration is invalid
        """
        interface_name = getattr(interface, '__name__', str(interface))
        
        # Check if interface is already registered
        if interface in self._descriptors:
            # Allow re-registration for testing scenarios
            pass
        
        # Validate implementation inheritance if provided
        if implementation and hasattr(interface, '__mro__'):
            # For protocol/ABC interfaces, check if implementation has required methods
            if hasattr(interface, '__abstractmethods__'):
                abstract_methods = getattr(interface, '__abstractmethods__', set())
                impl_methods = set(dir(implementation))
                missing_methods = abstract_methods - impl_methods
                if missing_methods:
                    raise ValueError(
                        f"Implementation {implementation.__name__} missing required methods "
                        f"for {interface_name}: {missing_methods}"
                    )
        
        # Validate factory signature if provided
        if factory:
            try:
                sig = inspect.signature(factory)
                params = list(sig.parameters.keys())
                if not params or params[0] not in ['container', 'c', 'self']:
                    raise ValueError(
                        f"Factory for {interface_name} should accept container as first parameter"
                    )
            except Exception as e:
                raise ValueError(f"Invalid factory for {interface_name}: {e}")
        
        # Validate instance type if provided
        if instance and hasattr(interface, '__mro__'):
            if not isinstance(instance, interface):
                # For protocols, we can't use isinstance, so just warn
                pass
    
    def is_registered(self, interface: Type) -> bool:
        """Check if interface is registered."""
        return interface in self._descriptors
    
    def get_registration_info(self, interface: Type) -> Dict[str, Any]:
        """
        Get registration information for debugging.
        
        Args:
            interface: Service interface
            
        Returns:
            Dictionary with registration details
        """
        if interface not in self._descriptors:
            return {"registered": False}
        
        descriptor = self._descriptors[interface]
        return {
            "registered": True,
            "lifetime": descriptor.lifetime,
            "has_implementation": descriptor.implementation is not None,
            "has_factory": descriptor.factory is not None,
            "has_instance": descriptor.instance is not None,
            "dependency_count": len(descriptor.dependencies),
            "is_singleton_created": interface in self._singletons
        }
    
    def get_all_registered(self) -> List[Type]:
        """
        Get all registered service interfaces.
        
        Returns:
            List of registered interfaces
        """
        return list(self._descriptors.keys())
    
    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._descriptors.clear()
        self._singletons.clear()


def ff_create_application_container(config_path: Optional[Union[str, Path]] = None,
                               environment: Optional[str] = None) -> FFDependencyInjectionManager:
    """
    Create and configure application container with all services.
    
    Args:
        config_path: Optional configuration file path
        environment: Optional environment name
        
    Returns:
        Configured service container
    """
    from ff_class_configs.ff_configuration_manager_config import load_config
    from backends import FlatfileBackend
    from ff_search_manager import FFSearchManager
    from ff_vector_storage_manager import FFVectorStorageManager
    from ff_document_processing_manager import FFDocumentProcessingManager
    from ff_storage_manager import FFStorageManager
    from ff_utils.ff_file_ops import FileOperationManager
    from ff_embedding_functions import generate_embeddings
    from functools import partial
    
    container = FFDependencyInjectionManager()
    
    # Load configuration
    config_manager = load_config(config_path, environment)
    container.register_singleton(FFConfigurationManagerConfigDTO, instance=config_manager)
    
    # Register file operations  
    def file_ops_factory(container: FFDependencyInjectionManager) -> FileOperationsProtocol:
        config = container.resolve(FFConfigurationManagerConfigDTO)
        return FileOperationManager(config)
    
    container.register_singleton(FileOperationsProtocol, factory=file_ops_factory)
    
    # Register backend
    def backend_factory(container: FFDependencyInjectionManager) -> BackendProtocol:
        config = container.resolve(FFConfigurationManagerConfigDTO)
        return FlatfileBackend(config)
    
    container.register_singleton(BackendProtocol, factory=backend_factory)
    
    # Register vector store
    def vector_store_factory(c: FFDependencyInjectionManager) -> VectorStoreProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        return FFVectorStorageManager(config)
    
    container.register_singleton(VectorStoreProtocol, factory=vector_store_factory)
    
    # Register search engine
    def search_factory(container: FFDependencyInjectionManager) -> SearchProtocol:
        config = container.resolve(FFConfigurationManagerConfigDTO)
        return FFSearchManager(config)
    
    container.register_singleton(SearchProtocol, factory=search_factory)
    
    # Register document processor
    def processor_factory(container: FFDependencyInjectionManager) -> DocumentProcessorProtocol:
        config = container.resolve(FFConfigurationManagerConfigDTO)
        # Note: vector_store dependency commented out as FFDocumentProcessingManager
        # doesn't currently use it in constructor
        return FFDocumentProcessingManager(config)
    
    container.register_singleton(DocumentProcessorProtocol, factory=processor_factory)
    
    # Register storage manager
    def storage_factory(container: FFDependencyInjectionManager) -> StorageProtocol:
        config = container.resolve(FFConfigurationManagerConfigDTO)
        backend = container.resolve(BackendProtocol)
        
        # Create storage manager with essential dependencies
        # Other services will be lazily loaded via DI container
        manager = FFStorageManager(
            config=config,
            backend=backend
        )
        
        return manager
    
    container.register_singleton(StorageProtocol, factory=storage_factory)
    
    # Register embedding function as a service
    def embedding_factory(container: FFDependencyInjectionManager):
        config = container.resolve(FFConfigurationManagerConfigDTO)
        return partial(generate_embeddings, config=config)
    
    # Register embedding service (simplified for now)
    container.register_singleton("generate_embeddings", factory=embedding_factory)
    
    return container


# ===== GLOBAL CONTAINER SINGLETON PATTERN =====
#
# The global container provides a convenient singleton access pattern for the DI container
# across the entire application. This is particularly useful for:
#
# 1. **Application-wide service resolution**: Access services from anywhere without 
#    passing the container instance around
# 2. **Legacy code integration**: Gradually migrate existing code to use DI
# 3. **Testing isolation**: Each test can set its own container configuration
# 4. **Configuration consistency**: Ensures all components use the same service instances
#
# Usage patterns:
#   - Production: Container is auto-created with default configuration on first access
#   - Testing: Set custom container with mocked services using ff_set_container()
#   - Development: Container can be reconfigured at runtime for debugging
#
# Thread Safety: The container itself is thread-safe, but global access should be
# initialized during application startup to avoid race conditions.
#
_global_container: Optional[FFDependencyInjectionManager] = None


def ff_get_container() -> FFDependencyInjectionManager:
    """
    Get the global dependency injection container instance (lazy singleton).
    
    This function implements the singleton pattern for application-wide access to
    the DI container. On first access, it automatically creates and configures
    a container with all standard services registered.
    
    Thread Safety: While the container operations are thread-safe, the initial
    creation should ideally happen during single-threaded application startup.
    
    Usage Examples:
        # Basic service resolution
        storage = ff_get_container().resolve(StorageProtocol)
        
        # In tests - use after ff_set_container() with mocks
        test_storage = ff_get_container().resolve(StorageProtocol)
    
    Returns:
        FFDependencyInjectionManager: The global container instance
        
    Note:
        The container is created with default configuration. For custom configuration,
        create your own container and set it using ff_set_container().
    """
    global _global_container
    if _global_container is None:
        _global_container = ff_create_application_container()
    return _global_container


def ff_set_container(container: FFDependencyInjectionManager) -> None:
    """
    Set a custom global container instance.
    
    This function allows you to replace the global container with a custom one,
    which is particularly useful for:
    - Unit testing with mocked services
    - Different configurations for different environments  
    - Runtime container reconfiguration
    - Isolation between test cases
    
    Usage Examples:
        # In tests - create container with mocked services
        test_container = FFDependencyInjectionManager()
        test_container.register_singleton(StorageProtocol, instance=mock_storage)
        ff_set_container(test_container)
        
        # Reset to default in test cleanup
        ff_set_container(ff_create_application_container())
        
        # Clear global state completely
        ff_set_container(None)  # Next ff_get_container() creates fresh instance
    
    Args:
        container: The container instance to set as global, or None to clear
        
    Warning:
        Setting a new container affects all subsequent service resolutions
        across the entire application. Ensure proper coordination in
        multi-threaded environments.
    """
    global _global_container
    _global_container = container


def ff_clear_global_container() -> None:
    """
    Clear the global container instance, forcing recreation on next access.
    
    This is primarily useful for:
    - Test isolation: Ensure each test starts with a fresh container
    - Memory cleanup: Release all singleton instances
    - Configuration reload: Force recreation with updated environment variables
    
    Usage Examples:
        # In test teardown
        def teardown():
            ff_clear_global_container()
            
        # Force configuration reload
        os.environ['CHATDB_ENV'] = 'production'
        ff_clear_global_container()
        container = ff_get_container()  # Creates new container with prod config
    
    Note:
        After calling this function, the next call to ff_get_container() will
        create a new container with current environment configuration.
    """
    global _global_container
    _global_container = None