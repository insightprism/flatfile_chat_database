"""
Dependency injection container for flatfile chat database.

Provides centralized service registration and resolution with support for
singletons, factories, and scoped instances.
"""

import asyncio
from typing import Dict, Type, Any, Callable, Optional, TypeVar, Union, List
from contextlib import asynccontextmanager
from pathlib import Path
import inspect

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_protocols import (
    StorageProtocol, SearchProtocol, VectorStoreProtocol,
    DocumentProcessorProtocol, BackendProtocol, FileOperationsProtocol
)

T = TypeVar('T')


class FFFFServiceLifetime:
    """Service lifetime options."""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance for container lifetime
    SCOPED = "scoped"       # Single instance per scope


class FFFFServiceDescriptor:
    """Describes a registered service."""
    
    def __init__(self, 
                 interface: Type,
                 implementation: Optional[Type] = None,
                 factory: Optional[Callable] = None,
                 instance: Optional[Any] = None,
                 lifetime: str = FFFFServiceLifetime.TRANSIENT,
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


class FFFFServiceScope:
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
        
        if descriptor.lifetime == FFFFServiceLifetime.SCOPED:
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
        self._descriptors: Dict[Type, FFFFServiceDescriptor] = {}
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
    async def create_scope(self):
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
        Create service instance.
        
        Args:
            descriptor: Service descriptor
            scope: Optional scope for resolving dependencies
            
        Returns:
            Service instance
        """
        # If instance provided, return it
        if descriptor.instance:
            return descriptor.instance
        
        # If factory provided, use it
        if descriptor.factory:
            return descriptor.factory(self)
        
        # Otherwise, instantiate implementation with dependencies
        if descriptor.implementation:
            # Resolve dependencies
            resolved_deps = {}
            for dep_type in descriptor.dependencies:
                resolved_deps[dep_type.__name__.lower()] = self.resolve(dep_type, scope)
            
            # Create instance
            return descriptor.implementation(**resolved_deps)
        
        raise ValueError("No way to create instance")
    
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
    
    def get_all_registered(self) -> List[Type]:
        """
        Get all registered service interfaces.
        
        Returns:
            List of registered interfaces
        """
        return list(self._descriptors.keys())
    
    def is_registered(self, interface: Type) -> bool:
        """
        Check if service is registered.
        
        Args:
            interface: Service interface
            
        Returns:
            True if registered
        """
        return interface in self._descriptors
    
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
    from config_new.manager import load_config
    from backends.flatfile import FlatfileBackend
    from ff_search_manager import FFSearchManager
    from ff_vector_storage_manager import FFVectorStorageManager
    from ff_document_processing_manager import FFDocumentProcessingManager
    from ff_storage_manager import FFStorageManager
    from ff_utils.ff_file_ops import FFFileOperationManager
    
    container = FFDependencyInjectionManager()
    
    # Load configuration
    config_manager = load_config(config_path, environment)
    container.register_singleton(FFConfigurationManagerConfigDTO, instance=config_manager)
    
    # Register file operations
    def file_ops_factory(c: FFDependencyInjectionManager) -> FileOperationsProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        return FFFileOperationManager(config.storage)
    
    container.register_singleton(FileOperationsProtocol, factory=file_ops_factory)
    
    # Register backend
    def backend_factory(c: FFDependencyInjectionManager) -> BackendProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        return FlatfileBackend(config.storage)
    
    container.register_singleton(BackendProtocol, factory=backend_factory)
    
    # Register vector store
    def vector_store_factory(c: FFDependencyInjectionManager) -> VectorStoreProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        return FFVectorStorageManager(config.vector)
    
    container.register_singleton(VectorStoreProtocol, factory=vector_store_factory)
    
    # Register search engine
    def search_factory(c: FFDependencyInjectionManager) -> SearchProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        return FFSearchManager(config.search)
    
    container.register_singleton(SearchProtocol, factory=search_factory)
    
    # Register document processor
    def processor_factory(c: FFDependencyInjectionManager) -> DocumentProcessorProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        vector_store = c.resolve(VectorStoreProtocol)
        return FFDocumentProcessingManager(config.document, vector_store)
    
    container.register_singleton(DocumentProcessorProtocol, factory=processor_factory)
    
    # Register storage manager
    def storage_factory(c: FFDependencyInjectionManager) -> StorageProtocol:
        config = c.resolve(FFConfigurationManagerConfigDTO)
        backend = c.resolve(BackendProtocol)
        search = c.resolve(SearchProtocol)
        vector_store = c.resolve(VectorStoreProtocol)
        processor = c.resolve(DocumentProcessorProtocol)
        
        # Create storage manager with all dependencies
        manager = FFStorageManager(
            config=config.storage,
            backend=backend
        )
        
        # Inject other services
        manager.search_engine = search
        manager.vector_storage = vector_store
        manager.document_processor = processor
        
        return manager
    
    container.register_singleton(StorageProtocol, factory=storage_factory)
    
    return container


# Global container instance (optional)
_global_container: Optional[FFDependencyInjectionManager] = None


def ff_get_container() -> FFDependencyInjectionManager:
    """
    Get global container instance.
    
    Returns:
        Global container (creates if not exists)
    """
    global _global_container
    if _global_container is None:
        _global_container = ff_create_application_container()
    return _global_container


def ff_set_container(container: FFDependencyInjectionManager) -> None:
    """
    Set global container instance.
    
    Args:
        container: Container to set as global
    """
    global _global_container
    _global_container = container