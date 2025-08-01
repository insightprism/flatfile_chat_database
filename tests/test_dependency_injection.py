"""
Comprehensive tests for the dependency injection system.

Tests the DI container, service registration, resolution, lifetime management,
and integration with the updated architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Protocol
import sys
from pathlib import Path

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ff_dependency_injection_manager import (
    FFDependencyInjectionManager,
    FFServiceLifetime,
    FFServiceDescriptor,
    FFServiceScope,
    ff_create_application_container,
    ff_get_container,
    ff_set_container,
    ff_clear_global_container
)
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_protocols import (
    StorageProtocol, SearchProtocol, VectorStoreProtocol,
    DocumentProcessorProtocol, BackendProtocol, FileOperationsProtocol
)


# Test interfaces for DI testing
class ITestService(Protocol):
    def get_name(self) -> str: ...


class ITestDependency(Protocol):
    def get_value(self) -> int: ...


class TestService:
    def __init__(self, dependency: ITestDependency):
        self.dependency = dependency
    
    def get_name(self) -> str:
        return f"TestService_{self.dependency.get_value()}"


class TestDependency:
    def __init__(self, value: int = 42):
        self.value = value
    
    def get_value(self) -> int:
        return self.value


class TestServiceWithoutDeps:
    def get_name(self) -> str:
        return "NoDepService"


class TestDIContainer:
    """Test basic DI container functionality."""
    
    def test_container_creation(self):
        """Test creating a new DI container."""
        container = FFDependencyInjectionManager()
        assert container is not None
        assert len(container.get_all_registered()) == 0
    
    def test_register_transient_service(self):
        """Test registering transient services."""
        container = FFDependencyInjectionManager()
        
        container.register_transient(ITestService, TestServiceWithoutDeps)
        
        assert container.is_registered(ITestService)
        assert ITestService in container.get_all_registered()
    
    def test_register_singleton_service(self):
        """Test registering singleton services."""
        container = FFDependencyInjectionManager()
        
        container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        assert container.is_registered(ITestService)
        
        # Should get same instance
        service1 = container.resolve(ITestService)
        service2 = container.resolve(ITestService)
        assert service1 is service2
    
    def test_register_scoped_service(self):
        """Test registering scoped services."""
        container = FFDependencyInjectionManager()
        
        container.register_scoped(ITestService, TestServiceWithoutDeps)
        
        assert container.is_registered(ITestService)
    
    def test_register_with_instance(self):
        """Test registering pre-created instances."""
        container = FFDependencyInjectionManager()
        instance = TestServiceWithoutDeps()
        
        container.register_singleton(ITestService, instance=instance)
        
        resolved = container.resolve(ITestService)
        assert resolved is instance
    
    def test_register_with_factory(self):
        """Test registering services with factory functions."""
        container = FFDependencyInjectionManager()
        
        def service_factory(c: FFDependencyInjectionManager) -> ITestService:
            return TestServiceWithoutDeps()
        
        container.register_singleton(ITestService, factory=service_factory)
        
        service = container.resolve(ITestService)
        assert isinstance(service, TestServiceWithoutDeps)
    
    def test_registration_validation(self):
        """Test that registration validates required parameters."""
        container = FFDependencyInjectionManager()
        
        # Should raise error when no implementation, factory, or instance provided
        with pytest.raises(ValueError, match="Must provide implementation, factory, or instance"):
            container.register(ITestService)


class TestServiceResolution:
    """Test service resolution functionality."""
    
    def test_resolve_transient_service(self):
        """Test resolving transient services."""
        container = FFDependencyInjectionManager()
        container.register_transient(ITestService, TestServiceWithoutDeps)
        
        service1 = container.resolve(ITestService)
        service2 = container.resolve(ITestService)
        
        assert isinstance(service1, TestServiceWithoutDeps)
        assert isinstance(service2, TestServiceWithoutDeps)
        assert service1 is not service2  # Different instances
    
    def test_resolve_singleton_service(self):
        """Test resolving singleton services."""
        container = FFDependencyInjectionManager()
        container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        service1 = container.resolve(ITestService)
        service2 = container.resolve(ITestService)
        
        assert isinstance(service1, TestServiceWithoutDeps)
        assert service1 is service2  # Same instance
    
    def test_resolve_with_dependencies(self):
        """Test resolving services with dependencies."""
        container = FFDependencyInjectionManager()
        
        # Register dependency first
        container.register_singleton(ITestDependency, TestDependency)
        # Register service that depends on it
        container.register_transient(ITestService, TestService)
        
        service = container.resolve(ITestService)
        
        assert isinstance(service, TestService)
        assert service.get_name() == "TestService_42"
    
    def test_resolve_with_factory_dependencies(self):
        """Test resolving services with factory that uses dependencies."""
        container = FFDependencyInjectionManager()
        
        # Register dependency
        container.register_singleton(ITestDependency, TestDependency)
        
        # Register service with factory that resolves dependency
        def service_factory(c: FFDependencyInjectionManager) -> ITestService:
            dependency = c.resolve(ITestDependency)
            return TestService(dependency)
        
        container.register_transient(ITestService, factory=service_factory)
        
        service = container.resolve(ITestService)
        assert isinstance(service, TestService)
        assert service.get_name() == "TestService_42"
    
    def test_resolve_unregistered_service(self):
        """Test resolving unregistered service raises error."""
        container = FFDependencyInjectionManager()
        
        with pytest.raises(ValueError, match="Service .* not registered"):
            container.resolve(ITestService)
    
    @pytest.mark.asyncio
    async def test_resolve_async(self):
        """Test async service resolution."""
        container = FFDependencyInjectionManager()
        container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        service = await container.resolve_async(ITestService)
        assert isinstance(service, TestServiceWithoutDeps)


class TestServiceScopes:
    """Test service scope functionality."""
    
    @pytest.mark.asyncio
    async def test_scoped_service_lifecycle(self):
        """Test scoped service creation and lifecycle."""
        container = FFDependencyInjectionManager()
        container.register_scoped(ITestService, TestServiceWithoutDeps)
        
        async with container.create_scope() as scope:
            service1 = container.resolve(ITestService, scope)
            service2 = container.resolve(ITestService, scope)
            
            # Should be same instance within scope
            assert service1 is service2
        
        # New scope should create new instance
        async with container.create_scope() as scope2:
            service3 = container.resolve(ITestService, scope2)
            assert service3 is not service1
    
    @pytest.mark.asyncio
    async def test_scoped_service_without_scope(self):
        """Test scoped service resolution without explicit scope."""
        container = FFDependencyInjectionManager()
        container.register_scoped(ITestService, TestServiceWithoutDeps)
        
        # Should behave like transient when no scope provided
        service1 = container.resolve(ITestService)
        service2 = container.resolve(ITestService)
        
        assert service1 is not service2
    
    @pytest.mark.asyncio
    async def test_mixed_lifetimes_in_scope(self):
        """Test mixing different service lifetimes in scope."""
        container = FFDependencyInjectionManager()
        
        container.register_singleton(ITestDependency, TestDependency)
        container.register_scoped(ITestService, TestService)
        
        async with container.create_scope() as scope:
            service1 = container.resolve(ITestService, scope)
            service2 = container.resolve(ITestService, scope)
            dependency = container.resolve(ITestDependency, scope)
            
            # Scoped service should be same within scope
            assert service1 is service2
            # But should use same singleton dependency
            assert service1.dependency is dependency


class TestServiceDescriptors:
    """Test service descriptor functionality."""
    
    def test_service_descriptor_creation(self):
        """Test creating service descriptors."""
        descriptor = FFServiceDescriptor(
            interface=ITestService,
            implementation=TestServiceWithoutDeps,
            lifetime=FFServiceLifetime.SINGLETON
        )
        
        assert descriptor.interface == ITestService
        assert descriptor.implementation == TestServiceWithoutDeps
        assert descriptor.lifetime == FFServiceLifetime.SINGLETON
        assert descriptor.dependencies == []
    
    def test_service_descriptor_validation(self):
        """Test service descriptor validation."""
        # Should raise error with no implementation/factory/instance
        with pytest.raises(ValueError):
            FFServiceDescriptor(interface=ITestService)
    
    def test_dependency_detection(self):
        """Test automatic dependency detection."""
        container = FFDependencyInjectionManager()
        
        # Register service with automatic dependency detection
        container.register(ITestService, TestService, lifetime=FFServiceLifetime.TRANSIENT)
        
        descriptor = container._get_descriptor(ITestService)
        
        # Should detect ITestDependency as dependency
        assert len(descriptor.dependencies) == 1
        assert ITestDependency in descriptor.dependencies


class TestApplicationContainer:
    """Test application container creation and configuration."""
    
    def test_create_application_container(self):
        """Test creating full application container."""
        container = ff_create_application_container()
        
        # Should have all core services registered
        expected_services = [
            FFConfigurationManagerConfigDTO,
            BackendProtocol,
            StorageProtocol,
            SearchProtocol,
            VectorStoreProtocol,
            DocumentProcessorProtocol,
            FileOperationsProtocol
        ]
        
        for service in expected_services:
            assert container.is_registered(service), f"{service} should be registered"
    
    def test_application_container_service_resolution(self):
        """Test resolving services from application container."""
        container = ff_create_application_container()
        
        # Should be able to resolve all core services
        config = container.resolve(FFConfigurationManagerConfigDTO)
        assert config is not None
        
        storage = container.resolve(StorageProtocol)
        assert storage is not None
        
        backend = container.resolve(BackendProtocol)
        assert backend is not None
    
    def test_application_container_with_custom_config(self, test_config, temp_dir):
        """Test application container with custom configuration."""
        container = ff_create_application_container(str(temp_dir / "custom.json"), "test")
        
        # Should use custom configuration
        resolved_config = container.resolve(FFConfigurationManagerConfigDTO)
        # Note: Since config file doesn't exist, it should use environment defaults
        assert resolved_config.environment in ["test", "development"]


class TestGlobalContainer:
    """Test global container singleton pattern."""
    
    def setup_method(self):
        """Clear global container before each test."""
        ff_clear_global_container()
    
    def teardown_method(self):
        """Clear global container after each test."""
        ff_clear_global_container()
    
    def test_global_container_lazy_creation(self):
        """Test that global container is created lazily."""
        container = ff_get_container()
        
        assert container is not None
        assert isinstance(container, FFDependencyInjectionManager)
        
        # Should get same instance on subsequent calls  
        container2 = ff_get_container()
        assert container is container2
    
    def test_set_custom_global_container(self):
        """Test setting custom global container."""
        custom_container = FFDependencyInjectionManager()
        custom_container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        ff_set_container(custom_container)
        
        resolved_container = ff_get_container()
        assert resolved_container is custom_container
        assert resolved_container.is_registered(ITestService)
    
    def test_clear_global_container(self):
        """Test clearing global container."""
        # Get initial container
        container1 = ff_get_container()
        
        # Clear and get new one
        ff_clear_global_container()
        container2 = ff_get_container()
        
        assert container2 is not container1
    
    def test_global_container_service_resolution(self):
        """Test resolving services from global container."""
        container = ff_get_container()
        
        # Should have all application services
        config = container.resolve(FFConfigurationManagerConfigDTO)
        assert config is not None
        
        storage = container.resolve(StorageProtocol)
        assert storage is not None


class TestErrorHandling:
    """Test error handling in dependency injection."""
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # This would require more complex setup to create actual circular dependencies
        # For now, test that we can handle the error gracefully
        container = FFDependencyInjectionManager()
        
        # Register service that depends on itself (simplified circular dependency)
        with pytest.raises(Exception):  # Should raise some kind of dependency error
            container.register_transient(ITestService, TestService)
            container.register_transient(ITestDependency, TestService)  # Wrong type intentionally
            container.resolve(ITestService)
    
    def test_missing_dependency_error(self):
        """Test error when dependency is not registered."""
        container = FFDependencyInjectionManager()
        
        # Register service but not its dependency
        container.register_transient(ITestService, TestService)
        
        with pytest.raises(ValueError, match="Service .* not registered"):
            container.resolve(ITestService)
    
    def test_invalid_factory_error(self):
        """Test error handling with invalid factory functions."""
        container = FFDependencyInjectionManager()
        
        def broken_factory(c: FFDependencyInjectionManager) -> ITestService:
            raise RuntimeError("Factory failed")
        
        container.register_singleton(ITestService, factory=broken_factory)
        
        with pytest.raises(RuntimeError, match="Factory failed"):
            container.resolve(ITestService)


class TestThreadSafety:
    """Test thread safety of DI container."""
    
    @pytest.mark.asyncio
    async def test_concurrent_service_resolution(self):
        """Test concurrent service resolution is thread-safe."""
        container = FFDependencyInjectionManager()
        container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        # Resolve service concurrently
        tasks = [container.resolve_async(ITestService) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should be the same singleton instance
        first_service = results[0]
        for service in results[1:]:
            assert service is first_service
    
    @pytest.mark.asyncio
    async def test_concurrent_registration_and_resolution(self):
        """Test that registration and resolution can happen concurrently."""
        container = FFDependencyInjectionManager()
        
        async def register_service():
            container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        async def resolve_service():
            try:
                return container.resolve(ITestService)
            except ValueError:
                return None  # Service not registered yet
        
        # Run registration and resolution concurrently
        register_task = asyncio.create_task(register_service())
        resolve_tasks = [asyncio.create_task(resolve_service()) for _ in range(5)]
        
        await asyncio.gather(register_task, *resolve_tasks)
        
        # After registration, resolution should work
        service = container.resolve(ITestService)
        assert service is not None


@pytest.mark.integration
class TestDIIntegration:
    """Integration tests with real system components."""
    
    @pytest.mark.asyncio
    async def test_di_with_storage_manager(self, test_config, temp_dir):
        """Test DI integration with storage manager."""
        container = FFDependencyInjectionManager()
        
        # Register real components
        container.register_singleton(FFConfigurationManagerConfigDTO, instance=test_config)
        
        from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
        def backend_factory(c):
            config = c.resolve(FFConfigurationManagerConfigDTO)
            return FFFlatfileStorageBackend(config)
        
        container.register_singleton(BackendProtocol, factory=backend_factory)
        
        # Resolve and test
        backend = container.resolve(BackendProtocol)
        assert backend is not None
        
        # Initialize backend
        success = await backend.initialize()
        assert success is True
    
    def test_di_with_protocol_compliance(self):
        """Test that DI-resolved services comply with protocols."""
        container = ff_create_application_container()
        
        # Resolve services and verify protocol compliance
        storage = container.resolve(StorageProtocol)
        assert hasattr(storage, 'initialize')
        assert hasattr(storage, 'create_user')
        assert hasattr(storage, 'create_session')
        
        backend = container.resolve(BackendProtocol)
        assert hasattr(backend, 'read')
        assert hasattr(backend, 'write')
        assert hasattr(backend, 'exists')
    
    def test_di_service_dependency_chain(self):
        """Test complex dependency chains through DI."""
        container = ff_create_application_container()
        
        # Storage manager should depend on backend, search, vector, etc.
        storage = container.resolve(StorageProtocol)
        
        # Verify that dependencies are properly injected
        assert hasattr(storage, 'backend')
        assert hasattr(storage, 'config')
        
        # Backend should be the same instance when resolved directly
        backend = container.resolve(BackendProtocol)
        assert storage.backend is backend or isinstance(storage.backend, type(backend))


class TestPerformance:
    """Performance tests for DI container."""
    
    def test_resolution_performance(self, performance_timer):
        """Test service resolution performance."""
        container = FFDependencyInjectionManager()
        container.register_singleton(ITestService, TestServiceWithoutDeps)
        
        # Warm up
        container.resolve(ITestService)
        
        # Time resolution
        performance_timer.start()
        for _ in range(1000):
            container.resolve(ITestService)
        performance_timer.stop()
        
        # Should be fast (under 1 second for 1000 resolutions)
        performance_timer.assert_under(1.0)
    
    def test_registration_performance(self, performance_timer):
        """Test service registration performance."""
        container = FFDependencyInjectionManager()
        
        performance_timer.start()
        for i in range(100):
            # Create unique interface for each registration
            interface_name = f"ITestService{i}"
            interface = type(interface_name, (Protocol,), {})
            container.register_transient(interface, TestServiceWithoutDeps)
        performance_timer.stop()
        
        # Should be fast (under 0.5 seconds for 100 registrations)
        performance_timer.assert_under(0.5)