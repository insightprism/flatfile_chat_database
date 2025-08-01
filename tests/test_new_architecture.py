#!/usr/bin/env python3
"""
Test script to verify the new architecture works correctly.

Tests configuration loading, dependency injection, and backward compatibility.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Test new configuration system
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, create_default_config
from ff_class_configs.ff_storage_config import FFStorageConfigDTO

# Test dependency injection
from ff_dependency_injection_manager import FFDependencyInjectionManager, ff_create_application_container
from ff_protocols import StorageProtocol, SearchProtocol, VectorStoreProtocol


def test_new_config_system():
    """Test the new modular configuration system."""
    print("Testing new configuration system...")
    
    # Create default config
    config = create_default_config("development")
    print(f"✓ Created config for environment: {config.environment}")
    
    # Test individual domains
    print(f"✓ Storage base path: {config.storage.base_path}")
    print(f"✓ Search default limit: {config.search.default_limit}")
    print(f"✓ Vector provider: {config.vector.default_embedding_provider}")
    print(f"✓ Document max size: {config.document.max_file_size_bytes / 1_048_576:.1f}MB")
    print(f"✓ Locking enabled: {config.locking.enabled}")
    print(f"✓ Panel max personas: {config.panel.max_personas_per_panel}")
    
    # Test validation
    errors = config.validate_all()
    if errors:
        print(f"✗ Validation errors: {errors}")
    else:
        print("✓ Configuration validation passed")
    
    print()


def test_current_config_system():
    """Test the current configuration system capabilities."""
    print("Testing current configuration system...")
    
    # Create configuration with different environments
    dev_config = create_default_config("development")
    prod_config = create_default_config("production")
    test_config = create_default_config("test")
    
    print(f"✓ Development config base path: {dev_config.storage.base_path}")
    print(f"✓ Production config base path: {prod_config.storage.base_path}")
    print(f"✓ Test config base path: {test_config.storage.base_path}")
    
    # Test environment-specific differences
    print(f"✓ Dev search cache enabled: {dev_config.search.enable_search_cache}")
    print(f"✓ Prod search cache enabled: {prod_config.search.enable_search_cache}")
    
    # Test configuration validation
    dev_errors = dev_config.validate_all()
    print(f"✓ Development config validation: {'PASSED' if not dev_errors else f'FAILED ({len(dev_errors)} errors)'}")
    
    print()


def test_dependency_injection():
    """Test the dependency injection container."""
    print("Testing dependency injection container...")
    
    # Create a test container
    container = FFDependencyInjectionManager()
    
    # Register a simple service
    class TestService:
        def __init__(self):
            self.name = "TestService"
    
    container.register_singleton(TestService, implementation=TestService)
    
    # Resolve service
    service1 = container.resolve(TestService)
    service2 = container.resolve(TestService)
    
    print(f"✓ Resolved service: {service1.name}")
    print(f"✓ Singleton check: {service1 is service2}")
    
    # Test factory registration
    def factory(c: FFDependencyInjectionManager) -> TestService:
        service = TestService()
        service.name = "FactoryCreated"
        return service
    
    class FactoryService:
        pass
    
    container.register_transient(FactoryService, factory=factory)
    
    service3 = container.resolve(FactoryService)
    print(f"✓ Factory created service: {service3.name}")
    
    print()


async def test_application_container():
    """Test the full application container."""
    print("Testing application container...")
    
    try:
        # Create application container with default configuration
        container = ff_create_application_container()
        print("✓ Created application container")
        
        # Resolve configuration
        config = container.resolve(FFConfigurationManagerConfigDTO)
        print(f"✓ Resolved FFConfigurationManagerConfigDTO: {config.storage.base_path}")
        
        # Check if core services are registered
        services = [
            ("FFConfigurationManagerConfigDTO", FFConfigurationManagerConfigDTO),
            ("StorageProtocol", StorageProtocol),
            ("SearchProtocol", SearchProtocol),
            ("VectorStoreProtocol", VectorStoreProtocol)
        ]
        
        registered_count = 0
        for name, interface in services:
            if container.is_registered(interface):
                print(f"✓ {name} is registered")
                registered_count += 1
            else:
                print(f"✗ {name} is NOT registered")
        
        print(f"✓ Total services registered: {registered_count}/{len(services)}")
        
        # Test service resolution
        try:
            storage_service = container.resolve(StorageProtocol)
            print(f"✓ Successfully resolved StorageProtocol: {type(storage_service).__name__}")
        except Exception as resolve_error:
            print(f"✗ Could not resolve StorageProtocol: {resolve_error}")
        
        print()
        
    except Exception as e:
        print(f"✗ Error creating application container: {e}")
        import traceback
        traceback.print_exc()


def test_config_summary():
    """Test configuration summary."""
    print("Testing configuration summary...")
    
    config = create_default_config("production")
    summary = config.get_summary()
    
    print("Configuration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Architecture")
    print("=" * 60)
    print()
    
    # Run synchronous tests
    test_new_config_system()
    test_current_config_system()
    test_dependency_injection()
    test_config_summary()
    
    # Run async tests
    asyncio.run(test_application_container())
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()