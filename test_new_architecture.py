#!/usr/bin/env python3
"""
Test script to verify the new architecture works correctly.

Tests configuration loading, dependency injection, and backward compatibility.
"""

import asyncio
from pathlib import Path

# Test new configuration system
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, create_default_config
from ff_class_configs.ff_storage_config import FFStorageConfig as NewStorageConfig

# Test backward compatibility
from ff_config_legacy_adapter import StorageConfig as FFLegacyStorageConfig, ff_load_config

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


def test_backward_compatibility():
    """Test backward compatibility with old StorageConfig."""
    print("Testing backward compatibility...")
    
    # Load config using old interface
    config = FFLegacyStorageConfig()
    
    # Test old attribute access
    print(f"✓ storage_base_path: {config.storage_base_path}")
    print(f"✓ max_message_size_bytes: {config.max_message_size_bytes}")
    print(f"✓ search_include_message_content: {config.search_include_message_content}")
    print(f"✓ vector_search_top_k: {config.vector_search_top_k}")
    print(f"✓ panel_max_personas: {config.panel_max_personas}")
    
    # Test that changes propagate
    config.storage_base_path = "./test_data"
    print(f"✓ Changed storage_base_path: {config.storage_base_path}")
    
    # Verify it's using the new system internally
    if hasattr(config, '_manager'):
        print(f"✓ Using new ConfigurationManager internally")
        print(f"✓ Manager storage path: {config._manager.storage.base_path}")
    
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
        # Create application container
        container = ff_create_application_container(environment="development")
        print("✓ Created application container")
        
        # Resolve configuration
        config = container.resolve(FFConfigurationManagerConfigDTO)
        print(f"✓ Resolved FFConfigurationManagerConfigDTO: {config.storage.base_path}")
        
        # Check if services are registered
        services = [
            ("FFConfigurationManagerConfigDTO", FFConfigurationManagerConfigDTO),
            ("StorageProtocol", StorageProtocol),
            ("SearchProtocol", SearchProtocol),
            ("VectorStoreProtocol", VectorStoreProtocol)
        ]
        
        for name, interface in services:
            if container.is_registered(interface):
                print(f"✓ {name} is registered")
            else:
                print(f"✗ {name} is NOT registered")
        
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
    test_backward_compatibility()
    test_dependency_injection()
    test_config_summary()
    
    # Run async tests
    asyncio.run(test_application_container())
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()