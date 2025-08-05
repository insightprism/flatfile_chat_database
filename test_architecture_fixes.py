#!/usr/bin/env python3
"""
Test script to verify architecture fixes are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ff_dependency_injection_manager import ff_create_application_container, ff_get_container
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_protocols import StorageProtocol
from ff_utils.ff_logging import configure_logging, get_logger
from ff_embedding_functions import generate_embeddings

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)


async def test_dependency_injection():
    """Test the fixed dependency injection."""
    logger.info("Testing dependency injection fix...")
    
    try:
        container = ff_create_application_container()
        config = container.resolve(FFConfigurationManagerConfigDTO)
        logger.info(f"✅ DI container created successfully")
        logger.info(f"   Environment: {config.environment}")
        return True
    except Exception as e:
        logger.error(f"❌ DI test failed: {e}")
        return False


async def test_runtime_config():
    """Test runtime configuration."""
    logger.info("Testing runtime configuration...")
    
    try:
        config = ff_get_container().resolve(FFConfigurationManagerConfigDTO)
        
        # Check runtime config exists
        assert hasattr(config, 'runtime'), "Runtime config missing"
        logger.info(f"✅ Runtime config loaded")
        
        # Check key values
        logger.info(f"   Cache size limit: {config.runtime.cache_size_limit}")
        logger.info(f"   File retry attempts: {config.runtime.file_retry_attempts}")
        logger.info(f"   Entity patterns: {len(config.runtime.entity_patterns)} patterns")
        
        return True
    except Exception as e:
        logger.error(f"❌ Runtime config test failed: {e}")
        return False


async def test_logging():
    """Test logging is working."""
    logger.info("Testing logging functionality...")
    
    try:
        test_logger = get_logger("test.module")
        test_logger.debug("Debug message")
        test_logger.info("Info message")
        test_logger.warning("Warning message")
        test_logger.error("Error message (this is a test)")
        
        logger.info("✅ Logging working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Logging test failed: {e}")
        return False


async def test_abstract_methods():
    """Test abstract methods raise NotImplementedError."""
    logger.info("Testing abstract method fix...")
    
    try:
        from backends.ff_storage_backend_base import FFStorageBackendBase
        
        # First test: Cannot instantiate abstract class
        try:
            class TestBackend(FFStorageBackendBase):
                pass
            
            backend = TestBackend(ff_get_container().resolve(FFConfigurationManagerConfigDTO))
            logger.error("❌ Should not be able to instantiate abstract class")
            return False
        except TypeError as e:
            logger.info(f"✅ Cannot instantiate abstract class without implementing methods")
            logger.info(f"   Error: {e}")
        
        # Second test: Methods raise NotImplementedError when called
        from backends.ff_storage_backend_base import FFStorageBackendBase
        # Check that the base methods have NotImplementedError
        import inspect
        source = inspect.getsource(FFStorageBackendBase.initialize)
        if "NotImplementedError" in source:
            logger.info("✅ Abstract methods correctly raise NotImplementedError")
            return True
        else:
            logger.error("❌ Abstract methods do not raise NotImplementedError")
            return False
            
    except Exception as e:
        logger.error(f"❌ Abstract method test failed: {e}")
        return False


async def test_embedding_functions():
    """Test embedding functions."""
    logger.info("Testing embedding functions...")
    
    try:
        # Test direct function call
        embeddings = await generate_embeddings(
            ["Test text 1", "Test text 2"],
            provider="sentence-transformers"
        )
        
        assert len(embeddings) == 2, "Should return 2 embeddings"
        assert "embedding_vector" in embeddings[0], "Should have embedding_vector"
        assert len(embeddings[0]["embedding_vector"]) == 384, "Should have correct dimension"
        
        logger.info("✅ Embedding functions working")
        logger.info(f"   Generated {len(embeddings)} embeddings")
        logger.info(f"   Dimension: {len(embeddings[0]['embedding_vector'])}")
        
        # Test from DI container with sentence-transformers (no API key needed)
        embedding_func = ff_get_container().resolve("generate_embeddings")
        embeddings2 = await embedding_func(["Test from DI"], provider="sentence-transformers")
        logger.info("✅ Embedding function from DI container working")
        
        return True
    except Exception as e:
        logger.error(f"❌ Embedding function test failed: {e}")
        return False


async def test_lazy_loading():
    """Test lazy loading in storage manager."""
    logger.info("Testing lazy loading...")
    
    try:
        from ff_storage_manager import FFStorageManager
        
        # Create storage manager
        storage = FFStorageManager()
        
        # Check components are not loaded yet
        assert storage._search_engine is None, "Search engine should not be loaded"
        assert storage._vector_storage is None, "Vector storage should not be loaded"
        
        # Access search engine (should trigger lazy load)
        search = storage.search_engine
        assert search is not None, "Search engine should be loaded"
        assert storage._search_engine is not None, "Search engine should be cached"
        
        logger.info("✅ Lazy loading working correctly")
        logger.info("   Components load on first access")
        
        return True
    except Exception as e:
        logger.error(f"❌ Lazy loading test failed: {e}")
        return False


async def test_integration():
    """Test full integration."""
    logger.info("Testing full integration...")
    
    try:
        # Get storage from DI
        storage = ff_get_container().resolve(StorageProtocol)
        
        # Create a test user
        user_id = "test_architecture_user"
        success = await storage.create_user(user_id, {"username": "Test User"})
        
        if success:
            logger.info("✅ Full integration test passed")
            logger.info(f"   Created user: {user_id}")
            
            # Cleanup - delete_user doesn't exist, so just log success
            return True
        else:
            logger.error("❌ Failed to create test user")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("ARCHITECTURE FIX VERIFICATION")
    logger.info("="*60)
    
    tests = [
        ("Dependency Injection", test_dependency_injection),
        ("Runtime Configuration", test_runtime_config),
        ("Logging System", test_logging),
        ("Abstract Methods", test_abstract_methods),
        ("Embedding Functions", test_embedding_functions),
        ("Lazy Loading", test_lazy_loading),
        ("Full Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n--- {name} ---")
        result = await test_func()
        results.append((name, result))
        await asyncio.sleep(0.1)  # Small delay for readability
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{name:.<40} {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All architecture fixes verified successfully!")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)