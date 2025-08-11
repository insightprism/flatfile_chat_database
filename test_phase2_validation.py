"""
Phase 2 validation script for Chat Application Bridge System.

Validates bridge implementation and factory methods.
"""

import asyncio
import sys
import tempfile
import traceback
from pathlib import Path

async def test_config_class():
    """Test ChatAppStorageConfig class."""
    try:
        from ff_chat_integration import ChatAppStorageConfig, ConfigurationError
        
        # Test valid configuration
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced",
            cache_size_mb=100
        )
        assert config.storage_path == "./test_data"
        assert config.performance_mode == "balanced"
        print("✓ ChatAppStorageConfig creation successful")
        
        # Test configuration validation
        try:
            invalid_config = ChatAppStorageConfig(
                storage_path="./test_data",
                performance_mode="invalid_mode"  # Invalid mode
            )
            print("✗ Configuration validation should have failed")
            return False
        except ConfigurationError:
            print("✓ Configuration validation working correctly")
        
        # Test configuration serialization
        config_dict = config.to_dict()
        assert "storage_path" in config_dict
        assert config_dict["performance"]["mode"] == "balanced"
        print("✓ Configuration serialization working")
        
        return True
        
    except Exception as e:
        print(f"✗ Config class test failed: {e}")
        traceback.print_exc()
        return False

async def test_bridge_creation():
    """Test FFChatAppBridge creation without initialization."""
    try:
        from ff_chat_integration import FFChatAppBridge, ChatAppStorageConfig
        
        # Test bridge creation with config object
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced"
        )
        
        bridge = FFChatAppBridge(config)
        assert bridge is not None
        assert bridge.config.storage_path == "./test_data"
        assert not bridge._initialized  # Should not be initialized yet
        print("✓ Bridge creation with config object successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_preset_functionality():
    """Test preset configuration functionality."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        # Test preset configuration retrieval
        speed_preset = FFChatAppBridge._get_preset_config("speed")
        assert speed_preset["performance_mode"] == "speed"
        assert speed_preset["cache_size_mb"] == 200
        print("✓ Speed preset configuration correct")
        
        balanced_preset = FFChatAppBridge._get_preset_config("balanced")
        assert balanced_preset["performance_mode"] == "balanced"
        assert balanced_preset["backup_enabled"] is True
        print("✓ Balanced preset configuration correct")
        
        quality_preset = FFChatAppBridge._get_preset_config("quality")
        assert quality_preset["performance_mode"] == "quality"
        assert quality_preset["enable_compression"] is True
        print("✓ Quality preset configuration correct")
        
        # Test invalid preset
        try:
            invalid_preset = FFChatAppBridge._get_preset_config("invalid_preset")
            print("✗ Should have raised ConfigurationError for invalid preset")
            return False
        except Exception:  # Should catch ConfigurationError
            print("✓ Invalid preset handling working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Preset functionality test failed: {e}")
        traceback.print_exc()
        return False

async def test_configuration_validation():
    """Test configuration validation functionality."""
    try:
        from ff_chat_integration import ChatAppStorageConfig, ConfigurationError
        
        # Test valid configuration
        valid_config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="speed",
            cache_size_mb=50,
            message_batch_size=100
        )
        assert len(valid_config.validate()) == 0  # No validation errors
        print("✓ Valid configuration passes validation")
        
        # Test cache size validation
        try:
            invalid_cache_config = ChatAppStorageConfig(
                storage_path="./test_data",
                cache_size_mb=5  # Too small
            )
            print("✗ Should have failed validation for small cache size")
            return False
        except ConfigurationError:
            print("✓ Cache size validation working correctly")
        
        # Test batch size validation
        try:
            invalid_batch_config = ChatAppStorageConfig(
                storage_path="./test_data",
                message_batch_size=0  # Too small
            )
            print("✗ Should have failed validation for invalid batch size")
            return False
        except ConfigurationError:
            print("✓ Batch size validation working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation test failed: {e}")
        traceback.print_exc()
        return False

async def test_bridge_methods():
    """Test bridge methods without full initialization."""
    try:
        from ff_chat_integration import FFChatAppBridge, ChatAppStorageConfig
        
        # Create bridge without initialization
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced"
        )
        bridge = FFChatAppBridge(config)
        
        # Test methods that should work without initialization
        
        # Test that get_standardized_config raises error when not initialized
        try:
            bridge.get_standardized_config()
            print("✗ Should have raised RuntimeError for uninitialized bridge")
            return False
        except RuntimeError:
            print("✓ Uninitialized bridge correctly raises RuntimeError")
        
        # Test that get_data_layer raises error when not initialized
        try:
            bridge.get_data_layer()
            print("✗ Should have raised RuntimeError for uninitialized bridge")
            return False
        except RuntimeError:
            print("✓ Uninitialized data layer access correctly raises RuntimeError")
        
        # Test that get_capabilities raises error when not initialized
        try:
            await bridge.get_capabilities()
            print("✗ Should have raised RuntimeError for uninitialized bridge")
            return False
        except RuntimeError:
            print("✓ Uninitialized capabilities access correctly raises RuntimeError")
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge methods test failed: {e}")
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test error handling in bridge."""
    try:
        from ff_chat_integration import FFChatAppBridge, ConfigurationError, InitializationError
        
        # Test invalid performance mode
        try:
            bridge = await FFChatAppBridge.create_for_chat_app(
                storage_path="./test_data",
                options={"performance_mode": "invalid_mode"}
            )
            print("✗ Should have raised ConfigurationError for invalid performance mode")
            return False
        except ConfigurationError:
            print("✓ Invalid performance mode error handling working")
        except InitializationError as e:
            # InitializationError with ConfigurationError as cause is also acceptable
            if "Configuration validation failed" in str(e):
                print("✓ Invalid performance mode error handling working")
            else:
                print(f"✗ Unexpected InitializationError: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 2 validation tests."""
    print("Phase 2 Validation - Chat Application Bridge Implementation")
    print("=" * 65)
    
    tests = [
        ("Configuration Class", test_config_class),
        ("Bridge Creation", test_bridge_creation),
        ("Preset Functionality", test_preset_functionality),
        ("Configuration Validation", test_configuration_validation),
        ("Bridge Methods", test_bridge_methods),
        ("Error Handling", test_error_handling)
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
            traceback.print_exc()
    
    print(f"\n" + "=" * 65)
    print(f"Phase 2 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 2 implementation is ready for Phase 3!")
        print("\nKey Phase 2 achievements:")
        print("- ✅ ChatAppStorageConfig class with validation")
        print("- ✅ FFChatAppBridge class with factory methods")
        print("- ✅ Performance presets (speed, balanced, quality)")
        print("- ✅ Direct integration approach (no wrappers)")
        print("- ✅ Comprehensive error handling and validation")
        return True
    else:
        print("✗ Phase 2 needs fixes before proceeding to Phase 3")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)