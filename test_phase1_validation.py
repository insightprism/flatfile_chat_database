"""
Phase 1 validation script for Chat Application Bridge System.

Validates that the module structure and exception handling are working correctly.
"""

import sys
import traceback
from pathlib import Path

def test_module_import():
    """Test that the module can be imported successfully."""
    try:
        import ff_chat_integration
        print("✓ Module import successful")
        return True
    except ImportError as e:
        print(f"✗ Module import failed: {e}")
        return False

def test_exception_classes():
    """Test that all exception classes work correctly."""
    try:
        from ff_chat_integration import (
            ChatIntegrationError,
            ConfigurationError,
            InitializationError,
            StorageError,
            SearchError,
            PerformanceError
        )
        
        # Test base exception
        try:
            raise ChatIntegrationError(
                "Test error",
                context={"test": True},
                suggestions=["This is a test"]
            )
        except ChatIntegrationError as e:
            assert "Test error" in str(e)
            assert e.context["test"] is True
            assert "This is a test" in e.suggestions
            print("✓ ChatIntegrationError working correctly")
        
        # Test configuration error
        try:
            raise ConfigurationError(
                "Invalid config",
                config_field="test_field",
                config_value="invalid_value"
            )
        except ConfigurationError as e:
            assert e.context["config_field"] == "test_field"
            assert e.context["config_value"] == "invalid_value"
            print("✓ ConfigurationError working correctly")
        
        # Test utility functions
        from ff_chat_integration import create_validation_error
        
        error = create_validation_error("test_field", "bad_value", "good_format")
        assert isinstance(error, ConfigurationError)
        assert "test_field" in str(error)
        print("✓ Utility functions working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Exception classes test failed: {e}")
        traceback.print_exc()
        return False

def test_module_metadata():
    """Test module metadata and info functions."""
    try:
        import ff_chat_integration
        
        version = ff_chat_integration.get_version()
        assert version == "1.0.0"
        print(f"✓ Version info correct: {version}")
        
        info = ff_chat_integration.get_module_info()
        assert info["name"] == "ff_chat_integration"
        assert info["version"] == "1.0.0"
        assert len(info["exception_classes"]) == 6
        print("✓ Module metadata working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Module metadata test failed: {e}")
        return False

def main():
    """Run all Phase 1 validation tests."""
    print("Phase 1 Validation - Chat Application Bridge Infrastructure")
    print("=" * 60)
    
    tests = [
        ("Module Import", test_module_import),
        ("Exception Classes", test_exception_classes),
        ("Module Metadata", test_module_metadata)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"Test {test_name} failed!")
    
    print(f"\n" + "=" * 60)
    print(f"Phase 1 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 1 implementation is ready for Phase 2!")
        return True
    else:
        print("✗ Phase 1 needs fixes before proceeding to Phase 2")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)