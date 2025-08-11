"""
Phase 4 validation script for Chat Application Bridge System.

Validates configuration factory, presets, and template management.
"""

import asyncio
import sys
import tempfile
import traceback
import json
from pathlib import Path

async def test_factory_creation():
    """Test configuration factory creation."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        templates = factory.list_templates()
        
        assert len(templates) >= 5  # Should have at least 5 templates
        assert "development" in templates
        assert "production" in templates
        assert "high_performance" in templates
        assert "lightweight" in templates
        assert "testing" in templates
        print(f"✓ Factory created with {len(templates)} templates")
        
        # Test template descriptions
        for name, info in templates.items():
            assert "description" in info
            assert "performance_mode" in info
            assert "features" in info
            assert "use_cases" in info
            assert "recommended_for" in info
        print("✓ All templates have required metadata")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_template_creation():
    """Test creating configurations from templates."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Test development template
        dev_config = factory.create_from_template("development", "./test_data")
        assert dev_config.storage_path == "./test_data"
        assert dev_config.environment == "development"
        assert dev_config.performance_mode == "balanced"
        assert dev_config.cache_size_mb == 50
        print("✓ Development template creation successful")
        
        # Test production template with overrides
        prod_config = factory.create_from_template(
            "production", 
            "./prod_data_test",
            {"cache_size_mb": 300, "enable_compression": False}
        )
        assert prod_config.cache_size_mb == 300
        assert prod_config.enable_compression is False
        assert prod_config.environment == "production"
        print("✓ Production template with overrides successful")
        
        # Test high_performance template
        hp_config = factory.create_from_template("high_performance", "./hp_data")
        assert hp_config.performance_mode == "speed"
        assert hp_config.enable_vector_search is False  # Disabled for performance
        assert hp_config.cache_size_mb == 500
        print("✓ High performance template creation successful")
        
        # Test lightweight template
        light_config = factory.create_from_template("lightweight", "./light_data")
        assert light_config.cache_size_mb == 25
        assert light_config.enable_analytics is False
        assert light_config.max_session_size_mb == 10
        print("✓ Lightweight template creation successful")
        
        # Test testing template
        test_config = factory.create_from_template("testing", "./test_env_data")
        assert test_config.environment == "test"
        assert test_config.cache_size_mb == 30
        assert test_config.max_session_size_mb == 5
        print("✓ Testing template creation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Template creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_environment_configs():
    """Test environment-specific configuration creation."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Test development environment
        dev_env = factory.create_for_environment("development", "./dev_env", "balanced")
        assert dev_env.environment == "development"
        assert dev_env.performance_mode == "balanced"
        assert dev_env.cache_size_mb == 50
        assert dev_env.enable_compression is False
        print("✓ Development environment config successful")
        
        # Test production environment
        prod_env = factory.create_for_environment("production", "./prod_env", "quality")
        assert prod_env.environment == "production"
        assert prod_env.performance_mode == "quality"
        assert prod_env.cache_size_mb == 200
        assert prod_env.backup_enabled is True
        print("✓ Production environment config successful")
        
        # Test test environment
        test_env = factory.create_for_environment("test", "./test_env", "speed")
        assert test_env.environment == "test"
        assert test_env.performance_mode == "speed"
        assert test_env.cache_size_mb == 30
        assert test_env.max_session_size_mb == 5
        print("✓ Test environment config successful")
        
        # Test staging environment (should use production template)
        staging_env = factory.create_for_environment("staging", "./staging_env")
        assert staging_env.environment == "staging"
        assert staging_env.cache_size_mb == 100
        assert staging_env.backup_enabled is True
        print("✓ Staging environment config successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment config test failed: {e}")
        traceback.print_exc()
        return False

async def test_use_case_configs():
    """Test use-case-specific configuration creation."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Test AI assistant use case
        ai_config = factory.create_for_use_case("ai_assistant", "./ai_data")
        assert ai_config.enable_vector_search is True
        assert ai_config.enable_analytics is True
        print("✓ AI assistant use case config successful")
        
        # Test high volume chat
        hv_config = factory.create_for_use_case("high_volume_chat", "./hv_data")
        assert hv_config.performance_mode == "speed"
        assert hv_config.enable_analytics is False
        print("✓ High volume chat use case config successful")
        
        # Test simple chat
        simple_config = factory.create_for_use_case("simple_chat", "./simple_data")
        assert simple_config.enable_vector_search is False
        assert simple_config.enable_streaming is False
        print("✓ Simple chat use case config successful")
        
        # Test gaming chat
        gaming_config = factory.create_for_use_case("gaming_chat", "./gaming_data")
        assert gaming_config.performance_mode == "speed"
        assert gaming_config.enable_analytics is False
        print("✓ Gaming chat use case config successful")
        
        # Test testing chat
        testing_config = factory.create_for_use_case("testing_chat", "./testing_data")
        assert testing_config.environment == "test"
        assert testing_config.cache_size_mb == 30
        print("✓ Testing chat use case config successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Use case config test failed: {e}")
        traceback.print_exc()
        return False

async def test_convenience_functions():
    """Test convenience factory functions."""
    try:
        from ff_chat_integration import (
            create_chat_config_for_development,
            create_chat_config_for_production,
            create_chat_config_for_testing,
            get_chat_app_presets
        )
        
        # Test development config creation
        dev_config = create_chat_config_for_development("./dev_test")
        assert dev_config.environment == "development"
        assert dev_config.storage_path == "./dev_test"
        print("✓ Development config convenience function works")
        
        # Test production config creation  
        prod_config = create_chat_config_for_production(
            "./prod_test_data",
            performance_level="speed"
        )
        assert prod_config.performance_mode == "speed"
        assert prod_config.environment == "production"
        print("✓ Production config convenience function works")
        
        # Test testing config creation
        test_config = create_chat_config_for_testing("./testing_test")
        assert test_config.environment == "test"
        assert test_config.storage_path == "./testing_test"
        print("✓ Testing config convenience function works")
        
        # Test presets
        presets = get_chat_app_presets()
        assert len(presets) >= 5
        assert "development" in presets
        assert "production" in presets
        assert "high_performance" in presets
        assert "lightweight" in presets
        assert "testing" in presets
        print(f"✓ Presets function returned {len(presets)} presets")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        traceback.print_exc()
        return False

async def test_validation_and_optimization():
    """Test configuration validation and optimization."""
    try:
        from ff_chat_integration import FFChatConfigFactory, ChatAppStorageConfig
        
        factory = FFChatConfigFactory()
        
        # Test good configuration
        good_config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced",
            cache_size_mb=100
        )
        
        results = factory.validate_and_optimize(good_config)
        assert results["valid"] is True
        assert "optimization_score" in results
        assert "estimated_performance" in results
        assert results["optimization_score"] > 0
        print("✓ Good configuration validation successful")
        
        # Test configuration with warnings
        warning_config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="speed",
            cache_size_mb=25,  # Small cache
            enable_vector_search=True  # May conflict with speed mode
        )
        
        warning_results = factory.validate_and_optimize(warning_config)
        assert warning_results["valid"] is True
        assert len(warning_results["warnings"]) > 0
        assert len(warning_results["recommendations"]) > 0
        print("✓ Configuration with warnings properly identified")
        
        # Test problematic configuration
        try:
            bad_config = ChatAppStorageConfig(
                storage_path="",  # Empty path
                performance_mode="invalid_mode",  # Invalid mode
                cache_size_mb=5  # Too small
            )
            # This should raise ConfigurationError due to validation in __post_init__
            print("✗ Should have raised ConfigurationError")
            return False
        except Exception:
            print("✓ Bad configuration properly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        traceback.print_exc()
        return False

async def test_bridge_preset_integration():
    """Test bridge integration with presets."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "preset_test")
            
            # Test preset creation
            bridge = await FFChatAppBridge.create_from_preset(
                "development", 
                storage_path
            )
            
            assert bridge._initialized
            config = bridge.get_standardized_config()
            assert config["environment"] == "development"
            print("✓ Bridge preset integration successful")
            
            await bridge.close()
            
            # Test use case creation
            bridge2 = await FFChatAppBridge.create_for_use_case(
                "simple_chat",
                storage_path
            )
            
            assert bridge2._initialized
            config2 = bridge2.get_standardized_config()
            print("✓ Bridge use case integration successful")
            
            await bridge2.close()
            
            # Test preset with overrides
            bridge3 = await FFChatAppBridge.create_from_preset(
                "production",
                storage_path,
                {"cache_size_mb": 250, "enable_analytics": False}
            )
            
            assert bridge3._initialized
            config3 = bridge3.get_standardized_config()
            assert config3["performance"]["cache_size_mb"] == 250
            print("✓ Bridge preset with overrides successful")
            
            await bridge3.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge preset integration test failed: {e}")
        traceback.print_exc()
        return False

async def test_migration_utility():
    """Test configuration migration from wrapper format."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Simulate old wrapper configuration
        wrapper_config = {
            "base_path": "./old_wrapper_data",
            "cache_size_limit": 150,
            "enable_vector_search": True,
            "enable_compression": True,
            "performance_mode": "speed",
            "environment": "production",
            "message_batch_size": 75,
            "history_page_size": 25
        }
        
        # Migrate to bridge configuration
        bridge_config = factory.migrate_from_wrapper_config(wrapper_config)
        
        assert bridge_config.storage_path == "./old_wrapper_data"
        assert bridge_config.cache_size_mb == 150
        assert bridge_config.enable_vector_search is True
        assert bridge_config.enable_compression is True
        assert bridge_config.performance_mode == "speed"
        assert bridge_config.environment == "production"
        assert bridge_config.message_batch_size == 75
        assert bridge_config.history_page_size == 25
        print("✓ Configuration migration successful")
        
        # Test migration with minimal wrapper config
        minimal_wrapper = {
            "base_path": "./minimal_data"
        }
        
        minimal_config = factory.migrate_from_wrapper_config(minimal_wrapper)
        assert minimal_config.storage_path == "./minimal_data"
        assert minimal_config.cache_size_mb == 100  # Default
        print("✓ Minimal configuration migration successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration utility test failed: {e}")
        traceback.print_exc()
        return False

async def test_template_file_operations():
    """Test template file loading and exporting."""
    try:
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test template export
            export_path = temp_path / "exported_development.json"
            success = factory.export_template("development", export_path)
            assert success is True
            assert export_path.exists()
            print("✓ Template export successful")
            
            # Verify exported template content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data["name"] == "development"
            assert "configuration" in exported_data
            assert "performance_optimizations" in exported_data
            assert "use_cases" in exported_data
            print("✓ Exported template content valid")
            
            # Test template loading from file
            loaded_template = factory.load_template_from_file(export_path)
            assert loaded_template.name == "development"
            assert loaded_template.description == "Optimized for development with debugging features"
            print("✓ Template loading from file successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Template file operations test failed: {e}")
        traceback.print_exc()
        return False

async def test_utility_functions():
    """Test additional utility functions."""
    try:
        from ff_chat_integration import (
            validate_chat_app_config,
            optimize_chat_config,
            get_recommended_config_for_use_case,
            ChatAppStorageConfig
        )
        
        # Test validation utility
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced"
        )
        
        issues = validate_chat_app_config(config)
        print(f"✓ Configuration validation utility returned {len(issues)} issues")
        
        # Test optimization utility
        optimization = optimize_chat_config(config)
        assert "optimization_score" in optimization
        assert "recommendations" in optimization
        print("✓ Configuration optimization utility successful")
        
        # Test use case recommendations
        recommendation = get_recommended_config_for_use_case("ai_assistant")
        assert "use_case" in recommendation
        assert "recommended_template" in recommendation
        assert recommendation["recommended_template"] == "feature_rich"
        print("✓ Use case recommendation utility successful")
        
        # Test unknown use case
        unknown_rec = get_recommended_config_for_use_case("unknown_use_case")
        assert "error" in unknown_rec or "recommended_template" in unknown_rec
        print("✓ Unknown use case handled properly")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 4 validation tests."""
    print("Phase 4 Validation - Configuration Factory and Presets")
    print("=" * 55)
    
    tests = [
        ("Factory Creation", test_factory_creation),
        ("Template Creation", test_template_creation),
        ("Environment Configs", test_environment_configs),
        ("Use Case Configs", test_use_case_configs),
        ("Convenience Functions", test_convenience_functions),
        ("Validation & Optimization", test_validation_and_optimization),
        ("Bridge Preset Integration", test_bridge_preset_integration),
        ("Migration Utility", test_migration_utility),
        ("Template File Operations", test_template_file_operations),
        ("Utility Functions", test_utility_functions)
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
    
    print(f"\n" + "=" * 55)
    print(f"Phase 4 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 4 implementation is ready for Phase 5!")
        return True
    else:
        print("✗ Phase 4 needs fixes before proceeding to Phase 5") 
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)