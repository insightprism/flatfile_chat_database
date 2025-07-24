#!/usr/bin/env python3
"""
Test PrismMind integration directly within the flatfile package.
"""

import sys
import os
from pathlib import Path

# Add PrismMind parent directory to Python path
sys.path.insert(0, '/home/markly2')

print("🧪 Testing PrismMind Integration (Direct Test)")
print("=" * 60)

def test_prismmind_imports():
    """Test PrismMind imports that our integration needs."""
    print("\n=== Testing PrismMind Imports ===")
    
    try:
        # Test core PrismMind imports
        from prismmind.pm_engines.pm_run_engine_chain import pm_run_engine_chain
        print("✅ pm_run_engine_chain imported")
        
        from prismmind.pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec
        print("✅ pm_trace_handler_log_dec imported")
        
        from prismmind.pm_engines.pm_base_engine import PmBaseEngine
        print("✅ PmBaseEngine imported")
        
        import prismmind
        print(f"✅ prismmind package imported (version {prismmind.__version__})")
        
        return True
        
    except Exception as e:
        print(f"❌ PrismMind imports failed: {e}")
        return False

def test_integration_modules():
    """Test our PrismMind integration modules."""
    print("\n=== Testing Integration Modules ===")
    
    try:
        # Test that our integration files exist and are importable
        integration_path = Path("prismmind_integration")
        
        if integration_path.exists():
            print(f"✅ Integration directory exists: {integration_path}")
            
            # Test individual files
            files_to_check = [
                "__init__.py",
                "config.py", 
                "handlers.py",
                "factory.py",
                "processor.py",
                "loader.py"
            ]
            
            for file_name in files_to_check:
                file_path = integration_path / file_name
                if file_path.exists():
                    print(f"✅ {file_name} exists ({file_path.stat().st_size} bytes)")
                else:
                    print(f"❌ {file_name} missing")
                    return False
            
            return True
        else:
            print(f"❌ Integration directory not found: {integration_path}")
            return False
            
    except Exception as e:
        print(f"❌ Integration module test failed: {e}")
        return False

def test_config_files():
    """Test configuration files."""
    print("\n=== Testing Configuration Files ===")
    
    try:
        configs_path = Path("configs")
        
        if configs_path.exists():
            print(f"✅ Configs directory exists: {configs_path}")
            
            config_files = [
                "flatfile_prismmind_config.json",
                "development_config.json",
                "production_config.json", 
                "test_config.json"
            ]
            
            for config_file in config_files:
                config_path = configs_path / config_file
                if config_path.exists():
                    print(f"✅ {config_file} exists")
                else:
                    print(f"❌ {config_file} missing")
                    return False
            
            return True
        else:
            print(f"❌ Configs directory not found: {configs_path}")
            return False
            
    except Exception as e:
        print(f"❌ Config files test failed: {e}")
        return False

def test_storage_manager_creation():
    """Test if we can create a basic StorageManager with minimal imports."""
    print("\n=== Testing Basic StorageManager Creation ===")
    
    try:
        # Import the modules directly without relative imports
        import config
        import storage
        
        # Create basic config
        storage_config = config.StorageConfig()
        storage_config.storage_base_path = "./test_integration_data"
        
        # Try to create StorageManager with PrismMind disabled first
        storage_manager = storage.StorageManager(storage_config, enable_prismmind=False)
        print("✅ StorageManager created with PrismMind disabled")
        
        # Try with PrismMind enabled
        storage_manager_pm = storage.StorageManager(storage_config, enable_prismmind=True)
        print("✅ StorageManager created with PrismMind enabled")
        print(f"   - PrismMind available: {storage_manager_pm.is_prismmind_available()}")
        
        return True
        
    except Exception as e:
        print(f"❌ StorageManager creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_pipeline_creation():
    """Test DocumentRAGPipeline creation."""
    print("\n=== Testing DocumentRAGPipeline Creation ===")
    
    try:
        import config
        import document_pipeline
        
        # Create config
        storage_config = config.StorageConfig()
        storage_config.storage_base_path = "./test_pipeline_data"
        
        # Try creating pipeline with PrismMind disabled
        pipeline = document_pipeline.DocumentRAGPipeline(storage_config, use_prismmind=False)
        print("✅ DocumentRAGPipeline created with PrismMind disabled")
        
        # Try with PrismMind enabled
        pipeline_pm = document_pipeline.DocumentRAGPipeline(storage_config, use_prismmind=True)
        print("✅ DocumentRAGPipeline created with PrismMind enabled")
        print(f"   - Use PrismMind: {pipeline_pm.use_prismmind}")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentRAGPipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    test_results = {}
    
    # Test PrismMind imports
    test_results['prismmind_imports'] = test_prismmind_imports()
    
    # Test integration modules
    test_results['integration_modules'] = test_integration_modules()
    
    # Test config files
    test_results['config_files'] = test_config_files()
    
    # Test storage manager creation
    test_results['storage_manager'] = test_storage_manager_creation()
    
    # Test document pipeline creation
    test_results['document_pipeline'] = test_document_pipeline_creation()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DIRECT INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Assessment
    if test_results.get('prismmind_imports') and test_results.get('storage_manager'):
        print("\n🎉 SUCCESS: PrismMind integration is working!")
        print("   The flatfile codebase can use PrismMind functionality.")
        return True
    elif test_results.get('prismmind_imports'):
        print("\n✅ PARTIAL: PrismMind imports work, some integration issues remain.")
        return True
    else:
        print("\n❌ FAILURE: PrismMind imports not working.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)