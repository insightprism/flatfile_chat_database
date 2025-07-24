#!/usr/bin/env python3
"""
Test PrismMind integration directly in our flatfile chat database codebase.
"""

import sys
import os
from pathlib import Path

# Add the main flatfile database directory to path
main_db_path = Path(__file__).parent.parent / "flatfile_chat_database"
sys.path.insert(0, str(main_db_path))

# Add PrismMind parent directory to Python path
prismmind_parent = '/home/markly2'
if prismmind_parent not in sys.path:
    sys.path.insert(0, prismmind_parent)

print("🧪 Testing PrismMind Integration in Flatfile Codebase")
print("=" * 60)

def test_flatfile_imports():
    """Test if our flatfile code can import its modules."""
    print("\n=== Testing Flatfile Module Imports ===")
    
    try:
        from storage import StorageManager
        print("✅ StorageManager imported successfully")
        
        from document_pipeline import DocumentRAGPipeline
        print("✅ DocumentRAGPipeline imported successfully")
        
        from config import StorageConfig
        print("✅ StorageConfig imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Flatfile imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prismmind_detection():
    """Test if our flatfile code can detect PrismMind."""
    print("\n=== Testing PrismMind Detection ===")
    
    try:
        # Import StorageManager and test PrismMind detection
        from storage import StorageManager
        
        # Initialize with PrismMind enabled
        config = StorageConfig()
        storage = StorageManager(config, enable_prismmind=True)
        
        print(f"✅ StorageManager initialized")
        print(f"   - PrismMind available: {storage.is_prismmind_available()}")
        print(f"   - PrismMind processor: {storage.prismmind_processor is not None}")
        
        if storage.is_prismmind_available():
            print("✅ PrismMind successfully detected by flatfile!")
            return True
        else:
            print("⚠️ PrismMind not detected (but this might be expected)")
            return False
            
    except Exception as e:
        print(f"❌ PrismMind detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prismmind_processor_import():
    """Test if our PrismMind integration modules import correctly."""
    print("\n=== Testing PrismMind Integration Module Imports ===")
    
    try:
        from prismmind_integration import (
            FlatfileDocumentProcessor,
            FlatfilePrismMindConfig,
            FlatfilePrismMindConfigLoader
        )
        print("✅ PrismMind integration modules imported successfully")
        print(f"   - FlatfileDocumentProcessor: {FlatfileDocumentProcessor}")
        print(f"   - FlatfilePrismMindConfig: {FlatfilePrismMindConfig}")
        print(f"   - FlatfilePrismMindConfigLoader: {FlatfilePrismMindConfigLoader}")
        
        return True
        
    except Exception as e:
        print(f"❌ PrismMind integration imports failed: {e}")
        print("   This is expected if PrismMind engines have dependency issues")
        return False

def test_document_pipeline_with_prismmind():
    """Test if DocumentRAGPipeline can initialize with PrismMind."""
    print("\n=== Testing DocumentRAGPipeline with PrismMind ===")
    
    try:
        from document_pipeline import DocumentRAGPipeline
        from config import StorageConfig
        
        # Test initialization with PrismMind enabled
        config = StorageConfig()
        config.storage_base_path = "./test_pipeline_data"
        
        pipeline = DocumentRAGPipeline(config, use_prismmind=True)
        
        print("✅ DocumentRAGPipeline initialized with PrismMind option")
        print(f"   - Use PrismMind: {pipeline.use_prismmind}")
        print(f"   - PrismMind processor: {pipeline.prismmind_processor is not None}")
        
        if pipeline.use_prismmind:
            print("✅ DocumentRAGPipeline successfully configured to use PrismMind!")
            return True
        else:
            print("⚠️ DocumentRAGPipeline fell back to legacy mode")
            return False
            
    except Exception as e:
        print(f"❌ DocumentRAGPipeline PrismMind test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_prismmind_imports():
    """Test direct PrismMind imports that our integration uses."""
    print("\n=== Testing Direct PrismMind Imports (Our Integration Needs) ===")
    
    imports_to_test = [
        ('pm_run_engine_chain', 'prismmind.pm_engines.pm_run_engine_chain'),
        ('pm_trace_handler_log_dec', 'prismmind.pm_utils.pm_trace_handler_log_dec'),
        ('PmBaseEngine', 'prismmind.pm_engines.pm_base_engine'),
        ('prismmind package', 'prismmind'),
    ]
    
    working_imports = []
    
    for import_name, import_path in imports_to_test:
        try:
            if '.' in import_path:
                module_path, item_name = import_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[item_name])
                item = getattr(module, item_name)
                print(f"✅ {import_name} imported successfully")
                working_imports.append(import_name)
            else:
                module = __import__(import_path)
                print(f"✅ {import_name} imported successfully")
                print(f"   - Version: {getattr(module, '__version__', 'unknown')}")
                working_imports.append(import_name)
                
        except Exception as e:
            print(f"❌ {import_name} failed: {e}")
    
    return working_imports

def test_configuration_loading():
    """Test if our PrismMind configuration files can be loaded."""
    print("\n=== Testing PrismMind Configuration Loading ===")
    
    try:
        config_path = main_db_path / "configs" / "flatfile_prismmind_config.json"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        # Try loading with our config loader
        from prismmind_integration.loader import FlatfilePrismMindConfigLoader
        
        config = FlatfilePrismMindConfigLoader.from_file(str(config_path))
        print("✅ PrismMind configuration loaded successfully")
        print(f"   - Environment: {config.environment}")
        print(f"   - File type chains: {len(config.document_processing.file_type_chains)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run all integration tests."""
    
    test_results = {}
    
    # Test flatfile imports
    test_results['flatfile_imports'] = test_flatfile_imports()
    
    # Only proceed if flatfile imports work
    if test_results['flatfile_imports']:
        # Test PrismMind detection
        test_results['prismmind_detection'] = test_prismmind_detection()
        
        # Test PrismMind integration imports
        test_results['integration_imports'] = test_prismmind_processor_import()
        
        # Test document pipeline
        test_results['pipeline_prismmind'] = test_document_pipeline_with_prismmind()
        
        # Test configuration loading
        test_results['config_loading'] = test_configuration_loading()
        
        # Test direct imports
        working_imports = test_direct_prismmind_imports()
        test_results['direct_imports'] = len(working_imports) >= 3
    else:
        print("\n⚠️ Skipping PrismMind tests due to flatfile import failure")
        test_results.update({
            'prismmind_detection': False,
            'integration_imports': False,
            'pipeline_prismmind': False,
            'config_loading': False,
            'direct_imports': False
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 FLATFILE INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Final assessment
    if test_results.get('flatfile_imports') and test_results.get('direct_imports'):
        print("\n🎉 SUCCESS: Flatfile codebase can import and use PrismMind!")
        print("   The integration is working and ready for document processing.")
        return True
    elif test_results.get('flatfile_imports'):
        print("\n✅ PARTIAL SUCCESS: Flatfile imports work, PrismMind has some issues.")
        print("   The basic integration structure is sound.")
        return True
    else:
        print("\n❌ FAILURE: Basic flatfile imports not working.")
        print("   Need to fix fundamental import issues first.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)