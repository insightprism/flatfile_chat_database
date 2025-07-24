#!/usr/bin/env python3
"""
Simple test for PrismMind integration with flatfile chat database.
"""

import sys
import asyncio
import time
from pathlib import Path
import json

# Add the main flatfile database directory to path
main_db_path = Path(__file__).parent.parent / "flatfile_chat_database"
sys.path.insert(0, str(main_db_path))

# Add PrismMind engines to path  
sys.path.append('/home/markly2/PrismMind_v2/pm_engines')

# Test files
TEST_FILES = [
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/test_file.txt',
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/mp_earnings_2024q4.pdf'
]

def test_imports():
    """Test if we can import the required modules"""
    print("=== Testing Module Imports ===")
    
    # Test core imports
    try:
        import config
        print("✓ config module imported")
    except ImportError as e:
        print(f"❌ config import failed: {e}")
        return False
    
    try:
        import storage
        print("✓ storage module imported") 
    except ImportError as e:
        print(f"❌ storage import failed: {e}")
        return False
        
    try:
        import document_pipeline
        print("✓ document_pipeline module imported")
    except ImportError as e:
        print(f"❌ document_pipeline import failed: {e}")
        return False
    
    # Test PrismMind imports
    print("\n--- Testing PrismMind Module Availability ---")
    prismmind_modules = [
        'pm_trace_logger',
        'pm_engine_config', 
        'pm_run_engine_chain',
        'pm_resolve_input_source_async'
    ]
    
    available_count = 0
    for module_name in prismmind_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name} available")
            available_count += 1
        except ImportError:
            print(f"❌ {module_name} not available")
    
    if available_count == len(prismmind_modules):
        print("✓ All PrismMind modules available")
        return True
    else:
        print(f"⚠️ Only {available_count}/{len(prismmind_modules)} PrismMind modules available")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        config_path = main_db_path / "configs" / "flatfile_prismmind_config.json"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        print(f"✓ Configuration loaded from {config_path}")
        print(f"  - Environment: {config_data.get('environment', 'unknown')}")
        print(f"  - Flatfile config keys: {list(config_data.get('flatfile_config', {}).keys())}")
        print(f"  - Has document processing: {'document_processing' in config_data}")
        print(f"  - Has integration settings: {'integration_settings' in config_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_access():
    """Test access to test files"""
    print("\n=== Testing Test File Access ===")
    
    for file_path in TEST_FILES:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size
            print(f"✓ {path.name} found ({file_size} bytes)")
        else:
            print(f"❌ {path.name} not found at {file_path}")
            return False
    
    return True

async def test_basic_storage_initialization():
    """Test basic storage initialization without PrismMind"""
    print("\n=== Testing Basic Storage Initialization ===")
    
    try:
        import config
        import storage
        
        # Create basic config
        storage_config = config.StorageConfig()
        storage_config.storage_base_path = "./test_simple_data"
        
        # Initialize storage without PrismMind
        storage_manager = storage.StorageManager(storage_config, enable_prismmind=False)
        await storage_manager.initialize()
        
        print("✓ StorageManager initialized without PrismMind")
        print(f"  - Initialized: {storage_manager._initialized}")
        print(f"  - Base path: {storage_manager.base_path}")
        print(f"  - PrismMind available: {storage_manager.is_prismmind_available()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_prismmind_storage_initialization():
    """Test storage initialization with PrismMind"""
    print("\n=== Testing PrismMind Storage Initialization ===")
    
    try:
        import config
        import storage
        
        # Create basic config
        storage_config = config.StorageConfig()
        storage_config.storage_base_path = "./test_prismmind_data"
        
        # Try to initialize storage with PrismMind
        storage_manager = storage.StorageManager(storage_config, enable_prismmind=True)
        await storage_manager.initialize()
        
        print("✓ StorageManager initialized with PrismMind attempt")
        print(f"  - Initialized: {storage_manager._initialized}")
        print(f"  - PrismMind available: {storage_manager.is_prismmind_available()}")
        
        if storage_manager.is_prismmind_available():
            print("✓ PrismMind integration is working!")
            return True
        else:
            print("⚠️ PrismMind integration not available (but storage works)")
            return False
        
    except Exception as e:
        print(f"❌ PrismMind storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_legacy_document_processing():
    """Test legacy document processing with text file"""
    print("\n=== Testing Legacy Document Processing ===")
    
    try:
        import config
        import document_pipeline
        
        # Create config
        storage_config = config.StorageConfig()
        storage_config.storage_base_path = "./test_legacy_data"
        
        # Initialize pipeline without PrismMind
        pipeline = document_pipeline.DocumentRAGPipeline(storage_config, use_prismmind=False)
        
        # Test with text file
        text_file = TEST_FILES[0]  # test_file.txt
        
        print(f"Processing {Path(text_file).name} with legacy pipeline...")
        start_time = time.time()
        
        result = await pipeline.process_document(
            document_path=text_file,
            user_id="test_user",
            session_id="test_session",
            document_id="legacy_test_doc"
        )
        
        processing_time = time.time() - start_time
        
        print(f"Result: {result.success}")
        print(f"Processing time: {processing_time:.2f}s")
        
        if result.success:
            print(f"✓ Legacy processing successful")
            print(f"  - Document ID: {result.document_id}")
            print(f"  - Chunks: {result.chunk_count}")
            print(f"  - Vectors: {result.vector_count}")
            return True
        else:
            print(f"❌ Legacy processing failed: {result.error}")
            return False
        
    except Exception as e:
        print(f"❌ Legacy processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Simple PrismMind Integration Tests")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Test imports
    test_results["imports"] = test_imports()
    
    # Test configuration
    test_results["config"] = test_config_loading()
    
    # Test file access
    test_results["files"] = test_file_access()
    
    # Only proceed with async tests if basic tests pass
    if test_results["imports"]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test basic storage
            test_results["basic_storage"] = loop.run_until_complete(test_basic_storage_initialization())
            
            # Test PrismMind storage
            test_results["prismmind_storage"] = loop.run_until_complete(test_prismmind_storage_initialization())
            
            # Test legacy processing (should always work)
            test_results["legacy_processing"] = loop.run_until_complete(test_legacy_document_processing())
            
        finally:
            loop.close()
    else:
        test_results["basic_storage"] = False
        test_results["prismmind_storage"] = False
        test_results["legacy_processing"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Determine overall success
    critical_tests = ["imports", "basic_storage", "legacy_processing"]
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        if test_results.get("prismmind_storage", False):
            print("🎉 All tests passed! PrismMind integration is fully working.")
        else:
            print("✅ Core functionality works. PrismMind integration may need setup.")
        return True
    else:
        print("❌ Critical functionality failed. Check setup and dependencies.")
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