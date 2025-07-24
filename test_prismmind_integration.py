#!/usr/bin/env python3
"""
Test script for PrismMind integration with flatfile chat database.

This script tests the complete processing chain using real test files.
"""

import sys
import asyncio
import time
from pathlib import Path
import os

# Add current directory and PrismMind engines to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.append('/home/markly2/PrismMind_v2/pm_engines')

# Change to the correct directory
os.chdir(current_dir)

# Import our integration modules
try:
    from storage import StorageManager
    from document_pipeline import DocumentRAGPipeline  
    from config import StorageConfig
    print("✓ Successfully imported core modules")
except ImportError as e:
    print(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

# Test files
TEST_FILES = [
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/test_file.txt',
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/mp_earnings_2024q4.pdf'
]

async def test_storage_manager_integration():
    """Test PrismMind integration via StorageManager"""
    print("\n=== Testing StorageManager PrismMind Integration ===")
    
    try:
        # Initialize storage manager
        config = StorageConfig()
        storage = StorageManager(config, enable_prismmind=True)
        await storage.initialize()
        
        print(f"✓ Storage manager initialized")
        print(f"✓ PrismMind available: {storage.is_prismmind_available()}")
        
        if not storage.is_prismmind_available():
            print("❌ PrismMind integration not available")
            return False
        
        # Test processing each file
        results = []
        for i, file_path in enumerate(TEST_FILES):
            if not Path(file_path).exists():
                print(f"❌ Test file not found: {file_path}")
                continue
                
            print(f"\n--- Processing {Path(file_path).name} ---")
            start_time = time.time()
            
            result = await storage.process_document_with_prismmind(
                document_path=file_path,
                user_id="test_user",
                session_id="test_session",
                document_id=f"test_doc_{i}",
                metadata={"test": True, "file_type": Path(file_path).suffix}
            )
            
            processing_time = time.time() - start_time
            results.append(result)
            
            print(f"Result: {result}")
            print(f"Processing time: {processing_time:.2f}s")
            
            if result.get("success"):
                print(f"✓ Successfully processed {Path(file_path).name}")
                print(f"  - Document ID: {result.get('document_id')}")
                print(f"  - Chunks created: {result.get('chunk_count', 0)}")
                print(f"  - Vectors created: {result.get('vector_count', 0)}")
            else:
                print(f"❌ Failed to process {Path(file_path).name}: {result.get('error')}")
        
        return all(r.get("success", False) for r in results)
        
    except Exception as e:
        print(f"❌ StorageManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_document_pipeline_integration():
    """Test PrismMind integration via DocumentRAGPipeline"""
    print("\n=== Testing DocumentRAGPipeline PrismMind Integration ===")
    
    try:
        # Initialize pipeline with PrismMind
        config = StorageConfig()
        pipeline = DocumentRAGPipeline(config, use_prismmind=True)
        
        print(f"✓ Pipeline initialized with PrismMind: {pipeline.use_prismmind}")
        
        if not pipeline.use_prismmind:
            print("❌ PrismMind not enabled in pipeline")
            return False
        
        # Test processing files
        results = []
        for i, file_path in enumerate(TEST_FILES):
            if not Path(file_path).exists():
                print(f"❌ Test file not found: {file_path}")
                continue
                
            print(f"\n--- Processing {Path(file_path).name} via Pipeline ---")
            start_time = time.time()
            
            result = await pipeline.process_document(
                document_path=file_path,
                user_id="test_user_pipeline",
                session_id="test_session_pipeline",
                document_id=f"pipeline_doc_{i}",
                metadata={"test": True, "method": "pipeline"}
            )
            
            processing_time = time.time() - start_time
            results.append(result)
            
            print(f"Result success: {result.success}")
            print(f"Processing time: {processing_time:.2f}s")
            
            if result.success:
                print(f"✓ Successfully processed {Path(file_path).name}")
                print(f"  - Document ID: {result.document_id}")
                print(f"  - Chunks: {result.chunk_count}")
                print(f"  - Vectors: {result.vector_count}")
                print(f"  - Processing time: {result.processing_time:.2f}s")
            else:
                print(f"❌ Failed to process {Path(file_path).name}: {result.error}")
        
        return all(r.success for r in results)
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_legacy_fallback():
    """Test that legacy processing still works"""
    print("\n=== Testing Legacy Fallback ===")
    
    try:
        # Initialize pipeline without PrismMind
        config = StorageConfig()
        pipeline = DocumentRAGPipeline(config, use_prismmind=False)
        
        print(f"✓ Pipeline initialized without PrismMind: {not pipeline.use_prismmind}")
        
        # Test with text file (should work in legacy mode)
        text_file = TEST_FILES[0]  # test_file.txt
        
        print(f"--- Processing {Path(text_file).name} via Legacy ---")
        start_time = time.time()
        
        result = await pipeline.process_document(
            document_path=text_file,
            user_id="test_user_legacy",
            session_id="test_session_legacy",
            document_id="legacy_doc_0",
            metadata={"test": True, "method": "legacy"}
        )
        
        processing_time = time.time() - start_time
        
        print(f"Result success: {result.success}")
        print(f"Processing time: {processing_time:.2f}s")
        
        if result.success:
            print(f"✓ Legacy processing works for {Path(text_file).name}")
            print(f"  - Document ID: {result.document_id}")
            print(f"  - Chunks: {result.chunk_count}")
            print(f"  - Processing method: {result.metadata.get('processing_method', 'unknown')}")
            return True
        else:
            print(f"❌ Legacy processing failed: {result.error}")
            return False
        
    except Exception as e:
        print(f"❌ Legacy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_configuration_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        from prismmind_integration.loader import FlatfilePrismMindConfigLoader
        
        # Test loading default config
        config_path = "configs/flatfile_prismmind_config.json"
        if Path(config_path).exists():
            config = FlatfilePrismMindConfigLoader.from_file(config_path)
            print(f"✓ Loaded configuration from {config_path}")
            print(f"  - Environment: {config.environment}")
            print(f"  - Processing chains defined: {len(config.document_processing.file_type_chains)}")
            print(f"  - Handler strategies: {len(config.handler_strategies.default_strategies)}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting PrismMind Integration Tests")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Test configuration loading first
    test_results["config"] = await test_configuration_loading()
    
    # Test storage manager integration
    test_results["storage"] = await test_storage_manager_integration()
    
    # Test document pipeline integration  
    test_results["pipeline"] = await test_document_pipeline_integration()
    
    # Test legacy fallback
    test_results["legacy"] = await test_legacy_fallback()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():12} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PrismMind integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)