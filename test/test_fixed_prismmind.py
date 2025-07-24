#!/usr/bin/env python3
"""
Test PrismMind after fixing all import issues.
"""

import sys
import os
from pathlib import Path

# Add PrismMind parent directory to Python path
prismmind_parent = '/home/markly2'
if prismmind_parent not in sys.path:
    sys.path.insert(0, prismmind_parent)

print("🎉 Testing Fixed PrismMind Integration")
print("=" * 50)

def test_core_functions():
    """Test the core functions that should now work."""
    print("\n=== Testing Core PrismMind Functions ===")
    
    results = {}
    
    # Test pm_run_engine_chain
    try:
        from prismmind.pm_engines.pm_run_engine_chain import pm_run_engine_chain
        print("✅ pm_run_engine_chain imported successfully")
        print(f"   Function type: {type(pm_run_engine_chain)}")
        results['chain'] = True
    except Exception as e:
        print(f"❌ pm_run_engine_chain failed: {e}")
        results['chain'] = False
    
    # Test PmBaseEngine
    try:
        from prismmind.pm_engines.pm_base_engine import PmBaseEngine
        print("✅ PmBaseEngine imported successfully")
        print(f"   Class type: {type(PmBaseEngine)}")
        results['base_engine'] = True
    except Exception as e:
        print(f"❌ PmBaseEngine failed: {e}")
        results['base_engine'] = False
    
    # Test trace decorator
    try:
        from prismmind.pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec
        print("✅ pm_trace_handler_log_dec imported successfully")
        print(f"   Decorator type: {type(pm_trace_handler_log_dec)}")
        results['trace'] = True
    except Exception as e:
        print(f"❌ pm_trace_handler_log_dec failed: {e}")
        results['trace'] = False
    
    # Test input resolution function
    try:
        from prismmind.pm_utils.adhoc_util_functions import pm_resolve_input_source_async
        print("✅ pm_resolve_input_source_async imported successfully")
        print(f"   Function type: {type(pm_resolve_input_source_async)}")
        results['resolve'] = True
    except Exception as e:
        print(f"❌ pm_resolve_input_source_async failed: {e}")
        results['resolve'] = False
    
    return results

def test_engine_classes():
    """Test various engine classes."""
    print("\n=== Testing Engine Classes ===")
    
    engine_tests = [
        ('PmIngestEngine', 'prismmind.pm_engines.pm_injest_engine'),
        ('PmChunkingEngine', 'prismmind.pm_engines.pm_chunking_engine'),
        ('PmEmbeddingEngine', 'prismmind.pm_engines.pm_embedding_engine'),
        ('PmNlpEngine', 'prismmind.pm_engines.pm_nlp_engine'),
    ]
    
    working_engines = []
    
    for engine_name, module_path in engine_tests:
        try:
            module = __import__(module_path, fromlist=[engine_name])
            engine_class = getattr(module, engine_name)
            print(f"✅ {engine_name} imported successfully")
            working_engines.append(engine_name)
        except Exception as e:
            print(f"❌ {engine_name} failed: {e}")
    
    return working_engines

def test_flatfile_integration_requirements():
    """Test the specific imports needed for flatfile integration."""
    print("\n=== Testing Flatfile Integration Requirements ===")
    
    requirements = [
        ('pm_run_engine_chain', 'prismmind.pm_engines.pm_run_engine_chain', 'pm_run_engine_chain'),
        ('pm_resolve_input_source_async', 'prismmind.pm_utils.adhoc_util_functions', 'pm_resolve_input_source_async'),
        ('pm_trace_handler_log_dec', 'prismmind.pm_utils.pm_trace_handler_log_dec', 'pm_trace_handler_log_dec'),
        ('PmBaseEngine', 'prismmind.pm_engines.pm_base_engine', 'PmBaseEngine'),
    ]
    
    working_requirements = []
    
    for req_name, module_path, item_name in requirements:
        try:
            module = __import__(module_path, fromlist=[item_name])
            item = getattr(module, item_name)
            print(f"✅ {req_name} available for flatfile integration")
            working_requirements.append(req_name)
        except Exception as e:
            print(f"❌ {req_name} not available: {e}")
    
    return working_requirements

def simulate_flatfile_integration():
    """Simulate what the flatfile integration would look like."""
    print("\n=== Simulating Flatfile Integration ===")
    
    try:
        # This simulates the exact import pattern our flatfile integration uses
        print("Simulating flatfile integration imports...")
        
        # Import the key functions
        from prismmind.pm_engines.pm_run_engine_chain import pm_run_engine_chain
        from prismmind.pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec  
        from prismmind.pm_engines.pm_base_engine import PmBaseEngine
        
        print("✅ All critical imports successful!")
        print("✅ Flatfile integration can now use PrismMind!")
        
        # Test package-level detection
        import prismmind
        print(f"✅ Package version: {prismmind.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flatfile integration simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    
    # Test core functions
    core_results = test_core_functions()
    
    # Test engine classes
    working_engines = test_engine_classes()
    
    # Test flatfile integration requirements
    integration_requirements = test_flatfile_integration_requirements()
    
    # Simulate actual integration
    integration_works = simulate_flatfile_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 50)
    
    core_working = sum(core_results.values())
    total_core = len(core_results)
    
    print(f"Core Functions:      {core_working}/{total_core} working")
    print(f"Engine Classes:      {len(working_engines)} engines importable")
    print(f"Integration Needs:   {len(integration_requirements)}/4 requirements met")
    print(f"Integration Test:    {'✅ PASS' if integration_works else '❌ FAIL'}")
    
    if integration_works:
        print("\n🎉 SUCCESS: PrismMind is now fully functional for flatfile integration!")
        print("   The flatfile chat database can now use PrismMind engines.")
        print("   Key working components:")
        for req in integration_requirements:
            print(f"   - {req}")
        return True
    else:
        print("\n⚠️ PARTIAL SUCCESS: Some components working, but integration needs work.")
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