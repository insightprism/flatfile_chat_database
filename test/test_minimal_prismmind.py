#!/usr/bin/env python3
"""
Minimal test for PrismMind integration using direct path addition.
"""

import sys
import os
from pathlib import Path

# Add PrismMind parent directory to Python path
prismmind_parent = '/home/markly2'
if prismmind_parent not in sys.path:
    sys.path.insert(0, prismmind_parent)

print("🚀 Testing Minimal PrismMind Integration")
print("=" * 50)

def test_basic_imports():
    """Test basic imports from our new package structure."""
    print("\n=== Testing Basic Imports ===")
    
    try:
        # Test main package import
        import prismmind
        print("✓ prismmind package imported")
        print(f"  - Version: {getattr(prismmind, '__version__', 'unknown')}")
        
        # Test availability check
        availability = prismmind.check_availability()
        print(f"  - Available components: {len(availability['available'])}")
        print(f"  - Missing components: {len(availability['missing'])}")
        
        if availability['available']:
            print(f"  - Working: {', '.join(availability['available'][:3])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import prismmind: {e}")
        return False

def test_engine_imports():
    """Test engine module imports."""
    print("\n=== Testing Engine Imports ===")
    
    try:
        from prismmind import pm_engines
        print("✓ pm_engines imported")
        
        # Check available engines
        available = pm_engines.get_available_engines()
        print(f"  - Available engines: {len(available['available'])}")
        print(f"  - Engine chain available: {available['chain_available']}")
        
        if available['available']:
            print(f"  - Engines: {', '.join(available['available'])}")
        
        return len(available['available']) > 0
        
    except Exception as e:
        print(f"❌ Failed to import engines: {e}")
        return False

def test_key_functions():
    """Test key function imports."""
    print("\n=== Testing Key Function Imports ===")
    
    results = {}
    
    # Test pm_run_engine_chain
    try:
        from prismmind import pm_run_engine_chain
        if pm_run_engine_chain is not None:
            print("✓ pm_run_engine_chain imported")
            results['chain'] = True
        else:
            print("❌ pm_run_engine_chain is None")
            results['chain'] = False
    except Exception as e:
        print(f"❌ pm_run_engine_chain import failed: {e}")
        results['chain'] = False
    
    # Test pm_resolve_input_source_async
    try:
        from prismmind import pm_resolve_input_source_async
        if pm_resolve_input_source_async is not None:
            print("✓ pm_resolve_input_source_async imported")
            results['resolve'] = True
        else:
            print("❌ pm_resolve_input_source_async is None")
            results['resolve'] = False
    except Exception as e:
        print(f"❌ pm_resolve_input_source_async import failed: {e}")
        results['resolve'] = False
    
    # Test trace decorator
    try:
        from prismmind import pm_trace_handler_log_dec
        if pm_trace_handler_log_dec is not None:
            print("✓ pm_trace_handler_log_dec imported")
            results['trace'] = True
        else:
            print("❌ pm_trace_handler_log_dec is None")
            results['trace'] = False
    except Exception as e:
        print(f"❌ pm_trace_handler_log_dec import failed: {e}")
        results['trace'] = False
    
    return any(results.values())

def test_integration_compatibility():
    """Test compatibility with our flatfile integration."""
    print("\n=== Testing Integration Compatibility ===")
    
    try:
        # Test the exact imports our integration uses
        from pm_run_engine_chain import pm_run_engine_chain
        print("✓ Direct pm_run_engine_chain import works")
        
        from pm_resolve_input_source_async import pm_resolve_input_source_async  
        print("✓ Direct pm_resolve_input_source_async import works")
        
        from pm_trace_handler_log_dec import pm_trace_handler_log_dec
        print("✓ Direct pm_trace_handler_log_dec import works")
        
        print("✅ All required imports for flatfile integration work!")
        return True
        
    except Exception as e:
        print(f"❌ Integration compatibility failed: {e}")
        return False

def main():
    """Run all tests."""
    
    test_results = {}
    
    # Run all tests
    test_results['basic'] = test_basic_imports()
    test_results['engines'] = test_engine_imports()  
    test_results['functions'] = test_key_functions()
    test_results['integration'] = test_integration_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 MINIMAL TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():15} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if test_results.get('integration', False):
        print("\n🎉 SUCCESS: PrismMind can be imported for flatfile integration!")
        print("   The flatfile PrismMind integration should now work.")
    elif test_results.get('basic', False):
        print("\n✅ PARTIAL: PrismMind package structure works, but some imports failed.")
        print("   This may be due to missing dependencies.")
    else:
        print("\n❌ FAILED: PrismMind package setup needs more work.")
    
    return passed >= 3

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