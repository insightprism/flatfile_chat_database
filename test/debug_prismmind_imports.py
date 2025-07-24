#!/usr/bin/env python3
"""
Debug script to identify specific import issues in PrismMind.
"""

import sys
import os
from pathlib import Path

# Add PrismMind parent directory to Python path
prismmind_parent = '/home/markly2'
if prismmind_parent not in sys.path:
    sys.path.insert(0, prismmind_parent)

print("🔍 Debugging PrismMind Import Issues")
print("=" * 50)

def test_direct_engine_imports():
    """Test direct imports from engine files."""
    print("\n=== Testing Direct Engine File Imports ===")
    
    try:
        # Test importing the run chain directly
        print("Trying: from prismmind.pm_engines.pm_run_engine_chain import pm_run_engine_chain")
        from prismmind.pm_engines.pm_run_engine_chain import pm_run_engine_chain
        print("✓ pm_run_engine_chain imported directly")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("  Let's check if the file exists...")
        
        run_chain_path = Path("/home/markly2/prismmind/pm_engines/pm_run_engine_chain.py")
        if run_chain_path.exists():
            print(f"  ✓ File exists: {run_chain_path}")
            print("  Let's try importing the module file directly...")
            
            try:
                # Add the specific engine directory to path
                engines_path = str(run_chain_path.parent)
                if engines_path not in sys.path:
                    sys.path.insert(0, engines_path)
                
                import pm_run_engine_chain
                print("  ✓ Module imported as pm_run_engine_chain")
                
                if hasattr(pm_run_engine_chain, 'pm_run_engine_chain'):
                    print("  ✓ Function pm_run_engine_chain found in module")
                    return True
                else:
                    print("  ❌ Function pm_run_engine_chain not found in module")
                    print(f"  Available attributes: {dir(pm_run_engine_chain)[:5]}...")
                    
            except Exception as e2:
                print(f"  ❌ Module import failed: {e2}")
        else:
            print(f"  ❌ File does not exist: {run_chain_path}")
        
        return False

def test_base_engine_import():
    """Test base engine import."""
    print("\n=== Testing Base Engine Import ===")
    
    try:
        print("Trying: from prismmind.pm_engines.pm_base_engine import PmBaseEngine")
        from prismmind.pm_engines.pm_base_engine import PmBaseEngine
        print("✓ PmBaseEngine imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_trace_decorator_import():
    """Test trace decorator import."""
    print("\n=== Testing Trace Decorator Import ===")
    
    try:
        print("Trying: from prismmind.pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec")
        from prismmind.pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec
        print("✓ pm_trace_handler_log_dec imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_alternative_imports():
    """Test alternative import methods."""
    print("\n=== Testing Alternative Import Methods ===")
    
    # Add engines directory directly to path
    engines_dir = "/home/markly2/prismmind/pm_engines"
    utils_dir = "/home/markly2/prismmind/pm_utils"
    config_dir = "/home/markly2/prismmind/pm_config"
    
    for directory in [engines_dir, utils_dir, config_dir]:
        if directory not in sys.path:
            sys.path.insert(0, directory)
    
    results = {}
    
    # Test direct module imports
    modules_to_test = [
        'pm_run_engine_chain',
        'pm_base_engine', 
        'pm_trace_handler_log_dec',
        'pm_base_engine_config'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name} imported successfully")
            results[module_name] = True
        except Exception as e:
            print(f"❌ {module_name} failed: {e}")
            results[module_name] = False
    
    return results

def show_file_structure():
    """Show the actual file structure to verify."""
    print("\n=== PrismMind File Structure ===")
    
    prismmind_dir = Path("/home/markly2/prismmind")
    
    if not prismmind_dir.exists():
        print("❌ PrismMind directory not found!")
        return
    
    print(f"📁 {prismmind_dir}")
    print(f"  📄 __init__.py exists: {(prismmind_dir / '__init__.py').exists()}")
    
    # Check engines directory
    engines_dir = prismmind_dir / "pm_engines"
    if engines_dir.exists():
        print(f"  📁 pm_engines/")
        print(f"    📄 __init__.py exists: {(engines_dir / '__init__.py').exists()}")
        print(f"    📄 pm_run_engine_chain.py exists: {(engines_dir / 'pm_run_engine_chain.py').exists()}")
        print(f"    📄 pm_base_engine.py exists: {(engines_dir / 'pm_base_engine.py').exists()}")
    
    # Check utils directory  
    utils_dir = prismmind_dir / "pm_utils"
    if utils_dir.exists():
        print(f"  📁 pm_utils/")
        print(f"    📄 __init__.py exists: {(utils_dir / '__init__.py').exists()}")
        print(f"    📄 pm_trace_handler_log_dec.py exists: {(utils_dir / 'pm_trace_handler_log_dec.py').exists()}")

def main():
    """Run all debug tests."""
    
    show_file_structure()
    
    test_results = {}
    test_results['direct_engine'] = test_direct_engine_imports()
    test_results['base_engine'] = test_base_engine_import()
    test_results['trace_decorator'] = test_trace_decorator_import()
    
    alternative_results = test_alternative_imports()
    test_results.update(alternative_results)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 DEBUG RESULTS SUMMARY")
    print("=" * 50)
    
    working_imports = [name for name, success in test_results.items() if success]
    failed_imports = [name for name, success in test_results.items() if not success]
    
    if working_imports:
        print(f"✅ Working imports: {', '.join(working_imports)}")
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
    
    print(f"\nOverall: {len(working_imports)}/{len(test_results)} imports working")
    
    if len(working_imports) >= len(failed_imports):
        print("\n✅ Most imports are working. PrismMind can be made functional.")
    else:
        print("\n⚠️ Many imports are failing. More dependency fixes needed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()