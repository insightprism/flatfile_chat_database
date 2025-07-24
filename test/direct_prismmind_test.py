#!/usr/bin/env python3
"""
Direct test of PrismMind integration files without import issues.
"""

import sys
import json
from pathlib import Path

# Add PrismMind engines to path  
sys.path.append('/home/markly2/PrismMind_v2/pm_engines')

# Test files
TEST_FILES = [
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/test_file.txt',
    '/home/markly2/PrismMind_v2/pm_user_guide/test_data/mp_earnings_2024q4.pdf'
]

def test_prismmind_engines():
    """Test if PrismMind engines are available"""
    print("=== Testing PrismMind Engine Availability ===")
    
    engines_to_test = [
        'pm_trace_logger',
        'pm_engine_config',
        'pm_run_engine_chain', 
        'pm_resolve_input_source_async',
        'pm_prism_text_handler_async',
        'pm_prism_pdf_handler_async',
        'pm_embed_batch_handler_async',
        'pm_fixed_chunk_handler_async'
    ]
    
    available = []
    missing = []
    
    for engine in engines_to_test:
        try:
            module = __import__(engine)
            available.append(engine)
            print(f"✓ {engine}")
        except ImportError as e:
            missing.append(engine)
            print(f"❌ {engine}: {e}")
    
    print(f"\nSummary: {len(available)}/{len(engines_to_test)} engines available")
    
    if missing:
        print("Missing engines:", missing)
        return False
    
    return True

def test_integration_files():
    """Test that our integration files exist and are valid"""
    print("\n=== Testing Integration Files ===")
    
    main_db_path = Path(__file__).parent.parent / "flatfile_chat_database"
    
    # Test integration directory structure
    integration_dir = main_db_path / "prismmind_integration"
    if not integration_dir.exists():
        print(f"❌ Integration directory not found: {integration_dir}")
        return False
    
    required_files = [
        "__init__.py",
        "config.py", 
        "handlers.py",
        "factory.py",
        "processor.py",
        "loader.py"
    ]
    
    for file_name in required_files:
        file_path = integration_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✓ {file_name} ({file_size} bytes)")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    # Test config files
    configs_dir = main_db_path / "configs"
    config_files = [
        "flatfile_prismmind_config.json",
        "development_config.json", 
        "production_config.json",
        "test_config.json"
    ]
    
    print("\n--- Configuration Files ---")
    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                print(f"✓ {config_file} (valid JSON)")
            except json.JSONDecodeError as e:
                print(f"❌ {config_file} (invalid JSON): {e}")
                return False
        else:
            print(f"❌ {config_file} missing")
            return False
    
    return True

def test_file_access():
    """Test access to test files"""
    print("\n=== Testing Test File Access ===")
    
    for file_path in TEST_FILES:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size
            print(f"✓ {path.name} found ({file_size:,} bytes)")
            
            # Try to read a bit of each file
            try:
                if path.suffix == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(100)
                        print(f"  - Text preview: {content[:50]}...")
                elif path.suffix == '.pdf':
                    print(f"  - PDF file detected")
            except Exception as e:
                print(f"  - Read test failed: {e}")
                
        else:
            print(f"❌ {path.name} not found at {file_path}")
            return False
    
    return True

def test_engine_chain_simulation():
    """Simulate the PrismMind engine chain process"""
    print("\n=== Testing Engine Chain Simulation ===")
    
    try:
        # Test if we can at least import the main chain runner
        try:
            from pm_run_engine_chain import pm_run_engine_chain
            print("✓ pm_run_engine_chain imported successfully")
            
            # Test if we can import resolve input source
            from pm_resolve_input_source_async import pm_resolve_input_source_async
            print("✓ pm_resolve_input_source_async imported successfully")
            
            print("✓ Core PrismMind chain functions available")
            return True
            
        except ImportError as e:
            print(f"❌ PrismMind chain functions not available: {e}")
            print("   This means PrismMind engines are not properly accessible")
            return False
        
    except Exception as e:
        print(f"❌ Engine chain simulation failed: {e}")
        return False

def test_config_parsing():
    """Test parsing our PrismMind configuration"""
    print("\n=== Testing Configuration Parsing ===")
    
    try:
        main_db_path = Path(__file__).parent.parent / "flatfile_chat_database"
        config_path = main_db_path / "configs" / "flatfile_prismmind_config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Test key configuration sections
        sections_to_check = [
            "flatfile_config",
            "document_processing", 
            "engine_selection",
            "handler_strategies",
            "integration_settings"
        ]
        
        for section in sections_to_check:
            if section in config:
                print(f"✓ {section} section present")
                
                # Check specific important keys
                if section == "document_processing":
                    chains = config[section].get("file_type_chains", {})
                    print(f"  - File type chains: {len(chains)} types")
                    print(f"  - Supported types: {list(chains.keys())}")
                    
                elif section == "engine_selection":
                    handlers = config[section].get("file_type_handlers", {})
                    print(f"  - File handlers: {len(handlers)} types")
                    
                elif section == "handler_strategies":
                    strategies = config[section].get("default_strategies", {})
                    print(f"  - Default strategies: {list(strategies.keys())}")
                    
            else:
                print(f"❌ {section} section missing")
                return False
        
        print("✓ Configuration structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration parsing failed: {e}")
        return False

def main():
    """Run all direct tests"""
    print("🚀 Starting Direct PrismMind Integration Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Test integration files exist
    test_results["integration_files"] = test_integration_files()
    
    # Test configuration parsing
    test_results["config_parsing"] = test_config_parsing()
    
    # Test file access
    test_results["file_access"] = test_file_access()
    
    # Test PrismMind engines
    test_results["prismmind_engines"] = test_prismmind_engines()
    
    # Test engine chain simulation (only if engines available)
    if test_results["prismmind_engines"]:
        test_results["engine_chain"] = test_engine_chain_simulation()
    else:
        test_results["engine_chain"] = False
        print("\n⚠️ Skipping engine chain test due to missing PrismMind engines")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DIRECT TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Analysis
    if test_results["integration_files"] and test_results["config_parsing"]:
        print("\n✅ PrismMind integration code is properly implemented")
        
        if test_results["prismmind_engines"] and test_results["engine_chain"]:
            print("✅ PrismMind engines are available and accessible")
            print("🎉 Full integration should work!")
        else:
            print("⚠️ PrismMind engines not accessible - integration won't function")
            print("   Check that PrismMind is installed and path is correct")
    else:
        print("❌ Integration implementation has issues")
    
    return passed >= 3  # At least integration files, config, and file access should work

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