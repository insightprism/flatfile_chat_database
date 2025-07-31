#!/usr/bin/env python3
"""
Test the exact imports your notebook is trying to use.
"""

import sys
from pathlib import Path

# Simulate what your notebook does
print("🔧 Testing notebook imports...")

try:
    # Import configuration system
    print("1. Testing configuration import...")
    from ff_class_configs.ff_configuration_manager_config import load_config
    config = load_config()
    print(f"   ✅ Configuration loaded: {type(config).__name__}")
    
    # Import storage manager  
    print("2. Testing storage manager import...")
    from ff_storage_manager import FFStorageManager
    print(f"   ✅ Storage manager imported: {FFStorageManager}")
    
    # Test backend import
    print("3. Testing backend import...")
    from backends import FlatfileBackend
    import inspect
    backend_file = inspect.getfile(FlatfileBackend)
    print(f"   ✅ Backend imported from: {backend_file}")
    if 'flatfile_chat_database_v2' in backend_file:
        print("   ✅ Correct v2 backend!")
    else:
        print("   ⚠️ Wrong backend version!")
    
    # Test storage manager creation
    print("4. Testing storage manager creation...")
    config.storage.base_path = "./test_data"
    storage_manager = FFStorageManager(config)
    print(f"   ✅ Storage manager created successfully!")
    
    print("\n🎉 ALL TESTS PASSED - Your notebook should work now!")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()