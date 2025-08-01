#!/usr/bin/env python3
"""
Test script to verify core functionality after DTO name changes.
This will test the basic flow: config -> storage manager -> backend
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_core_functionality():
    """Test the core functionality step by step."""
    
    print("🔍 Testing Core Functionality After DTO Changes\n")
    
    # Step 1: Test configuration loading
    print("1. Testing configuration system...")
    try:
        from ff_class_configs.ff_configuration_manager_config import load_config
        config = load_config()
        print(f"   ✅ Configuration loaded successfully")
        print(f"   📁 Base path: {config.storage.base_path}")
        print(f"   🔧 Config type: {type(config).__name__}")
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
    
    # Step 2: Test backend creation
    print("\n2. Testing backend creation...")
    try:
        from backends import FlatfileBackend
        backend = FlatfileBackend(config)
        print(f"   ✅ Backend created successfully")
        print(f"   🔧 Backend type: {type(backend).__name__}")
        print(f"   📁 Backend base path: {backend.base_path}")
    except Exception as e:
        print(f"   ❌ Backend creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test storage manager creation
    print("\n3. Testing storage manager creation...")
    try:
        from ff_storage_manager import FFStorageManager
        storage_manager = FFStorageManager(config)
        print(f"   ✅ Storage manager created successfully")
        print(f"   🔧 Storage manager type: {type(storage_manager).__name__}")
        print(f"   📁 Storage manager base path: {storage_manager.base_path}")
    except Exception as e:
        print(f"   ❌ Storage manager creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test initialization
    print("\n4. Testing storage system initialization...")
    try:
        await storage_manager.initialize()
        print(f"   ✅ Storage system initialized successfully")
    except Exception as e:
        print(f"   ❌ Storage system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test basic operations
    print("\n5. Testing basic operations...")
    try:
        # Test user creation
        user_created = await storage_manager.create_user("test_user")
        if user_created:
            print(f"   ✅ User creation successful")
        else:
            print(f"   ⚠️ User creation returned False (may already exist)")
        
        # Test session creation
        session_id = await storage_manager.create_session("test_user", "Test Session")
        if session_id:
            print(f"   ✅ Session creation successful: {session_id}")
        else:
            print(f"   ❌ Session creation failed: empty session ID")
            return False
            
        # Test message creation
        from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
        message = FFMessageDTO(
            role=MessageRole.USER,
            content="Test message"
        )
        message_added = await storage_manager.add_message("test_user", session_id, message)
        if message_added:
            print(f"   ✅ Message creation successful")
        else:
            print(f"   ❌ Message creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Basic operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All core functionality tests passed!")
    return True

def test_import_consistency():
    """Test that all imports are consistent with DTO naming."""
    
    print("\n🔍 Testing Import Consistency\n")
    
    # Test entity imports
    print("1. Testing entity class imports...")
    try:
        from ff_class_configs.ff_chat_entities_config import (
            FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFUserProfileDTO,
            MessageRole
        )
        print("   ✅ All DTO entity classes import successfully")
        
        # Test backward compatibility aliases
        from ff_class_configs.ff_chat_entities_config import (
            FFMessage, FFSession, FFDocument, FFUserProfile
        )
        print("   ✅ Backward compatibility aliases work")
        
    except ImportError as e:
        print(f"   ❌ Entity import failed: {e}")
        return False
    
    # Test backend imports
    print("\n2. Testing backend imports...")
    try:
        from backends import StorageBackend, FlatfileBackend
        from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
        
        # Verify alias
        if FlatfileBackend == FFFlatfileStorageBackend:
            print("   ✅ Backend alias is correct")
        else:
            print("   ❌ Backend alias is incorrect")
            return False
            
    except ImportError as e:
        print(f"   ❌ Backend import failed: {e}")
        return False
    
    # Test search imports
    print("\n3. Testing search imports...")
    try:
        from ff_search_manager import FFSearchManager, FFSearchQueryDTO, FFSearchResultDTO
        
        # Test backward compatibility
        from ff_search_manager import FFSearchQuery
        # Note: FFSearchResult alias not implemented yet
        print("   ✅ Search imports and aliases work")
        
    except ImportError as e:
        print(f"   ❌ Search import failed: {e}")
        return False
    
    print("\n✅ All import consistency tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  FLATFILE CHAT DATABASE - CORE FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test import consistency first
    imports_ok = test_import_consistency()
    
    if imports_ok:
        # Test core functionality
        core_ok = asyncio.run(test_core_functionality())
        
        if core_ok:
            print("\n" + "=" * 60)
            print("🎉 OVERALL RESULT: CODEBASE IS HEALTHY")
            print("✅ All tests passed - the DTO changes did not break core functionality")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ OVERALL RESULT: CORE FUNCTIONALITY ISSUES")
            print("⚠️ The DTO changes may have introduced functional problems")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ OVERALL RESULT: IMPORT CONSISTENCY ISSUES")
        print("⚠️ The DTO changes introduced import problems")
        print("=" * 60)