#!/usr/bin/env python3
"""
Simple test runner for integration tests without pytest.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the package
from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, Session, 
    SituationalContext, Document, UserProfile, Persona,
    PanelMessage, PanelInsight
)

# Import test classes
from test_storage_integration import (
    storage_manager,
    TestUserManagement,
    TestSessionManagement,
    TestMessageManagement,
    TestDocumentManagement,
    TestContextManagement,
    TestPanelManagement,
    TestPersonaManagement,
    TestCompleteWorkflow
)


async def run_test_method(test_class, method_name, storage):
    """Run a single test method"""
    print(f"  Running {method_name}...", end=" ")
    try:
        test_instance = test_class()
        method = getattr(test_instance, method_name)
        await method(storage)
        print("✓ PASSED")
        return True
    except Exception as e:
        print("✗ FAILED")
        print(f"    Error: {e}")
        traceback.print_exc()
        return False


async def run_test_class(test_class_name, test_class):
    """Run all test methods in a test class"""
    print(f"\n{test_class_name}:")
    
    # Get all test methods
    test_methods = [
        method for method in dir(test_class)
        if method.startswith('test_') and callable(getattr(test_class, method))
    ]
    
    passed = 0
    failed = 0
    
    # Create storage manager for this test class
    storage = None
    async for s in storage_manager():
        storage = s
        break
    
    for method_name in test_methods:
        if await run_test_method(test_class, method_name, storage):
            passed += 1
        else:
            failed += 1
    
    return passed, failed


async def main():
    """Run all integration tests"""
    print("Running Flatfile Chat Database Integration Tests")
    print("=" * 50)
    
    test_classes = [
        ("User Management", TestUserManagement),
        ("Session Management", TestSessionManagement),
        ("Message Management", TestMessageManagement),
        ("Document Management", TestDocumentManagement),
        ("Context Management", TestContextManagement),
        ("Panel Management", TestPanelManagement),
        ("Persona Management", TestPersonaManagement),
        ("Complete Workflow", TestCompleteWorkflow)
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name, test_class in test_classes:
        passed, failed = await run_test_class(test_name, test_class)
        total_passed += passed
        total_failed += failed
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {total_passed} passed, {total_failed} failed")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)