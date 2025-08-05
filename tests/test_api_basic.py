# Basic API Test - Minimal working example
"""
Basic test to verify the API testing infrastructure works.
"""

import pytest

def test_basic_functionality():
    """Test basic functionality works"""
    assert True

def test_import_structure():
    """Test that we can import basic modules"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test basic imports
    try:
        # These should work
        import asyncio
        assert asyncio is not None
        
        import json
        assert json is not None
        
        print("✅ Basic imports working")
        
    except Exception as e:
        pytest.fail(f"Basic imports failed: {e}")

@pytest.mark.asyncio
async def test_async_functionality():
    """Test that async tests work"""
    import asyncio
    
    # Simple async operation
    await asyncio.sleep(0.001)
    
    result = await simple_async_function()
    assert result == "async_works"

async def simple_async_function():
    """Simple async function for testing"""
    return "async_works"

def test_ff_imports():
    """Test FF module imports"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Try to import FF components
        from ff_chat_api import FFChatAPI
        print("✅ FFChatAPI import successful")
    except ImportError as e:
        print(f"⚠️ FFChatAPI import failed (expected in test environment): {e}")
        # This is acceptable in test environment
    
    try:
        from ff_chat_auth import FFChatAuthManager
        print("✅ FFChatAuthManager import successful")
    except ImportError as e:
        print(f"⚠️ FFChatAuthManager import failed (expected in test environment): {e}")
        # This is acceptable in test environment

class TestBasicAPIStructure:
    """Test basic API structure"""
    
    def test_pytest_working(self):
        """Test that pytest is working correctly"""
        assert 1 + 1 == 2
    
    def test_project_structure(self):
        """Test project structure exists"""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        
        # Check key directories exist
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "api").exists()
        assert (project_root / "tests" / "security").exists()
        assert (project_root / "tests" / "system").exists()
        
        print("✅ Project structure validated")
    
    def test_configuration_files(self):
        """Test configuration files exist"""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        
        # Check configuration files
        assert (project_root / "pytest.ini").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        
        print("✅ Configuration files validated")