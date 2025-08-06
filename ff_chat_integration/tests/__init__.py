"""
Test infrastructure for Chat Application Bridge System.

Provides common test utilities, fixtures, and helpers for comprehensive testing.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock

# Test configuration
TEMP_TEST_DATA_PREFIX = "ff_chat_bridge_test_"
DEFAULT_TEST_TIMEOUT = 30  # seconds
PERFORMANCE_TEST_ITERATIONS = 10


class TestDataManager:
    """Manages test data creation and cleanup."""
    
    def __init__(self):
        self.temp_dirs = []
        self.test_users = []
        self.test_sessions = []
    
    def create_temp_storage_path(self) -> str:
        """Create temporary storage path for testing."""
        temp_dir = tempfile.mkdtemp(prefix=TEMP_TEST_DATA_PREFIX)
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_all(self):
        """Clean up all test data."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        self.temp_dirs.clear()
        self.test_users.clear()
        self.test_sessions.clear()


class BridgeTestHelper:
    """Helper for bridge testing scenarios."""
    
    @staticmethod
    async def create_test_bridge(config_overrides: Optional[Dict] = None):
        """Create bridge for testing with cleanup."""
        from ff_chat_integration import FFChatAppBridge
        
        test_data = TestDataManager()
        storage_path = test_data.create_temp_storage_path()
        
        options = {"performance_mode": "balanced"}
        if config_overrides:
            options.update(config_overrides)
        
        bridge = await FFChatAppBridge.create_for_chat_app(storage_path, options)
        
        # Attach cleanup function
        bridge._test_cleanup = test_data.cleanup_all
        
        return bridge
    
    @staticmethod
    async def cleanup_test_bridge(bridge):
        """Clean up test bridge and data."""
        try:
            await bridge.close()
            if hasattr(bridge, '_test_cleanup'):
                bridge._test_cleanup()
        except Exception:
            pass


# Common test fixtures
@pytest.fixture
async def test_bridge():
    """Create test bridge with automatic cleanup."""
    bridge = await BridgeTestHelper.create_test_bridge()
    yield bridge
    await BridgeTestHelper.cleanup_test_bridge(bridge)


@pytest.fixture
async def test_bridge_with_data():
    """Create test bridge with sample data."""
    bridge = await BridgeTestHelper.create_test_bridge()
    
    # Add sample data
    data_layer = bridge.get_data_layer()
    
    # Create test user
    user_id = "test_user_with_data"
    await data_layer.storage.create_user(user_id, {"name": "Test User"})
    
    # Create test session
    session_id = await data_layer.storage.create_session(user_id, "Test Session")
    
    # Add test messages
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help with Python."}
    ]
    
    for msg in test_messages:
        await data_layer.store_chat_message(user_id, session_id, msg)
    
    # Attach test data info
    bridge._test_user_id = user_id
    bridge._test_session_id = session_id
    
    yield bridge
    await BridgeTestHelper.cleanup_test_bridge(bridge)


class PerformanceTester:
    """Performance testing utilities."""
    
    @staticmethod
    async def benchmark_operation(operation_func, iterations: int = PERFORMANCE_TEST_ITERATIONS):
        """Benchmark an async operation."""
        import time
        
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            await operation_func()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "average_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "iterations": iterations,
            "all_times": times
        }
    
    @staticmethod
    def assert_performance_improvement(new_time: float, baseline_time: float, 
                                     expected_improvement: float = 0.3):
        """Assert performance improvement meets expectations."""
        actual_improvement = (baseline_time - new_time) / baseline_time
        assert actual_improvement >= expected_improvement, \
            f"Expected {expected_improvement:.1%} improvement, got {actual_improvement:.1%}"


# Mock factories
class MockFactories:
    """Factories for creating mock objects."""
    
    @staticmethod
    def create_mock_storage_manager():
        """Create mock storage manager."""
        mock = AsyncMock()
        mock.create_user.return_value = True
        mock.create_session.return_value = "mock_session_123"
        mock.add_message.return_value = True
        mock.get_messages.return_value = []
        mock.search_messages.return_value = []
        return mock
    
    @staticmethod
    def create_mock_config():
        """Create mock configuration."""
        from ff_chat_integration import ChatAppStorageConfig
        return ChatAppStorageConfig(
            storage_path="./mock_data",
            performance_mode="balanced"
        )