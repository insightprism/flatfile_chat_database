"""
Error handling and edge case tests for the flatfile chat database system.

Tests error conditions, edge cases, and system resilience to ensure
the recent architecture changes maintain robust error handling.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import sys

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import create_default_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
from ff_dependency_injection_manager import FFDependencyInjectionManager, ff_create_application_container
from ff_protocols import StorageProtocol, BackendProtocol
from ff_utils.ff_file_ops import FFFileOperationError, FFAtomicWriteError, FFLockTimeoutError


class TestConfigurationErrors:
    """Test error handling in configuration system."""
    
    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        config = create_default_config("test")
        
        # Set invalid values
        config.storage.base_path = ""  # Empty path
        config.storage.max_message_size_bytes = -1  # Negative size
        config.locking.timeout_seconds = -5.0  # Negative timeout
        config.search.default_limit = 0  # Zero limit
        
        # Validation should catch these
        errors = config.validate_all()
        assert len(errors) > 0
        
        # Specific errors should be reported
        error_text = " ".join(errors)
        assert "base_path" in error_text or "empty" in error_text.lower()
        assert "negative" in error_text.lower() or "positive" in error_text.lower()
    
    def test_configuration_file_errors(self, temp_dir):
        """Test handling of configuration file errors."""
        from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
        
        # Test missing file
        missing_file = temp_dir / "missing_config.json"
        with pytest.raises(FileNotFoundError):
            FFConfigurationManagerConfigDTO.from_file(missing_file)
        
        # Test invalid JSON
        invalid_json_file = temp_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content }")
        
        with pytest.raises(Exception):  # JSON decode error
            FFConfigurationManagerConfigDTO.from_file(invalid_json_file)
        
        # Test permission error simulation
        if os.name != 'nt':  # Skip on Windows due to permission handling differences
            restricted_file = temp_dir / "restricted.json"
            restricted_file.write_text('{"storage": {"base_path": "/test"}}')
            restricted_file.chmod(0o000)  # No permissions
            
            try:
                with pytest.raises(PermissionError):
                    FFConfigurationManagerConfigDTO.from_file(restricted_file)
            finally:
                restricted_file.chmod(0o644)  # Restore permissions for cleanup


class TestDependencyInjectionErrors:
    """Test error handling in dependency injection system."""
    
    def test_unregistered_service_resolution(self):
        """Test resolving unregistered services."""
        container = FFDependencyInjectionManager()
        
        class UnregisteredService:
            pass
        
        with pytest.raises(ValueError, match="Service .* not registered"):
            container.resolve(UnregisteredService)
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        container = FFDependencyInjectionManager()
        
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a: 'ServiceA'):
                self.service_a = service_a
        
        container.register_transient(ServiceA, ServiceA)
        container.register_transient(ServiceB, ServiceB)
        
        # This should detect the circular dependency
        with pytest.raises(Exception):  # Could be RecursionError or custom error
            container.resolve(ServiceA)
    
    def test_factory_function_errors(self):
        """Test error handling in factory functions."""
        container = FFDependencyInjectionManager()
        
        def failing_factory(c: FFDependencyInjectionManager):
            raise RuntimeError("Factory intentionally failed")
        
        class TestService:
            pass
        
        container.register_singleton(TestService, factory=failing_factory)
        
        with pytest.raises(RuntimeError, match="Factory intentionally failed"):
            container.resolve(TestService)
    
    def test_invalid_service_registration(self):
        """Test invalid service registration scenarios."""
        container = FFDependencyInjectionManager()
        
        class TestService:
            pass
        
        # Test registration with no implementation, factory, or instance
        with pytest.raises(ValueError, match="Must provide implementation, factory, or instance"):
            container.register(TestService)
        
        # Test invalid lifetime
        with pytest.raises((ValueError, AttributeError)):
            container.register(TestService, TestService, lifetime="invalid_lifetime")


class TestStorageErrors:
    """Test error handling in storage operations."""
    
    @pytest.mark.asyncio
    async def test_storage_initialization_failures(self, temp_dir):
        """Test storage initialization failure scenarios."""
        config = create_default_config("test")
        
        # Test with invalid path (no permissions)
        if os.name != 'nt':  # Skip on Windows
            config.storage.base_path = "/root/invalid_storage_path"
            backend = FFFlatfileStorageBackend(config)
            
            result = await backend.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_invalid_user_operations(self, storage_manager):
        """Test invalid user operations."""
        # Test operations with invalid user IDs
        invalid_user_ids = ["", None, "user/with/slashes", "user\x00null"]
        
        for invalid_id in invalid_user_ids:
            if invalid_id is None:
                with pytest.raises((TypeError, AttributeError)):
                    await storage_manager.create_user(invalid_id)
            else:
                # Should handle gracefully or raise appropriate error
                try:
                    result = await storage_manager.create_user(invalid_id)
                    # If it succeeds, that's also acceptable
                except (ValueError, OSError):
                    # Expected for invalid characters
                    pass
    
    @pytest.mark.asyncio
    async def test_message_size_limits(self, storage_manager):
        """Test message size limit enforcement."""
        await storage_manager.create_user("size_test_user")
        session_id = await storage_manager.create_session("size_test_user", "Size Test")
        
        # Create message exceeding size limit
        max_size = storage_manager.config.storage.max_message_size_bytes
        oversized_content = "X" * (max_size + 1000)
        
        oversized_message = FFMessageDTO(
            role=MessageRole.USER,
            content=oversized_content
        )
        
        # Should fail gracefully
        result = await storage_manager.add_message("size_test_user", session_id, oversized_message)
        assert result is False
        
        # Verify session state is consistent
        session = await storage_manager.get_session("size_test_user", session_id)
        assert session.message_count == 0
    
    @pytest.mark.asyncio
    async def test_document_size_limits(self, storage_manager):
        """Test document size limit enforcement."""
        await storage_manager.create_user("doc_size_user")
        session_id = await storage_manager.create_session("doc_size_user", "Doc Size Test")
        
        # Create document exceeding size limit
        max_size = storage_manager.config.storage.max_document_size_bytes
        oversized_content = b"X" * (max_size + 1000)
        
        # Should fail gracefully
        doc_id = await storage_manager.save_document(
            "doc_size_user", session_id, "huge_file.txt", oversized_content
        )
        assert doc_id == ""  # Empty string indicates failure
        
        # Verify no document was stored
        documents = await storage_manager.list_documents("doc_size_user", session_id)
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_file_extensions(self, storage_manager):
        """Test invalid file extension handling."""
        await storage_manager.create_user("ext_test_user")
        session_id = await storage_manager.create_session("ext_test_user", "Extension Test")
        
        # Test various invalid extensions
        invalid_files = [
            ("malware.exe", b"executable content"),
            ("script.bat", b"batch script"),
            ("virus.scr", b"screensaver executable")
        ]
        
        for filename, content in invalid_files:
            doc_id = await storage_manager.save_document(
                "ext_test_user", session_id, filename, content
            )
            assert doc_id == "", f"Should reject {filename}"
    
    @pytest.mark.asyncio
    async def test_nonexistent_data_access(self, storage_manager):
        """Test accessing nonexistent data."""
        # Test accessing nonexistent user
        profile = await storage_manager.get_user_profile("nonexistent_user")
        assert profile is None
        
        # Test accessing nonexistent session
        session = await storage_manager.get_session("nonexistent_user", "nonexistent_session")
        assert session is None
        
        # Test getting messages from nonexistent session
        messages = await storage_manager.get_all_messages("nonexistent_user", "nonexistent_session")
        assert len(messages) == 0
        
        # Test getting documents from nonexistent session
        documents = await storage_manager.list_documents("nonexistent_user", "nonexistent_session")
        assert len(documents) == 0
        
        # Test getting context from nonexistent session
        context = await storage_manager.get_context("nonexistent_user", "nonexistent_session")
        assert context is None
    
    @pytest.mark.asyncio
    async def test_session_stats_for_nonexistent_session(self, storage_manager):
        """Test session statistics for nonexistent session."""
        with pytest.raises(ValueError, match="Session .* not found"):
            await storage_manager.get_session_stats("nonexistent_user", "nonexistent_session")


class TestBackendErrors:
    """Test error handling in backend operations."""
    
    @pytest.mark.asyncio
    async def test_backend_permission_errors(self, test_config, temp_dir):
        """Test backend handling of permission errors."""
        backend = FFFlatfileStorageBackend(test_config)
        await backend.initialize()
        
        if os.name != 'nt':  # Skip on Windows due to permission handling differences
            # Create a file with no write permissions
            test_file = temp_dir / "readonly.txt"
            test_file.write_text("readonly content")
            test_file.chmod(0o444)  # Read-only
            
            try:
                # Attempt to write should fail gracefully
                result = await backend.write("readonly.txt", b"new content")
                assert result is False
            finally:
                test_file.chmod(0o644)  # Restore permissions for cleanup
    
    @pytest.mark.asyncio
    async def test_backend_disk_full_simulation(self, backend):
        """Test backend behavior when disk is full."""
        # This is difficult to test without actually filling the disk
        # Instead, we test with very large data that might exceed available space
        
        # Create a very large piece of data (100MB)
        large_data = b"X" * (100 * 1024 * 1024)
        
        try:
            # This might fail due to disk space or succeed if space is available
            result = await backend.write("large_file.bin", large_data)
            # Either result is acceptable - just shouldn't crash
            assert isinstance(result, bool)
        except OSError:
            # Expected if disk is full or filesystem doesn't support large files
            pass
    
    @pytest.mark.asyncio
    async def test_backend_concurrent_write_conflicts(self, backend):
        """Test backend handling of concurrent write conflicts."""
        key = "concurrent_conflict.txt"
        
        async def write_data(data: bytes):
            return await backend.write(key, data)
        
        # Attempt many concurrent writes to same file
        tasks = [write_data(f"Data from task {i}".encode()) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed, none should raise unhandled exceptions
        success_count = sum(1 for r in results if r is True)
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        
        assert success_count > 0, "At least some writes should succeed"
        # Exceptions are acceptable as long as they're handled
        if exception_count > 0:
            # Verify they're expected exception types
            for r in results:
                if isinstance(r, Exception):
                    assert isinstance(r, (OSError, FFFileOperationError))
    
    @pytest.mark.asyncio
    async def test_backend_invalid_key_handling(self, backend):
        """Test backend handling of invalid keys."""
        invalid_keys = [
            None,
            "",
            "file\x00null.txt",  # Null character
            "file\n\r.txt",      # Control characters
            "../../../etc/passwd",  # Path traversal attempt
        ]
        
        for invalid_key in invalid_keys:
            try:
                if invalid_key is None:
                    with pytest.raises((TypeError, AttributeError)):
                        await backend.write(invalid_key, b"data")
                else:
                    # Should handle gracefully
                    result = await backend.write(invalid_key, b"data")
                    # Either succeed (if sanitized) or fail gracefully
                    assert isinstance(result, bool)
            except (ValueError, OSError):
                # Expected for invalid keys
                pass


class TestConcurrencyErrors:
    """Test error handling in concurrent scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_creation(self, app_container):
        """Test concurrent creation of same user."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        async def create_user():
            return await storage.create_user("concurrent_user", {"test": True})
        
        # Try to create same user concurrently
        tasks = [create_user() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed, others should fail gracefully
        success_count = sum(1 for r in results if r is True)
        failure_count = sum(1 for r in results if r is False)
        
        assert success_count == 1, "Exactly one user creation should succeed"
        assert failure_count == 4, "Four should fail due to duplicate user"
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, app_container):
        """Test concurrent operations on same session."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        await storage.create_user("concurrent_session_user")
        session_id = await storage.create_session("concurrent_session_user", "Concurrent Test")
        
        async def add_message(i: int):
            message = FFMessageDTO(role=MessageRole.USER, content=f"Message {i}")
            return await storage.add_message("concurrent_session_user", session_id, message)
        
        # Add many messages concurrently
        tasks = [add_message(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        success_count = sum(1 for r in results if r is True)
        assert success_count >= 15, "Most concurrent messages should succeed"
        
        # Verify final state is consistent
        final_session = await storage.get_session("concurrent_session_user", session_id)
        final_messages = await storage.get_all_messages("concurrent_session_user", session_id)
        
        assert final_session.message_count == len(final_messages)
        assert len(final_messages) == success_count
    
    @pytest.mark.asyncio
    async def test_concurrent_file_lock_timeout(self, temp_dir, test_config):
        """Test file lock timeout handling."""
        from ff_utils.ff_file_ops import FFFileLock
        
        test_file = temp_dir / "lock_timeout_test.txt"
        
        # Create first lock with long timeout
        lock1 = FFFileLock(test_file, timeout=10.0, config=test_config)
        await lock1.acquire()
        
        # Create second lock with short timeout
        lock2 = FFFileLock(test_file, timeout=0.1, config=test_config)
        
        # Should timeout quickly
        with pytest.raises(FFLockTimeoutError):
            await lock2.acquire()
        
        # Clean up
        await lock1.release()


class TestResourceExhaustion:
    """Test behavior under resource exhaustion."""
    
    @pytest.mark.asyncio
    async def test_many_open_files(self, app_container):
        """Test behavior with many open files."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        await storage.create_user("file_test_user")
        
        # Create many sessions (each might open files)
        session_ids = []
        for i in range(50):  # Reasonable number to avoid system limits
            try:
                session_id = await storage.create_session("file_test_user", f"Session {i}")
                if session_id:
                    session_ids.append(session_id)
            except OSError:
                # Expected if we hit file descriptor limits
                break
        
        # Should create at least some sessions
        assert len(session_ids) > 10, "Should create multiple sessions"
        
        # Verify sessions are accessible
        for session_id in session_ids[:5]:  # Test first 5
            session = await storage.get_session("file_test_user", session_id)
            assert session is not None
    
    @pytest.mark.asyncio
    async def test_memory_intensive_operations(self, app_container):
        """Test memory-intensive operations."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        await storage.create_user("memory_test_user")
        session_id = await storage.create_session("memory_test_user", "Memory Test")
        
        # Create many messages
        for i in range(100):
            try:
                # Create moderately sized messages
                content = f"Message {i}: " + "X" * 1000  # 1KB each
                message = FFMessageDTO(role=MessageRole.USER, content=content)
                
                result = await storage.add_message("memory_test_user", session_id, message)
                if not result:
                    break  # Stop if we hit limits
            except MemoryError:
                # Expected if we exhaust memory
                break
        
        # Should have created some messages
        messages = await storage.get_all_messages("memory_test_user", session_id)
        assert len(messages) > 10, "Should create multiple messages"


class TestDataCorruption:
    """Test handling of data corruption scenarios."""
    
    @pytest.mark.asyncio
    async def test_corrupted_json_files(self, backend, temp_dir):
        """Test handling of corrupted JSON files."""
        # Create a corrupted JSON file directly
        corrupted_file = temp_dir / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"incomplete": "json"')  # Missing closing brace
        
        # Attempt to read should handle gracefully
        result = await backend.read("corrupted.json")
        if result:
            # If we got data, it should be the raw bytes
            assert isinstance(result, bytes)
        else:
            # None is also acceptable if file doesn't exist in backend's view
            assert result is None
    
    @pytest.mark.asyncio
    async def test_partial_file_writes(self, backend):
        """Test handling of partial file writes."""
        # This is hard to test directly, but we can test the recovery
        key = "partial_write_test.txt"
        complete_data = b"This is complete data that should be written atomically"
        
        # Normal write should succeed
        result = await backend.write(key, complete_data)
        assert result is True
        
        # Read should get complete data
        read_data = await backend.read(key)
        assert read_data == complete_data
        
        # Test overwrite
        new_data = b"This is the new complete data"
        result = await backend.write(key, new_data)
        assert result is True
        
        # Should get new complete data, not partial
        read_data = await backend.read(key)
        assert read_data == new_data


class TestNetworkAndIOErrors:
    """Test handling of network and I/O errors."""
    
    @pytest.mark.asyncio
    async def test_file_system_errors(self, app_container):
        """Test various filesystem error conditions."""
        storage = app_container.resolve(StorageProtocol)
        backend = app_container.resolve(BackendProtocol)
        
        # Mock various filesystem errors
        original_write = backend.write
        
        async def failing_write(key: str, data: bytes):
            if "fail_" in key:
                raise OSError("Simulated filesystem error")
            return await original_write(key, data)
        
        # Patch the write method
        backend.write = failing_write
        
        await storage.initialize()
        await storage.create_user("fs_error_user")
        session_id = await storage.create_session("fs_error_user", "FS Error Test")
        
        # Normal operation should work
        normal_message = FFMessageDTO(role=MessageRole.USER, content="Normal message")
        result = await storage.add_message("fs_error_user", session_id, normal_message)
        assert result is True
        
        # Operation that triggers error should be handled
        # (This depends on implementation - might succeed if error is in different layer)
        try:
            doc_result = await storage.save_document(
                "fs_error_user", session_id, "fail_document.txt", b"content"
            )
            # If it succeeds, that's fine too
            assert isinstance(doc_result, str)
        except OSError:
            # Expected error is acceptable
            pass
        
        # Restore original method
        backend.write = original_write


class TestRecoveryScenarios:
    """Test system recovery from various error conditions."""
    
    @pytest.mark.asyncio
    async def test_recovery_from_initialization_failure(self, temp_dir):
        """Test recovery after initialization failure."""
        config = create_default_config("test")
        config.storage.base_path = str(temp_dir)
        
        # Create storage manager
        backend = FFFlatfileStorageBackend(config)
        storage = FFStorageManager(config, backend)
        
        # First initialization should succeed
        result1 = await storage.initialize()
        assert result1 is True
        
        # Second initialization should also succeed (idempotent)
        result2 = await storage.initialize()
        assert result2 is True
        
        # Should be able to use storage normally
        user_created = await storage.create_user("recovery_user")
        assert user_created is True
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, app_container):
        """Test graceful degradation when components fail."""
        storage = app_container.resolve(StorageProtocol)
        await storage.initialize()
        
        # Create test data
        await storage.create_user("degradation_user")
        session_id = await storage.create_session("degradation_user", "Degradation Test")
        
        # Add initial message
        message = FFMessageDTO(role=MessageRole.USER, content="Initial message")
        result = await storage.add_message("degradation_user", session_id, message)
        assert result is True
        
        # Even if some operations fail, basic operations should still work
        user_exists = await storage.user_exists("degradation_user")
        assert user_exists is True
        
        session = await storage.get_session("degradation_user", session_id)
        assert session is not None
        
        messages = await storage.get_all_messages("degradation_user", session_id)
        assert len(messages) >= 1