"""
Comprehensive tests for the flatfile storage backend implementation.

Tests the FFFlatfileStorageBackend class including file operations,
locking mechanisms, directory management, and error handling.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, mock_open
from concurrent.futures import ThreadPoolExecutor
import sys

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
from ff_utils.ff_file_ops import FFFileLock, FFFileOperationError, FFAtomicWriteError, FFLockTimeoutError


class TestBackendInitialization:
    """Test backend initialization and setup."""
    
    @pytest.mark.asyncio
    async def test_backend_initialization_success(self, test_config, temp_dir):
        """Test successful backend initialization."""
        backend = FFFlatfileStorageBackend(test_config)
        
        result = await backend.initialize()
        assert result is True
        
        # Base directory should exist
        assert backend.base_path.exists()
        assert backend.base_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_backend_initialization_creates_subdirectories(self, test_config, temp_dir):
        """Test that initialization creates required subdirectories."""
        backend = FFFlatfileStorageBackend(test_config)
        
        await backend.initialize()
        
        # Check that standard subdirectories are created
        expected_subdirs = [
            test_config.panel.global_personas_directory,
            test_config.panel.panel_sessions_directory,
            test_config.storage.system_config_directory
        ]
        
        for subdir in expected_subdirs:
            subdir_path = backend.base_path / subdir
            assert subdir_path.exists(), f"Subdirectory {subdir} should be created"
            assert subdir_path.is_dir(), f"{subdir} should be a directory"
    
    @pytest.mark.asyncio
    async def test_backend_initialization_failure(self, test_config):
        """Test backend initialization with permission errors."""
        # Create backend with invalid path (root directory that requires permissions)
        test_config.storage.base_path = "/root/invalid_path"
        backend = FFFlatfileStorageBackend(test_config)
        
        result = await backend.initialize()
        assert result is False
    
    def test_backend_absolute_path_resolution(self, test_config, temp_dir):
        """Test that backend resolves paths to absolute paths."""
        backend = FFFlatfileStorageBackend(test_config)
        
        assert backend.base_path.is_absolute()
        assert str(backend.base_path) == str(Path(test_config.storage.base_path).resolve())


class TestBasicFileOperations:
    """Test basic file read/write operations."""
    
    @pytest.mark.asyncio
    async def test_write_and_read_data(self, backend):
        """Test writing and reading data."""
        test_data = b"Hello, world! This is test data."
        key = "test_files/hello.txt"
        
        # Write data
        write_result = await backend.write(key, test_data)
        assert write_result is True
        
        # Read data back
        read_data = await backend.read(key)
        assert read_data == test_data
    
    @pytest.mark.asyncio
    async def test_write_creates_directories(self, backend):
        """Test that write operations create necessary directories."""
        test_data = b"Directory creation test"
        key = "deep/nested/directory/structure/test.txt"
        
        write_result = await backend.write(key, test_data)
        assert write_result is True
        
        # Verify directory structure was created
        file_path = backend.base_path / key
        assert file_path.exists()
        assert file_path.parent.exists()
        assert file_path.parent.is_dir()
    
    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, backend):
        """Test reading nonexistent file returns None."""
        result = await backend.read("nonexistent/file.txt")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_write_empty_data(self, backend):
        """Test writing empty data."""
        key = "empty_file.txt"
        empty_data = b""
        
        write_result = await backend.write(key, empty_data)
        assert write_result is True
        
        read_data = await backend.read(key)
        assert read_data == empty_data
    
    @pytest.mark.asyncio
    async def test_write_binary_data(self, backend):
        """Test writing and reading binary data."""
        # Create some binary data
        binary_data = bytes(range(256))
        key = "binary_test.bin"
        
        write_result = await backend.write(key, binary_data)
        assert write_result is True
        
        read_data = await backend.read(key)
        assert read_data == binary_data
    
    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, backend):
        """Test overwriting existing file."""
        key = "overwrite_test.txt"
        original_data = b"Original content"
        new_data = b"New content that overwrites"
        
        # Write original data
        await backend.write(key, original_data)
        original_read = await backend.read(key)
        assert original_read == original_data
        
        # Overwrite with new data
        await backend.write(key, new_data)
        new_read = await backend.read(key)
        assert new_read == new_data


class TestAppendOperations:
    """Test file append operations."""
    
    @pytest.mark.asyncio
    async def test_append_to_existing_file(self, backend):
        """Test appending to existing file."""
        key = "append_test.txt"
        initial_data = b"Initial content\n"
        append_data = b"Appended content\n"
        
        # Write initial data
        await backend.write(key, initial_data)
        
        # Append more data
        append_result = await backend.append(key, append_data)
        assert append_result is True
        
        # Read combined data
        combined_data = await backend.read(key)
        assert combined_data == initial_data + append_data
    
    @pytest.mark.asyncio
    async def test_append_to_nonexistent_file(self, backend):
        """Test appending to nonexistent file creates it."""
        key = "new_append_file.txt"
        append_data = b"First content via append"
        
        append_result = await backend.append(key, append_data)
        assert append_result is True
        
        read_data = await backend.read(key)
        assert read_data == append_data
    
    @pytest.mark.asyncio
    async def test_multiple_appends(self, backend):
        """Test multiple append operations."""
        key = "multi_append.txt"
        
        data_chunks = [
            b"Chunk 1\n",
            b"Chunk 2\n", 
            b"Chunk 3\n"
        ]
        
        # Append multiple chunks
        for chunk in data_chunks:
            result = await backend.append(key, chunk)
            assert result is True
        
        # Read final result
        final_data = await backend.read(key)
        expected_data = b"".join(data_chunks)
        assert final_data == expected_data


class TestFileExistence:
    """Test file existence checking."""
    
    @pytest.mark.asyncio
    async def test_exists_for_existing_file(self, backend):
        """Test exists() returns True for existing file."""
        key = "exists_test.txt"
        test_data = b"File exists"
        
        # Create file
        await backend.write(key, test_data)
        
        # Check existence
        exists = await backend.exists(key)
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_exists_for_nonexistent_file(self, backend):
        """Test exists() returns False for nonexistent file."""
        exists = await backend.exists("definitely_does_not_exist.txt")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_exists_for_directory(self, backend):
        """Test exists() behavior for directories."""
        # Create a directory by writing a file in it
        key = "test_dir/file.txt"
        await backend.write(key, b"content")
        
        # Check if directory itself is detected as existing
        # Note: This tests the current implementation behavior
        dir_exists = await backend.exists("test_dir")
        # The behavior may vary depending on implementation
        # Just ensure it doesn't crash
        assert isinstance(dir_exists, bool)


class TestFileDeletion:
    """Test file deletion operations."""
    
    @pytest.mark.asyncio
    async def test_delete_existing_file(self, backend):
        """Test deleting existing file."""
        key = "delete_test.txt"
        test_data = b"File to be deleted"
        
        # Create file
        await backend.write(key, test_data)
        assert await backend.exists(key) is True
        
        # Delete file
        delete_result = await backend.delete(key)
        assert delete_result is True
        
        # Verify file is gone
        assert await backend.exists(key) is False
        assert await backend.read(key) is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, backend):
        """Test deleting nonexistent file."""
        delete_result = await backend.delete("nonexistent_file.txt")
        # Should handle gracefully (implementation may vary)
        assert isinstance(delete_result, bool)
    
    @pytest.mark.asyncio
    async def test_delete_directory_with_contents(self, backend):
        """Test deleting directory with contents."""
        # Create files in directory
        files = [
            "delete_dir/file1.txt",
            "delete_dir/file2.txt", 
            "delete_dir/subdir/file3.txt"
        ]
        
        for file_key in files:
            await backend.write(file_key, b"content")
        
        # Delete entire directory
        delete_result = await backend.delete("delete_dir")
        assert delete_result is True
        
        # Verify all files are gone
        for file_key in files:
            assert await backend.exists(file_key) is False


class TestKeyListing:
    """Test key listing operations."""
    
    @pytest.mark.asyncio
    async def test_list_keys_in_directory(self, backend):
        """Test listing keys in a directory."""
        # Create multiple files
        files = {
            "list_test/file1.txt": b"content1",
            "list_test/file2.txt": b"content2",
            "list_test/subdir/file3.txt": b"content3"
        }
        
        for key, content in files.items():
            await backend.write(key, content)
        
        # List keys
        keys = await backend.list_keys("list_test")
        
        # Should find files (exact behavior depends on implementation)
        assert isinstance(keys, list)
        assert len(keys) > 0
        
        # Check that some expected files are found
        key_strings = [str(k) for k in keys]
        has_file1 = any("file1.txt" in k for k in key_strings)
        has_file2 = any("file2.txt" in k for k in key_strings)
        
        assert has_file1 or has_file2, "Should find at least one of the created files"
    
    @pytest.mark.asyncio
    async def test_list_keys_empty_directory(self, backend):
        """Test listing keys in empty directory."""
        keys = await backend.list_keys("empty_directory")
        assert isinstance(keys, list)
        # Empty directory should return empty list or handle gracefully
    
    @pytest.mark.asyncio
    async def test_list_keys_with_pattern(self, backend):
        """Test listing keys with pattern matching."""
        # Create files with different extensions
        files = {
            "pattern_test/document.txt": b"text content",
            "pattern_test/image.jpg": b"image content",
            "pattern_test/data.json": b"json content",
            "pattern_test/readme.md": b"markdown content"
        }
        
        for key, content in files.items():
            await backend.write(key, content)
        
        # List with pattern (if supported)
        try:
            txt_keys = await backend.list_keys("pattern_test", pattern="*.txt")
            if txt_keys:  # If pattern matching is supported
                assert isinstance(txt_keys, list)
                # Should find only .txt files
                txt_found = any("document.txt" in str(k) for k in txt_keys)
                jpg_found = any("image.jpg" in str(k) for k in txt_keys)
                assert txt_found
                assert not jpg_found
        except (TypeError, ValueError):
            # Pattern matching might not be supported
            pass


class TestConcurrentOperations:
    """Test concurrent file operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_writes_different_files(self, backend):
        """Test concurrent writes to different files."""
        async def write_file(i):
            key = f"concurrent/file_{i}.txt"
            data = f"Content for file {i}".encode()
            return await backend.write(key, data)
        
        # Write multiple files concurrently
        tasks = [write_file(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All writes should succeed
        assert all(results)
        
        # Verify all files exist
        for i in range(10):
            key = f"concurrent/file_{i}.txt"
            exists = await backend.exists(key)
            assert exists is True
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_same_file(self, backend):
        """Test concurrent read/write operations on same file."""
        key = "concurrent_rw.txt"
        initial_data = b"Initial content"
        
        # Write initial data
        await backend.write(key, initial_data)
        
        async def read_file():
            return await backend.read(key)
        
        async def write_file():
            return await backend.write(key, b"Updated content")
        
        # Run concurrent read and write
        read_task = asyncio.create_task(read_file())
        write_task = asyncio.create_task(write_file())
        
        read_result, write_result = await asyncio.gather(read_task, write_task)
        
        # Write should succeed
        assert write_result is True
        
        # Read should return some valid content (either old or new)
        assert read_result in [initial_data, b"Updated content"]
    
    @pytest.mark.asyncio
    async def test_concurrent_appends_same_file(self, backend):
        """Test concurrent append operations to same file."""
        key = "concurrent_append.txt"
        
        async def append_data(i):
            data = f"Line {i}\n".encode()
            return await backend.append(key, data)
        
        # Append multiple lines concurrently
        tasks = [append_data(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All appends should succeed
        assert all(results)
        
        # Read final content
        final_content = await backend.read(key)
        assert final_content is not None
        
        # Should contain all lines (order may vary)
        content_str = final_content.decode()
        for i in range(5):
            assert f"Line {i}" in content_str


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_write_with_invalid_path_characters(self, backend):
        """Test writing with invalid path characters."""
        # Test with various potentially problematic characters
        problematic_keys = [
            "test\x00null.txt",  # Null character
            "test/../parent.txt",  # Parent directory reference
            "test/./current.txt",  # Current directory reference
        ]
        
        for key in problematic_keys:
            try:
                # Some keys might be sanitized or rejected
                result = await backend.write(key, b"test content")
                # If write succeeds, verify read works
                if result:
                    read_data = await backend.read(key)
                    assert read_data == b"test content"
            except (ValueError, OSError):
                # Some invalid keys should raise errors
                pass
    
    @pytest.mark.asyncio
    async def test_write_very_long_key(self, backend):
        """Test writing with very long file path."""
        # Create a very long key
        long_key = "long/" * 50 + "file.txt"
        
        try:
            result = await backend.write(long_key, b"content")
            if result:
                # If write succeeds, read should work too
                read_data = await backend.read(long_key)
                assert read_data == b"content"
        except (OSError, ValueError):
            # Some systems may reject very long paths
            pass
    
    @pytest.mark.asyncio
    async def test_operations_with_none_key(self, backend):
        """Test operations with None key."""
        with pytest.raises((TypeError, AttributeError)):
            await backend.write(None, b"data")
        
        with pytest.raises((TypeError, AttributeError)):
            await backend.read(None)
        
        with pytest.raises((TypeError, AttributeError)):
            await backend.exists(None)
    
    @pytest.mark.asyncio
    async def test_write_with_none_data(self, backend):
        """Test writing None data."""
        with pytest.raises((TypeError, AttributeError)):
            await backend.write("test.txt", None)
    
    @pytest.mark.asyncio
    async def test_write_with_string_data(self, backend):
        """Test writing string data (should expect bytes)."""
        with pytest.raises((TypeError, AttributeError)):
            await backend.write("test.txt", "string data instead of bytes")


class TestFileLocking:
    """Test file locking mechanisms."""
    
    @pytest.mark.asyncio
    async def test_file_lock_creation(self, temp_dir, test_config):
        """Test creating file locks."""
        test_file = temp_dir / "lock_test.txt"
        
        lock = FFFileLock(test_file, timeout=1.0, config=test_config)
        assert lock.path == test_file
        assert lock.lock_path == test_file.with_suffix(test_file.suffix + '.lock')
        assert lock.timeout == 1.0
    
    @pytest.mark.asyncio
    async def test_file_lock_acquire_release(self, temp_dir, test_config):
        """Test acquiring and releasing file locks."""
        test_file = temp_dir / "lock_acquire_test.txt"
        
        lock = FFFileLock(test_file, config=test_config)
        
        # Acquire lock
        await lock.acquire()
        assert lock.acquired_at is not None
        
        # Release lock
        await lock.release()
        assert not lock.lock_path.exists()  # Lock file should be cleaned up
    
    @pytest.mark.asyncio
    async def test_file_lock_timeout(self, temp_dir, test_config):
        """Test file lock timeout behavior."""
        test_file = temp_dir / "lock_timeout_test.txt"
        
        # Create first lock and hold it
        lock1 = FFFileLock(test_file, timeout=0.1, config=test_config)
        await lock1.acquire()
        
        # Try to acquire second lock - should timeout
        lock2 = FFFileLock(test_file, timeout=0.1, config=test_config)
        
        with pytest.raises(FFLockTimeoutError):
            await lock2.acquire()
        
        # Clean up
        await lock1.release()
    
    @pytest.mark.asyncio
    async def test_file_lock_context_manager(self, temp_dir, test_config):
        """Test file lock as context manager."""
        test_file = temp_dir / "lock_context_test.txt"
        
        lock = FFFileLock(test_file, config=test_config)
        
        async with lock:
            # Lock should be acquired
            assert lock.acquired_at is not None
            assert lock.lock_path.exists()
        
        # Lock should be released after context
        assert not lock.lock_path.exists()


class TestAtomicOperations:
    """Test atomic file operations."""
    
    @pytest.mark.asyncio
    async def test_atomic_write_success(self, backend):
        """Test successful atomic write operation."""
        key = "atomic_test.txt"
        data = b"Atomic write test content"
        
        result = await backend.write(key, data)
        assert result is True
        
        # Verify content
        read_data = await backend.read(key)
        assert read_data == data
    
    @pytest.mark.asyncio
    async def test_atomic_write_large_data(self, backend):
        """Test atomic write with large data."""
        key = "large_atomic.txt"
        # Create 1MB of data
        large_data = b"X" * (1024 * 1024)
        
        result = await backend.write(key, large_data)
        assert result is True
        
        # Verify all data was written
        read_data = await backend.read(key)
        assert len(read_data) == len(large_data)
        assert read_data == large_data
    
    @pytest.mark.asyncio
    async def test_atomic_write_interrupted_simulation(self, backend, temp_dir):
        """Test atomic write behavior when simulating interruption."""
        key = "interrupted_write.txt"
        data = b"This write might be interrupted"
        
        # This test is more about ensuring the implementation uses
        # proper atomic writes that would survive interruption
        result = await backend.write(key, data)
        assert result is True
        
        # In a real atomic write, even if interrupted, we should either
        # get the complete file or no file at all, never partial content
        read_data = await backend.read(key)
        assert read_data == data  # Complete content or None


class TestPerformance:
    """Performance tests for backend operations."""
    
    @pytest.mark.asyncio
    async def test_write_performance(self, backend, performance_timer):
        """Test write operation performance."""
        data = b"Performance test data"
        
        performance_timer.start()
        for i in range(100):
            await backend.write(f"perf/write_{i}.txt", data)
        performance_timer.stop()
        
        # Should complete 100 writes in reasonable time
        performance_timer.assert_under(5.0)  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_read_performance(self, backend, performance_timer):
        """Test read operation performance."""
        data = b"Performance test data for reading"
        key = "perf_read.txt"
        
        # Write test file
        await backend.write(key, data)
        
        performance_timer.start()
        for _ in range(100):
            read_data = await backend.read(key)
            assert read_data == data
        performance_timer.stop()
        
        # Should complete 100 reads in reasonable time
        performance_timer.assert_under(2.0)  # 2 seconds max
    
    @pytest.mark.asyncio
    async def test_exists_performance(self, backend, performance_timer):
        """Test exists operation performance."""
        # Create test file
        await backend.write("perf_exists.txt", b"exists test")
        
        performance_timer.start()
        for _ in range(100):
            exists = await backend.exists("perf_exists.txt")
            assert exists is True
        performance_timer.stop()
        
        # Should complete 100 existence checks quickly
        performance_timer.assert_under(1.0)  # 1 second max


@pytest.mark.integration
class TestBackendIntegration:
    """Integration tests for backend with other components."""
    
    @pytest.mark.asyncio
    async def test_backend_with_storage_manager(self, test_config, temp_dir):
        """Test backend integration with storage manager."""
        from ff_storage_manager import FFStorageManager
        
        backend = FFFlatfileStorageBackend(test_config)
        await backend.initialize()
        
        storage_manager = FFStorageManager(test_config, backend)
        await storage_manager.initialize()
        
        # Test that storage manager can use backend
        assert storage_manager.backend == backend
        
        # Test storage operation that uses backend
        result = await storage_manager.create_user("integration_user")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_backend_with_dependency_injection(self, di_container, test_config):
        """Test backend with dependency injection."""
        from ff_protocols import BackendProtocol
        
        backend = FFFlatfileStorageBackend(test_config)
        di_container.register_singleton(BackendProtocol, instance=backend)
        
        resolved_backend = di_container.resolve(BackendProtocol)
        assert resolved_backend is backend
        
        # Test basic operation
        await resolved_backend.initialize()
        result = await resolved_backend.write("di_test.txt", b"DI test content")
        assert result is True


class TestBackendEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    @pytest.mark.asyncio
    async def test_unicode_filenames(self, backend):
        """Test handling of unicode filenames."""
        unicode_keys = [
            "unicode/测试文件.txt",  # Chinese
            "unicode/файл.txt",     # Russian
            "unicode/файл_émoji_🎉.txt"  # Mixed with emoji
        ]
        
        for key in unicode_keys:
            try:
                # Some filesystems/systems may not support unicode
                result = await backend.write(key, b"unicode test")
                if result:
                    read_data = await backend.read(key)
                    assert read_data == b"unicode test"
            except (UnicodeError, OSError):
                # Some systems may not support unicode filenames
                pass
    
    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, backend):
        """Test handling content with special characters."""
        key = "special_content.txt"
        
        # Content with various special characters and encodings
        special_content = "Special: \x00\x01\x02\n\r\t🎉中文русский".encode('utf-8')
        
        result = await backend.write(key, special_content)
        assert result is True
        
        read_data = await backend.read(key)
        assert read_data == special_content
    
    @pytest.mark.asyncio
    async def test_empty_directory_operations(self, backend):
        """Test operations with empty directory names."""
        # Test with empty key
        with pytest.raises((ValueError, TypeError)):
            await backend.write("", b"content")
        
        # Test with just directory separator
        try:
            result = await backend.write("/", b"content")
            # Behavior may vary by implementation
        except (ValueError, OSError):
            pass
    
    @pytest.mark.asyncio
    async def test_repeated_operations_same_file(self, backend):
        """Test repeated operations on the same file."""
        key = "repeated_ops.txt"
        
        # Perform many operations on same file
        for i in range(10):
            data = f"Content iteration {i}".encode()
            
            # Write
            result = await backend.write(key, data)
            assert result is True
            
            # Read
            read_data = await backend.read(key)
            assert read_data == data
            
            # Check existence
            exists = await backend.exists(key)
            assert exists is True
        
        # Final cleanup
        delete_result = await backend.delete(key)
        assert delete_result is True