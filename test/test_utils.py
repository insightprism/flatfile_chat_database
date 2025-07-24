"""
Tests for utility functions.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
import json
import sys

# Add the parent directory to the path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.utils import (
    atomic_write, safe_read, ensure_directory,
    write_json, read_json, append_jsonl, read_jsonl,
    sanitize_filename, generate_session_id, get_user_path
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def config():
    """Create test configuration"""
    return StorageConfig(
        storage_base_path="./test_data",
        atomic_write_temp_suffix=".tmp",
        create_parent_directories=True
    )


class TestFileOperations:
    """Test file operations"""
    
    @pytest.mark.asyncio
    async def test_atomic_write_and_read(self, temp_dir, config):
        """Test atomic write and read operations"""
        test_file = temp_dir / "test.txt"
        test_data = "Hello, World!"
        
        # Write data
        result = await atomic_write(test_file, test_data, config, mode='w')
        assert result is True
        assert test_file.exists()
        
        # Read data back
        data = await safe_read(test_file, mode='r')
        assert data == test_data
    
    @pytest.mark.asyncio
    async def test_atomic_write_binary(self, temp_dir, config):
        """Test atomic write with binary data"""
        test_file = temp_dir / "test.bin"
        test_data = b"Binary data \x00\x01\x02"
        
        # Write binary data
        result = await atomic_write(test_file, test_data, config, mode='wb')
        assert result is True
        
        # Read binary data back
        data = await safe_read(test_file, mode='rb')
        assert data == test_data
    
    @pytest.mark.asyncio
    async def test_safe_read_nonexistent(self, temp_dir):
        """Test reading non-existent file returns None"""
        test_file = temp_dir / "nonexistent.txt"
        data = await safe_read(test_file)
        assert data is None
    
    @pytest.mark.asyncio
    async def test_ensure_directory(self, temp_dir):
        """Test directory creation"""
        test_dir = temp_dir / "sub" / "dir" / "path"
        result = await ensure_directory(test_dir)
        assert result is True
        assert test_dir.exists()
        assert test_dir.is_dir()


class TestJSONOperations:
    """Test JSON utilities"""
    
    @pytest.mark.asyncio
    async def test_write_and_read_json(self, temp_dir, config):
        """Test JSON write and read"""
        test_file = temp_dir / "test.json"
        test_data = {
            "name": "Test User",
            "age": 30,
            "tags": ["python", "async", "storage"]
        }
        
        # Write JSON
        result = await write_json(test_file, test_data, config)
        assert result is True
        
        # Read JSON back
        data = await read_json(test_file, config)
        assert data == test_data
    
    @pytest.mark.asyncio
    async def test_jsonl_operations(self, temp_dir, config):
        """Test JSONL append and read"""
        test_file = temp_dir / "messages.jsonl"
        
        # Append multiple entries
        entries = [
            {"id": 1, "message": "First"},
            {"id": 2, "message": "Second"},
            {"id": 3, "message": "Third"}
        ]
        
        for entry in entries:
            result = await append_jsonl(test_file, entry, config)
            assert result is True
        
        # Read all entries
        data = await read_jsonl(test_file, config)
        assert len(data) == 3
        assert data == entries
        
        # Read with limit
        data = await read_jsonl(test_file, config, limit=2)
        assert len(data) == 2
        assert data == entries[:2]
        
        # Read with offset
        data = await read_jsonl(test_file, config, offset=1)
        assert len(data) == 2
        assert data == entries[1:]


class TestPathUtilities:
    """Test path management utilities"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Test various problematic filenames
        assert sanitize_filename("hello world.txt") == "hello_world.txt"
        assert sanitize_filename("file/with\\slashes.pdf") == "file_with_slashes.pdf"
        assert sanitize_filename("special!@#$%chars.doc") == "special_____chars.doc"
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("...") == "unnamed"
        
        # Test unicode normalization
        assert sanitize_filename("café.txt") == "caf.txt"  # é is removed
        
        # Test length truncation
        long_name = "a" * 300 + ".txt"
        sanitized = sanitize_filename(long_name, max_length=255)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")
    
    def test_generate_session_id(self, config):
        """Test session ID generation"""
        session_id = generate_session_id(config)
        assert session_id.startswith(config.session_id_prefix)
        assert len(session_id) == len(config.session_id_prefix) + 1 + 8 + 1 + 6  # prefix_YYYYMMDD_HHMMSS
    
    def test_get_user_path(self, config):
        """Test user path construction"""
        base_path = Path(config.storage_base_path)
        user_path = get_user_path(base_path, "test_user")
        assert user_path == base_path / "test_user"
        
        # Test with special characters in user ID
        user_path = get_user_path(base_path, "user@example.com")
        assert user_path == base_path / "user_example.com"


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration passes validation"""
        config = StorageConfig()
        config.validate()  # Should not raise
    
    def test_invalid_config(self):
        """Test invalid configuration raises error"""
        config = StorageConfig(max_message_size_bytes=-1)
        with pytest.raises(ValueError):
            config.validate()
        
        config = StorageConfig(context_confidence_threshold=1.5)
        with pytest.raises(ValueError):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])