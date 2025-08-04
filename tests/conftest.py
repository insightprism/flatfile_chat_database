"""
Pytest configuration and shared fixtures for the flatfile chat database test suite.

Provides common test utilities, fixtures, and mock factories to support
comprehensive testing of all system components.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Import core system components
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO, create_default_config
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFSituationalContextDTO, 
    FFUserProfileDTO, FFPersonaDTO, MessageRole
)
from ff_dependency_injection_manager import FFDependencyInjectionManager, ff_create_application_container
from ff_storage_manager import FFStorageManager
from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
from ff_protocols import (
    StorageProtocol, SearchProtocol, VectorStoreProtocol,
    DocumentProcessorProtocol, BackendProtocol, FileOperationsProtocol
)


# ===== PYTEST CONFIGURATION =====

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "e2e: End-to-end tests with full system")
    config.addinivalue_line("markers", "performance: Performance regression tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ===== DIRECTORY AND FILE FIXTURES =====

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test isolation."""
    temp_path = tempfile.mkdtemp(prefix="ff_test_")
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "test_data"


# ===== CONFIGURATION FIXTURES =====

@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths."""
    config = create_default_config("test")
    config.storage.base_path = str(temp_dir)
    config.storage.validate_json_on_read = True
    config.locking.enabled = True
    config.locking.timeout_seconds = 1.0
    config.search.enable_search_cache = False
    config.vector.cache_enabled = False
    config.document.cache_processed_documents = False
    return config


@pytest.fixture
def production_config(temp_dir):
    """Create a production-like configuration for testing."""
    config = create_default_config("production")
    config.storage.base_path = str(temp_dir)
    config.storage.validate_json_on_read = False
    config.locking.timeout_seconds = 30.0
    config.search.enable_search_cache = True
    return config


@pytest.fixture
def invalid_config():
    """Create an invalid configuration for error testing."""
    config = create_default_config("test")
    config.storage.base_path = ""  # Invalid empty path
    config.storage.max_message_size_bytes = -1  # Invalid negative size
    config.locking.timeout_seconds = -5.0  # Invalid negative timeout
    return config


# ===== DEPENDENCY INJECTION FIXTURES =====

@pytest.fixture
def di_container(test_config):
    """Create a clean dependency injection container for testing."""
    container = FFDependencyInjectionManager()
    container.register_singleton(FFConfigurationManagerConfigDTO, instance=test_config)
    return container


@pytest.fixture
def app_container(test_config, temp_dir):
    """Create a full application container with real components."""
    # Override config with test paths
    container = ff_create_application_container()
    container._singletons[FFConfigurationManagerConfigDTO] = test_config
    return container


@pytest.fixture
def mock_container():
    """Create a container with mocked services for unit testing."""
    container = FFDependencyInjectionManager()
    
    # Mock configuration
    mock_config = Mock(spec=FFConfigurationManagerConfigDTO)
    mock_config.storage.base_path = "/tmp/test"
    mock_config.storage.message_id_length = 8
    container.register_singleton(FFConfigurationManagerConfigDTO, instance=mock_config)
    
    # Mock services
    container.register_singleton(BackendProtocol, instance=Mock(spec=BackendProtocol))
    container.register_singleton(SearchProtocol, instance=Mock(spec=SearchProtocol))
    container.register_singleton(VectorStoreProtocol, instance=Mock(spec=VectorStoreProtocol))
    container.register_singleton(DocumentProcessorProtocol, instance=Mock(spec=DocumentProcessorProtocol))
    container.register_singleton(FileOperationsProtocol, instance=Mock(spec=FileOperationsProtocol))
    
    return container


# ===== STORAGE COMPONENT FIXTURES =====

@pytest.fixture
async def storage_manager(test_config, temp_dir):
    """Create a storage manager with test configuration."""
    backend = FFFlatfileStorageBackend(test_config)
    await backend.initialize()
    
    manager = FFStorageManager(test_config, backend)
    await manager.initialize()
    return manager


@pytest.fixture
async def mock_storage_manager():
    """Create a mock storage manager for unit testing."""
    mock_manager = AsyncMock(spec=FFStorageManager)
    
    # Configure common return values
    mock_manager.create_user.return_value = True
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.add_message.return_value = True
    mock_manager.user_exists.return_value = True
    mock_manager.get_session.return_value = sample_session()
    
    return mock_manager


@pytest.fixture
async def backend(test_config):
    """Create a flatfile backend for testing."""
    backend = FFFlatfileStorageBackend(test_config)
    await backend.initialize()
    return backend


# ===== SAMPLE DATA FACTORIES =====

def sample_user_profile(user_id: str = "test_user") -> FFUserProfileDTO:
    """Create a sample user profile for testing."""
    return FFUserProfileDTO(
        user_id=user_id,
        username=f"user_{user_id}",
        preferences={"theme": "dark", "language": "en"},
        metadata={"created_via": "test", "test_flag": True}
    )


def sample_session(user_id: str = "test_user", session_id: str = "test_session_123") -> FFSessionDTO:
    """Create a sample session for testing."""
    return FFSessionDTO(
        session_id=session_id,
        user_id=user_id,
        title="Test Chat Session",
        message_count=0,
        metadata={"test_session": True}
    )


def sample_message(
    role: MessageRole = MessageRole.USER,
    content: str = "Test message content",
    message_id: str = "msg_test123"
) -> FFMessageDTO:
    """Create a sample message for testing."""
    return FFMessageDTO(
        id=message_id,
        role=role,
        content=content,
        timestamp=datetime.now().isoformat(),
        metadata={"test_message": True}
    )


def sample_document(filename: str = "test_doc.txt") -> FFDocumentDTO:
    """Create a sample document for testing."""
    return FFDocumentDTO(
        filename=filename,
        original_name=filename,
        path=f"documents/{filename}",
        mime_type="text/plain",
        size=1024,
        uploaded_by="test_user",
        metadata={"test_document": True}
    )


def sample_context() -> FFSituationalContextDTO:
    """Create a sample situational context for testing."""
    return FFSituationalContextDTO(
        summary="Test conversation about testing",
        key_points=["Testing is important", "We're testing the system"],
        entities={"topics": ["testing", "system"], "users": ["test_user"]},
        confidence=0.85,
        metadata={"test_context": True}
    )


# ===== SAMPLE DATA FIXTURES =====

@pytest.fixture
def sample_users() -> List[FFUserProfileDTO]:
    """Create multiple sample users for testing."""
    return [
        sample_user_profile("alice"),
        sample_user_profile("bob"),
        sample_user_profile("charlie")
    ]


@pytest.fixture
def sample_messages() -> List[FFMessageDTO]:
    """Create multiple sample messages for testing."""
    return [
        sample_message(MessageRole.USER, "Hello, how are you?", "msg_001"),
        sample_message(MessageRole.ASSISTANT, "I'm doing well, thank you!", "msg_002"),
        sample_message(MessageRole.USER, "Can you help me with testing?", "msg_003"),
        sample_message(MessageRole.ASSISTANT, "Of course! I'd be happy to help.", "msg_004")
    ]


@pytest.fixture
def sample_documents() -> List[FFDocumentDTO]:
    """Create multiple sample documents for testing."""
    return [
        sample_document("readme.txt"),
        sample_document("guide.md"),
        sample_document("config.json")
    ]


# ===== TEST UTILITIES =====

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_conversation(message_count: int = 10) -> List[FFMessageDTO]:
        """Generate a realistic test conversation."""
        messages = []
        for i in range(message_count):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            content = f"Test message {i + 1} from {role.value}"
            msg_id = f"msg_{i:03d}"
            messages.append(sample_message(role, content, msg_id))
        return messages
    
    @staticmethod
    def create_large_message(size_bytes: int = 1024) -> FFMessageDTO:
        """Create a message with specific size for testing limits."""
        content = "A" * size_bytes
        return sample_message(MessageRole.USER, content, "msg_large")
    
    @staticmethod
    def create_test_session_with_messages(
        user_id: str = "test_user",
        session_id: str = "test_session",
        message_count: int = 5
    ) -> tuple[FFSessionDTO, List[FFMessageDTO]]:
        """Create a session with associated messages."""
        session = sample_session(user_id, session_id)
        session.message_count = message_count
        messages = TestDataGenerator.create_test_conversation(message_count)
        return session, messages


@pytest.fixture
def test_data_generator():
    """Provide the test data generator utility."""
    return TestDataGenerator


# ===== ASSERTION HELPERS =====

class TestAssertions:
    """Custom assertion helpers for testing."""
    
    @staticmethod
    def assert_valid_session(session: FFSessionDTO):
        """Assert that a session object is valid."""
        assert session.session_id
        assert session.user_id
        assert session.title
        assert session.created_at
        assert isinstance(session.message_count, int)
        assert session.message_count >= 0
    
    @staticmethod
    def assert_valid_message(message: FFMessageDTO):
        """Assert that a message object is valid."""
        assert message.id
        assert message.role in [role.value for role in MessageRole]
        assert message.content
        assert message.timestamp
    
    @staticmethod
    def assert_valid_user_profile(profile: FFUserProfileDTO):
        """Assert that a user profile is valid."""
        assert profile.user_id
        assert profile.created_at
        assert isinstance(profile.preferences, dict)
        assert isinstance(profile.metadata, dict)
    
    @staticmethod
    async def assert_directory_structure(base_path: Path, expected_dirs: List[str]):
        """Assert that expected directory structure exists."""
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"


@pytest.fixture
def test_assertions():
    """Provide test assertion helpers."""
    return TestAssertions


# ===== PERFORMANCE TESTING UTILITIES =====

@pytest.fixture
def performance_timer():
    """Simple performance timing utility."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
        
        @property
        def elapsed_seconds(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0.0
        
        def assert_under(self, max_seconds: float):
            assert self.elapsed_seconds <= max_seconds, \
                f"Operation took {self.elapsed_seconds:.2f}s, expected under {max_seconds}s"
    
    return Timer()


# ===== MOCK FACTORIES =====

@pytest.fixture
def mock_backend():
    """Create a mock backend for unit testing."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.initialize.return_value = True
    backend.read.return_value = b'{"test": "data"}'
    backend.write.return_value = True
    backend.exists.return_value = True
    backend.delete.return_value = True
    backend.list_keys.return_value = ["key1", "key2", "key3"]
    return backend


@pytest.fixture
def mock_search_engine():
    """Create a mock search engine for testing."""
    search = AsyncMock(spec=SearchProtocol)
    search.search.return_value = []
    search.search_by_entities.return_value = []
    search.extract_entities.return_value = {"topics": ["test"]}
    return search


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    vector_store = AsyncMock(spec=VectorStoreProtocol)
    vector_store.store_vectors.return_value = True
    vector_store.search_similar.return_value = []
    vector_store.delete_document_vectors.return_value = True
    vector_store.get_vector_stats.return_value = {"total_vectors": 0}
    return vector_store


# ===== ERROR SIMULATION UTILITIES =====

@pytest.fixture
def error_simulator():
    """Utility for simulating various error conditions."""
    class ErrorSimulator:
        @staticmethod
        def file_not_found_error():
            return FileNotFoundError("Test file not found")
        
        @staticmethod
        def permission_error():
            return PermissionError("Test permission denied")
        
        @staticmethod
        def disk_full_error():
            return OSError("Test disk full")
        
        @staticmethod
        def network_error():
            return ConnectionError("Test network error")
        
        @staticmethod
        def timeout_error():
            return TimeoutError("Test operation timeout")
    
    return ErrorSimulator()


# ===== CLEANUP UTILITIES =====

@pytest.fixture(autouse=True)
async def cleanup_global_state():
    """Automatically clean up global state after each test."""
    yield
    
    # Clear any global DI container state
    from ff_dependency_injection_manager import ff_clear_global_container
    ff_clear_global_container()