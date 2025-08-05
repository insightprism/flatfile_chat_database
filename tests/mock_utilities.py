"""
Advanced mock utilities for comprehensive testing.

Provides sophisticated mocking capabilities, behavior simulation,
and realistic error injection for thorough testing of all components.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ff_protocols import (
    StorageProtocol, SearchProtocol, VectorStoreProtocol,
    DocumentProcessorProtocol, BackendProtocol, FileOperationsProtocol,
    CompressionProtocol, StreamingProtocol, EmbeddingProtocol, ChunkingProtocol
)
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFSearchResultDTO, FFSearchQueryDTO
)


@dataclass
class MockBehaviorConfig:
    """Configuration for mock behavior patterns."""
    success_rate: float = 0.95              # Success rate for operations (0.0-1.0)
    latency_min_ms: int = 10                # Minimum operation latency
    latency_max_ms: int = 100               # Maximum operation latency
    error_patterns: List[str] = None        # Error patterns to simulate
    realistic_data: bool = True             # Generate realistic vs simple data
    state_tracking: bool = True             # Track state changes
    call_logging: bool = False              # Log all mock calls


class AdvancedMockFactory:
    """Factory for creating sophisticated mocks with realistic behavior."""
    
    def __init__(self, config: Optional[MockBehaviorConfig] = None):
        """Initialize with behavior configuration."""
        self.config = config or MockBehaviorConfig()
        if self.config.error_patterns is None:
            self.config.error_patterns = ["network", "disk", "permission", "timeout"]
        
        # Internal state tracking
        self._call_counts = {}
        self._state_data = {}
        self._error_counters = {}
    
    # === Storage Protocol Mocks ===
    
    def create_storage_mock(self, **overrides) -> Mock:
        """Create a sophisticated storage protocol mock."""
        mock_storage = AsyncMock(spec=StorageProtocol)
        
        # Track internal state
        self._state_data["users"] = {}
        self._state_data["sessions"] = {}
        self._state_data["messages"] = {}
        
        # Configure realistic behaviors
        mock_storage.initialize = self._create_async_method(
            "initialize", lambda: True, **overrides.get("initialize", {})
        )
        
        mock_storage.create_user = self._create_async_method(
            "create_user", self._mock_create_user, **overrides.get("create_user", {})
        )
        
        mock_storage.user_exists = self._create_async_method(
            "user_exists", self._mock_user_exists, **overrides.get("user_exists", {})
        )
        
        mock_storage.create_session = self._create_async_method(
            "create_session", self._mock_create_session, **overrides.get("create_session", {})
        )
        
        mock_storage.get_session = self._create_async_method(
            "get_session", self._mock_get_session, **overrides.get("get_session", {})
        )
        
        mock_storage.add_message = self._create_async_method(
            "add_message", self._mock_add_message, **overrides.get("add_message", {})
        )
        
        mock_storage.get_all_messages = self._create_async_method(
            "get_all_messages", self._mock_get_messages, **overrides.get("get_all_messages", {})
        )
        
        # Apply any additional overrides
        for method_name, method_config in overrides.items():
            if hasattr(mock_storage, method_name) and "return_value" in method_config:
                getattr(mock_storage, method_name).return_value = method_config["return_value"]
        
        return mock_storage
    
    # === Backend Protocol Mocks ===
    
    def create_backend_mock(self, **overrides) -> Mock:
        """Create a sophisticated backend protocol mock."""
        mock_backend = AsyncMock(spec=BackendProtocol)
        
        # Track internal file system state
        self._state_data["files"] = {}
        
        mock_backend.initialize = self._create_async_method(
            "initialize", lambda: True, **overrides.get("initialize", {})
        )
        
        mock_backend.read = self._create_async_method(
            "read", self._mock_read_file, **overrides.get("read", {})
        )
        
        mock_backend.write = self._create_async_method(
            "write", self._mock_write_file, **overrides.get("write", {})
        )
        
        mock_backend.exists = self._create_async_method(
            "exists", self._mock_file_exists, **overrides.get("exists", {})
        )
        
        mock_backend.delete = self._create_async_method(
            "delete", self._mock_delete_file, **overrides.get("delete", {})
        )
        
        mock_backend.list_keys = self._create_async_method(
            "list_keys", self._mock_list_keys, **overrides.get("list_keys", {})
        )
        
        return mock_backend
    
    # === Search Protocol Mocks ===
    
    def create_search_mock(self, **overrides) -> Mock:
        """Create a sophisticated search protocol mock."""
        mock_search = AsyncMock(spec=SearchProtocol)
        
        # Track search index state
        self._state_data["search_index"] = {}
        
        mock_search.search = self._create_async_method(
            "search", self._mock_search, **overrides.get("search", {})
        )
        
        mock_search.build_search_index = self._create_async_method(
            "build_search_index", self._mock_build_index, **overrides.get("build_search_index", {})
        )
        
        return mock_search
    
    # === Vector Store Protocol Mocks ===
    
    def create_vector_store_mock(self, **overrides) -> Mock:
        """Create a sophisticated vector store protocol mock."""
        mock_vector = AsyncMock(spec=VectorStoreProtocol)
        
        # Track vector state
        self._state_data["vectors"] = {}
        
        mock_vector.store_embeddings = self._create_async_method(
            "store_embeddings", lambda session_id, embeddings: True,
            **overrides.get("store_embeddings", {})
        )
        
        mock_vector.similarity_search = self._create_async_method(
            "similarity_search", self._mock_similarity_search,
            **overrides.get("similarity_search", {})
        )
        
        return mock_vector
    
    # === Error Injection Utilities ===
    
    def create_failing_mock(
        self,
        protocol_class: Type,
        failure_rate: float = 0.5,
        error_types: Optional[List[Exception]] = None
    ) -> Mock:
        """Create a mock that fails randomly with specified rate."""
        error_types = error_types or [
            ConnectionError("Mock network error"),
            FileNotFoundError("Mock file not found"),
            PermissionError("Mock permission denied"),
            TimeoutError("Mock operation timeout")
        ]
        
        mock_obj = AsyncMock(spec=protocol_class)
        
        # Override all async methods to potentially fail
        for attr_name in dir(protocol_class):
            if not attr_name.startswith('_'):
                attr = getattr(protocol_class, attr_name)
                if asyncio.iscoroutinefunction(attr):
                    setattr(
                        mock_obj,
                        attr_name,
                        self._create_failing_method(attr_name, failure_rate, error_types)
                    )
        
        return mock_obj
    
    def create_slow_mock(
        self,
        protocol_class: Type,
        min_delay: float = 1.0,
        max_delay: float = 5.0
    ) -> Mock:
        """Create a mock that simulates slow operations."""
        mock_obj = AsyncMock(spec=protocol_class)
        
        # Override all async methods to be slow
        for attr_name in dir(protocol_class):
            if not attr_name.startswith('_'):
                attr = getattr(protocol_class, attr_name)
                if asyncio.iscoroutinefunction(attr):
                    setattr(
                        mock_obj,
                        attr_name,
                        self._create_slow_method(attr_name, min_delay, max_delay)
                    )
        
        return mock_obj
    
    # === State Management ===
    
    def get_mock_state(self, category: str) -> Dict[str, Any]:
        """Get current state of mocked data."""
        return self._state_data.get(category, {})
    
    def reset_mock_state(self):
        """Reset all mock state data."""
        self._state_data.clear()
        self._call_counts.clear()
        self._error_counters.clear()
    
    def get_call_count(self, method_name: str) -> int:
        """Get call count for a specific method."""
        return self._call_counts.get(method_name, 0)
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get comprehensive call statistics."""
        return {
            "call_counts": self._call_counts.copy(),
            "error_counts": self._error_counters.copy(),
            "total_calls": sum(self._call_counts.values()),
            "total_errors": sum(self._error_counters.values())
        }
    
    # === Context Managers ===
    
    @asynccontextmanager
    async def temporary_failure(self, mock_obj: Mock, method_name: str, error: Exception):
        """Temporarily make a method fail, then restore it."""
        original_method = getattr(mock_obj, method_name)
        
        async def failing_method(*args, **kwargs):
            raise error
        
        setattr(mock_obj, method_name, failing_method)
        try:
            yield
        finally:
            setattr(mock_obj, method_name, original_method)
    
    @contextmanager
    def mock_time_freeze(self, frozen_time: datetime):
        """Freeze time for consistent timestamp testing."""
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            yield
    
    @contextmanager
    def mock_random_seed(self, seed: int = 42):
        """Set random seed for reproducible mock behavior."""
        original_state = random.getstate()
        random.seed(seed)
        try:
            yield
        finally:
            random.setstate(original_state)
    
    # === Private Helper Methods ===
    
    def _create_async_method(self, method_name: str, func: Callable, **config) -> AsyncMock:
        """Create an async method with configured behavior."""
        async def method_wrapper(*args, **kwargs):
            # Track call count
            self._call_counts[method_name] = self._call_counts.get(method_name, 0) + 1
            
            # Log calls if enabled
            if self.config.call_logging:
                print(f"Mock call: {method_name}({args}, {kwargs})")
            
            # Simulate latency
            if self.config.latency_max_ms > 0:
                delay = random.randint(self.config.latency_min_ms, self.config.latency_max_ms) / 1000
                await asyncio.sleep(delay)
            
            # Simulate errors based on success rate
            if random.random() > self.config.success_rate:
                error_type = random.choice(self.config.error_patterns)
                self._error_counters[method_name] = self._error_counters.get(method_name, 0) + 1
                raise self._create_error(error_type)
            
            # Execute the actual function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return AsyncMock(side_effect=method_wrapper)
    
    def _create_failing_method(self, method_name: str, failure_rate: float, error_types: List[Exception]) -> AsyncMock:
        """Create a method that fails with specified probability."""
        async def failing_method(*args, **kwargs):
            if random.random() < failure_rate:
                raise random.choice(error_types)
            return True  # Default success return
        
        return AsyncMock(side_effect=failing_method)
    
    def _create_slow_method(self, method_name: str, min_delay: float, max_delay: float) -> AsyncMock:
        """Create a method that simulates slow operations."""
        async def slow_method(*args, **kwargs):
            delay = random.uniform(min_delay, max_delay)
            await asyncio.sleep(delay)
            return True  # Default success return
        
        return AsyncMock(side_effect=slow_method)
    
    def _create_error(self, error_type: str) -> Exception:
        """Create an appropriate error based on type."""
        error_map = {
            "network": ConnectionError("Simulated network error"),
            "disk": OSError("Simulated disk error"),
            "permission": PermissionError("Simulated permission error"),
            "timeout": TimeoutError("Simulated timeout error"),
            "file_not_found": FileNotFoundError("Simulated file not found"),
            "memory": MemoryError("Simulated memory error")
        }
        return error_map.get(error_type, Exception(f"Simulated {error_type} error"))
    
    # === Mock Implementation Methods ===
    
    async def _mock_create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool:
        """Mock user creation with state tracking."""
        if user_id in self._state_data["users"]:
            return False  # User already exists
        
        self._state_data["users"][user_id] = {
            "profile": profile or {},
            "created_at": datetime.now().isoformat()
        }
        return True
    
    async def _mock_user_exists(self, user_id: str) -> bool:
        """Mock user existence check."""
        return user_id in self._state_data["users"]
    
    async def _mock_create_session(self, user_id: str, title: Optional[str] = None) -> str:
        """Mock session creation with state tracking."""
        if user_id not in self._state_data["users"]:
            raise ValueError(f"User {user_id} does not exist")
        
        session_id = f"session_{len(self._state_data['sessions'])}"
        self._state_data["sessions"][session_id] = {
            "user_id": user_id,
            "title": title or "Mock Session",
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }
        return session_id
    
    async def _mock_get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]:
        """Mock session retrieval."""
        session_data = self._state_data["sessions"].get(session_id)
        if not session_data or session_data["user_id"] != user_id:
            return None
        
        return FFSessionDTO(
            session_id=session_id,
            user_id=user_id,
            title=session_data["title"],
            created_at=session_data["created_at"],
            updated_at=session_data["created_at"]
        )
    
    async def _mock_add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool:
        """Mock message addition with state tracking."""
        session_key = f"{user_id}:{session_id}"
        if session_key not in self._state_data["messages"]:
            self._state_data["messages"][session_key] = []
        
        self._state_data["messages"][session_key].append(message)
        
        # Update session message count
        if session_id in self._state_data["sessions"]:
            self._state_data["sessions"][session_id]["message_count"] += 1
        
        return True
    
    async def _mock_get_messages(self, user_id: str, session_id: str) -> List[FFMessageDTO]:
        """Mock message retrieval."""
        session_key = f"{user_id}:{session_id}"
        return self._state_data["messages"].get(session_key, [])
    
    async def _mock_read_file(self, key: str) -> Optional[bytes]:
        """Mock file reading with state tracking."""
        return self._state_data["files"].get(key)
    
    async def _mock_write_file(self, key: str, data: bytes) -> bool:
        """Mock file writing with state tracking."""
        self._state_data["files"][key] = data
        return True
    
    async def _mock_file_exists(self, key: str) -> bool:
        """Mock file existence check."""
        return key in self._state_data["files"]
    
    async def _mock_delete_file(self, key: str) -> bool:
        """Mock file deletion with state tracking."""
        if key in self._state_data["files"]:
            del self._state_data["files"][key]
            return True
        return False
    
    async def _mock_list_keys(self, prefix: str = "", pattern: Optional[str] = None) -> List[str]:
        """Mock key listing with filtering."""
        keys = list(self._state_data["files"].keys())
        
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        
        if pattern:
            import re
            regex_pattern = pattern.replace("*", ".*")
            keys = [k for k in keys if re.match(regex_pattern, k)]
        
        return keys
    
    async def _mock_search(self, query: str, user_id: Optional[str] = None, 
                          session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Mock search with realistic results."""
        # Simple mock search - return some relevant results
        results = []
        
        for i in range(random.randint(0, 5)):
            results.append({
                "id": f"result_{i}",
                "type": "message",
                "content": f"Mock search result {i} for query: {query}",
                "relevance": random.uniform(0.5, 1.0),
                "user_id": user_id or "mock_user",
                "session_id": "mock_session"
            })
        
        return results
    
    async def _mock_build_index(self, user_id: str) -> Dict[str, Any]:
        """Mock search index building."""
        return {
            "indexed_messages": random.randint(10, 100),
            "indexed_documents": random.randint(0, 20),
            "build_time_seconds": random.uniform(1.0, 5.0)
        }
    
    async def _mock_similarity_search(self, query_embedding: List[float], 
                                    session_id: Optional[str] = None,
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """Mock vector similarity search."""
        results = []
        
        for i in range(min(limit, random.randint(0, 5))):
            results.append({
                "id": f"vector_result_{i}",
                "similarity": random.uniform(0.6, 0.95),
                "content": f"Mock vector result {i}",
                "metadata": {"type": "message"}
            })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)


# === Specialized Mock Builders ===

class IntegrationMockBuilder:
    """Builder for creating integrated mocks for complex testing scenarios."""
    
    def __init__(self):
        self.factory = AdvancedMockFactory()
        self.mocks = {}
    
    def with_storage(self, **overrides) -> 'IntegrationMockBuilder':
        """Add storage mock to the integration."""
        self.mocks["storage"] = self.factory.create_storage_mock(**overrides)
        return self
    
    def with_backend(self, **overrides) -> 'IntegrationMockBuilder':
        """Add backend mock to the integration."""
        self.mocks["backend"] = self.factory.create_backend_mock(**overrides)
        return self
    
    def with_search(self, **overrides) -> 'IntegrationMockBuilder':
        """Add search mock to the integration."""
        self.mocks["search"] = self.factory.create_search_mock(**overrides)
        return self
    
    def with_vector_store(self, **overrides) -> 'IntegrationMockBuilder':
        """Add vector store mock to the integration."""
        self.mocks["vector_store"] = self.factory.create_vector_store_mock(**overrides)
        return self
    
    def with_failing_component(self, component: str, failure_rate: float = 0.5) -> 'IntegrationMockBuilder':
        """Add a component that fails randomly."""
        protocol_map = {
            "storage": StorageProtocol,
            "backend": BackendProtocol,
            "search": SearchProtocol,
            "vector_store": VectorStoreProtocol
        }
        
        if component in protocol_map:
            self.mocks[component] = self.factory.create_failing_mock(
                protocol_map[component], failure_rate
            )
        
        return self
    
    def build(self) -> Dict[str, Mock]:
        """Build and return the integration mocks."""
        return self.mocks.copy()


# === Convenience Instances ===

# Default factory for general use
mock_factory = AdvancedMockFactory()

# Factory for reliable testing (high success rate, low latency)
reliable_mock_factory = AdvancedMockFactory(MockBehaviorConfig(
    success_rate=0.99,
    latency_min_ms=1,
    latency_max_ms=10,
    call_logging=False
))

# Factory for stress testing (lower success rate, higher latency)
stress_mock_factory = AdvancedMockFactory(MockBehaviorConfig(
    success_rate=0.8,
    latency_min_ms=50,
    latency_max_ms=500,
    error_patterns=["network", "timeout", "disk", "memory"]
))

# Builder for integration scenarios
integration_builder = IntegrationMockBuilder