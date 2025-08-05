# Phase 1d: Protocols and Configuration Updates

## Overview
This phase updates the protocol definitions and configuration systems to support the new v2 data models while maintaining compatibility with existing v1 interfaces where possible.

## Scope
- Update storage protocols for v2 models
- Enhance configuration system for new features
- Create adapter protocols for v1→v2 compatibility
- Define new search and query protocols

## 1. Enhanced Storage Protocols

### 1.1 Base Storage Protocol V2
```python
from typing import Protocol, TypeVar, Generic, Optional, List, Dict, Any, AsyncIterator
from abc import abstractmethod
import asyncio
from datetime import datetime

T = TypeVar('T')
K = TypeVar('K')

class StorageProtocolV2(Protocol, Generic[T, K]):
    """Enhanced storage protocol supporting advanced operations"""
    
    @abstractmethod
    async def create(self, item: T) -> K:
        """Create a new item"""
        pass
    
    @abstractmethod
    async def read(self, key: K) -> Optional[T]:
        """Read item by key"""
        pass
    
    @abstractmethod
    async def update(self, key: K, item: T) -> bool:
        """Update existing item"""
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """Delete item by key"""
        pass
    
    @abstractmethod
    async def exists(self, key: K) -> bool:
        """Check if item exists"""
        pass
    
    @abstractmethod
    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> List[T]:
        """List items with pagination and filtering"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count items matching filters"""
        pass
    
    @abstractmethod
    async def batch_create(self, items: List[T]) -> List[K]:
        """Create multiple items efficiently"""
        pass
    
    @abstractmethod
    async def batch_read(self, keys: List[K]) -> List[Optional[T]]:
        """Read multiple items efficiently"""
        pass
    
    @abstractmethod
    async def batch_update(self, updates: List[tuple[K, T]]) -> List[bool]:
        """Update multiple items efficiently"""
        pass
    
    @abstractmethod
    async def batch_delete(self, keys: List[K]) -> List[bool]:
        """Delete multiple items efficiently"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> AsyncIterator[List[T]]:
        """Stream items in batches"""
        pass
```

### 1.2 Searchable Storage Protocol
```python
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

class SearchableStorageProtocol(StorageProtocolV2[T, K], Protocol):
    """Storage protocol with search capabilities"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_fields: Optional[List[str]] = None
    ) -> List[Tuple[T, float]]:
        """Full-text search with relevance scores"""
        pass
    
    @abstractmethod
    async def vector_search(
        self,
        embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[T, float]]:
        """Vector similarity search"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5  # Weight between text and vector search
    ) -> List[Tuple[T, float]]:
        """Hybrid text + vector search"""
        pass
    
    @abstractmethod
    async def create_index(
        self,
        field: str,
        index_type: str = "btree"
    ) -> bool:
        """Create index on field"""
        pass
    
    @abstractmethod
    async def drop_index(self, field: str) -> bool:
        """Drop index on field"""
        pass
```

### 1.3 Versioned Storage Protocol
```python
class VersionedStorageProtocol(StorageProtocolV2[T, K], Protocol):
    """Storage protocol with versioning support"""
    
    @abstractmethod
    async def get_version(self, key: K, version: int) -> Optional[T]:
        """Get specific version of item"""
        pass
    
    @abstractmethod
    async def get_versions(
        self,
        key: K,
        limit: int = 10
    ) -> List[Tuple[T, int, datetime]]:
        """Get version history"""
        pass
    
    @abstractmethod
    async def restore_version(self, key: K, version: int) -> bool:
        """Restore item to specific version"""
        pass
    
    @abstractmethod
    async def purge_versions(
        self,
        key: K,
        keep_last: int = 1
    ) -> int:
        """Purge old versions, return count deleted"""
        pass
```

## 2. Manager Protocols

### 2.1 Enhanced Message Manager Protocol
```python
from ff_manager_protocol import ManagerProtocol
from ff_protocols import MessageV2DTO, MessageContentType

class MessageManagerV2Protocol(ManagerProtocol):
    """Enhanced message manager protocol"""
    
    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        content: Dict[str, Any],
        content_type: MessageContentType,
        role: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageV2DTO:
        """Add message with flexible content"""
        pass
    
    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        content_types: Optional[List[MessageContentType]] = None
    ) -> List[MessageV2DTO]:
        """Get messages with filtering"""
        pass
    
    @abstractmethod
    async def search_messages(
        self,
        session_id: str,
        query: str,
        content_types: Optional[List[MessageContentType]] = None,
        limit: int = 10
    ) -> List[Tuple[MessageV2DTO, float]]:
        """Search messages with relevance"""
        pass
    
    @abstractmethod
    async def get_message_thread(
        self,
        message_id: str,
        depth: int = 10
    ) -> List[MessageV2DTO]:
        """Get message thread/conversation"""
        pass
    
    @abstractmethod
    async def update_message_metadata(
        self,
        message_id: str,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """Update message metadata"""
        pass
```

### 2.2 Memory Manager Protocol
```python
from ff_protocols import MemoryEntryDTO, MemoryType

class MemoryManagerProtocol(ManagerProtocol):
    """Memory management protocol"""
    
    @abstractmethod
    async def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> MemoryEntryDTO:
        """Store new memory"""
        pass
    
    @abstractmethod
    async def recall_memories(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Tuple[MemoryEntryDTO, float]]:
        """Recall relevant memories"""
        pass
    
    @abstractmethod
    async def reinforce_memory(
        self,
        memory_id: str,
        reinforcement: float = 0.1
    ) -> bool:
        """Reinforce memory importance"""
        pass
    
    @abstractmethod
    async def decay_memories(
        self,
        user_id: str,
        decay_rate: float = 0.01
    ) -> int:
        """Apply decay to memories, return count affected"""
        pass
    
    @abstractmethod
    async def consolidate_memories(
        self,
        user_id: str,
        threshold: float = 0.8
    ) -> List[MemoryEntryDTO]:
        """Consolidate similar memories"""
        pass
```

### 2.3 Agent Manager Protocol
```python
from ff_protocols import AgentConfigurationDTO, AgentMessage

class AgentManagerProtocol(ManagerProtocol):
    """Agent management protocol"""
    
    @abstractmethod
    async def register_agent(
        self,
        config: AgentConfigurationDTO
    ) -> str:
        """Register new agent"""
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[AgentConfigurationDTO]:
        """Get agent configuration"""
        pass
    
    @abstractmethod
    async def update_agent(
        self,
        agent_id: str,
        config: AgentConfigurationDTO
    ) -> bool:
        """Update agent configuration"""
        pass
    
    @abstractmethod
    async def route_message(
        self,
        message: MessageV2DTO,
        available_agents: List[str]
    ) -> Optional[str]:
        """Route message to appropriate agent"""
        pass
    
    @abstractmethod
    async def get_agent_history(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get agent interaction history"""
        pass
```

## 3. Configuration System

### 3.1 Enhanced Configuration Schema
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class StorageConfigV2:
    """Enhanced storage configuration"""
    base_path: Path
    storage_type: str = "flatfile"  # flatfile, sqlite, hybrid
    
    # Performance settings
    cache_enabled: bool = True
    cache_size_mb: int = 100
    batch_size: int = 1000
    compression_enabled: bool = True
    compression_level: int = 6
    
    # Indexing settings
    index_enabled: bool = True
    index_fields: List[str] = field(default_factory=list)
    vector_index_enabled: bool = False
    vector_dimensions: int = 768
    
    # Versioning settings
    versioning_enabled: bool = False
    max_versions: int = 10
    version_cleanup_days: int = 30
    
    # Search settings
    search_enabled: bool = True
    search_min_length: int = 2
    search_max_results: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "base_path": str(self.base_path),
            "storage_type": self.storage_type,
            "cache_enabled": self.cache_enabled,
            "cache_size_mb": self.cache_size_mb,
            "batch_size": self.batch_size,
            "compression_enabled": self.compression_enabled,
            "compression_level": self.compression_level,
            "index_enabled": self.index_enabled,
            "index_fields": self.index_fields,
            "vector_index_enabled": self.vector_index_enabled,
            "vector_dimensions": self.vector_dimensions,
            "versioning_enabled": self.versioning_enabled,
            "max_versions": self.max_versions,
            "version_cleanup_days": self.version_cleanup_days,
            "search_enabled": self.search_enabled,
            "search_min_length": self.search_min_length,
            "search_max_results": self.search_max_results
        }

@dataclass
class MemoryConfigV2:
    """Memory system configuration"""
    enabled: bool = True
    
    # Storage settings
    storage_backend: str = "flatfile"  # flatfile, redis, hybrid
    persistence_enabled: bool = True
    
    # Memory settings
    max_memories_per_user: int = 10000
    consolidation_threshold: float = 0.8
    importance_threshold: float = 0.1
    
    # Decay settings
    decay_enabled: bool = True
    decay_rate: float = 0.01
    decay_interval_hours: int = 24
    
    # Embedding settings
    embedding_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_size: int = 1000

@dataclass
class MultimodalConfigV2:
    """Multimodal content configuration"""
    enabled: bool = True
    
    # Storage settings
    storage_path: Path = field(default_factory=lambda: Path("multimodal"))
    max_file_size_mb: int = 100
    allowed_image_types: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "gif", "webp"]
    )
    allowed_audio_types: List[str] = field(
        default_factory=lambda: ["mp3", "wav", "ogg", "m4a"]
    )
    allowed_video_types: List[str] = field(
        default_factory=lambda: ["mp4", "avi", "mov", "webm"]
    )
    allowed_document_types: List[str] = field(
        default_factory=lambda: ["pdf", "doc", "docx", "txt", "md"]
    )
    
    # Processing settings
    generate_thumbnails: bool = True
    thumbnail_size: tuple[int, int] = (256, 256)
    extract_text: bool = True
    generate_embeddings: bool = True
    
    # Deduplication settings
    deduplication_enabled: bool = True
    deduplication_threshold: float = 0.95

@dataclass
class TraceConfigV2:
    """Trace system configuration"""
    enabled: bool = True
    
    # Storage settings
    storage_backend: str = "flatfile"
    retention_days: int = 30
    
    # Trace settings
    trace_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    include_payloads: bool = True
    include_embeddings: bool = False
    max_payload_size: int = 10000
    
    # Performance settings
    async_writing: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 5

@dataclass
class ChatDatabaseConfigV2:
    """Main configuration for chat database v2"""
    storage: StorageConfigV2
    memory: MemoryConfigV2 = field(default_factory=MemoryConfigV2)
    multimodal: MultimodalConfigV2 = field(default_factory=MultimodalConfigV2)
    trace: TraceConfigV2 = field(default_factory=TraceConfigV2)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "multi_agent": True,
        "memory_system": True,
        "multimodal": True,
        "tracing": True,
        "vector_search": True,
        "versioning": False
    })
    
    # Compatibility settings
    v1_compatibility: bool = True
    auto_migrate: bool = True
    
    @classmethod
    def from_v1_config(cls, v1_config: Dict[str, Any]) -> "ChatDatabaseConfigV2":
        """Create v2 config from v1 config"""
        storage_config = StorageConfigV2(
            base_path=Path(v1_config.get("base_path", "data")),
            storage_type="flatfile"
        )
        
        return cls(storage=storage_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "storage": self.storage.to_dict(),
            "memory": dataclasses.asdict(self.memory),
            "multimodal": dataclasses.asdict(self.multimodal),
            "trace": dataclasses.asdict(self.trace),
            "features": self.features,
            "v1_compatibility": self.v1_compatibility,
            "auto_migrate": self.auto_migrate
        }
```

### 3.2 Configuration Manager
```python
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

class ConfigurationManagerV2:
    """Manages v2 configuration with validation and migration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self._config: Optional[ChatDatabaseConfigV2] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Check if v1 config
            if "storage" not in data and "base_path" in data:
                self._config = ChatDatabaseConfigV2.from_v1_config(data)
                if self._config.auto_migrate:
                    self.save_config()
            else:
                self._config = self._create_from_dict(data)
        else:
            # Create default config
            self._config = ChatDatabaseConfigV2(
                storage=StorageConfigV2(base_path=Path("data"))
            )
            self.save_config()
    
    def _create_from_dict(self, data: Dict[str, Any]) -> ChatDatabaseConfigV2:
        """Create config from dictionary"""
        storage_data = data.get("storage", {})
        storage_config = StorageConfigV2(
            base_path=Path(storage_data.get("base_path", "data")),
            **{k: v for k, v in storage_data.items() if k != "base_path"}
        )
        
        memory_config = MemoryConfigV2(**data.get("memory", {}))
        multimodal_config = MultimodalConfigV2(**data.get("multimodal", {}))
        trace_config = TraceConfigV2(**data.get("trace", {}))
        
        return ChatDatabaseConfigV2(
            storage=storage_config,
            memory=memory_config,
            multimodal=multimodal_config,
            trace=trace_config,
            features=data.get("features", {}),
            v1_compatibility=data.get("v1_compatibility", True),
            auto_migrate=data.get("auto_migrate", True)
        )
    
    def save_config(self) -> None:
        """Save configuration to file"""
        if self._config:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix == '.yaml':
                    yaml.dump(self._config.to_dict(), f, default_flow_style=False)
                else:
                    json.dump(self._config.to_dict(), f, indent=2)
    
    @property
    def config(self) -> ChatDatabaseConfigV2:
        """Get current configuration"""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config
    
    def get_feature(self, feature: str) -> bool:
        """Check if feature is enabled"""
        return self.config.features.get(feature, False)
    
    def update_feature(self, feature: str, enabled: bool) -> None:
        """Update feature flag"""
        self.config.features[feature] = enabled
        self.save_config()
```

## 4. Adapter Protocols

### 4.1 V1 to V2 Storage Adapter
```python
from typing import Optional, List, Dict, Any
from ff_storage_protocol import StorageProtocol
from ff_protocols import MessageDTO, SessionDTO

class V1ToV2StorageAdapter(StorageProtocolV2[T, K]):
    """Adapts v1 storage to v2 protocol"""
    
    def __init__(self, v1_storage: StorageProtocol):
        self.v1_storage = v1_storage
        self._converters = {
            MessageDTO: self._convert_message_v1_to_v2,
            SessionDTO: self._convert_session_v1_to_v2
        }
    
    async def create(self, item: T) -> K:
        """Create item using v1 storage"""
        v1_item = self._convert_to_v1(item)
        return await self.v1_storage.create(v1_item)
    
    async def read(self, key: K) -> Optional[T]:
        """Read item using v1 storage"""
        v1_item = await self.v1_storage.read(key)
        if v1_item:
            return self._convert_to_v2(v1_item)
        return None
    
    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> List[T]:
        """List items with v1 storage limitations"""
        # V1 doesn't support advanced filtering
        all_items = await self.v1_storage.list()
        
        # Apply basic filtering in memory
        if filters:
            filtered = []
            for item in all_items:
                match = True
                for key, value in filters.items():
                    if hasattr(item, key) and getattr(item, key) != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            all_items = filtered
        
        # Convert to v2
        v2_items = [self._convert_to_v2(item) for item in all_items]
        
        # Apply sorting
        if sort_by:
            v2_items.sort(
                key=lambda x: getattr(x, sort_by, None),
                reverse=sort_desc
            )
        
        # Apply pagination
        return v2_items[offset:offset + limit]
    
    def _convert_to_v1(self, item: T) -> Any:
        """Convert v2 item to v1"""
        # Implementation depends on item type
        raise NotImplementedError
    
    def _convert_to_v2(self, item: Any) -> T:
        """Convert v1 item to v2"""
        # Implementation depends on item type
        item_type = type(item)
        if item_type in self._converters:
            return self._converters[item_type](item)
        raise ValueError(f"No converter for type {item_type}")
    
    def _convert_message_v1_to_v2(self, v1_msg: MessageDTO) -> MessageV2DTO:
        """Convert v1 message to v2"""
        from ff_protocols import MessageV2DTO, MessageContentType
        
        return MessageV2DTO(
            id=v1_msg.id,
            session_id=v1_msg.session_id,
            role=v1_msg.role,
            content={"text": v1_msg.content},
            content_type=MessageContentType.TEXT,
            timestamp=v1_msg.timestamp,
            metadata=v1_msg.metadata or {},
            parent_id=None,
            thread_id=None,
            version=1,
            embeddings=None
        )
    
    def _convert_session_v1_to_v2(self, v1_session: SessionDTO) -> SessionV2DTO:
        """Convert v1 session to v2"""
        from ff_protocols import SessionV2DTO, SessionType
        
        return SessionV2DTO(
            id=v1_session.id,
            user_id=v1_session.user_id,
            title=v1_session.metadata.get("title", "Untitled Session")
            if v1_session.metadata else "Untitled Session",
            session_type=SessionType.CHAT,
            created_at=v1_session.created_at,
            updated_at=v1_session.updated_at,
            metadata=v1_session.metadata or {},
            summary="",
            tags=[],
            participants=[v1_session.user_id],
            config={},
            status="active"
        )
```

## 5. Testing

### 5.1 Protocol Tests
```python
import pytest
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

class MockStorageV2(StorageProtocolV2[Dict[str, Any], str]):
    """Mock implementation for testing"""
    
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.id_counter = 0
    
    async def create(self, item: Dict[str, Any]) -> str:
        self.id_counter += 1
        key = f"item_{self.id_counter}"
        self.store[key] = item.copy()
        return key
    
    async def read(self, key: str) -> Optional[Dict[str, Any]]:
        return self.store.get(key)
    
    async def update(self, key: str, item: Dict[str, Any]) -> bool:
        if key in self.store:
            self.store[key] = item.copy()
            return True
        return False
    
    async def delete(self, key: str) -> bool:
        if key in self.store:
            del self.store[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self.store
    
    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> List[Dict[str, Any]]:
        items = list(self.store.values())
        
        # Apply filters
        if filters:
            filtered = []
            for item in items:
                match = True
                for key, value in filters.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            items = filtered
        
        # Apply sorting
        if sort_by:
            items.sort(
                key=lambda x: x.get(sort_by),
                reverse=sort_desc
            )
        
        # Apply pagination
        return items[offset:offset + limit]
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        items = await self.list(filters=filters)
        return len(items)
    
    async def batch_create(self, items: List[Dict[str, Any]]) -> List[str]:
        keys = []
        for item in items:
            key = await self.create(item)
            keys.append(key)
        return keys
    
    async def batch_read(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        return [await self.read(key) for key in keys]
    
    async def batch_update(
        self,
        updates: List[tuple[str, Dict[str, Any]]]
    ) -> List[bool]:
        results = []
        for key, item in updates:
            result = await self.update(key, item)
            results.append(result)
        return results
    
    async def batch_delete(self, keys: List[str]) -> List[bool]:
        return [await self.delete(key) for key in keys]
    
    async def stream(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        items = await self.list(filters=filters)
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]


@pytest.mark.asyncio
async def test_storage_protocol_v2():
    """Test enhanced storage protocol"""
    storage = MockStorageV2()
    
    # Test create
    item1 = {"name": "test1", "value": 100}
    key1 = await storage.create(item1)
    assert key1 == "item_1"
    
    # Test read
    retrieved = await storage.read(key1)
    assert retrieved == item1
    
    # Test update
    item1["value"] = 200
    success = await storage.update(key1, item1)
    assert success is True
    retrieved = await storage.read(key1)
    assert retrieved["value"] == 200
    
    # Test batch operations
    items = [
        {"name": "test2", "value": 300},
        {"name": "test3", "value": 400}
    ]
    keys = await storage.batch_create(items)
    assert len(keys) == 2
    
    # Test list with filters
    filtered = await storage.list(filters={"value": 300})
    assert len(filtered) == 1
    assert filtered[0]["name"] == "test2"
    
    # Test sorting
    sorted_items = await storage.list(sort_by="value", sort_desc=True)
    assert sorted_items[0]["value"] == 400
    assert sorted_items[1]["value"] == 300
    assert sorted_items[2]["value"] == 200
    
    # Test streaming
    batch_count = 0
    async for batch in storage.stream(batch_size=2):
        batch_count += 1
        assert len(batch) <= 2
    assert batch_count == 2


@pytest.mark.asyncio
async def test_configuration_manager():
    """Test configuration management"""
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        # Test default config creation
        manager = ConfigurationManagerV2(config_path)
        assert manager.config.storage.base_path == Path("data")
        assert manager.config.v1_compatibility is True
        
        # Test feature management
        assert manager.get_feature("multi_agent") is True
        manager.update_feature("multi_agent", False)
        assert manager.get_feature("multi_agent") is False
        
        # Test config persistence
        manager2 = ConfigurationManagerV2(config_path)
        assert manager2.get_feature("multi_agent") is False


@pytest.mark.asyncio
async def test_v1_to_v2_adapter():
    """Test v1 to v2 adapter"""
    # This would require actual v1 storage implementation
    # Placeholder for now
    pass
```

### 5.2 Integration Tests
```python
@pytest.mark.asyncio
async def test_protocol_compliance():
    """Test that implementations comply with protocols"""
    from typing import get_type_hints
    
    # Check MockStorageV2 implements StorageProtocolV2
    storage = MockStorageV2()
    protocol_methods = [
        "create", "read", "update", "delete", "exists",
        "list", "count", "batch_create", "batch_read",
        "batch_update", "batch_delete", "stream"
    ]
    
    for method in protocol_methods:
        assert hasattr(storage, method)
        assert callable(getattr(storage, method))


def test_configuration_schema():
    """Test configuration schema completeness"""
    config = ChatDatabaseConfigV2(
        storage=StorageConfigV2(base_path=Path("data"))
    )
    
    # Check all subsystems have config
    assert hasattr(config, "storage")
    assert hasattr(config, "memory")
    assert hasattr(config, "multimodal")
    assert hasattr(config, "trace")
    
    # Check serialization
    config_dict = config.to_dict()
    assert "storage" in config_dict
    assert "features" in config_dict
    
    # Check v1 compatibility
    v1_config = {"base_path": "old_data"}
    v2_config = ChatDatabaseConfigV2.from_v1_config(v1_config)
    assert v2_config.storage.base_path == Path("old_data")
```

## Implementation Checklist

### Protocol Updates
- [ ] Create `ff_storage_protocol_v2.py` with enhanced protocols
- [ ] Create `ff_manager_protocols_v2.py` with manager protocols
- [ ] Create `ff_adapter_protocols.py` with compatibility adapters
- [ ] Update existing protocol imports to use v2 where appropriate

### Configuration System
- [ ] Create `ff_config_v2.py` with new configuration classes
- [ ] Create `ff_config_manager_v2.py` with configuration manager
- [ ] Create migration utilities for v1 → v2 configs
- [ ] Update dependency injection to use v2 configs

### Testing
- [ ] Create `tests/test_protocols_v2.py`
- [ ] Create `tests/test_configuration_v2.py`
- [ ] Create `tests/test_adapters.py`
- [ ] Update existing tests to work with v2 protocols

### Documentation
- [ ] Document new protocol methods and parameters
- [ ] Document configuration schema and options
- [ ] Create migration guide for v1 → v2
- [ ] Update API documentation

## Next Steps
After completing Phase 1d:
1. Review all Phase 1 components for consistency
2. Begin Phase 2: Core Managers Refactoring
3. Start with MessageManagerV2 implementation
4. Ensure v1 compatibility throughout

## Notes
- All protocols maintain backward compatibility through adapters
- Configuration system supports automatic v1 → v2 migration
- Feature flags allow gradual rollout of new capabilities
- Extensive testing ensures protocol compliance