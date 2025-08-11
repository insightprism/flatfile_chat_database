# Phase 1: Multi-Layered Memory System Implementation

## 🎯 Phase Overview

Implement a sophisticated multi-tiered memory system that extends your existing context management to support PrismMind's advanced memory capabilities. This system will provide automatic retention policies, memory compression, and intelligent archival while maintaining full backward compatibility with your current `FFContextManager`.

## 📋 Requirements Analysis

### **Current State Assessment**
Your system already has:
- ✅ `FFContextManager` - Basic situational context management
- ✅ `FFSituationalContextDTO` - Context data model
- ✅ Configuration-driven architecture with DTOs
- ✅ Async file operations with atomic writes

### **Memory Layer Requirements**
Based on PrismMind's use cases, implement 5 memory layers:
1. **Immediate Memory** (2-hour retention) - Current conversation context
2. **Short-term Memory** (24-hour retention) - Recent interactions
3. **Medium-term Memory** (1-week retention) - Weekly patterns and important topics
4. **Long-term Memory** (1-year retention) - Persistent patterns and significant conversations
5. **Permanent Memory** (never expires) - Critical insights and user preferences

## 🏗️ Architecture Design

### **Memory Layer Hierarchy**
```
users/{user_id}/memory_layers/
├── immediate.jsonl           # Last 2 hours of context
├── short_term.jsonl          # Last 24 hours of context  
├── medium_term.jsonl         # Last week of context
├── long_term.jsonl          # Last year of context
├── permanent.jsonl          # Never expires
├── compression_history.json # Archival and compression logs
└── layer_metadata.json     # Layer statistics and health
```

### **Memory Flow Architecture**
```
New Context Item
      ↓
[Immediate Memory] → (2hr) → [Short-term Memory] → (24hr) → [Medium-term Memory]
                                    ↓                              ↓
                              [Compression]                  [Compression]
                                    ↓                              ↓
                              (Relevance > 0.7)              (Relevance > 0.8)
                                    ↓                              ↓
                            [Long-term Memory] ← (1yr) ← [Long-term Memory]
                                    ↓
                              (Critical Items)
                                    ↓
                            [Permanent Memory]
```

## 📊 Data Models

### **1. Memory Layer Configuration DTO**

```python
# ff_class_configs/ff_memory_layer_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class MemoryLayerType(str, Enum):
    """Memory layer types with different retention policies."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"

@dataclass
class FFMemoryLayerConfigDTO:
    """Configuration for multi-layered memory system."""
    
    # Retention policies (in hours, -1 = never expires)
    retention_policies: Dict[str, int] = field(default_factory=lambda: {
        "immediate": 2,        # 2 hours
        "short_term": 24,      # 24 hours
        "medium_term": 168,    # 1 week (168 hours)
        "long_term": 8760,     # 1 year (8760 hours)
        "permanent": -1        # Never expires
    })
    
    # Maximum items per layer before compression
    compression_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "immediate": 100,      # Compress after 100 items
        "short_term": 500,     # Compress after 500 items
        "medium_term": 1000,   # Compress after 1000 items
        "long_term": 2000,     # Compress after 2000 items
        "permanent": 5000      # Compress after 5000 items
    })
    
    # Relevance thresholds for promotion between layers
    promotion_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "immediate_to_short": 0.5,    # Promote if relevance > 0.5
        "short_to_medium": 0.6,       # Promote if relevance > 0.6
        "medium_to_long": 0.7,        # Promote if relevance > 0.7
        "long_to_permanent": 0.8      # Promote if relevance > 0.8
    })
    
    # Compression settings
    enable_compression: bool = True
    compression_strategy: str = "summarization"  # "summarization", "clustering", "importance"
    compression_ratio: float = 0.3  # Target 30% of original size after compression
    
    # Auto-archival settings
    enable_auto_archival: bool = True
    archival_check_interval_hours: int = 6  # Check every 6 hours
    
    # Performance settings
    max_memory_search_results: int = 50
    memory_search_timeout_seconds: int = 5
    enable_memory_indexing: bool = True
    
    # LLM integration for summarization
    summarization_enabled: bool = True
    summarization_model: str = "claude-3-sonnet"
    summarization_max_tokens: int = 200
    summarization_temperature: float = 0.3
```

### **2. Memory Context DTO**

```python
# ff_class_configs/ff_chat_entities_config.py (extend existing file)

@dataclass
class FFMemoryContextDTO:
    """Individual memory context item with layer-specific metadata."""
    
    # Core content
    content: str
    context_type: str  # "conversation", "insight", "preference", "pattern"
    
    # Temporal information
    timestamp: str = field(default_factory=current_timestamp)
    layer: str = MemoryLayerType.IMMEDIATE.value
    created_at: str = field(default_factory=current_timestamp)
    last_accessed: str = field(default_factory=current_timestamp)
    
    # Relevance and importance
    relevance_score: float = 0.5  # 0.0 to 1.0
    importance_score: float = 0.5  # 0.0 to 1.0  
    access_count: int = 0
    
    # Context relationships
    related_session_ids: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    entity_mentions: List[str] = field(default_factory=list)
    
    # Memory-specific metadata
    compression_metadata: Dict[str, Any] = field(default_factory=dict)
    promotion_history: List[str] = field(default_factory=list)  # Track layer promotions
    source_layer: Optional[str] = None  # Original layer when compressed
    
    # User and session context
    user_id: str = ""
    session_id: Optional[str] = None
    
    def promote_to_layer(self, new_layer: str) -> None:
        """Promote memory item to a higher layer."""
        self.promotion_history.append(f"{self.layer} -> {new_layer} at {current_timestamp()}")
        self.layer = new_layer
        self.last_accessed = current_timestamp()
        self.access_count += 1
    
    def update_relevance(self, new_score: float) -> None:
        """Update relevance score and access information."""
        self.relevance_score = max(0.0, min(1.0, new_score))
        self.last_accessed = current_timestamp()
        self.access_count += 1

@dataclass
class FFMemoryCompressionResultDTO:
    """Result of memory compression operation."""
    
    original_count: int
    compressed_count: int
    compression_ratio: float
    compression_strategy: str
    summary_content: str
    preserved_items: List[FFMemoryContextDTO]
    compression_timestamp: str = field(default_factory=current_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## 🔧 Implementation Specifications

### **1. Memory Layer Manager**

```python
# ff_memory_layer_manager.py

"""
Multi-layered memory management system.

Provides sophisticated memory hierarchy with automatic retention policies,
compression, and intelligent archival while integrating with existing context system.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_memory_layer_config import FFMemoryLayerConfigDTO, MemoryLayerType
from ff_class_configs.ff_chat_entities_config import FFMemoryContextDTO, FFMemoryCompressionResultDTO
from ff_utils.ff_file_ops import ff_atomic_write, ff_ensure_directory
from ff_utils.ff_json_utils import ff_read_jsonl, ff_append_jsonl, ff_write_json, ff_read_json
from ff_utils.ff_logging import get_logger

class FFMemoryLayerManager:
    """
    Multi-layered memory management system following flatfile patterns.
    
    Manages five memory layers with automatic retention, compression, and archival:
    - Immediate: 2-hour retention for current conversation context
    - Short-term: 24-hour retention for recent interactions  
    - Medium-term: 1-week retention for weekly patterns
    - Long-term: 1-year retention for persistent patterns
    - Permanent: Never expires for critical insights
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize memory layer manager."""
        self.config = config
        self.memory_config = getattr(config, 'memory_layer', FFMemoryLayerConfigDTO())
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Memory layer file paths
        self.layer_files = {
            MemoryLayerType.IMMEDIATE: "immediate.jsonl",
            MemoryLayerType.SHORT_TERM: "short_term.jsonl", 
            MemoryLayerType.MEDIUM_TERM: "medium_term.jsonl",
            MemoryLayerType.LONG_TERM: "long_term.jsonl",
            MemoryLayerType.PERMANENT: "permanent.jsonl"
        }
        
        # In-memory caches for performance
        self._memory_caches: Dict[str, List[FFMemoryContextDTO]] = {}
        self._last_cleanup = datetime.now()
        
    def _get_memory_path(self, user_id: str) -> Path:
        """Get memory directory path for user."""
        return self.base_path / "users" / user_id / "memory_layers"
    
    async def initialize_user_memory(self, user_id: str) -> bool:
        """Initialize memory layer structure for user."""
        try:
            memory_path = self._get_memory_path(user_id)
            await ff_ensure_directory(memory_path)
            
            # Create layer metadata file if it doesn't exist
            metadata_path = memory_path / "layer_metadata.json"
            if not metadata_path.exists():
                metadata = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "last_maintenance": datetime.now().isoformat(),
                    "layer_stats": {layer.value: {"item_count": 0, "last_compression": None} 
                                  for layer in MemoryLayerType},
                    "compression_history": []
                }
                await ff_write_json(metadata_path, metadata, self.config)
            
            self.logger.info(f"Initialized memory layers for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory for user {user_id}: {e}")
            return False
    
    async def add_memory_context(
        self, 
        user_id: str, 
        context: FFMemoryContextDTO,
        target_layer: MemoryLayerType = MemoryLayerType.IMMEDIATE
    ) -> bool:
        """Add context item to specified memory layer."""
        try:
            # Ensure user memory is initialized
            await self.initialize_user_memory(user_id)
            
            # Set user context
            context.user_id = user_id
            context.layer = target_layer.value
            
            # Get layer file path
            memory_path = self._get_memory_path(user_id)
            layer_file = memory_path / self.layer_files[target_layer]
            
            # Append to layer file
            await ff_append_jsonl(layer_file, [context.to_dict()], self.config)
            
            # Update cache
            cache_key = f"{user_id}_{target_layer.value}"
            if cache_key not in self._memory_caches:
                self._memory_caches[cache_key] = []
            self._memory_caches[cache_key].append(context)
            
            # Check if compression is needed
            if await self._should_compress_layer(user_id, target_layer):
                await self._compress_layer(user_id, target_layer)
            
            self.logger.debug(f"Added context to {target_layer.value} layer for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add memory context for user {user_id}: {e}")
            return False
    
    async def get_memory_contexts(
        self,
        user_id: str,
        layers: Optional[List[MemoryLayerType]] = None,
        max_results: Optional[int] = None,
        relevance_threshold: float = 0.0
    ) -> List[FFMemoryContextDTO]:
        """Retrieve memory contexts from specified layers."""
        try:
            if layers is None:
                layers = list(MemoryLayerType)
            
            all_contexts = []
            
            for layer in layers:
                layer_contexts = await self._load_layer_contexts(user_id, layer)
                
                # Filter by relevance threshold
                filtered_contexts = [
                    ctx for ctx in layer_contexts 
                    if ctx.relevance_score >= relevance_threshold
                ]
                
                all_contexts.extend(filtered_contexts)
            
            # Sort by relevance score and recency
            all_contexts.sort(
                key=lambda x: (x.relevance_score, x.last_accessed), 
                reverse=True
            )
            
            # Apply max results limit
            if max_results:
                all_contexts = all_contexts[:max_results]
            
            return all_contexts
            
        except Exception as e:
            self.logger.error(f"Failed to get memory contexts for user {user_id}: {e}")
            return []
    
    async def search_memory(
        self,
        user_id: str,
        query: str,
        layers: Optional[List[MemoryLayerType]] = None,
        max_results: int = 20
    ) -> List[Tuple[FFMemoryContextDTO, float]]:
        """Search memory contexts with relevance scoring."""
        try:
            contexts = await self.get_memory_contexts(user_id, layers)
            
            # Simple text-based relevance scoring
            results = []
            query_words = set(query.lower().split())
            
            for context in contexts:
                # Calculate relevance based on content overlap
                content_words = set(context.content.lower().split())
                overlap = len(query_words.intersection(content_words))
                
                if overlap > 0:
                    relevance = (overlap / len(query_words)) * context.relevance_score
                    results.append((context, relevance))
            
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to search memory for user {user_id}: {e}")
            return []
    
    async def run_memory_maintenance(self, user_id: str) -> Dict[str, Any]:
        """Run comprehensive memory maintenance tasks."""
        try:
            maintenance_results = {
                "user_id": user_id,
                "start_time": datetime.now().isoformat(),
                "tasks_completed": [],
                "errors": []
            }
            
            # 1. Clean expired memories
            try:
                expired_count = await self._clean_expired_memories(user_id)
                maintenance_results["tasks_completed"].append(f"cleaned_{expired_count}_expired")
            except Exception as e:
                maintenance_results["errors"].append(f"cleanup_expired: {str(e)}")
            
            # 2. Auto-promote memories between layers
            try:
                promoted_count = await self._auto_promote_memories(user_id)
                maintenance_results["tasks_completed"].append(f"promoted_{promoted_count}_memories")
            except Exception as e:
                maintenance_results["errors"].append(f"auto_promote: {str(e)}")
            
            # 3. Compress layers that exceed thresholds
            try:
                compressed_layers = await self._compress_all_layers(user_id)
                maintenance_results["tasks_completed"].append(f"compressed_{len(compressed_layers)}_layers")
            except Exception as e:
                maintenance_results["errors"].append(f"compress_layers: {str(e)}")
            
            # 4. Update layer metadata
            try:
                await self._update_layer_metadata(user_id)
                maintenance_results["tasks_completed"].append("updated_metadata")
            except Exception as e:
                maintenance_results["errors"].append(f"update_metadata: {str(e)}")
            
            maintenance_results["end_time"] = datetime.now().isoformat()
            maintenance_results["success"] = len(maintenance_results["errors"]) == 0
            
            return maintenance_results
            
        except Exception as e:
            self.logger.error(f"Memory maintenance failed for user {user_id}: {e}")
            return {"success": False, "error": str(e)}
    
    # Private helper methods
    
    async def _load_layer_contexts(self, user_id: str, layer: MemoryLayerType) -> List[FFMemoryContextDTO]:
        """Load contexts from specific memory layer."""
        try:
            memory_path = self._get_memory_path(user_id)
            layer_file = memory_path / self.layer_files[layer]
            
            if not layer_file.exists():
                return []
            
            context_data = await ff_read_jsonl(layer_file, self.config)
            contexts = [FFMemoryContextDTO.from_dict(data) for data in context_data]
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Failed to load {layer.value} contexts for user {user_id}: {e}")
            return []
    
    async def _should_compress_layer(self, user_id: str, layer: MemoryLayerType) -> bool:
        """Check if layer should be compressed based on thresholds."""
        try:
            contexts = await self._load_layer_contexts(user_id, layer)
            threshold = self.memory_config.compression_thresholds.get(layer.value, 1000)
            
            return len(contexts) > threshold
            
        except Exception as e:
            self.logger.error(f"Failed to check compression threshold for {layer.value}: {e}")
            return False
    
    async def _compress_layer(self, user_id: str, layer: MemoryLayerType) -> FFMemoryCompressionResultDTO:
        """Compress memory layer using configured strategy."""
        try:
            contexts = await self._load_layer_contexts(user_id, layer)
            
            if not contexts:
                return FFMemoryCompressionResultDTO(0, 0, 0.0, "none", "", [])
            
            # Group contexts by topic/session for better compression
            grouped_contexts = self._group_contexts_for_compression(contexts)
            
            compressed_contexts = []
            total_compressed = 0
            
            for group_key, group_contexts in grouped_contexts.items():
                if len(group_contexts) > 5:  # Only compress groups with multiple items
                    summary = await self._create_compression_summary(group_contexts)
                    
                    # Create compressed context item
                    compressed_context = FFMemoryContextDTO(
                        content=summary,
                        context_type="compressed_summary",
                        layer=layer.value,
                        user_id=user_id,
                        relevance_score=max(ctx.relevance_score for ctx in group_contexts),
                        importance_score=max(ctx.importance_score for ctx in group_contexts),
                        topic_tags=list(set(tag for ctx in group_contexts for tag in ctx.topic_tags)),
                        compression_metadata={
                            "original_count": len(group_contexts),
                            "group_key": group_key,
                            "compression_strategy": self.memory_config.compression_strategy
                        }
                    )
                    
                    compressed_contexts.append(compressed_context)
                    total_compressed += len(group_contexts)
                else:
                    # Keep small groups as-is
                    compressed_contexts.extend(group_contexts)
            
            # Write compressed contexts back to file
            memory_path = self._get_memory_path(user_id)
            layer_file = memory_path / self.layer_files[layer]
            
            compressed_data = [ctx.to_dict() for ctx in compressed_contexts]
            await ff_write_json(layer_file.with_suffix('.json'), compressed_data, self.config)
            
            # Create backup of original
            backup_file = layer_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
            original_data = [ctx.to_dict() for ctx in contexts]
            await ff_write_json(backup_file, original_data, self.config)
            
            # Replace original with compressed version
            await ff_write_json(layer_file.with_suffix('.jsonl'), compressed_data, self.config)
            
            compression_ratio = (len(compressed_contexts) / len(contexts)) if contexts else 0.0
            
            return FFMemoryCompressionResultDTO(
                original_count=len(contexts),
                compressed_count=len(compressed_contexts),
                compression_ratio=compression_ratio,
                compression_strategy=self.memory_config.compression_strategy,
                summary_content=f"Compressed {len(contexts)} items to {len(compressed_contexts)}",
                preserved_items=compressed_contexts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compress {layer.value} layer for user {user_id}: {e}")
            return FFMemoryCompressionResultDTO(0, 0, 0.0, "failed", str(e), [])
```

### **2. Memory Protocol Interface**

```python
# ff_protocols/ff_memory_layer_protocol.py

"""Protocol interface for memory layer management."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ff_class_configs.ff_chat_entities_config import FFMemoryContextDTO
from ff_class_configs.ff_memory_layer_config import MemoryLayerType

class MemoryLayerProtocol(ABC):
    """Protocol interface for memory layer management operations."""
    
    @abstractmethod
    async def initialize_user_memory(self, user_id: str) -> bool:
        """Initialize memory layer structure for user."""
        pass
    
    @abstractmethod
    async def add_memory_context(
        self, 
        user_id: str, 
        context: FFMemoryContextDTO,
        target_layer: MemoryLayerType = MemoryLayerType.IMMEDIATE
    ) -> bool:
        """Add context item to specified memory layer."""
        pass
    
    @abstractmethod
    async def get_memory_contexts(
        self,
        user_id: str,
        layers: Optional[List[MemoryLayerType]] = None,
        max_results: Optional[int] = None,
        relevance_threshold: float = 0.0
    ) -> List[FFMemoryContextDTO]:
        """Retrieve memory contexts from specified layers."""
        pass
    
    @abstractmethod
    async def search_memory(
        self,
        user_id: str,
        query: str,
        layers: Optional[List[MemoryLayerType]] = None,
        max_results: int = 20
    ) -> List[Tuple[FFMemoryContextDTO, float]]:
        """Search memory contexts with relevance scoring."""
        pass
    
    @abstractmethod
    async def run_memory_maintenance(self, user_id: str) -> Dict[str, Any]:
        """Run comprehensive memory maintenance tasks."""
        pass
```

### **3. Integration with Configuration Manager**

```python
# ff_class_configs/ff_configuration_manager_config.py (extend existing)

# Add to existing FFConfigurationManagerConfigDTO class:

@dataclass
class FFConfigurationManagerConfigDTO:
    # ... existing fields ...
    
    memory_layer: FFMemoryLayerConfigDTO = field(default_factory=FFMemoryLayerConfigDTO)
    
    # ... rest of existing implementation ...
```

### **4. Enhanced Context Manager Integration**

```python
# ff_context_manager.py (extend existing)

# Add methods to integrate with memory layers:

class FFContextManager:
    """Enhanced context manager with memory layer integration."""
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        # ... existing initialization ...
        
        # Add memory layer manager
        self.memory_manager = FFMemoryLayerManager(config)
    
    async def add_context_with_memory(
        self,
        user_id: str,
        context: FFSituationalContextDTO,
        auto_promote_to_memory: bool = True
    ) -> bool:
        """Add context and optionally promote to memory layers."""
        
        # Add to existing context system
        success = await self.add_context(user_id, context)
        
        if success and auto_promote_to_memory:
            # Convert to memory context
            memory_context = FFMemoryContextDTO(
                content=context.context_content,
                context_type="situational",
                user_id=user_id,
                session_id=context.session_id,
                relevance_score=context.relevance_score,
                topic_tags=context.context_tags
            )
            
            # Add to immediate memory layer
            await self.memory_manager.add_memory_context(
                user_id, 
                memory_context, 
                MemoryLayerType.IMMEDIATE
            )
        
        return success
```

## 🧪 Testing Specifications

### **Unit Tests**

```python
# tests/test_memory_layer_manager.py

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from ff_memory_layer_manager import FFMemoryLayerManager
from ff_class_configs.ff_memory_layer_config import FFMemoryLayerConfigDTO, MemoryLayerType
from ff_class_configs.ff_chat_entities_config import FFMemoryContextDTO

class TestMemoryLayerManager:
    
    @pytest.fixture
    async def memory_manager(self):
        """Create memory manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FFConfigurationManagerConfigDTO()
            config.storage.base_path = temp_dir
            config.memory_layer = FFMemoryLayerConfigDTO()
            
            manager = FFMemoryLayerManager(config)
            yield manager
    
    @pytest.mark.asyncio
    async def test_initialize_user_memory(self, memory_manager):
        """Test user memory initialization."""
        user_id = "test_user"
        
        success = await memory_manager.initialize_user_memory(user_id)
        assert success
        
        # Check directory structure created
        memory_path = memory_manager._get_memory_path(user_id)
        assert memory_path.exists()
        assert (memory_path / "layer_metadata.json").exists()
    
    @pytest.mark.asyncio
    async def test_add_memory_context(self, memory_manager):
        """Test adding context to memory layers."""
        user_id = "test_user"
        await memory_manager.initialize_user_memory(user_id)
        
        context = FFMemoryContextDTO(
            content="Test memory content",
            context_type="conversation",
            user_id=user_id,
            relevance_score=0.8
        )
        
        success = await memory_manager.add_memory_context(
            user_id, context, MemoryLayerType.IMMEDIATE
        )
        assert success
        
        # Verify context was added
        contexts = await memory_manager.get_memory_contexts(
            user_id, [MemoryLayerType.IMMEDIATE]
        )
        assert len(contexts) == 1
        assert contexts[0].content == "Test memory content"
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test memory search functionality."""
        user_id = "test_user"
        await memory_manager.initialize_user_memory(user_id)
        
        # Add test contexts
        contexts = [
            FFMemoryContextDTO(
                content="Python programming tutorial",
                context_type="conversation",
                user_id=user_id,
                relevance_score=0.9
            ),
            FFMemoryContextDTO(
                content="JavaScript web development",
                context_type="conversation", 
                user_id=user_id,
                relevance_score=0.7
            )
        ]
        
        for context in contexts:
            await memory_manager.add_memory_context(
                user_id, context, MemoryLayerType.IMMEDIATE
            )
        
        # Search for Python-related content
        results = await memory_manager.search_memory(user_id, "Python programming")
        
        assert len(results) >= 1
        assert results[0][0].content == "Python programming tutorial"
        assert results[0][1] > 0  # Has relevance score
    
    @pytest.mark.asyncio
    async def test_memory_maintenance(self, memory_manager):
        """Test memory maintenance operations."""
        user_id = "test_user"
        await memory_manager.initialize_user_memory(user_id)
        
        # Add some test data
        context = FFMemoryContextDTO(
            content="Test maintenance content",
            context_type="conversation",
            user_id=user_id
        )
        await memory_manager.add_memory_context(user_id, context)
        
        # Run maintenance
        results = await memory_manager.run_memory_maintenance(user_id)
        
        assert results["success"]
        assert len(results["tasks_completed"]) > 0
        assert len(results["errors"]) == 0
```

### **Integration Tests**

```python
# tests/test_memory_integration.py

class TestMemoryIntegration:
    
    @pytest.mark.asyncio
    async def test_context_manager_memory_integration(self):
        """Test integration between context manager and memory layers."""
        # Test that situational contexts automatically promote to memory
        pass
    
    @pytest.mark.asyncio
    async def test_memory_layer_promotion(self):
        """Test automatic promotion between memory layers."""
        # Test that high-relevance items get promoted to higher layers
        pass
    
    @pytest.mark.asyncio
    async def test_memory_compression(self):
        """Test memory compression functionality."""
        # Test that memory layers compress when thresholds are exceeded
        pass
```

## 📈 Success Criteria

### **Functional Requirements**
- ✅ All 5 memory layers operational with configurable retention policies
- ✅ Automatic archival and promotion between layers based on relevance
- ✅ Memory compression reduces storage while preserving important information
- ✅ Search functionality works across all memory layers
- ✅ Integration with existing context system maintains backward compatibility

### **Performance Requirements**
- ✅ Memory operations complete within configured timeout (5 seconds default)
- ✅ Memory search returns results within 1 second for typical queries
- ✅ Memory maintenance runs without blocking normal operations
- ✅ Memory compression achieves target compression ratio (30% default)

### **Integration Requirements**
- ✅ Existing `FFContextManager` functionality unchanged
- ✅ New memory capabilities accessible through configuration
- ✅ Memory layers properly isolated per user
- ✅ All operations follow existing async patterns and error handling

### **Testing Requirements**
- ✅ Unit test coverage > 90% for all new components
- ✅ Integration tests validate memory layer interactions
- ✅ Performance tests validate memory operation speed
- ✅ End-to-end tests validate full memory lifecycle

## 🚀 Implementation Checklist

### **Phase 1A: Core Data Models**
- [ ] Create `ff_memory_layer_config.py` with complete configuration DTOs
- [ ] Extend `ff_chat_entities_config.py` with memory context DTOs
- [ ] Create `ff_memory_layer_protocol.py` with abstract interface
- [ ] Update configuration manager to include memory layer config

### **Phase 1B: Memory Layer Manager**
- [ ] Implement `FFMemoryLayerManager` with core functionality
- [ ] Add memory initialization and context management
- [ ] Implement memory search and retrieval operations
- [ ] Add memory maintenance and cleanup operations

### **Phase 1C: Memory Compression**
- [ ] Implement memory compression strategies
- [ ] Add automatic archival between layers
- [ ] Create compression result tracking
- [ ] Add compression performance monitoring

### **Phase 1D: Integration & Testing**
- [ ] Integrate memory manager with existing context system
- [ ] Update dependency injection container registration
- [ ] Create comprehensive unit test suite
- [ ] Create integration tests with existing components
- [ ] Performance test memory operations
- [ ] Validate backward compatibility

### **Phase 1E: Documentation & Validation**
- [ ] Update configuration documentation
- [ ] Create memory system usage examples
- [ ] Validate all success criteria met
- [ ] Performance benchmark memory operations
- [ ] Create migration guide for existing data

This comprehensive specification provides everything needed to implement the multi-layered memory system while maintaining your excellent architectural standards and ensuring full backward compatibility.