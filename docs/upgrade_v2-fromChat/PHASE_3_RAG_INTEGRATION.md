# Phase 3: RAG Integration & Knowledge Base Management

## 🎯 Phase Overview

Implement sophisticated Retrieval-Augmented Generation (RAG) capabilities that extend your existing vector storage system to support PrismMind's advanced knowledge-based conversation features. This includes personal knowledge base management, intelligent document indexing, conversation-specific context retrieval, and integration with the multi-layered memory system.

## 📋 Requirements Analysis

### **Current State Assessment**
Your system already has excellent foundations:
- ✅ `FFVectorStorageManager` - Vector storage with NumPy arrays and JSONL indexing
- ✅ `FFDocumentManager` - Document processing and metadata tracking
- ✅ `FFEmbeddingManager` - Embedding generation and management
- ✅ `FFSearchManager` - Advanced search with vector similarity
- ✅ Vector storage structure with embeddings and indices

### **RAG Enhancement Requirements**
Based on PrismMind's knowledge-intensive use cases:
1. **Personal Knowledge Bases** - User-specific document collections and expertise domains
2. **Intelligent Retrieval** - Context-aware document retrieval with relevance scoring
3. **RAG Context Management** - Conversation-specific context injection and tracking
4. **Knowledge Base Lifecycle** - Automatic indexing, updating, and maintenance
5. **Memory Integration** - Knowledge bases feeding into multi-layered memory system

## 🏗️ Architecture Design

### **Enhanced Knowledge Base Storage Structure**
```
users/{user_id}/knowledge_bases/
├── {kb_name}/
│   ├── kb_config.json              # Knowledge base configuration
│   ├── documents/                  # Source documents
│   │   ├── {doc_id}_{filename}     # Original documents
│   │   └── metadata.json          # Document metadata
│   ├── vectors/                    # Document embeddings
│   │   ├── embeddings.npy          # Vector embeddings
│   │   ├── vector_index.jsonl      # Vector metadata index
│   │   └── chunk_index.jsonl       # Document chunk mappings
│   ├── retrieval_cache/            # Cached query results
│   │   ├── query_cache.json        # Recent query results
│   │   └── performance_stats.json  # Retrieval performance metrics
│   └── kb_analytics.json           # Knowledge base usage analytics
└── global_kb_index.json            # User's knowledge base registry

users/{user_id}/rag_sessions/
├── {session_id}/
│   ├── session_config.json         # RAG session configuration
│   ├── active_knowledge_bases.json # Linked knowledge bases
│   ├── messages.jsonl              # Conversation messages
│   └── rag_context/                # Retrieved context per message
│       ├── msg_{message_id}_context.json  # Context for specific message
│       └── context_usage_stats.json       # Context utilization metrics
```

### **RAG Processing Flow**
```
User Query
    ↓
[Query Analysis] → [Knowledge Base Selection] → [Multi-KB Retrieval]
    ↓                      ↓                         ↓
[Context Ranking] → [Context Synthesis] → [RAG Context Injection]
    ↓                      ↓                         ↓
[LLM Generation] → [Response Enhancement] → [Context Usage Tracking]
    ↓
[Memory Layer Integration] → [Knowledge Base Updates]
```

## 📊 Data Models

### **1. Knowledge Base Configuration**

```python
# ff_class_configs/ff_knowledge_base_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class KnowledgeBaseType(str, Enum):
    """Types of knowledge bases."""
    PERSONAL_DOCS = "personal_documents"
    WORK_KNOWLEDGE = "work_knowledge"
    RESEARCH_PAPERS = "research_papers"
    REFERENCE_MATERIALS = "reference_materials"
    CONVERSATION_HISTORY = "conversation_history"
    CUSTOM_DOMAIN = "custom_domain"

class RetrievalStrategy(str, Enum):
    """Retrieval strategies for different use cases."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCHING = "keyword_matching"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    CONTEXTUAL_RELEVANCE = "contextual_relevance"
    TEMPORAL_RELEVANCE = "temporal_relevance"

class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC_CHUNKING = "semantic_chunking"
    HIERARCHICAL_CHUNKING = "hierarchical_chunking"

@dataclass
class FFKnowledgeBaseConfigDTO:
    """Configuration for knowledge base management and RAG operations."""
    
    # Knowledge base settings
    max_knowledge_bases_per_user: int = 10
    max_documents_per_kb: int = 1000
    max_kb_size_mb: int = 500
    enable_kb_versioning: bool = True
    
    # Document processing settings
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_CHUNKING
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_chunk_size: int = 1024
    min_chunk_size: int = 100
    
    # Embedding settings
    embedding_model: str = "nomic-embed-text-v1.5"
    embedding_dimensions: int = 768
    normalize_embeddings: bool = True
    enable_batch_embedding: bool = True
    batch_size: int = 32
    
    # Retrieval settings
    default_retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RETRIEVAL
    max_retrieval_results: int = 20
    min_relevance_threshold: float = 0.3
    enable_retrieval_reranking: bool = True
    reranking_model: Optional[str] = None
    
    # RAG context settings
    max_context_length: int = 4000  # tokens
    context_overlap_handling: str = "merge"  # "merge", "truncate", "prioritize"
    enable_context_compression: bool = True
    context_relevance_decay: float = 0.1  # per hour
    
    # Caching settings
    enable_retrieval_cache: bool = True
    cache_ttl_hours: int = 24
    max_cache_entries: int = 1000
    cache_compression: bool = True
    
    # Performance settings
    retrieval_timeout_seconds: int = 10
    embedding_timeout_seconds: int = 30
    enable_async_processing: bool = True
    max_concurrent_operations: int = 5
    
    # Integration settings
    integrate_with_memory_layers: bool = True
    memory_integration_threshold: float = 0.7
    enable_cross_kb_retrieval: bool = True
    enable_kb_analytics: bool = True
    
    # Maintenance settings
    enable_auto_maintenance: bool = True
    maintenance_interval_hours: int = 24
    enable_duplicate_detection: bool = True
    duplicate_similarity_threshold: float = 0.95

@dataclass
class FFRAGSessionConfigDTO:
    """Configuration for RAG-enabled conversation sessions."""
    
    # Session RAG settings
    enable_rag: bool = True
    active_knowledge_bases: List[str] = field(default_factory=list)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RETRIEVAL
    max_context_per_message: int = 3  # Number of context chunks per message
    
    # Context injection settings
    context_injection_mode: str = "automatic"  # "automatic", "explicit", "hybrid"
    context_visibility: str = "hidden"  # "hidden", "visible", "optional"
    enable_context_citations: bool = True
    citation_format: str = "academic"  # "academic", "numeric", "parenthetical"
    
    # Adaptive settings
    enable_adaptive_retrieval: bool = True
    context_quality_threshold: float = 0.5
    enable_retrieval_feedback: bool = True
    feedback_learning_rate: float = 0.1
    
    # Performance settings
    context_cache_enabled: bool = True
    precompute_embeddings: bool = True
    enable_parallel_retrieval: bool = True
```

### **2. Knowledge Base Data Models**

```python
# ff_class_configs/ff_chat_entities_config.py (extend existing)

@dataclass
class FFKnowledgeBaseDTO:
    """Knowledge base with metadata and configuration."""
    
    # Basic information
    kb_id: str = field(default_factory=lambda: f"kb_{uuid.uuid4().hex[:12]}")
    user_id: str = ""
    name: str = ""
    description: str = ""
    kb_type: KnowledgeBaseType = KnowledgeBaseType.PERSONAL_DOCS
    
    # Timestamps
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    last_accessed: str = field(default_factory=current_timestamp)
    
    # Content statistics
    document_count: int = 0
    total_chunks: int = 0
    total_size_bytes: int = 0
    embedding_count: int = 0
    
    # Configuration
    chunking_config: Dict[str, Any] = field(default_factory=dict)
    embedding_config: Dict[str, Any] = field(default_factory=dict)
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    
    # Usage statistics
    query_count: int = 0
    retrieval_count: int = 0
    average_query_time_ms: float = 0.0
    last_maintenance: Optional[str] = None
    
    # Access control
    is_active: bool = True
    access_permissions: List[str] = field(default_factory=list)
    sharing_enabled: bool = False
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_statistics(self, query_time_ms: float) -> None:
        """Update usage statistics."""
        self.query_count += 1
        self.retrieval_count += 1
        self.last_accessed = current_timestamp()
        
        # Update average query time
        if self.average_query_time_ms == 0.0:
            self.average_query_time_ms = query_time_ms
        else:
            alpha = 0.1  # Exponential moving average factor
            self.average_query_time_ms = (alpha * query_time_ms + 
                                        (1 - alpha) * self.average_query_time_ms)

@dataclass
class FFRAGContextDTO:
    """Retrieved context for RAG-enhanced responses."""
    
    # Context identification
    context_id: str = field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:12]}")
    kb_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    
    # Content
    content: str = ""
    title: Optional[str] = None
    document_title: str = ""
    
    # Retrieval metadata
    relevance_score: float = 0.0
    retrieval_method: str = "semantic_similarity"
    query_used: str = ""
    
    # Context positioning
    chunk_index: int = 0
    total_chunks: int = 1
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    
    # Source information
    source_path: str = ""
    source_type: str = "document"  # "document", "web", "conversation", "generated"
    last_updated: str = field(default_factory=current_timestamp)
    
    # Usage tracking
    used_in_response: bool = False
    user_feedback: Optional[str] = None  # "helpful", "not_helpful", "irrelevant"
    usage_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_as_used(self) -> None:
        """Mark context as used in response generation."""
        self.used_in_response = True
        self.usage_count += 1

@dataclass
class FFRAGRetrievalResultDTO:
    """Results of RAG retrieval operation."""
    
    # Query information
    query: str = ""
    retrieval_strategy: str = "hybrid_retrieval"
    timestamp: str = field(default_factory=current_timestamp)
    
    # Results
    contexts: List[FFRAGContextDTO] = field(default_factory=list)
    total_results: int = 0
    retrieval_time_ms: float = 0.0
    
    # Quality metrics
    average_relevance: float = 0.0
    context_diversity: float = 0.0  # How diverse the retrieved contexts are
    coverage_score: float = 0.0     # How well contexts cover the query
    
    # Knowledge base coverage
    kb_coverage: Dict[str, int] = field(default_factory=dict)  # kb_id -> result_count
    source_diversity: Dict[str, int] = field(default_factory=dict)  # source_type -> count
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_quality_metrics(self) -> None:
        """Calculate quality metrics for retrieval results."""
        if not self.contexts:
            return
        
        # Average relevance
        self.average_relevance = sum(ctx.relevance_score for ctx in self.contexts) / len(self.contexts)
        
        # Context diversity (based on content similarity)
        if len(self.contexts) > 1:
            # Simple diversity measure based on source diversity
            unique_docs = len(set(ctx.document_id for ctx in self.contexts))
            self.context_diversity = unique_docs / len(self.contexts)
        else:
            self.context_diversity = 1.0
        
        # Coverage score (placeholder - could be enhanced with more sophisticated metrics)
        self.coverage_score = min(1.0, len(self.contexts) / 5)  # Assume 5 contexts provide full coverage

@dataclass
class FFKnowledgeBaseAnalyticsDTO:
    """Analytics for knowledge base usage and performance."""
    
    # Basic info
    kb_id: str = ""
    analysis_period_start: str = field(default_factory=current_timestamp)
    analysis_period_end: str = field(default_factory=current_timestamp)
    
    # Usage statistics
    total_queries: int = 0
    total_retrievals: int = 0
    unique_users: int = 0
    average_query_frequency: float = 0.0
    
    # Performance metrics
    average_retrieval_time_ms: float = 0.0
    average_relevance_score: float = 0.0
    success_rate: float = 0.0  # Percentage of queries with relevant results
    
    # Content metrics
    most_retrieved_documents: List[Tuple[str, int]] = field(default_factory=list)
    popular_query_patterns: List[Tuple[str, int]] = field(default_factory=list)
    underutilized_content: List[str] = field(default_factory=list)
    
    # Quality metrics
    user_satisfaction_score: float = 0.0
    context_utilization_rate: float = 0.0  # How often retrieved contexts are actually used
    duplicate_content_percentage: float = 0.0
    
    # Recommendations
    optimization_suggestions: List[str] = field(default_factory=list)
    maintenance_tasks: List[str] = field(default_factory=list)
    
    # Trends
    query_volume_trend: str = "stable"  # "increasing", "decreasing", "stable"
    performance_trend: str = "stable"
    content_growth_rate: float = 0.0
```

## 🔧 Implementation Specifications

### **1. Knowledge Base Manager**

```python
# ff_knowledge_base_manager.py

"""
Knowledge Base Management System.

Provides comprehensive knowledge base lifecycle management including
creation, document indexing, retrieval optimization, and maintenance.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_knowledge_base_config import (
    FFKnowledgeBaseConfigDTO, KnowledgeBaseType, RetrievalStrategy, ChunkingStrategy
)
from ff_class_configs.ff_chat_entities_config import (
    FFKnowledgeBaseDTO, FFRAGContextDTO, FFRAGRetrievalResultDTO, FFKnowledgeBaseAnalyticsDTO
)
from ff_utils.ff_file_ops import ff_atomic_write, ff_ensure_directory
from ff_utils.ff_json_utils import ff_read_jsonl, ff_append_jsonl, ff_write_json, ff_read_json
from ff_utils.ff_logging import get_logger
from ff_embedding_manager import FFEmbeddingManager
from ff_vector_storage_manager import FFVectorStorageManager

class FFKnowledgeBaseManager:
    """
    Comprehensive knowledge base management following flatfile patterns.
    
    Manages the complete lifecycle of knowledge bases including:
    - Knowledge base creation and configuration
    - Document indexing with intelligent chunking
    - Vector embedding generation and storage
    - Intelligent retrieval with multiple strategies
    - Performance optimization and maintenance
    - Integration with memory layers and conversation context
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize knowledge base manager."""
        self.config = config
        self.kb_config = getattr(config, 'knowledge_base', FFKnowledgeBaseConfigDTO())
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Initialize dependent managers
        self.embedding_manager = FFEmbeddingManager(config)
        self.vector_storage = FFVectorStorageManager(config)
        
        # Knowledge base registry cache
        self._kb_registry_cache: Dict[str, Dict[str, FFKnowledgeBaseDTO]] = {}
        
        # Retrieval cache
        self._retrieval_cache: Dict[str, FFRAGRetrievalResultDTO] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def _get_user_kb_path(self, user_id: str) -> Path:
        """Get user's knowledge base directory path."""
        return self.base_path / "users" / user_id / "knowledge_bases"
    
    def _get_kb_path(self, user_id: str, kb_id: str) -> Path:
        """Get specific knowledge base path."""
        return self._get_user_kb_path(user_id) / kb_id
    
    async def create_knowledge_base(
        self,
        user_id: str,
        name: str,
        description: str = "",
        kb_type: KnowledgeBaseType = KnowledgeBaseType.PERSONAL_DOCS,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[FFKnowledgeBaseDTO]:
        """Create new knowledge base for user."""
        try:
            # Check user's KB limit
            user_kbs = await self.get_user_knowledge_bases(user_id)
            if len(user_kbs) >= self.kb_config.max_knowledge_bases_per_user:
                self.logger.warning(f"User {user_id} has reached KB limit")
                return None
            
            # Create knowledge base DTO
            kb = FFKnowledgeBaseDTO(
                user_id=user_id,
                name=name,
                description=description,
                kb_type=kb_type
            )
            
            # Apply configuration overrides
            if config_overrides:
                kb.chunking_config.update(config_overrides.get('chunking', {}))
                kb.embedding_config.update(config_overrides.get('embedding', {}))
                kb.retrieval_config.update(config_overrides.get('retrieval', {}))
            
            # Create KB directory structure
            kb_path = self._get_kb_path(user_id, kb.kb_id)
            await ff_ensure_directory(kb_path)
            
            for subdir in ["documents", "vectors", "retrieval_cache"]:
                await ff_ensure_directory(kb_path / subdir)
            
            # Save KB configuration
            await ff_write_json(kb_path / "kb_config.json", kb.to_dict(), self.config)
            
            # Initialize empty indices
            await ff_write_json(kb_path / "documents" / "metadata.json", {}, self.config)
            await ff_write_json(kb_path / "vectors" / "vector_index.jsonl", [], self.config)
            await ff_write_json(kb_path / "vectors" / "chunk_index.jsonl", [], self.config)
            
            # Initialize analytics
            analytics = FFKnowledgeBaseAnalyticsDTO(kb_id=kb.kb_id)
            await ff_write_json(kb_path / "kb_analytics.json", analytics.to_dict(), self.config)
            
            # Update user's KB registry
            await self._update_user_kb_registry(user_id, kb)
            
            self.logger.info(f"Created knowledge base {kb.kb_id} for user {user_id}")
            return kb
            
        except Exception as e:
            self.logger.error(f"Failed to create knowledge base for user {user_id}: {e}")
            return None
    
    async def add_document_to_kb(
        self,
        user_id: str,
        kb_id: str,
        document_path: str,
        document_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document to knowledge base with processing and indexing."""
        try:
            kb = await self.get_knowledge_base(user_id, kb_id)
            if not kb:
                self.logger.error(f"Knowledge base {kb_id} not found for user {user_id}")
                return False
            
            # Check KB size limits
            if kb.document_count >= self.kb_config.max_documents_per_kb:
                self.logger.warning(f"Knowledge base {kb_id} has reached document limit")
                return False
            
            # Generate document ID
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Process document content into chunks
            chunks = await self._chunk_document(document_content, kb.chunking_config)
            
            if not chunks:
                self.logger.warning(f"No chunks generated for document {document_path}")
                return False
            
            # Generate embeddings for chunks
            embeddings = await self.embedding_manager.generate_embeddings(
                chunks, kb.embedding_config
            )
            
            if not embeddings or len(embeddings) != len(chunks):
                self.logger.error(f"Failed to generate embeddings for document {document_path}")
                return False
            
            # Store document and vectors
            kb_path = self._get_kb_path(user_id, kb_id)
            
            # Save document content
            doc_filename = f"{doc_id}_{Path(document_path).name}"
            doc_path = kb_path / "documents" / doc_filename
            await ff_atomic_write(doc_path, document_content.encode('utf-8'))
            
            # Update document metadata
            doc_metadata = {
                "doc_id": doc_id,
                "original_path": document_path,
                "filename": doc_filename,
                "content_length": len(document_content),
                "chunk_count": len(chunks),
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            await self._update_document_metadata(user_id, kb_id, doc_id, doc_metadata)
            
            # Store vectors using existing vector storage manager
            await self.vector_storage.store_vectors(
                session_id=f"{user_id}_{kb_id}",
                document_id=doc_id,
                chunks=chunks,
                vectors=embeddings,
                metadata={
                    "kb_id": kb_id,
                    "user_id": user_id,
                    "document_path": document_path
                }
            )
            
            # Update KB statistics
            kb.document_count += 1
            kb.total_chunks += len(chunks)
            kb.total_size_bytes += len(document_content.encode('utf-8'))
            kb.embedding_count += len(embeddings)
            kb.updated_at = datetime.now().isoformat()
            
            await self._save_knowledge_base(kb)
            
            self.logger.info(f"Added document {doc_id} to KB {kb_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document to KB {kb_id}: {e}")
            return False
    
    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        kb_ids: Optional[List[str]] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RETRIEVAL,
        max_results: int = 10,
        min_relevance: float = 0.3
    ) -> FFRAGRetrievalResultDTO:
        """Retrieve relevant context from knowledge bases."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{user_id}_{hash(query)}_{hash(str(kb_ids))}_{retrieval_strategy.value}"
            if self._is_cache_valid(cache_key):
                cached_result = self._retrieval_cache[cache_key]
                self.logger.debug(f"Retrieved cached results for query: {query[:50]}...")
                return cached_result
            
            # Get knowledge bases
            if kb_ids is None:
                user_kbs = await self.get_user_knowledge_bases(user_id)
                kb_ids = [kb.kb_id for kb in user_kbs if kb.is_active]
            
            if not kb_ids:
                return FFRAGRetrievalResultDTO(
                    query=query,
                    retrieval_strategy=retrieval_strategy.value,
                    contexts=[],
                    total_results=0
                )
            
            # Perform retrieval based on strategy
            all_contexts = []
            
            if retrieval_strategy == RetrievalStrategy.SEMANTIC_SIMILARITY:
                all_contexts = await self._semantic_retrieval(user_id, query, kb_ids, max_results * 2)
            elif retrieval_strategy == RetrievalStrategy.KEYWORD_MATCHING:
                all_contexts = await self._keyword_retrieval(user_id, query, kb_ids, max_results * 2)
            elif retrieval_strategy == RetrievalStrategy.HYBRID_RETRIEVAL:
                semantic_contexts = await self._semantic_retrieval(user_id, query, kb_ids, max_results)
                keyword_contexts = await self._keyword_retrieval(user_id, query, kb_ids, max_results)
                all_contexts = self._merge_retrieval_results(semantic_contexts, keyword_contexts)
            elif retrieval_strategy == RetrievalStrategy.CONTEXTUAL_RELEVANCE:
                all_contexts = await self._contextual_retrieval(user_id, query, kb_ids, max_results * 2)
            
            # Filter by relevance threshold
            filtered_contexts = [
                ctx for ctx in all_contexts 
                if ctx.relevance_score >= min_relevance
            ]
            
            # Sort by relevance and limit results
            filtered_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
            final_contexts = filtered_contexts[:max_results]
            
            # Create result
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            result = FFRAGRetrievalResultDTO(
                query=query,
                retrieval_strategy=retrieval_strategy.value,
                contexts=final_contexts,
                total_results=len(final_contexts),
                retrieval_time_ms=retrieval_time
            )
            
            result.calculate_quality_metrics()
            
            # Cache result
            if self.kb_config.enable_retrieval_cache:
                self._retrieval_cache[cache_key] = result
                self._cache_timestamps[cache_key] = datetime.now()
            
            # Update analytics
            await self._update_retrieval_analytics(user_id, kb_ids, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve context for query '{query}': {e}")
            return FFRAGRetrievalResultDTO(
                query=query,
                retrieval_strategy=retrieval_strategy.value,
                contexts=[],
                total_results=0
            )
    
    async def get_user_knowledge_bases(self, user_id: str) -> List[FFKnowledgeBaseDTO]:
        """Get all knowledge bases for user."""
        try:
            # Check cache first
            if user_id in self._kb_registry_cache:
                return list(self._kb_registry_cache[user_id].values())
            
            # Load from file system
            user_kb_path = self._get_user_kb_path(user_id)
            if not user_kb_path.exists():
                return []
            
            kbs = []
            for kb_dir in user_kb_path.iterdir():
                if kb_dir.is_dir():
                    kb_config_file = kb_dir / "kb_config.json"
                    if kb_config_file.exists():
                        try:
                            kb_data = await ff_read_json(kb_config_file, self.config)
                            kb = FFKnowledgeBaseDTO.from_dict(kb_data)
                            kbs.append(kb)
                        except Exception as e:
                            self.logger.warning(f"Failed to load KB config from {kb_dir}: {e}")
            
            # Update cache
            self._kb_registry_cache[user_id] = {kb.kb_id: kb for kb in kbs}
            
            return kbs
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge bases for user {user_id}: {e}")
            return []
    
    async def get_knowledge_base(self, user_id: str, kb_id: str) -> Optional[FFKnowledgeBaseDTO]:
        """Get specific knowledge base."""
        try:
            user_kbs = await self.get_user_knowledge_bases(user_id)
            for kb in user_kbs:
                if kb.kb_id == kb_id:
                    return kb
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base {kb_id} for user {user_id}: {e}")
            return None
    
    async def get_kb_analytics(self, user_id: str, kb_id: str) -> Optional[FFKnowledgeBaseAnalyticsDTO]:
        """Get knowledge base analytics."""
        try:
            kb_path = self._get_kb_path(user_id, kb_id)
            analytics_file = kb_path / "kb_analytics.json"
            
            if analytics_file.exists():
                analytics_data = await ff_read_json(analytics_file, self.config)
                return FFKnowledgeBaseAnalyticsDTO.from_dict(analytics_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics for KB {kb_id}: {e}")
            return None
    
    async def run_kb_maintenance(self, user_id: str, kb_id: str) -> Dict[str, Any]:
        """Run comprehensive knowledge base maintenance."""
        try:
            maintenance_results = {
                "kb_id": kb_id,
                "start_time": datetime.now().isoformat(),
                "tasks_completed": [],
                "errors": []
            }
            
            kb = await self.get_knowledge_base(user_id, kb_id)
            if not kb:
                return {"success": False, "error": "Knowledge base not found"}
            
            # 1. Clean retrieval cache
            try:
                await self._clean_retrieval_cache(user_id, kb_id)
                maintenance_results["tasks_completed"].append("cleaned_retrieval_cache")
            except Exception as e:
                maintenance_results["errors"].append(f"clean_cache: {str(e)}")
            
            # 2. Update embeddings for modified documents
            try:
                updated_count = await self._update_stale_embeddings(user_id, kb_id)
                maintenance_results["tasks_completed"].append(f"updated_{updated_count}_embeddings")
            except Exception as e:
                maintenance_results["errors"].append(f"update_embeddings: {str(e)}")
            
            # 3. Detect and remove duplicate content
            try:
                if self.kb_config.enable_duplicate_detection:
                    removed_count = await self._remove_duplicate_content(user_id, kb_id)
                    maintenance_results["tasks_completed"].append(f"removed_{removed_count}_duplicates")
            except Exception as e:
                maintenance_results["errors"].append(f"remove_duplicates: {str(e)}")
            
            # 4. Update analytics
            try:
                await self._generate_kb_analytics(user_id, kb_id)
                maintenance_results["tasks_completed"].append("updated_analytics")
            except Exception as e:
                maintenance_results["errors"].append(f"update_analytics: {str(e)}")
            
            # 5. Optimize vector indices
            try:
                await self._optimize_vector_indices(user_id, kb_id)
                maintenance_results["tasks_completed"].append("optimized_indices")
            except Exception as e:
                maintenance_results["errors"].append(f"optimize_indices: {str(e)}")
            
            # Update KB maintenance timestamp
            kb.last_maintenance = datetime.now().isoformat()
            await self._save_knowledge_base(kb)
            
            maintenance_results["end_time"] = datetime.now().isoformat()
            maintenance_results["success"] = len(maintenance_results["errors"]) == 0
            
            return maintenance_results
            
        except Exception as e:
            self.logger.error(f"KB maintenance failed for {kb_id}: {e}")
            return {"success": False, "error": str(e)}
    
    # Private helper methods
    
    async def _chunk_document(self, content: str, chunking_config: Dict[str, Any]) -> List[str]:
        """Chunk document content based on configuration."""
        strategy = chunking_config.get('strategy', self.kb_config.chunking_strategy.value)
        chunk_size = chunking_config.get('chunk_size', self.kb_config.chunk_size)
        chunk_overlap = chunking_config.get('chunk_overlap', self.kb_config.chunk_overlap)
        
        if strategy == ChunkingStrategy.FIXED_SIZE.value:
            return self._fixed_size_chunking(content, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SENTENCE_BOUNDARY.value:
            return self._sentence_boundary_chunking(content, chunk_size)
        elif strategy == ChunkingStrategy.SEMANTIC_CHUNKING.value:
            return await self._semantic_chunking(content, chunk_size)
        else:
            # Default to fixed size
            return self._fixed_size_chunking(content, chunk_size, chunk_overlap)
    
    def _fixed_size_chunking(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Split content into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end].strip()
            
            if chunk and len(chunk) >= self.kb_config.min_chunk_size:
                chunks.append(chunk)
            
            if end >= len(content):
                break
                
            start = end - overlap
        
        return chunks
    
    def _sentence_boundary_chunking(self, content: str, target_size: int) -> List[str]:
        """Split content at sentence boundaries while respecting target size."""
        # Simple sentence splitting (could be enhanced with NLP)
        sentences = content.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= target_size:
                current_chunk += ". " + sentence
            else:
                if current_chunk and len(current_chunk) >= self.kb_config.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.kb_config.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _semantic_chunking(self, content: str, target_size: int) -> List[str]:
        """Advanced semantic chunking (placeholder for more sophisticated implementation)."""
        # For now, fall back to sentence boundary chunking
        # In a full implementation, this would use semantic similarity
        # to group related sentences together
        return self._sentence_boundary_chunking(content, target_size)
    
    async def _semantic_retrieval(
        self, 
        user_id: str, 
        query: str, 
        kb_ids: List[str], 
        max_results: int
    ) -> List[FFRAGContextDTO]:
        """Perform semantic similarity-based retrieval."""
        try:
            # Generate query embedding
            query_embeddings = await self.embedding_manager.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            all_contexts = []
            
            for kb_id in kb_ids:
                # Use vector storage manager for similarity search
                similar_chunks = await self.vector_storage.similarity_search(
                    query_embedding=query_embedding,
                    session_id=f"{user_id}_{kb_id}",
                    limit=max_results
                )
                
                # Convert to RAG context DTOs
                for chunk_data in similar_chunks:
                    context = FFRAGContextDTO(
                        kb_id=kb_id,
                        document_id=chunk_data.get('document_id', ''),
                        chunk_id=chunk_data.get('chunk_id', ''),
                        content=chunk_data.get('chunk_text', ''),
                        relevance_score=chunk_data.get('similarity_score', 0.0),
                        retrieval_method="semantic_similarity",
                        query_used=query
                    )
                    all_contexts.append(context)
            
            return all_contexts
            
        except Exception as e:
            self.logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    async def _keyword_retrieval(
        self, 
        user_id: str, 
        query: str, 
        kb_ids: List[str], 
        max_results: int
    ) -> List[FFRAGContextDTO]:
        """Perform keyword-based retrieval."""
        try:
            # Simple keyword matching implementation
            # In production, this would use more sophisticated text search
            query_words = set(query.lower().split())
            
            all_contexts = []
            
            for kb_id in kb_ids:
                # Load chunks from vector index
                kb_path = self._get_kb_path(user_id, kb_id)
                chunk_index_file = kb_path / "vectors" / "chunk_index.jsonl"
                
                if not chunk_index_file.exists():
                    continue
                
                chunk_data = await ff_read_jsonl(chunk_index_file, self.config)
                
                for chunk in chunk_data:
                    chunk_words = set(chunk.get('chunk_text', '').lower().split())
                    overlap = len(query_words.intersection(chunk_words))
                    
                    if overlap > 0:
                        relevance = overlap / len(query_words)
                        
                        context = FFRAGContextDTO(
                            kb_id=kb_id,
                            document_id=chunk.get('document_id', ''),
                            chunk_id=chunk.get('chunk_id', ''),
                            content=chunk.get('chunk_text', ''),
                            relevance_score=relevance,
                            retrieval_method="keyword_matching",
                            query_used=query
                        )
                        all_contexts.append(context)
            
            # Sort by relevance and limit
            all_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
            return all_contexts[:max_results]
            
        except Exception as e:
            self.logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    def _merge_retrieval_results(
        self, 
        semantic_results: List[FFRAGContextDTO], 
        keyword_results: List[FFRAGContextDTO]
    ) -> List[FFRAGContextDTO]:
        """Merge and deduplicate results from different retrieval methods."""
        # Create a map to avoid duplicates
        merged_contexts = {}
        
        # Add semantic results with weight
        for context in semantic_results:
            key = f"{context.kb_id}_{context.document_id}_{context.chunk_id}"
            context.relevance_score *= 0.7  # Weight semantic results
            merged_contexts[key] = context
        
        # Add keyword results with weight
        for context in keyword_results:
            key = f"{context.kb_id}_{context.document_id}_{context.chunk_id}"
            context.relevance_score *= 0.3  # Weight keyword results
            
            if key in merged_contexts:
                # Combine scores for hybrid result
                merged_contexts[key].relevance_score += context.relevance_score
            else:
                merged_contexts[key] = context
        
        return list(merged_contexts.values())
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached retrieval result is still valid."""
        if not self.kb_config.enable_retrieval_cache:
            return False
        
        if cache_key not in self._retrieval_cache:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.kb_config.cache_ttl_hours * 3600)
```

### **2. RAG Context Manager**

```python
# ff_rag_context_manager.py

"""
RAG Context Management System.

Manages conversation-specific RAG context including context injection,
usage tracking, and integration with conversation flow.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_knowledge_base_config import FFRAGSessionConfigDTO, RetrievalStrategy
from ff_class_configs.ff_chat_entities_config import FFRAGContextDTO, FFRAGRetrievalResultDTO
from ff_utils.ff_file_ops import ff_ensure_directory
from ff_utils.ff_json_utils import ff_write_json, ff_read_json, ff_append_jsonl
from ff_utils.ff_logging import get_logger
from ff_knowledge_base_manager import FFKnowledgeBaseManager

class FFRAGContextManager:
    """
    RAG context management for conversation sessions.
    
    Handles RAG context injection, tracking, and optimization for
    individual conversation sessions with knowledge base integration.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize RAG context manager."""
        self.config = config
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Initialize knowledge base manager
        self.kb_manager = FFKnowledgeBaseManager(config)
        
        # Active RAG sessions
        self.active_sessions: Dict[str, FFRAGSessionConfigDTO] = {}
        
    def _get_rag_session_path(self, user_id: str, session_id: str) -> Path:
        """Get RAG session directory path."""
        return self.base_path / "users" / user_id / "rag_sessions" / session_id
    
    async def initialize_rag_session(
        self,
        user_id: str,
        session_id: str,
        kb_ids: List[str],
        session_config: Optional[FFRAGSessionConfigDTO] = None
    ) -> bool:
        """Initialize RAG-enabled conversation session."""
        try:
            if session_config is None:
                session_config = FFRAGSessionConfigDTO(active_knowledge_bases=kb_ids)
            
            # Create session directory
            session_path = self._get_rag_session_path(user_id, session_id)
            await ff_ensure_directory(session_path)
            await ff_ensure_directory(session_path / "rag_context")
            
            # Save session configuration
            session_data = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "config": session_config.to_dict(),
                "active_knowledge_bases": kb_ids,
                "context_usage_stats": {
                    "total_retrievals": 0,
                    "contexts_used": 0,
                    "average_relevance": 0.0
                }
            }
            
            await ff_write_json(session_path / "session_config.json", session_data, self.config)
            
            # Track active session
            self.active_sessions[f"{user_id}_{session_id}"] = session_config
            
            self.logger.info(f"Initialized RAG session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG session {session_id}: {e}")
            return False
    
    async def get_rag_context_for_message(
        self,
        user_id: str,
        session_id: str,
        message_content: str,
        message_id: str,
        max_context_chunks: Optional[int] = None
    ) -> List[FFRAGContextDTO]:
        """Get RAG context for specific message."""
        try:
            session_key = f"{user_id}_{session_id}"
            
            if session_key not in self.active_sessions:
                # Try to load session
                if not await self._load_rag_session(user_id, session_id):
                    return []
            
            session_config = self.active_sessions[session_key]
            
            if not session_config.enable_rag or not session_config.active_knowledge_bases:
                return []
            
            # Determine max context chunks
            if max_context_chunks is None:
                max_context_chunks = session_config.max_context_per_message
            
            # Retrieve relevant context
            retrieval_result = await self.kb_manager.retrieve_context(
                user_id=user_id,
                query=message_content,
                kb_ids=session_config.active_knowledge_bases,
                retrieval_strategy=session_config.retrieval_strategy,
                max_results=max_context_chunks * 2,  # Get extra for filtering
                min_relevance=session_config.context_quality_threshold
            )
            
            # Filter and select best contexts
            selected_contexts = retrieval_result.contexts[:max_context_chunks]
            
            # Save context for this message
            if selected_contexts:
                await self._save_message_context(
                    user_id, session_id, message_id, selected_contexts, retrieval_result
                )
            
            # Update session statistics
            await self._update_session_stats(user_id, session_id, retrieval_result)
            
            return selected_contexts
            
        except Exception as e:
            self.logger.error(f"Failed to get RAG context for message {message_id}: {e}")
            return []
    
    async def format_context_for_injection(
        self,
        contexts: List[FFRAGContextDTO],
        injection_mode: str = "automatic",
        include_citations: bool = True
    ) -> str:
        """Format RAG context for injection into conversation."""
        try:
            if not contexts:
                return ""
            
            if injection_mode == "hidden":
                # Context is used but not shown to user
                formatted_contexts = []
                for i, context in enumerate(contexts):
                    formatted_contexts.append(f"[Context {i+1}]: {context.content}")
                return "\n\n".join(formatted_contexts)
            
            elif injection_mode == "visible":
                # Context is shown to user with formatting
                formatted_contexts = []
                for i, context in enumerate(contexts):
                    citation = ""
                    if include_citations and context.document_title:
                        citation = f" (Source: {context.document_title})"
                    
                    formatted_contexts.append(
                        f"**Relevant Information {i+1}**{citation}:\n{context.content}"
                    )
                return "\n\n" + "\n\n".join(formatted_contexts) + "\n\n"
            
            else:  # "optional" or other modes
                # Provide context in a way that can be optionally shown
                formatted_contexts = []
                for context in contexts:
                    formatted_contexts.append(context.content)
                return "\n".join(formatted_contexts)
            
        except Exception as e:
            self.logger.error(f"Failed to format context for injection: {e}")
            return ""
    
    async def track_context_usage(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        used_contexts: List[str],  # Context IDs that were actually used
        user_feedback: Optional[str] = None
    ) -> None:
        """Track which contexts were actually used in response generation."""
        try:
            # Load message context
            message_contexts = await self._load_message_context(user_id, session_id, message_id)
            
            if not message_contexts:
                return
            
            # Mark contexts as used and apply feedback
            for context in message_contexts:
                if context.context_id in used_contexts:
                    context.mark_as_used()
                    if user_feedback:
                        context.user_feedback = user_feedback
            
            # Save updated context
            await self._save_message_context(
                user_id, session_id, message_id, message_contexts, None
            )
            
            # Update knowledge base analytics
            for context in message_contexts:
                if context.context_id in used_contexts:
                    await self._update_kb_usage_analytics(
                        user_id, context.kb_id, context.relevance_score, user_feedback
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to track context usage for message {message_id}: {e}")
    
    async def get_session_rag_statistics(
        self, 
        user_id: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Get RAG usage statistics for session."""
        try:
            session_path = self._get_rag_session_path(user_id, session_id)
            stats_file = session_path / "rag_context" / "context_usage_stats.json"
            
            if stats_file.exists():
                return await ff_read_json(stats_file, self.config)
            
            return {
                "total_retrievals": 0,
                "contexts_used": 0,
                "average_relevance": 0.0,
                "most_used_knowledge_bases": [],
                "context_effectiveness": 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get RAG statistics for session {session_id}: {e}")
            return {}
    
    # Private helper methods
    
    async def _load_rag_session(self, user_id: str, session_id: str) -> bool:
        """Load existing RAG session configuration."""
        try:
            session_path = self._get_rag_session_path(user_id, session_id)
            config_file = session_path / "session_config.json"
            
            if not config_file.exists():
                return False
            
            session_data = await ff_read_json(config_file, self.config)
            session_config = FFRAGSessionConfigDTO.from_dict(session_data.get('config', {}))
            
            session_key = f"{user_id}_{session_id}"
            self.active_sessions[session_key] = session_config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load RAG session {session_id}: {e}")
            return False
    
    async def _save_message_context(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        contexts: List[FFRAGContextDTO],
        retrieval_result: Optional[FFRAGRetrievalResultDTO]
    ) -> None:
        """Save RAG context for specific message."""
        try:
            session_path = self._get_rag_session_path(user_id, session_id)
            context_file = session_path / "rag_context" / f"msg_{message_id}_context.json"
            
            context_data = {
                "message_id": message_id,
                "timestamp": datetime.now().isoformat(),
                "contexts": [ctx.to_dict() for ctx in contexts],
                "retrieval_metadata": retrieval_result.to_dict() if retrieval_result else None
            }
            
            await ff_write_json(context_file, context_data, self.config)
            
        except Exception as e:
            self.logger.error(f"Failed to save context for message {message_id}: {e}")
    
    async def _load_message_context(
        self, 
        user_id: str, 
        session_id: str, 
        message_id: str
    ) -> List[FFRAGContextDTO]:
        """Load RAG context for specific message."""
        try:
            session_path = self._get_rag_session_path(user_id, session_id)
            context_file = session_path / "rag_context" / f"msg_{message_id}_context.json"
            
            if not context_file.exists():
                return []
            
            context_data = await ff_read_json(context_file, self.config)
            contexts = [
                FFRAGContextDTO.from_dict(ctx_data) 
                for ctx_data in context_data.get('contexts', [])
            ]
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Failed to load context for message {message_id}: {e}")
            return []
```

### **3. RAG Integration Protocol**

```python
# ff_protocols/ff_knowledge_base_protocol.py

"""Protocol interface for knowledge base and RAG operations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ff_class_configs.ff_knowledge_base_config import KnowledgeBaseType, RetrievalStrategy
from ff_class_configs.ff_chat_entities_config import (
    FFKnowledgeBaseDTO, FFRAGContextDTO, FFRAGRetrievalResultDTO, FFKnowledgeBaseAnalyticsDTO
)

class KnowledgeBaseProtocol(ABC):
    """Protocol interface for knowledge base management operations."""
    
    @abstractmethod
    async def create_knowledge_base(
        self,
        user_id: str,
        name: str,
        description: str = "",
        kb_type: KnowledgeBaseType = KnowledgeBaseType.PERSONAL_DOCS,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[FFKnowledgeBaseDTO]:
        """Create new knowledge base for user."""
        pass
    
    @abstractmethod
    async def add_document_to_kb(
        self,
        user_id: str,
        kb_id: str,
        document_path: str,
        document_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document to knowledge base with processing and indexing."""
        pass
    
    @abstractmethod
    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        kb_ids: Optional[List[str]] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RETRIEVAL,
        max_results: int = 10,
        min_relevance: float = 0.3
    ) -> FFRAGRetrievalResultDTO:
        """Retrieve relevant context from knowledge bases."""
        pass
    
    @abstractmethod
    async def get_user_knowledge_bases(self, user_id: str) -> List[FFKnowledgeBaseDTO]:
        """Get all knowledge bases for user."""
        pass
    
    @abstractmethod
    async def get_knowledge_base(self, user_id: str, kb_id: str) -> Optional[FFKnowledgeBaseDTO]:
        """Get specific knowledge base."""
        pass
    
    @abstractmethod
    async def get_kb_analytics(self, user_id: str, kb_id: str) -> Optional[FFKnowledgeBaseAnalyticsDTO]:
        """Get knowledge base analytics."""
        pass
    
    @abstractmethod
    async def run_kb_maintenance(self, user_id: str, kb_id: str) -> Dict[str, Any]:
        """Run comprehensive knowledge base maintenance."""
        pass

class RAGContextProtocol(ABC):
    """Protocol interface for RAG context management operations."""
    
    @abstractmethod
    async def initialize_rag_session(
        self,
        user_id: str,
        session_id: str,
        kb_ids: List[str],
        session_config: Optional[Any] = None
    ) -> bool:
        """Initialize RAG-enabled conversation session."""
        pass
    
    @abstractmethod
    async def get_rag_context_for_message(
        self,
        user_id: str,
        session_id: str,
        message_content: str,
        message_id: str,
        max_context_chunks: Optional[int] = None
    ) -> List[FFRAGContextDTO]:
        """Get RAG context for specific message."""
        pass
    
    @abstractmethod
    async def format_context_for_injection(
        self,
        contexts: List[FFRAGContextDTO],
        injection_mode: str = "automatic",
        include_citations: bool = True
    ) -> str:
        """Format RAG context for injection into conversation."""
        pass
    
    @abstractmethod
    async def track_context_usage(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        used_contexts: List[str],
        user_feedback: Optional[str] = None
    ) -> None:
        """Track which contexts were actually used in response generation."""
        pass
    
    @abstractmethod
    async def get_session_rag_statistics(
        self, 
        user_id: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Get RAG usage statistics for session."""
        pass
```

## 🧪 Testing Specifications

### **Unit Tests**

```python
# tests/test_knowledge_base_manager.py

import pytest
import tempfile
from pathlib import Path

from ff_knowledge_base_manager import FFKnowledgeBaseManager
from ff_class_configs.ff_knowledge_base_config import FFKnowledgeBaseConfigDTO, KnowledgeBaseType, RetrievalStrategy

class TestKnowledgeBaseManager:
    
    @pytest.fixture
    async def kb_manager(self):
        """Create knowledge base manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FFConfigurationManagerConfigDTO()
            config.storage.base_path = temp_dir
            config.knowledge_base = FFKnowledgeBaseConfigDTO()
            
            manager = FFKnowledgeBaseManager(config)
            yield manager
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base(self, kb_manager):
        """Test knowledge base creation."""
        user_id = "test_user"
        
        kb = await kb_manager.create_knowledge_base(
            user_id=user_id,
            name="Test KB",
            description="Test knowledge base",
            kb_type=KnowledgeBaseType.PERSONAL_DOCS
        )
        
        assert kb is not None
        assert kb.user_id == user_id
        assert kb.name == "Test KB"
        assert kb.kb_type == KnowledgeBaseType.PERSONAL_DOCS
        
        # Check directory structure
        kb_path = kb_manager._get_kb_path(user_id, kb.kb_id)
        assert kb_path.exists()
        assert (kb_path / "documents").exists()
        assert (kb_path / "vectors").exists()
        assert (kb_path / "retrieval_cache").exists()
    
    @pytest.mark.asyncio
    async def test_add_document_to_kb(self, kb_manager):
        """Test adding document to knowledge base."""
        user_id = "test_user"
        
        # Create KB
        kb = await kb_manager.create_knowledge_base(
            user_id, "Test KB", "Test KB"
        )
        
        # Add document
        document_content = """
        This is a test document about artificial intelligence.
        It covers machine learning, neural networks, and deep learning.
        The document is designed to test the knowledge base functionality.
        """
        
        success = await kb_manager.add_document_to_kb(
            user_id=user_id,
            kb_id=kb.kb_id,
            document_path="test_doc.txt",
            document_content=document_content,
            metadata={"topic": "AI", "type": "test"}
        )
        
        assert success
        
        # Verify KB statistics updated
        updated_kb = await kb_manager.get_knowledge_base(user_id, kb.kb_id)
        assert updated_kb.document_count == 1
        assert updated_kb.total_chunks > 0
        assert updated_kb.total_size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_context(self, kb_manager):
        """Test context retrieval from knowledge base."""
        user_id = "test_user"
        
        # Create KB and add document
        kb = await kb_manager.create_knowledge_base(user_id, "AI KB", "AI Knowledge")
        
        await kb_manager.add_document_to_kb(
            user_id, kb.kb_id, "ai_doc.txt",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms."
        )
        
        # Test retrieval
        result = await kb_manager.retrieve_context(
            user_id=user_id,
            query="What is machine learning?",
            kb_ids=[kb.kb_id],
            retrieval_strategy=RetrievalStrategy.HYBRID_RETRIEVAL,
            max_results=5
        )
        
        assert result is not None
        assert result.query == "What is machine learning?"
        assert len(result.contexts) > 0
        assert result.total_results > 0
        assert result.retrieval_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_kb_maintenance(self, kb_manager):
        """Test knowledge base maintenance."""
        user_id = "test_user"
        
        # Create KB with some content
        kb = await kb_manager.create_knowledge_base(user_id, "Maintenance KB", "Test KB")
        
        await kb_manager.add_document_to_kb(
            user_id, kb.kb_id, "doc1.txt", "Test document content for maintenance."
        )
        
        # Run maintenance
        result = await kb_manager.run_kb_maintenance(user_id, kb.kb_id)
        
        assert result["success"] is True
        assert len(result["tasks_completed"]) > 0
        assert len(result["errors"]) == 0
```

### **Integration Tests**

```python
# tests/test_rag_integration.py

class TestRAGIntegration:
    
    @pytest.mark.asyncio
    async def test_rag_session_workflow(self):
        """Test complete RAG session workflow."""
        # Test RAG session creation, context retrieval, and usage tracking
        pass
    
    @pytest.mark.asyncio
    async def test_memory_layer_integration(self):
        """Test integration between RAG and memory layers."""
        # Test that RAG contexts feed into memory layers appropriately
        pass
    
    @pytest.mark.asyncio
    async def test_cross_kb_retrieval(self):
        """Test retrieval across multiple knowledge bases."""
        # Test that contexts from multiple KBs are properly merged and ranked
        pass
```

## 📈 Success Criteria

### **Functional Requirements**
- ✅ Personal knowledge bases support all document types and processing strategies
- ✅ Intelligent retrieval provides relevant context with multiple strategies
- ✅ RAG context management integrates seamlessly with conversation flow
- ✅ Knowledge base maintenance optimizes performance and removes duplicates
- ✅ Integration with existing vector storage and memory systems

### **Performance Requirements**
- ✅ Document indexing completes within 30 seconds for typical documents
- ✅ Context retrieval returns results within 5 seconds for complex queries
- ✅ Knowledge base operations scale to configured limits (1000 docs/KB)
- ✅ Memory usage remains reasonable for large knowledge bases

### **Integration Requirements**
- ✅ Existing vector storage functionality enhanced, not replaced
- ✅ RAG contexts integrate with memory layers for long-term learning
- ✅ All operations follow existing async patterns and error handling
- ✅ Configuration-driven behavior with comprehensive settings

### **Testing Requirements**
- ✅ Unit test coverage > 90% for all new components
- ✅ Integration tests validate RAG workflow end-to-end
- ✅ Performance tests validate retrieval speed and accuracy
- ✅ Load tests validate concurrent KB operations

## 🚀 Implementation Checklist

### **Phase 3A: Core Knowledge Base Management**
- [ ] Create `ff_knowledge_base_config.py` with comprehensive configuration DTOs
- [ ] Extend `ff_chat_entities_config.py` with RAG data models
- [ ] Create `ff_knowledge_base_protocol.py` with abstract interfaces
- [ ] Update configuration manager to include KB config

### **Phase 3B: Knowledge Base Manager**
- [ ] Implement `FFKnowledgeBaseManager` with full lifecycle management
- [ ] Add document processing with multiple chunking strategies
- [ ] Implement intelligent retrieval with multiple strategies
- [ ] Add comprehensive maintenance and optimization

### **Phase 3C: RAG Context Management**
- [ ] Implement `FFRAGContextManager` for session-specific context
- [ ] Add context injection and formatting capabilities
- [ ] Create context usage tracking and analytics
- [ ] Implement integration with conversation flow

### **Phase 3D: Integration & Testing**
- [ ] Integrate with existing vector storage and embedding systems
- [ ] Update dependency injection container registration
- [ ] Create comprehensive unit test suite
- [ ] Create integration tests with memory layers and conversation management
- [ ] Performance test knowledge base operations
- [ ] Validate backward compatibility

### **Phase 3E: Documentation & Validation**
- [ ] Update RAG system documentation
- [ ] Create knowledge base management usage examples
- [ ] Validate all success criteria met
- [ ] Performance benchmark RAG operations
- [ ] Create migration guide for existing vector data

This comprehensive specification provides everything needed to implement sophisticated RAG capabilities while maintaining your excellent architectural standards and ensuring full integration with existing systems.