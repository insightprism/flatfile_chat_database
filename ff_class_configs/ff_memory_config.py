"""
FF Memory Component Configuration

Configuration for FF memory component following existing FF configuration 
patterns and integrating with FF vector storage for memory management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from .ff_base_config import FFBaseConfigDTO, validate_positive, validate_range


class FFMemoryType(str, Enum):
    """Types of memory supported by FF memory component"""
    WORKING = "working"          # Current session context
    EPISODIC = "episodic"        # Conversation episodes and events
    SEMANTIC = "semantic"        # Knowledge and facts
    PROCEDURAL = "procedural"    # Learned behaviors and patterns


class FFMemoryRetentionPolicy(str, Enum):
    """Memory retention policies"""
    PERMANENT = "permanent"      # Never delete
    TIME_BASED = "time_based"    # Delete after time period
    COUNT_BASED = "count_based"  # Keep only N most recent
    RELEVANCE_BASED = "relevance_based"  # Keep most relevant


@dataclass
class FFMemoryConfigDTO(FFBaseConfigDTO):
    """FF Memory component configuration following FF patterns"""
    
    # Memory type settings
    enabled_memory_types: List[str] = field(default_factory=lambda: ["working", "episodic", "semantic"])
    
    # Working memory settings (current session)
    working_memory_size: int = 20
    working_memory_timeout: int = 3600  # seconds
    working_memory_cleanup_interval: int = 300  # seconds
    
    # Episodic memory settings (conversation history)
    episodic_retention_days: int = 30
    episodic_max_entries: int = 1000
    episodic_retention_policy: str = FFMemoryRetentionPolicy.TIME_BASED.value
    
    # Semantic memory settings (knowledge extraction)
    semantic_similarity_threshold: float = 0.7
    semantic_max_entries: int = 500
    semantic_extraction_enabled: bool = True
    semantic_consolidation_threshold: int = 10  # messages before consolidation
    
    # Procedural memory settings (behavior patterns)
    procedural_enabled: bool = False
    procedural_learning_threshold: int = 5  # repetitions to form pattern
    procedural_max_patterns: int = 100
    
    # FF vector storage integration settings
    use_ff_vector_storage: bool = True
    embedding_dimension: int = 384
    similarity_metric: str = "cosine"
    vector_index_type: str = "hnsw"
    
    # Memory retrieval settings
    retrieval_similarity_threshold: float = 0.6
    max_retrieved_memories: int = 5
    retrieval_diversity_factor: float = 0.3  # 0.0 = pure similarity, 1.0 = pure diversity
    
    # Memory consolidation settings
    enable_memory_consolidation: bool = True
    consolidation_interval_hours: int = 24
    consolidation_similarity_threshold: float = 0.8
    
    # Performance settings
    enable_memory_caching: bool = True
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 1800  # seconds
    
    # Privacy and security settings
    enable_memory_encryption: bool = False
    memory_anonymization_level: str = "none"  # none, partial, full
    sensitive_data_filter: bool = True
    
    def validate(self) -> List[str]:
        """Validate memory configuration"""
        errors = []
        
        # Validate enabled memory types
        valid_types = [t.value for t in FFMemoryType]
        for memory_type in self.enabled_memory_types:
            if memory_type not in valid_types:
                errors.append(f"Invalid memory type '{memory_type}'. Must be one of {valid_types}")
        
        if not self.enabled_memory_types:
            errors.append("At least one memory type must be enabled")
        
        # Validate working memory settings
        error = validate_positive(self.working_memory_size, "working_memory_size")
        if error:
            errors.append(error)
            
        error = validate_positive(self.working_memory_timeout, "working_memory_timeout")
        if error:
            errors.append(error)
            
        error = validate_positive(self.working_memory_cleanup_interval, "working_memory_cleanup_interval")
        if error:
            errors.append(error)
        
        # Validate episodic memory settings
        error = validate_positive(self.episodic_retention_days, "episodic_retention_days")
        if error:
            errors.append(error)
            
        error = validate_positive(self.episodic_max_entries, "episodic_max_entries")
        if error:
            errors.append(error)
            
        valid_policies = [p.value for p in FFMemoryRetentionPolicy]
        if self.episodic_retention_policy not in valid_policies:
            errors.append(f"episodic_retention_policy must be one of {valid_policies}")
        
        # Validate semantic memory settings
        error = validate_range(self.semantic_similarity_threshold, 0.0, 1.0, "semantic_similarity_threshold")
        if error:
            errors.append(error)
            
        error = validate_positive(self.semantic_max_entries, "semantic_max_entries")
        if error:
            errors.append(error)
            
        error = validate_positive(self.semantic_consolidation_threshold, "semantic_consolidation_threshold")
        if error:
            errors.append(error)
        
        # Validate procedural memory settings
        if self.procedural_enabled:
            error = validate_positive(self.procedural_learning_threshold, "procedural_learning_threshold")
            if error:
                errors.append(error)
                
            error = validate_positive(self.procedural_max_patterns, "procedural_max_patterns")
            if error:
                errors.append(error)
        
        # Validate vector storage settings
        if self.use_ff_vector_storage:
            error = validate_positive(self.embedding_dimension, "embedding_dimension")
            if error:
                errors.append(error)
                
            valid_metrics = ["cosine", "euclidean", "dot_product"]
            if self.similarity_metric not in valid_metrics:
                errors.append(f"similarity_metric must be one of {valid_metrics}")
                
            valid_index_types = ["hnsw", "flat", "ivf"]
            if self.vector_index_type not in valid_index_types:
                errors.append(f"vector_index_type must be one of {valid_index_types}")
        
        # Validate retrieval settings
        error = validate_range(self.retrieval_similarity_threshold, 0.0, 1.0, "retrieval_similarity_threshold")
        if error:
            errors.append(error)
            
        error = validate_positive(self.max_retrieved_memories, "max_retrieved_memories")
        if error:
            errors.append(error)
            
        error = validate_range(self.retrieval_diversity_factor, 0.0, 1.0, "retrieval_diversity_factor")
        if error:
            errors.append(error)
        
        # Validate consolidation settings
        if self.enable_memory_consolidation:
            error = validate_positive(self.consolidation_interval_hours, "consolidation_interval_hours")
            if error:
                errors.append(error)
                
            error = validate_range(self.consolidation_similarity_threshold, 0.0, 1.0, "consolidation_similarity_threshold")
            if error:
                errors.append(error)
        
        # Validate cache settings
        if self.enable_memory_caching:
            error = validate_positive(self.memory_cache_size, "memory_cache_size")
            if error:
                errors.append(error)
                
            error = validate_positive(self.memory_cache_ttl, "memory_cache_ttl")
            if error:
                errors.append(error)
        
        # Validate privacy settings
        valid_anonymization = ["none", "partial", "full"]
        if self.memory_anonymization_level not in valid_anonymization:
            errors.append(f"memory_anonymization_level must be one of {valid_anonymization}")
        
        return errors
    
    def get_working_memory_config(self) -> Dict[str, Any]:
        """Get working memory configuration"""
        return {
            "enabled": FFMemoryType.WORKING.value in self.enabled_memory_types,
            "size": self.working_memory_size,
            "timeout": self.working_memory_timeout,
            "cleanup_interval": self.working_memory_cleanup_interval
        }
    
    def get_episodic_memory_config(self) -> Dict[str, Any]:
        """Get episodic memory configuration"""
        return {
            "enabled": FFMemoryType.EPISODIC.value in self.enabled_memory_types,
            "retention_days": self.episodic_retention_days,
            "max_entries": self.episodic_max_entries,
            "retention_policy": self.episodic_retention_policy
        }
    
    def get_semantic_memory_config(self) -> Dict[str, Any]:
        """Get semantic memory configuration"""
        return {
            "enabled": FFMemoryType.SEMANTIC.value in self.enabled_memory_types,
            "similarity_threshold": self.semantic_similarity_threshold,
            "max_entries": self.semantic_max_entries,
            "extraction_enabled": self.semantic_extraction_enabled,
            "consolidation_threshold": self.semantic_consolidation_threshold
        }
    
    def get_procedural_memory_config(self) -> Dict[str, Any]:
        """Get procedural memory configuration"""
        return {
            "enabled": self.procedural_enabled and FFMemoryType.PROCEDURAL.value in self.enabled_memory_types,
            "learning_threshold": self.procedural_learning_threshold,
            "max_patterns": self.procedural_max_patterns
        }
    
    def get_vector_storage_config(self) -> Dict[str, Any]:
        """Get FF vector storage configuration"""
        return {
            "enabled": self.use_ff_vector_storage,
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": self.similarity_metric,
            "vector_index_type": self.vector_index_type
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get memory retrieval configuration"""
        return {
            "similarity_threshold": self.retrieval_similarity_threshold,
            "max_retrieved_memories": self.max_retrieved_memories,
            "diversity_factor": self.retrieval_diversity_factor
        }
    
    def get_consolidation_config(self) -> Dict[str, Any]:
        """Get memory consolidation configuration"""
        return {
            "enabled": self.enable_memory_consolidation,
            "interval_hours": self.consolidation_interval_hours,
            "similarity_threshold": self.consolidation_similarity_threshold
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "caching_enabled": self.enable_memory_caching,
            "cache_size": self.memory_cache_size,
            "cache_ttl": self.memory_cache_ttl
        }
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy and security configuration"""
        return {
            "encryption_enabled": self.enable_memory_encryption,
            "anonymization_level": self.memory_anonymization_level,
            "sensitive_data_filter": self.sensitive_data_filter
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()
        
        # Additional post-initialization validation
        if hasattr(self, '_post_init_done'):
            return
        self._post_init_done = True
        
        # Ensure at least working memory is enabled
        if not self.enabled_memory_types:
            self.enabled_memory_types = [FFMemoryType.WORKING.value]
        
        # Disable procedural memory if not explicitly in enabled types
        if FFMemoryType.PROCEDURAL.value not in self.enabled_memory_types:
            self.procedural_enabled = False
        
        # Adjust thresholds for consistency
        if self.retrieval_similarity_threshold > self.semantic_similarity_threshold:
            self.retrieval_similarity_threshold = self.semantic_similarity_threshold
        
        if self.consolidation_similarity_threshold < self.semantic_similarity_threshold:
            self.consolidation_similarity_threshold = self.semantic_similarity_threshold


@dataclass
class FFMemoryUseCaseConfigDTO(FFBaseConfigDTO):
    """Configuration for memory usage in specific use cases"""
    
    # Use case specific settings
    use_case_name: str = "memory_chat"
    memory_importance: str = "high"  # low, medium, high, critical
    
    # Memory type preferences for the use case
    preferred_memory_types: List[str] = field(default_factory=lambda: ["working", "episodic"])
    memory_retrieval_strategy: str = "hybrid"  # similarity, recency, hybrid, relevance
    
    # Context integration settings
    memory_context_weight: float = 0.7  # How much to weight memory vs current context
    max_memory_context_items: int = 5
    memory_context_decay: float = 0.9  # How much to reduce weight over time
    
    # Use case specific thresholds
    custom_similarity_threshold: Optional[float] = None
    custom_retrieval_limit: Optional[int] = None
    custom_consolidation_settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate memory use case configuration"""
        errors = []
        
        # Validate use case name
        if not self.use_case_name or not self.use_case_name.strip():
            errors.append("use_case_name cannot be empty")
        
        # Validate memory importance
        valid_importance = ["low", "medium", "high", "critical"]
        if self.memory_importance not in valid_importance:
            errors.append(f"memory_importance must be one of {valid_importance}")
        
        # Validate preferred memory types
        valid_types = [t.value for t in FFMemoryType]
        for memory_type in self.preferred_memory_types:
            if memory_type not in valid_types:
                errors.append(f"Invalid preferred memory type '{memory_type}'. Must be one of {valid_types}")
        
        # Validate retrieval strategy
        valid_strategies = ["similarity", "recency", "hybrid", "relevance"]
        if self.memory_retrieval_strategy not in valid_strategies:
            errors.append(f"memory_retrieval_strategy must be one of {valid_strategies}")
        
        # Validate context weight
        error = validate_range(self.memory_context_weight, 0.0, 1.0, "memory_context_weight")
        if error:
            errors.append(error)
        
        # Validate max context items
        error = validate_positive(self.max_memory_context_items, "max_memory_context_items")
        if error:
            errors.append(error)
        
        # Validate context decay
        error = validate_range(self.memory_context_decay, 0.0, 1.0, "memory_context_decay")
        if error:
            errors.append(error)
        
        # Validate custom thresholds
        if self.custom_similarity_threshold is not None:
            error = validate_range(self.custom_similarity_threshold, 0.0, 1.0, "custom_similarity_threshold")
            if error:
                errors.append(error)
        
        if self.custom_retrieval_limit is not None:
            error = validate_positive(self.custom_retrieval_limit, "custom_retrieval_limit")
            if error:
                errors.append(error)
        
        return errors
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get memory retrieval configuration for this use case"""
        return {
            "strategy": self.memory_retrieval_strategy,
            "preferred_types": self.preferred_memory_types,
            "context_weight": self.memory_context_weight,
            "max_context_items": self.max_memory_context_items,
            "context_decay": self.memory_context_decay,
            "custom_similarity_threshold": self.custom_similarity_threshold,
            "custom_retrieval_limit": self.custom_retrieval_limit
        }
    
    def get_importance_config(self) -> Dict[str, Any]:
        """Get memory importance configuration"""
        return {
            "importance_level": self.memory_importance,
            "consolidation_settings": self.custom_consolidation_settings
        }