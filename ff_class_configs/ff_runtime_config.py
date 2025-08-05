"""
Runtime configuration for dynamic behavior settings.

Centralizes all runtime constants and patterns that were previously hardcoded.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .ff_base_config import FFBaseConfigDTO


@dataclass
class FFRuntimeConfigDTO(FFBaseConfigDTO):
    """
    Runtime behavior configuration.
    
    Contains all previously hardcoded values for caching, retries, and patterns.
    """
    
    # Cache settings
    cache_size_limit: int = 100
    cache_ttl_seconds: int = 300
    
    # File operation retry settings
    file_retry_delay_ms: int = 10
    file_retry_max_delay_ms: int = 1000
    file_retry_attempts: int = 3
    file_retry_backoff_factor: float = 2.0
    
    # Health check thresholds
    disk_usage_warning_threshold: float = 90.0
    disk_usage_critical_threshold: float = 95.0
    
    # Search defaults
    default_search_limit: int = 100
    similarity_threshold: float = 0.7
    vector_search_top_k: int = 10
    hybrid_search_vector_weight: float = 0.5
    
    # Compression settings
    compression_min_size_bytes: int = 1024
    compression_min_reduction_percent: float = 10.0
    compression_chunk_size_bytes: int = 1048576  # 1MB
    
    # Lock timeout settings
    default_lock_timeout_seconds: float = 30.0
    
    # Protocol method defaults
    vector_search_default_limit: int = 100
    document_processor_max_keywords: int = 10
    document_processor_max_summary_length: int = 500
    storage_default_session_limit: int = 50
    storage_default_message_limit: int = 100
    
    # Business logic thresholds
    large_session_threshold_bytes: int = 1_000_000  # 1MB
    large_session_warning_enabled: bool = True
    
    # File extensions and MIME types
    document_content_extension: str = ".txt"
    context_file_extension: str = ".json"
    persona_file_extension: str = ".json"
    insight_file_extension: str = ".json"
    
    # Operational settings
    path_component_min_length: int = 2  # Minimum path parts for user ID extraction
    user_id_path_index: int = 1  # Index of user ID in path parts
    enable_large_session_warnings: bool = True
    enable_empty_session_notifications: bool = True
    
    # Validation rules
    min_user_id_length: int = 1
    max_user_id_length: int = 100
    min_session_name_length: int = 1
    max_session_name_length: int = 200
    min_filename_length: int = 1
    max_filename_length: int = 255
    
    # Content validation
    min_message_content_length: int = 1
    min_document_content_length: int = 0  # Allow empty documents
    max_panel_name_length: int = 100
    max_persona_name_length: int = 100
    
    # Search entity patterns
    entity_patterns: Dict[str, str] = field(default_factory=lambda: {
        "urls": r'https?://[^\s]+',
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "code_blocks": r'```[\s\S]*?```',
        "mentions": r'@\w+',
        "hashtags": r'#\w+',
        "numbers": r'\b\d+\.?\d*\b',
        "phone_numbers": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ip_addresses": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        "file_paths": r'(?:[a-zA-Z]:)?[\\/](?:[^\\/\s]+[\\/])*[^\\/\s]+',
        "github_repos": r'(?:https?://)?github\.com/[\w-]+/[\w.-]+',
        "jira_tickets": r'\b[A-Z]{2,}-\d+\b',
        "uuids": r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    })
    
    # Additional runtime settings
    jsonl_page_size: int = 1000
    default_encoding: str = "utf-8"
    temp_dir_cleanup_interval_seconds: int = 3600
    max_concurrent_operations: int = 10
    
    def validate(self) -> list[str]:
        """
        Validate runtime configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if self.cache_size_limit < 1:
            errors.append("cache_size_limit must be at least 1")
            
        if self.file_retry_attempts < 1:
            errors.append("file_retry_attempts must be at least 1")
            
        if self.file_retry_delay_ms < 1:
            errors.append("file_retry_delay_ms must be at least 1")
            
        if self.file_retry_backoff_factor < 1.0:
            errors.append("file_retry_backoff_factor must be at least 1.0")
            
        if not (0.0 <= self.disk_usage_warning_threshold <= 100.0):
            errors.append("disk_usage_warning_threshold must be between 0 and 100")
            
        if not (0.0 <= self.disk_usage_critical_threshold <= 100.0):
            errors.append("disk_usage_critical_threshold must be between 0 and 100")
            
        if self.disk_usage_warning_threshold >= self.disk_usage_critical_threshold:
            errors.append("disk_usage_warning_threshold must be less than critical_threshold")
            
        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append("similarity_threshold must be between 0.0 and 1.0")
            
        if not (0.0 <= self.hybrid_search_vector_weight <= 1.0):
            errors.append("hybrid_search_vector_weight must be between 0.0 and 1.0")
            
        if self.compression_min_size_bytes < 0:
            errors.append("compression_min_size_bytes must be non-negative")
            
        if not (0.0 <= self.compression_min_reduction_percent <= 100.0):
            errors.append("compression_min_reduction_percent must be between 0 and 100")
        
        # Validate validation rules themselves
        if self.min_user_id_length < 0:
            errors.append("min_user_id_length must be non-negative")
        if self.max_user_id_length < self.min_user_id_length:
            errors.append("max_user_id_length must be >= min_user_id_length")
        
        if self.min_session_name_length < 0:
            errors.append("min_session_name_length must be non-negative")
        if self.max_session_name_length < self.min_session_name_length:
            errors.append("max_session_name_length must be >= min_session_name_length")
        
        if self.min_filename_length < 0:
            errors.append("min_filename_length must be non-negative")
        if self.max_filename_length < self.min_filename_length:
            errors.append("max_filename_length must be >= min_filename_length")
            
        if not self.entity_patterns:
            errors.append("entity_patterns cannot be empty")
            
        return errors
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'FFRuntimeConfigDTO':
        """Create from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "cache_size_limit": self.cache_size_limit,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "file_retry_delay_ms": self.file_retry_delay_ms,
            "file_retry_max_delay_ms": self.file_retry_max_delay_ms,
            "file_retry_attempts": self.file_retry_attempts,
            "file_retry_backoff_factor": self.file_retry_backoff_factor,
            "disk_usage_warning_threshold": self.disk_usage_warning_threshold,
            "disk_usage_critical_threshold": self.disk_usage_critical_threshold,
            "default_search_limit": self.default_search_limit,
            "similarity_threshold": self.similarity_threshold,
            "vector_search_top_k": self.vector_search_top_k,
            "hybrid_search_vector_weight": self.hybrid_search_vector_weight,
            "compression_min_size_bytes": self.compression_min_size_bytes,
            "compression_min_reduction_percent": self.compression_min_reduction_percent,
            "compression_chunk_size_bytes": self.compression_chunk_size_bytes,
            "default_lock_timeout_seconds": self.default_lock_timeout_seconds,
            "vector_search_default_limit": self.vector_search_default_limit,
            "document_processor_max_keywords": self.document_processor_max_keywords,
            "document_processor_max_summary_length": self.document_processor_max_summary_length,
            "storage_default_session_limit": self.storage_default_session_limit,
            "storage_default_message_limit": self.storage_default_message_limit,
            "large_session_threshold_bytes": self.large_session_threshold_bytes,
            "large_session_warning_enabled": self.large_session_warning_enabled,
            "document_content_extension": self.document_content_extension,
            "context_file_extension": self.context_file_extension,
            "persona_file_extension": self.persona_file_extension,
            "insight_file_extension": self.insight_file_extension,
            "path_component_min_length": self.path_component_min_length,
            "user_id_path_index": self.user_id_path_index,
            "enable_large_session_warnings": self.enable_large_session_warnings,
            "enable_empty_session_notifications": self.enable_empty_session_notifications,
            "min_user_id_length": self.min_user_id_length,
            "max_user_id_length": self.max_user_id_length,
            "min_session_name_length": self.min_session_name_length,
            "max_session_name_length": self.max_session_name_length,
            "min_filename_length": self.min_filename_length,
            "max_filename_length": self.max_filename_length,
            "min_message_content_length": self.min_message_content_length,
            "min_document_content_length": self.min_document_content_length,
            "max_panel_name_length": self.max_panel_name_length,
            "max_persona_name_length": self.max_persona_name_length,
            "entity_patterns": self.entity_patterns,
            "jsonl_page_size": self.jsonl_page_size,
            "default_encoding": self.default_encoding,
            "temp_dir_cleanup_interval_seconds": self.temp_dir_cleanup_interval_seconds,
            "max_concurrent_operations": self.max_concurrent_operations
        }