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
            "entity_patterns": self.entity_patterns,
            "jsonl_page_size": self.jsonl_page_size,
            "default_encoding": self.default_encoding,
            "temp_dir_cleanup_interval_seconds": self.temp_dir_cleanup_interval_seconds,
            "max_concurrent_operations": self.max_concurrent_operations
        }