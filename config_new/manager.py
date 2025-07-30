"""
Configuration manager that composes all domain configurations.

Provides centralized configuration management with validation,
environment overrides, and persistence.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

from .base import BaseConfig
from .storage import StorageConfig
from .search import SearchConfig
from .vector import VectorConfig
from .document import DocumentConfig
from .locking import LockingConfig
from .panel import PanelConfig


@dataclass
class ConfigurationManager:
    """
    Central configuration manager for all domains.
    
    Manages loading, validation, and access to all configuration domains
    while maintaining separation of concerns.
    """
    
    storage: StorageConfig = field(default_factory=StorageConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    locking: LockingConfig = field(default_factory=LockingConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    
    # Manager settings
    config_file_path: Optional[Path] = None
    environment: Optional[str] = None  # "development", "production", "test"
    auto_reload: bool = False
    validate_on_load: bool = True
    
    def __post_init__(self):
        """Initialize configuration manager."""
        if self.config_file_path:
            self.load_from_file(self.config_file_path)
        
        # Apply environment overrides
        self._apply_environment_overrides()
        
        # Validate if required
        if self.validate_on_load:
            errors = self.validate_all()
            if errors:
                raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], environment: Optional[str] = None) -> 'ConfigurationManager':
        """
        Create configuration manager from file.
        
        Args:
            file_path: Path to configuration file
            environment: Environment name
            
        Returns:
            Configured manager instance
        """
        return cls(
            config_file_path=Path(file_path),
            environment=environment
        )
    
    @classmethod
    def from_environment(cls, environment: str) -> 'ConfigurationManager':
        """
        Create configuration manager for specific environment.
        
        Args:
            environment: Environment name ("development", "production", "test")
            
        Returns:
            Configured manager instance
        """
        manager = cls(environment=environment)
        
        # Load environment-specific defaults
        if environment == "development":
            manager.storage.base_path = "./dev_data"
            manager.storage.validate_json_on_read = True
            manager.locking.deadlock_detection_enabled = True
            manager.search.enable_search_cache = False
            
        elif environment == "production":
            manager.storage.base_path = "/var/lib/chatdb/data"
            manager.storage.validate_json_on_read = False
            manager.storage.jsonl_read_buffer_size = 16384
            manager.locking.retry_max_attempts = 200
            manager.search.cache_ttl_seconds = 600
            
        elif environment == "test":
            manager.storage.base_path = "./test_data"
            manager.storage.max_messages_per_session = 1000
            manager.locking.timeout_seconds = 5.0
            manager.search.max_total_results = 100
        
        return manager
    
    def load_from_file(self, file_path: Path) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to configuration file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load each domain configuration
        if "storage" in data:
            self.storage = StorageConfig.from_dict(data["storage"])
        if "search" in data:
            self.search = SearchConfig.from_dict(data["search"])
        if "vector" in data:
            self.vector = VectorConfig.from_dict(data["vector"])
        if "document" in data:
            self.document = DocumentConfig.from_dict(data["document"])
        if "locking" in data:
            self.locking = LockingConfig.from_dict(data["locking"])
        if "panel" in data:
            self.panel = PanelConfig.from_dict(data["panel"])
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save configuration
        """
        data = {
            "storage": self.storage.to_dict(),
            "search": self.search.to_dict(),
            "vector": self.vector.to_dict(),
            "document": self.document.to_dict(),
            "locking": self.locking.to_dict(),
            "panel": self.panel.to_dict()
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to all configurations."""
        # Apply overrides to each domain with specific prefixes
        self.storage = self.storage.apply_environment_overrides("CHATDB_STORAGE")
        self.search = self.search.apply_environment_overrides("CHATDB_SEARCH")
        self.vector = self.vector.apply_environment_overrides("CHATDB_VECTOR")
        self.document = self.document.apply_environment_overrides("CHATDB_DOCUMENT")
        self.locking = self.locking.apply_environment_overrides("CHATDB_LOCKING")
        self.panel = self.panel.apply_environment_overrides("CHATDB_PANEL")
        
        # Also check for general overrides
        self._apply_general_overrides()
    
    def _apply_general_overrides(self) -> None:
        """Apply general environment overrides that affect multiple domains."""
        # Check for common overrides
        if "CHATDB_BASE_PATH" in os.environ:
            self.storage.base_path = os.environ["CHATDB_BASE_PATH"]
        
        if "CHATDB_MAX_FILE_SIZE" in os.environ:
            size = int(os.environ["CHATDB_MAX_FILE_SIZE"])
            self.storage.max_document_size_bytes = size
            self.document.max_file_size_bytes = size
        
        if "CHATDB_ENABLE_CACHING" in os.environ:
            enable = os.environ["CHATDB_ENABLE_CACHING"].lower() in ('true', '1', 'yes')
            self.search.enable_search_cache = enable
            self.vector.cache_enabled = enable
            self.document.cache_processed_documents = enable
    
    def validate_all(self) -> List[str]:
        """
        Validate all configurations.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate each domain
        for name, config in [
            ("storage", self.storage),
            ("search", self.search),
            ("vector", self.vector),
            ("document", self.document),
            ("locking", self.locking),
            ("panel", self.panel)
        ]:
            domain_errors = config.validate()
            for error in domain_errors:
                errors.append(f"{name}: {error}")
        
        # Cross-domain validation
        errors.extend(self._validate_cross_domain())
        
        return errors
    
    def _validate_cross_domain(self) -> List[str]:
        """
        Validate configuration consistency across domains.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Ensure document size limits are consistent
        if self.document.max_file_size_bytes > self.storage.max_document_size_bytes:
            errors.append(
                f"document.max_file_size_bytes ({self.document.max_file_size_bytes}) "
                f"exceeds storage.max_document_size_bytes ({self.storage.max_document_size_bytes})"
            )
        
        # Ensure search limits are reasonable
        if self.search.max_results_per_session > self.storage.max_messages_per_session:
            errors.append(
                f"search.max_results_per_session ({self.search.max_results_per_session}) "
                f"exceeds storage.max_messages_per_session ({self.storage.max_messages_per_session})"
            )
        
        # Ensure panel limits are consistent
        if self.panel.max_messages_per_panel > self.storage.max_messages_per_session:
            errors.append(
                f"panel.max_messages_per_panel ({self.panel.max_messages_per_panel}) "
                f"exceeds storage.max_messages_per_session ({self.storage.max_messages_per_session})"
            )
        
        return errors
    
    def get_domain_config(self, domain: str) -> Optional[BaseConfig]:
        """
        Get configuration for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Domain configuration or None
        """
        return getattr(self, domain, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all configurations to dictionary.
        
        Returns:
            Dictionary of all configurations
        """
        return {
            "storage": self.storage.to_dict(),
            "search": self.search.to_dict(),
            "vector": self.vector.to_dict(),
            "document": self.document.to_dict(),
            "locking": self.locking.to_dict(),
            "panel": self.panel.to_dict(),
            "environment": self.environment
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for logging/debugging.
        
        Returns:
            Summary of key configuration values
        """
        return {
            "environment": self.environment,
            "base_path": self.storage.base_path,
            "locking_enabled": self.locking.enabled,
            "vector_provider": self.vector.default_embedding_provider,
            "search_cache_enabled": self.search.enable_search_cache,
            "document_extensions": len(self.document.allowed_extensions),
            "max_file_size_mb": self.document.max_file_size_bytes / 1_048_576,
            "panel_types": list(self.panel.panel_types.keys())
        }
    
    def __repr__(self) -> str:
        """String representation of configuration manager."""
        return f"ConfigurationManager(environment={self.environment}, base_path={self.storage.base_path})"


# Convenience functions
def load_config(
    config_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> ConfigurationManager:
    """
    Load configuration with automatic environment detection.
    
    Args:
        config_path: Optional path to configuration file
        environment: Optional environment override
        
    Returns:
        Configured manager instance
    """
    # Auto-detect environment if not specified
    if environment is None:
        environment = os.environ.get("CHATDB_ENV", "development")
    
    # Check for config file in environment
    if config_path is None:
        env_config_path = os.environ.get("CHATDB_CONFIG_PATH")
        if env_config_path:
            config_path = Path(env_config_path)
    
    # Create manager
    if config_path:
        return ConfigurationManager.from_file(config_path, environment)
    else:
        return ConfigurationManager.from_environment(environment)


def create_default_config(environment: str = "development") -> ConfigurationManager:
    """
    Create default configuration for an environment.
    
    Args:
        environment: Target environment
        
    Returns:
        Default configuration manager
    """
    return ConfigurationManager.from_environment(environment)