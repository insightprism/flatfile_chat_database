"""
Storage configuration for core file operations.

Handles configuration for file storage, paths, limits, and behaviors.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .base import BaseConfig, validate_positive, validate_non_empty


@dataclass
class StorageConfig(BaseConfig):
    """
    Core storage configuration.
    
    Manages file paths, size limits, and storage behaviors.
    """
    
    # Storage locations
    base_path: str = "./data"
    user_data_directory: str = "users"
    session_data_directory: str = "sessions"
    system_config_directory: str = "system"
    
    # File naming patterns
    session_id_prefix: str = "chat_session"
    session_timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Core file names
    profile_filename: str = "profile.json"
    session_metadata_filename: str = "session.json"
    messages_filename: str = "messages.jsonl"
    
    # Storage limits
    max_message_size_bytes: int = 1_048_576  # 1MB
    max_document_size_bytes: int = 104_857_600  # 100MB
    max_messages_per_session: int = 100_000
    max_sessions_per_user: int = 1_000
    
    # File operation settings
    atomic_write_temp_suffix: str = ".tmp"
    backup_before_delete: bool = True
    auto_cleanup_empty_directories: bool = True
    validate_json_on_read: bool = True
    create_parent_directories: bool = True
    
    # Performance settings
    jsonl_read_buffer_size: int = 8192
    file_operation_max_retries: int = 3
    file_operation_retry_delay_ms: int = 10
    
    # ID generation settings
    message_id_length: int = 12
    temp_file_suffix_length: int = 8
    
    def validate(self) -> List[str]:
        """
        Validate storage configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate paths
        if error := validate_non_empty(self.base_path, "base_path"):
            errors.append(error)
        if error := validate_non_empty(self.user_data_directory, "user_data_directory"):
            errors.append(error)
        if error := validate_non_empty(self.session_data_directory, "session_data_directory"):
            errors.append(error)
        
        # Validate size limits
        if error := validate_positive(self.max_message_size_bytes, "max_message_size_bytes"):
            errors.append(error)
        if error := validate_positive(self.max_document_size_bytes, "max_document_size_bytes"):
            errors.append(error)
        if error := validate_positive(self.max_messages_per_session, "max_messages_per_session"):
            errors.append(error)
        if error := validate_positive(self.max_sessions_per_user, "max_sessions_per_user"):
            errors.append(error)
        
        # Validate performance settings
        if error := validate_positive(self.jsonl_read_buffer_size, "jsonl_read_buffer_size"):
            errors.append(error)
        if error := validate_positive(self.file_operation_max_retries, "file_operation_max_retries"):
            errors.append(error)
        if error := validate_positive(self.file_operation_retry_delay_ms, "file_operation_retry_delay_ms"):
            errors.append(error)
        
        # Validate ID generation
        if error := validate_positive(self.message_id_length, "message_id_length"):
            errors.append(error)
        if error := validate_positive(self.temp_file_suffix_length, "temp_file_suffix_length"):
            errors.append(error)
        
        return errors