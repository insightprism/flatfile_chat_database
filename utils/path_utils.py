"""
Path management utilities for the flatfile chat database.

Provides consistent path construction and naming conventions based on
the configuration.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import unicodedata

from flatfile_chat_database.config import StorageConfig


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem storage.
    
    Args:
        filename: Original filename
        max_length: Maximum length for filename
        
    Returns:
        Sanitized filename safe for all filesystems
    """
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove non-ASCII characters
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    
    # Remove leading/trailing underscores and dots
    filename = filename.strip('._')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    # Truncate if too long (preserve extension if present)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def generate_session_id(config: StorageConfig) -> str:
    """
    Generate unique session ID using configured format.
    
    Args:
        config: Storage configuration
        
    Returns:
        Session ID like "chat_session_20240722_143022_123456"
    """
    timestamp = datetime.now().strftime(config.session_timestamp_format)
    # Add microseconds to ensure uniqueness
    microseconds = datetime.now().strftime("%f")[:6]
    return f"{config.session_id_prefix}_{timestamp}_{microseconds}"


def generate_panel_id(config: StorageConfig) -> str:
    """
    Generate unique panel ID using configured format.
    
    Args:
        config: Storage configuration
        
    Returns:
        Panel ID like "panel_20240722_143022_123456"
    """
    timestamp = datetime.now().strftime(config.session_timestamp_format)
    # Add microseconds to ensure uniqueness
    microseconds = datetime.now().strftime("%f")[:6]
    return f"{config.panel_id_prefix}_{timestamp}_{microseconds}"


def generate_context_snapshot_id(config: StorageConfig) -> str:
    """
    Generate context snapshot filename.
    
    Args:
        config: Storage configuration
        
    Returns:
        Snapshot ID like "context_20240722_143022_123456"
    """
    timestamp = datetime.now().strftime(config.context_snapshot_timestamp_format)
    # Add microseconds to ensure uniqueness
    microseconds = datetime.now().strftime("%f")[:6]
    return f"context_{timestamp}_{microseconds}"


def get_base_path(config: StorageConfig) -> Path:
    """
    Get base storage path from configuration.
    
    Args:
        config: Storage configuration
        
    Returns:
        Base path as Path object
    """
    return Path(config.storage_base_path).resolve()


def get_user_path(base_path: Path, user_id: str, config: Optional['StorageConfig'] = None) -> Path:
    """
    Get user directory path.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        config: Storage configuration (if None, uses 'users' as default)
        
    Returns:
        User directory path
    """
    # Sanitize user_id for filesystem
    safe_user_id = sanitize_filename(user_id)
    if config:
        return base_path / config.user_data_directory_name / safe_user_id
    # Default for backward compatibility
    return base_path / "users" / safe_user_id


def get_session_path(base_path: Path, user_id: str, session_id: str, config: Optional['StorageConfig'] = None) -> Path:
    """
    Get session directory path.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        session_id: Session identifier
        config: Storage configuration (optional)
        
    Returns:
        Session directory path
    """
    user_path = get_user_path(base_path, user_id, config)
    return user_path / session_id


def get_panel_path(base_path: Path, panel_id: str, config: StorageConfig) -> Path:
    """
    Get panel session directory path.
    
    Args:
        base_path: Base storage path
        panel_id: Panel identifier
        config: Storage configuration
        
    Returns:
        Panel directory path
    """
    return base_path / config.panel_sessions_directory_name / panel_id


def get_global_personas_path(base_path: Path, config: StorageConfig) -> Path:
    """
    Get global personas directory path.
    
    Args:
        base_path: Base storage path
        config: Storage configuration
        
    Returns:
        Global personas directory path
    """
    return base_path / config.global_personas_directory_name


def get_user_personas_path(base_path: Path, user_id: str, config: StorageConfig) -> Path:
    """
    Get user-specific personas directory path.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        config: Storage configuration
        
    Returns:
        User personas directory path
    """
    user_path = get_user_path(base_path, user_id, config)
    return user_path / config.panel_personas_directory_name


def get_documents_path(session_path: Path, config: StorageConfig) -> Path:
    """
    Get documents directory for a session.
    
    Args:
        session_path: Session directory path
        config: Storage configuration
        
    Returns:
        Documents directory path
    """
    return session_path / config.document_storage_subdirectory_name


def get_context_history_path(session_path: Path, config: StorageConfig) -> Path:
    """
    Get context history directory for a session.
    
    Args:
        session_path: Session directory path
        config: Storage configuration
        
    Returns:
        Context history directory path
    """
    return session_path / config.context_history_subdirectory_name


def get_panel_insights_path(panel_path: Path, config: StorageConfig) -> Path:
    """
    Get insights directory for a panel.
    
    Args:
        panel_path: Panel directory path
        config: Storage configuration
        
    Returns:
        Panel insights directory path
    """
    return panel_path / config.panel_insights_directory_name


def build_file_paths(session_path: Path, config: StorageConfig) -> dict:
    """
    Build all standard file paths for a session.
    
    Args:
        session_path: Session directory path
        config: Storage configuration
        
    Returns:
        Dictionary of file paths
    """
    return {
        "session_metadata": session_path / config.session_metadata_filename,
        "messages": session_path / config.messages_filename,
        "situational_context": session_path / config.situational_context_filename,
        "documents": get_documents_path(session_path, config),
        "document_metadata": get_documents_path(session_path, config) / config.document_metadata_filename,
        "context_history": get_context_history_path(session_path, config)
    }


def build_panel_file_paths(panel_path: Path, config: StorageConfig) -> dict:
    """
    Build all standard file paths for a panel session.
    
    Args:
        panel_path: Panel directory path
        config: Storage configuration
        
    Returns:
        Dictionary of file paths
    """
    return {
        "panel_metadata": panel_path / config.panel_metadata_filename,
        "messages": panel_path / config.messages_filename,
        "personas": panel_path / config.panel_personas_directory_name,
        "insights": get_panel_insights_path(panel_path, config)
    }


def parse_session_id(session_id: str, config: StorageConfig) -> Optional[datetime]:
    """
    Parse timestamp from session ID.
    
    Args:
        session_id: Session identifier
        config: Storage configuration
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        # Extract timestamp part after prefix
        prefix = f"{config.session_id_prefix}_"
        if session_id.startswith(prefix):
            timestamp_str = session_id[len(prefix):]
            return datetime.strptime(timestamp_str, config.session_timestamp_format)
    except:
        pass
    return None


def is_valid_session_id(session_id: str, config: StorageConfig) -> bool:
    """
    Check if session ID follows expected format.
    
    Args:
        session_id: Session identifier to validate
        config: Storage configuration
        
    Returns:
        True if valid format
    """
    pattern = f"^{re.escape(config.session_id_prefix)}_\\d{{8}}_\\d{{6}}$"
    return bool(re.match(pattern, session_id))


def is_valid_panel_id(panel_id: str, config: StorageConfig) -> bool:
    """
    Check if panel ID follows expected format.
    
    Args:
        panel_id: Panel identifier to validate
        config: Storage configuration
        
    Returns:
        True if valid format
    """
    pattern = f"^{re.escape(config.panel_id_prefix)}_\\d{{8}}_\\d{{6}}$"
    return bool(re.match(pattern, panel_id))


# Centralized key generation functions for backend storage

def get_user_key(base_path: Path, user_id: str, config: StorageConfig) -> str:
    """
    Get backend storage key for user directory.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        config: Storage configuration
        
    Returns:
        Backend key string
    """
    user_path = base_path / config.user_data_directory_name / sanitize_filename(user_id)
    return str(user_path.relative_to(base_path))


def get_session_key(base_path: Path, user_id: str, session_id: str, config: StorageConfig) -> str:
    """
    Get backend storage key for session.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        session_id: Session identifier
        config: Storage configuration
        
    Returns:
        Backend key string
    """
    session_path = get_session_path(base_path, user_id, session_id, config)
    return str(session_path.relative_to(base_path))


def get_profile_key(base_path: Path, user_id: str, config: StorageConfig) -> str:
    """
    Get backend storage key for user profile.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        config: Storage configuration
        
    Returns:
        Backend key string
    """
    user_path = base_path / config.user_data_directory_name / sanitize_filename(user_id)
    profile_path = user_path / config.user_profile_filename
    return str(profile_path.relative_to(base_path))


def get_messages_key(base_path: Path, user_id: str, session_id: str, config: StorageConfig) -> str:
    """
    Get backend storage key for session messages.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        session_id: Session identifier
        config: Storage configuration
        
    Returns:
        Backend key string
    """
    session_path = get_session_path(base_path, user_id, session_id, config)
    messages_path = session_path / config.messages_filename
    return str(messages_path.relative_to(base_path))


def get_session_metadata_key(base_path: Path, user_id: str, session_id: str, config: StorageConfig) -> str:
    """
    Get backend storage key for session metadata.
    
    Args:
        base_path: Base storage path
        user_id: User identifier
        session_id: Session identifier
        config: Storage configuration
        
    Returns:
        Backend key string
    """
    session_path = get_session_path(base_path, user_id, session_id, config)
    metadata_path = session_path / config.session_metadata_filename
    return str(metadata_path.relative_to(base_path))


