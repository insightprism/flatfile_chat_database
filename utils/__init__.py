"""
Utility functions for file operations, JSON handling, and path management.
"""

from .file_ops import (
    atomic_write,
    atomic_append,
    safe_read,
    ensure_directory,
    safe_delete,
    file_exists,
    directory_exists,
    list_files,
    get_file_size,
    get_file_operation_manager
)

from .json_utils import (
    write_json,
    read_json,
    append_jsonl,
    read_jsonl,
    read_jsonl_paginated
)

from .path_utils import (
    get_user_path,
    get_session_path,
    get_panel_path,
    get_global_personas_path,
    get_user_personas_path,
    get_documents_path,
    get_context_history_path,
    generate_session_id,
    generate_panel_id,
    generate_context_snapshot_id,
    sanitize_filename,
    build_file_paths,
    build_panel_file_paths,
    # New centralized key functions
    get_user_key,
    get_session_key,
    get_profile_key,
    get_messages_key,
    get_session_metadata_key
)

__all__ = [
    # file_ops
    "atomic_write",
    "atomic_append",
    "safe_read",
    "ensure_directory",
    "safe_delete",
    "file_exists",
    "directory_exists",
    "list_files",
    "get_file_size",
    "get_file_operation_manager",
    # json_utils
    "write_json",
    "read_json",
    "append_jsonl", 
    "read_jsonl",
    "read_jsonl_paginated",
    # path_utils
    "get_user_path",
    "get_session_path",
    "get_panel_path",
    "get_global_personas_path",
    "get_user_personas_path",
    "get_documents_path",
    "get_context_history_path",
    "generate_session_id",
    "generate_panel_id",
    "generate_context_snapshot_id",
    "sanitize_filename",
    "build_file_paths",
    "build_panel_file_paths",
    # New centralized key functions
    "get_user_key",
    "get_session_key",
    "get_profile_key",
    "get_messages_key",
    "get_session_metadata_key"
]