"""
Utility functions for file operations, JSON handling, and path management.
"""

from ff_utils.ff_file_ops import (
    ff_atomic_write,
    ff_atomic_append,
    ff_safe_read,
    ff_ensure_directory,
    ff_safe_delete,
    ff_file_exists,
    ff_directory_exists,
    ff_list_files,
    ff_get_file_size,
    ff_get_file_operation_manager,
    # Classes
    FFFileLock,
    FFFileOperationError,
    FFAtomicWriteError,
    FFLockTimeoutError
)

from ff_utils.ff_json_utils import (
    ff_write_json,
    ff_read_json,
    ff_append_jsonl,
    ff_read_jsonl,
    ff_read_jsonl_paginated,
    # Classes
    FFJSONError
)

from ff_utils.ff_path_utils import (
    ff_get_user_path,
    ff_get_session_path,
    ff_get_panel_path,
    ff_get_global_personas_path,
    ff_get_user_personas_path,
    ff_get_documents_path,
    ff_get_context_history_path,
    ff_generate_session_id,
    ff_generate_panel_id,
    ff_generate_context_snapshot_id,
    ff_sanitize_filename,
    ff_build_file_paths,
    ff_build_panel_file_paths,
    # New centralized key functions
    ff_get_user_key,
    ff_get_session_key,
    ff_get_profile_key,
    ff_get_messages_key,
    ff_get_session_metadata_key
)

__all__ = [
    # file_ops functions
    "ff_atomic_write",
    "ff_atomic_append",
    "ff_safe_read",
    "ff_ensure_directory",
    "ff_safe_delete",
    "ff_file_exists",
    "ff_directory_exists",
    "ff_list_files",
    "ff_get_file_size",
    "ff_get_file_operation_manager",
    # file_ops classes
    "FFFileLock",
    "FFFileOperationError",
    "FFAtomicWriteError",
    "FFLockTimeoutError",
    # json_utils functions
    "ff_write_json",
    "ff_read_json",
    "ff_append_jsonl", 
    "ff_read_jsonl",
    "ff_read_jsonl_paginated",
    # json_utils classes
    "FFJSONError",
    # path_utils functions
    "ff_get_user_path",
    "ff_get_session_path",
    "ff_get_panel_path",
    "ff_get_global_personas_path",
    "ff_get_user_personas_path",
    "ff_get_documents_path",
    "ff_get_context_history_path",
    "ff_generate_session_id",
    "ff_generate_panel_id",
    "ff_generate_context_snapshot_id",
    "ff_sanitize_filename",
    "ff_build_file_paths",
    "ff_build_panel_file_paths",
    # New centralized key functions
    "ff_get_user_key",
    "ff_get_session_key",
    "ff_get_profile_key",
    "ff_get_messages_key",
    "ff_get_session_metadata_key"
]