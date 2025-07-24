#!/usr/bin/env python3
"""
Script to demonstrate the directory structure created by the flatfile chat database.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database.storage import StorageManager
from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.models import Message, Document, SituationalContext, Panel, Persona


def print_tree(directory, prefix="", is_last=True):
    """Print directory tree structure"""
    if prefix == "":
        print(directory)
    
    items = sorted(list(directory.iterdir()))
    
    for i, item in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = "└── " if is_last_item else "├── "
        print(prefix + current_prefix + item.name)
        
        if item.is_dir():
            extension = "    " if is_last_item else "│   "
            print_tree(item, prefix + extension, is_last_item)


async def create_sample_data():
    """Create sample data to show the directory structure"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Creating sample data in: {temp_dir}\n")
    
    try:
        # Create storage with custom config to show configuration options
        config = StorageConfig(
            storage_base_path=temp_dir,
            user_data_directory_name="users",  # Default
            panel_sessions_directory_name="panels",  # Default
            global_personas_directory_name="personas",  # Default
            system_config_directory_name="system",  # Default
        )
        
        manager = StorageManager(config=config)
        await manager.initialize()
        
        # Create users
        users = ["alice", "bob"]
        for user_id in users:
            await manager.create_user(user_id, {"username": user_id.capitalize()})
            
            # Create sessions for each user
            for i in range(2):
                session_id = await manager.create_session(user_id, f"Session {i+1}")
                
                # Add messages
                await manager.add_message(user_id, session_id, 
                    Message(role="user", content=f"Hello from {user_id}"))
                await manager.add_message(user_id, session_id,
                    Message(role="assistant", content=f"Hello {user_id}, how can I help?"))
                
                # Add document (only for first session)
                if i == 0:
                    await manager.save_document(
                        user_id, session_id, "example.txt",
                        b"Sample document content"
                    )
                
                # Add context
                context = SituationalContext(
                    summary=f"Chat session {i+1} for {user_id}",
                    key_points=[f"Point 1", f"Point 2"],
                    entities={"user": [user_id]},
                    confidence=0.9
                )
                await manager.update_context(user_id, session_id, context)
                
                # Save context snapshot
                await manager.save_context_snapshot(user_id, session_id, context)
        
        # Note: Panels and personas would also be created here in a real application
        
        # Show the directory structure
        print("Directory Structure:")
        print("=" * 50)
        print_tree(Path(temp_dir))
        
        # Show what each directory contains
        print("\n\nDirectory Contents Explanation:")
        print("=" * 50)
        print("""
📁 {base_path}/
├── 📁 users/                    # All user data (configurable: user_data_directory_name)
│   ├── 📁 alice/                # Individual user directory
│   │   ├── 📄 profile.json      # User profile and preferences
│   │   └── 📁 chat_session_*/   # Session directories (prefix configurable)
│   │       ├── 📄 session.json  # Session metadata (configurable: session_metadata_filename)
│   │       ├── 📄 messages.jsonl # Chat messages (configurable: messages_filename)
│   │       ├── 📄 context.json  # Current context (configurable: situational_context_filename)
│   │       ├── 📁 documents/    # Document storage (configurable: document_storage_subdirectory_name)
│   │       │   ├── 📄 metadata.json  # Document metadata
│   │       │   └── 📄 {doc_id}      # Actual document files
│   │       └── 📁 context_history/  # Context snapshots (configurable: context_history_subdirectory_name)
│   │           └── 📄 {timestamp}.json  # Historical context snapshots
│   └── 📁 bob/                  # Another user
│       └── ... (same structure)
├── 📁 panels/                   # Panel sessions (configurable: panel_sessions_directory_name)
│   └── 📁 panel_123/            # Individual panel
│       ├── 📄 metadata.json     # Panel configuration
│       └── 📄 messages.jsonl    # Panel messages
├── 📁 personas/                 # Global personas (configurable: global_personas_directory_name)
│   └── 📄 helpful_assistant.json # Persona definition
└── 📁 system/                   # System config (configurable: system_config_directory_name)
    └── 📄 (system files)        # System configuration files

Note: All directory and file names are configurable via StorageConfig!
        """)
        
        # Show example of custom configuration
        print("\nExample with Custom Configuration:")
        print("=" * 50)
        print("""
# You can customize all paths via StorageConfig:
config = StorageConfig(
    storage_base_path="/data/chats",
    user_data_directory_name="customers",        # Instead of "users"
    session_id_prefix="conversation",            # Instead of "chat_session"
    messages_filename="chat_log.jsonl",          # Instead of "messages.jsonl"
    session_metadata_filename="info.json",       # Instead of "session.json"
    document_storage_subdirectory_name="files",  # Instead of "documents"
    # ... and many more options
)

This would create:
📁 /data/chats/
├── 📁 customers/
│   └── 📁 alice/
│       └── 📁 conversation_20250723_150000/
│           ├── 📄 info.json
│           ├── 📄 chat_log.jsonl
│           └── 📁 files/
│               └── ...
        """)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    asyncio.run(create_sample_data())