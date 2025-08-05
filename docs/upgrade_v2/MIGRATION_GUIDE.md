# Migration Guide: Flatfile Database v2 to Modular Chat Platform

## Overview

This guide provides step-by-step instructions for migrating from the current Flatfile Database v2 to the new modular chat platform architecture. The migration is designed to be performed incrementally with minimal disruption.

## Pre-Migration Checklist

- [ ] Backup all existing data
- [ ] Document current custom configurations
- [ ] Identify active integrations
- [ ] Review breaking changes
- [ ] Set up test environment
- [ ] Prepare rollback plan

## Migration Phases

### Phase 1: Environment Preparation (Day 1)

#### 1.1 Create New Directory Structure
```bash
# Create new modular platform structure alongside existing
mkdir -p modular_chat_platform/{core,modules,storage,config}
mkdir -p modular_chat_platform/modules/{text_chat,memory,rag,multi_agent}
mkdir -p modular_chat_platform/modules/{tool_use,persona,multimodal,topic_router,trace_logger}
```

#### 1.2 Install Dependencies
```bash
# Core dependencies
pip install aiofiles msgpack pydantic

# Module-specific dependencies
pip install numpy faiss-cpu  # For RAG module
pip install Pillow opencv-python  # For multimodal
```

#### 1.3 Set Up Configuration
```bash
# Copy and adapt configuration templates
cp templates/config/* modular_chat_platform/config/
```

### Phase 2: Core Infrastructure (Day 2-3)

#### 2.1 Implement Message Bus
```python
# core/message_bus.py
from collections import defaultdict
from typing import Dict, List, Callable, Any
import asyncio

class MessageBus:
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._middleware: List[Callable] = []
    
    def on(self, event_type: str):
        def decorator(func: Callable):
            self._handlers[event_type].append(func)
            return func
        return decorator
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        # Apply middleware
        for middleware in self._middleware:
            data = await middleware(event_type, data)
        
        # Call handlers
        tasks = []
        for handler in self._handlers[event_type]:
            tasks.append(handler(data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Global message bus instance
message_bus = MessageBus()
```

#### 2.2 Create Module Loader
```python
# core/module_loader.py
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ModuleLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.modules: Dict[str, Any] = {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        with open(self.config_path / "modules.json") as f:
            return json.load(f)
    
    async def load_module(self, module_name: str) -> Optional[Any]:
        if module_name in self.modules:
            return self.modules[module_name]
        
        if module_name not in self.config["modules"]:
            return None
        
        module_config = self.config["modules"][module_name]
        if not module_config.get("enabled", False):
            return None
        
        # Dynamic import
        module = importlib.import_module(f"modules.{module_name}")
        
        # Initialize module
        if hasattr(module, "initialize"):
            await module.initialize(module_config)
        
        self.modules[module_name] = module
        return module
```

### Phase 3: Data Migration Scripts (Day 4-5)

#### 3.1 Storage Structure Mapper
```python
# migration/storage_mapper.py
from pathlib import Path
import json
import shutil
from typing import Dict, List

class StorageMapper:
    """Maps old storage structure to new modular structure."""
    
    def __init__(self, old_base: Path, new_base: Path):
        self.old_base = old_base
        self.new_base = new_base
    
    def migrate_sessions(self):
        """Migrate chat sessions to new structure."""
        old_sessions = self.old_base / "users"
        new_sessions = self.new_base / "text_chat" / "sessions"
        
        for user_dir in old_sessions.iterdir():
            if not user_dir.is_dir():
                continue
            
            user_id = user_dir.name
            new_user_dir = new_sessions / user_id
            new_user_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all chat sessions
            for session_file in user_dir.glob("chat_session_*/session.json"):
                session_id = session_file.parent.name
                self._migrate_session(user_id, session_id)
    
    def _migrate_session(self, user_id: str, session_id: str):
        """Migrate individual session."""
        old_session_path = self.old_base / "users" / user_id / session_id
        new_session_path = self.new_base / "text_chat" / "sessions" / user_id / session_id
        
        new_session_path.mkdir(parents=True, exist_ok=True)
        
        # Migrate session metadata
        old_meta = old_session_path / "session.json"
        if old_meta.exists():
            with open(old_meta) as f:
                data = json.load(f)
            
            # Transform to new format
            new_meta = {
                "id": session_id,
                "user_id": user_id,
                "title": data.get("session_name", "Untitled"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "metadata": data.get("metadata", {})
            }
            
            with open(new_session_path / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
        
        # Migrate messages
        old_messages = old_session_path / "messages.jsonl"
        if old_messages.exists():
            shutil.copy(old_messages, new_session_path / "messages.jsonl")
```

#### 3.2 Configuration Migrator
```python
# migration/config_migrator.py
import json
from typing import Dict, Any

def migrate_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform old configuration to new modular format."""
    
    new_config = {
        "modules": {
            "text_chat": {
                "enabled": True,
                "max_message_length": old_config.get("storage", {}).get("max_message_size_bytes", 65536),
                "session": {
                    "max_sessions_per_user": old_config.get("storage", {}).get("max_sessions_per_user", 100)
                }
            },
            "memory": {
                "enabled": True,
                "short_term": {
                    "ttl_seconds": 3600
                },
                "long_term": {
                    "max_items_per_user": 10000
                }
            },
            "rag": {
                "enabled": old_config.get("vector", {}).get("enable_vector_storage", False),
                "embedding": {
                    "provider": old_config.get("vector", {}).get("default_embedding_provider", "openai")
                }
            }
        },
        "storage": {
            "backend": "flatfile",
            "base_path": old_config.get("storage", {}).get("base_path", "./data")
        }
    }
    
    return new_config
```

### Phase 4: Module Implementation (Day 6-10)

#### 4.1 Implement Text Chat Module
```python
# modules/text_chat/__init__.py
from .chat import send_message, create_session, get_messages
from ..core import message_bus, ModuleContext

async def initialize(config: Dict):
    """Initialize text chat module."""
    # Register event handlers
    message_bus.on("chat.message.send")(handle_message)
    message_bus.on("chat.session.create")(handle_create_session)

async def handle_message(data: Dict):
    """Handle incoming chat message."""
    message = await send_message(
        session_id=data["session_id"],
        content=data["content"],
        role=data.get("role", "user")
    )
    
    # Emit response event
    await message_bus.emit("chat.message.received", {
        "message": message.dict()
    })

# modules/text_chat/chat.py
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path

async def send_message(
    session_id: str,
    content: str,
    role: str = "user",
    metadata: Optional[Dict] = None
) -> Message:
    """Send a message in a chat session."""
    message = Message(
        id=f"msg_{datetime.now().timestamp()}",
        session_id=session_id,
        role=role,
        content=content,
        timestamp=datetime.now().isoformat(),
        metadata=metadata or {}
    )
    
    # Store message
    session_path = get_session_path(session_id)
    messages_file = session_path / "messages.jsonl"
    
    async with aiofiles.open(messages_file, "a") as f:
        await f.write(json.dumps(message.dict()) + "\n")
    
    return message
```

#### 4.2 Implement Memory Module
```python
# modules/memory/__init__.py
from typing import Any, Optional, List
import json
from datetime import datetime, timedelta

class MemoryModule:
    def __init__(self, config: Dict):
        self.config = config
        self.short_term_cache = {}
        self.ttl = config.get("short_term", {}).get("ttl_seconds", 3600)
    
    async def store(self, key: str, value: Any, memory_type: str = "short_term"):
        """Store information in memory."""
        if memory_type == "short_term":
            self.short_term_cache[key] = {
                "value": value,
                "timestamp": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=self.ttl)
            }
        else:
            # Store in long-term storage
            await self._store_long_term(key, value)
    
    async def recall(self, key: str, memory_type: str = "short_term") -> Optional[Any]:
        """Retrieve memory by key."""
        if memory_type == "short_term":
            entry = self.short_term_cache.get(key)
            if entry and entry["expires_at"] > datetime.now():
                return entry["value"]
        else:
            return await self._recall_long_term(key)
        
        return None
```

### Phase 5: Testing & Validation (Day 11-12)

#### 5.1 Migration Validation Script
```python
# migration/validate.py
import json
from pathlib import Path
from typing import List, Tuple

class MigrationValidator:
    def __init__(self, old_path: Path, new_path: Path):
        self.old_path = old_path
        self.new_path = new_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks."""
        self.validate_sessions()
        self.validate_messages()
        self.validate_documents()
        self.validate_config()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def validate_sessions(self):
        """Validate session migration."""
        old_sessions = set()
        new_sessions = set()
        
        # Count old sessions
        for user_dir in (self.old_path / "users").iterdir():
            if user_dir.is_dir():
                for session_dir in user_dir.glob("chat_session_*"):
                    old_sessions.add(f"{user_dir.name}/{session_dir.name}")
        
        # Count new sessions
        sessions_path = self.new_path / "text_chat" / "sessions"
        if sessions_path.exists():
            for user_dir in sessions_path.iterdir():
                if user_dir.is_dir():
                    for session_dir in user_dir.iterdir():
                        if session_dir.is_dir():
                            new_sessions.add(f"{user_dir.name}/{session_dir.name}")
        
        # Compare
        missing = old_sessions - new_sessions
        if missing:
            self.errors.append(f"Missing sessions: {missing}")
        
        extra = new_sessions - old_sessions
        if extra:
            self.warnings.append(f"Extra sessions found: {extra}")
```

#### 5.2 Test Suite
```python
# tests/test_migration.py
import pytest
import asyncio
from pathlib import Path
from migration.storage_mapper import StorageMapper

@pytest.fixture
def test_data_path(tmp_path):
    """Create test data structure."""
    old_path = tmp_path / "old"
    new_path = tmp_path / "new"
    
    # Create sample old structure
    user_path = old_path / "users" / "test_user"
    session_path = user_path / "chat_session_20240115_100000_123456"
    session_path.mkdir(parents=True)
    
    # Create sample session
    session_data = {
        "session_id": "chat_session_20240115_100000_123456",
        "session_name": "Test Chat",
        "created_at": "2024-01-15T10:00:00Z"
    }
    
    with open(session_path / "session.json", "w") as f:
        json.dump(session_data, f)
    
    return old_path, new_path

def test_session_migration(test_data_path):
    """Test session migration."""
    old_path, new_path = test_data_path
    mapper = StorageMapper(old_path, new_path)
    
    mapper.migrate_sessions()
    
    # Verify migration
    migrated = new_path / "text_chat" / "sessions" / "test_user" / "chat_session_20240115_100000_123456"
    assert migrated.exists()
    
    # Check metadata transformation
    with open(migrated / "metadata.json") as f:
        metadata = json.load(f)
    
    assert metadata["title"] == "Test Chat"
    assert metadata["user_id"] == "test_user"
```

### Phase 6: Cutover Process (Day 13-14)

#### 6.1 Incremental Migration
```bash
#!/bin/bash
# migration/cutover.sh

# Step 1: Stop writes to old system
echo "Stopping old system..."
systemctl stop flatfile-chat

# Step 2: Final data sync
echo "Running final migration..."
python migration/run_migration.py --final

# Step 3: Validate migration
echo "Validating migration..."
python migration/validate.py
if [ $? -ne 0 ]; then
    echo "Migration validation failed!"
    exit 1
fi

# Step 4: Start new system
echo "Starting modular platform..."
systemctl start modular-chat

# Step 5: Health check
echo "Running health checks..."
python migration/health_check.py
```

#### 6.2 Rollback Plan
```python
# migration/rollback.py
import shutil
from pathlib import Path
from datetime import datetime

class RollbackManager:
    def __init__(self, backup_path: Path):
        self.backup_path = backup_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_backup(self, source_path: Path):
        """Create backup before migration."""
        backup_dest = self.backup_path / f"backup_{self.timestamp}"
        shutil.copytree(source_path, backup_dest)
        return backup_dest
    
    def rollback(self, backup_name: str, target_path: Path):
        """Rollback to specific backup."""
        backup_source = self.backup_path / backup_name
        if not backup_source.exists():
            raise ValueError(f"Backup {backup_name} not found")
        
        # Clear current data
        if target_path.exists():
            shutil.rmtree(target_path)
        
        # Restore backup
        shutil.copytree(backup_source, target_path)
```

## Migration Patterns

### Pattern 1: Adapter Pattern (Gradual Migration)
```python
# adapters/legacy_adapter.py
class LegacyStorageAdapter:
    """Adapter to use new modules with old API."""
    
    def __init__(self, module_context):
        self.context = module_context
        self._cache = {}
    
    async def create_session(self, user_id: str, session_name: str):
        """Legacy create_session method using new modules."""
        # Use new text_chat module
        chat_module = await self.context.get_module("text_chat")
        session = await chat_module.create_session(
            user_id=user_id,
            title=session_name
        )
        
        # Maintain backward compatibility
        return {
            "session_id": session.id,
            "session_name": session.title,
            "created_at": session.created_at
        }
```

### Pattern 2: Parallel Running
```python
# parallel/dual_writer.py
class DualWriter:
    """Write to both old and new systems during migration."""
    
    def __init__(self, old_storage, new_modules):
        self.old_storage = old_storage
        self.new_modules = new_modules
    
    async def write_message(self, session_id: str, content: str):
        """Write to both systems."""
        # Write to old system
        old_result = await self.old_storage.add_message(
            session_id=session_id,
            message={"content": content}
        )
        
        # Write to new system
        new_result = await self.new_modules["text_chat"].send_message(
            session_id=session_id,
            content=content
        )
        
        return old_result, new_result
```

## Common Issues and Solutions

### Issue 1: Large Data Volumes
**Problem**: Migration takes too long for large datasets.

**Solution**: Batch processing with checkpoints
```python
async def migrate_in_batches(mapper: StorageMapper, batch_size: int = 1000):
    """Migrate data in batches with progress tracking."""
    checkpoint_file = Path("migration_checkpoint.json")
    
    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {"processed": 0, "total": 0}
    
    # Process in batches
    items = list(mapper.get_items_to_migrate())
    checkpoint["total"] = len(items)
    
    for i in range(checkpoint["processed"], len(items), batch_size):
        batch = items[i:i + batch_size]
        await mapper.migrate_batch(batch)
        
        # Update checkpoint
        checkpoint["processed"] = i + len(batch)
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
```

### Issue 2: Schema Differences
**Problem**: Old data doesn't match new schema requirements.

**Solution**: Schema transformation layer
```python
class SchemaTransformer:
    """Transform old schemas to new format."""
    
    def transform_message(self, old_message: Dict) -> Dict:
        """Transform old message format to new."""
        return {
            "id": old_message.get("message_id", f"msg_{time.time()}"),
            "content": old_message.get("content", ""),
            "role": old_message.get("role", "user"),
            "timestamp": old_message.get("timestamp", datetime.now().isoformat()),
            "metadata": {
                "legacy_id": old_message.get("id"),
                **old_message.get("metadata", {})
            }
        }
```

### Issue 3: Missing Dependencies
**Problem**: New modules have unmet dependencies.

**Solution**: Dependency checker
```python
def check_dependencies() -> List[str]:
    """Check for missing dependencies."""
    missing = []
    
    required = {
        "numpy": "1.24.0",
        "faiss-cpu": "1.7.4",
        "aiofiles": "23.0.0"
    }
    
    for package, min_version in required.items():
        try:
            module = importlib.import_module(package.replace("-", "_"))
            # Check version if available
            if hasattr(module, "__version__"):
                if version.parse(module.__version__) < version.parse(min_version):
                    missing.append(f"{package}>={min_version}")
        except ImportError:
            missing.append(package)
    
    return missing
```

## Post-Migration Tasks

### 1. Performance Optimization
```python
# optimization/index_builder.py
async def build_indices(storage_path: Path):
    """Build performance indices after migration."""
    indices = {
        "user_sessions": {},
        "message_search": {},
        "document_registry": {}
    }
    
    # Build user-session index
    sessions_path = storage_path / "text_chat" / "sessions"
    for user_dir in sessions_path.iterdir():
        if user_dir.is_dir():
            user_id = user_dir.name
            sessions = [s.name for s in user_dir.iterdir() if s.is_dir()]
            indices["user_sessions"][user_id] = sessions
    
    # Save indices
    with open(storage_path / "indices" / "master_index.json", "w") as f:
        json.dump(indices, f)
```

### 2. Monitoring Setup
```python
# monitoring/health_check.py
async def health_check_all_modules(context: ModuleContext) -> Dict[str, str]:
    """Check health of all modules."""
    results = {}
    
    modules = [
        "text_chat", "memory", "rag", "multi_agent",
        "tool_use", "persona", "multimodal", "topic_router", "trace_logger"
    ]
    
    for module_name in modules:
        try:
            module = await context.get_module(module_name)
            if module and hasattr(module, "health_check"):
                status = await module.health_check()
                results[module_name] = "healthy" if status else "unhealthy"
            else:
                results[module_name] = "not_loaded"
        except Exception as e:
            results[module_name] = f"error: {str(e)}"
    
    return results
```

### 3. User Communication
```markdown
# Email Template: Migration Complete

Dear User,

We've successfully upgraded our chat platform to a new modular architecture. Here's what's new:

**New Features:**
- Enhanced multi-modal support (images, documents, audio)
- Improved memory and context management
- Multi-agent conversations
- Advanced RAG capabilities

**What's Changed:**
- Faster response times
- Better conversation context
- More reliable service

**Action Required:**
- Please log out and log back in
- Clear your browser cache
- Update any API integrations

If you experience any issues, please contact support.

Best regards,
The Platform Team
```

## Conclusion

This migration guide provides a comprehensive approach to upgrading from Flatfile Database v2 to the modular chat platform. The key to success is:

1. **Incremental Migration**: Module by module
2. **Thorough Testing**: At each phase
3. **Data Validation**: Ensure no data loss
4. **Rollback Capability**: Always have a way back
5. **Clear Communication**: Keep users informed

Follow this guide carefully, and the migration should proceed smoothly with minimal disruption to service.