# Flatfile Chat Database System - Design Specification

## 1. Executive Summary

The Flatfile Chat Database System is a file-based storage solution designed to provide zero-configuration, human-readable data persistence for AI chat applications. It serves as a lightweight alternative to traditional databases while maintaining a clear migration path for future scaling needs.

## 2. Purpose and Objectives

### 2.1 Purpose
To create a simple, modular, and extensible storage system that:
- Requires no database setup or administration
- Stores all chat-related data in an organized file structure
- Provides a database-like API for easy integration
- Supports complex AI chat scenarios including multi-persona panels and document analysis

### 2.2 Core Objectives
1. **Zero Configuration**: Works immediately upon installation
2. **Human Readable**: All data stored in JSON/JSONL format for easy debugging
3. **Modular Design**: Clean separation of concerns for easy maintenance
4. **Migration Ready**: Abstract interface allows seamless transition to database backends
5. **Feature Complete**: Supports all common chat application requirements
6. **Performance**: Efficient for small to medium-scale deployments
7. **Data Integrity**: Atomic operations prevent data corruption

## 3. System Architecture

### 3.1 High-Level Architecture
```
┌─────────────────────────────────────────┐
│         Chat Application Layer          │
├─────────────────────────────────────────┤
│          Storage API Interface          │
│         (StorageManager Class)          │
├─────────────────────────────────────────┤
│           Storage Backend               │
│    ┌──────────────┬──────────────┐     │
│    │  Flatfile    │   Database   │     │
│    │  Backend     │   Backend    │     │
│    │  (Current)   │   (Future)   │     │
│    └──────────────┴──────────────┘     │
└─────────────────────────────────────────┘
```

### 3.2 Component Architecture
```
flatfile_chat_database/
├── storage.py          # Main StorageManager with public API
├── models.py           # Data models (Message, Session, Panel, etc.)
├── backends/
│   ├── base.py        # Abstract backend interface
│   ├── flatfile.py    # Flatfile implementation
│   └── database.py    # Future database implementation
├── utils/
│   ├── file_ops.py    # Atomic file operations
│   ├── path_utils.py  # Path management
│   └── json_utils.py  # JSON/JSONL handling
└── config.py          # Configuration management
```

## 4. Data Models

### 4.1 Core Models

```python
@dataclass
class Message:
    """Individual chat message"""
    id: str                     # Unique message ID
    role: str                   # "user", "assistant", or persona_id
    content: str                # Message content
    timestamp: str              # ISO format timestamp
    attachments: List[str]      # File references
    metadata: Dict[str, Any]    # Additional data
    
@dataclass
class Session:
    """Chat session metadata"""
    id: str                     # Session ID (format: chat_session_YYYYMMDD_HHMMSS)
    user_id: str                # Owner of the session
    title: str                  # Human-readable title
    created_at: str             # ISO format timestamp
    updated_at: str             # ISO format timestamp
    metadata: Dict[str, Any]    # Session configuration
    
@dataclass
class Panel:
    """Multi-persona panel session"""
    id: str                     # Panel ID (format: panel_YYYYMMDD_HHMMSS)
    type: str                   # "multi_persona", "focus_group", etc.
    personas: List[str]         # Active persona IDs
    created_at: str             # ISO format timestamp
    config: Dict[str, Any]      # Panel configuration
    
@dataclass
class SituationalContext:
    """Conversation context snapshot"""
    summary: str                # Context summary
    key_points: List[str]       # Important points
    entities: Dict[str, List[str]]  # Extracted entities
    timestamp: str              # When context was captured
    confidence: float           # Context confidence score
    
@dataclass
class Document:
    """Document metadata"""
    filename: str               # Original filename
    path: str                   # Storage path
    mime_type: str              # Content type
    size: int                   # File size in bytes
    uploaded_at: str            # Upload timestamp
    analysis: Dict[str, Any]    # Analysis results
```

## 5. Directory Structure

### 5.1 Standard Chat Sessions
```
data/
├── {user_id}/
│   ├── profile.json                        # User profile and preferences
│   ├── chat_session_YYYYMMDD_HHMMSS/      # Individual chat session
│   │   ├── session.json                    # Session metadata
│   │   ├── messages.jsonl                  # Chat messages (append-only)
│   │   ├── situational_context.json        # Current context
│   │   ├── context_history/                # Context evolution
│   │   │   └── context_YYYYMMDD_HHMMSS.json
│   │   └── documents/                      # Uploaded files
│   │       ├── metadata.json               # Document metadata
│   │       └── {files}                     # Actual files
│   └── personas/                           # User-specific personas
│       └── {persona_id}.json
```

### 5.2 Panel Sessions
```
data/
├── panel_sessions/
│   └── panel_YYYYMMDD_HHMMSS/
│       ├── panel.json                      # Panel configuration
│       ├── messages.jsonl                  # All panel messages
│       ├── personas/                       # Active persona snapshots
│       │   └── {persona_id}.json
│       └── insights/                       # Analysis and conclusions
│           └── insight_YYYYMMDD_HHMMSS.json
```

### 5.3 Global Resources
```
data/
├── personas_global/                        # Shared personas
│   └── {persona_id}.json
└── system/                                # System configuration
    └── config.json
```

## 6. Configuration System

### 6.1 Configuration Design Principles
1. **No hardcoded values** - All configurable values in config file
2. **Meaningful names** - Clear, specific variable names
3. **Grouped by function** - Related settings together
4. **Environment aware** - Support for dev/staging/prod configs
5. **Type safe** - Using dataclasses for validation

### 6.2 Configuration Structure

```python
@dataclass
class StorageConfig:
    """Main configuration for storage system"""
    
    # Storage locations
    storage_base_path: str = "./data"
    user_data_directory_name: str = "users"  # Not just "users" - be specific
    panel_sessions_directory_name: str = "panel_sessions"
    global_personas_directory_name: str = "personas_global"
    system_config_directory_name: str = "system"
    
    # File naming patterns
    session_id_prefix: str = "chat_session"
    panel_id_prefix: str = "panel"
    session_timestamp_format: str = "%Y%m%d_%H%M%S"
    context_snapshot_timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # File names
    user_profile_filename: str = "profile.json"
    session_metadata_filename: str = "session.json"
    messages_filename: str = "messages.jsonl"
    situational_context_filename: str = "situational_context.json"
    document_metadata_filename: str = "metadata.json"
    
    # Storage limits
    max_message_size_bytes: int = 1_048_576  # 1MB
    max_document_size_bytes: int = 104_857_600  # 100MB
    max_messages_per_session: int = 100_000
    max_sessions_per_user: int = 1_000
    max_context_history_snapshots: int = 100
    
    # Performance settings
    message_pagination_default_limit: int = 100
    session_list_default_limit: int = 50
    search_results_default_limit: int = 20
    jsonl_read_buffer_size: int = 8192
    
    # Behavior settings
    atomic_write_temp_suffix: str = ".tmp"
    backup_before_delete: bool = True
    auto_cleanup_empty_directories: bool = True
    validate_json_on_read: bool = True
    
    # Search configuration
    search_include_message_content: bool = True
    search_include_context: bool = True
    search_include_metadata: bool = False
    full_text_search_min_word_length: int = 3
    
    # Panel configuration
    panel_max_personas: int = 10
    panel_insight_retention_days: int = 90
    panel_message_threading_enabled: bool = True
    
    # Document handling
    allowed_document_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".md", ".json", ".csv", ".png", ".jpg", ".jpeg"
    ])
    document_storage_subdirectory_name: str = "documents"
    document_analysis_subdirectory_name: str = "analysis"
    
    # Context management
    context_history_subdirectory_name: str = "context_history"
    context_summary_max_length: int = 500
    context_key_points_max_count: int = 10
    context_confidence_threshold: float = 0.7

@dataclass
class DevelopmentConfig(StorageConfig):
    """Development environment overrides"""
    storage_base_path: str = "./dev_data"
    validate_json_on_read: bool = True
    backup_before_delete: bool = True
    
@dataclass
class ProductionConfig(StorageConfig):
    """Production environment overrides"""
    storage_base_path: str = "/var/lib/chatdb/data"
    validate_json_on_read: bool = False  # Performance
    message_pagination_default_limit: int = 50
    search_results_default_limit: int = 10
```

### 6.3 Configuration Loading

```python
def load_config(config_path: str = None, environment: str = None) -> StorageConfig:
    """Load configuration from file and environment"""
    
    # Default config
    config = StorageConfig()
    
    # Load from JSON file if provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = json.load(f)
            config = StorageConfig(**config_dict)
    
    # Apply environment-specific overrides
    if environment == "development":
        config = DevelopmentConfig()
    elif environment == "production":
        config = ProductionConfig()
    
    # Allow environment variables to override
    for field in dataclasses.fields(config):
        env_var = f"CHATDB_{field.name.upper()}"
        if env_var in os.environ:
            setattr(config, field.name, os.environ[env_var])
    
    return config
```

### 6.4 Usage Example

```python
# Load configuration
config = load_config(
    config_path="./config/storage.json",
    environment=os.getenv("ENVIRONMENT", "development")
)

# Initialize storage with config
storage = StorageManager(config=config)

# Access configuration values
max_size = config.max_document_size_bytes
session_prefix = config.session_id_prefix
```

## 7. API Design

### 7.1 Main Storage Interface

```python
class StorageManager:
    """Main API interface for all storage operations"""
    
    def __init__(self, config: StorageConfig = None):
        """Initialize storage with configuration"""
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.storage_base_path)
    
    # === User Management ===
    async def create_user(self, user_id: str, profile: Dict = None) -> bool:
        """Create new user with optional profile"""
        
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile"""
        
    async def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """Update user profile"""
    
    # === Session Management ===
    async def create_session(self, user_id: str, title: str = None) -> str:
        """Create new chat session, returns session_id"""
        
    async def get_session(self, user_id: str, session_id: str) -> Optional[Session]:
        """Get session metadata"""
        
    async def update_session(self, user_id: str, session_id: str, updates: Dict) -> bool:
        """Update session metadata"""
        
    async def list_sessions(self, user_id: str, limit: int = None, offset: int = 0) -> List[Session]:
        """List user's sessions with pagination"""
        
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete session and all associated data"""
    
    # === Message Management ===
    async def add_message(self, user_id: str, session_id: str, message: Message) -> bool:
        """Add message to session"""
        
    async def get_messages(self, user_id: str, session_id: str, 
                          limit: int = None, offset: int = 0) -> List[Message]:
        """Get messages with pagination"""
        
    async def search_messages(self, user_id: str, query: str, 
                            session_id: str = None) -> List[Message]:
        """Search messages across sessions"""
    
    # === Document Management ===
    async def save_document(self, user_id: str, session_id: str, 
                          filename: str, content: bytes, 
                          metadata: Dict = None) -> str:
        """Save document and return storage path"""
        
    async def get_document(self, user_id: str, session_id: str, 
                         filename: str) -> Optional[bytes]:
        """Retrieve document content"""
        
    async def list_documents(self, user_id: str, session_id: str) -> List[Document]:
        """List all documents in session"""
        
    async def update_document_analysis(self, user_id: str, session_id: str,
                                     filename: str, analysis: Dict) -> bool:
        """Store document analysis results"""
    
    # === Context Management ===
    async def update_context(self, user_id: str, session_id: str, 
                           context: SituationalContext) -> bool:
        """Update current situational context"""
        
    async def get_context(self, user_id: str, session_id: str) -> Optional[SituationalContext]:
        """Get current context"""
        
    async def save_context_snapshot(self, user_id: str, session_id: str,
                                  context: SituationalContext) -> bool:
        """Save context to history"""
        
    async def get_context_history(self, user_id: str, session_id: str,
                                limit: int = None) -> List[SituationalContext]:
        """Get context evolution history"""
    
    # === Panel Management ===
    async def create_panel(self, panel_type: str, personas: List[str],
                         config: Dict = None) -> str:
        """Create multi-persona panel session"""
        
    async def add_panel_message(self, panel_id: str, message: Message) -> bool:
        """Add message to panel"""
        
    async def get_panel_messages(self, panel_id: str, 
                               limit: int = None) -> List[Message]:
        """Get panel conversation"""
        
    async def save_panel_insight(self, panel_id: str, insight: Dict) -> bool:
        """Save panel analysis or conclusion"""
    
    # === Persona Management ===
    async def save_persona(self, persona_id: str, data: Dict, 
                         user_id: str = None) -> bool:
        """Save global or user persona"""
        
    async def get_persona(self, persona_id: str, 
                        user_id: str = None) -> Optional[Dict]:
        """Get persona definition"""
        
    async def list_personas(self, user_id: str = None) -> List[Dict]:
        """List available personas"""
```

### 7.2 Backend Interface

```python
class StorageBackend(ABC):
    """Abstract backend interface for storage operations"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
    
    @abstractmethod
    async def read(self, key: str) -> Optional[bytes]:
        """Read data by key"""
        
    @abstractmethod
    async def write(self, key: str, data: bytes) -> bool:
        """Write data with key"""
        
    @abstractmethod
    async def append(self, key: str, data: bytes) -> bool:
        """Append data to existing key"""
        
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data by key"""
        
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        
    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix"""
```

## 8. Implementation Details

### 8.1 Configuration-Driven Operations

All operations use configuration values:
```python
class StorageManager:
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.storage_base_path)
        
    def _get_user_path(self, user_id: str) -> Path:
        """Build user path from config"""
        return self.base_path / user_id
        
    def _get_session_path(self, user_id: str, session_id: str) -> Path:
        """Build session path from config"""
        return self._get_user_path(user_id) / session_id
        
    def generate_session_id(self) -> str:
        """Generate session ID using configured format"""
        timestamp = datetime.now().strftime(self.config.session_timestamp_format)
        return f"{self.config.session_id_prefix}_{timestamp}"
        
    async def save_document(self, user_id: str, session_id: str, 
                          filename: str, content: bytes) -> str:
        """Save document with size validation from config"""
        if len(content) > self.config.max_document_size_bytes:
            raise ValidationError(
                f"Document size {len(content)} exceeds limit of "
                f"{self.config.max_document_size_bytes} bytes"
            )
        
        # Check allowed extensions
        ext = Path(filename).suffix.lower()
        if ext not in self.config.allowed_document_extensions:
            raise ValidationError(
                f"File extension {ext} not allowed. "
                f"Allowed: {self.config.allowed_document_extensions}"
            )
        
        # Save to configured documents directory
        doc_dir = (self._get_session_path(user_id, session_id) / 
                   self.config.document_storage_subdirectory_name)
        doc_dir.mkdir(parents=True, exist_ok=True)
        # ... rest of implementation
```

### 8.2 Atomic Operations
All write operations use atomic file operations to prevent data corruption:
```python
def atomic_write(path: Path, data: bytes, config: StorageConfig):
    temp_path = path.with_suffix(config.atomic_write_temp_suffix)
    temp_path.write_bytes(data)
    temp_path.rename(path)  # Atomic on POSIX systems
```

### 8.3 JSONL Handling
Messages use JSONL format for efficient append operations:
```python
def append_jsonl(path: Path, entry: Dict, config: StorageConfig):
    """Append with configured buffer size"""
    with open(path, 'a', buffering=config.jsonl_read_buffer_size) as f:
        json.dump(entry, f)
        f.write('\n')
        
async def read_jsonl_paginated(path: Path, config: StorageConfig, 
                             limit: int = None, offset: int = 0) -> List[Dict]:
    """Read JSONL with pagination from config"""
    limit = limit or config.message_pagination_default_limit
    entries = []
    
    with open(path, 'r', buffering=config.jsonl_read_buffer_size) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(entries) >= limit:
                break
            entries.append(json.loads(line))
    
    return entries
```

### 8.4 Error Handling
Consistent error handling with custom exceptions:
```python
class StorageError(Exception): pass
class NotFoundError(StorageError): pass
class ValidationError(StorageError): pass
class IntegrityError(StorageError): pass
class ConfigurationError(StorageError): pass
class StorageLimitExceeded(StorageError): pass
```

## 9. Use Cases

### 9.1 Basic Chat Session
```python
# Initialize storage with configuration
config = StorageConfig(
    storage_base_path="/var/myapp/data",
    max_messages_per_session=50000,
    message_pagination_default_limit=200
)
storage = StorageManager(config=config)

# Create user and session
await storage.create_user("john_doe")
session_id = await storage.create_session("john_doe", "Python Help")

# Chat interaction
user_msg = Message(role="user", content="How do I read a file in Python?")
await storage.add_message("john_doe", session_id, user_msg)

assistant_msg = Message(role="assistant", content="You can use the open() function...")
await storage.add_message("john_doe", session_id, assistant_msg)
```

### 8.2 Document Analysis
```python
# Upload document
doc_path = await storage.save_document(
    "john_doe", session_id, "report.pdf", pdf_bytes
)

# Store analysis
await storage.update_document_analysis(
    "john_doe", session_id, "report.pdf",
    {"summary": "Quarterly report", "sentiment": "positive"}
)
```

### 8.3 Multi-Persona Panel
```python
# Create panel
panel_id = await storage.create_panel(
    "analysis_team",
    ["analyst_ai", "critic_ai", "summarizer_ai"]
)

# Panel discussion
msg = Message(role="analyst_ai", content="Based on the data...")
await storage.add_panel_message(panel_id, msg)
```

### 8.4 Context Tracking
```python
# Update context
context = SituationalContext(
    summary="Discussing Python file operations",
    key_points=["User is beginner", "Needs simple examples"],
    entities={"topics": ["file I/O", "Python"], "level": ["beginner"]}
)
await storage.update_context("john_doe", session_id, context)
```

## 9. Migration Strategy

### 9.1 Database Migration Path
The system is designed for easy migration to a database backend:

1. **Implement Database Backend**: Create new backend implementing StorageBackend interface
2. **Data Export**: Built-in export utilities to extract all data
3. **Schema Mapping**: Clear mapping from file structure to database tables
4. **Gradual Migration**: Support running both backends simultaneously

### 9.2 Export Format
```python
async def export_all_data(self) -> Dict:
    """Export all data in database-ready format"""
    return {
        "users": [...],
        "sessions": [...],
        "messages": [...],
        "documents": [...],
        "personas": [...]
    }
```

## 10. Performance Considerations

### 10.1 Scalability Limits
- Suitable for: Up to 1000 users, 100K messages per session
- Performance degrades with: Very large individual files, millions of messages

### 10.2 Optimization Strategies
- Message pagination prevents loading entire history
- JSONL format allows streaming reads
- Directory structure enables efficient file system operations
- Optional compression for large documents

## 11. Security Considerations

### 11.1 Access Control
- File system permissions provide basic security
- User isolation through directory structure
- No built-in authentication (handled by application layer)

### 11.2 Data Privacy
- All data stored in plain text (encryption is application responsibility)
- Sensitive data should be encrypted before storage
- Document sanitization before storage

## 12. Future Enhancements

### 12.1 Planned Features
- Built-in backup and restore
- Data compression options
- Search indexing for better performance
- WebSocket support for real-time updates
- Multi-node synchronization

### 12.2 Backend Options
- SQLite for single-user improvements
- PostgreSQL for full database features
- Redis for caching layer
- S3-compatible storage for cloud deployment

## 13. Success Criteria

The system will be considered successful when it:
1. Provides stable storage for chat applications
2. Requires zero configuration to start using
3. Handles all defined use cases reliably
4. Maintains data integrity under concurrent access
5. Offers clear migration path to database systems
6. Achieves sub-100ms response times for common operations
7. Supports at least 100 concurrent sessions without degradation