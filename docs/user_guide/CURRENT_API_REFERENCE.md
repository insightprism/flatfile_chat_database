# API Reference - Current Implementation

This document provides comprehensive API documentation for the improved flatfile chat database system, covering all managers, protocols, and configuration options.

## Table of Contents
1. [Main Storage Manager](#main-storage-manager)
2. [Specialized Managers](#specialized-managers)
3. [Protocol Interfaces](#protocol-interfaces)
4. [Configuration System](#configuration-system)
5. [Dependency Injection](#dependency-injection)
6. [Data Models](#data-models)
7. [Utility Functions](#utility-functions)

## Main Storage Manager

### FFStorageManager

The primary interface for all storage operations. Provides a unified API that coordinates with specialized managers.

```python
class FFStorageManager:
    def __init__(self, config: Optional[FFConfigurationManagerConfigDTO] = None, 
                 backend: Optional[StorageBackend] = None,
                 enable_prismmind: bool = True) -> None
```

#### Core Operations

```python
# Initialization
async def initialize(self) -> bool
    """Initialize the storage system."""

# User Management
async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool
    """Create new user with optional profile."""

async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]
    """Get user profile data."""

async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool
    """Update user profile with provided changes."""

async def user_exists(self, user_id: str) -> bool
    """Check if user exists."""

async def list_users(self) -> List[str]
    """Get list of all user IDs."""

# Session Management
async def create_session(self, user_id: str, title: Optional[str] = None) -> str
    """Create new chat session, returns session ID."""

async def get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]
    """Get session metadata."""

async def delete_session(self, user_id: str, session_id: str) -> bool
    """Delete session and all associated data."""

# Message Operations
async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool
    """Add message to session."""

async def get_all_messages(self, user_id: str, session_id: str) -> List[FFMessageDTO]
    """Get all messages from session."""

# Search Operations
async def advanced_search(self, query: FFSearchQueryDTO) -> List[FFSearchResultDTO]
    """Perform advanced search across sessions."""

async def build_search_index(self, user_id: str) -> Dict[str, Any]
    """Build search index for user."""

# Document Operations
async def list_documents(self, user_id: str, session_id: str) -> List[FFDocumentDTO]
    """List documents in session."""

# Statistics
async def get_session_stats(self, user_id: str, session_id: str) -> Dict[str, Any]
    """Get comprehensive session statistics."""
```

#### Property Access (Lazy-Loaded Services)

```python
@property
def search_engine(self) -> FFSearchManager
    """Lazy-load search engine via dependency injection."""

@property  
def vector_storage(self) -> FFVectorStorageManager
    """Lazy-load vector storage via dependency injection."""

@property
def chunking_engine(self) -> FFChunkingManager
    """Lazy-load chunking engine."""

@property
def embedding_engine(self) -> FFEmbeddingManager
    """Lazy-load embedding engine."""

@property
def document_processor(self) -> FFDocumentProcessingManager
    """Lazy-load document processor via dependency injection."""

@property
def prismmind_processor(self) -> Optional[Any]
    """Lazy-load PrismMind processor if available."""

@property
def prismmind_available(self) -> bool
    """Check if PrismMind integration is available."""
```

## Specialized Managers

### FFUserManager

Handles user profiles and user-specific operations.

```python
class FFUserManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend)

# User Operations
async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool
    """Create new user with optional profile data."""

async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]
    """Retrieve user profile."""

async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool
    """Update user profile with changes."""

async def store_user_profile(self, profile: FFUserProfileDTO) -> bool
    """Store complete user profile object."""

async def user_exists(self, user_id: str) -> bool
    """Check if user exists."""

async def list_users(self) -> List[str]
    """Get list of all user IDs."""
```

### FFSessionManager

Manages chat sessions and messages.

```python
class FFSessionManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend)

# Session Operations
async def create_session(self, user_id: str, session_name: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str
    """Create new chat session."""

async def get_session(self, user_id: str, session_id: str) -> Optional[FFSessionDTO]
    """Retrieve session metadata."""

async def list_sessions(self, user_id: str, limit: Optional[int] = None, 
                       offset: int = 0) -> List[FFSessionDTO]
    """List user sessions."""

async def update_session_metadata(self, user_id: str, session_id: str, 
                                 updates: Dict[str, Any]) -> bool
    """Update session metadata."""

async def session_exists(self, user_id: str, session_id: str) -> bool
    """Check if session exists."""

# Message Operations
async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool
    """Add message to session."""

async def get_messages(self, user_id: str, session_id: str, 
                      limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]
    """Retrieve messages from session."""

async def get_all_messages(self, user_id: str, session_id: str) -> List[FFMessageDTO]
    """Retrieve all messages from session."""

# Statistics
async def get_session_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]
    """Get comprehensive session statistics."""
```

### FFDocumentManager

Handles document storage and retrieval.

```python
class FFDocumentManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend)

# Document Operations
async def store_document(self, user_id: str, session_id: str, filename: str, 
                        content: str, metadata: Optional[Dict[str, Any]] = None) -> str
    """Store document content and metadata, returns document ID."""

async def get_document(self, user_id: str, session_id: str, doc_id: str) -> Optional[FFDocumentDTO]
    """Retrieve document by ID."""

async def list_documents(self, user_id: str, session_id: str) -> List[FFDocumentDTO]
    """List all documents in session (without content)."""

async def update_document_analysis(self, user_id: str, session_id: str,
                                  doc_id: str, analysis: Dict[str, Any]) -> bool
    """Update document with analysis results."""

async def delete_document(self, user_id: str, session_id: str, doc_id: str) -> bool
    """Delete document and its metadata."""

async def get_document_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]
    """Get document statistics for session."""
```

### FFContextManager

Manages situational context and context history.

```python
class FFContextManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend)

# Context Operations
async def store_context(self, user_id: str, session_id: str, 
                       context: FFSituationalContextDTO) -> str
    """Store situational context, returns context ID."""

async def get_context(self, user_id: str, session_id: str) -> Optional[FFSituationalContextDTO]
    """Get current context for session."""

async def update_context(self, user_id: str, session_id: str, 
                        updates: Dict[str, Any]) -> bool
    """Update context with changes."""

async def create_context_snapshot(self, user_id: str, session_id: str, 
                                 snapshot_name: str) -> str
    """Create context snapshot, returns snapshot ID."""

async def restore_context_snapshot(self, user_id: str, session_id: str, 
                                  snapshot_id: str) -> bool
    """Restore context from snapshot."""

async def list_context_snapshots(self, user_id: str, session_id: str) -> List[Dict[str, str]]
    """List available context snapshots."""

async def delete_context(self, user_id: str, session_id: str, snapshot_id: str) -> bool
    """Delete context snapshot."""

async def get_context_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]
    """Get context statistics for session."""
```

### FFPanelManager

Manages multi-persona panels and personas.

```python
class FFPanelManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend)

# Panel Operations
async def create_panel(self, user_id: str, personas: List[str], panel_name: str,
                      metadata: Optional[Dict[str, Any]] = None) -> str
    """Create multi-persona panel, returns panel ID."""

async def get_panel(self, user_id: str, panel_id: str) -> Optional[FFPanelDTO]
    """Retrieve panel metadata."""

async def list_panels(self, user_id: str) -> List[FFPanelDTO]
    """List user panels."""

async def panel_exists(self, user_id: str, panel_id: str) -> bool
    """Check if panel exists."""

# Panel Message Operations
async def add_panel_message(self, user_id: str, panel_id: str, 
                           message: FFPanelMessageDTO) -> bool
    """Add message to panel."""

async def get_panel_messages(self, user_id: str, panel_id: str, 
                            limit: Optional[int] = None, offset: int = 0) -> List[FFPanelMessageDTO]
    """Retrieve messages from panel."""

# Panel Insights
async def save_panel_insight(self, panel_id: str, insight: FFPanelInsightDTO) -> bool
    """Save panel insight."""

# Persona Operations
async def store_persona(self, persona: FFPersonaDTO, global_persona: bool = False) -> bool
    """Store persona (global or user-specific)."""

async def get_persona(self, persona_id: str, user_id: Optional[str] = None) -> Optional[FFPersonaDTO]
    """Retrieve persona by ID."""

async def list_personas(self, user_id: Optional[str] = None, 
                       include_global: bool = True) -> List[FFPersonaDTO]
    """List available personas."""
```

### FFSearchManager

Provides search capabilities across sessions.

```python
class FFSearchManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO)

# Search Operations
async def search(self, query: FFSearchQueryDTO) -> List[FFSearchResultDTO]
    """Perform search with advanced query parameters."""

async def extract_entities(self, text: str) -> Dict[str, List[str]]
    """Extract entities from text."""

async def build_search_index(self, user_id: str) -> Dict[str, Any]
    """Build search index for user."""

# Vector Search (if enabled)
async def vector_search(self, query: FFSearchQueryDTO) -> List[FFSearchResultDTO]
    """Perform vector similarity search."""

async def search_enhanced(self, query: FFSearchQueryDTO) -> List[FFSearchResultDTO]
    """Perform hybrid search (text + vector)."""
```

### FFStreamingManager

Provides streaming capabilities for large datasets.

```python
class FFMessageStreamerManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, 
                 stream_config: Optional[FFStreamConfigDTO] = None)

# Message Streaming
async def stream_messages(self, user_id: str, session_id: str,
                         start_offset: int = 0,
                         max_messages: Optional[int] = None) -> AsyncIterator[List[FFMessageDTO]]
    """Stream messages in chunks from a session."""

async def stream_messages_reverse(self, user_id: str, session_id: str,
                                 limit: Optional[int] = None) -> AsyncIterator[List[FFMessageDTO]]
    """Stream messages in reverse order (most recent first)."""

async def parallel_stream_sessions(self, user_id: str, 
                                  session_ids: List[str]) -> AsyncIterator[Tuple[str, List[FFMessageDTO]]]
    """Stream messages from multiple sessions in parallel."""

class FFExportStreamerManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO, 
                 stream_config: Optional[FFStreamConfigDTO] = None)

# Export Streaming
async def stream_session_export(self, user_id: str, session_id: str,
                               include_documents: bool = True,
                               include_context: bool = True) -> AsyncIterator[Dict[str, Any]]
    """Stream session data for export."""

async def stream_user_export(self, user_id: str,
                            session_limit: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]
    """Stream all user data for export."""

class FFLazyLoaderManager:
    def __init__(self, config: FFConfigurationManagerConfigDTO)

# Lazy Loading
async def get_message_lazy(self, user_id: str, session_id: str, 
                          message_index: int) -> Optional[FFMessageDTO]
    """Lazily load a single message by index."""

async def get_session_metadata_lazy(self, user_id: str, 
                                   session_id: str) -> Optional[Dict[str, Any]]
    """Lazily load session metadata."""

def clear_cache(self) -> None
    """Clear the lazy loading cache."""
```

## Protocol Interfaces

### StorageProtocol

Main storage operations interface.

```python
class StorageProtocol(ABC):
    @abstractmethod
    async def initialize(self) -> bool
    
    @abstractmethod
    async def create_user(self, user_id: str, profile: Optional[Dict[str, Any]] = None) -> bool
    
    @abstractmethod
    async def create_session(self, user_id: str, title: Optional[str] = None) -> str
    
    @abstractmethod
    async def add_message(self, user_id: str, session_id: str, message: FFMessageDTO) -> bool
    
    @abstractmethod
    async def get_messages(self, user_id: str, session_id: str, 
                          limit: Optional[int] = None, offset: int = 0) -> List[FFMessageDTO]
```

### SearchProtocol

Search operations interface.

```python
class SearchProtocol(ABC):
    @abstractmethod
    async def search(self, query: str, user_id: Optional[str] = None, 
                    session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]
    
    @abstractmethod
    async def build_search_index(self, user_id: str) -> Dict[str, Any]
```

### BackendProtocol

Low-level storage backend interface.

```python
class BackendProtocol(ABC):
    @abstractmethod
    async def initialize(self) -> bool
    
    @abstractmethod
    async def read(self, key: str) -> Optional[bytes]
    
    @abstractmethod
    async def write(self, key: str, data: bytes) -> bool
    
    @abstractmethod
    async def delete(self, key: str) -> bool
    
    @abstractmethod
    async def exists(self, key: str) -> bool
    
    @abstractmethod
    async def list_keys(self, prefix: str = "", pattern: Optional[str] = None) -> List[str]
```

## Configuration System

### FFConfigurationManagerConfigDTO

Main configuration container composing all domain-specific configs.

```python
@dataclass
class FFConfigurationManagerConfigDTO:
    storage: FFStorageConfig          # File paths and storage settings
    search: FFSearchConfig            # Search behavior and indexing
    vector: FFVectorConfig            # Vector embeddings and similarity
    document: FFDocumentConfig        # Document processing settings
    locking: FFLockingConfig          # File locking configuration
    panel: FFPanelConfig              # Panel and persona settings
    runtime: FFRuntimeConfig          # Runtime behavior and validation
```

### Configuration Loading

```python
def load_config(config_path: Optional[Union[str, Path]] = None, 
               environment: Optional[str] = None) -> FFConfigurationManagerConfigDTO
    """Load configuration with environment-specific overrides."""

# Environment-aware loading
config = load_config(environment="development")  # Loads dev-specific settings
config = load_config(environment="production")   # Loads prod-specific settings
```

### Key Configuration Classes

#### FFStorageConfig
```python
@dataclass
class FFStorageConfig:
    base_path: str = "./data"
    user_data_directory_name: str = "users"
    session_id_prefix: str = "chat_session_"
    messages_filename: str = "messages.jsonl"
    session_metadata_filename: str = "session.json"
    user_profile_filename: str = "profile.json"
    enable_file_locking: bool = True
    max_message_size_bytes: int = 1_000_000  # 1MB
    max_document_size_bytes: int = 10_000_000  # 10MB
```

#### FFRuntimeConfig
```python
@dataclass  
class FFRuntimeConfig:
    # Business logic thresholds
    large_session_threshold_bytes: int = 10_000_000  # 10MB
    storage_default_message_limit: int = 1000
    storage_default_session_limit: int = 100
    cache_size_limit: int = 100
    
    # Validation rules (configurable)
    user_id_min_length: int = 3
    user_id_max_length: int = 50
    session_name_min_length: int = 1
    session_name_max_length: int = 200
    filename_min_length: int = 1
    filename_max_length: int = 255
    message_content_min_length: int = 1
    message_content_max_length: int = 100_000
    document_content_max_length: int = 10_000_000
    
    # File extensions
    document_content_extension: str = ".txt"
    context_file_extension: str = ".json"
    persona_file_extension: str = ".json"
    insight_file_extension: str = ".json"
```

## Dependency Injection

### FFDependencyInjectionManager

Container for service registration and resolution.

```python
class FFDependencyInjectionManager:
    def __init__(self) -> None
    
    # Service Registration
    def register(self, interface: Type[T],
                 implementation: Optional[Type[T]] = None,
                 factory: Optional[Callable] = None,
                 instance: Optional[T] = None,
                 lifetime: str = FFServiceLifetime.TRANSIENT) -> None
    
    def register_singleton(self, interface: Type[T], 
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable] = None,
                          instance: Optional[T] = None) -> None
    
    def register_transient(self, interface: Type[T],
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable] = None) -> None
    
    def register_scoped(self, interface: Type[T],
                       implementation: Optional[Type[T]] = None,
                       factory: Optional[Callable] = None) -> None
    
    # Service Resolution
    def resolve(self, interface: Type[T], scope: Optional[FFServiceScope] = None) -> T
    
    async def resolve_async(self, interface: Type[T], 
                           scope: Optional[FFServiceScope] = None) -> T
    
    # Scope Management
    @asynccontextmanager
    async def create_scope(self) -> AsyncGenerator[FFServiceScope, None]
    
    # Container Introspection
    def is_registered(self, interface: Type) -> bool
    
    def get_registration_info(self, interface: Type) -> Dict[str, Any]
    
    def get_all_registered(self) -> List[Type]
    
    def clear(self) -> None
```

### Global Container Functions

```python
def ff_create_application_container(config_path: Optional[Union[str, Path]] = None,
                                   environment: Optional[str] = None) -> FFDependencyInjectionManager
    """Create and configure application container with all services."""

def ff_get_container() -> FFDependencyInjectionManager
    """Get the global dependency injection container instance."""

def ff_set_container(container: FFDependencyInjectionManager) -> None
    """Set the global container instance."""

def ff_clear_global_container() -> None
    """Clear the global container (creates fresh on next access)."""
```

## Data Models

### Core DTOs

#### FFMessageDTO
```python
@dataclass
class FFMessageDTO:
    role: str                           # "user", "assistant", "system"
    content: str                        # Message content
    timestamp: Optional[str] = None     # ISO format timestamp
    message_id: Optional[str] = None    # Unique message identifier
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFMessageDTO'
    
    def to_dict(self) -> Dict[str, Any]
```

#### FFSessionDTO
```python
@dataclass
class FFSessionDTO:
    session_id: str                     # Unique session identifier
    user_id: str                        # Owner user ID
    title: Optional[str] = None         # Session title/name
    created_at: Optional[str] = None    # ISO format timestamp
    updated_at: Optional[str] = None    # ISO format timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def session_name(self) -> str       # Alias for title
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFSessionDTO'
    
    def to_dict(self) -> Dict[str, Any]
```

#### FFDocumentDTO
```python
@dataclass
class FFDocumentDTO:
    document_id: str                    # Unique document identifier
    filename: str                       # Original filename
    content: str                        # Document content
    size: int                          # Content size in bytes
    content_type: str                  # File extension/MIME type
    uploaded_at: Optional[str] = None   # ISO format timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFDocumentDTO'
    
    def to_dict(self) -> Dict[str, Any]
```

### Search DTOs

#### FFSearchQueryDTO
```python
@dataclass
class FFSearchQueryDTO:
    query: str                          # Search query string
    user_id: Optional[str] = None       # Limit to specific user
    session_ids: Optional[List[str]] = None  # Limit to specific sessions
    start_date: Optional[datetime] = None    # Date range start
    end_date: Optional[datetime] = None      # Date range end
    message_roles: Optional[List[str]] = None # Filter by message roles
    entities: Optional[Dict[str, List[str]]] = None  # Entity filters
    include_documents: bool = False     # Include document content
    include_context: bool = False       # Include context data
    max_results: int = 100             # Maximum results to return
    min_relevance_score: float = 0.0   # Minimum relevance threshold
    
    # Vector search parameters
    use_vector_search: bool = False     # Enable vector similarity
    similarity_threshold: float = 0.7   # Vector similarity threshold
    embedding_provider: str = "nomic-ai"  # Embedding provider
    hybrid_search: bool = False         # Combine text + vector search
    vector_weight: float = 0.5         # Weight for vector vs text scores
```

#### FFSearchResultDTO
```python
@dataclass
class FFSearchResultDTO:
    id: str                            # Result identifier
    type: str                          # "message", "document", "context"
    content: str                       # Result content/snippet
    user_id: str                       # Associated user
    session_id: str                    # Associated session
    relevance_score: float             # Relevance score (0.0-1.0)
    highlights: List[Tuple[int, int]] = field(default_factory=list)  # Highlight positions
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'FFSearchResultDTO') -> bool  # For sorting by relevance
```

## Utility Functions

### Validation Functions (`ff_utils.ff_validation`)

```python
def validate_user_id(user_id: str, config: FFConfigurationManagerConfigDTO) -> List[str]
    """Validate user ID using configurable rules."""

def validate_session_name(session_name: str, config: FFConfigurationManagerConfigDTO) -> List[str]
    """Validate session name using configurable rules."""

def validate_filename(filename: str, config: FFConfigurationManagerConfigDTO) -> List[str]
    """Validate filename using configurable rules."""

def validate_message_content(content: str, config: FFConfigurationManagerConfigDTO) -> List[str]
    """Validate message content using configurable rules."""

def validate_document_content(content: str, config: FFConfigurationManagerConfigDTO) -> List[str]
    """Validate document content using configurable rules."""
```

### File Operations (`ff_utils.ff_file_ops`)

```python
async def ff_write_json(file_path: Path, data: Dict[str, Any], 
                       config: FFConfigurationManagerConfigDTO) -> bool
    """Write JSON data to file with locking support."""

async def ff_read_json(file_path: Path, 
                      config: FFConfigurationManagerConfigDTO) -> Optional[Dict[str, Any]]
    """Read JSON data from file."""

async def ff_append_jsonl(file_path: Path, data: Dict[str, Any], 
                         config: FFConfigurationManagerConfigDTO) -> bool
    """Append JSON line to JSONL file."""

async def ff_read_jsonl(file_path: Path, config: FFConfigurationManagerConfigDTO,
                       limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]
    """Read JSONL file with pagination support."""

async def ff_read_jsonl_paginated(file_path: Path, config: FFConfigurationManagerConfigDTO,
                                 page_size: int = 100, page: int = 0) -> Dict[str, Any]
    """Read JSONL file with pagination metadata."""
```

### Path Utilities (`ff_utils.ff_path_utils`)

```python
def ff_get_user_path(base_path: Path, user_id: str, 
                    config: FFConfigurationManagerConfigDTO) -> Path
    """Get user directory path."""

def ff_get_session_path(base_path: Path, user_id: str, session_id: str,
                       config: FFConfigurationManagerConfigDTO) -> Path
    """Get session directory path."""

def ff_get_documents_path(base_path: Path, user_id: str, session_id: str,
                         config: FFConfigurationManagerConfigDTO) -> Path
    """Get documents directory path."""

def ff_generate_session_id(config: FFConfigurationManagerConfigDTO) -> str
    """Generate unique session ID with configured prefix."""

def ff_sanitize_filename(filename: str) -> str
    """Sanitize filename for safe storage."""
```

### Logging (`ff_utils.ff_logging`)

```python
def get_logger(name: str) -> logging.Logger
    """Get configured logger instance."""

def setup_logging(level: str = "INFO", 
                 format_string: Optional[str] = None) -> None
    """Setup logging configuration."""
```

## Usage Examples

### Basic Operations

```python
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO

# Initialize storage
storage = FFStorageManager()
await storage.initialize()

# Create user and session
await storage.create_user("alice", {"name": "Alice Smith"})
session_id = await storage.create_session("alice", "My Chat")

# Add messages
message = FFMessageDTO(role="user", content="Hello!")
await storage.add_message("alice", session_id, message)

# Search messages
from ff_class_configs.ff_chat_entities_config import FFSearchQueryDTO
query = FFSearchQueryDTO(query="hello", user_id="alice")
results = await storage.advanced_search(query)
```

### Using Dependency Injection

```python
from ff_dependency_injection_manager import ff_get_container
from ff_protocols import StorageProtocol, SearchProtocol

# Get services via DI
container = ff_get_container()
storage = container.resolve(StorageProtocol)
search = container.resolve(SearchProtocol)

# Check container state
print("Services registered:", len(container.get_all_registered()))
print("Storage info:", container.get_registration_info(StorageProtocol))
```

### Configuration Customization

```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Load environment-specific config
config = load_config(environment="development")

# Create storage with custom config
storage = FFStorageManager(config=config)
await storage.initialize()
```

This API reference covers all the major components and their interfaces in the current implementation. All methods include proper type annotations and comprehensive documentation.