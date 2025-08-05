# API Reference

Complete API documentation for the Flatfile Chat Database system.

## 📚 Overview

This reference covers all public methods, classes, and interfaces available in the Flatfile Chat Database system. The API is designed to be async-first, type-safe, and easy to use.

## 🏗️ Core Classes

### FFStorageManager

The main interface for all database operations.

```python
class FFStorageManager:
    """Main storage manager for the flatfile chat database system."""
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: Optional[BackendProtocol] = None)
    async def initialize(self) -> bool
    async def close(self) -> None
```

#### Initialization
```python
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config

config = load_config()
storage = FFStorageManager(config)
await storage.initialize()
```

## 👤 User Management Methods

### create_user
```python
async def create_user(
    self, 
    user_id: str, 
    profile_data: Optional[Dict[str, Any]] = None
) -> bool
```
Creates a new user in the system.

**Parameters:**
- `user_id` (str): Unique identifier for the user
- `profile_data` (Dict[str, Any], optional): User profile information

**Returns:**
- `bool`: True if user was created successfully, False if user already exists

**Example:**
```python
# Simple user creation
success = await storage.create_user("alice")

# User with profile
success = await storage.create_user("bob", {
    "name": "Bob Smith",
    "email": "bob@example.com",
    "role": "admin",
    "preferences": {"theme": "dark"}
})
```

### user_exists
```python
async def user_exists(self, user_id: str) -> bool
```
Checks if a user exists in the system.

**Parameters:**
- `user_id` (str): User identifier to check

**Returns:**
- `bool`: True if user exists, False otherwise

**Example:**
```python
if await storage.user_exists("alice"):
    print("Alice exists in the system")
```

### get_user_profile
```python
async def get_user_profile(self, user_id: str) -> Optional[FFUserProfileDTO]
```
Retrieves user profile information.

**Parameters:**
- `user_id` (str): User identifier

**Returns:**
- `FFUserProfileDTO | None`: User profile object or None if not found

**Example:**
```python
profile = await storage.get_user_profile("alice")
if profile:
    print(f"User: {profile.name}, Email: {profile.email}")
```

### update_user_profile
```python
async def update_user_profile(
    self, 
    user_id: str, 
    profile_updates: Dict[str, Any]
) -> bool
```
Updates user profile information.

**Parameters:**
- `user_id` (str): User identifier
- `profile_updates` (Dict[str, Any]): Profile fields to update

**Returns:**
- `bool`: True if update was successful

**Example:**
```python
success = await storage.update_user_profile("alice", {
    "name": "Alice Johnson-Smith",
    "preferences": {"theme": "light", "notifications": False}
})
```

### delete_user
```python
async def delete_user(self, user_id: str, force: bool = False) -> bool
```
Deletes a user and optionally all associated data.

**Parameters:**
- `user_id` (str): User identifier
- `force` (bool): If True, deletes all user data; if False, fails if user has data

**Returns:**
- `bool`: True if deletion was successful

**Example:**
```python
# Safe delete (fails if user has sessions)
success = await storage.delete_user("alice")

# Force delete (removes all user data)
success = await storage.delete_user("alice", force=True)
```

## 💬 Session Management Methods

### create_session
```python
async def create_session(
    self, 
    user_id: str, 
    title: str = "New Chat Session",
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]
```
Creates a new chat session for a user.

**Parameters:**
- `user_id` (str): User identifier
- `title` (str): Session title
- `metadata` (Dict[str, Any], optional): Additional session metadata

**Returns:**
- `str | None`: Session ID if successful, None otherwise

**Example:**
```python
session_id = await storage.create_session(
    "alice", 
    "Customer Support Chat",
    {"priority": "high", "category": "billing"}
)
```

### get_session
```python
async def get_session(
    self, 
    user_id: str, 
    session_id: str
) -> Optional[FFSessionDTO]
```
Retrieves session information.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier

**Returns:**
- `FFSessionDTO | None`: Session object or None if not found

**Example:**
```python
session = await storage.get_session("alice", session_id)
if session:
    print(f"Session: {session.title}, Messages: {session.message_count}")
```

### list_sessions
```python
async def list_sessions(
    self, 
    user_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    include_metadata: bool = False
) -> List[FFSessionDTO]
```
Lists all sessions for a user.

**Parameters:**
- `user_id` (str): User identifier
- `limit` (int, optional): Maximum number of sessions to return
- `offset` (int): Number of sessions to skip
- `include_metadata` (bool): Whether to include session metadata

**Returns:**
- `List[FFSessionDTO]`: List of session objects

**Example:**
```python
# Get all sessions
sessions = await storage.list_sessions("alice")

# Get recent 10 sessions with metadata
recent_sessions = await storage.list_sessions(
    "alice", 
    limit=10, 
    include_metadata=True
)
```

### update_session
```python
async def update_session(
    self, 
    user_id: str, 
    session_id: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool
```
Updates session information.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `title` (str, optional): New session title
- `metadata` (Dict[str, Any], optional): Metadata updates

**Returns:**
- `bool`: True if update was successful

**Example:**
```python
success = await storage.update_session(
    "alice", 
    session_id,
    title="Resolved Support Issue",
    metadata={"status": "resolved", "resolution_time": "2023-10-15T14:30:00Z"}
)
```

### delete_session
```python
async def delete_session(self, user_id: str, session_id: str) -> bool
```
Deletes a session and all its messages.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier

**Returns:**
- `bool`: True if deletion was successful

**Example:**
```python
success = await storage.delete_session("alice", session_id)
```

## 📝 Message Methods

### add_message
```python
async def add_message(
    self, 
    user_id: str, 
    session_id: str, 
    message: FFMessageDTO
) -> bool
```
Adds a message to a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `message` (FFMessageDTO): Message object to add

**Returns:**
- `bool`: True if message was added successfully

**Example:**
```python
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

message = FFMessageDTO(
    role=MessageRole.USER,
    content="Hello, I need help with my account",
    metadata={"urgent": True}
)

success = await storage.add_message("alice", session_id, message)
```

### get_all_messages
```python
async def get_all_messages(
    self, 
    user_id: str, 
    session_id: str
) -> List[FFMessageDTO]
```
Retrieves all messages from a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier

**Returns:**
- `List[FFMessageDTO]`: List of all messages in chronological order

**Example:**
```python
messages = await storage.get_all_messages("alice", session_id)
for message in messages:
    print(f"[{message.role.value}]: {message.content}")
```

### get_messages
```python
async def get_messages(
    self, 
    user_id: str, 
    session_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    reverse: bool = False
) -> List[FFMessageDTO]
```
Retrieves messages with pagination options.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `limit` (int, optional): Maximum number of messages to return
- `offset` (int): Number of messages to skip
- `reverse` (bool): If True, returns messages in reverse chronological order

**Returns:**
- `List[FFMessageDTO]`: List of messages

**Example:**
```python
# Get last 10 messages
recent = await storage.get_messages("alice", session_id, limit=10, reverse=True)

# Get next 10 messages (pagination)
next_batch = await storage.get_messages("alice", session_id, limit=10, offset=10)
```

### delete_message
```python
async def delete_message(
    self, 
    user_id: str, 
    session_id: str, 
    message_id: str
) -> bool
```
Deletes a specific message from a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `message_id` (str): Message identifier

**Returns:**
- `bool`: True if deletion was successful

**Example:**
```python
success = await storage.delete_message("alice", session_id, "msg_123")
```

### stream_messages
```python
async def stream_messages(
    self, 
    user_id: str, 
    session_id: str,
    buffer_size: int = 100,
    timeout_seconds: Optional[float] = None
) -> AsyncIterator[List[FFMessageDTO]]
```
Streams messages in real-time as they are added to a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `buffer_size` (int): Number of messages to buffer before yielding
- `timeout_seconds` (float, optional): Stream timeout

**Returns:**
- `AsyncIterator[List[FFMessageDTO]]`: Async iterator of message batches

**Example:**
```python
async for message_batch in storage.stream_messages("alice", session_id):
    for message in message_batch:
        print(f"New message: {message.content}")
        # Process message in real-time
```

## 🔍 Search Methods

### search_messages
```python
async def search_messages(
    self, 
    user_id: str, 
    query: str,
    session_ids: Optional[List[str]] = None,
    limit: int = 100,
    min_relevance_score: float = 0.0
) -> List[FFSearchResultDTO]
```
Searches for messages matching a text query.

**Parameters:**
- `user_id` (str): User identifier
- `query` (str): Search query
- `session_ids` (List[str], optional): Limit search to specific sessions
- `limit` (int): Maximum number of results
- `min_relevance_score` (float): Minimum relevance threshold

**Returns:**
- `List[FFSearchResultDTO]`: List of search results

**Example:**
```python
results = await storage.search_messages(
    "alice", 
    "billing issue refund",
    session_ids=[session_id],
    limit=20,
    min_relevance_score=0.5
)

for result in results:
    print(f"Score: {result.relevance_score:.2f}")
    print(f"Content: {result.content}")
```

### search_similar_messages
```python
async def search_similar_messages(
    self, 
    user_id: str, 
    query: str,
    session_ids: Optional[List[str]] = None,
    similarity_threshold: float = 0.7,
    limit: int = 50
) -> List[FFSearchResultDTO]
```
Searches for semantically similar messages using vector embeddings.

**Parameters:**
- `user_id` (str): User identifier
- `query` (str): Query text for similarity matching
- `session_ids` (List[str], optional): Limit search to specific sessions
- `similarity_threshold` (float): Minimum similarity score
- `limit` (int): Maximum number of results

**Returns:**
- `List[FFSearchResultDTO]`: List of similar messages

**Example:**
```python
similar = await storage.search_similar_messages(
    "alice",
    "help with account problems",
    similarity_threshold=0.8,
    limit=10
)
```

## 📄 Document Methods

### store_document
```python
async def store_document(
    self, 
    user_id: str, 
    session_id: str, 
    document: FFDocumentDTO
) -> Optional[str]
```
Stores a document in a session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `document` (FFDocumentDTO): Document object to store

**Returns:**
- `str | None`: Document ID if successful, None otherwise

**Example:**
```python
from ff_class_configs.ff_chat_entities_config import FFDocumentDTO

document = FFDocumentDTO(
    filename="report.pdf",
    original_name="Q3 Sales Report.pdf",
    path="/uploads/report.pdf",
    mime_type="application/pdf",
    size=1048576,
    uploaded_by="alice",
    metadata={"category": "financial", "quarter": "Q3"}
)

doc_id = await storage.store_document("alice", session_id, document)
```

### get_document
```python
async def get_document(
    self, 
    user_id: str, 
    session_id: str, 
    document_id: str
) -> Optional[FFDocumentDTO]
```
Retrieves a document by ID.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `document_id` (str): Document identifier

**Returns:**
- `FFDocumentDTO | None`: Document object or None if not found

### list_documents
```python
async def list_documents(
    self, 
    user_id: str, 
    session_id: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[FFDocumentDTO]
```
Lists documents for a user or session.

**Parameters:**
- `user_id` (str): User identifier
- `session_id` (str, optional): Limit to specific session
- `document_type` (str, optional): Filter by MIME type
- `limit` (int, optional): Maximum number of documents

**Returns:**
- `List[FFDocumentDTO]`: List of documents

### search_documents
```python
async def search_documents(
    self, 
    user_id: str, 
    query: str,
    session_ids: Optional[List[str]] = None,
    document_types: Optional[List[str]] = None,
    limit: int = 50
) -> List[FFSearchResultDTO]
```
Searches through document content.

**Parameters:**
- `user_id` (str): User identifier
- `query` (str): Search query
- `session_ids` (List[str], optional): Limit to specific sessions
- `document_types` (List[str], optional): Filter by MIME types
- `limit` (int): Maximum results

**Returns:**
- `List[FFSearchResultDTO]`: Search results from documents

### delete_document
```python
async def delete_document(
    self, 
    user_id: str, 
    session_id: str, 
    document_id: str
) -> bool
```
Deletes a document from storage.

## 🎭 Persona & Panel Methods

### create_persona
```python
async def create_persona(self, persona: FFPersonaDTO) -> bool
```
Creates a new persona for multi-character conversations.

**Parameters:**
- `persona` (FFPersonaDTO): Persona object to create

**Returns:**
- `bool`: True if creation was successful

### get_persona
```python
async def get_persona(self, persona_id: str) -> Optional[FFPersonaDTO]
```
Retrieves a persona by ID.

### create_panel
```python
async def create_panel(self, panel: FFPanelDTO) -> Optional[str]
```
Creates a panel discussion with multiple personas.

### add_panel_message
```python
async def add_panel_message(
    self, 
    panel_id: str, 
    persona_id: str, 
    message: FFMessageDTO
) -> bool
```
Adds a message to a panel discussion from a specific persona.

## 🔍 Context Methods

### extract_situational_context
```python
async def extract_situational_context(
    self, 
    user_id: str, 
    session_id: str
) -> Optional[FFSituationalContextDTO]
```
Extracts situational context from a conversation.

### save_context
```python
async def save_context(
    self, 
    user_id: str, 
    session_id: str, 
    context: FFSituationalContextDTO
) -> Optional[str]
```
Saves extracted context for future reference.

### get_context
```python
async def get_context(
    self, 
    user_id: str, 
    context_id: str
) -> Optional[FFSituationalContextDTO]
```
Retrieves saved context by ID.

## 📊 Statistics & Analytics Methods

### get_user_stats
```python
async def get_user_stats(self, user_id: str) -> FFUserStatsDTO
```
Gets comprehensive statistics for a user.

**Returns:**
- `FFUserStatsDTO`: User statistics including session count, message count, activity patterns

### get_session_stats
```python
async def get_session_stats(
    self, 
    user_id: str, 
    session_id: str
) -> FFSessionStatsDTO
```
Gets statistics for a specific session.

### get_storage_stats
```python
async def get_storage_stats(self) -> FFStorageStatsDTO
```
Gets system-wide storage statistics.

### get_performance_metrics
```python
async def get_performance_metrics(self) -> FFPerformanceMetricsDTO
```
Gets system performance metrics.

## 🔧 Configuration Methods

### get_configuration
```python
def get_configuration(self) -> FFConfigurationManagerConfigDTO
```
Returns the current configuration.

### update_configuration
```python
async def update_configuration(
    self, 
    config_updates: Dict[str, Any]
) -> bool
```
Updates configuration at runtime.

## 🧹 Maintenance Methods

### cleanup_old_data
```python
async def cleanup_old_data(
    self, 
    days_old: int = 30,
    dry_run: bool = True
) -> Dict[str, int]
```
Cleans up old data based on age.

**Parameters:**
- `days_old` (int): Age threshold in days
- `dry_run` (bool): If True, returns what would be deleted without actually deleting

**Returns:**
- `Dict[str, int]`: Counts of items that were/would be deleted

### optimize_storage
```python
async def optimize_storage(self, user_id: Optional[str] = None) -> bool
```
Optimizes storage by compacting files and rebuilding indices.

### backup_data
```python
async def backup_data(
    self, 
    backup_path: str,
    user_id: Optional[str] = None
) -> bool
```
Creates a backup of user data or entire system.

### restore_data
```python
async def restore_data(
    self, 
    backup_path: str,
    user_id: Optional[str] = None
) -> bool
```
Restores data from a backup.

## 📋 Data Transfer Objects (DTOs)

### FFMessageDTO
```python
@dataclass
class FFMessageDTO:
    role: MessageRole
    content: str
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### FFSessionDTO
```python
@dataclass
class FFSessionDTO:
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
```

### FFUserProfileDTO
```python
@dataclass
class FFUserProfileDTO:
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    created_at: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### FFDocumentDTO
```python
@dataclass
class FFDocumentDTO:
    filename: str
    original_name: str
    path: str
    mime_type: str
    size: int
    uploaded_by: str
    uploaded_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### FFSearchResultDTO
```python
@dataclass
class FFSearchResultDTO:
    id: str
    type: str  # "message" or "document"
    content: str
    user_id: str
    session_id: str
    relevance_score: float
    similarity_score: Optional[float] = None
    highlights: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
```

## 🔢 Enums

### MessageRole
```python
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
```

## 🚨 Exception Classes

### FFStorageError
```python
class FFStorageError(Exception):
    """Base exception for storage operations."""
    pass
```

### FFUserNotFoundError
```python
class FFUserNotFoundError(FFStorageError):
    """Raised when a user is not found."""
    pass
```

### FFSessionNotFoundError
```python
class FFSessionNotFoundError(FFStorageError):
    """Raised when a session is not found."""
    pass
```

### FFValidationError
```python
class FFValidationError(FFStorageError):
    """Raised when input validation fails."""
    pass
```

## 🎯 Usage Examples

### Complete Workflow Example
```python
async def complete_workflow_example():
    """Example of a complete workflow using the API."""
    
    # Initialize
    config = load_config()
    storage = FFStorageManager(config)
    await storage.initialize()
    
    try:
        # Create user
        success = await storage.create_user("demo_user", {
            "name": "Demo User",
            "email": "demo@example.com"
        })
        
        if not success:
            print("User already exists")
        
        # Create session
        session_id = await storage.create_session(
            "demo_user", 
            "API Demo Session"
        )
        
        # Add messages
        messages = [
            FFMessageDTO(role=MessageRole.USER, content="Hello API!"),
            FFMessageDTO(role=MessageRole.ASSISTANT, content="Hello! How can I help?"),
            FFMessageDTO(role=MessageRole.USER, content="Show me how the API works"),
        ]
        
        for message in messages:
            await storage.add_message("demo_user", session_id, message)
        
        # Search messages
        results = await storage.search_messages("demo_user", "API")
        print(f"Found {len(results)} messages about API")
        
        # Get statistics
        stats = await storage.get_user_stats("demo_user")
        print(f"User has {stats.total_sessions} sessions with {stats.total_messages} messages")
        
    except FFStorageError as e:
        print(f"Storage error: {e}")
    finally:
        await storage.close()

await complete_workflow_example()
```

## 🔗 Type Hints Reference

All methods use proper type hints for better IDE support and type checking:

```python
from typing import Optional, List, Dict, Any, AsyncIterator
from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFUserProfileDTO, FFDocumentDTO,
    FFSearchResultDTO, MessageRole
)
```

## 🎉 API Summary

The API provides:

- ✅ **Comprehensive Coverage** - Full CRUD operations for all entities
- ✅ **Type Safety** - Complete type annotations and DTOs
- ✅ **Async Support** - All operations are async for better performance
- ✅ **Error Handling** - Specific exceptions for different error conditions
- ✅ **Search Capabilities** - Both text and semantic search
- ✅ **Streaming Support** - Real-time message streaming
- ✅ **Analytics** - Rich statistics and performance metrics
- ✅ **Flexibility** - Configurable and extensible design

This API reference provides everything needed to integrate the Flatfile Chat Database into your applications. All methods are designed to be intuitive, well-documented, and performant.