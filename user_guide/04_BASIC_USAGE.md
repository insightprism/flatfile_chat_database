# Basic Usage Guide

Learn the fundamental operations of the Flatfile Chat Database system through practical examples and detailed explanations.

## 🎯 Core Concepts Review

Before diving into usage, let's review the key concepts:

- **Users**: Individuals who participate in conversations
- **Sessions**: Individual chat conversations or threads
- **Messages**: Individual messages within sessions
- **Storage Manager**: Main interface for all database operations

## 🚀 Getting Started

### Basic Setup
```python
import asyncio
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

async def setup_storage():
    """Initialize the storage system."""
    config = load_config()
    storage = FFStorageManager(config)
    await storage.initialize()
    return storage

# Use in all examples
storage = await setup_storage()
```

## 👤 User Management

### Creating Users

#### Simple User Creation
```python
# Create user with just an ID
user_created = await storage.create_user("alice")
print(f"User created: {user_created}")  # True if successful
```

#### User with Profile Information
```python
# Create user with detailed profile
profile_data = {
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "role": "customer_support",
    "preferences": {
        "theme": "dark",
        "language": "en",
        "notifications": True
    },
    "metadata": {
        "department": "support",
        "hire_date": "2023-01-15",
        "skills": ["customer_service", "technical_support"]
    }
}

user_created = await storage.create_user("alice", profile_data)
```

#### Batch User Creation
```python
# Create multiple users
users_to_create = [
    ("alice", {"name": "Alice Johnson", "role": "support"}),
    ("bob", {"name": "Bob Smith", "role": "developer"}),
    ("charlie", {"name": "Charlie Brown", "role": "manager"})
]

for user_id, profile in users_to_create:
    success = await storage.create_user(user_id, profile)
    print(f"Created {user_id}: {success}")
```

### Checking User Existence
```python
# Check if user exists
if await storage.user_exists("alice"):
    print("Alice exists in the system")
else:
    print("Alice not found")
```

### Retrieving User Information
```python
# Get user profile
user_profile = await storage.get_user_profile("alice")
if user_profile:
    print(f"User: {user_profile.name}")
    print(f"Email: {user_profile.email}")
    print(f"Preferences: {user_profile.preferences}")
```

### Updating User Profiles
```python
# Update user profile
updated_profile = {
    "name": "Alice Johnson-Smith",  # Name change
    "preferences": {
        "theme": "light",           # Changed preference
        "language": "en",
        "notifications": False      # Disabled notifications
    }
}

success = await storage.update_user_profile("alice", updated_profile)
print(f"Profile updated: {success}")
```

## 💬 Session Management

### Creating Sessions

#### Basic Session Creation
```python
# Create a new chat session
session_id = await storage.create_session("alice", "Customer Support Chat")
print(f"Created session: {session_id}")
```

#### Session with Metadata
```python
# Create session with additional metadata
session_metadata = {
    "topic": "billing_inquiry",
    "priority": "high",
    "assigned_agent": "support_bot_v2",
    "customer_type": "premium"
}

session_id = await storage.create_session(
    user_id="alice",
    title="Billing Inquiry - Premium Customer",
    metadata=session_metadata
)
```

### Retrieving Session Information
```python
# Get session details
session = await storage.get_session("alice", session_id)
if session:
    print(f"Session ID: {session.session_id}")
    print(f"Title: {session.title}")
    print(f"Created: {session.created_at}")
    print(f"Messages: {session.message_count}")
    print(f"Metadata: {session.metadata}")
```

### Listing User Sessions
```python
# Get all sessions for a user
sessions = await storage.list_sessions("alice")
print(f"Alice has {len(sessions)} sessions:")

for session in sessions:
    print(f"  - {session.title} ({session.message_count} messages)")
```

#### Advanced Session Listing
```python
# List sessions with filtering
recent_sessions = await storage.list_sessions(
    user_id="alice",
    limit=10,           # Only get 10 most recent
    include_metadata=True
)

# Sort by message count
busy_sessions = sorted(sessions, key=lambda s: s.message_count, reverse=True)
print("Most active sessions:")
for session in busy_sessions[:5]:
    print(f"  - {session.title}: {session.message_count} messages")
```

### Updating Sessions
```python
# Update session title
await storage.update_session("alice", session_id, title="Resolved Billing Issue")

# Update session metadata
new_metadata = {
    "status": "resolved",
    "resolution_time": "2023-10-15T14:30:00Z",
    "satisfaction_score": 5
}
await storage.update_session("alice", session_id, metadata=new_metadata)
```

## 📝 Message Operations

### Creating Messages

#### Basic Message Creation
```python
# Create different types of messages
user_message = FFMessageDTO(
    role=MessageRole.USER,
    content="Hello, I need help with my billing."
)

assistant_message = FFMessageDTO(
    role=MessageRole.ASSISTANT,
    content="I'd be happy to help you with your billing inquiry. What specific issue are you facing?"
)

system_message = FFMessageDTO(
    role=MessageRole.SYSTEM,
    content="Customer support session initiated. Agent: AI Assistant"
)
```

#### Messages with Metadata
```python
# Message with rich metadata
detailed_message = FFMessageDTO(
    role=MessageRole.USER,
    content="I was charged twice for my subscription this month.",
    metadata={
        "timestamp": "2023-10-15T10:30:00Z",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0...",
        "session_duration": 120,
        "sentiment": "frustrated",
        "intent": "billing_issue",
        "entities": ["subscription", "duplicate_charge"]
    }
)
```

### Adding Messages to Sessions
```python
# Add messages to a session
messages = [user_message, assistant_message, system_message]

for message in messages:
    success = await storage.add_message("alice", session_id, message)
    if success:
        print(f"Added message: {message.content[:30]}...")
    else:
        print(f"Failed to add message: {message.content[:30]}...")
```

### Retrieving Messages

#### Get All Messages
```python
# Retrieve all messages in a session
all_messages = await storage.get_all_messages("alice", session_id)
print(f"Retrieved {len(all_messages)} messages")

for i, message in enumerate(all_messages, 1):
    print(f"{i}. [{message.role.value}]: {message.content}")
```

#### Get Recent Messages
```python
# Get only the last 5 messages
recent_messages = await storage.get_messages("alice", session_id, limit=5)
print("Recent conversation:")
for message in recent_messages:
    print(f"[{message.role.value}]: {message.content}")
```

#### Get Messages with Filtering
```python
# Get messages from a specific role
user_messages = [msg for msg in all_messages if msg.role == MessageRole.USER]
print(f"User sent {len(user_messages)} messages")

# Get messages after a certain time
from datetime import datetime, timedelta
cutoff_time = datetime.now() - timedelta(hours=1)

recent_messages = [
    msg for msg in all_messages 
    if datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00')) > cutoff_time
]
```

### Message Streaming
```python
# Stream messages in real-time (useful for live conversations)
async for message_batch in storage.stream_messages("alice", session_id):
    for message in message_batch:
        print(f"New message: [{message.role.value}] {message.content}")
        
        # Process message in real-time
        if message.role == MessageRole.USER:
            # Trigger response generation
            await process_user_message(message)
```

## 🔍 Search and Discovery

### Basic Text Search
```python
# Search for messages containing specific terms
search_results = await storage.search_messages("alice", "billing")
print(f"Found {len(search_results)} messages about billing")

for result in search_results:
    print(f"- Session: {result.session_id}")
    print(f"  Content: {result.content[:100]}...")
    print(f"  Relevance: {result.relevance_score:.2f}")
```

### Advanced Search
```python
# Search with more specific parameters
advanced_results = await storage.search_messages(
    user_id="alice",
    query="subscription refund",
    session_ids=[session_id],  # Search only in specific sessions
    limit=20,                  # Limit results
    min_relevance_score=0.5    # Only high-relevance results
)
```

### Cross-User Search (if permissions allow)
```python
# Search across multiple users
multi_user_results = await storage.search_messages(
    user_id=None,  # Search all users
    query="technical issue",
    limit=50
)

# Group results by user
from collections import defaultdict
results_by_user = defaultdict(list)
for result in multi_user_results:
    results_by_user[result.user_id].append(result)

for user_id, user_results in results_by_user.items():
    print(f"{user_id}: {len(user_results)} results")
```

## 📊 Statistics and Analytics

### User Statistics
```python
# Get user statistics
user_stats = await storage.get_user_stats("alice")
print(f"User Statistics for Alice:")
print(f"  Total sessions: {user_stats.total_sessions}")
print(f"  Total messages: {user_stats.total_messages}")
print(f"  Average messages per session: {user_stats.avg_messages_per_session:.1f}")
print(f"  First activity: {user_stats.first_activity}")
print(f"  Last activity: {user_stats.last_activity}")
```

### Session Statistics
```python
# Get statistics for a specific session
session_stats = await storage.get_session_stats("alice", session_id)
print(f"Session Statistics:")
print(f"  Message count: {session_stats.message_count}")
print(f"  Duration: {session_stats.duration_minutes} minutes")
print(f"  User messages: {session_stats.user_message_count}")
print(f"  Assistant messages: {session_stats.assistant_message_count}")
print(f"  Average message length: {session_stats.avg_message_length:.1f} chars")
```

### System-Wide Statistics
```python
# Get overall system statistics
system_stats = await storage.get_storage_stats()
print(f"System Statistics:")
print(f"  Total users: {system_stats.total_users}")
print(f"  Total sessions: {system_stats.total_sessions}")
print(f"  Total messages: {system_stats.total_messages}")
print(f"  Storage size: {system_stats.storage_size_mb:.1f} MB")
print(f"  Average messages per user: {system_stats.avg_messages_per_user:.1f}")
```

## 🗂️ Working with Conversations

### Complete Conversation Example
```python
async def create_sample_conversation():
    """Create a complete conversation example."""
    
    # Create participants
    await storage.create_user("customer", {
        "name": "John Customer",
        "type": "customer"
    })
    
    await storage.create_user("agent", {
        "name": "Sarah Agent",
        "type": "support_agent"
    })
    
    # Create session
    session_id = await storage.create_session(
        "customer", 
        "Product Return Request"
    )
    
    # Create conversation flow
    conversation = [
        ("customer", "USER", "Hi, I need to return a product I ordered last week."),
        ("agent", "ASSISTANT", "Hello! I'd be happy to help you with your return. Can you provide me with your order number?"),
        ("customer", "USER", "Sure, it's ORDER-12345. The product arrived damaged."),
        ("agent", "ASSISTANT", "I'm sorry to hear the product arrived damaged. Let me look up your order... I can see ORDER-12345 here. I'll process a return label for you right away."),
        ("customer", "USER", "That's great! How long will the refund take once you receive it?"),
        ("agent", "ASSISTANT", "Once we receive the returned item, we'll process your refund within 3-5 business days. You'll receive an email confirmation when it's processed."),
        ("customer", "USER", "Perfect, thank you for your help!"),
        ("agent", "ASSISTANT", "You're welcome! Is there anything else I can help you with today?"),
        ("customer", "USER", "No, that's everything. Have a great day!"),
        ("agent", "ASSISTANT", "Thank you, you too! Your return label will be emailed to you within the next hour.")
    ]
    
    # Add messages to session
    for user_id, role, content in conversation:
        message = FFMessageDTO(
            role=MessageRole[role],
            content=content
        )
        await storage.add_message(user_id, session_id, message)
    
    return session_id

# Create the conversation
conversation_session = await create_sample_conversation()
```

### Conversation Analysis
```python
# Analyze the conversation
messages = await storage.get_all_messages("customer", conversation_session)

# Calculate conversation metrics
total_messages = len(messages)
customer_messages = len([m for m in messages if m.role == MessageRole.USER])
agent_messages = len([m for m in messages if m.role == MessageRole.ASSISTANT])

print(f"Conversation Analysis:")
print(f"  Total messages: {total_messages}")
print(f"  Customer messages: {customer_messages}")
print(f"  Agent messages: {agent_messages}")
print(f"  Response ratio: {agent_messages/customer_messages:.2f}")

# Calculate response times (simplified)
response_times = []
for i in range(len(messages) - 1):
    if (messages[i].role == MessageRole.USER and 
        messages[i+1].role == MessageRole.ASSISTANT):
        # In a real implementation, you'd calculate actual time differences
        # Here we'll simulate response time analysis
        response_times.append(f"Response to message {i+1}")

print(f"  Agent responses: {len(response_times)}")
```

## 🔄 Bulk Operations

### Bulk Message Import
```python
async def import_conversation_history(user_id: str, conversation_data: list):
    """Import a large conversation history efficiently."""
    
    session_id = await storage.create_session(user_id, "Imported Conversation")
    
    # Process messages in batches for efficiency
    batch_size = 50
    for i in range(0, len(conversation_data), batch_size):
        batch = conversation_data[i:i + batch_size]
        
        # Add batch of messages
        for msg_data in batch:
            message = FFMessageDTO(
                role=MessageRole[msg_data['role']],
                content=msg_data['content'],
                metadata=msg_data.get('metadata', {})
            )
            await storage.add_message(user_id, session_id, message)
        
        print(f"Imported batch {i//batch_size + 1}: {len(batch)} messages")
    
    return session_id

# Example usage
large_conversation = [
    {"role": "USER", "content": f"Message {i}", "metadata": {"batch": i//10}}
    for i in range(500)  # 500 messages
]

imported_session = await import_conversation_history("alice", large_conversation)
```

### Bulk User Creation
```python
async def create_users_from_csv(csv_data: list):
    """Create multiple users from CSV-like data."""
    
    created_count = 0
    failed_count = 0
    
    for row in csv_data:
        try:
            user_id = row['user_id']
            profile = {
                'name': row['name'],
                'email': row['email'],
                'department': row.get('department', ''),
                'role': row.get('role', 'user')
            }
            
            success = await storage.create_user(user_id, profile)
            if success:
                created_count += 1
            else:
                failed_count += 1
                print(f"Failed to create user: {user_id}")
                
        except Exception as e:
            failed_count += 1
            print(f"Error creating user {row.get('user_id', 'unknown')}: {e}")
    
    print(f"Bulk user creation completed: {created_count} created, {failed_count} failed")
    return created_count, failed_count

# Example usage
user_data = [
    {"user_id": "emp001", "name": "John Doe", "email": "john@example.com", "department": "IT"},
    {"user_id": "emp002", "name": "Jane Smith", "email": "jane@example.com", "department": "HR"},
    {"user_id": "emp003", "name": "Bob Johnson", "email": "bob@example.com", "department": "Sales"},
]

await create_users_from_csv(user_data)
```

## 🧹 Data Management

### Session Cleanup
```python
async def cleanup_old_sessions(user_id: str, days_old: int = 30):
    """Clean up sessions older than specified days."""
    
    from datetime import datetime, timedelta
    
    sessions = await storage.list_sessions(user_id)
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    old_sessions = []
    for session in sessions:
        session_date = datetime.fromisoformat(session.created_at.replace('Z', '+00:00'))
        if session_date < cutoff_date:
            old_sessions.append(session)
    
    print(f"Found {len(old_sessions)} sessions older than {days_old} days")
    
    # In a full implementation, you'd have a delete_session method
    # For now, we'll just report what would be deleted
    for session in old_sessions:
        print(f"Would delete: {session.title} (created {session.created_at})")
    
    return len(old_sessions)

# Example usage
old_session_count = await cleanup_old_sessions("alice", 60)  # 60 days old
```

### Data Export
```python
async def export_user_data(user_id: str, output_file: str):
    """Export all user data to JSON file."""
    
    import json
    from pathlib import Path
    
    # Collect all user data
    export_data = {
        "user_id": user_id,
        "profile": None,
        "sessions": [],
        "export_timestamp": datetime.now().isoformat()
    }
    
    # Get user profile
    profile = await storage.get_user_profile(user_id)
    if profile:
        export_data["profile"] = {
            "name": profile.name,
            "email": profile.email,
            "preferences": profile.preferences,
            "metadata": profile.metadata
        }
    
    # Get all sessions and messages
    sessions = await storage.list_sessions(user_id)
    for session in sessions:
        messages = await storage.get_all_messages(user_id, session.session_id)
        
        session_data = {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        }
        export_data["sessions"].append(session_data)
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported data for {user_id} to {output_path}")
    print(f"  Sessions: {len(export_data['sessions'])}")
    print(f"  Total messages: {sum(s['message_count'] for s in export_data['sessions'])}")

# Example usage
await export_user_data("alice", "alice_export.json")
```

## 🔒 Error Handling

### Robust Error Handling
```python
async def safe_storage_operations():
    """Demonstrate proper error handling for storage operations."""
    
    try:
        # User creation with error handling
        user_created = await storage.create_user("new_user")
        if not user_created:
            print("User creation failed - user may already exist")
        
        # Session creation with validation
        if await storage.user_exists("new_user"):
            session_id = await storage.create_session("new_user", "Test Session")
            if session_id:
                print(f"Session created successfully: {session_id}")
            else:
                print("Session creation failed")
        else:
            print("Cannot create session - user does not exist")
        
        # Message addition with error handling
        message = FFMessageDTO(
            role=MessageRole.USER,
            content="Test message with proper error handling"
        )
        
        success = await storage.add_message("new_user", session_id, message)
        if success:
            print("Message added successfully")
        else:
            print("Message addition failed")
            
    except FileNotFoundError as e:
        print(f"Storage file not found: {e}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except ValueError as e:
        print(f"Invalid value provided: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

await safe_storage_operations()
```

## 🎯 Best Practices

### 1. Efficient Message Handling
```python
# Good: Batch operations when possible
messages_to_add = [create_message(i) for i in range(10)]
for message in messages_to_add:
    await storage.add_message("user", session_id, message)

# Better: Use transaction-like patterns (if available)
# await storage.add_messages_batch("user", session_id, messages_to_add)
```

### 2. Proper Resource Management
```python
# Always initialize storage properly
async def with_storage():
    storage = None
    try:
        config = load_config()
        storage = FFStorageManager(config)
        await storage.initialize()
        
        # Do your operations here
        yield storage
        
    finally:
        # Clean up if needed
        if storage and hasattr(storage, 'close'):
            await storage.close()
```

### 3. Validate Input Data
```python
def validate_message_content(content: str) -> bool:
    """Validate message content before adding."""
    if not content or not content.strip():
        return False
    if len(content) > 10000:  # Example limit
        return False
    return True

# Use validation
content = "User message content"
if validate_message_content(content):
    message = FFMessageDTO(role=MessageRole.USER, content=content)
    await storage.add_message("user", session_id, message)
else:
    print("Invalid message content")
```

## 🎉 Summary

You now know how to:

- ✅ **Create and manage users** with profiles and metadata
- ✅ **Create and organize sessions** for different conversations
- ✅ **Add and retrieve messages** with different roles and content
- ✅ **Search through messages** using text queries
- ✅ **Get statistics and analytics** about usage patterns
- ✅ **Handle errors gracefully** with proper exception handling
- ✅ **Perform bulk operations** efficiently
- ✅ **Export and manage data** for backup and analysis

## 🚀 Next Steps

Now that you understand basic usage, you can explore:

- **[Advanced Features](05_ADVANCED_FEATURES.md)** - Document handling, vector search, streaming
- **[API Reference](06_API_REFERENCE.md)** - Complete method documentation
- **[Examples & Tutorials](07_EXAMPLES.md)** - Real-world usage patterns
- **[Performance & Optimization](08_PERFORMANCE.md)** - Scaling and optimization techniques

The basic operations covered here form the foundation for building sophisticated chat applications with persistent storage, search capabilities, and comprehensive data management.