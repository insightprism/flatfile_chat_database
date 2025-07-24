#!/usr/bin/env python3
"""
Quick demonstration of the Flatfile Chat Database.

Shows how to use the main features with a simple example.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, SituationalContext, SearchQuery
)


async def main():
    """Demonstrate the flatfile chat database"""
    
    print("🗂️  Flatfile Chat Database Demo")
    print("=" * 50)
    
    # 1. Initialize the storage system
    print("\n1️⃣  Initializing storage system...")
    config = StorageConfig(storage_base_path="./demo_chat_data")
    storage = StorageManager(config)
    await storage.initialize()
    print("✅ Storage initialized")
    
    # 2. Create a user
    print("\n2️⃣  Creating user 'alice'...")
    await storage.create_user("alice", {
        "username": "Alice Johnson",
        "preferences": {"theme": "dark", "language": "en"}
    })
    print("✅ User created")
    
    # 3. Start a chat session
    print("\n3️⃣  Starting a chat session...")
    session_id = await storage.create_session("alice", "Python Programming Help")
    print(f"✅ Session created: {session_id}")
    
    # 4. Add some messages
    print("\n4️⃣  Adding conversation messages...")
    messages = [
        ("user", "Hi! I'm learning Python and need help with lists"),
        ("assistant", "Hello! I'd be happy to help you with Python lists. What would you like to know?"),
        ("user", "How do I add items to a list?"),
        ("assistant", "You can add items to a list in several ways:\n1. `append()` - adds one item to the end\n2. `extend()` - adds multiple items\n3. `insert()` - adds at specific position"),
        ("user", "Can you show me an example?"),
        ("assistant", "```python\n# Create a list\nfruits = ['apple', 'banana']\n\n# Append one item\nfruits.append('orange')\n# Result: ['apple', 'banana', 'orange']\n\n# Extend with multiple items\nfruits.extend(['grape', 'mango'])\n# Result: ['apple', 'banana', 'orange', 'grape', 'mango']\n\n# Insert at position\nfruits.insert(1, 'pear')\n# Result: ['apple', 'pear', 'banana', 'orange', 'grape', 'mango']\n```")
    ]
    
    for role, content in messages:
        msg = Message(role=role, content=content)
        await storage.add_message("alice", session_id, msg)
    
    print(f"✅ Added {len(messages)} messages")
    
    # 5. Save a document
    print("\n5️⃣  Saving a code example document...")
    code_example = b"""# Python List Methods Example
fruits = ['apple', 'banana', 'orange']

# Common list methods:
fruits.append('grape')        # Add to end
fruits.remove('banana')       # Remove specific item
fruits.pop()                  # Remove and return last item
fruits.sort()                 # Sort in place
fruits.reverse()              # Reverse order
"""
    
    doc_id = await storage.save_document(
        "alice", session_id, "list_methods.py", code_example,
        metadata={"type": "code_example", "language": "python"}
    )
    print(f"✅ Document saved: {doc_id}")
    
    # 6. Update context
    print("\n6️⃣  Updating conversation context...")
    context = SituationalContext(
        summary="Alice is learning Python list operations",
        key_points=["New to Python", "Learning about lists", "Prefers code examples"],
        entities={"topics": ["Python", "lists", "data structures"], "skill_level": ["beginner"]},
        confidence=0.9
    )
    await storage.update_context("alice", session_id, context)
    print("✅ Context updated")
    
    # 7. Search messages
    print("\n7️⃣  Searching for messages about 'list'...")
    results = await storage.search_messages("alice", "list")
    print(f"✅ Found {len(results)} messages containing 'list'")
    for i, msg in enumerate(results[:3], 1):
        print(f"   {i}. {msg.role}: {msg.content[:60]}...")
    
    # 8. Advanced search
    print("\n8️⃣  Advanced search for Python code...")
    query = SearchQuery(
        query="python",
        user_id="alice",
        include_documents=True,
        message_roles=["assistant"]  # Only assistant responses
    )
    adv_results = await storage.advanced_search(query)
    print(f"✅ Found {len(adv_results)} results")
    
    # 9. Create another session
    print("\n9️⃣  Creating another session...")
    session2_id = await storage.create_session("alice", "Web Development Questions")
    
    # Add different topic messages
    web_messages = [
        ("user", "What's the difference between HTML and CSS?"),
        ("assistant", "HTML (HyperText Markup Language) provides the structure and content of web pages, while CSS (Cascading Style Sheets) handles the visual styling and layout.")
    ]
    
    for role, content in web_messages:
        await storage.add_message("alice", session2_id, Message(role=role, content=content))
    
    print("✅ Created second session with web development topic")
    
    # 10. List all sessions
    print("\n🔟 Listing Alice's sessions...")
    sessions = await storage.list_sessions("alice")
    for session in sessions:
        messages = await storage.get_messages("alice", session.id, limit=1)
        preview = messages[0].content[:50] if messages else "No messages"
        print(f"   • {session.title} ({session.message_count} messages)")
        print(f"     Preview: {preview}...")
    
    # 11. Extract entities from text
    print("\n1️⃣1️⃣ Entity extraction demo...")
    sample_text = "Check out https://python.org for tutorials. Email me at alice@example.com about the #python basics."
    entities = await storage.extract_entities(sample_text)
    print("✅ Extracted entities:")
    for entity_type, values in entities.items():
        print(f"   • {entity_type}: {values}")
    
    # 12. Time-based search
    print("\n1️⃣2️⃣ Searching recent messages (last hour)...")
    recent_results = await storage.search_by_time_range(
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now(),
        user_id="alice"
    )
    print(f"✅ Found {len(recent_results)} messages in the last hour")
    
    # Summary
    print("\n" + "=" * 50)
    print("✨ Demo Complete!")
    print("\nThe flatfile chat database provides:")
    print("  ✓ User management")
    print("  ✓ Session organization") 
    print("  ✓ Message storage with pagination")
    print("  ✓ Document handling")
    print("  ✓ Context tracking")
    print("  ✓ Basic and advanced search")
    print("  ✓ Entity extraction")
    print("  ✓ Time-based queries")
    print("\nAll data is stored in: ./demo_chat_data/")
    
    # Show directory structure
    print("\n📁 Storage structure:")
    demo_path = Path("./demo_chat_data")
    if demo_path.exists():
        for user_dir in sorted(demo_path.iterdir()):
            if user_dir.is_dir() and not user_dir.name.startswith('.'):
                print(f"└── {user_dir.name}/")
                for session_dir in sorted(user_dir.iterdir())[:3]:  # Show first 3
                    if session_dir.is_dir():
                        print(f"    └── {session_dir.name}/")
                        for file in sorted(session_dir.iterdir())[:5]:  # Show first 5 files
                            if file.is_file():
                                print(f"        └── {file.name}")


if __name__ == "__main__":
    asyncio.run(main())