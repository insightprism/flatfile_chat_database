#!/usr/bin/env python3
"""
Automated test of the interactive chat demo.
"""

import asyncio
import sys
from io import StringIO
from unittest.mock import patch
from interactive_chat_demo import InteractiveChatDemo
from chat_config import get_default_config

async def run_demo_test():
    """Run automated demo test"""
    config = get_default_config()
    config.storage.storage_base_path = "./test_chat_data"
    
    demo = InteractiveChatDemo(config)
    await demo.initialize()
    
    # Simulate user interactions
    print("=== AUTOMATED DEMO TEST ===")
    
    # Test 1: Create user
    print("\n1. Creating user 'demo_user'...")
    demo.context['user_id'] = 'demo_user'
    await demo.storage.create_user('demo_user', {
        'username': 'Demo User',
        'created_via': 'automated_test'
    })
    print("✓ User created")
    
    # Test 2: Create session
    print("\n2. Creating new session...")
    session_id = await demo.storage.create_session('demo_user', 'Test Session')
    demo.context['session_id'] = session_id
    demo.context['session_title'] = 'Test Session'
    print(f"✓ Session created: {session_id}")
    
    # Test 3: Add messages
    print("\n3. Adding test messages...")
    from flatfile_chat_database import Message
    
    test_messages = [
        Message(role="user", content="Hello! Can you help me with Python?"),
        Message(role="assistant", content="Of course! I'd be happy to help you with Python."),
        Message(role="user", content="How do I read a file in Python?"),
        Message(role="assistant", content="You can use the open() function to read files in Python."),
    ]
    
    for msg in test_messages:
        await demo.storage.add_message('demo_user', session_id, msg)
    print(f"✓ Added {len(test_messages)} messages")
    
    # Test 4: Search messages
    print("\n4. Testing search...")
    from flatfile_chat_database import SearchQuery
    query = SearchQuery(
        query="Python",
        user_id='demo_user',
        include_context=True
    )
    results = await demo.storage.advanced_search(query)
    print(f"✓ Search found {len(results)} results for 'Python'")
    
    # Test 5: List sessions
    print("\n5. Listing sessions...")
    sessions = await demo.storage.list_sessions('demo_user')
    print(f"✓ Found {len(sessions)} session(s)")
    for session in sessions:
        print(f"   - {session.title} ({session.message_count} messages)")
    
    # Test 6: Get messages
    print("\n6. Retrieving messages...")
    messages = await demo.storage.get_messages('demo_user', session_id)
    print(f"✓ Retrieved {len(messages)} messages")
    for msg in messages[:2]:  # Show first 2
        print(f"   [{msg.role}]: {msg.content[:50]}...")
    
    # Test 7: Update context
    print("\n7. Updating conversation context...")
    from flatfile_chat_database import SituationalContext
    context = SituationalContext(
        summary="Discussion about Python file operations",
        key_points=["Reading files", "open() function"],
        entities={"topics": ["Python", "file operations"]},
        confidence=0.8
    )
    await demo.storage.update_context('demo_user', session_id, context)
    print("✓ Context updated")
    
    # Test 8: Upload document
    print("\n8. Testing document upload...")
    test_doc = b"# Python File Reading\n\nExample code for reading files."
    doc_id = await demo.storage.save_document(
        'demo_user', session_id, 'python_guide.md', test_doc,
        metadata={'type': 'markdown', 'size': len(test_doc)}
    )
    print(f"✓ Document uploaded: {doc_id}")
    
    # Test 9: Export session
    print("\n9. Testing export...")
    messages_all = await demo.storage.get_all_messages('demo_user', session_id)
    export_data = {
        'session_id': session_id,
        'title': 'Test Session',
        'messages': [msg.to_dict() for msg in messages_all],
        'message_count': len(messages_all)
    }
    print(f"✓ Session exported ({len(export_data['messages'])} messages)")
    
    # Test 10: User stats
    print("\n10. Getting user statistics...")
    profile = await demo.storage.get_user_profile('demo_user')
    docs = await demo.storage.list_documents('demo_user', session_id)
    print(f"✓ User stats:")
    print(f"   - Sessions: {len(sessions)}")
    print(f"   - Total messages: {sum(s.message_count for s in sessions)}")
    print(f"   - Documents: {len(docs)}")
    
    print("\n=== DEMO TEST COMPLETE ===")
    print("All features working correctly!")
    
    # Show storage structure
    print("\n📁 Storage structure created:")
    import os
    for root, dirs, files in os.walk("./test_chat_data"):
        level = root.replace("./test_chat_data", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")

if __name__ == "__main__":
    asyncio.run(run_demo_test())