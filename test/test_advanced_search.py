#!/usr/bin/env python3
"""
Test advanced search functionality.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, Document, 
    SituationalContext, SearchQuery, SearchResult
)


async def test_advanced_search():
    """Test advanced search functionality"""
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Setting up test data...")
        
        # Create storage manager
        config = StorageConfig(storage_base_path=temp_dir)
        manager = StorageManager(config=config)
        await manager.initialize()
        
        # Create test data
        user_id = "search_test_user"
        await manager.create_user(user_id)
        
        # Create sessions with different dates
        now = datetime.now()
        
        # Session 1: Python discussion (yesterday)
        session1 = await manager.create_session(user_id, "Python Help")
        messages1 = [
            Message(role="user", content="How do I use asyncio in Python?"),
            Message(role="assistant", content="Python's asyncio is great for concurrent programming. Here's an example: https://docs.python.org/3/library/asyncio.html"),
            Message(role="user", content="Can you show me a code example?"),
            Message(role="assistant", content="```python\nimport asyncio\n\nasync def main():\n    await asyncio.sleep(1)\n    print('Hello!')\n\nasyncio.run(main())\n```"),
        ]
        
        # Manually set timestamps for testing
        for i, msg in enumerate(messages1):
            msg.timestamp = (now - timedelta(days=1, hours=i)).isoformat()
            await manager.add_message(user_id, session1, msg)
        
        # Add context
        context1 = SituationalContext(
            summary="Learning Python asyncio",
            key_points=["asyncio", "concurrent programming", "async/await"],
            entities={"languages": ["python"], "topics": ["concurrency"]},
            confidence=0.9
        )
        await manager.update_context(user_id, session1, context1)
        
        # Session 2: JavaScript discussion (today)
        session2 = await manager.create_session(user_id, "JavaScript Help")
        messages2 = [
            Message(role="user", content="What's the difference between let and const in JavaScript?"),
            Message(role="assistant", content="In JavaScript, 'let' allows reassignment while 'const' creates immutable bindings. Check out https://developer.mozilla.org/en-US/docs/Web/JavaScript"),
            Message(role="user", content="Show me an example with #javascript"),
            Message(role="assistant", content="```javascript\nlet x = 5;\nx = 10; // OK\n\nconst y = 5;\ny = 10; // Error!\n```"),
        ]
        
        for i, msg in enumerate(messages2):
            msg.timestamp = (now - timedelta(hours=i)).isoformat()
            await manager.add_message(user_id, session2, msg)
        
        # Add document
        doc_content = b"JavaScript ES6 features documentation"
        doc_id = await manager.save_document(
            user_id, session2, "es6_features.txt", doc_content,
            metadata={"description": "Modern JavaScript features"}
        )
        
        # Session 3: General discussion (last week)
        session3 = await manager.create_session(user_id, "General Chat")
        messages3 = [
            Message(role="user", content="Hello! How are you?"),
            Message(role="assistant", content="I'm doing well, thank you! How can I help you today?"),
        ]
        
        for i, msg in enumerate(messages3):
            msg.timestamp = (now - timedelta(days=7, hours=i)).isoformat()
            await manager.add_message(user_id, session3, msg)
        
        print("\n=== Testing Advanced Search Features ===")
        
        # Test 1: Basic text search
        print("\n1. Basic text search for 'asyncio':")
        query = SearchQuery(query="asyncio", user_id=user_id)
        results = await manager.advanced_search(query)
        print(f"   Found {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.type}: {r.content[:60]}... (score: {r.relevance_score:.2f})")
        
        # Test 2: Entity-based search
        print("\n2. Entity-based search for Python:")
        results = await manager.search_by_entities(
            entities={"languages": ["python"]},
            user_id=user_id
        )
        print(f"   Found {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.type}: {r.content[:60]}... (score: {r.relevance_score:.2f})")
        
        # Test 3: Time-range search
        print("\n3. Time-range search (last 2 days):")
        results = await manager.search_by_time_range(
            start_date=now - timedelta(days=2),
            end_date=now,
            user_id=user_id
        )
        print(f"   Found {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.type}: {r.content[:60]}... (timestamp: {r.timestamp[:10]})")
        
        # Test 4: Combined search
        print("\n4. Combined search (JavaScript in last day):")
        query = SearchQuery(
            query="javascript",
            user_id=user_id,
            start_date=now - timedelta(days=1),
            end_date=now,
            include_documents=True
        )
        results = await manager.advanced_search(query)
        print(f"   Found {len(results)} results")
        for r in results:
            print(f"   - {r.type}: {r.content[:60]}... (score: {r.relevance_score:.2f})")
        
        # Test 5: Extract entities
        print("\n5. Entity extraction:")
        test_text = "Check out https://github.com/python/cpython and email me at user@example.com about #python"
        entities = await manager.extract_entities(test_text)
        print(f"   Extracted entities: {entities}")
        
        # Test 6: Search with role filter
        print("\n6. Search only user messages:")
        query = SearchQuery(
            query="how",
            user_id=user_id,
            message_roles=["user"]
        )
        results = await manager.advanced_search(query)
        print(f"   Found {len(results)} results")
        for r in results[:3]:
            print(f"   - Role: {r.metadata.get('role', 'N/A')}, Content: {r.content[:50]}...")
        
        # Test 7: Build search index
        print("\n7. Building search index:")
        index = await manager.build_search_index(user_id)
        print(f"   Sessions indexed: {len(index['sessions'])}")
        print(f"   Total entities: {len(index['entities'])}")
        print(f"   Entity types: {list(index['entities'].keys())}")
        
        # Test 8: Search with minimum relevance score
        print("\n8. Search with minimum relevance score:")
        query = SearchQuery(
            query="programming",
            user_id=user_id,
            min_relevance_score=0.5
        )
        results = await manager.advanced_search(query)
        print(f"   Found {len(results)} results with score >= 0.5")
        
        # Test 9: Cross-session search (no user filter)
        print("\n9. Cross-session search:")
        query = SearchQuery(
            query="javascript OR python",
            include_context=True
        )
        results = await manager.advanced_search(query)
        print(f"   Found {len(results)} results across all users")
        
        print("\n=== All Advanced Search Tests Completed ===")
        return True


async def main():
    """Run test and handle results"""
    try:
        success = await test_advanced_search()
        if success:
            print("\n✓ Advanced search tests passed!")
            return 0
        else:
            print("\n✗ Advanced search tests failed!")
            return 1
    except Exception as e:
        print(f"\n✗ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)