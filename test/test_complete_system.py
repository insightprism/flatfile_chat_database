#!/usr/bin/env python3
"""
Comprehensive test suite for the Flatfile Chat Database.

This demonstrates how to use and test all features of the system.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, Session, Document,
    SituationalContext, SearchQuery, SearchResult
)


class TestColors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}=== {text} ==={TestColors.END}")


def print_success(text):
    """Print success message"""
    print(f"{TestColors.GREEN}✓ {text}{TestColors.END}")


def print_error(text):
    """Print error message"""
    print(f"{TestColors.RED}✗ {text}{TestColors.END}")


def print_info(text):
    """Print info message"""
    print(f"{TestColors.BLUE}→ {text}{TestColors.END}")


async def test_basic_operations(manager: StorageManager):
    """Test basic CRUD operations"""
    print_header("Testing Basic Operations")
    
    # 1. User Management
    print_info("Testing user management...")
    
    # Create user
    user_id = "test_user_001"
    profile = {
        "username": "John Doe",
        "preferences": {
            "theme": "dark",
            "language": "en",
            "ai_model": "gpt-4"
        },
        "metadata": {
            "created_from": "test_suite",
            "api_version": "1.0"
        }
    }
    
    result = await manager.create_user(user_id, profile)
    assert result, "Failed to create user"
    print_success("User created successfully")
    
    # Retrieve user
    retrieved_profile = await manager.get_user_profile(user_id)
    assert retrieved_profile is not None, "Failed to retrieve user"
    assert retrieved_profile["username"] == "John Doe", "Username mismatch"
    print_success("User retrieved successfully")
    
    # Update user
    updates = {"preferences": {"theme": "light"}}
    result = await manager.update_user_profile(user_id, updates)
    assert result, "Failed to update user"
    
    updated_profile = await manager.get_user_profile(user_id)
    assert updated_profile["preferences"]["theme"] == "light", "Update not applied"
    print_success("User updated successfully")
    
    # 2. Session Management
    print_info("\nTesting session management...")
    
    # Create session
    session_id = await manager.create_session(user_id, "Test Chat Session")
    assert session_id, "Failed to create session"
    assert session_id.startswith("chat_session_"), "Invalid session ID format"
    print_success(f"Session created: {session_id}")
    
    # Get session
    session = await manager.get_session(user_id, session_id)
    assert session is not None, "Failed to retrieve session"
    assert session.title == "Test Chat Session", "Session title mismatch"
    print_success("Session retrieved successfully")
    
    # List sessions
    sessions = await manager.list_sessions(user_id)
    assert len(sessions) >= 1, "No sessions found"
    print_success(f"Listed {len(sessions)} sessions")
    
    # 3. Message Management
    print_info("\nTesting message management...")
    
    # Add messages
    messages = [
        Message(role="user", content="Hello! How can you help me today?"),
        Message(role="assistant", content="I'm here to help! What would you like to know?"),
        Message(role="user", content="Can you explain Python decorators?"),
        Message(role="assistant", content="Python decorators are a way to modify or enhance functions...")
    ]
    
    for msg in messages:
        result = await manager.add_message(user_id, session_id, msg)
        assert result, f"Failed to add message: {msg.content[:30]}..."
    print_success(f"Added {len(messages)} messages")
    
    # Retrieve messages
    retrieved_messages = await manager.get_messages(user_id, session_id)
    assert len(retrieved_messages) == len(messages), "Message count mismatch"
    print_success("Messages retrieved successfully")
    
    # Test pagination
    page1 = await manager.get_messages(user_id, session_id, limit=2, offset=0)
    page2 = await manager.get_messages(user_id, session_id, limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2, "Pagination failed"
    print_success("Message pagination working")
    
    return True


async def test_document_handling(manager: StorageManager):
    """Test document storage and retrieval"""
    print_header("Testing Document Handling")
    
    # Setup
    user_id = "doc_test_user"
    await manager.create_user(user_id)
    session_id = await manager.create_session(user_id, "Document Test Session")
    
    # 1. Upload text document
    print_info("Uploading text document...")
    text_content = b"This is a test document containing important information about Python."
    doc_id = await manager.save_document(
        user_id, session_id, "python_guide.txt", text_content,
        metadata={"type": "tutorial", "topic": "Python"}
    )
    assert doc_id, "Failed to save document"
    print_success(f"Document saved with ID: {doc_id}")
    
    # 2. Retrieve document
    print_info("Retrieving document...")
    retrieved_content = await manager.get_document(user_id, session_id, doc_id)
    assert retrieved_content == text_content, "Document content mismatch"
    print_success("Document retrieved successfully")
    
    # 3. List documents
    print_info("Listing documents...")
    docs = await manager.list_documents(user_id, session_id)
    assert len(docs) == 1, "Document count mismatch"
    assert docs[0].original_name == "python_guide.txt", "Document name mismatch"
    print_success(f"Found {len(docs)} documents")
    
    # 4. Update document analysis
    print_info("Adding document analysis...")
    analysis = {
        "summary": "A guide about Python programming",
        "keywords": ["Python", "programming", "tutorial"],
        "word_count": 10,
        "language": "en"
    }
    result = await manager.update_document_analysis(user_id, session_id, doc_id, analysis)
    assert result, "Failed to update analysis"
    print_success("Document analysis added")
    
    # 5. Test file type restrictions
    print_info("Testing file type restrictions...")
    try:
        await manager.save_document(user_id, session_id, "test.exe", b"executable")
        print_error("Should have rejected .exe file")
        return False
    except:
        print_success("Correctly rejected invalid file type")
    
    return True


async def test_context_management(manager: StorageManager):
    """Test situational context tracking"""
    print_header("Testing Context Management")
    
    # Setup
    user_id = "context_test_user"
    await manager.create_user(user_id)
    session_id = await manager.create_session(user_id, "Context Test Session")
    
    # 1. Create and update context
    print_info("Creating situational context...")
    context = SituationalContext(
        summary="User is learning about machine learning fundamentals",
        key_points=[
            "Interested in neural networks",
            "Beginner level",
            "Prefers practical examples"
        ],
        entities={
            "topics": ["machine learning", "neural networks"],
            "skill_level": ["beginner"],
            "preferences": ["practical", "examples"]
        },
        confidence=0.85
    )
    
    result = await manager.update_context(user_id, session_id, context)
    assert result, "Failed to update context"
    print_success("Context updated successfully")
    
    # 2. Retrieve current context
    print_info("Retrieving current context...")
    current_context = await manager.get_context(user_id, session_id)
    assert current_context is not None, "Failed to retrieve context"
    assert current_context.summary == context.summary, "Context summary mismatch"
    assert current_context.confidence == 0.85, "Confidence mismatch"
    print_success("Current context retrieved")
    
    # 3. Save context snapshots
    print_info("Creating context history...")
    for i in range(3):
        snapshot = SituationalContext(
            summary=f"Learning progress: Stage {i+1}",
            key_points=[f"Completed module {i+1}"],
            entities={"progress": [f"stage_{i+1}"]},
            confidence=0.7 + i * 0.1
        )
        result = await manager.save_context_snapshot(user_id, session_id, snapshot)
        assert result, f"Failed to save snapshot {i+1}"
        await asyncio.sleep(0.01)  # Ensure different timestamps
    print_success("Created 3 context snapshots")
    
    # 4. Retrieve context history
    print_info("Retrieving context history...")
    history = await manager.get_context_history(user_id, session_id)
    assert len(history) >= 3, "Context history incomplete"
    assert history[0].summary == "Learning progress: Stage 3", "History order incorrect"
    print_success(f"Retrieved {len(history)} context snapshots")
    
    return True


async def test_search_capabilities(manager: StorageManager):
    """Test basic and advanced search features"""
    print_header("Testing Search Capabilities")
    
    # Setup test data
    print_info("Setting up search test data...")
    user_id = "search_test_user"
    await manager.create_user(user_id)
    
    # Create multiple sessions with different topics
    topics = {
        "python_basics": ["variables", "functions", "classes"],
        "web_development": ["HTML", "CSS", "JavaScript"],
        "data_science": ["pandas", "numpy", "matplotlib"]
    }
    
    for topic, keywords in topics.items():
        session_id = await manager.create_session(user_id, f"Learning {topic}")
        
        # Add messages with keywords
        for keyword in keywords:
            await manager.add_message(
                user_id, session_id,
                Message(role="user", content=f"Tell me about {keyword}")
            )
            await manager.add_message(
                user_id, session_id,
                Message(role="assistant", content=f"{keyword} is an important concept in {topic}...")
            )
    
    print_success("Created 3 sessions with 18 messages")
    
    # 1. Basic search
    print_info("\nTesting basic search...")
    results = await manager.search_messages(user_id, "Python")
    assert len(results) > 0, "No results found for 'Python'"
    print_success(f"Found {len(results)} messages containing 'Python'")
    
    # 2. Advanced text search
    print_info("Testing advanced search...")
    query = SearchQuery(
        query="functions",
        user_id=user_id,
        message_roles=["user"]  # Only search user messages
    )
    results = await manager.advanced_search(query)
    assert len(results) > 0, "No results for advanced search"
    assert all(r.metadata.get("role") == "user" for r in results), "Role filter not working"
    print_success(f"Found {len(results)} user messages about 'functions'")
    
    # 3. Entity-based search
    print_info("Testing entity extraction and search...")
    test_text = "Check out https://python.org and email me at user@example.com"
    entities = await manager.extract_entities(test_text)
    assert "urls" in entities, "URL extraction failed"
    assert "emails" in entities, "Email extraction failed"
    print_success(f"Extracted entities: {list(entities.keys())}")
    
    # 4. Time-range search
    print_info("Testing time-range search...")
    now = datetime.now()
    results = await manager.search_by_time_range(
        start_date=now - timedelta(hours=1),
        end_date=now,
        user_id=user_id
    )
    assert len(results) > 0, "No results in time range"
    print_success(f"Found {len(results)} recent messages")
    
    # 5. Search with relevance scoring
    print_info("Testing search relevance...")
    query = SearchQuery(
        query="pandas numpy",
        user_id=user_id,
        min_relevance_score=0.5
    )
    results = await manager.advanced_search(query)
    if results:
        print_success(f"Top result: '{results[0].content[:50]}...' (score: {results[0].relevance_score:.2f})")
    
    return True


async def test_panel_sessions(manager: StorageManager):
    """Test multi-persona panel functionality"""
    print_header("Testing Panel Sessions")
    
    # 1. Create personas
    print_info("Creating personas...")
    personas = [
        {"id": "analyst", "name": "Data Analyst", "expertise": ["data analysis", "statistics"]},
        {"id": "engineer", "name": "Software Engineer", "expertise": ["coding", "architecture"]},
        {"id": "designer", "name": "UX Designer", "expertise": ["user experience", "design"]}
    ]
    
    for persona in personas:
        result = await manager.save_persona(persona["id"], persona)
        assert result, f"Failed to save persona {persona['id']}"
    print_success("Created 3 personas")
    
    # 2. Create panel session
    print_info("Creating panel session...")
    panel_id = await manager.create_panel(
        "expert_discussion",
        ["analyst", "engineer", "designer"],
        config={"topic": "Building a Data Dashboard"}
    )
    assert panel_id, "Failed to create panel"
    print_success(f"Created panel: {panel_id}")
    
    # 3. Add panel messages
    print_info("Adding panel discussion...")
    from flatfile_chat_database.models import PanelMessage
    
    messages = [
        PanelMessage(
            role="analyst",
            content="We need to focus on data visualization and insights",
            persona_id="analyst"
        ),
        PanelMessage(
            role="engineer",
            content="I'll handle the backend API and data processing",
            persona_id="engineer"
        ),
        PanelMessage(
            role="designer",
            content="Let's ensure the dashboard is intuitive and user-friendly",
            persona_id="designer"
        )
    ]
    
    for msg in messages:
        result = await manager.add_panel_message(panel_id, msg)
        assert result, "Failed to add panel message"
    print_success("Added panel messages")
    
    # 4. Retrieve panel messages
    print_info("Retrieving panel discussion...")
    panel_messages = await manager.get_panel_messages(panel_id)
    assert len(panel_messages) == len(messages), "Panel message count mismatch"
    print_success(f"Retrieved {len(panel_messages)} panel messages")
    
    # 5. Save panel insight
    print_info("Saving panel insight...")
    from flatfile_chat_database.models import PanelInsight
    
    insight = PanelInsight(
        panel_id=panel_id,
        type="consensus",
        content="Team agrees on building a real-time dashboard with modular components",
        consensus_level=0.9,
        supporting_messages=[msg.id for msg in messages]
    )
    
    result = await manager.save_panel_insight(panel_id, insight)
    assert result, "Failed to save insight"
    print_success("Panel insight saved")
    
    return True


async def test_performance(manager: StorageManager):
    """Run performance tests"""
    print_header("Testing Performance")
    
    user_id = "perf_test_user"
    await manager.create_user(user_id)
    
    # 1. Session creation performance
    print_info("Testing session creation speed...")
    start = asyncio.get_event_loop().time()
    session_id = await manager.create_session(user_id, "Performance Test")
    duration = (asyncio.get_event_loop().time() - start) * 1000
    print_success(f"Session creation: {duration:.2f}ms (target: <10ms)")
    assert duration < 10, "Session creation too slow"
    
    # 2. Message append performance
    print_info("Testing message append speed...")
    times = []
    for i in range(10):
        msg = Message(role="user", content=f"Test message {i}")
        start = asyncio.get_event_loop().time()
        await manager.add_message(user_id, session_id, msg)
        duration = (asyncio.get_event_loop().time() - start) * 1000
        times.append(duration)
    
    avg_time = sum(times) / len(times)
    print_success(f"Message append average: {avg_time:.2f}ms (target: <5ms)")
    assert avg_time < 5, "Message append too slow"
    
    # 3. Message retrieval performance
    print_info("Testing message retrieval speed...")
    start = asyncio.get_event_loop().time()
    messages = await manager.get_messages(user_id, session_id)
    duration = (asyncio.get_event_loop().time() - start) * 1000
    print_success(f"Retrieved {len(messages)} messages in {duration:.2f}ms (target: <50ms)")
    assert duration < 50, "Message retrieval too slow"
    
    return True


async def test_error_handling(manager: StorageManager):
    """Test error handling and edge cases"""
    print_header("Testing Error Handling")
    
    # 1. Non-existent user
    print_info("Testing non-existent user handling...")
    profile = await manager.get_user_profile("non_existent_user")
    assert profile is None, "Should return None for non-existent user"
    print_success("Correctly handled non-existent user")
    
    # 2. Invalid session
    print_info("Testing invalid session handling...")
    messages = await manager.get_messages("user1", "invalid_session")
    assert messages == [], "Should return empty list for invalid session"
    print_success("Correctly handled invalid session")
    
    # 3. Large message handling
    print_info("Testing large message handling...")
    user_id = "error_test_user"
    await manager.create_user(user_id)
    session_id = await manager.create_session(user_id)
    
    # Try to add a message that exceeds size limit
    huge_content = "x" * (manager.config.max_message_size_bytes + 1000)
    huge_msg = Message(role="user", content=huge_content)
    result = await manager.add_message(user_id, session_id, huge_msg)
    assert not result, "Should reject oversized message"
    print_success("Correctly rejected oversized message")
    
    # 4. Concurrent access
    print_info("Testing concurrent access...")
    async def add_message_concurrent(i):
        msg = Message(role="user", content=f"Concurrent message {i}")
        return await manager.add_message(user_id, session_id, msg)
    
    # Add 10 messages concurrently
    tasks = [add_message_concurrent(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert all(results), "Some concurrent writes failed"
    
    # Verify all messages were saved
    messages = await manager.get_messages(user_id, session_id)
    assert len(messages) >= 10, "Not all concurrent messages were saved"
    print_success("Handled concurrent access correctly")
    
    return True


async def run_all_tests():
    """Run all tests and report results"""
    print(f"{TestColors.BOLD}Flatfile Chat Database - Complete Test Suite{TestColors.END}")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize storage manager
        config = StorageConfig(
            storage_base_path=temp_dir,
            max_messages_per_session=10000,
            session_list_default_limit=50
        )
        manager = StorageManager(config=config)
        await manager.initialize()
        
        # Run all test suites
        test_suites = [
            ("Basic Operations", test_basic_operations),
            ("Document Handling", test_document_handling),
            ("Context Management", test_context_management),
            ("Search Capabilities", test_search_capabilities),
            ("Panel Sessions", test_panel_sessions),
            ("Performance", test_performance),
            ("Error Handling", test_error_handling)
        ]
        
        results = {}
        for test_name, test_func in test_suites:
            try:
                result = await test_func(manager)
                results[test_name] = result
            except Exception as e:
                print_error(f"{test_name} failed with error: {e}")
                import traceback
                traceback.print_exc()
                results[test_name] = False
        
        # Print summary
        print_header("Test Summary")
        all_passed = True
        for test_name, passed in results.items():
            if passed:
                print_success(f"{test_name}: PASSED")
            else:
                print_error(f"{test_name}: FAILED")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print_success("All tests passed! The flatfile chat database is working correctly.")
        else:
            print_error("Some tests failed. Please check the errors above.")
        
        return all_passed


async def example_usage():
    """Show example usage of the storage system"""
    print_header("Example Usage")
    
    # Initialize
    config = StorageConfig(storage_base_path="./chat_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create a user
    await storage.create_user("alice", {"username": "Alice Smith"})
    
    # Start a chat session
    session_id = await storage.create_session("alice", "Python Help")
    
    # Add messages
    await storage.add_message("alice", session_id, 
                            Message(role="user", content="How do I read a CSV file in Python?"))
    
    await storage.add_message("alice", session_id,
                            Message(role="assistant", content="You can use pandas: `df = pd.read_csv('file.csv')`"))
    
    # Search across sessions
    results = await storage.search_messages("alice", "Python")
    print(f"Found {len(results)} messages about Python")
    
    print_info("See the test code for more examples!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Flatfile Chat Database")
    parser.add_argument("--example", action="store_true", help="Show example usage")
    args = parser.parse_args()
    
    if args.example:
        asyncio.run(example_usage())
    else:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)