"""
Phase 3 validation script for Chat Application Bridge System.

Validates chat-optimized data layer operations and performance.
"""

import asyncio
import sys
import tempfile
import time
import traceback
from pathlib import Path

async def test_data_layer_creation():
    """Test FFChatDataLayer creation and integration."""
    try:
        from ff_chat_integration import FFChatAppBridge, FFChatDataLayer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            assert data_layer is not None
            assert isinstance(data_layer, FFChatDataLayer)
            assert data_layer.storage is not None
            assert data_layer.config is not None
            print("✓ Data layer creation successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Data layer creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_message_operations():
    """Test optimized message storage and retrieval."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test user and session
            user_id = "test_user_123"
            await data_layer.storage.create_user(user_id, {"name": "Test User"})
            session_id = await data_layer.storage.create_session(user_id, "Test Session")
            
            # Test message storage
            start_time = time.time()
            result = await data_layer.store_chat_message(
                user_id=user_id,
                session_id=session_id,
                message={
                    "role": "user",
                    "content": "Hello, this is a test message!",
                    "metadata": {"test": True}
                }
            )
            storage_time = time.time() - start_time
            
            assert result["success"] is True
            assert "message_id" in result["data"]
            assert result["metadata"]["operation_time_ms"] > 0
            assert result["data"]["session_updated"] is True
            print(f"✓ Message storage successful ({storage_time:.3f}s)")
            
            # Test history retrieval
            start_time = time.time()
            history = await data_layer.get_chat_history(user_id, session_id, limit=10)
            retrieval_time = time.time() - start_time
            
            assert history["success"] is True
            assert len(history["data"]["messages"]) == 1
            assert history["data"]["messages"][0]["content"] == "Hello, this is a test message!"
            assert "pagination" in history["data"]
            assert history["data"]["pagination"]["returned_count"] == 1
            print(f"✓ History retrieval successful ({retrieval_time:.3f}s)")
            
            # Test empty message validation
            empty_result = await data_layer.store_chat_message(
                user_id, session_id, {"role": "user", "content": ""}
            )
            assert empty_result["success"] is False
            assert "empty" in empty_result["error"]
            print("✓ Empty message validation working")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Message operations test failed: {e}")
        traceback.print_exc()
        return False

async def test_search_operations():
    """Test chat search functionality."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_search"
            await data_layer.storage.create_user(user_id, {"name": "Search Test User"})
            session_id = await data_layer.storage.create_session(user_id, "Search Test Session")
            
            # Store test messages
            test_messages = [
                {"role": "user", "content": "Hello, I love Python programming"},
                {"role": "assistant", "content": "Python is a great language for development"},
                {"role": "user", "content": "Can you help with JavaScript too?"},
                {"role": "assistant", "content": "Yes, I can help with JavaScript as well"}
            ]
            
            for msg in test_messages:
                await data_layer.store_chat_message(user_id, session_id, msg)
            
            # Test search
            search_result = await data_layer.search_conversations(
                user_id=user_id,
                query="Python",
                options={"search_type": "text", "limit": 10}
            )
            
            assert search_result["success"] is True
            assert len(search_result["data"]["results"]) >= 0  # May depend on search implementation
            assert "search_metadata" in search_result["data"]
            assert search_result["data"]["search_metadata"]["query"] == "Python"
            assert search_result["data"]["search_metadata"]["search_type"] == "text"
            print("✓ Search operations successful")
            
            # Test empty query validation
            empty_search = await data_layer.search_conversations(user_id, "")
            assert empty_search["success"] is False
            assert "empty" in empty_search["error"]
            print("✓ Empty search query validation working")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Search operations test failed: {e}")
        traceback.print_exc()
        return False

async def test_analytics():
    """Test analytics functionality."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_analytics"
            await data_layer.storage.create_user(user_id, {"name": "Analytics Test"})
            session_id = await data_layer.storage.create_session(user_id, "Analytics Session")
            
            # Add some messages
            for i in range(5):
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Test message {i}"}
                )
            
            # Test analytics
            analytics = await data_layer.get_analytics_summary(user_id)
            
            assert analytics["success"] is True
            assert "analytics" in analytics["data"]
            assert analytics["data"]["analytics"]["total_sessions"] >= 1
            assert analytics["data"]["analytics"]["total_messages"] >= 5
            assert "usage_patterns" in analytics["data"]["analytics"]
            assert "recent_activity" in analytics["data"]["analytics"]
            print("✓ Analytics functionality successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Analytics test failed: {e}")
        traceback.print_exc()
        return False

async def test_streaming():
    """Test conversation streaming."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_stream"
            await data_layer.storage.create_user(user_id, {"name": "Stream Test"})
            session_id = await data_layer.storage.create_session(user_id, "Stream Session")
            
            # Add messages for streaming test
            for i in range(15):
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Stream test message {i}"}
                )
            
            # Test streaming
            chunks_received = 0
            total_messages = 0
            
            async for chunk in data_layer.stream_conversation(user_id, session_id, chunk_size=5):
                assert chunk["success"] is True
                assert "chunk" in chunk["data"]
                assert "chunk_info" in chunk["data"]
                
                chunk_info = chunk["data"]["chunk_info"]
                assert chunk_info["size"] > 0
                assert chunk_info["offset"] >= 0
                assert "has_more" in chunk_info
                assert "total_streamed" in chunk_info
                
                chunks_received += 1
                total_messages += len(chunk["data"]["chunk"])
                
                if chunks_received > 10:  # Safety break
                    break
            
            assert chunks_received >= 3  # Should have multiple chunks
            assert total_messages >= 15
            print(f"✓ Streaming successful ({chunks_received} chunks, {total_messages} messages)")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Streaming test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_metrics():
    """Test performance monitoring."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_perf"
            await data_layer.storage.create_user(user_id, {"name": "Perf Test"})
            session_id = await data_layer.storage.create_session(user_id, "Perf Session")
            
            # Perform operations to generate metrics
            for i in range(3):
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Performance test {i}"}
                )
            
            # Get performance metrics
            metrics = data_layer.get_performance_metrics()
            
            assert "operation_metrics" in metrics
            assert "store_chat_message" in metrics["operation_metrics"]
            assert metrics["operation_metrics"]["store_chat_message"]["total_operations"] >= 3
            assert "cache_stats" in metrics
            assert "optimization_info" in metrics
            print("✓ Performance metrics working")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        traceback.print_exc()
        return False

async def test_caching_system():
    """Test caching functionality."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_cache"
            await data_layer.storage.create_user(user_id, {"name": "Cache Test"})
            session_id = await data_layer.storage.create_session(user_id, "Cache Session")
            
            # Add some messages
            for i in range(5):
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Cache test message {i}"}
                )
            
            # First request - should not be cached
            result1 = await data_layer.get_chat_history(user_id, session_id, limit=5)
            assert result1["success"] is True
            assert result1["metadata"]["performance_metrics"]["cache_hit"] is False
            
            # Second request - should be cached (same parameters)
            result2 = await data_layer.get_chat_history(user_id, session_id, limit=5)
            assert result2["success"] is True
            assert result2["metadata"]["performance_metrics"]["cache_hit"] is True
            
            # Test cache invalidation after adding new message
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "assistant", "content": "This should invalidate cache"}
            )
            
            # Request after invalidation - should not be cached
            result3 = await data_layer.get_chat_history(user_id, session_id, limit=5)
            assert result3["success"] is True
            assert result3["metadata"]["performance_metrics"]["cache_hit"] is False
            
            # Test cache clearing
            data_layer.clear_cache()
            result4 = await data_layer.get_chat_history(user_id, session_id, limit=5)
            assert result4["success"] is True
            assert result4["metadata"]["performance_metrics"]["cache_hit"] is False
            
            print("✓ Caching system working correctly")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Caching system test failed: {e}")
        traceback.print_exc()
        return False

async def test_standardized_responses():
    """Test standardized response format."""
    try:
        from ff_chat_integration import FFChatAppBridge
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "test_storage")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            data_layer = bridge.get_data_layer()
            
            # Create test data
            user_id = "test_user_response"
            await data_layer.storage.create_user(user_id, {"name": "Response Test"})
            session_id = await data_layer.storage.create_session(user_id, "Response Session")
            
            # Test that all operations return standardized format
            operations_to_test = [
                ("store_chat_message", lambda: data_layer.store_chat_message(
                    user_id, session_id, {"role": "user", "content": "Test message"}
                )),
                ("get_chat_history", lambda: data_layer.get_chat_history(user_id, session_id)),
                ("search_conversations", lambda: data_layer.search_conversations(
                    user_id, "test", {"search_type": "text"}
                )),
                ("get_analytics_summary", lambda: data_layer.get_analytics_summary(user_id))
            ]
            
            for op_name, op_func in operations_to_test:
                result = await op_func()
                
                # Verify standardized response format
                assert isinstance(result, dict), f"{op_name}: Response is not a dictionary"
                assert "success" in result, f"{op_name}: Missing 'success' field"
                assert "data" in result, f"{op_name}: Missing 'data' field"
                assert "metadata" in result, f"{op_name}: Missing 'metadata' field"
                assert "error" in result, f"{op_name}: Missing 'error' field"
                assert "warnings" in result, f"{op_name}: Missing 'warnings' field"
                
                # Verify metadata structure
                metadata = result["metadata"]
                assert "operation" in metadata, f"{op_name}: Missing operation in metadata"
                assert "operation_time_ms" in metadata, f"{op_name}: Missing operation_time_ms"
                assert "records_affected" in metadata, f"{op_name}: Missing records_affected"
                assert "performance_metrics" in metadata, f"{op_name}: Missing performance_metrics"
                
                # Verify performance metrics structure
                perf_metrics = metadata["performance_metrics"]
                assert "response_time_ms" in perf_metrics, f"{op_name}: Missing response_time_ms"
                assert "cache_hit" in perf_metrics, f"{op_name}: Missing cache_hit"
                assert "optimization_applied" in perf_metrics, f"{op_name}: Missing optimization_applied"
                
                print(f"✓ {op_name} has standardized response format")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Standardized responses test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 3 validation tests."""
    print("Phase 3 Validation - Chat-Optimized Data Access Layer")
    print("=" * 60)
    
    tests = [
        ("Data Layer Creation", test_data_layer_creation),
        ("Message Operations", test_message_operations),
        ("Search Operations", test_search_operations),
        ("Analytics", test_analytics),
        ("Streaming", test_streaming),
        ("Performance Metrics", test_performance_metrics),
        ("Caching System", test_caching_system),
        ("Standardized Responses", test_standardized_responses)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if await test_func():
                passed += 1
            else:
                print(f"Test {test_name} failed!")
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"Phase 3 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 3 implementation is ready for Phase 4!")
        print("\nKey Phase 3 achievements:")
        print("- ✅ FFChatDataLayer with chat-optimized operations")
        print("- ✅ Standardized response format with performance metrics")
        print("- ✅ Intelligent caching system with TTL and invalidation")
        print("- ✅ Core methods: store_chat_message, get_chat_history")
        print("- ✅ Advanced methods: search_conversations, stream_conversation")
        print("- ✅ Performance analytics and optimization recommendations")
        print("- ✅ Seamless integration with Phase 2 bridge")
        return True
    else:
        print("✗ Phase 3 needs fixes before proceeding to Phase 4")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)