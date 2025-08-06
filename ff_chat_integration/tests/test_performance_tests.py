"""
Performance tests for Chat Application Bridge System.

Validates 30% performance improvement claims and benchmarks operations.
"""

import pytest
import asyncio
import time
import statistics
from pathlib import Path

from ff_chat_integration import FFChatAppBridge
from . import BridgeTestHelper, PerformanceTester


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    async def test_message_storage_performance(self):
        """Benchmark message storage performance."""
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "speed"})
        data_layer = bridge.get_data_layer()
        
        # Setup test data
        user_id = "perf_storage_user"
        await data_layer.storage.create_user(user_id, {"name": "Perf Test"})
        session_id = await data_layer.storage.create_session(user_id, "Perf Session")
        
        # Benchmark message storage
        async def store_message():
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": "Performance test message"}
            )
        
        benchmark = await PerformanceTester.benchmark_operation(store_message, iterations=20)
        
        # Target: 30% better than 100ms baseline (70ms)
        assert benchmark["average_ms"] < 70, f"Message storage too slow: {benchmark['average_ms']:.1f}ms"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_history_retrieval_performance(self):
        """Benchmark history retrieval performance."""
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "speed"})
        data_layer = bridge.get_data_layer()
        
        # Setup test data
        user_id = "perf_retrieval_user"
        await data_layer.storage.create_user(user_id, {"name": "Perf Test"})
        session_id = await data_layer.storage.create_session(user_id, "Perf Session")
        
        # Add test messages
        for i in range(50):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Test message {i}"}
            )
        
        # Benchmark history retrieval
        async def get_history():
            await data_layer.get_chat_history(user_id, session_id, limit=50)
        
        benchmark = await PerformanceTester.benchmark_operation(get_history, iterations=15)
        
        # Target: 30% better than 150ms baseline (105ms)
        assert benchmark["average_ms"] < 105, f"History retrieval too slow: {benchmark['average_ms']:.1f}ms"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_search_performance(self):
        """Benchmark search performance."""
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "speed"})
        data_layer = bridge.get_data_layer()
        
        # Setup test data
        user_id = "perf_search_user"
        await data_layer.storage.create_user(user_id, {"name": "Perf Test"})
        session_id = await data_layer.storage.create_session(user_id, "Perf Session")
        
        # Add searchable messages
        search_terms = ["Python", "JavaScript", "database", "performance", "optimization"]
        for i, term in enumerate(search_terms * 10):  # 50 messages
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Message about {term} number {i}"}
            )
        
        # Benchmark search
        async def search_messages():
            await data_layer.search_conversations(
                user_id, "Python", {"search_type": "text", "limit": 10}
            )
        
        benchmark = await PerformanceTester.benchmark_operation(search_messages, iterations=10)
        
        # Target: 30% better than 200ms baseline (140ms)
        assert benchmark["average_ms"] < 140, f"Search too slow: {benchmark['average_ms']:.1f}ms"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "balanced"})
        data_layer = bridge.get_data_layer()
        
        # Setup test data
        user_id = "perf_concurrent_user"
        await data_layer.storage.create_user(user_id, {"name": "Concurrent Test"})
        session_id = await data_layer.storage.create_session(user_id, "Concurrent Session")
        
        async def concurrent_operations():
            """Perform mixed operations concurrently."""
            tasks = []
            
            # Message storage tasks
            for i in range(5):
                task = data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Concurrent message {i}"}
                )
                tasks.append(task)
            
            # History retrieval tasks
            for i in range(3):
                task = data_layer.get_chat_history(user_id, session_id, limit=10)
                tasks.append(task)
            
            # Wait for all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Benchmark concurrent operations
        benchmark = await PerformanceTester.benchmark_operation(concurrent_operations, iterations=5)
        
        # Should handle concurrent load efficiently
        assert benchmark["average_ms"] < 500, f"Concurrent operations too slow: {benchmark['average_ms']:.1f}ms"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_memory_usage_performance(self):
        """Test memory usage during operations."""
        import psutil
        import gc
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        # Perform memory-intensive operations
        user_id = "memory_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Memory Test"})
        session_id = await data_layer.storage.create_session(user_id, "Memory Session")
        
        # Store many messages
        for i in range(100):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Memory test message {i} with some extra content"}
            )
        
        # Retrieve history multiple times
        for i in range(10):
            await data_layer.get_chat_history(user_id, session_id, limit=50)
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - baseline_memory
        
        # Should not use excessive memory
        assert memory_growth < 100, f"Excessive memory usage: {memory_growth:.1f}MB growth"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
        
        # Force garbage collection
        gc.collect()


class TestPerformanceComparison:
    """Compare bridge performance against baseline implementations."""
    
    async def test_vs_direct_storage_comparison(self):
        """Compare bridge performance vs direct storage usage."""
        # Create bridge
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "speed"})
        data_layer = bridge.get_data_layer()
        
        # Setup
        user_id = "comparison_user"
        await data_layer.storage.create_user(user_id, {"name": "Comparison Test"})
        session_id = await data_layer.storage.create_session(user_id, "Comparison Session")
        
        # Benchmark bridge operations
        async def bridge_store_message():
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": "Bridge test message"}
            )
        
        bridge_benchmark = await PerformanceTester.benchmark_operation(
            bridge_store_message, iterations=20
        )
        
        # Benchmark direct storage (simulated baseline)
        async def direct_store_message():
            from flatfile_chat_database.models import Message
            msg = Message(
                role="user",
                content="Direct test message",
                message_id=f"test_msg_{time.time()}"
            )
            await data_layer.storage.add_message(user_id, session_id, msg)
        
        direct_benchmark = await PerformanceTester.benchmark_operation(
            direct_store_message, iterations=20
        )
        
        # Bridge should add minimal overhead (less than 20% slower)
        overhead_ratio = bridge_benchmark["average_ms"] / direct_benchmark["average_ms"]
        assert overhead_ratio < 1.2, f"Bridge adds too much overhead: {overhead_ratio:.2f}x"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_cache_performance_benefit(self):
        """Test cache performance benefits."""
        bridge = await BridgeTestHelper.create_test_bridge({"cache_size_mb": 100})
        data_layer = bridge.get_data_layer()
        
        # Setup
        user_id = "cache_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Cache Test"})
        session_id = await data_layer.storage.create_session(user_id, "Cache Session")
        
        # Add messages
        for i in range(20):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Cache test message {i}"}
            )
        
        # First retrieval (no cache)
        first_start = time.time()
        result1 = await data_layer.get_chat_history(user_id, session_id, limit=20)
        first_time = (time.time() - first_start) * 1000
        
        # Second retrieval (should hit cache)
        second_start = time.time()
        result2 = await data_layer.get_chat_history(user_id, session_id, limit=20)
        second_time = (time.time() - second_start) * 1000
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Cache should provide performance benefit
        if result2["metadata"]["performance_metrics"]["cache_hit"]:
            cache_improvement = (first_time - second_time) / first_time
            assert cache_improvement > 0.1, f"Cache provides minimal benefit: {cache_improvement:.2%}"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    async def test_performance_baseline_comparison(self):
        """Test current performance against established baselines."""
        # These baselines represent expected performance targets
        baselines = {
            "message_storage": 50.0,  # ms
            "history_retrieval": 80.0,  # ms  
            "search_operation": 120.0,  # ms
        }
        
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "speed"})
        data_layer = bridge.get_data_layer()
        
        # Setup test data
        user_id = "baseline_user"
        await data_layer.storage.create_user(user_id, {"name": "Baseline Test"})
        session_id = await data_layer.storage.create_session(user_id, "Baseline Session")
        
        # Benchmark message storage
        async def store_operation():
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": "Baseline test message"}
            )
        
        storage_benchmark = await PerformanceTester.benchmark_operation(store_operation, 10)
        assert storage_benchmark["average_ms"] < baselines["message_storage"], \
            f"Message storage regression: {storage_benchmark['average_ms']:.1f}ms > {baselines['message_storage']}ms"
        
        # Add messages for history test
        for i in range(30):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"History test message {i}"}
            )
        
        # Benchmark history retrieval
        async def history_operation():
            await data_layer.get_chat_history(user_id, session_id, limit=30)
        
        history_benchmark = await PerformanceTester.benchmark_operation(history_operation, 10)
        assert history_benchmark["average_ms"] < baselines["history_retrieval"], \
            f"History retrieval regression: {history_benchmark['average_ms']:.1f}ms > {baselines['history_retrieval']}ms"
        
        # Benchmark search operation
        async def search_operation():
            await data_layer.search_conversations(
                user_id, "test", {"search_type": "text", "limit": 10}
            )
        
        search_benchmark = await PerformanceTester.benchmark_operation(search_operation, 5)
        assert search_benchmark["average_ms"] < baselines["search_operation"], \
            f"Search regression: {search_benchmark['average_ms']:.1f}ms > {baselines['search_operation']}ms"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_performance_consistency(self):
        """Test performance consistency across multiple runs."""
        bridge = await BridgeTestHelper.create_test_bridge({"performance_mode": "balanced"})
        data_layer = bridge.get_data_layer()
        
        # Setup
        user_id = "consistency_user"
        await data_layer.storage.create_user(user_id, {"name": "Consistency Test"})
        session_id = await data_layer.storage.create_session(user_id, "Consistency Session")
        
        # Run multiple benchmarks
        benchmarks = []
        for run in range(5):
            async def test_operation():
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Consistency test run {run}"}
                )
            
            benchmark = await PerformanceTester.benchmark_operation(test_operation, 10)
            benchmarks.append(benchmark["average_ms"])
        
        # Check for consistency (coefficient of variation < 0.3)
        mean_time = statistics.mean(benchmarks)
        std_dev = statistics.stdev(benchmarks)
        cv = std_dev / mean_time
        
        assert cv < 0.3, f"Performance inconsistent: CV={cv:.3f} (times: {benchmarks})"
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)