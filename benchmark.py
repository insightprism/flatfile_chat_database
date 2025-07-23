"""
Performance benchmarking for the flatfile chat database.

Measures performance of various operations to ensure they meet targets.
"""

import asyncio
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
import random
import string

from .config import StorageConfig
from .storage import StorageManager
from .models import Message, Document, SituationalContext
from .search import SearchQuery


class BenchmarkResult:
    """Container for benchmark results"""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.times: List[float] = []
        self.errors = 0
        self.start_time = None
        self.end_time = None
    
    def add_timing(self, duration: float):
        """Add a timing measurement"""
        self.times.append(duration)
    
    def add_error(self):
        """Record an error"""
        self.errors += 1
    
    @property
    def count(self) -> int:
        """Number of successful operations"""
        return len(self.times)
    
    @property
    def mean(self) -> float:
        """Average time in milliseconds"""
        return statistics.mean(self.times) * 1000 if self.times else 0
    
    @property
    def median(self) -> float:
        """Median time in milliseconds"""
        return statistics.median(self.times) * 1000 if self.times else 0
    
    @property
    def min(self) -> float:
        """Minimum time in milliseconds"""
        return min(self.times) * 1000 if self.times else 0
    
    @property
    def max(self) -> float:
        """Maximum time in milliseconds"""
        return max(self.times) * 1000 if self.times else 0
    
    @property
    def p95(self) -> float:
        """95th percentile in milliseconds"""
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] * 1000
    
    @property
    def throughput(self) -> float:
        """Operations per second"""
        if not self.times:
            return 0
        total_time = sum(self.times)
        return len(self.times) / total_time if total_time > 0 else 0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "operation": self.operation,
            "count": self.count,
            "errors": self.errors,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "p95_ms": round(self.p95, 2),
            "throughput_ops": round(self.throughput, 2)
        }


class PerformanceBenchmark:
    """
    Performance benchmarking suite for the flatfile chat database.
    """
    
    def __init__(self, iterations: int = 100, warmup: int = 10):
        """
        Initialize benchmark suite.
        
        Args:
            iterations: Number of iterations per test
            warmup: Number of warmup iterations
        """
        self.iterations = iterations
        self.warmup = warmup
        self.results: Dict[str, BenchmarkResult] = {}
    
    async def run_all_benchmarks(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all performance benchmarks.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Benchmark results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(storage_base_path=temp_dir)
            manager = StorageManager(config=config)
            await manager.initialize()
            
            if verbose:
                print("Running Performance Benchmarks")
                print("=" * 50)
            
            # Run benchmarks
            await self._benchmark_user_operations(manager, verbose)
            await self._benchmark_session_operations(manager, verbose)
            await self._benchmark_message_operations(manager, verbose)
            await self._benchmark_document_operations(manager, verbose)
            await self._benchmark_search_operations(manager, verbose)
            await self._benchmark_concurrent_operations(manager, verbose)
            
            # Compile results
            results = {
                "summary": self._get_summary(),
                "details": {name: result.summary() for name, result in self.results.items()},
                "performance_targets": self._check_performance_targets()
            }
            
            if verbose:
                print("\n" + "=" * 50)
                print("Benchmark Summary:")
                self._print_results()
            
            return results
    
    async def _benchmark_user_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark user-related operations"""
        if verbose:
            print("\nBenchmarking User Operations...")
        
        # User creation
        result = BenchmarkResult("user_create")
        for i in range(self.iterations + self.warmup):
            user_id = f"bench_user_{i}"
            start = time.perf_counter()
            try:
                await manager.create_user(user_id, {"username": f"User {i}"})
                duration = time.perf_counter() - start
                if i >= self.warmup:
                    result.add_timing(duration)
            except Exception:
                if i >= self.warmup:
                    result.add_error()
        self.results["user_create"] = result
        
        # User retrieval
        result = BenchmarkResult("user_get")
        for i in range(self.iterations + self.warmup):
            user_id = f"bench_user_{i % 10}"  # Reuse some users
            start = time.perf_counter()
            try:
                await manager.get_user_profile(user_id)
                duration = time.perf_counter() - start
                if i >= self.warmup:
                    result.add_timing(duration)
            except Exception:
                if i >= self.warmup:
                    result.add_error()
        self.results["user_get"] = result
    
    async def _benchmark_session_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark session-related operations"""
        if verbose:
            print("Benchmarking Session Operations...")
        
        # Create a test user
        user_id = "bench_session_user"
        await manager.create_user(user_id)
        
        # Session creation
        result = BenchmarkResult("session_create")
        session_ids = []
        for i in range(self.iterations + self.warmup):
            start = time.perf_counter()
            try:
                session_id = await manager.create_session(user_id, f"Session {i}")
                duration = time.perf_counter() - start
                if i >= self.warmup:
                    result.add_timing(duration)
                    session_ids.append(session_id)
            except Exception:
                if i >= self.warmup:
                    result.add_error()
        self.results["session_create"] = result
        
        # Session listing
        result = BenchmarkResult("session_list")
        for i in range(min(20, self.iterations)):  # Limit list operations
            start = time.perf_counter()
            try:
                await manager.list_sessions(user_id, limit=10)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        self.results["session_list"] = result
    
    async def _benchmark_message_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark message-related operations"""
        if verbose:
            print("Benchmarking Message Operations...")
        
        # Setup
        user_id = "bench_message_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        # Message append
        result = BenchmarkResult("message_append")
        for i in range(self.iterations + self.warmup):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Test message {i}: " + "x" * 100  # ~100 char message
            )
            start = time.perf_counter()
            try:
                await manager.add_message(user_id, session_id, msg)
                duration = time.perf_counter() - start
                if i >= self.warmup:
                    result.add_timing(duration)
            except Exception:
                if i >= self.warmup:
                    result.add_error()
        self.results["message_append"] = result
        
        # Message retrieval (100 messages)
        result = BenchmarkResult("message_get_100")
        for i in range(min(20, self.iterations)):  # Limit read operations
            start = time.perf_counter()
            try:
                await manager.get_messages(user_id, session_id, limit=100)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        self.results["message_get_100"] = result
    
    async def _benchmark_document_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark document-related operations"""
        if verbose:
            print("Benchmarking Document Operations...")
        
        # Setup
        user_id = "bench_doc_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id)
        
        # Document upload (1MB)
        result = BenchmarkResult("document_upload_1mb")
        doc_content = b"x" * (1024 * 1024)  # 1MB
        
        for i in range(min(20, self.iterations)):  # Limit large operations
            filename = f"test_doc_{i}.txt"
            start = time.perf_counter()
            try:
                await manager.save_document(user_id, session_id, filename, doc_content)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        self.results["document_upload_1mb"] = result
    
    async def _benchmark_search_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark search operations"""
        if verbose:
            print("Benchmarking Search Operations...")
        
        # Setup test data
        user_id = "bench_search_user"
        await manager.create_user(user_id)
        
        # Create sessions with searchable content
        for i in range(10):
            session_id = await manager.create_session(user_id, f"Search Session {i}")
            for j in range(100):
                msg = Message(
                    role="user" if j % 2 == 0 else "assistant",
                    content=f"Message about {'Python' if i < 5 else 'JavaScript'} programming"
                )
                await manager.add_message(user_id, session_id, msg)
        
        # Basic search
        result = BenchmarkResult("search_basic")
        for i in range(min(20, self.iterations)):
            query = "Python" if i % 2 == 0 else "JavaScript"
            start = time.perf_counter()
            try:
                await manager.search_messages(user_id, query)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        self.results["search_basic"] = result
        
        # Advanced search
        result = BenchmarkResult("search_advanced")
        for i in range(min(20, self.iterations)):
            query = SearchQuery(
                query="programming",
                user_id=user_id,
                start_date=datetime.now(),
                message_roles=["user"]
            )
            start = time.perf_counter()
            try:
                await manager.advanced_search(query)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        self.results["search_advanced"] = result
    
    async def _benchmark_concurrent_operations(self, manager: StorageManager, verbose: bool):
        """Benchmark concurrent operations"""
        if verbose:
            print("Benchmarking Concurrent Operations...")
        
        # Setup
        user_id = "bench_concurrent_user"
        await manager.create_user(user_id)
        
        # Concurrent message writes
        result = BenchmarkResult("concurrent_writes")
        
        async def write_message(session_id: str, msg_num: int):
            msg = Message(role="user", content=f"Concurrent message {msg_num}")
            await manager.add_message(user_id, session_id, msg)
        
        for i in range(min(10, self.iterations // 10)):
            session_id = await manager.create_session(user_id)
            
            start = time.perf_counter()
            try:
                # Write 10 messages concurrently
                tasks = [write_message(session_id, j) for j in range(10)]
                await asyncio.gather(*tasks)
                duration = time.perf_counter() - start
                result.add_timing(duration)
            except Exception:
                result.add_error()
        
        self.results["concurrent_writes"] = result
    
    def _check_performance_targets(self) -> Dict[str, bool]:
        """Check if performance targets are met"""
        targets = {
            "session_create": 10,      # < 10ms
            "message_append": 5,       # < 5ms
            "message_get_100": 50,     # < 50ms for 100 messages
            "search_basic": 200,       # < 200ms for 1000 sessions
            "document_upload_1mb": 100 # < 100ms for 1MB
        }
        
        results = {}
        for operation, target_ms in targets.items():
            if operation in self.results:
                actual_p95 = self.results[operation].p95
                results[operation] = {
                    "target_ms": target_ms,
                    "actual_p95_ms": round(actual_p95, 2),
                    "passed": actual_p95 <= target_ms
                }
        
        return results
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get overall summary"""
        total_ops = sum(r.count for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        
        return {
            "total_operations": total_ops,
            "total_errors": total_errors,
            "error_rate": total_errors / total_ops if total_ops > 0 else 0,
            "fastest_operation": min(self.results.items(), 
                                    key=lambda x: x[1].median)[0] if self.results else None,
            "slowest_operation": max(self.results.items(), 
                                   key=lambda x: x[1].median)[0] if self.results else None
        }
    
    def _print_results(self):
        """Print formatted results"""
        print("\nOperation Performance:")
        print("-" * 80)
        print(f"{'Operation':<25} {'Mean':<10} {'Median':<10} {'P95':<10} {'Throughput':<15}")
        print("-" * 80)
        
        for name, result in sorted(self.results.items()):
            print(f"{name:<25} {result.mean:>8.2f}ms {result.median:>8.2f}ms "
                  f"{result.p95:>8.2f}ms {result.throughput:>10.2f} ops/s")
        
        print("\nPerformance Targets:")
        print("-" * 60)
        targets = self._check_performance_targets()
        for op, data in targets.items():
            status = "✓ PASS" if data["passed"] else "✗ FAIL"
            print(f"{op:<25} Target: {data['target_ms']:>6}ms  "
                  f"Actual: {data['actual_p95_ms']:>6}ms  {status}")


async def run_quick_benchmark():
    """Run a quick benchmark with fewer iterations"""
    benchmark = PerformanceBenchmark(iterations=20, warmup=5)
    results = await benchmark.run_all_benchmarks(verbose=True)
    return results


if __name__ == "__main__":
    # Run benchmark when executed directly
    asyncio.run(run_quick_benchmark())