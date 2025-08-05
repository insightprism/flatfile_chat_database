"""
Performance testing framework for the flatfile chat database system.

Provides comprehensive performance testing utilities, benchmarking tools,
and regression detection for ensuring system performance standards.
"""

import asyncio
import time
import statistics
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
import json

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO
from tests.test_factories import TestDataFactory


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    items_processed: int = 0
    throughput_per_second: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Calculate throughput after initialization."""
        if self.duration_seconds > 0 and self.items_processed > 0:
            self.throughput_per_second = self.items_processed / self.duration_seconds


@dataclass
class PerformanceBenchmark:
    """Performance benchmark thresholds and expectations."""
    operation: str
    max_duration_seconds: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_throughput_per_second: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    
    def evaluate(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate metrics against benchmark thresholds."""
        results = {
            "operation": self.operation,
            "passed": True,
            "violations": [],
            "metrics": metrics
        }
        
        if self.max_duration_seconds and metrics.duration_seconds > self.max_duration_seconds:
            results["passed"] = False
            results["violations"].append(
                f"Duration {metrics.duration_seconds:.3f}s exceeds limit {self.max_duration_seconds}s"
            )
        
        if self.max_memory_mb and metrics.memory_usage_mb > self.max_memory_mb:
            results["passed"] = False
            results["violations"].append(
                f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds limit {self.max_memory_mb}MB"
            )
        
        if self.min_throughput_per_second and metrics.throughput_per_second < self.min_throughput_per_second:
            results["passed"] = False
            results["violations"].append(
                f"Throughput {metrics.throughput_per_second:.1f}/s below minimum {self.min_throughput_per_second}/s"
            )
        
        if self.max_cpu_percent and metrics.cpu_usage_percent > self.max_cpu_percent:
            results["passed"] = False
            results["violations"].append(
                f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds limit {self.max_cpu_percent}%"
            )
        
        return results


class PerformanceProfiler:
    """Advanced performance profiler with detailed metrics collection."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_time = None
        self.metrics_history = []
    
    @asynccontextmanager
    async def profile_operation(self, operation_name: str, items_count: int = 0):
        """Context manager for profiling async operations."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Record start metrics
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_time = self.process.cpu_percent()
        
        try:
            yield self
        finally:
            # Record end metrics
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            duration = end_time - self.start_time
            memory_usage = max(end_memory - self.start_memory, 0)
            cpu_usage = max(end_cpu, self.start_cpu_time)
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                items_processed=items_count
            )
            
            self.metrics_history.append(metrics)
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistical summary for a specific operation."""
        operation_metrics = [m for m in self.metrics_history if m.operation == operation_name]
        
        if not operation_metrics:
            return {"operation": operation_name, "count": 0}
        
        durations = [m.duration_seconds for m in operation_metrics]
        memory_usage = [m.memory_usage_mb for m in operation_metrics]
        throughputs = [m.throughput_per_second for m in operation_metrics if m.throughput_per_second > 0]
        
        return {
            "operation": operation_name,
            "count": len(operation_metrics),
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "memory_stats": {
                "mean": statistics.mean(memory_usage),
                "median": statistics.median(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage)
            },
            "throughput_stats": {
                "mean": statistics.mean(throughputs) if throughputs else 0,
                "median": statistics.median(throughputs) if throughputs else 0,
                "min": min(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0
            } if throughputs else None
        }
    
    def clear_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
    
    def export_metrics(self, file_path: Path):
        """Export metrics history to JSON file."""
        metrics_data = [
            {
                "operation": m.operation,
                "duration_seconds": m.duration_seconds,
                "memory_usage_mb": m.memory_usage_mb,
                "cpu_usage_percent": m.cpu_usage_percent,
                "items_processed": m.items_processed,
                "throughput_per_second": m.throughput_per_second,
                "timestamp": m.timestamp
            }
            for m in self.metrics_history
        ]
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)


class PerformanceTestSuite:
    """Comprehensive performance test suite for the storage system."""
    
    def __init__(self, config_environment: str = "test"):
        """Initialize the test suite."""
        self.config = load_config(environment=config_environment)
        self.profiler = PerformanceProfiler()
        self.factory = TestDataFactory()
        self.benchmarks = self._create_default_benchmarks()
    
    def _create_default_benchmarks(self) -> Dict[str, PerformanceBenchmark]:
        """Create default performance benchmarks."""
        return {
            "user_creation": PerformanceBenchmark(
                operation="user_creation",
                max_duration_seconds=0.1,
                max_memory_mb=50,
                min_throughput_per_second=100
            ),
            "session_creation": PerformanceBenchmark(
                operation="session_creation",
                max_duration_seconds=0.2,
                max_memory_mb=50,
                min_throughput_per_second=50
            ),
            "message_insertion": PerformanceBenchmark(
                operation="message_insertion",
                max_duration_seconds=0.05,
                max_memory_mb=20,
                min_throughput_per_second=200
            ),
            "message_retrieval": PerformanceBenchmark(
                operation="message_retrieval",
                max_duration_seconds=0.5,
                max_memory_mb=100,
                min_throughput_per_second=1000
            ),
            "search_operation": PerformanceBenchmark(
                operation="search_operation",
                max_duration_seconds=2.0,
                max_memory_mb=200,
                min_throughput_per_second=10
            ),
            "bulk_message_insertion": PerformanceBenchmark(
                operation="bulk_message_insertion",
                max_duration_seconds=10.0,
                max_memory_mb=500,
                min_throughput_per_second=100
            )
        }
    
    async def run_basic_operation_tests(self, storage: FFStorageManager) -> Dict[str, Any]:
        """Run basic operation performance tests."""
        results = {}
        
        # Test user creation performance
        async with self.profiler.profile_operation("user_creation", 10):
            for i in range(10):
                await storage.create_user(f"perf_user_{i}", {"name": f"User {i}"})
        
        results["user_creation"] = self._evaluate_latest_metrics("user_creation")
        
        # Test session creation performance
        async with self.profiler.profile_operation("session_creation", 20):
            for i in range(20):
                await storage.create_session(f"perf_user_{i % 10}", f"Session {i}")
        
        results["session_creation"] = self._evaluate_latest_metrics("session_creation")
        
        # Create a test session for message tests
        session_id = await storage.create_session("perf_user_0", "Message Test Session")
        
        # Test message insertion performance
        messages = self.factory.create_message_batch(100)
        async with self.profiler.profile_operation("message_insertion", len(messages)):
            for message in messages:
                await storage.add_message("perf_user_0", session_id, message)
        
        results["message_insertion"] = self._evaluate_latest_metrics("message_insertion")
        
        # Test message retrieval performance
        async with self.profiler.profile_operation("message_retrieval", 100):
            retrieved_messages = await storage.get_all_messages("perf_user_0", session_id)
            assert len(retrieved_messages) == 100
        
        results["message_retrieval"] = self._evaluate_latest_metrics("message_retrieval")
        
        return results
    
    async def run_bulk_operation_tests(self, storage: FFStorageManager) -> Dict[str, Any]:
        """Run bulk operation performance tests."""
        results = {}
        
        # Create test user and session
        await storage.create_user("bulk_user", {"name": "Bulk Test User"})
        session_id = await storage.create_session("bulk_user", "Bulk Test Session")
        
        # Test bulk message insertion
        bulk_messages = self.factory.create_message_batch(1000)
        async with self.profiler.profile_operation("bulk_message_insertion", len(bulk_messages)):
            for i, message in enumerate(bulk_messages):
                await storage.add_message("bulk_user", session_id, message)
                
                # Yield control periodically to prevent blocking
                if i % 100 == 0:
                    await asyncio.sleep(0.001)
        
        results["bulk_message_insertion"] = self._evaluate_latest_metrics("bulk_message_insertion")
        
        # Test bulk message retrieval
        async with self.profiler.profile_operation("bulk_message_retrieval", 1000):
            retrieved_messages = await storage.get_all_messages("bulk_user", session_id)
            assert len(retrieved_messages) == 1000
        
        results["bulk_message_retrieval"] = self._evaluate_latest_metrics("bulk_message_retrieval")
        
        return results
    
    async def run_concurrent_operation_tests(self, storage: FFStorageManager, concurrency: int = 10) -> Dict[str, Any]:
        """Run concurrent operation performance tests."""
        results = {}
        
        # Create test users concurrently
        async def create_user_batch(batch_id: int):
            for i in range(10):
                user_id = f"concurrent_user_{batch_id}_{i}"
                await storage.create_user(user_id, {"batch": batch_id, "index": i})
        
        async with self.profiler.profile_operation("concurrent_user_creation", concurrency * 10):
            tasks = [create_user_batch(i) for i in range(concurrency)]
            await asyncio.gather(*tasks)
        
        results["concurrent_user_creation"] = self._evaluate_latest_metrics("concurrent_user_creation")
        
        # Create sessions concurrently
        async def create_session_batch(batch_id: int):
            sessions = []
            for i in range(5):
                user_id = f"concurrent_user_{batch_id}_{i}"
                session_id = await storage.create_session(user_id, f"Concurrent Session {batch_id}_{i}")
                sessions.append((user_id, session_id))
            return sessions
        
        async with self.profiler.profile_operation("concurrent_session_creation", concurrency * 5):
            session_tasks = [create_session_batch(i) for i in range(concurrency)]
            session_results = await asyncio.gather(*session_tasks)
        
        results["concurrent_session_creation"] = self._evaluate_latest_metrics("concurrent_session_creation")
        
        # Insert messages concurrently
        async def insert_message_batch(user_sessions: List[tuple]):
            for user_id, session_id in user_sessions:
                messages = self.factory.create_message_batch(20)
                for message in messages:
                    await storage.add_message(user_id, session_id, message)
        
        all_sessions = [session for batch in session_results for session in batch]
        async with self.profiler.profile_operation("concurrent_message_insertion", len(all_sessions) * 20):
            # Split sessions into batches for concurrent processing
            batch_size = len(all_sessions) // concurrency
            session_batches = [
                all_sessions[i:i + batch_size] 
                for i in range(0, len(all_sessions), batch_size)
            ]
            
            message_tasks = [insert_message_batch(batch) for batch in session_batches if batch]
            await asyncio.gather(*message_tasks)
        
        results["concurrent_message_insertion"] = self._evaluate_latest_metrics("concurrent_message_insertion")
        
        return results
    
    async def run_memory_stress_tests(self, storage: FFStorageManager) -> Dict[str, Any]:
        """Run memory stress tests to identify memory leaks."""
        results = {}
        
        # Create large number of small operations
        await storage.create_user("memory_user", {"name": "Memory Test User"})
        session_id = await storage.create_session("memory_user", "Memory Test Session")
        
        # Monitor memory usage during many small operations
        memory_samples = []
        
        async with self.profiler.profile_operation("memory_stress_small_ops", 5000):
            for i in range(5000):
                # Create small message
                message = self.factory.create_message(content=f"Small message {i}")
                await storage.add_message("memory_user", session_id, message)
                
                # Sample memory every 100 operations
                if i % 100 == 0:
                    current_memory = self.profiler.process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    await asyncio.sleep(0.001)  # Yield control
        
        results["memory_stress_small_ops"] = self._evaluate_latest_metrics("memory_stress_small_ops")
        results["memory_stress_small_ops"]["memory_samples"] = memory_samples
        results["memory_stress_small_ops"]["memory_growth"] = memory_samples[-1] - memory_samples[0] if memory_samples else 0
        
        # Test with large messages
        large_messages = [self.factory.create_large_message(10) for _ in range(100)]  # 10KB each
        
        async with self.profiler.profile_operation("memory_stress_large_ops", len(large_messages)):
            for message in large_messages:
                await storage.add_message("memory_user", session_id, message)
        
        results["memory_stress_large_ops"] = self._evaluate_latest_metrics("memory_stress_large_ops")
        
        return results
    
    async def run_comprehensive_performance_suite(self, storage: FFStorageManager) -> Dict[str, Any]:
        """Run the complete performance test suite."""
        print("🚀 Starting comprehensive performance test suite...")
        
        suite_results = {
            "started_at": datetime.now().isoformat(),
            "basic_operations": {},
            "bulk_operations": {},
            "concurrent_operations": {},
            "memory_stress": {},
            "summary": {}
        }
        
        try:
            # Run basic operation tests
            print("  📊 Running basic operation tests...")
            suite_results["basic_operations"] = await self.run_basic_operation_tests(storage)
            
            # Run bulk operation tests
            print("  📈 Running bulk operation tests...")
            suite_results["bulk_operations"] = await self.run_bulk_operation_tests(storage)
            
            # Run concurrent operation tests
            print("  🔄 Running concurrent operation tests...")
            suite_results["concurrent_operations"] = await self.run_concurrent_operation_tests(storage, 5)
            
            # Run memory stress tests
            print("  🧠 Running memory stress tests...")
            suite_results["memory_stress"] = await self.run_memory_stress_tests(storage)
            
            # Generate summary
            suite_results["summary"] = self._generate_test_summary()
            suite_results["completed_at"] = datetime.now().isoformat()
            
            print("✅ Performance test suite completed successfully!")
            
        except Exception as e:
            print(f"❌ Performance test suite failed: {e}")
            suite_results["error"] = str(e)
            suite_results["failed_at"] = datetime.now().isoformat()
        
        return suite_results
    
    def _evaluate_latest_metrics(self, operation_name: str) -> Dict[str, Any]:
        """Evaluate latest metrics against benchmarks."""
        latest_metrics = self.profiler.get_latest_metrics()
        if not latest_metrics:
            return {"error": "No metrics available"}
        
        benchmark = self.benchmarks.get(operation_name)
        if benchmark:
            return benchmark.evaluate(latest_metrics)
        else:
            return {
                "operation": operation_name,
                "passed": True,
                "metrics": latest_metrics,
                "note": "No benchmark defined"
            }
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        all_operations = list(set(m.operation for m in self.profiler.metrics_history))
        
        summary = {
            "total_operations_tested": len(all_operations),
            "total_measurements": len(self.profiler.metrics_history),
            "operations_summary": {}
        }
        
        for operation in all_operations:
            summary["operations_summary"][operation] = self.profiler.get_operation_stats(operation)
        
        return summary


# === Convenience Functions ===

async def quick_performance_check(storage: FFStorageManager) -> Dict[str, Any]:
    """Run a quick performance check with basic operations."""
    suite = PerformanceTestSuite()
    return await suite.run_basic_operation_tests(storage)


async def run_performance_benchmark(storage: FFStorageManager, export_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run full performance benchmark and optionally export results."""
    suite = PerformanceTestSuite()
    results = await suite.run_comprehensive_performance_suite(storage)
    
    if export_path:
        # Export detailed metrics
        suite.profiler.export_metrics(export_path / "performance_metrics.json")
        
        # Export summary results
        with open(export_path / "performance_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


@asynccontextmanager
async def performance_monitor(operation_name: str):
    """Simple context manager for monitoring single operations."""
    profiler = PerformanceProfiler()
    async with profiler.profile_operation(operation_name):
        yield profiler
    
    metrics = profiler.get_latest_metrics()
    print(f"⏱️  {operation_name}: {metrics.duration_seconds:.3f}s, {metrics.memory_usage_mb:.1f}MB")


# === Performance Regression Detection ===

class PerformanceRegressionDetector:
    """Detect performance regressions by comparing test runs."""
    
    def __init__(self, baseline_file: Path):
        """Initialize with baseline performance data."""
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline performance data."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def detect_regressions(self, current_results: Dict[str, Any], 
                          threshold_percent: float = 20.0) -> Dict[str, Any]:
        """Detect performance regressions compared to baseline."""
        regressions = {
            "found": False,
            "regressions": [],
            "improvements": [],
            "summary": {}
        }
        
        if not self.baseline_data:
            return {"error": "No baseline data available"}
        
        # Compare operation performance
        for category in ["basic_operations", "bulk_operations"]:
            if category in current_results and category in self.baseline_data:
                category_regressions = self._compare_category_performance(
                    current_results[category],
                    self.baseline_data[category],
                    threshold_percent
                )
                
                if category_regressions["regressions"]:
                    regressions["found"] = True
                    regressions["regressions"].extend(category_regressions["regressions"])
                
                regressions["improvements"].extend(category_regressions["improvements"])
        
        return regressions
    
    def _compare_category_performance(self, current: Dict[str, Any], 
                                    baseline: Dict[str, Any],
                                    threshold: float) -> Dict[str, Any]:
        """Compare performance between current and baseline for a category."""
        regressions = []
        improvements = []
        
        for operation, current_data in current.items():
            if operation in baseline and "metrics" in current_data and "metrics" in baseline[operation]:
                current_metrics = current_data["metrics"]
                baseline_metrics = baseline[operation]["metrics"]
                
                # Compare duration
                duration_change = ((current_metrics["duration_seconds"] - baseline_metrics["duration_seconds"]) 
                                 / baseline_metrics["duration_seconds"]) * 100
                
                if duration_change > threshold:
                    regressions.append({
                        "operation": operation,
                        "metric": "duration",
                        "current": current_metrics["duration_seconds"],
                        "baseline": baseline_metrics["duration_seconds"],
                        "change_percent": duration_change
                    })
                elif duration_change < -threshold:
                    improvements.append({
                        "operation": operation,
                        "metric": "duration",
                        "current": current_metrics["duration_seconds"],
                        "baseline": baseline_metrics["duration_seconds"],
                        "improvement_percent": abs(duration_change)
                    })
        
        return {"regressions": regressions, "improvements": improvements}