"""
Integration testing helpers for the flatfile chat database system.

Provides utilities for end-to-end testing, test environment management,
and complex scenario orchestration for comprehensive integration testing.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import time

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import (
    load_config, FFConfigurationManagerConfigDTO
)
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSessionDTO
from tests.test_factories import TestDataFactory
from tests.mock_utilities import AdvancedMockFactory
from tests.performance_testing import PerformanceProfiler


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    use_temp_directory: bool = True
    cleanup_after_test: bool = True
    enable_performance_monitoring: bool = False
    test_data_seed: Optional[int] = None
    concurrent_operations: int = 1
    stress_test_enabled: bool = False


class IntegrationTestEnvironment:
    """Manages test environments for integration testing."""
    
    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        """Initialize test environment manager."""
        self.config = config or IntegrationTestConfig()
        self.temp_dir = None
        self.storage_manager = None
        self.test_config = None
        self.performance_profiler = None
        self.created_resources = []
        
    @asynccontextmanager
    async def create_test_environment(self) -> AsyncGenerator['IntegrationTestEnvironment', None]:
        """Create and manage a complete test environment."""
        try:
            await self._setup_environment()
            yield self
        finally:
            await self._cleanup_environment()
    
    async def _setup_environment(self):
        """Set up the test environment."""
        # Create temporary directory if needed
        if self.config.use_temp_directory:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="ff_integration_test_"))
        
        # Create test configuration
        base_config = load_config(environment="test")
        self.test_config = FFConfigurationManagerConfigDTO(
            storage=base_config.storage._replace(
                base_path=str(self.temp_dir) if self.temp_dir else base_config.storage.base_path
            ),
            search=base_config.search,
            vector=base_config.vector,
            document=base_config.document,
            streaming=base_config.streaming,
            compression=base_config.compression
        )
        
        # Initialize storage manager
        self.storage_manager = FFStorageManager(self.test_config)
        await self.storage_manager.initialize()
        
        # Set up performance monitoring if enabled
        if self.config.enable_performance_monitoring:
            self.performance_profiler = PerformanceProfiler()
    
    async def _cleanup_environment(self):
        """Clean up the test environment."""
        if self.config.cleanup_after_test:
            # Clean up created resources
            for resource in reversed(self.created_resources):
                try:
                    if hasattr(resource, 'cleanup'):
                        await resource.cleanup()
                except Exception as e:
                    print(f"Warning: Failed to cleanup resource {resource}: {e}")
            
            # Close storage manager
            if self.storage_manager and hasattr(self.storage_manager, 'close'):
                try:
                    await self.storage_manager.close()
                except Exception as e:
                    print(f"Warning: Failed to close storage manager: {e}")
            
            # Remove temporary directory
            if self.temp_dir and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove temp directory {self.temp_dir}: {e}")
    
    def track_resource(self, resource: Any):
        """Track a resource for cleanup."""
        self.created_resources.append(resource)
        return resource


class IntegrationTestScenario:
    """Orchestrates complex integration test scenarios."""
    
    def __init__(self, environment: IntegrationTestEnvironment):
        """Initialize with test environment."""
        self.env = environment
        self.factory = TestDataFactory()
        self.scenario_data = {}
    
    async def run_user_lifecycle_scenario(self, user_count: int = 5) -> Dict[str, Any]:
        """Run complete user lifecycle scenario."""
        scenario_results = {
            "users_created": 0,
            "sessions_created": 0,
            "messages_added": 0,
            "operations_completed": [],
            "errors": []
        }
        
        try:
            # Create users
            users = []
            for i in range(user_count):
                user_id = f"integration_user_{i}"
                user_profile = self.factory.create_user_profile(user_id=user_id)
                
                user_created = await self.env.storage_manager.create_user(
                    user_id, user_profile.to_dict() if hasattr(user_profile, 'to_dict') else {}
                )
                
                if user_created:
                    users.append(user_id)
                    scenario_results["users_created"] += 1
                    scenario_results["operations_completed"].append(f"user_created:{user_id}")
            
            # Create sessions for each user
            sessions = []
            for user_id in users:
                session_title = f"Integration Test Session for {user_id}"
                session_id = await self.env.storage_manager.create_session(user_id, session_title)
                
                if session_id:
                    sessions.append((user_id, session_id))
                    scenario_results["sessions_created"] += 1
                    scenario_results["operations_completed"].append(f"session_created:{session_id}")
            
            # Add messages to each session
            for user_id, session_id in sessions:
                messages = self.factory.create_conversation(turns=3)
                
                for message in messages:
                    message_added = await self.env.storage_manager.add_message(
                        user_id, session_id, message
                    )
                    
                    if message_added:
                        scenario_results["messages_added"] += 1
                        scenario_results["operations_completed"].append(f"message_added:{message.message_id}")
            
            # Store scenario data for validation
            self.scenario_data["user_lifecycle"] = {
                "users": users,
                "sessions": sessions,
                "created_at": time.time()
            }
            
        except Exception as e:
            scenario_results["errors"].append(str(e))
        
        return scenario_results
    
    async def run_concurrent_operations_scenario(self, concurrency: int = 10) -> Dict[str, Any]:
        """Run concurrent operations scenario."""
        scenario_results = {
            "concurrent_operations": concurrency,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time_seconds": 0,
            "operations_per_second": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Create concurrent user creation tasks
            async def create_user_session_messages(batch_id: int):
                user_id = f"concurrent_user_{batch_id}"
                
                try:
                    # Create user
                    user_created = await self.env.storage_manager.create_user(user_id)
                    if not user_created:
                        return False
                    
                    # Create session
                    session_id = await self.env.storage_manager.create_session(
                        user_id, f"Concurrent Session {batch_id}"
                    )
                    if not session_id:
                        return False
                    
                    # Add messages
                    messages = self.factory.create_message_batch(5)
                    for message in messages:
                        message_added = await self.env.storage_manager.add_message(
                            user_id, session_id, message
                        )
                        if not message_added:
                            return False
                    
                    return True
                    
                except Exception as e:
                    scenario_results["errors"].append(f"Batch {batch_id}: {str(e)}")
                    return False
            
            # Execute concurrent operations
            tasks = [create_user_session_messages(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    scenario_results["failed_operations"] += 1
                    scenario_results["errors"].append(str(result))
                elif result:
                    scenario_results["successful_operations"] += 1
                else:
                    scenario_results["failed_operations"] += 1
            
            end_time = time.time()
            scenario_results["total_time_seconds"] = end_time - start_time
            
            if scenario_results["total_time_seconds"] > 0:
                scenario_results["operations_per_second"] = (
                    scenario_results["successful_operations"] / scenario_results["total_time_seconds"]
                )
            
        except Exception as e:
            scenario_results["errors"].append(f"Concurrent scenario failed: {str(e)}")
        
        return scenario_results
    
    async def run_data_persistence_scenario(self) -> Dict[str, Any]:
        """Run data persistence validation scenario."""
        scenario_results = {
            "data_written": 0,
            "data_retrieved": 0,
            "data_matches": 0,
            "persistence_validated": False,
            "errors": []
        }
        
        try:
            # Create test data
            user_id = "persistence_test_user"
            await self.env.storage_manager.create_user(user_id)
            
            session_id = await self.env.storage_manager.create_session(user_id, "Persistence Test")
            original_messages = self.factory.create_message_batch(10)
            
            # Write data
            for message in original_messages:
                message_added = await self.env.storage_manager.add_message(
                    user_id, session_id, message
                )
                if message_added:
                    scenario_results["data_written"] += 1
            
            # Retrieve data
            retrieved_messages = await self.env.storage_manager.get_all_messages(user_id, session_id)
            scenario_results["data_retrieved"] = len(retrieved_messages)
            
            # Validate data consistency
            if len(retrieved_messages) == len(original_messages):
                for orig, retr in zip(original_messages, retrieved_messages):
                    if orig.content == retr.content and orig.role == retr.role:
                        scenario_results["data_matches"] += 1
            
            # Check persistence validation
            scenario_results["persistence_validated"] = (
                scenario_results["data_matches"] == len(original_messages)
            )
            
        except Exception as e:
            scenario_results["errors"].append(str(e))
        
        return scenario_results
    
    async def run_stress_test_scenario(self, operation_count: int = 1000) -> Dict[str, Any]:
        """Run stress test scenario with high volume operations."""
        scenario_results = {
            "target_operations": operation_count,
            "completed_operations": 0,
            "failed_operations": 0,
            "average_response_time": 0,
            "memory_usage_mb": 0,
            "errors": []
        }
        
        if not self.env.config.stress_test_enabled:
            scenario_results["errors"].append("Stress testing disabled in configuration")
            return scenario_results
        
        response_times = []
        start_time = time.time()
        
        try:
            # Create base user and session
            user_id = "stress_test_user"
            await self.env.storage_manager.create_user(user_id)
            session_id = await self.env.storage_manager.create_session(user_id, "Stress Test Session")
            
            # Run high-volume operations
            for i in range(operation_count):
                operation_start = time.time()
                
                try:
                    message = self.factory.create_message(
                        content=f"Stress test message {i}",
                        message_id=f"stress_msg_{i}"
                    )
                    
                    success = await self.env.storage_manager.add_message(user_id, session_id, message)
                    
                    if success:
                        scenario_results["completed_operations"] += 1
                    else:
                        scenario_results["failed_operations"] += 1
                    
                    operation_time = time.time() - operation_start
                    response_times.append(operation_time)
                    
                    # Yield control periodically
                    if i % 100 == 0:
                        await asyncio.sleep(0.001)
                    
                except Exception as e:
                    scenario_results["failed_operations"] += 1
                    scenario_results["errors"].append(f"Operation {i}: {str(e)}")
            
            # Calculate metrics
            if response_times:
                scenario_results["average_response_time"] = sum(response_times) / len(response_times)
            
            # Get memory usage (if performance profiler available)
            if self.env.performance_profiler:
                scenario_results["memory_usage_mb"] = (
                    self.env.performance_profiler.process.memory_info().rss / 1024 / 1024
                )
            
        except Exception as e:
            scenario_results["errors"].append(f"Stress test failed: {str(e)}")
        
        return scenario_results


class IntegrationTestSuite:
    """Complete integration test suite orchestrator."""
    
    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        """Initialize test suite."""
        self.config = config or IntegrationTestConfig()
        self.results = {}
    
    async def run_complete_integration_suite(self) -> Dict[str, Any]:
        """Run the complete integration test suite."""
        suite_results = {
            "started_at": time.time(),
            "scenarios": {},
            "overall_success": False,
            "summary": {}
        }
        
        async with IntegrationTestEnvironment(self.config).create_test_environment() as env:
            scenario_runner = IntegrationTestScenario(env)
            
            print("🚀 Starting comprehensive integration test suite...")
            
            # Run user lifecycle scenario
            print("  👤 Running user lifecycle scenario...")
            suite_results["scenarios"]["user_lifecycle"] = await scenario_runner.run_user_lifecycle_scenario()
            
            # Run concurrent operations scenario
            if self.config.concurrent_operations > 1:
                print(f"  🔄 Running concurrent operations scenario ({self.config.concurrent_operations} concurrent)...")
                suite_results["scenarios"]["concurrent_operations"] = await scenario_runner.run_concurrent_operations_scenario(
                    self.config.concurrent_operations
                )
            
            # Run data persistence scenario
            print("  💾 Running data persistence scenario...")
            suite_results["scenarios"]["data_persistence"] = await scenario_runner.run_data_persistence_scenario()
            
            # Run stress test scenario (if enabled)
            if self.config.stress_test_enabled:
                print("  ⚡ Running stress test scenario...")
                suite_results["scenarios"]["stress_test"] = await scenario_runner.run_stress_test_scenario()
            
            # Generate summary
            suite_results["summary"] = self._generate_suite_summary(suite_results["scenarios"])
            suite_results["overall_success"] = suite_results["summary"]["overall_success"]
            suite_results["completed_at"] = time.time()
            
            print(f"✅ Integration test suite completed in {suite_results['completed_at'] - suite_results['started_at']:.2f}s")
        
        return suite_results
    
    def _generate_suite_summary(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive suite summary."""
        summary = {
            "total_scenarios": len(scenarios),
            "successful_scenarios": 0,
            "failed_scenarios": 0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_errors": 0,
            "overall_success": True,
            "scenario_details": {}
        }
        
        for scenario_name, scenario_results in scenarios.items():
            scenario_success = len(scenario_results.get("errors", [])) == 0
            
            if scenario_success:
                summary["successful_scenarios"] += 1
            else:
                summary["failed_scenarios"] += 1
                summary["overall_success"] = False
            
            # Aggregate operation counts
            if "completed_operations" in scenario_results:
                summary["successful_operations"] += scenario_results["completed_operations"]
            if "failed_operations" in scenario_results:
                summary["failed_operations"] += scenario_results["failed_operations"]
            
            # Count errors
            error_count = len(scenario_results.get("errors", []))
            summary["total_errors"] += error_count
            
            # Store scenario details
            summary["scenario_details"][scenario_name] = {
                "success": scenario_success,
                "error_count": error_count,
                "key_metrics": self._extract_key_metrics(scenario_results)
            }
        
        summary["total_operations"] = summary["successful_operations"] + summary["failed_operations"]
        
        return summary
    
    def _extract_key_metrics(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from scenario results."""
        metrics = {}
        
        # Common metrics
        if "completed_operations" in scenario_results:
            metrics["completed_operations"] = scenario_results["completed_operations"]
        if "total_time_seconds" in scenario_results:
            metrics["duration_seconds"] = scenario_results["total_time_seconds"]
        if "operations_per_second" in scenario_results:
            metrics["throughput"] = scenario_results["operations_per_second"]
        
        # Specific metrics
        if "data_matches" in scenario_results:
            metrics["data_integrity"] = scenario_results["data_matches"]
        if "average_response_time" in scenario_results:
            metrics["avg_response_time"] = scenario_results["average_response_time"]
        
        return metrics


# === Convenience Functions ===

async def run_quick_integration_test() -> Dict[str, Any]:
    """Run a quick integration test with default settings."""
    config = IntegrationTestConfig(
        concurrent_operations=3,
        stress_test_enabled=False
    )
    
    suite = IntegrationTestSuite(config)
    return await suite.run_complete_integration_suite()


async def run_comprehensive_integration_test() -> Dict[str, Any]:
    """Run comprehensive integration test with all scenarios."""
    config = IntegrationTestConfig(
        concurrent_operations=10,
        stress_test_enabled=True,
        enable_performance_monitoring=True
    )
    
    suite = IntegrationTestSuite(config)
    return await suite.run_complete_integration_suite()


@asynccontextmanager
async def integration_test_environment(**kwargs) -> AsyncGenerator[IntegrationTestEnvironment, None]:
    """Convenient context manager for creating test environments."""
    config = IntegrationTestConfig(**kwargs)
    async with IntegrationTestEnvironment(config).create_test_environment() as env:
        yield env


# === Example Usage ===

if __name__ == "__main__":
    async def main():
        print("Running integration test suite...")
        results = await run_quick_integration_test()
        
        print(f"\nSuite Results:")
        print(f"Overall Success: {results['summary']['overall_success']}")
        print(f"Scenarios: {results['summary']['successful_scenarios']}/{results['summary']['total_scenarios']} successful")
        print(f"Operations: {results['summary']['successful_operations']}/{results['summary']['total_operations']} successful")
        
        if results['summary']['total_errors'] > 0:
            print(f"Errors: {results['summary']['total_errors']} total errors encountered")
        
        # Print scenario details
        for scenario_name, details in results['summary']['scenario_details'].items():
            status = "✅" if details['success'] else "❌"
            print(f"{status} {scenario_name}: {details['key_metrics']}")
    
    asyncio.run(main())