# FF Chat API Performance and Load Tests
"""
Performance and load tests for FF Chat API.
Tests system behavior under various load conditions.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.slow, pytest.mark.asyncio]

class TestFFChatAPIPerformance:
    """Test FF Chat API performance characteristics"""
    
    async def test_single_request_response_time(self, ff_chat_api, api_test_client, api_helper, performance_config):
        """Test single request response time"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Measure health check response time
        start_time = time.time()
        response = client.get("/health", headers=headers)
        end_time = time.time()
        
        response_time = end_time - start_time
        max_response_time = performance_config["max_response_time"]
        
        assert response_time < max_response_time, f"Response time {response_time:.2f}s exceeds limit {max_response_time}s"
        print(f"Health check response time: {response_time:.3f}s")
    
    async def test_chat_message_response_time(self, ff_chat_api, api_test_client, api_helper, performance_config):
        """Test chat message processing response time"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Measure message processing time
        start_time = time.time()
        response = client.post(
            f"/api/v1/chat/{session_id}/message",
            json={"message": "What is the capital of France?"},
            headers=headers
        )
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            max_response_time = performance_config["max_response_time"]
            
            assert response_time < max_response_time, f"Chat response time {response_time:.2f}s exceeds limit {max_response_time}s"
            print(f"Chat message response time: {response_time:.3f}s")
    
    async def test_session_creation_performance(self, ff_chat_api, api_test_client, api_helper):
        """Test session creation performance"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create multiple sessions and measure time
        num_sessions = 10
        start_time = time.time()
        
        for i in range(num_sessions):
            response = client.post(
                "/api/v1/sessions",
                json={"use_case": "basic_chat", "title": f"Performance test session {i}"},
                headers=headers
            )
            if hasattr(response, 'status_code'):
                assert response.status_code == 201
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_session = total_time / num_sessions
        
        assert avg_time_per_session < 1.0, f"Average session creation time {avg_time_per_session:.2f}s too high"
        print(f"Average session creation time: {avg_time_per_session:.3f}s")
    
    async def test_large_message_handling(self, ff_chat_api, api_test_client, api_helper):
        """Test performance with large messages"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test with increasingly large messages
        message_sizes = [1000, 5000, 10000, 50000]  # Character counts
        
        for size in message_sizes:
            large_message = "A" * size
            
            start_time = time.time()
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": large_message},
                headers=headers
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Allow more time for larger messages, but should still be reasonable
            max_time = 5.0 + (size / 10000)  # Base time + scaling factor
            
            if response.status_code == 200:
                assert response_time < max_time, f"Large message ({size} chars) took {response_time:.2f}s (limit: {max_time:.2f}s)"
                print(f"Message size {size} chars: {response_time:.3f}s")
            elif response.status_code == 413:  # Request Entity Too Large
                print(f"Message size {size} chars rejected (too large)")
                break

class TestFFChatAPIConcurrency:
    """Test FF Chat API concurrent request handling"""
    
    def _make_concurrent_requests(self, client, endpoint, headers, num_requests):
        """Helper to make concurrent requests"""
        results = []
        
        def make_request():
            try:
                start_time = time.time()
                response = client.get(endpoint, headers=headers)
                end_time = time.time()
                return {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                }
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        return results
    
    async def test_concurrent_health_checks(self, ff_chat_api, api_test_client, api_helper, performance_config):
        """Test concurrent health check requests"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        num_concurrent = performance_config["concurrent_users"]
        
        start_time = time.time()
        results = self._make_concurrent_requests(client, "/health", headers, num_concurrent)
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / len(results) * 100
        
        assert success_rate >= 90, f"Success rate {success_rate:.1f}% too low for concurrent requests"
        assert total_time < 10.0, f"Concurrent requests took {total_time:.2f}s (too long)"
        
        print(f"Concurrent health checks: {successful_requests}/{num_concurrent} successful in {total_time:.2f}s")
    
    async def test_concurrent_session_creation(self, ff_chat_api, api_test_client, api_helper):
        """Test concurrent session creation"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        num_concurrent = 10
        
        def create_session():
            try:
                start_time = time.time()
                response = client.post(
                    "/api/v1/sessions",
                    json={"use_case": "basic_chat", "title": "Concurrent test session"},
                    headers=headers
                )
                end_time = time.time()
                return {
                    "status_code": getattr(response, 'status_code', 0),
                    "response_time": end_time - start_time,
                    "success": getattr(response, 'status_code', 0) == 201
                }
            except Exception as e:
                return {"status_code": 0, "success": False, "error": str(e)}
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(create_session) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_creates = sum(1 for r in results if r["success"])
        
        success_rate = successful_creates / len(results) * 100
        assert success_rate >= 80, f"Concurrent session creation success rate {success_rate:.1f}% too low"
        
        print(f"Concurrent session creation: {successful_creates}/{num_concurrent} successful in {total_time:.2f}s")
    
    async def test_concurrent_chat_messages(self, ff_chat_api, api_test_client, api_helper):
        """Test concurrent chat message sending"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create a session first
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        num_concurrent = 10
        
        def send_message(msg_id):
            try:
                start_time = time.time()
                response = client.post(
                    f"/api/v1/chat/{session_id}/message",
                    json={"message": f"Concurrent test message {msg_id}"},
                    headers=headers
                )
                end_time = time.time()
                return {
                    "status_code": getattr(response, 'status_code', 0),
                    "response_time": end_time - start_time,
                    "success": getattr(response, 'status_code', 0) == 200,
                    "message_id": msg_id
                }
            except Exception as e:
                return {"status_code": 0, "success": False, "error": str(e), "message_id": msg_id}
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(send_message, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_messages = sum(1 for r in results if r["success"])
        
        success_rate = successful_messages / len(results) * 100
        assert success_rate >= 70, f"Concurrent message sending success rate {success_rate:.1f}% too low"
        
        print(f"Concurrent messages: {successful_messages}/{num_concurrent} successful in {total_time:.2f}s")

class TestFFChatAPILoadTesting:
    """Test FF Chat API under various load conditions"""
    
    async def test_sustained_load(self, ff_chat_api, api_test_client, api_helper, load_test_scenarios):
        """Test API under sustained load"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        scenario = load_test_scenarios["basic_load"]
        concurrent_users = scenario["concurrent_users"]
        test_duration = min(scenario["test_duration"], 10)  # Limit test duration
        
        def sustained_requests():
            """Make requests for the test duration"""
            end_time = time.time() + test_duration
            request_count = 0
            successful_requests = 0
            
            while time.time() < end_time:
                try:
                    response = client.get("/health", headers=headers)
                    request_count += 1
                    if getattr(response, 'status_code', 0) == 200:
                        successful_requests += 1
                    time.sleep(0.1)  # Small delay between requests
                except Exception:
                    request_count += 1
                    pass
            
            return {"total": request_count, "successful": successful_requests}
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(sustained_requests) for _ in range(concurrent_users)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_requests = sum(r["total"] for r in results)
        total_successful = sum(r["successful"] for r in results)
        
        success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        actual_duration = end_time - start_time
        
        assert success_rate >= 85, f"Sustained load success rate {success_rate:.1f}% too low"
        
        print(f"Sustained load test: {total_successful}/{total_requests} requests successful over {actual_duration:.1f}s")
    
    async def test_spike_load(self, ff_chat_api, api_test_client, api_helper, load_test_scenarios):
        """Test API response to sudden load spikes"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        scenario = load_test_scenarios["spike_test"]
        spike_users = min(scenario["concurrent_users"], 20)  # Limit for testing
        
        def spike_request():
            try:
                start_time = time.time()
                response = client.get("/health", headers=headers)
                end_time = time.time()
                return {
                    "success": getattr(response, 'status_code', 0) == 200,
                    "response_time": end_time - start_time
                }
            except Exception:
                return {"success": False, "response_time": 0}
        
        # Create sudden spike
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=spike_users) as executor:
            futures = [executor.submit(spike_request) for _ in range(spike_users)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / len(results) * 100
        
        # System should handle spike gracefully (may have some failures)
        assert success_rate >= 60, f"Spike load success rate {success_rate:.1f}% too low"
        assert total_time < 30.0, f"Spike load took {total_time:.2f}s (too long)"
        
        print(f"Spike load test: {successful_requests}/{spike_users} requests successful in {total_time:.2f}s")

class TestFFChatAPIMemoryUsage:
    """Test FF Chat API memory usage patterns"""
    
    async def test_memory_usage_with_many_sessions(self, ff_chat_api, api_test_client, api_helper):
        """Test memory usage when creating many sessions"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create many sessions to test memory usage
        num_sessions = 50
        created_sessions = []
        
        for i in range(num_sessions):
            try:
                session_id = api_helper.create_test_session(client, headers, "basic_chat")
                if session_id:
                    created_sessions.append(session_id)
            except Exception:
                pass  # Some may fail, that's acceptable
        
        # Verify we created a reasonable number of sessions
        success_rate = len(created_sessions) / num_sessions * 100
        assert success_rate >= 70, f"Session creation success rate {success_rate:.1f}% too low"
        
        print(f"Created {len(created_sessions)} sessions successfully")
        
        # Test that we can still interact with the API
        response = client.get("/health", headers=headers)
        assert getattr(response, 'status_code', 500) == 200, "API should still be responsive after creating many sessions"
    
    async def test_memory_usage_with_long_conversations(self, ff_chat_api, api_test_client, api_helper):
        """Test memory usage with long conversations"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "memory_chat")
        
        # Send many messages to create long conversation
        num_messages = 20
        successful_messages = 0
        
        for i in range(num_messages):
            try:
                response = client.post(
                    f"/api/v1/chat/{session_id}/message",
                    json={"message": f"This is message number {i} in a long conversation test"},
                    headers=headers
                )
                if getattr(response, 'status_code', 500) == 200:
                    successful_messages += 1
            except Exception:
                pass
        
        success_rate = successful_messages / num_messages * 100
        assert success_rate >= 70, f"Long conversation success rate {success_rate:.1f}% too low"
        
        print(f"Long conversation test: {successful_messages}/{num_messages} messages successful")
        
        # Verify API is still responsive
        response = client.get("/health", headers=headers)
        assert getattr(response, 'status_code', 500) == 200, "API should remain responsive after long conversation"

class TestFFChatAPIStressConditions:
    """Test FF Chat API under stress conditions"""
    
    async def test_rapid_session_creation_deletion(self, ff_chat_api, api_test_client, api_helper):
        """Test rapid session creation and deletion"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Rapidly create and delete sessions
        num_cycles = 10
        successful_cycles = 0
        
        for i in range(num_cycles):
            try:
                # Create session
                session_id = api_helper.create_test_session(client, headers, "basic_chat")
                
                if session_id:
                    # Immediately delete it
                    delete_response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
                    
                    if getattr(delete_response, 'status_code', 500) in [200, 204]:
                        successful_cycles += 1
            except Exception:
                pass
        
        success_rate = successful_cycles / num_cycles * 100
        assert success_rate >= 60, f"Rapid create/delete success rate {success_rate:.1f}% too low"
        
        print(f"Rapid create/delete test: {successful_cycles}/{num_cycles} cycles successful")
    
    async def test_mixed_operation_stress(self, ff_chat_api, api_test_client, api_helper):
        """Test mixed operations under stress"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create base session
        base_session = api_helper.create_test_session(client, headers, "basic_chat")
        
        operations = [
            lambda: client.get("/health", headers=headers),
            lambda: client.get("/api/v1/sessions", headers=headers),
            lambda: client.post(f"/api/v1/chat/{base_session}/message", 
                               json={"message": "Stress test message"}, headers=headers),
            lambda: client.get(f"/api/v1/chat/{base_session}/history", headers=headers)
        ]
        
        def mixed_operations():
            import random
            successful_ops = 0
            total_ops = 10
            
            for _ in range(total_ops):
                try:
                    operation = random.choice(operations)
                    response = operation()
                    if getattr(response, 'status_code', 500) < 400:
                        successful_ops += 1
                except Exception:
                    pass
                
                time.sleep(0.05)  # Small delay
            
            return {"successful": successful_ops, "total": total_ops}
        
        # Run mixed operations concurrently
        num_workers = 5
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]
        
        total_ops = sum(r["total"] for r in results)
        successful_ops = sum(r["successful"] for r in results)
        
        success_rate = (successful_ops / total_ops * 100) if total_ops > 0 else 0
        assert success_rate >= 70, f"Mixed operations success rate {success_rate:.1f}% too low"
        
        print(f"Mixed operations stress test: {successful_ops}/{total_ops} operations successful")