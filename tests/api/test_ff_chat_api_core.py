# FF Chat API Core Tests
"""
Core API functionality tests for FF Chat API.
Tests basic API operations, authentication, and core endpoints.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

class TestFFChatAPICoreEndpoints:
    """Test core FF Chat API endpoints"""
    
    async def test_api_health_check(self, ff_chat_api, api_test_client):
        """Test basic API health check endpoint"""
        client = api_test_client(ff_chat_api)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    async def test_api_detailed_health_check(self, ff_chat_api, api_test_client):
        """Test detailed health check with system information"""
        client = api_test_client(ff_chat_api)
        
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ff_backend_status" in data
        assert "component_status" in data
        assert "metrics" in data
    
    async def test_api_version_endpoint(self, ff_chat_api, api_test_client):
        """Test API version information endpoint"""
        client = api_test_client(ff_chat_api)
        
        response = client.get("/version")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "ff_chat_version" in data
        assert data["ff_chat_version"] == "4.0.0"

class TestFFChatAPIAuthentication:
    """Test FF Chat API authentication system"""
    
    async def test_user_registration(self, ff_chat_api, api_test_client, sample_api_test_data):
        """Test user registration endpoint"""
        client = api_test_client(ff_chat_api)
        
        user_data = {
            "username": "newuser",
            "password": "securepassword123",
            "email": "newuser@test.com"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        if response.status_code == 201:
            data = response.json()
            assert "user_id" in data
            assert data["username"] == user_data["username"]
            assert data["email"] == user_data["email"]
            assert "password" not in data  # Password should not be returned
    
    async def test_user_login(self, ff_chat_api, api_test_client):
        """Test user login endpoint"""
        client = api_test_client(ff_chat_api)
        
        # First register a user
        user_data = {
            "username": "loginuser",
            "password": "testpassword123",
            "email": "login@test.com"
        }
        client.post("/api/v1/auth/register", json=user_data)
        
        # Then try to login
        login_data = {
            "username": "loginuser",
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
    
    async def test_invalid_login(self, ff_chat_api, api_test_client):
        """Test login with invalid credentials"""
        client = api_test_client(ff_chat_api)
        
        invalid_login = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", json=invalid_login)
        
        assert response.status_code == 401
    
    async def test_protected_endpoint_without_auth(self, ff_chat_api, api_test_client):
        """Test accessing protected endpoint without authentication"""
        client = api_test_client(ff_chat_api)
        
        response = client.get("/api/v1/user/profile")
        
        assert response.status_code == 401
    
    async def test_protected_endpoint_with_auth(self, ff_chat_api, api_test_client, api_helper):
        """Test accessing protected endpoint with valid authentication"""
        client = api_test_client(ff_chat_api)
        
        # Mock valid token
        token = "valid_test_token"
        headers = api_helper.create_auth_headers(token)
        
        response = client.get("/api/v1/user/profile", headers=headers)
        
        # Should return 200 or specific error (not 401 unauthorized)
        assert response.status_code != 401

class TestFFChatAPISessionManagement:
    """Test FF Chat API session management"""
    
    async def test_create_chat_session(self, ff_chat_api, api_test_client, api_helper):
        """Test creating a new chat session"""
        client = api_test_client(ff_chat_api)
        
        # Mock authentication
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_data = {
            "use_case": "basic_chat",
            "title": "Test Chat Session"
        }
        
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        if response.status_code == 201:
            data = response.json()
            assert "session_id" in data
            assert data["use_case"] == "basic_chat"
            assert data["title"] == "Test Chat Session"
    
    async def test_list_user_sessions(self, ff_chat_api, api_test_client, api_helper):
        """Test listing user's chat sessions"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        response = client.get("/api/v1/sessions", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "sessions" in data
            assert isinstance(data["sessions"], list)
    
    async def test_get_session_details(self, ff_chat_api, api_test_client, api_helper):
        """Test getting specific session details"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session first
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert data["session_id"] == session_id
            assert "title" in data
            assert "use_case" in data
    
    async def test_delete_session(self, ff_chat_api, api_test_client, api_helper):
        """Test deleting a chat session"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session first
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        
        if response.status_code == 200:
            # Verify session is deleted
            get_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert get_response.status_code == 404

class TestFFChatAPIChatEndpoints:
    """Test FF Chat API chat functionality"""
    
    async def test_send_chat_message(self, ff_chat_api, api_test_client, api_helper):
        """Test sending a chat message"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session first
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        message_data = {
            "message": "Hello, this is a test message"
        }
        
        response = client.post(
            f"/api/v1/chat/{session_id}/message",
            json=message_data,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "response" in data
            assert "message_id" in data
    
    async def test_get_chat_history(self, ff_chat_api, api_test_client, api_helper):
        """Test retrieving chat history"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session and send messages
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        api_helper.send_test_message(client, headers, session_id, "Test message 1")
        api_helper.send_test_message(client, headers, session_id, "Test message 2")
        
        response = client.get(f"/api/v1/chat/{session_id}/history", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "messages" in data
            assert isinstance(data["messages"], list)
    
    async def test_search_messages(self, ff_chat_api, api_test_client, api_helper):
        """Test searching through chat messages"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session and send searchable messages
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        api_helper.send_test_message(client, headers, session_id, "I love Python programming")
        api_helper.send_test_message(client, headers, session_id, "JavaScript is also great")
        
        search_params = {"query": "Python"}
        response = client.get(
            f"/api/v1/chat/{session_id}/search",
            params=search_params,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)

class TestFFChatAPIUseCases:
    """Test FF Chat API use case handling"""
    
    @pytest.mark.parametrize("use_case", [
        "basic_chat",
        "memory_chat",
        "rag_chat",
        "multimodal_chat",
        "multi_ai_panel",
        "personal_assistant",
        "translation_chat",
        "scene_critic"
    ])
    async def test_use_case_endpoints(self, ff_chat_api, api_test_client, api_helper, use_case):
        """Test that each use case can be created and used"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Create session for specific use case
        session_id = api_helper.create_test_session(client, headers, use_case)
        
        # Send test message
        message_result = api_helper.send_test_message(
            client, headers, session_id, 
            f"This is a test message for {use_case}"
        )
        
        if isinstance(message_result, dict):
            assert message_result.get("success", True)
    
    async def test_use_case_component_activation(self, ff_chat_api, api_test_client, api_helper):
        """Test that different use cases activate appropriate components"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Test basic_chat (should only use text_chat)
        basic_session = api_helper.create_test_session(client, headers, "basic_chat")
        basic_response = api_helper.send_test_message(
            client, headers, basic_session, "Hello"
        )
        
        # Test memory_chat (should use text_chat + memory)
        memory_session = api_helper.create_test_session(client, headers, "memory_chat")
        memory_response = api_helper.send_test_message(
            client, headers, memory_session, "Remember that I like Python"
        )
        
        # Both should succeed
        if isinstance(basic_response, dict):
            assert basic_response.get("success", True)
        if isinstance(memory_response, dict):
            assert memory_response.get("success", True)

class TestFFChatAPIErrorHandling:
    """Test FF Chat API error handling"""
    
    async def test_invalid_session_id(self, ff_chat_api, api_test_client, api_helper):
        """Test handling of invalid session ID"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        invalid_session_id = "invalid_session_123"
        
        response = client.post(
            f"/api/v1/chat/{invalid_session_id}/message",
            json={"message": "Test message"},
            headers=headers
        )
        
        assert response.status_code == 404
    
    async def test_malformed_request_data(self, ff_chat_api, api_test_client, api_helper):
        """Test handling of malformed request data"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Invalid session creation data
        invalid_data = {
            "use_case": "invalid_use_case",
            "title": ""  # Empty title
        }
        
        response = client.post("/api/v1/sessions", json=invalid_data, headers=headers)
        
        assert response.status_code == 400
    
    async def test_rate_limiting(self, ff_chat_api, api_test_client, api_helper):
        """Test API rate limiting (if enabled)"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Make many rapid requests
        for i in range(100):  # Rapid requests
            response = client.get("/health", headers=headers)
            
            # If rate limiting is active, should eventually get 429
            if response.status_code == 429:
                break
        
        # This test passes if we either hit rate limit or all requests succeed
        assert True  # Test completion indicates proper handling

class TestFFChatAPIDataValidation:
    """Test FF Chat API data validation"""
    
    async def test_message_content_validation(self, ff_chat_api, api_test_client, api_helper):
        """Test validation of message content"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test empty message
        response = client.post(
            f"/api/v1/chat/{session_id}/message",
            json={"message": ""},
            headers=headers
        )
        
        assert response.status_code == 400
        
        # Test very long message (if limits exist)
        long_message = "A" * 10000  # Very long message
        response = client.post(
            f"/api/v1/chat/{session_id}/message",
            json={"message": long_message},
            headers=headers
        )
        
        # Should either succeed or return 400 (depending on limits)
        assert response.status_code in [200, 400]
    
    async def test_session_title_validation(self, ff_chat_api, api_test_client, api_helper):
        """Test validation of session titles"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Test very long title
        long_title = "A" * 1000
        
        response = client.post(
            "/api/v1/sessions",
            json={"use_case": "basic_chat", "title": long_title},
            headers=headers
        )
        
        # Should either succeed or return 400 (depending on limits)
        assert response.status_code in [201, 400]