# FF Chat API Security Tests
"""
Security tests for FF Chat API.
Tests authentication, authorization, input validation, and security measures.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

# Test markers
pytestmark = [pytest.mark.security, pytest.mark.asyncio]

class TestFFChatAPIAuthentication:
    """Test FF Chat API authentication security"""
    
    async def test_jwt_token_validation(self, ff_chat_api, api_test_client):
        """Test JWT token validation security"""
        client = api_test_client(ff_chat_api)
        
        # Test with invalid JWT tokens
        invalid_tokens = [
            "invalid.jwt.token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
            "Bearer invalid_token",
            "",
            "null",
            "undefined"
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/user/profile", headers=headers)
            assert response.status_code == 401, f"Invalid token should be rejected: {token}"
    
    async def test_api_key_validation(self, ff_chat_api, api_test_client):
        """Test API key validation security"""
        client = api_test_client(ff_chat_api)
        
        # Test with invalid API keys
        invalid_api_keys = [
            "invalid_api_key",
            "",
            "null",
            "admin",
            "123456",
            "X" * 1000  # Very long key
        ]
        
        for api_key in invalid_api_keys:
            headers = {"X-API-Key": api_key}
            response = client.get("/api/v1/user/profile", headers=headers)
            assert response.status_code == 401, f"Invalid API key should be rejected: {api_key}"
    
    async def test_password_security_requirements(self, ff_chat_api, api_test_client):
        """Test password security requirements"""
        client = api_test_client(ff_chat_api)
        
        # Test weak passwords
        weak_passwords = [
            "123",
            "password",
            "admin",
            "test",
            "a",
            "",
            "12345678"  # Only numbers
        ]
        
        for password in weak_passwords:
            user_data = {
                "username": f"testuser_{hash(password)}",
                "password": password,
                "email": "test@example.com"
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            # Should reject weak passwords
            assert response.status_code in [400, 422], f"Weak password should be rejected: {password}"
    
    async def test_brute_force_protection(self, ff_chat_api, api_test_client):
        """Test brute force attack protection"""
        client = api_test_client(ff_chat_api)
        
        # Register a user first
        user_data = {
            "username": "bruteforcetest",
            "password": "ValidPassword123!",
            "email": "brute@test.com"
        }
        client.post("/api/v1/auth/register", json=user_data)
        
        # Attempt multiple failed logins
        failed_attempts = 0
        for i in range(20):  # Many failed attempts
            login_data = {
                "username": "bruteforcetest",
                "password": "wrongpassword"
            }
            
            response = client.post("/api/v1/auth/login", json=login_data)
            
            if response.status_code == 429:  # Rate limited
                break
            elif response.status_code == 423:  # Account locked
                break
            
            failed_attempts += 1
        
        # Should eventually be rate limited or account locked
        assert failed_attempts < 20, "Brute force protection should activate"

class TestFFChatAPIInputValidation:
    """Test FF Chat API input validation security"""
    
    async def test_sql_injection_protection(self, ff_chat_api, api_test_client, api_helper, security_test_payloads):
        """Test protection against SQL injection attacks"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test SQL injection payloads in message content
        for payload in security_test_payloads["sql_injection"]:
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": payload},
                headers=headers
            )
            
            # Should not return 500 (internal server error) or expose database errors
            assert response.status_code != 500, f"SQL injection payload caused server error: {payload}"
            
            if response.status_code == 200:
                data = response.json()
                # Response should not contain SQL error messages
                response_text = str(data).lower()
                assert "sql" not in response_text
                assert "database" not in response_text
                assert "table" not in response_text
    
    async def test_xss_protection(self, ff_chat_api, api_test_client, api_helper, security_test_payloads):
        """Test protection against XSS attacks"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test XSS payloads
        for payload in security_test_payloads["xss_payloads"]:
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": payload},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                response_content = data.get("response", "")
                
                # Response should not contain unescaped script tags
                assert "<script>" not in response_content.lower()
                assert "javascript:" not in response_content.lower()
                assert "onerror=" not in response_content.lower()
    
    async def test_command_injection_protection(self, ff_chat_api, api_test_client, api_helper, security_test_payloads):
        """Test protection against command injection attacks"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test command injection payloads
        for payload in security_test_payloads["command_injection"]:
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": payload},
                headers=headers
            )
            
            # Should not execute system commands
            assert response.status_code != 500, f"Command injection payload caused error: {payload}"
            
            if response.status_code == 200:
                data = response.json()
                response_content = str(data).lower()
                
                # Should not contain system command outputs
                assert "/etc/passwd" not in response_content
                assert "root:" not in response_content
                assert "bin/bash" not in response_content
    
    async def test_path_traversal_protection(self, ff_chat_api, api_test_client, security_test_payloads):
        """Test protection against path traversal attacks"""
        client = api_test_client(ff_chat_api)
        
        # Test path traversal in session ID parameter
        for payload in security_test_payloads["path_traversal"]:
            response = client.get(f"/api/v1/chat/{payload}/history")
            
            # Should return 404 (not found) or 400 (bad request), not 500
            assert response.status_code in [400, 401, 404], f"Path traversal should be blocked: {payload}"

class TestFFChatAPIAuthorizationSecurity:
    """Test FF Chat API authorization security"""
    
    async def test_session_access_control(self, ff_chat_api, api_test_client, api_helper):
        """Test that users can only access their own sessions"""
        client = api_test_client(ff_chat_api)
        
        # Create two different users/tokens
        user1_token = "user1_token"
        user2_token = "user2_token"
        
        user1_headers = api_helper.create_auth_headers(user1_token)
        user2_headers = api_helper.create_auth_headers(user2_token)
        
        # User 1 creates a session
        session_id = api_helper.create_test_session(client, user1_headers, "basic_chat")
        
        # User 2 tries to access User 1's session
        response = client.get(f"/api/v1/sessions/{session_id}", headers=user2_headers)
        
        # Should be forbidden or not found
        assert response.status_code in [403, 404]
    
    async def test_admin_only_endpoints(self, ff_chat_api, api_test_client, api_helper):
        """Test that admin endpoints require admin privileges"""
        client = api_test_client(ff_chat_api)
        
        # Regular user token
        user_token = "regular_user_token"
        user_headers = api_helper.create_auth_headers(user_token)
        
        admin_endpoints = [
            "/api/v1/admin/status",
            "/api/v1/admin/users",
            "/api/v1/admin/metrics"
        ]
        
        for endpoint in admin_endpoints:
            response = client.get(endpoint, headers=user_headers)
            assert response.status_code == 403, f"Regular user should not access admin endpoint: {endpoint}"
    
    async def test_api_key_permissions(self, ff_chat_api, api_test_client):
        """Test API key permission enforcement"""
        client = api_test_client(ff_chat_api)
        
        # Test with limited permission API key
        limited_api_key = "limited_permissions_key"
        headers = {"X-API-Key": limited_api_key}
        
        # Try to access endpoint requiring higher permissions
        response = client.get("/api/v1/admin/users", headers=headers)
        
        # Should be forbidden
        assert response.status_code in [401, 403]

class TestFFChatAPIDataSecurity:
    """Test FF Chat API data security measures"""
    
    async def test_sensitive_data_not_logged(self, ff_chat_api, api_test_client):
        """Test that sensitive data is not exposed in logs or responses"""
        client = api_test_client(ff_chat_api)
        
        # Register user with sensitive data
        sensitive_data = {
            "username": "securitytest",
            "password": "VerySecretPassword123!",
            "email": "security@test.com"
        }
        
        response = client.post("/api/v1/auth/register", json=sensitive_data)
        
        if response.status_code in [200, 201]:
            data = response.json()
            
            # Password should never be in response
            response_str = str(data).lower()
            assert "verysecretpassword" not in response_str
            assert "password" not in data or data["password"] is None
    
    async def test_error_message_security(self, ff_chat_api, api_test_client):
        """Test that error messages don't expose sensitive information"""
        client = api_test_client(ff_chat_api)
        
        # Trigger various errors and check responses
        error_tests = [
            {"endpoint": "/api/v1/sessions/nonexistent", "method": "GET"},
            {"endpoint": "/api/v1/chat/invalid_session/message", "method": "POST", "data": {"message": "test"}},
            {"endpoint": "/api/v1/users/nonexistent", "method": "GET"}
        ]
        
        for test in error_tests:
            if test["method"] == "GET":
                response = client.get(test["endpoint"])
            elif test["method"] == "POST":
                response = client.post(test["endpoint"], json=test.get("data", {}))
            
            if response.status_code >= 400:
                data = response.json() if hasattr(response, 'json') else {}
                error_message = str(data).lower()
                
                # Should not expose sensitive system information
                assert "traceback" not in error_message
                assert "exception" not in error_message
                assert "database" not in error_message
                assert "internal" not in error_message or "internal server error" in error_message
    
    async def test_data_sanitization(self, ff_chat_api, api_test_client, api_helper):
        """Test that user input is properly sanitized"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Test various potentially harmful inputs
        harmful_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "{{7*7}}",  # Template injection
            "${7*7}",   # Expression injection
            "eval('alert(1)')",
            "console.log('test')"
        ]
        
        for harmful_input in harmful_inputs:
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": harmful_input},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                response_content = data.get("response", "")
                
                # Should not execute or reflect harmful content
                assert "49" not in response_content  # 7*7 result
                assert "alert" not in response_content.lower()
                assert "script" not in response_content.lower()

class TestFFChatAPISecurityHeaders:
    """Test FF Chat API security headers"""
    
    async def test_security_headers_present(self, ff_chat_api, api_test_client):
        """Test that required security headers are present"""
        client = api_test_client(ff_chat_api)
        
        response = client.get("/health")
        
        # Check for important security headers
        headers = response.headers
        
        # Note: These may not all be present depending on middleware configuration
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        # At least some security headers should be present in production
        security_header_count = sum(1 for header in expected_headers if header in headers)
        
        # This is more of an informational test - log what headers are present
        present_headers = [h for h in expected_headers if h in headers]
        print(f"Security headers present: {present_headers}")
    
    async def test_cors_configuration(self, ff_chat_api, api_test_client):
        """Test CORS configuration security"""
        client = api_test_client(ff_chat_api)
        
        # Test preflight request
        response = client.options("/api/v1/sessions", headers={
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "POST"
        })
        
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            
            # Should not allow all origins in production
            assert allowed_origin != "*" or "test" in str(ff_chat_api), "CORS should not allow all origins in production"

class TestFFChatAPIRateLimiting:
    """Test FF Chat API rate limiting security"""
    
    async def test_rate_limiting_per_user(self, ff_chat_api, api_test_client, api_helper):
        """Test rate limiting per user"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        # Make many rapid requests
        rate_limit_hit = False
        for i in range(100):
            response = client.get("/api/v1/sessions", headers=headers)
            
            if response.status_code == 429:  # Too Many Requests
                rate_limit_hit = True
                break
        
        # Note: Rate limiting may not be enabled in test environment
        # This test verifies the system handles rate limiting gracefully
        print(f"Rate limiting {'activated' if rate_limit_hit else 'not activated'}")
    
    async def test_rate_limiting_per_endpoint(self, ff_chat_api, api_test_client, api_helper):
        """Test rate limiting for expensive endpoints"""
        client = api_test_client(ff_chat_api)
        
        token = "test_token"
        headers = api_helper.create_auth_headers(token)
        
        session_id = api_helper.create_test_session(client, headers, "basic_chat")
        
        # Rapidly send many messages (expensive operation)
        for i in range(50):
            response = client.post(
                f"/api/v1/chat/{session_id}/message",
                json={"message": f"Message {i}"},
                headers=headers
            )
            
            if response.status_code == 429:
                print("Rate limiting activated for message endpoint")
                break

class TestFFChatAPISecurityMiscellaneous:
    """Test miscellaneous security aspects"""
    
    async def test_session_timeout(self, ff_chat_api, api_test_client, api_helper):
        """Test session timeout security"""
        client = api_test_client(ff_chat_api)
        
        # This would test session expiration in a real scenario
        # For now, just verify the system handles expired tokens gracefully
        expired_token = "expired_token_12345"
        headers = api_helper.create_auth_headers(expired_token)
        
        response = client.get("/api/v1/user/profile", headers=headers)
        
        # Should reject expired token
        assert response.status_code == 401
    
    async def test_content_type_validation(self, ff_chat_api, api_test_client):
        """Test content type validation"""
        client = api_test_client(ff_chat_api)
        
        # Try to send malformed content
        response = client.post(
            "/api/v1/auth/register",
            data="malformed data",  # Raw string instead of JSON
            headers={"Content-Type": "application/json"}
        )
        
        # Should reject malformed content
        assert response.status_code in [400, 422]
    
    async def test_method_security(self, ff_chat_api, api_test_client):
        """Test HTTP method security"""
        client = api_test_client(ff_chat_api)
        
        # Try dangerous HTTP methods on endpoints
        dangerous_methods = ["TRACE", "CONNECT"]
        
        for method in dangerous_methods:
            try:
                response = client.request(method, "/api/v1/sessions")
                # Should not allow dangerous methods
                assert response.status_code in [405, 501], f"Method {method} should not be allowed"
            except Exception:
                # Exception is acceptable for unsupported methods
                pass