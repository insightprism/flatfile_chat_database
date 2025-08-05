"""
FF Chat API Middleware and Security

Provides comprehensive middleware for security, logging, rate limiting,
request validation, and other cross-cutting concerns for the FF Chat API.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from ff_utils.ff_logging import get_logger
from ff_chat_auth import FFChatAuthManager, Permission

logger = get_logger(__name__)

class SecurityLevel(Enum):
    """Security levels for different endpoints"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    RESTRICTED = "restricted"
    ADMIN = "admin"

@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10

@dataclass
class SecurityRule:
    """Security rule for endpoints"""
    path_pattern: str
    security_level: SecurityLevel
    required_permissions: Set[Permission] = field(default_factory=set)
    rate_limit: Optional[RateLimitRule] = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE"})

@dataclass
class MiddlewareConfig:
    """Middleware configuration"""
    enable_cors: bool = True
    enable_rate_limiting: bool = True
    enable_request_logging: bool = True
    enable_security_headers: bool = True
    enable_request_validation: bool = True
    enable_compression: bool = True
    
    # Rate limiting defaults
    default_rate_limit: RateLimitRule = field(default_factory=RateLimitRule)
    
    # Security headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    })
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

class RequestTracker:
    """Tracks requests for rate limiting and analytics"""
    
    def __init__(self):
        self.request_history: Dict[str, List[datetime]] = {}
        self.request_stats: Dict[str, Dict[str, int]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
    
    def record_request(self, client_ip: str, endpoint: str, user_agent: str = ""):
        """Record a request for tracking"""
        current_time = datetime.now()
        
        # Track by IP
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        self.request_history[client_ip].append(current_time)
        
        # Track stats
        if client_ip not in self.request_stats:
            self.request_stats[client_ip] = {"total": 0, "endpoints": {}}
        
        self.request_stats[client_ip]["total"] += 1
        if endpoint not in self.request_stats[client_ip]["endpoints"]:
            self.request_stats[client_ip]["endpoints"][endpoint] = 0
        self.request_stats[client_ip]["endpoints"][endpoint] += 1
        
        # Detect suspicious patterns
        self._detect_suspicious_patterns(client_ip, endpoint, user_agent)
    
    def _detect_suspicious_patterns(self, client_ip: str, endpoint: str, user_agent: str):
        """Detect suspicious request patterns"""
        # Check for rapid requests
        recent_requests = [
            t for t in self.request_history.get(client_ip, [])
            if datetime.now() - t < timedelta(minutes=1)
        ]
        
        if len(recent_requests) > 100:  # More than 100 requests per minute
            self.suspicious_patterns[client_ip] = self.suspicious_patterns.get(client_ip, 0) + 1
        
        # Check for scanner patterns
        scanner_endpoints = ["/admin", "/.env", "/wp-admin", "/config", "/backup"]
        if any(pattern in endpoint for pattern in scanner_endpoints):
            self.suspicious_patterns[client_ip] = self.suspicious_patterns.get(client_ip, 0) + 5
        
        # Auto-block if too many suspicious activities
        if self.suspicious_patterns.get(client_ip, 0) > 10:
            self.blocked_ips.add(client_ip)
            logger.warning(f"Auto-blocked suspicious IP: {client_ip}")
    
    def is_rate_limited(self, client_ip: str, rule: RateLimitRule) -> tuple[bool, str]:
        """Check if client is rate limited"""
        current_time = datetime.now()
        requests = self.request_history.get(client_ip, [])
        
        # Clean old requests
        cutoff_day = current_time - timedelta(days=1)
        cutoff_hour = current_time - timedelta(hours=1)
        cutoff_minute = current_time - timedelta(minutes=1)
        
        requests = [r for r in requests if r > cutoff_day]
        self.request_history[client_ip] = requests
        
        # Check limits
        requests_last_minute = sum(1 for r in requests if r > cutoff_minute)
        requests_last_hour = sum(1 for r in requests if r > cutoff_hour)
        requests_last_day = len(requests)
        
        if requests_last_minute > rule.requests_per_minute:
            return True, f"Rate limit exceeded: {requests_last_minute}/{rule.requests_per_minute} per minute"
        
        if requests_last_hour > rule.requests_per_hour:
            return True, f"Rate limit exceeded: {requests_last_hour}/{rule.requests_per_hour} per hour"
        
        if requests_last_day > rule.requests_per_day:
            return True, f"Rate limit exceeded: {requests_last_day}/{rule.requests_per_day} per day"
        
        return False, ""
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked"""
        return client_ip in self.blocked_ips
    
    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """Get statistics for a client IP"""
        return {
            "total_requests": self.request_stats.get(client_ip, {}).get("total", 0),
            "endpoints": self.request_stats.get(client_ip, {}).get("endpoints", {}),
            "suspicious_score": self.suspicious_patterns.get(client_ip, 0),
            "is_blocked": client_ip in self.blocked_ips,
            "recent_requests": len([
                r for r in self.request_history.get(client_ip, [])
                if datetime.now() - r < timedelta(hours=1)
            ])
        }

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for FF Chat API"""
    
    def __init__(self, app, config: MiddlewareConfig, auth_manager: Optional[FFChatAuthManager] = None):
        super().__init__(app)
        self.config = config
        self.auth_manager = auth_manager
        self.request_tracker = RequestTracker()
        self.logger = get_logger(__name__)
        
        # Security rules for different endpoints
        self.security_rules = self._init_security_rules()
    
    def _init_security_rules(self) -> List[SecurityRule]:
        """Initialize security rules for endpoints"""
        return [
            # Public endpoints
            SecurityRule(
                path_pattern="/health",
                security_level=SecurityLevel.PUBLIC,
                rate_limit=RateLimitRule(requests_per_minute=30)
            ),
            SecurityRule(
                path_pattern="/docs",
                security_level=SecurityLevel.PUBLIC,
                rate_limit=RateLimitRule(requests_per_minute=10)
            ),
            
            # Authentication endpoints
            SecurityRule(
                path_pattern="/api/v1/auth/login",
                security_level=SecurityLevel.PUBLIC,
                rate_limit=RateLimitRule(requests_per_minute=5, burst_limit=2)
            ),
            SecurityRule(
                path_pattern="/api/v1/auth/register",
                security_level=SecurityLevel.PUBLIC,
                rate_limit=RateLimitRule(requests_per_minute=3, burst_limit=1)
            ),
            
            # Chat endpoints
            SecurityRule(
                path_pattern="/api/v1/sessions",
                security_level=SecurityLevel.AUTHENTICATED,
                required_permissions={Permission.CREATE_SESSION, Permission.VIEW_SESSION},
                rate_limit=RateLimitRule(requests_per_minute=30)
            ),
            SecurityRule(
                path_pattern="/api/v1/sessions/*/messages",
                security_level=SecurityLevel.AUTHENTICATED,
                required_permissions={Permission.SEND_MESSAGE, Permission.VIEW_MESSAGES},
                rate_limit=RateLimitRule(requests_per_minute=60)
            ),
            
            # Admin endpoints
            SecurityRule(
                path_pattern="/api/v1/admin/*",
                security_level=SecurityLevel.ADMIN,
                required_permissions={Permission.MANAGE_SYSTEM},
                rate_limit=RateLimitRule(requests_per_minute=10)
            ),
            
            # WebSocket endpoints
            SecurityRule(
                path_pattern="/ws/*",
                security_level=SecurityLevel.AUTHENTICATED,
                required_permissions={Permission.USE_WEBSOCKET},
                allowed_methods={"GET"}  # WebSocket upgrade
            )
        ]
    
    def _get_security_rule(self, path: str, method: str) -> Optional[SecurityRule]:
        """Get security rule for path and method"""
        import re
        
        for rule in self.security_rules:
            # Convert path pattern to regex
            pattern = rule.path_pattern.replace("*", ".*")
            if re.match(f"^{pattern}$", path):
                if method in rule.allowed_methods:
                    return rule
        
        # Default rule for unmatched paths
        return SecurityRule(
            path_pattern="*",
            security_level=SecurityLevel.AUTHENTICATED,
            rate_limit=self.config.default_rate_limit
        )
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Main middleware dispatch"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Get client info
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            # Check if IP is blocked
            if self.request_tracker.is_blocked(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "IP address blocked due to suspicious activity"}
                )
            
            # Get security rule for this endpoint
            security_rule = self._get_security_rule(request.url.path, request.method)
            
            # Rate limiting
            if self.config.enable_rate_limiting and security_rule.rate_limit:
                is_limited, message = self.request_tracker.is_rate_limited(client_ip, security_rule.rate_limit)
                if is_limited:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"error": message}
                    )
            
            # Request size validation
            if security_rule.max_request_size:
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > security_rule.max_request_size:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"error": "Request too large"}
                    )
            
            # Input validation
            if self.config.enable_request_validation:
                validation_error = await self._validate_request(request)
                if validation_error:
                    return validation_error
            
            # Security level checking
            auth_error = await self._check_security_level(request, security_rule)
            if auth_error:
                return auth_error
            
            # Record request
            self.request_tracker.record_request(client_ip, request.url.path, user_agent)
            
            # Log request if enabled
            if self.config.enable_request_logging:
                self._log_request(request, client_ip, request_id)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            if self.config.enable_security_headers:
                self._add_security_headers(response)
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{time.time() - start_time:.3f}"
            
            # Log response
            if self.config.enable_request_logging:
                self._log_response(request, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Middleware error for request {request_id}: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "request_id": request_id}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP address"""
        # Check for proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _validate_request(self, request: Request) -> Optional[JSONResponse]:
        """Validate incoming request"""
        try:
            # Check content type for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                
                if not content_type:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"error": "Content-Type header required"}
                    )
                
                # Validate JSON content
                if "application/json" in content_type:
                    try:
                        body = await request.body()
                        if body:
                            json.loads(body)
                    except json.JSONDecodeError:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"error": "Invalid JSON in request body"}
                        )
            
            # Check for suspicious patterns in URL
            suspicious_patterns = ["../", "..\\", "<script", "javascript:", "vbscript:", "onload="]
            url_path = request.url.path.lower()
            
            for pattern in suspicious_patterns:
                if pattern in url_path:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"error": "Invalid characters in URL"}
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Request validation failed"}
            )
    
    async def _check_security_level(self, request: Request, rule: SecurityRule) -> Optional[JSONResponse]:
        """Check if request meets security requirements"""
        if rule.security_level == SecurityLevel.PUBLIC:
            return None
        
        if not self.auth_manager:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Authentication not configured"}
            )
        
        # Check authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication required"}
            )
        
        # Extract token
        token = auth_header.split(" ")[1]
        
        try:
            # Verify token and get user
            user = None
            api_key = None
            
            # Try JWT first
            if self.auth_manager.config.enable_jwt:
                try:
                    payload = self.auth_manager.verify_token(token)
                    user_id = payload.get("sub")
                    if user_id:
                        user = self.auth_manager.users_cache.get(user_id)
                except:
                    pass
            
            # Try API key if JWT failed
            if not user and self.auth_manager.config.enable_api_keys:
                result = await self.auth_manager.verify_api_key(token)
                if result:
                    user, api_key = result
            
            if not user:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid token"}
                )
            
            # Check permissions
            for permission in rule.required_permissions:
                if not self.auth_manager.check_permission(user, permission, api_key):
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"error": f"Permission required: {permission.value}"}
                    )
            
            # Add user to request state
            request.state.user = user
            request.state.api_key = api_key
            
            return None
            
        except Exception as e:
            self.logger.error(f"Security check error: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication failed"}
            )
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.config.security_headers.items():
            response.headers[header] = value
    
    def _log_request(self, request: Request, client_ip: str, request_id: str):
        """Log incoming request"""
        self.logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {client_ip} - {request.headers.get('user-agent', 'unknown')}"
        )
    
    def _log_response(self, request: Request, response: Response, processing_time: float):
        """Log response"""
        self.logger.info(
            f"Response {getattr(request.state, 'request_id', 'unknown')}: "
            f"{response.status_code} - {processing_time:.3f}s"
        )

class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware"""
    
    def __init__(self, app, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Compress response if appropriate"""
        response = await call_next(request)
        
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/css",
            "application/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return response
        
        # Check response size
        if hasattr(response, 'body'):
            body_size = len(response.body) if response.body else 0
            if body_size < self.minimum_size:
                return response
        
        # Add compression header (actual compression would be handled by server/proxy)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        
        return response

def create_middleware_stack(config: MiddlewareConfig, auth_manager: Optional[FFChatAuthManager] = None) -> List[tuple]:
    """Create middleware stack for FF Chat API"""
    middleware_stack = []
    
    # Security middleware (first)
    middleware_stack.append((SecurityMiddleware, {"config": config, "auth_manager": auth_manager}))
    
    # Compression middleware (last)
    if config.enable_compression:
        middleware_stack.append((CompressionMiddleware, {}))
    
    return middleware_stack