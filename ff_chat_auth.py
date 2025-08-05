"""
FF Chat Authentication and Authorization

Provides comprehensive authentication and authorization for the FF Chat API,
including JWT tokens, role-based access control, and API key management.
"""

import hashlib
import secrets
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ff_utils.ff_logging import get_logger
from ff_core_storage_manager import FFCoreStorageManager

logger = get_logger(__name__)

class UserRole(Enum):
    """User roles for RBAC"""
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SYSTEM = "system"

class Permission(Enum):
    """System permissions"""
    # Session permissions
    CREATE_SESSION = "create_session"
    VIEW_SESSION = "view_session"
    DELETE_SESSION = "delete_session"
    
    # Message permissions
    SEND_MESSAGE = "send_message"
    VIEW_MESSAGES = "view_messages"
    DELETE_MESSAGE = "delete_message"
    
    # Search permissions
    SEARCH_MESSAGES = "search_messages"
    SEARCH_ALL_USERS = "search_all_users"
    
    # Admin permissions
    VIEW_USERS = "view_users"
    MANAGE_USERS = "manage_users"
    VIEW_METRICS = "view_metrics"
    MANAGE_SYSTEM = "manage_system"
    
    # API permissions
    USE_API = "use_api"
    USE_WEBSOCKET = "use_websocket"

@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: Optional[str] = None
    password_hash: Optional[str] = None
    roles: List[UserRole] = field(default_factory=lambda: [UserRole.USER])
    permissions: Set[Permission] = field(default_factory=set)
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIKey:
    """API key model"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True

@dataclass
class JWTConfig:
    """JWT configuration"""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

@dataclass
class AuthConfig:
    """Authentication configuration"""
    enable_jwt: bool = True
    enable_api_keys: bool = True
    require_email_verification: bool = False
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    jwt_config: JWTConfig = field(default_factory=JWTConfig)

class FFChatAuthManager:
    """
    FF Chat Authentication and Authorization Manager
    
    Provides comprehensive auth services including user management,
    JWT tokens, API keys, and role-based access control.
    """
    
    def __init__(self, config: AuthConfig, ff_storage: FFCoreStorageManager):
        """
        Initialize auth manager.
        
        Args:
            config: Authentication configuration
            ff_storage: FF storage manager for persistence
        """
        self.config = config
        self.ff_storage = ff_storage
        self.logger = get_logger(__name__)
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory caches (in production, use Redis or similar)
        self.users_cache: Dict[str, User] = {}
        self.api_keys_cache: Dict[str, APIKey] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # JWT security
        self.security = HTTPBearer()
        
        # Role permissions mapping
        self.role_permissions = self._init_role_permissions()
        
        self._initialized = False
    
    def _init_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-based permissions"""
        return {
            UserRole.USER: {
                Permission.CREATE_SESSION,
                Permission.VIEW_SESSION,
                Permission.DELETE_SESSION,
                Permission.SEND_MESSAGE,
                Permission.VIEW_MESSAGES,
                Permission.SEARCH_MESSAGES,
                Permission.USE_API,
                Permission.USE_WEBSOCKET
            },
            UserRole.MODERATOR: {
                Permission.CREATE_SESSION,
                Permission.VIEW_SESSION,
                Permission.DELETE_SESSION,
                Permission.SEND_MESSAGE,
                Permission.VIEW_MESSAGES,
                Permission.DELETE_MESSAGE,
                Permission.SEARCH_MESSAGES,
                Permission.SEARCH_ALL_USERS,
                Permission.USE_API,
                Permission.USE_WEBSOCKET
            },
            UserRole.ADMIN: {
                # All permissions
                *Permission,
            },
            UserRole.SYSTEM: {
                # All permissions
                *Permission,
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize auth manager"""
        try:
            self.logger.info("Initializing FF Chat Auth Manager...")
            
            # Load users and API keys from storage
            await self._load_users_from_storage()
            await self._load_api_keys_from_storage()
            
            # Create default admin user if none exists
            await self._ensure_default_admin()
            
            self._initialized = True
            self.logger.info("FF Chat Auth Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize auth manager: {e}")
            return False
    
    async def _load_users_from_storage(self):
        """Load users from FF storage"""
        try:
            # In a real implementation, this would load from FF storage
            # For now, we'll use in-memory storage
            pass
        except Exception as e:
            self.logger.error(f"Error loading users from storage: {e}")
    
    async def _load_api_keys_from_storage(self):
        """Load API keys from FF storage"""
        try:
            # In a real implementation, this would load from FF storage
            # For now, we'll use in-memory storage
            pass
        except Exception as e:
            self.logger.error(f"Error loading API keys from storage: {e}")
    
    async def _ensure_default_admin(self):
        """Ensure default admin user exists"""
        admin_users = [u for u in self.users_cache.values() if UserRole.ADMIN in u.roles]
        
        if not admin_users:
            # Create default admin
            admin_user = User(
                user_id="admin_001",
                username="admin",
                email="admin@localhost",
                password_hash=self.hash_password("admin123"),
                roles=[UserRole.ADMIN],
                permissions=self.role_permissions[UserRole.ADMIN]
            )
            
            self.users_cache[admin_user.user_id] = admin_user
            await self._save_user_to_storage(admin_user)
            
            self.logger.info("Created default admin user (username: admin, password: admin123)")
    
    async def _save_user_to_storage(self, user: User):
        """Save user to FF storage"""
        try:
            # In a real implementation, this would save to FF storage
            pass
        except Exception as e:
            self.logger.error(f"Error saving user to storage: {e}")
    
    async def _save_api_key_to_storage(self, api_key: APIKey):
        """Save API key to FF storage"""
        try:
            # In a real implementation, this would save to FF storage
            pass
        except Exception as e:
            self.logger.error(f"Error saving API key to storage: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.config.jwt_config.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(
            to_encode,
            self.config.jwt_config.secret_key,
            algorithm=self.config.jwt_config.algorithm
        )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.jwt_config.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(
            to_encode,
            self.config.jwt_config.secret_key,
            algorithm=self.config.jwt_config.algorithm
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_config.secret_key,
                algorithms=[self.config.jwt_config.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def create_user(self, username: str, password: str, email: Optional[str] = None, 
                         roles: List[UserRole] = None) -> User:
        """Create new user"""
        # Validate password
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")
        
        # Check if username exists
        if any(u.username == username for u in self.users_cache.values()):
            raise ValueError("Username already exists")
        
        # Create user
        user = User(
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            roles=roles or [UserRole.USER]
        )
        
        # Set permissions based on roles
        user.permissions = set()
        for role in user.roles:
            user.permissions.update(self.role_permissions.get(role, set()))
        
        # Store user
        self.users_cache[user.user_id] = user
        await self._save_user_to_storage(user)
        
        self.logger.info(f"Created user: {username} ({user.user_id})")
        return user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        # Check failed attempts
        if await self._is_account_locked(username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to failed login attempts"
            )
        
        # Find user
        user = None
        for u in self.users_cache.values():
            if u.username == username:
                user = u
                break
        
        if not user or not user.is_active:
            await self._record_failed_attempt(username)
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            await self._record_failed_attempt(username)
            return None
        
        # Clear failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        # Update last login
        user.last_login = datetime.now()
        await self._save_user_to_storage(user)
        
        return user
    
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[username]
        cutoff_time = datetime.now() - timedelta(minutes=self.config.lockout_duration_minutes)
        
        # Remove old attempts
        attempts = [a for a in attempts if a > cutoff_time]
        self.failed_attempts[username] = attempts
        
        return len(attempts) >= self.config.max_failed_attempts
    
    async def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now())
    
    async def create_api_key(self, user_id: str, name: str, 
                           permissions: Set[Permission] = None,
                           expires_at: Optional[datetime] = None) -> tuple[str, APIKey]:
        """Create API key for user"""
        user = self.users_cache.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate API key
        raw_key = f"ffchat_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Create API key object
        api_key = APIKey(
            key_id=f"key_{uuid.uuid4().hex[:8]}",
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or user.permissions.copy(),
            expires_at=expires_at
        )
        
        # Store API key
        self.api_keys_cache[api_key.key_id] = api_key
        user.api_keys.append(api_key.key_id)
        
        await self._save_api_key_to_storage(api_key)
        await self._save_user_to_storage(user)
        
        self.logger.info(f"Created API key: {name} for user {user_id}")
        return raw_key, api_key
    
    async def verify_api_key(self, raw_key: str) -> Optional[tuple[User, APIKey]]:
        """Verify API key and return associated user"""
        if not raw_key.startswith("ffchat_"):
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Find API key
        api_key = None
        for key in self.api_keys_cache.values():
            if key.key_hash == key_hash and key.is_active:
                api_key = key
                break
        
        if not api_key:
            return None
        
        # Check expiration
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            return None
        
        # Get user
        user = self.users_cache.get(api_key.user_id)
        if not user or not user.is_active:
            return None
        
        # Update last used
        api_key.last_used = datetime.now()
        await self._save_api_key_to_storage(api_key)
        
        return user, api_key
    
    def check_permission(self, user: User, permission: Permission, 
                        api_key: Optional[APIKey] = None) -> bool:
        """Check if user has permission"""
        if not user.is_active:
            return False
        
        # If using API key, check API key permissions
        if api_key:
            return permission in api_key.permissions
        
        # Check user permissions
        return permission in user.permissions
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
        """FastAPI dependency to get current authenticated user"""
        token = credentials.credentials
        
        # Try JWT token first
        if self.config.enable_jwt:
            try:
                payload = self.verify_token(token)
                user_id = payload.get("sub")
                if user_id:
                    user = self.users_cache.get(user_id)
                    if user and user.is_active:
                        return user
            except HTTPException:
                pass
        
        # Try API key
        if self.config.enable_api_keys:
            result = await self.verify_api_key(token)
            if result:
                user, api_key = result
                return user
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    def require_permission(self, permission: Permission):
        """FastAPI dependency factory for permission checking"""
        async def permission_checker(user: User = Depends(self.get_current_user)) -> User:
            if not self.check_permission(user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}"
                )
            return user
        
        return permission_checker
    
    def require_role(self, role: UserRole):
        """FastAPI dependency factory for role checking"""
        async def role_checker(user: User = Depends(self.get_current_user)) -> User:
            if role not in user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}"
                )
            return user
        
        return role_checker
    
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and return tokens"""
        user = await self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create tokens
        token_data = {"sub": user.user_id, "username": user.username}
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles]
            }
        }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        try:
            payload = self.verify_token(refresh_token)
            if payload.get("type") != "refresh":
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            user_id = payload.get("sub")
            user = self.users_cache.get(user_id)
            
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="User not found or inactive")
            
            # Create new access token
            token_data = {"sub": user.user_id, "username": user.username}
            access_token = self.create_access_token(token_data)
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        user = self.users_cache.get(user_id)
        if not user:
            return None
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active,
            "api_keys_count": len(user.api_keys)
        }
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        active_users = sum(1 for u in self.users_cache.values() if u.is_active)
        active_api_keys = sum(1 for k in self.api_keys_cache.values() if k.is_active)
        
        return {
            "total_users": len(self.users_cache),
            "active_users": active_users,
            "total_api_keys": len(self.api_keys_cache),
            "active_api_keys": active_api_keys,
            "locked_accounts": len(self.failed_attempts),
            "jwt_enabled": self.config.enable_jwt,
            "api_keys_enabled": self.config.enable_api_keys
        }