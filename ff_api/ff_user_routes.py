"""
FF User Routes - User management and authentication endpoints

Provides REST API endpoints for user registration, authentication,
profile management, and user settings.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, EmailStr

from ff_chat_auth import FFChatAuthManager, User, Permission, UserRole
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)

# Pydantic models
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password")
    email: Optional[EmailStr] = Field(None, description="Email address")

class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token")

class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., description="API key name")
    permissions: Optional[List[str]] = Field(None, description="API key permissions")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")

class UpdateProfileRequest(BaseModel):
    email: Optional[EmailStr] = Field(None, description="New email address")
    metadata: Optional[Dict[str, Any]] = Field(None, description="User metadata")

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    created_at: str
    last_login: Optional[str]
    is_active: bool

class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]
    last_used: Optional[str]
    is_active: bool

# Create router
router = APIRouter(prefix="/api/v1/users", tags=["Users"])

@router.post("/register", response_model=UserResponse)
async def register_user(
    request: RegisterRequest,
    auth_manager: FFChatAuthManager = Depends()
):
    """Register a new user account."""
    try:
        user = await auth_manager.create_user(
            username=request.username,
            password=request.password,
            email=request.email
        )
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            roles=[role.value for role in user.roles],
            created_at=user.created_at.isoformat(),
            last_login=None,
            is_active=user.is_active
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/login")
async def login_user(
    request: LoginRequest,
    auth_manager: FFChatAuthManager = Depends()
):
    """Authenticate user and return access tokens."""
    try:
        result = await auth_manager.login(request.username, request.password)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/refresh")
async def refresh_access_token(
    request: RefreshTokenRequest,
    auth_manager: FFChatAuthManager = Depends()
):
    """Refresh access token using refresh token."""
    try:
        result = await auth_manager.refresh_token(request.refresh_token)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    user: User = Depends(lambda: None)
):
    """Get current user's profile information."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        roles=[role.value for role in user.roles],
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
        is_active=user.is_active
    )

@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    request: UpdateProfileRequest,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """Update current user's profile information."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        # Update user information
        if request.email:
            user.email = request.email
        if request.metadata:
            user.metadata.update(request.metadata)
        
        # Save changes
        await auth_manager._save_user_to_storage(user)
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            roles=[role.value for role in user.roles],
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None,
            is_active=user.is_active
        )
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """Change user's password."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        # Verify current password
        if not auth_manager.verify_password(request.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user.password_hash = auth_manager.hash_password(request.new_password)
        await auth_manager._save_user_to_storage(user)
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/api-keys", response_model=Dict[str, Any])
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """Create a new API key for the user."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        # Parse permissions
        permissions = set()
        if request.permissions:
            for perm_str in request.permissions:
                try:
                    permissions.add(Permission(perm_str))
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid permission: {perm_str}"
                    )
        else:
            permissions = user.permissions.copy()
        
        # Set expiration
        expires_at = None
        if request.expires_days:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(days=request.expires_days)
        
        # Create API key
        raw_key, api_key = await auth_manager.create_api_key(
            user_id=user.user_id,
            name=request.name,
            permissions=permissions,
            expires_at=expires_at
        )
        
        return {
            "api_key": raw_key,
            "key_info": APIKeyResponse(
                key_id=api_key.key_id,
                name=api_key.name,
                permissions=[p.value for p in api_key.permissions],
                created_at=api_key.created_at.isoformat(),
                expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
                last_used=None,
                is_active=api_key.is_active
            ),
            "warning": "Store this API key securely. It cannot be retrieved again."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """List user's API keys (without the actual key values)."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        api_keys = []
        for key_id in user.api_keys:
            api_key = auth_manager.api_keys_cache.get(key_id)
            if api_key:
                api_keys.append(APIKeyResponse(
                    key_id=api_key.key_id,
                    name=api_key.name,
                    permissions=[p.value for p in api_key.permissions],
                    created_at=api_key.created_at.isoformat(),
                    expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
                    last_used=api_key.last_used.isoformat() if api_key.last_used else None,
                    is_active=api_key.is_active
                ))
        
        return api_keys
        
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """Revoke an API key."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        # Check if key belongs to user
        if key_id not in user.api_keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        # Revoke key
        api_key = auth_manager.api_keys_cache.get(key_id)
        if api_key:
            api_key.is_active = False
            await auth_manager._save_api_key_to_storage(api_key)
        
        # Remove from user's key list
        user.api_keys.remove(key_id)
        await auth_manager._save_user_to_storage(user)
        
        return {"message": "API key revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/stats")
async def get_user_stats(
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(lambda: None)
):
    """Get user's usage statistics."""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    try:
        # Basic user stats
        stats = {
            "user_id": user.user_id,
            "username": user.username,
            "account_created": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "api_keys_count": len(user.api_keys),
            "roles": [role.value for role in user.roles],
            "permissions_count": len(user.permissions)
        }
        
        # Add additional stats if available
        # In a real implementation, this would query usage metrics
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))