"""
FF Admin Routes - Administrative endpoints

Provides REST API endpoints for system administration, user management,
monitoring, and maintenance operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field

from ff_chat_auth import FFChatAuthManager, User, Permission, UserRole
from ff_chat_application import FFChatApplication
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)

# Pydantic models
class SystemStatus(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    ff_backend_status: Dict[str, str]
    component_status: Dict[str, str]
    active_sessions: int
    total_users: int
    system_resources: Dict[str, Any]

class AdminUserResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    created_at: str
    last_login: Optional[str]
    is_active: bool
    api_keys_count: int
    session_count: int

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=lambda: ["user"])

class UpdateUserRequest(BaseModel):
    email: Optional[str] = None
    roles: Optional[List[str]] = None
    is_active: Optional[bool] = None

class SystemMetrics(BaseModel):
    timestamp: str
    performance_metrics: Dict[str, Any]
    usage_metrics: Dict[str, Any]
    error_metrics: Dict[str, Any]
    resource_metrics: Dict[str, Any]

class MaintenanceRequest(BaseModel):
    operation: str = Field(..., description="Maintenance operation")
    parameters: Optional[Dict[str, Any]] = None
    dry_run: bool = Field(False, description="Perform dry run without changes")

# Create router with admin dependency
def require_admin_permission(permission: Permission):
    """Dependency factory for admin permissions"""
    async def admin_check(user: User = Depends(lambda: None)):
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        
        if not (hasattr(user, 'permissions') and permission in user.permissions):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
        
        return user
    return admin_check

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(require_admin_permission(Permission.VIEW_METRICS))
):
    """Get comprehensive system status and health information."""
    try:
        import time
        import psutil
        
        # Get system uptime (placeholder)
        uptime_seconds = time.time() - 1640995200  # Placeholder start time
        
        # Get FF backend status
        ff_status = {}
        if chat_app.ff_storage:
            ff_status["storage"] = "healthy"
        else:
            ff_status["storage"] = "unavailable"
        
        # Get component status
        components_info = await chat_app.get_components_info()
        component_status = {}
        for name, info in components_info.items():
            component_status[name] = "active" if info.get("initialized", False) else "inactive"
        
        # Get system metrics
        try:
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_resources = {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
        except:
            system_resources = {"error": "Unable to get system resources"}
        
        return SystemStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            ff_backend_status=ff_status,
            component_status=component_status,
            active_sessions=len(getattr(chat_app, 'active_sessions', {})),
            total_users=0,  # Would get from auth manager
            system_resources=system_resources
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users", response_model=List[AdminUserResponse])
async def list_all_users(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    active_only: bool = Query(False),
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(require_admin_permission(Permission.VIEW_USERS))
):
    """List all users in the system (admin only)."""
    try:
        users = []
        for user_obj in auth_manager.users_cache.values():
            if active_only and not user_obj.is_active:
                continue
            
            users.append(AdminUserResponse(
                user_id=user_obj.user_id,
                username=user_obj.username,
                email=user_obj.email,
                roles=[role.value for role in user_obj.roles],
                permissions=[perm.value for perm in user_obj.permissions],
                created_at=user_obj.created_at.isoformat(),
                last_login=user_obj.last_login.isoformat() if user_obj.last_login else None,
                is_active=user_obj.is_active,
                api_keys_count=len(user_obj.api_keys),
                session_count=0  # Would query from chat app
            ))
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return users[start_idx:end_idx]
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/users", response_model=AdminUserResponse)
async def create_user_admin(
    request: CreateUserRequest,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(require_admin_permission(Permission.MANAGE_USERS))
):
    """Create a new user (admin only)."""
    try:
        # Parse roles
        roles = []
        for role_str in request.roles:
            try:
                roles.append(UserRole(role_str))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid role: {role_str}"
                )
        
        # Create user
        new_user = await auth_manager.create_user(
            username=request.username,
            password=request.password,
            email=request.email,
            roles=roles
        )
        
        return AdminUserResponse(
            user_id=new_user.user_id,
            username=new_user.username,
            email=new_user.email,
            roles=[role.value for role in new_user.roles],
            permissions=[perm.value for perm in new_user.permissions],
            created_at=new_user.created_at.isoformat(),
            last_login=None,
            is_active=new_user.is_active,
            api_keys_count=0,
            session_count=0
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users/{user_id}", response_model=AdminUserResponse)
async def get_user_admin(
    user_id: str,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(require_admin_permission(Permission.VIEW_USERS))
):
    """Get detailed user information (admin only)."""
    try:
        target_user = auth_manager.users_cache.get(user_id)
        if not target_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        return AdminUserResponse(
            user_id=target_user.user_id,
            username=target_user.username,
            email=target_user.email,
            roles=[role.value for role in target_user.roles],
            permissions=[perm.value for perm in target_user.permissions],
            created_at=target_user.created_at.isoformat(),
            last_login=target_user.last_login.isoformat() if target_user.last_login else None,
            is_active=target_user.is_active,
            api_keys_count=len(target_user.api_keys),
            session_count=0  # Would query from chat app
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.put("/users/{user_id}", response_model=AdminUserResponse)
async def update_user_admin(
    user_id: str,
    request: UpdateUserRequest,
    auth_manager: FFChatAuthManager = Depends(),
    user: User = Depends(require_admin_permission(Permission.MANAGE_USERS))
):
    """Update user information (admin only)."""
    try:
        target_user = auth_manager.users_cache.get(user_id)
        if not target_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Update fields
        if request.email is not None:
            target_user.email = request.email
        
        if request.roles is not None:
            roles = []
            for role_str in request.roles:
                try:
                    roles.append(UserRole(role_str))
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid role: {role_str}"
                    )
            target_user.roles = roles
            
            # Update permissions based on new roles
            target_user.permissions = set()
            for role in target_user.roles:
                target_user.permissions.update(auth_manager.role_permissions.get(role, set()))
        
        if request.is_active is not None:
            target_user.is_active = request.is_active
        
        # Save changes
        await auth_manager._save_user_to_storage(target_user)
        
        return AdminUserResponse(
            user_id=target_user.user_id,
            username=target_user.username,
            email=target_user.email,
            roles=[role.value for role in target_user.roles],
            permissions=[perm.value for perm in target_user.permissions],
            created_at=target_user.created_at.isoformat(),
            last_login=target_user.last_login.isoformat() if target_user.last_login else None,
            is_active=target_user.is_active,
            api_keys_count=len(target_user.api_keys),
            session_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    include_detailed: bool = Query(False, description="Include detailed metrics"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(require_admin_permission(Permission.VIEW_METRICS))
):
    """Get comprehensive system metrics and analytics."""
    try:
        # Get chat system metrics
        chat_metrics = await chat_app.get_metrics()
        
        # Performance metrics
        performance_metrics = {
            "avg_response_time": chat_metrics.get("avg_response_time", 0),
            "total_requests": chat_metrics.get("total_requests", 0),
            "successful_requests": chat_metrics.get("successful_requests", 0),
            "failed_requests": chat_metrics.get("failed_requests", 0)
        }
        
        # Usage metrics
        usage_metrics = {
            "active_sessions": chat_metrics.get("active_sessions", 0),
            "total_messages": chat_metrics.get("total_messages", 0),
            "unique_users": chat_metrics.get("unique_users", 0),
            "use_cases_used": chat_metrics.get("use_cases_used", {})
        }
        
        # Error metrics
        error_metrics = {
            "error_rate": chat_metrics.get("error_rate", 0),
            "component_errors": chat_metrics.get("component_errors", {}),
            "recent_errors": chat_metrics.get("recent_errors", []) if include_detailed else []
        }
        
        # Resource metrics
        resource_metrics = {
            "memory_usage": chat_metrics.get("memory_usage", {}),
            "storage_usage": chat_metrics.get("storage_usage", {}),
            "component_resource_usage": chat_metrics.get("component_resource_usage", {})
        }
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            usage_metrics=usage_metrics,
            error_metrics=error_metrics,
            resource_metrics=resource_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/maintenance")
async def perform_maintenance(
    request: MaintenanceRequest,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(require_admin_permission(Permission.MANAGE_SYSTEM))
):
    """Perform system maintenance operations."""
    try:
        operation = request.operation.lower()
        parameters = request.parameters or {}
        
        if operation == "cleanup_sessions":
            # Clean up inactive sessions
            result = await _cleanup_inactive_sessions(chat_app, parameters, request.dry_run)
            
        elif operation == "optimize_storage":
            # Optimize FF storage
            result = await _optimize_ff_storage(chat_app, parameters, request.dry_run)
            
        elif operation == "rebuild_indexes":
            # Rebuild search indexes
            result = await _rebuild_search_indexes(chat_app, parameters, request.dry_run)
            
        elif operation == "clear_cache":
            # Clear system caches
            result = await _clear_system_caches(chat_app, parameters, request.dry_run)
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown maintenance operation: {operation}"
            )
        
        return {
            "operation": operation,
            "dry_run": request.dry_run,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "performed_by": user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing maintenance: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", description="Log level filter"),
    limit: int = Query(100, ge=1, le=1000),
    component: Optional[str] = Query(None, description="Filter by component"),
    user: User = Depends(require_admin_permission(Permission.MANAGE_SYSTEM))
):
    """Get system logs (admin only)."""
    try:
        # In a real implementation, this would read from log files or logging service
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "ff_chat_api",
                "message": "System running normally",
                "metadata": {}
            }
        ]
        
        return {
            "logs": logs,
            "total_count": len(logs),
            "filters": {
                "level": level,
                "component": component,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Maintenance operation implementations
async def _cleanup_inactive_sessions(chat_app: FFChatApplication, parameters: Dict[str, Any], dry_run: bool):
    """Clean up inactive sessions"""
    # Implementation would go here
    return {"cleaned_sessions": 0, "dry_run": dry_run}

async def _optimize_ff_storage(chat_app: FFChatApplication, parameters: Dict[str, Any], dry_run: bool):
    """Optimize FF storage"""
    # Implementation would go here
    return {"optimized_files": 0, "space_saved_mb": 0, "dry_run": dry_run}

async def _rebuild_search_indexes(chat_app: FFChatApplication, parameters: Dict[str, Any], dry_run: bool):
    """Rebuild search indexes"""
    # Implementation would go here
    return {"indexes_rebuilt": 0, "dry_run": dry_run}

async def _clear_system_caches(chat_app: FFChatApplication, parameters: Dict[str, Any], dry_run: bool):
    """Clear system caches"""
    # Implementation would go here
    return {"caches_cleared": 0, "dry_run": dry_run}