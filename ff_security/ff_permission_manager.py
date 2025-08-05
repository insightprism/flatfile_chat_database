"""
FF Permission Manager - Access Control and Permission Management

Provides fine-grained permission management for tool execution,
file access, and system resources in the FF Chat System.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_tools_config import FFToolsSecurityConfigDTO, FFToolSecurityLevel

logger = get_logger(__name__)


class FFPermissionLevel(Enum):
    """Permission levels for different operations"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    FULL = "full"


class FFResourceType(Enum):
    """Types of resources that can be controlled"""
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    SYSTEM_COMMANDS = "system_commands"
    PROCESSES = "processes"
    MEMORY = "memory"
    CPU = "cpu"
    TOOLS = "tools"
    APIS = "apis"
    DATABASES = "databases"


class FFPermissionScope(Enum):
    """Scope of permission application"""
    GLOBAL = "global"
    SESSION = "session"
    USER = "user"
    TOOL = "tool"
    RESOURCE = "resource"


@dataclass
class FFPermission:
    """Individual permission entry"""
    resource_type: FFResourceType
    resource_identifier: str  # Specific resource (file path, command name, etc.)
    permission_level: FFPermissionLevel
    scope: FFPermissionScope
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFPermissionRequest:
    """Request for permission to access a resource"""
    resource_type: FFResourceType
    resource_identifier: str
    requested_level: FFPermissionLevel
    requester_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""


@dataclass
class FFPermissionResult:
    """Result of permission check"""
    granted: bool
    permission_level: FFPermissionLevel
    reason: str
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    audit_info: Dict[str, Any] = field(default_factory=dict)


class FFPermissionManager:
    """
    Fine-grained permission management system for FF tools and resources.
    
    Provides role-based access control, resource-specific permissions,
    and audit logging for all permission checks and grants.
    """
    
    def __init__(self, security_config: FFToolsSecurityConfigDTO):
        """
        Initialize FF permission manager.
        
        Args:
            security_config: Security configuration
        """
        self.security_config = security_config
        self.logger = get_logger(__name__)
        
        # Permission storage
        self.permissions: Dict[str, List[FFPermission]] = {}  # user_id -> permissions
        self.role_permissions: Dict[str, List[FFPermission]] = {}  # role -> permissions
        self.global_permissions: List[FFPermission] = []
        
        # Role assignments
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> roles
        
        # Audit logging
        self.permission_audit_log: List[Dict[str, Any]] = []
        
        # Permission cache for performance
        self.permission_cache: Dict[str, FFPermissionResult] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Initialize default permissions and roles
        self._initialize_default_permissions()
        self._initialize_default_roles()
    
    def _initialize_default_permissions(self):
        """Initialize default permission structure"""
        # Default global permissions based on security config
        default_perms = [
            # File system permissions
            FFPermission(
                resource_type=FFResourceType.FILE_SYSTEM,
                resource_identifier="/tmp/*",
                permission_level=FFPermissionLevel.WRITE,
                scope=FFPermissionScope.GLOBAL,
                granted_by="system",
                granted_at=datetime.now()
            ),
            
            # Network permissions for allowed domains
            FFPermission(
                resource_type=FFResourceType.NETWORK,
                resource_identifier="https://*",
                permission_level=FFPermissionLevel.READ,
                scope=FFPermissionScope.GLOBAL,
                granted_by="system",
                granted_at=datetime.now(),
                conditions={"allowed_domains": self.security_config.allowed_domains}
            ),
            
            # Basic system commands
            FFPermission(
                resource_type=FFResourceType.SYSTEM_COMMANDS,
                resource_identifier="safe_commands",
                permission_level=FFPermissionLevel.EXECUTE,
                scope=FFPermissionScope.GLOBAL,
                granted_by="system",
                granted_at=datetime.now(),
                conditions={"allowed_commands": self.security_config.allowed_commands}
            )
        ]
        
        self.global_permissions.extend(default_perms)
    
    def _initialize_default_roles(self):
        """Initialize default roles and their permissions"""
        # Guest role - minimal permissions
        guest_permissions = [
            FFPermission(
                resource_type=FFResourceType.FILE_SYSTEM,
                resource_identifier="/tmp/guest/*",
                permission_level=FFPermissionLevel.READ,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            ),
            FFPermission(
                resource_type=FFResourceType.TOOLS,
                resource_identifier="web_search",
                permission_level=FFPermissionLevel.EXECUTE,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            )
        ]
        self.role_permissions["guest"] = guest_permissions
        
        # User role - standard permissions
        user_permissions = [
            FFPermission(
                resource_type=FFResourceType.FILE_SYSTEM,
                resource_identifier="/tmp/user/*",
                permission_level=FFPermissionLevel.WRITE,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            ),
            FFPermission(
                resource_type=FFResourceType.TOOLS,
                resource_identifier="*",
                permission_level=FFPermissionLevel.EXECUTE,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now(),
                conditions={"security_level": "restricted"}
            ),
            FFPermission(
                resource_type=FFResourceType.NETWORK,
                resource_identifier="*",
                permission_level=FFPermissionLevel.READ,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now(),
                conditions={"allowed_domains": self.security_config.allowed_domains}
            )
        ]
        self.role_permissions["user"] = user_permissions
        
        # Admin role - elevated permissions
        admin_permissions = [
            FFPermission(
                resource_type=FFResourceType.FILE_SYSTEM,
                resource_identifier="*",
                permission_level=FFPermissionLevel.FULL,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            ),
            FFPermission(
                resource_type=FFResourceType.TOOLS,
                resource_identifier="*",
                permission_level=FFPermissionLevel.FULL,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            ),
            FFPermission(
                resource_type=FFResourceType.SYSTEM_COMMANDS,
                resource_identifier="*",
                permission_level=FFPermissionLevel.EXECUTE,
                scope=FFPermissionScope.USER,
                granted_by="system",
                granted_at=datetime.now()
            )
        ]
        self.role_permissions["admin"] = admin_permissions
    
    def check_permission(self,
                        user_id: str,
                        resource_type: FFResourceType,
                        resource_identifier: str,
                        requested_level: FFPermissionLevel,
                        context: Optional[Dict[str, Any]] = None) -> FFPermissionResult:
        """
        Check if user has permission to access a resource.
        
        Args:
            user_id: User requesting access
            resource_type: Type of resource
            resource_identifier: Specific resource identifier
            requested_level: Level of access requested
            context: Additional context for permission check
            
        Returns:
            FFPermissionResult: Permission check result
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(user_id, resource_type, resource_identifier, requested_level)
            
            # Check cache first
            cached_result = self._get_cached_permission(cache_key)
            if cached_result:
                return cached_result
            
            # Perform permission check
            result = self._evaluate_permissions(user_id, resource_type, resource_identifier, requested_level, context)
            
            # Cache result
            self._cache_permission_result(cache_key, result)
            
            # Log permission check
            self._log_permission_check(user_id, resource_type, resource_identifier, requested_level, result, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return FFPermissionResult(
                granted=False,
                permission_level=FFPermissionLevel.NONE,
                reason=f"Permission check error: {str(e)}",
                audit_info={"error": str(e)}
            )
    
    def _evaluate_permissions(self,
                            user_id: str,
                            resource_type: FFResourceType,
                            resource_identifier: str,
                            requested_level: FFPermissionLevel,
                            context: Optional[Dict[str, Any]]) -> FFPermissionResult:
        """Evaluate permissions from all sources"""
        highest_level = FFPermissionLevel.NONE
        granted_by = None
        conditions = {}
        expires_at = None
        
        # Check global permissions
        for perm in self.global_permissions:
            if self._permission_matches(perm, resource_type, resource_identifier):
                if self._check_permission_conditions(perm, context):
                    if self._permission_level_sufficient(perm.permission_level, requested_level):
                        if self._permission_level_higher(perm.permission_level, highest_level):
                            highest_level = perm.permission_level
                            granted_by = f"global:{perm.granted_by}"
                            conditions.update(perm.conditions)
                            expires_at = perm.expires_at
        
        # Check user-specific permissions
        user_permissions = self.permissions.get(user_id, [])
        for perm in user_permissions:
            if self._permission_matches(perm, resource_type, resource_identifier):
                if self._check_permission_conditions(perm, context):
                    if self._permission_level_sufficient(perm.permission_level, requested_level):
                        if self._permission_level_higher(perm.permission_level, highest_level):
                            highest_level = perm.permission_level
                            granted_by = f"user:{perm.granted_by}"
                            conditions.update(perm.conditions)
                            expires_at = perm.expires_at
        
        # Check role-based permissions
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            role_permissions = self.role_permissions.get(role, [])
            for perm in role_permissions:
                if self._permission_matches(perm, resource_type, resource_identifier):
                    if self._check_permission_conditions(perm, context):
                        if self._permission_level_sufficient(perm.permission_level, requested_level):
                            if self._permission_level_higher(perm.permission_level, highest_level):
                                highest_level = perm.permission_level
                                granted_by = f"role:{role}:{perm.granted_by}"
                                conditions.update(perm.conditions)
                                expires_at = perm.expires_at
        
        # Determine final result
        granted = self._permission_level_sufficient(highest_level, requested_level)
        reason = f"Permission {'granted' if granted else 'denied'} by {granted_by}" if granted_by else "No matching permissions found"
        
        return FFPermissionResult(
            granted=granted,
            permission_level=highest_level,
            reason=reason,
            expires_at=expires_at,
            conditions=conditions,
            audit_info={
                "evaluated_by": "ff_permission_manager",
                "user_id": user_id,
                "resource": f"{resource_type.value}:{resource_identifier}",
                "granted_by": granted_by
            }
        )
    
    def _permission_matches(self,
                          permission: FFPermission,
                          resource_type: FFResourceType,
                          resource_identifier: str) -> bool:
        """Check if permission matches the requested resource"""
        # Check resource type
        if permission.resource_type != resource_type:
            return False
        
        # Check resource identifier with wildcard support
        perm_id = permission.resource_identifier
        
        # Exact match
        if perm_id == resource_identifier:
            return True
        
        # Wildcard match
        if perm_id.endswith("*"):
            prefix = perm_id[:-1]
            if resource_identifier.startswith(prefix):
                return True
        
        # Pattern matching for more complex cases
        if perm_id.startswith("pattern:"):
            import re
            pattern = perm_id[8:]  # Remove "pattern:" prefix
            if re.match(pattern, resource_identifier):
                return True
        
        return False
    
    def _check_permission_conditions(self,
                                   permission: FFPermission,
                                   context: Optional[Dict[str, Any]]) -> bool:
        """Check if permission conditions are met"""
        if not permission.conditions:
            return True
        
        if not context:
            context = {}
        
        # Check expiration
        if permission.expires_at and datetime.now() > permission.expires_at:
            return False
        
        # Check allowed domains condition
        if "allowed_domains" in permission.conditions:
            allowed_domains = permission.conditions["allowed_domains"]
            requested_domain = context.get("domain")
            if requested_domain and requested_domain not in allowed_domains:
                return False
        
        # Check security level condition
        if "security_level" in permission.conditions:
            required_level = permission.conditions["security_level"]
            current_level = context.get("security_level", "restricted")
            security_levels = ["read_only", "restricted", "sandboxed", "trusted"]
            if security_levels.index(current_level) < security_levels.index(required_level):
                return False
        
        # Check time-based conditions
        if "time_restrictions" in permission.conditions:
            time_restrictions = permission.conditions["time_restrictions"]
            current_hour = datetime.now().hour
            allowed_hours = time_restrictions.get("allowed_hours", [])
            if allowed_hours and current_hour not in allowed_hours:
                return False
        
        # Check usage limits
        if "usage_limit" in permission.conditions:
            # This would require tracking usage - simplified for now
            pass
        
        return True
    
    def _permission_level_sufficient(self, granted_level: FFPermissionLevel, requested_level: FFPermissionLevel) -> bool:
        """Check if granted permission level is sufficient for request"""
        level_hierarchy = {
            FFPermissionLevel.NONE: 0,
            FFPermissionLevel.READ: 1,
            FFPermissionLevel.WRITE: 2,
            FFPermissionLevel.EXECUTE: 3,
            FFPermissionLevel.ADMIN: 4,
            FFPermissionLevel.FULL: 5
        }
        
        return level_hierarchy[granted_level] >= level_hierarchy[requested_level]
    
    def _permission_level_higher(self, level1: FFPermissionLevel, level2: FFPermissionLevel) -> bool:
        """Check if level1 is higher than level2"""
        level_hierarchy = {
            FFPermissionLevel.NONE: 0,
            FFPermissionLevel.READ: 1,
            FFPermissionLevel.WRITE: 2,
            FFPermissionLevel.EXECUTE: 3,
            FFPermissionLevel.ADMIN: 4,
            FFPermissionLevel.FULL: 5
        }
        
        return level_hierarchy[level1] > level_hierarchy[level2]
    
    def grant_permission(self,
                        user_id: str,
                        permission: FFPermission,
                        granted_by: str) -> bool:
        """
        Grant a permission to a user.
        
        Args:
            user_id: User to grant permission to
            permission: Permission to grant
            granted_by: Who is granting the permission
            
        Returns:
            bool: True if permission granted successfully
        """
        try:
            # Update permission metadata
            permission.granted_by = granted_by
            permission.granted_at = datetime.now()
            
            # Add to user permissions
            if user_id not in self.permissions:
                self.permissions[user_id] = []
            
            self.permissions[user_id].append(permission)
            
            # Clear cache for this user
            self._clear_user_cache(user_id)
            
            # Log permission grant
            self._log_permission_grant(user_id, permission, granted_by)
            
            self.logger.info(f"Granted permission to {user_id}: {permission.resource_type.value}:{permission.resource_identifier}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to grant permission: {e}")
            return False
    
    def revoke_permission(self,
                         user_id: str,
                         resource_type: FFResourceType,
                         resource_identifier: str,
                         revoked_by: str) -> bool:
        """
        Revoke a permission from a user.
        
        Args:
            user_id: User to revoke permission from
            resource_type: Type of resource
            resource_identifier: Specific resource identifier
            revoked_by: Who is revoking the permission
            
        Returns:
            bool: True if permission revoked successfully
        """
        try:
            if user_id not in self.permissions:
                return False
            
            # Find and remove matching permissions
            original_count = len(self.permissions[user_id])
            self.permissions[user_id] = [
                perm for perm in self.permissions[user_id]
                if not (perm.resource_type == resource_type and perm.resource_identifier == resource_identifier)
            ]
            
            removed_count = original_count - len(self.permissions[user_id])
            
            if removed_count > 0:
                # Clear cache for this user
                self._clear_user_cache(user_id)
                
                # Log permission revocation
                self._log_permission_revocation(user_id, resource_type, resource_identifier, revoked_by)
                
                self.logger.info(f"Revoked {removed_count} permissions from {user_id}: {resource_type.value}:{resource_identifier}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to revoke permission: {e}")
            return False
    
    def assign_role(self, user_id: str, role: str, assigned_by: str) -> bool:
        """Assign a role to a user"""
        try:
            if role not in self.role_permissions:
                self.logger.warning(f"Role {role} does not exist")
                return False
            
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role)
            
            # Clear cache for this user
            self._clear_user_cache(user_id)
            
            # Log role assignment
            self._log_role_assignment(user_id, role, assigned_by)
            
            self.logger.info(f"Assigned role {role} to user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to assign role: {e}")
            return False
    
    def remove_role(self, user_id: str, role: str, removed_by: str) -> bool:
        """Remove a role from a user"""
        try:
            if user_id not in self.user_roles:
                return False
            
            if role in self.user_roles[user_id]:
                self.user_roles[user_id].remove(role)
                
                # Clear cache for this user
                self._clear_user_cache(user_id)
                
                # Log role removal
                self._log_role_removal(user_id, role, removed_by)
                
                self.logger.info(f"Removed role {role} from user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove role: {e}")
            return False
    
    def _generate_cache_key(self, user_id: str, resource_type: FFResourceType, resource_identifier: str, level: FFPermissionLevel) -> str:
        """Generate cache key for permission result"""
        key_data = f"{user_id}:{resource_type.value}:{resource_identifier}:{level.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_permission(self, cache_key: str) -> Optional[FFPermissionResult]:
        """Get cached permission result if still valid"""
        if cache_key not in self.permission_cache:
            return None
        
        result, cached_at = self.permission_cache[cache_key]
        
        # Check if cache entry is still valid
        if time.time() - cached_at > self.cache_ttl_seconds:
            del self.permission_cache[cache_key]
            return None
        
        return result
    
    def _cache_permission_result(self, cache_key: str, result: FFPermissionResult) -> None:
        """Cache permission result"""
        self.permission_cache[cache_key] = (result, time.time())
    
    def _clear_user_cache(self, user_id: str) -> None:
        """Clear cached permissions for a user"""
        keys_to_remove = []
        for cache_key in self.permission_cache:
            # Cache keys start with user_id hash
            user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
            if cache_key.startswith(user_hash):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.permission_cache[key]
    
    def _log_permission_check(self, user_id: str, resource_type: FFResourceType, resource_identifier: str, 
                            level: FFPermissionLevel, result: FFPermissionResult, context: Optional[Dict[str, Any]]) -> None:
        """Log permission check for audit"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "permission_check",
            "user_id": user_id,
            "resource_type": resource_type.value,
            "resource_identifier": resource_identifier,
            "requested_level": level.value,
            "granted": result.granted,
            "granted_level": result.permission_level.value,
            "reason": result.reason,
            "context": context
        }
        
        self.permission_audit_log.append(audit_entry)
        
        # Keep audit log size manageable
        if len(self.permission_audit_log) > 10000:
            self.permission_audit_log = self.permission_audit_log[-5000:]
    
    def _log_permission_grant(self, user_id: str, permission: FFPermission, granted_by: str) -> None:
        """Log permission grant for audit"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "permission_grant",
            "user_id": user_id,
            "resource_type": permission.resource_type.value,
            "resource_identifier": permission.resource_identifier,
            "permission_level": permission.permission_level.value,
            "granted_by": granted_by,
            "scope": permission.scope.value
        }
        
        self.permission_audit_log.append(audit_entry)
    
    def _log_permission_revocation(self, user_id: str, resource_type: FFResourceType, resource_identifier: str, revoked_by: str) -> None:
        """Log permission revocation for audit"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "permission_revoke",
            "user_id": user_id,
            "resource_type": resource_type.value,
            "resource_identifier": resource_identifier,
            "revoked_by": revoked_by
        }
        
        self.permission_audit_log.append(audit_entry)
    
    def _log_role_assignment(self, user_id: str, role: str, assigned_by: str) -> None:
        """Log role assignment for audit"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "role_assign",
            "user_id": user_id,
            "role": role,
            "assigned_by": assigned_by
        }
        
        self.permission_audit_log.append(audit_entry)
    
    def _log_role_removal(self, user_id: str, role: str, removed_by: str) -> None:
        """Log role removal for audit"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "role_remove",
            "user_id": user_id,
            "role": role,
            "removed_by": removed_by
        }
        
        self.permission_audit_log.append(audit_entry)
    
    def get_user_permissions(self, user_id: str) -> List[FFPermission]:
        """Get all permissions for a user (direct + role-based)"""
        all_permissions = []
        
        # Direct permissions
        all_permissions.extend(self.permissions.get(user_id, []))
        
        # Role-based permissions
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            all_permissions.extend(self.role_permissions.get(role, []))
        
        return all_permissions
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles assigned to a user"""
        return self.user_roles.get(user_id, set())
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self.permission_audit_log[-limit:] if self.permission_audit_log else []
    
    def cleanup_expired_permissions(self) -> int:
        """Remove expired permissions and return count of removed permissions"""
        removed_count = 0
        current_time = datetime.now()
        
        # Clean user permissions
        for user_id in self.permissions:
            original_count = len(self.permissions[user_id])
            self.permissions[user_id] = [
                perm for perm in self.permissions[user_id]
                if not (perm.expires_at and current_time > perm.expires_at)
            ]
            removed_count += original_count - len(self.permissions[user_id])
        
        # Clean role permissions
        for role in self.role_permissions:
            original_count = len(self.role_permissions[role])
            self.role_permissions[role] = [
                perm for perm in self.role_permissions[role]
                if not (perm.expires_at and current_time > perm.expires_at)
            ]
            removed_count += original_count - len(self.role_permissions[role])
        
        # Clean global permissions
        original_count = len(self.global_permissions)
        self.global_permissions = [
            perm for perm in self.global_permissions
            if not (perm.expires_at and current_time > perm.expires_at)
        ]
        removed_count += original_count - len(self.global_permissions)
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} expired permissions")
        
        return removed_count