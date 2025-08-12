"""
User Context Manager for Flatfile Database.

This module provides authoritative user identification for the flatfile database system.
It determines the actual system user and prevents personas from being treated as users.

The flatfile database is the sole authority on user identification - the chat app
should query this to know who the current user is, not provide user identification.
"""

import os
import getpass
from typing import Optional, Dict, Any
from pathlib import Path
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)


class FFUserContextManager:
    """
    Manages user authentication and context for the flatfile database.
    
    This is the authoritative source for user identification. The chat application
    should query this manager to determine the current user, rather than attempting
    to specify users itself.
    """
    
    # Default user for testing and development
    DEFAULT_USER = "test123"
    
    # Known persona prefixes that should be rejected as users
    PERSONA_PREFIXES = ["persona_", "agent_", "ai_", "bot_"]
    
    def __init__(self, override_user: Optional[str] = None):
        """
        Initialize the user context manager.
        
        Args:
            override_user: Optional user override for testing purposes
        """
        self._override_user = override_user
        self._cached_user = None
        logger.info(f"FFUserContextManager initialized with override: {override_user}")
    
    def get_current_user(self) -> str:
        """
        Get the current authenticated user.
        
        This is the authoritative method for determining who the current user is.
        The chat application should call this to know the user, not provide it.
        
        Returns:
            The authenticated user identifier
        """
        # Use cached value if available
        if self._cached_user:
            return self._cached_user
        
        # Check for override (testing purposes)
        if self._override_user:
            self._cached_user = self._override_user
            logger.info(f"Using override user: {self._cached_user}")
            return self._cached_user
        
        # Try to get system user
        user = None
        
        # Method 1: Try os.getlogin() - most reliable on Unix
        try:
            user = os.getlogin()
            if user:
                logger.info(f"Got user from os.getlogin(): {user}")
        except (OSError, AttributeError):
            pass
        
        # Method 2: Try getpass.getuser() - works on more systems
        if not user:
            try:
                user = getpass.getuser()
                if user:
                    logger.info(f"Got user from getpass.getuser(): {user}")
            except:
                pass
        
        # Method 3: Check environment variables
        if not user:
            user = os.environ.get('USER') or os.environ.get('USERNAME')
            if user:
                logger.info(f"Got user from environment: {user}")
        
        # Method 4: Use default user
        if not user:
            user = self.DEFAULT_USER
            logger.info(f"Using default user: {user}")
        
        # Cache the result
        self._cached_user = user
        return user
    
    def is_valid_user(self, user_id: str) -> bool:
        """
        Check if a given user_id is a valid user (not a persona).
        
        Args:
            user_id: The user identifier to validate
            
        Returns:
            True if this is a valid user, False if it's a persona
        """
        if not user_id:
            return False
        
        # Check if it starts with known persona prefixes
        user_lower = user_id.lower()
        for prefix in self.PERSONA_PREFIXES:
            if user_lower.startswith(prefix):
                logger.info(f"Rejected persona as user: {user_id}")
                return False
        
        # Check if it's a known persona name (without prefix)
        known_personas = [
            "alex_patterson", "betty_rodriguez", "bob_wilson",
            "creative_writer", "technical_expert", "helpful_assistant",
            "business_analyst", "jake_thompson", "john_smith",
            "marcus_chen", "maria_garcia", "sarah_johnson"
        ]
        
        if user_id.lower() in known_personas:
            logger.warning(f"Rejected known persona name as user: {user_id}")
            return False
        
        return True
    
    def validate_and_get_user(self, provided_user: Optional[str] = None) -> str:
        """
        Validate a provided user or get the current user.
        
        If a user is provided, validate it's not a persona.
        If no user is provided or validation fails, return the current authenticated user.
        
        Args:
            provided_user: Optional user identifier provided by caller
            
        Returns:
            A valid user identifier (never a persona)
        """
        # If a user was provided, validate it
        if provided_user:
            if self.is_valid_user(provided_user):
                logger.info(f"Validated provided user: {provided_user}")
                return provided_user
            else:
                logger.info(f"Invalid user provided: {provided_user}, using authenticated user instead")
        
        # Return the authenticated user
        return self.get_current_user()
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current user.
        
        Returns:
            Dictionary containing user information
        """
        current_user = self.get_current_user()
        
        return {
            "user_id": current_user,
            "is_default": current_user == self.DEFAULT_USER,
            "is_override": self._override_user is not None,
            "authentication_method": self._get_auth_method(),
            "is_authenticated": current_user != self.DEFAULT_USER
        }
    
    def _get_auth_method(self) -> str:
        """Get the method used for authentication."""
        if self._override_user:
            return "override"
        elif self._cached_user == self.DEFAULT_USER:
            return "default"
        else:
            return "system"
    
    def set_override_user(self, user: Optional[str] = None):
        """
        Set or clear the override user (for testing).
        
        Args:
            user: User to override with, or None to clear override
        """
        if user and not self.is_valid_user(user):
            raise ValueError(f"Cannot set override to persona: {user}")
        
        self._override_user = user
        self._cached_user = None  # Clear cache to force re-evaluation
        logger.info(f"Override user set to: {user}")
    
    def clear_cache(self):
        """Clear the cached user to force re-evaluation."""
        self._cached_user = None
        logger.info("User cache cleared")


# Global instance for easy access
_global_user_context = None

def get_user_context() -> FFUserContextManager:
    """
    Get the global user context manager instance.
    
    Returns:
        The global FFUserContextManager instance
    """
    global _global_user_context
    if _global_user_context is None:
        _global_user_context = FFUserContextManager()
    return _global_user_context

def get_current_user() -> str:
    """
    Convenience function to get the current user.
    
    Returns:
        The current authenticated user identifier
    """
    return get_user_context().get_current_user()