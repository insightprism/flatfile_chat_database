"""
FF Chat API Routes Package

Contains all API route modules for the FF Chat system.
"""

from .ff_chat_routes import router as chat_router
from .ff_session_routes import router as session_router
from .ff_user_routes import router as user_router
from .ff_admin_routes import router as admin_router
from .ff_health_routes import router as health_router

__all__ = [
    "chat_router",
    "session_router", 
    "user_router",
    "admin_router",
    "health_router"
]