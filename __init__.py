"""
Flatfile Chat Database System

A simple, modular file-based storage solution for AI chat applications.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from .config import StorageConfig, load_config
from .models import (
    Message, Session, Panel, SituationalContext, Document,
    UserProfile, Persona, PanelMessage, PanelInsight
)
from .storage import StorageManager
from .search import SearchQuery, SearchResult, AdvancedSearchEngine

__all__ = [
    # Main API
    "StorageManager",
    # Configuration
    "StorageConfig",
    "load_config",
    # Models
    "Message",
    "Session", 
    "Panel",
    "SituationalContext",
    "Document",
    "UserProfile",
    "Persona",
    "PanelMessage",
    "PanelInsight",
    # Search
    "SearchQuery",
    "SearchResult",
    "AdvancedSearchEngine"
]