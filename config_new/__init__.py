"""
Configuration package for flatfile chat database.

This package provides domain-specific configuration classes following
clean architecture principles with clear separation of concerns.
"""

from .base import BaseConfig
from .storage import StorageConfig
from .search import SearchConfig
from .vector import VectorConfig
from .document import DocumentConfig
from .locking import LockingConfig
from .panel import PanelConfig
from .manager import ConfigurationManager

__all__ = [
    'BaseConfig',
    'StorageConfig',
    'SearchConfig',
    'VectorConfig',
    'DocumentConfig',
    'LockingConfig',
    'PanelConfig',
    'ConfigurationManager'
]