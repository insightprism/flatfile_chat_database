"""
Flatfile-PrismMind Integration Module

This module provides seamless integration between PrismMind's proven engine
architecture and flatfile chat database storage system.

Key Components:
- Configuration classes for complete system setup
- Handlers for bridging PrismMind to flatfile storage  
- Document processor using PrismMind engines
- Configuration loading and factory systems

Philosophy: Maximum PrismMind reuse, minimal custom code, configuration-driven.
"""

from flatfile_chat_database.prismmind_integration.config import (
    FlatfilePrismMindConfig,
    FlatfileDocumentProcessingConfig,
    FlatfileEngineSelectionConfig,
    FlatfileHandlerStrategiesConfig,
    FlatfileIntegrationConfig
)

from flatfile_chat_database.prismmind_integration.handlers import ff_store_vectors_handler_async

from flatfile_chat_database.prismmind_integration.factory import FlatfilePrismMindConfigFactory

from flatfile_chat_database.prismmind_integration.processor import FlatfileDocumentProcessor

from flatfile_chat_database.prismmind_integration.loader import FlatfilePrismMindConfigLoader

__all__ = [
    'FlatfilePrismMindConfig',
    'FlatfileDocumentProcessingConfig', 
    'FlatfileEngineSelectionConfig',
    'FlatfileHandlerStrategiesConfig',
    'FlatfileIntegrationConfig',
    'ff_store_vectors_handler_async',
    'FlatfilePrismMindConfigFactory',
    'FlatfileDocumentProcessor',
    'FlatfilePrismMindConfigLoader'
]

__version__ = "1.0.0"