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

from .ff_prismmind_config import (
    FFPrismMindConfigDTO,
    FFDocumentProcessingConfigDTO,
    FFEngineSelectionConfigDTO,
    FFHandlerStrategiesConfigDTO,
    FFIntegrationConfigDTO
)

from .ff_prismmind_storage_handlers import ff_store_vectors_handler_async

from .ff_prismmind_engine_factory import FFPrismMindConfigFactory

from .ff_prismmind_document_processor import FFDocumentProcessor

from .ff_prismmind_config_loader import FFPrismMindConfigLoader

__all__ = [
    'FFPrismMindConfigDTO',
    'FFDocumentProcessingConfigDTO', 
    'FFEngineSelectionConfigDTO',
    'FFHandlerStrategiesConfigDTO',
    'FFIntegrationConfigDTO',
    'ff_store_vectors_handler_async',
    'FFPrismMindConfigFactory',
    'FFDocumentProcessor',
    'FFPrismMindConfigLoader'
]

__version__ = "1.0.0"