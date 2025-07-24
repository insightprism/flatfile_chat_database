"""
Configuration classes for Flatfile-PrismMind integration.

All configuration is dataclass-based, following PrismMind's configuration-driven
philosophy. Zero hard-coding, everything configurable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from flatfile_chat_database.config import StorageConfig


@dataclass
class FlatfileDocumentProcessingConfig:
    """Configuration for document processing workflows"""
    
    # Processing chain definition (no hard-coding!)
    default_processing_chain: List[str] = field(default_factory=lambda: [
        "pm_injest_engine",
        "pm_nlp_engine",
        "pm_chunking_engine", 
        "pm_embedding_engine",
        "ff_storage_engine"
    ])
    
    # File type processing chains (configurable per file type)
    file_type_chains: Dict[str, List[str]] = field(default_factory=lambda: {
        "application/pdf": [
            "pm_injest_engine", 
            "pm_nlp_engine", 
            "pm_chunking_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ],
        "image/jpeg": [
            "pm_injest_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ],
        "image/png": [
            "pm_injest_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ],
        "text/plain": [
            "pm_injest_engine", 
            "pm_chunking_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ],
        "text/markdown": [
            "pm_injest_engine", 
            "pm_chunking_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ],
        "url": [
            "pm_injest_engine", 
            "pm_nlp_engine", 
            "pm_chunking_engine", 
            "pm_embedding_engine", 
            "ff_storage_engine"
        ]
    })
    
    # Environment-specific configurations
    environment_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "development": {
            "enable_tracing": True,
            "log_level": "DEBUG",
            "timeout_seconds": 60,
            "max_retries": 1
        },
        "staging": {
            "enable_tracing": True,
            "log_level": "INFO",
            "timeout_seconds": 45,
            "max_retries": 2
        },
        "production": {
            "enable_tracing": False,
            "log_level": "WARNING", 
            "timeout_seconds": 30,
            "max_retries": 3
        }
    })
    
    # Processing options
    skip_nlp_for_file_types: List[str] = field(default_factory=lambda: [
        "image/jpeg", "image/png", "image/bmp"
    ])
    
    enable_parallel_processing: bool = True
    max_concurrent_documents: int = 5


@dataclass
class FlatfileEngineSelectionConfig:
    """Configuration for engine and handler selection"""
    
    # File type to handler mapping (completely configurable)
    file_type_handlers: Dict[str, str] = field(default_factory=lambda: {
        "application/pdf": "pm_prism_pdf_handler_async",
        "text/plain": "pm_prism_text_handler_async",
        "text/markdown": "pm_prism_text_handler_async",
        "text/html": "pm_prism_text_handler_async",
        "image/jpeg": "pm_tesseract_ocr_reader_handler_async",
        "image/png": "pm_tesseract_ocr_reader_handler_async", 
        "image/bmp": "pm_tesseract_ocr_reader_handler_async",
        "url": "pm_playwright_web_image_handler_async"
    })
    
    # Fallback handlers for when primary handlers fail
    fallback_handlers: Dict[str, str] = field(default_factory=lambda: {
        "application/pdf": "pm_prism_text_handler_async",
        "image/jpeg": "pm_prism_text_handler_async",
        "image/png": "pm_prism_text_handler_async",
        "url": "pm_prism_text_handler_async"
    })
    
    # NLP handler selection
    nlp_handlers: Dict[str, str] = field(default_factory=lambda: {
        "basic": "pm_basic_text_cleaner_handler_async",
        "spacy": "pm_spacy_text_cleaner_handler_async",
        "advanced": "pm_advanced_nlp_handler_async"
    })
    
    # Chunking strategy handlers
    chunking_handlers: Dict[str, str] = field(default_factory=lambda: {
        "fixed": "pm_fixed_chunk_handler_async",
        "sentence": "pm_sentence_chunk_handler_async",
        "optimized_summary": "pm_optimize_chunk_handler_async",
        "semantic": "pm_semantic_chunk_handler_async"
    })
    
    # Embedding provider handlers  
    embedding_handlers: Dict[str, str] = field(default_factory=lambda: {
        "nomic-ai": "pm_embed_batch_handler_async",
        "openai": "pm_openai_embed_handler_async",
        "openai-3-small": "pm_openai_3_small_handler_async",
        "openai-3-large": "pm_openai_3_large_handler_async"
    })


@dataclass
class FlatfileHandlerStrategiesConfig:
    """Configuration for handler strategies and parameters"""
    
    # Default strategies (no magic strings!)
    default_strategies: Dict[str, str] = field(default_factory=lambda: {
        "nlp_strategy": "spacy",
        "chunking_strategy": "optimized_summary", 
        "embedding_provider": "nomic-ai"
    })
    
    # Strategy parameters (no magic numbers!)
    strategy_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "chunking": {
            "optimized_summary": {
                "chunk_size": 800,
                "chunk_overlap": 100,
                "sentence_per_chunk": 5,
                "sentence_overlap": 1,
                "sentence_buffer": 2,
                "max_tokens_per_chunk": 800,
                "min_tokens_per_chunk": 128,
                "chunk_overlap_sentences": 1
            },
            "fixed": {
                "chunk_size": 512,
                "chunk_overlap": 64
            },
            "sentence": {
                "sentence_per_chunk": 3,
                "sentence_overlap": 1,
                "sentence_buffer": 1
            },
            "semantic": {
                "similarity_threshold": 0.8,
                "min_chunk_size": 200,
                "max_chunk_size": 1000
            }
        },
        "embedding": {
            "nomic-ai": {
                "vector_dimension": 768,
                "normalize_vectors": True,
                "batch_size": 32,
                "device": "auto"
            },
            "openai": {
                "vector_dimension": 1536,
                "normalize_vectors": True,
                "batch_size": 16,
                "model_name": "text-embedding-ada-002"
            },
            "openai-3-small": {
                "vector_dimension": 1536,
                "normalize_vectors": True,
                "batch_size": 16,
                "model_name": "text-embedding-3-small"
            },
            "openai-3-large": {
                "vector_dimension": 3072,
                "normalize_vectors": True,
                "batch_size": 8,
                "model_name": "text-embedding-3-large"
            }
        },
        "nlp": {
            "spacy": {
                "model_name": "en_core_web_sm",
                "remove_stopwords": True,
                "remove_punctuation": False,
                "lowercase": True
            },
            "basic": {
                "remove_extra_whitespace": True,
                "remove_special_chars": False,
                "normalize_unicode": True
            }
        }
    })
    
    # File type specific parameters
    file_type_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "pdf": {
            "max_pages": None,
            "clean_text": True,
            "extract_images": False,
            "preserve_layout": False
        },
        "image": {
            "ocr_language": "eng",
            "confidence_threshold": 0.6,
            "preprocessing": True,
            "deskew": True,
            "denoise": True
        },
        "web": {
            "wait_for_load": True,
            "timeout_ms": 30000,
            "extract_images": False,
            "follow_redirects": True,
            "user_agent": "FlatfileBot/1.0"
        },
        "text": {
            "encoding": "utf-8",
            "detect_encoding": True,
            "normalize_line_endings": True
        }
    })


@dataclass  
class FlatfileIntegrationConfig:
    """Configuration for PrismMind-Flatfile integration"""
    
    # Storage integration settings
    storage_integration: Dict[str, Any] = field(default_factory=lambda: {
        "auto_store_vectors": True,
        "store_intermediate_results": False,
        "cleanup_temp_files": True,
        "backup_before_overwrite": True,
        "compress_vectors": False
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        "concurrent_documents": 5,
        "vector_batch_size": 50,
        "memory_limit_mb": 1024,
        "timeout_per_document_seconds": 300,
        "enable_caching": True,
        "cache_ttl_seconds": 3600
    })
    
    # Error handling configuration
    error_handling: Dict[str, Any] = field(default_factory=lambda: {
        "retry_attempts": 3,
        "retry_delay_seconds": 1,
        "exponential_backoff": True,
        "continue_on_error": True,
        "log_errors": True,
        "raise_on_critical_error": True
    })
    
    # Monitoring and observability
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "enable_metrics": True,
        "enable_tracing": True,
        "trace_sampling_rate": 1.0,
        "metrics_interval_seconds": 60,
        "log_performance": True
    })


@dataclass
class FlatfilePrismMindConfig:
    """Master configuration combining flatfile and PrismMind settings"""
    
    # Flatfile storage configuration
    flatfile_config: StorageConfig
    
    # Document processing configuration
    document_processing: FlatfileDocumentProcessingConfig = field(
        default_factory=FlatfileDocumentProcessingConfig
    )
    
    # Engine selection configuration
    engine_selection: FlatfileEngineSelectionConfig = field(
        default_factory=FlatfileEngineSelectionConfig
    )
    
    # Handler strategy configuration
    handler_strategies: FlatfileHandlerStrategiesConfig = field(
        default_factory=FlatfileHandlerStrategiesConfig
    )
    
    # Integration configuration
    integration_settings: FlatfileIntegrationConfig = field(
        default_factory=FlatfileIntegrationConfig
    )
    
    # Environment settings
    environment: str = "development"
    
    def get_current_environment_config(self) -> Dict[str, Any]:
        """Get configuration for current environment"""
        return self.document_processing.environment_configs.get(
            self.environment, 
            self.document_processing.environment_configs["development"]
        )
    
    def get_handler_for_file_type(self, file_type: str) -> str:
        """Get appropriate handler for file type with fallback"""
        handler = self.engine_selection.file_type_handlers.get(file_type)
        if not handler:
            # Try fallback handler
            handler = self.engine_selection.fallback_handlers.get(file_type)
        if not handler:
            # Default fallback
            handler = "pm_prism_text_handler_async"
        return handler
    
    def get_processing_chain_for_file_type(self, file_type: str) -> List[str]:
        """Get processing chain for specific file type"""
        return self.document_processing.file_type_chains.get(
            file_type, 
            self.document_processing.default_processing_chain
        )
    
    def get_strategy_parameters(self, strategy_type: str, strategy_name: str) -> Dict[str, Any]:
        """Get parameters for specific strategy"""
        return self.handler_strategies.strategy_parameters.get(strategy_type, {}).get(
            strategy_name, {}
        )
    
    def get_file_type_parameters(self, file_type: str) -> Dict[str, Any]:
        """Get parameters for specific file type"""
        # Extract base type (e.g., "pdf" from "application/pdf")
        base_type = file_type.split('/')[-1] if '/' in file_type else file_type
        return self.handler_strategies.file_type_parameters.get(base_type, {})