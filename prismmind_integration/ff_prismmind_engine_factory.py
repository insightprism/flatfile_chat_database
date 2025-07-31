"""
Configuration factory for creating PrismMind configurations with flatfile integration.

This factory transforms flatfile configurations into PrismMind-compatible
configurations while maintaining the configuration-driven philosophy.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import PrismMind configuration types if available
try:
    from pm_config.pm_injest_engine_config import (
        pm_injest_engine_config_dto, 
        pm_injest_handler_config_dto
    )
    from pm_config.pm_nlp_engine_config import (
        pm_nlp_engine_config_dto
    )
    from pm_config.pm_chunking_engine_config import (
        pm_chunking_engine_config_dto, 
        pm_chunking_handler_config_dto,
        pm_get_chunking_engine_settings
    )
    from pm_config.pm_embedding_engine_config import (
        pm_embedding_engine_config_dto, 
        pm_embedding_handler_config_dto
    )
    from pm_engines.pm_injest_engine import PmInjestEngine
    from pm_engines.pm_nlp_engine import PmNLPEngine
    from pm_engines.pm_chunking_engine import PmChunkingEngine
    from pm_engines.pm_embedding_engine import PmEmbeddingEngine
    from pm_engines.pm_base_engine import PmBaseEngine
    PRISMMIND_AVAILABLE = True
except ImportError:
    # Fallback classes if PrismMind not available
    class MockDto:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    pm_injest_engine_config_dto = MockDto
    pm_injest_handler_config_dto = MockDto
    pm_nlp_engine_config_dto = MockDto
    pm_chunking_engine_config_dto = MockDto
    pm_chunking_handler_config_dto = MockDto
    pm_embedding_engine_config_dto = MockDto
    pm_embedding_handler_config_dto = MockDto
    PmInjestEngine = object
    PmNLPEngine = object
    PmChunkingEngine = object
    PmEmbeddingEngine = object
    PmBaseEngine = object
    
    def pm_get_chunking_engine_settings():
        return {"PM_CHUNKING_STRATEGY_MAP": {}}
    
    PRISMMIND_AVAILABLE = False

from .ff_prismmind_config import FFPrismMindConfig


class FFPrismMindConfigFactory:
    """Factory for creating PrismMind configurations with flatfile integration"""
    
    @staticmethod
    def create_injest_engine_config(
        file_type: str,
        config: FFPrismMindConfig
    ) -> Tuple[pm_injest_engine_config_dto, pm_injest_handler_config_dto]:
        """
        Create PrismMind injest engine config from flatfile config.
        
        Args:
            file_type: MIME type of the file to process
            config: Flatfile-PrismMind configuration
            
        Returns:
            Tuple of (engine_config, handler_config)
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create configurations.")
        
        # Get handler for file type from configuration
        handler_name = config.get_handler_for_file_type(file_type)
        
        # Get file type specific parameters
        file_params = config.get_file_type_parameters(file_type)
        
        # Create engine configuration
        engine_config = pm_injest_engine_config_dto(
            handler_name=handler_name
        )
        
        # Create handler configuration with file-specific parameters
        handler_config = pm_injest_handler_config_dto(**file_params)
        
        return engine_config, handler_config
    
    @staticmethod
    def create_nlp_engine_config(
        config: FFPrismMindConfig,
        strategy_override: Optional[str] = None
    ) -> pm_nlp_engine_config_dto:
        """
        Create PrismMind NLP engine config from flatfile config.
        
        Args:
            config: Flatfile-PrismMind configuration
            strategy_override: Optional strategy override
            
        Returns:
            NLP engine configuration
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create configurations.")
        
        # Get NLP strategy from config or override
        nlp_strategy = strategy_override or config.handler_strategies.default_strategies["nlp_strategy"]
        
        # Get handler name for strategy
        handler_name = config.engine_selection.nlp_handlers.get(nlp_strategy)
        
        if not handler_name:
            raise ValueError(f"Unknown NLP strategy: {nlp_strategy}")
        
        # Get strategy parameters
        strategy_params = config.get_strategy_parameters("nlp", nlp_strategy)
        
        return pm_nlp_engine_config_dto(
            handler_name=handler_name,
            **strategy_params
        )
    
    @staticmethod
    def create_chunking_engine_config(
        config: FFPrismMindConfig,
        strategy_override: Optional[str] = None
    ) -> Tuple[pm_chunking_engine_config_dto, pm_chunking_handler_config_dto]:
        """
        Create PrismMind chunking engine config from flatfile config.
        
        Args:
            config: Flatfile-PrismMind configuration
            strategy_override: Optional strategy override
            
        Returns:
            Tuple of (engine_config, handler_config)
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create configurations.")
        
        # Get chunking strategy from config or override
        chunking_strategy = strategy_override or config.handler_strategies.default_strategies["chunking_strategy"]
        
        # Get handler name for strategy
        handler_name = config.engine_selection.chunking_handlers.get(chunking_strategy)
        
        if not handler_name:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        # Get strategy parameters
        strategy_params = config.get_strategy_parameters("chunking", chunking_strategy)
        
        # Create engine configuration
        engine_config = pm_chunking_engine_config_dto(
            handler_name=handler_name
        )
        
        # Create handler configuration with strategy parameters
        handler_config = pm_chunking_handler_config_dto(**strategy_params)
        
        # Set the chunking strategy in handler config
        handler_config.defined_chunking_strategy = chunking_strategy
        
        return engine_config, handler_config
    
    @staticmethod  
    def create_embedding_engine_config_with_storage(
        embedding_provider: str,
        storage_metadata: Dict[str, Any],
        config: FFPrismMindConfig,
        provider_override: Optional[str] = None
    ) -> Tuple[pm_embedding_engine_config_dto, pm_embedding_handler_config_dto]:
        """
        Create embedding config that automatically stores in flatfile.
        
        Args:
            embedding_provider: Embedding provider to use
            storage_metadata: Metadata for flatfile storage (user_id, session_id, etc.)
            config: Flatfile-PrismMind configuration
            provider_override: Optional provider override
            
        Returns:
            Tuple of (engine_config, handler_config)
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create configurations.")
        
        # Use override or default provider
        provider = provider_override or embedding_provider
        
        # Get embedding parameters
        embedding_params = config.get_strategy_parameters("embedding", provider)
        
        # Create engine configuration with storage handler
        engine_config = pm_embedding_engine_config_dto(
            embedding_provider=provider,
            handler_name="ff_store_vectors_handler_async",  # Our custom storage handler
            **{k: v for k, v in embedding_params.items() if k in ['vector_dimension', 'model_name']}
        )
        
        # Create handler configuration with embedding and storage parameters
        handler_config = pm_embedding_handler_config_dto(
            vector_normalized_flag=embedding_params.get("normalize_vectors", True),
            return_only_vector_flag=False,
            metadata={
                **storage_metadata,
                "flatfile_config": config.flatfile_config,
                "embedding_provider": provider,
                "embedding_params": embedding_params
            }
        )
        
        return engine_config, handler_config
    
    @staticmethod
    def create_standard_embedding_engine_config(
        embedding_provider: str,
        config: FFPrismMindConfig,
        provider_override: Optional[str] = None
    ) -> Tuple[pm_embedding_engine_config_dto, pm_embedding_handler_config_dto]:
        """
        Create standard embedding config without flatfile storage.
        
        Args:
            embedding_provider: Embedding provider to use
            config: Flatfile-PrismMind configuration
            provider_override: Optional provider override
            
        Returns:
            Tuple of (engine_config, handler_config)
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create configurations.")
        
        # Use override or default provider
        provider = provider_override or embedding_provider
        
        # Get handler name for provider
        handler_name = config.engine_selection.embedding_handlers.get(provider)
        
        if not handler_name:
            raise ValueError(f"Unknown embedding provider: {provider}")
        
        # Get embedding parameters
        embedding_params = config.get_strategy_parameters("embedding", provider)
        
        # Create engine configuration
        engine_config = pm_embedding_engine_config_dto(
            embedding_provider=provider,
            handler_name=handler_name,
            **{k: v for k, v in embedding_params.items() if k in ['vector_dimension', 'model_name']}
        )
        
        # Create handler configuration
        handler_config = pm_embedding_handler_config_dto(
            vector_normalized_flag=embedding_params.get("normalize_vectors", True),
            return_only_vector_flag=False,
            metadata={
                "embedding_provider": provider,
                "embedding_params": embedding_params
            }
        )
        
        return engine_config, handler_config
    
    @staticmethod
    def create_complete_processing_chain(
        file_type: str,
        user_id: str,
        session_id: str,
        document_id: str,
        config: FFPrismMindConfig,
        strategy_overrides: Optional[Dict[str, str]] = None
    ) -> List[PmBaseEngine]:
        """
        Create complete engine chain from configuration.
        
        Args:
            file_type: MIME type of the file
            user_id: User identifier
            session_id: Session identifier
            document_id: Document identifier
            config: Flatfile-PrismMind configuration
            strategy_overrides: Optional strategy overrides
            
        Returns:
            List of configured PrismMind engines
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create engines.")
        
        engines = []
        overrides = strategy_overrides or {}
        
        # Get processing chain for file type
        processing_chain = config.get_processing_chain_for_file_type(file_type)
        
        # Storage metadata for final storage
        storage_metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "document_id": document_id
        }
        
        for engine_name in processing_chain:
            if engine_name == "pm_injest_engine":
                # Create injest engine
                engine_config, handler_config = FFPrismMindConfigFactory.create_injest_engine_config(
                    file_type, config
                )
                
                engine = PmInjestEngine(
                    engine_config=engine_config,
                    handler_config=handler_config
                )
                engines.append(engine)
                
            elif engine_name == "pm_nlp_engine":
                # Skip NLP for certain file types
                if file_type in config.document_processing.skip_nlp_for_file_types:
                    continue
                
                # Create NLP engine
                engine_config = FFPrismMindConfigFactory.create_nlp_engine_config(
                    config, overrides.get("nlp_strategy")
                )
                
                engine = PmNLPEngine(engine_config=engine_config)
                engines.append(engine)
                
            elif engine_name == "pm_chunking_engine":
                # Create chunking engine
                engine_config, handler_config = FFPrismMindConfigFactory.create_chunking_engine_config(
                    config, overrides.get("chunking_strategy")
                )
                
                engine = PmChunkingEngine(
                    engine_config=engine_config,
                    handler_config=handler_config
                )
                engines.append(engine)
                
            elif engine_name == "pm_embedding_engine":
                # Create standard embedding engine (no storage)
                embedding_provider = overrides.get("embedding_provider") or config.handler_strategies.default_strategies["embedding_provider"]
                
                engine_config, handler_config = FFPrismMindConfigFactory.create_standard_embedding_engine_config(
                    embedding_provider, config
                )
                
                engine = PmEmbeddingEngine(
                    engine_config=engine_config,
                    handler_config=handler_config
                )
                engines.append(engine)
                
            elif engine_name == "ff_storage_engine":
                # Create embedding engine with flatfile storage
                embedding_provider = overrides.get("embedding_provider") or config.handler_strategies.default_strategies["embedding_provider"]
                
                engine_config, handler_config = FFPrismMindConfigFactory.create_embedding_engine_config_with_storage(
                    embedding_provider, storage_metadata, config
                )
                
                engine = PmEmbeddingEngine(
                    engine_config=engine_config,
                    handler_config=handler_config
                )
                engines.append(engine)
        
        return engines
    
    @staticmethod
    def create_engine_by_name(
        engine_name: str,
        config: FFPrismMindConfig,
        **kwargs
    ) -> PmBaseEngine:
        """
        Create a single engine by name with configuration.
        
        Args:
            engine_name: Name of the engine to create
            config: Flatfile-PrismMind configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured PrismMind engine
        """
        if not PRISMMIND_AVAILABLE:
            raise RuntimeError("PrismMind not available. Cannot create engines.")
        
        if engine_name == "pm_injest_engine":
            file_type = kwargs.get("file_type", "text/plain")
            engine_config, handler_config = FFPrismMindConfigFactory.create_injest_engine_config(
                file_type, config
            )
            return PmInjestEngine(engine_config=engine_config, handler_config=handler_config)
            
        elif engine_name == "pm_nlp_engine":
            engine_config = FFPrismMindConfigFactory.create_nlp_engine_config(
                config, kwargs.get("strategy_override")
            )
            return PmNLPEngine(engine_config=engine_config)
            
        elif engine_name == "pm_chunking_engine":
            engine_config, handler_config = FFPrismMindConfigFactory.create_chunking_engine_config(
                config, kwargs.get("strategy_override")
            )
            return PmChunkingEngine(engine_config=engine_config, handler_config=handler_config)
            
        elif engine_name == "pm_embedding_engine":
            embedding_provider = kwargs.get("embedding_provider") or config.handler_strategies.default_strategies["embedding_provider"]
            engine_config, handler_config = FFPrismMindConfigFactory.create_standard_embedding_engine_config(
                embedding_provider, config
            )
            return PmEmbeddingEngine(engine_config=engine_config, handler_config=handler_config)
            
        else:
            raise ValueError(f"Unknown engine name: {engine_name}")
    
    @staticmethod
    def validate_configuration(config: FFPrismMindConfig) -> List[str]:
        """
        Validate configuration for completeness and consistency.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate required handlers exist
        for file_type, handler_name in config.engine_selection.file_type_handlers.items():
            if not handler_name:
                errors.append(f"No handler specified for file type: {file_type}")
        
        # Validate strategy parameters
        for strategy_type, strategies in config.handler_strategies.strategy_parameters.items():
            for strategy_name, params in strategies.items():
                if not params:
                    errors.append(f"No parameters for {strategy_type} strategy: {strategy_name}")
        
        # Validate default strategies exist
        defaults = config.handler_strategies.default_strategies
        
        if defaults["nlp_strategy"] not in config.engine_selection.nlp_handlers:
            errors.append(f"Default NLP strategy not found: {defaults['nlp_strategy']}")
        
        if defaults["chunking_strategy"] not in config.engine_selection.chunking_handlers:
            errors.append(f"Default chunking strategy not found: {defaults['chunking_strategy']}")
        
        if defaults["embedding_provider"] not in config.engine_selection.embedding_handlers:
            errors.append(f"Default embedding provider not found: {defaults['embedding_provider']}")
        
        return errors