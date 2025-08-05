"""
FF Chat Use Case Configuration

Configuration for FF chat use case management following
existing FF configuration patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from .ff_base_config import FFBaseConfigDTO


@dataclass
class FFChatUseCaseConfigDTO(FFBaseConfigDTO):
    """FF Chat use case configuration"""
    
    # Use case loading settings
    config_file_path: str = "config/ff_chat_use_cases.yaml"
    auto_reload_config: bool = False
    validate_use_cases: bool = True
    
    # Processing settings
    default_processing_mode: str = "ff_storage"
    enable_fallback_processing: bool = True
    max_processing_retries: int = 3
    
    # Component settings
    component_timeout: int = 30     # seconds
    enable_component_caching: bool = True
    
    # Use case specific settings
    max_use_cases: int = 50
    enable_custom_use_cases: bool = True
    use_case_validation_timeout: int = 5  # seconds
    
    def validate(self):
        """Validate use case configuration"""
        errors = []
        
        valid_modes = ["ff_storage", "ff_enhanced", "ff_full"]
        if self.default_processing_mode not in valid_modes:
            errors.append(f"default_processing_mode must be one of {valid_modes}")
        
        if self.max_processing_retries < 0:
            errors.append("max_processing_retries must be non-negative")
            
        if self.component_timeout <= 0:
            errors.append("component_timeout must be positive")
            
        if self.max_use_cases <= 0:
            errors.append("max_use_cases must be positive")
            
        if self.use_case_validation_timeout <= 0:
            errors.append("use_case_validation_timeout must be positive")
            
        return errors
    
    def __post_init__(self):
        """Validate use case configuration"""
        super().__post_init__()
        
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")