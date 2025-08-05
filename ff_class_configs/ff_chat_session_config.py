"""
FF Chat Session Configuration

Configuration for FF chat session management following
existing FF configuration patterns.
"""

from dataclasses import dataclass
from .ff_base_config import FFBaseConfigDTO


@dataclass
class FFChatSessionConfigDTO(FFBaseConfigDTO):
    """FF Chat session configuration"""
    
    # Session lifecycle settings
    session_timeout: int = 3600  # seconds
    cleanup_interval: int = 300   # seconds
    max_inactive_time: int = 1800 # seconds
    
    # Session processing settings
    enable_concurrent_processing: bool = False
    max_processing_time: int = 30  # seconds
    
    # Session persistence settings
    auto_save_interval: int = 60   # seconds
    persist_session_state: bool = True
    
    def validate(self):
        """Validate chat session configuration"""
        errors = []
        
        if self.session_timeout <= 0:
            errors.append("session_timeout must be positive")
        
        if self.cleanup_interval <= 0:
            errors.append("cleanup_interval must be positive")
            
        if self.max_inactive_time <= 0:
            errors.append("max_inactive_time must be positive")
            
        if self.max_processing_time <= 0:
            errors.append("max_processing_time must be positive")
            
        if self.auto_save_interval <= 0:
            errors.append("auto_save_interval must be positive")
            
        return errors
    
    def __post_init__(self):
        """Validate chat session configuration"""
        super().__post_init__()
        
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")