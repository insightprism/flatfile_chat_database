"""
FF Chat Application Configuration

Enhanced configuration for FF chat application following
existing FF configuration patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .ff_base_config import FFBaseConfigDTO
from .ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from .ff_chat_session_config import FFChatSessionConfigDTO
from .ff_chat_use_case_config import FFChatUseCaseConfigDTO


@dataclass
class FFChatApplicationConfigDTO(FFConfigurationManagerConfigDTO):
    """
    FF Chat Application configuration extending existing FF config.
    
    Inherits all existing FF configurations and adds chat-specific settings.
    """
    
    # Chat-specific configurations
    chat_session: FFChatSessionConfigDTO = field(default_factory=FFChatSessionConfigDTO)
    chat_use_cases: FFChatUseCaseConfigDTO = field(default_factory=FFChatUseCaseConfigDTO)
    
    # Chat application settings
    default_use_case: str = "basic_chat"
    max_concurrent_sessions: int = 1000
    enable_session_persistence: bool = True
    enable_real_time_features: bool = True
    
    # Performance settings
    session_pool_size: int = 100
    message_batch_size: int = 10
    enable_session_caching: bool = True
    
    # Security settings
    enable_user_isolation: bool = True
    max_sessions_per_user: int = 10
    session_id_length: int = 12
    
    def __post_init__(self):
        """Initialize chat application configuration"""
        super().__post_init__()
        
        # Validate chat-specific settings
        if self.max_concurrent_sessions <= 0:
            raise ValueError("max_concurrent_sessions must be positive")
        
        if not self.default_use_case:
            raise ValueError("default_use_case cannot be empty")
            
        if self.session_pool_size <= 0:
            raise ValueError("session_pool_size must be positive")
            
        if self.message_batch_size <= 0:
            raise ValueError("message_batch_size must be positive")
            
        if self.max_sessions_per_user <= 0:
            raise ValueError("max_sessions_per_user must be positive")
            
        if self.session_id_length <= 4:
            raise ValueError("session_id_length must be greater than 4")
    
    def get_chat_summary(self) -> Dict[str, Any]:
        """
        Get chat application configuration summary for logging/debugging.
        
        Returns:
            Summary of key chat configuration values
        """
        base_summary = self.get_summary()
        chat_summary = {
            "default_use_case": self.default_use_case,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "session_persistence_enabled": self.enable_session_persistence,
            "real_time_features_enabled": self.enable_real_time_features,
            "session_timeout": self.chat_session.session_timeout,
            "processing_mode": self.chat_use_cases.default_processing_mode,
            "user_isolation_enabled": self.enable_user_isolation,
            "max_sessions_per_user": self.max_sessions_per_user
        }
        
        return {**base_summary, "chat_config": chat_summary}