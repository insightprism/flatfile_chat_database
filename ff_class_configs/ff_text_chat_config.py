"""
FF Text Chat Component Configuration

Configuration for FF text chat component following existing FF 
configuration patterns and integrating with FF storage backend.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .ff_base_config import FFBaseConfigDTO, validate_positive, validate_range


@dataclass
class FFTextChatConfigDTO(FFBaseConfigDTO):
    """FF Text Chat component configuration following FF patterns"""
    
    # Text processing settings
    max_message_length: int = 4000
    response_format: str = "markdown"
    context_window: int = 10
    
    # Response generation settings (placeholder for LLM integration)
    temperature: float = 0.7
    max_tokens: int = 1000
    model_name: str = "gpt-3.5-turbo"
    
    # FF storage integration settings
    enable_message_search: bool = True
    enable_context_retrieval: bool = True
    max_search_results: int = 5
    
    # Response behavior settings
    response_style: str = "conversational"
    timeout_seconds: int = 30
    enable_streaming: bool = False
    
    # Message validation settings
    enable_content_filtering: bool = False
    min_message_length: int = 1
    allowed_message_types: List[str] = field(default_factory=lambda: ["text", "markdown"])
    
    # Context management settings
    enable_conversation_summary: bool = False
    summary_trigger_threshold: int = 50  # messages
    max_context_length: int = 8000  # characters
    
    # Performance settings
    cache_responses: bool = False
    cache_ttl_seconds: int = 300
    enable_batch_processing: bool = False
    
    def validate(self) -> List[str]:
        """Validate text chat configuration"""
        errors = []
        
        # Validate message length settings
        error = validate_positive(self.max_message_length, "max_message_length")
        if error:
            errors.append(error)
            
        error = validate_positive(self.min_message_length, "min_message_length")
        if error:
            errors.append(error)
            
        if self.min_message_length >= self.max_message_length:
            errors.append("min_message_length must be less than max_message_length")
        
        # Validate context window
        error = validate_positive(self.context_window, "context_window")
        if error:
            errors.append(error)
            
        # Validate temperature range
        error = validate_range(self.temperature, 0.0, 2.0, "temperature")
        if error:
            errors.append(error)
            
        # Validate max tokens
        error = validate_positive(self.max_tokens, "max_tokens")
        if error:
            errors.append(error)
            
        # Validate response format
        valid_formats = ["text", "markdown", "html", "json"]
        if self.response_format not in valid_formats:
            errors.append(f"response_format must be one of {valid_formats}")
            
        # Validate response style
        valid_styles = ["conversational", "formal", "technical", "creative", "concise"]
        if self.response_style not in valid_styles:
            errors.append(f"response_style must be one of {valid_styles}")
            
        # Validate timeout
        error = validate_positive(self.timeout_seconds, "timeout_seconds")
        if error:
            errors.append(error)
            
        # Validate search settings
        if self.enable_context_retrieval:
            error = validate_positive(self.max_search_results, "max_search_results")
            if error:
                errors.append(error)
                
        # Validate summary settings
        if self.enable_conversation_summary:
            error = validate_positive(self.summary_trigger_threshold, "summary_trigger_threshold")
            if error:
                errors.append(error)
                
        # Validate context length
        error = validate_positive(self.max_context_length, "max_context_length")
        if error:
            errors.append(error)
            
        # Validate cache settings
        if self.cache_responses:
            error = validate_positive(self.cache_ttl_seconds, "cache_ttl_seconds")
            if error:
                errors.append(error)
                
        # Validate allowed message types
        if not self.allowed_message_types:
            errors.append("allowed_message_types cannot be empty")
            
        return errors
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for text generation"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "timeout_seconds": self.timeout_seconds
        }
    
    def get_context_config(self) -> Dict[str, Any]:
        """Get context management configuration"""
        return {
            "context_window": self.context_window,
            "enable_context_retrieval": self.enable_context_retrieval,
            "max_search_results": self.max_search_results,
            "max_context_length": self.max_context_length,
            "enable_conversation_summary": self.enable_conversation_summary,
            "summary_trigger_threshold": self.summary_trigger_threshold
        }
    
    def get_message_validation_config(self) -> Dict[str, Any]:
        """Get message validation configuration"""
        return {
            "max_message_length": self.max_message_length,
            "min_message_length": self.min_message_length,
            "allowed_message_types": self.allowed_message_types,
            "enable_content_filtering": self.enable_content_filtering
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "cache_responses": self.cache_responses,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "enable_streaming": self.enable_streaming,
            "enable_batch_processing": self.enable_batch_processing
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()
        
        # Additional post-initialization validation
        if hasattr(self, '_post_init_done'):
            return
        self._post_init_done = True
        
        # Ensure model name is set
        if not self.model_name or not self.model_name.strip():
            self.model_name = "gpt-3.5-turbo"
            
        # Ensure allowed message types includes text
        if "text" not in self.allowed_message_types:
            self.allowed_message_types.append("text")
            
        # Adjust context length if needed
        if self.max_context_length < self.max_message_length:
            self.max_context_length = self.max_message_length * 2


@dataclass 
class FFTextChatUseCaseConfigDTO(FFBaseConfigDTO):
    """Configuration for specific text chat use cases"""
    
    # Use case specific settings  
    use_case_name: str = "basic_chat"
    persona_settings: Dict[str, Any] = field(default_factory=dict)
    context_instructions: str = ""
    
    # Response customization
    response_prefix: str = ""
    response_suffix: str = ""
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    
    # Behavioral settings
    creativity_level: float = 0.7  # 0.0 to 1.0
    formality_level: str = "moderate"  # casual, moderate, formal
    expertise_level: str = "general"   # general, specialized, expert
    
    # Integration settings
    enable_external_tools: bool = False
    enable_memory_integration: bool = False
    enable_multimodal: bool = False
    
    def validate(self) -> List[str]:
        """Validate use case configuration"""
        errors = []
        
        # Validate use case name
        if not self.use_case_name or not self.use_case_name.strip():
            errors.append("use_case_name cannot be empty")
            
        # Validate creativity level
        error = validate_range(self.creativity_level, 0.0, 1.0, "creativity_level")
        if error:
            errors.append(error)
            
        # Validate formality level
        valid_formality = ["casual", "moderate", "formal"]
        if self.formality_level not in valid_formality:
            errors.append(f"formality_level must be one of {valid_formality}")
            
        # Validate expertise level
        valid_expertise = ["general", "specialized", "expert"]
        if self.expertise_level not in valid_expertise:
            errors.append(f"expertise_level must be one of {valid_expertise}")
            
        return errors
    
    def get_persona_config(self) -> Dict[str, Any]:
        """Get persona configuration for the use case"""
        return {
            "use_case_name": self.use_case_name,
            "creativity_level": self.creativity_level,
            "formality_level": self.formality_level,
            "expertise_level": self.expertise_level,
            "context_instructions": self.context_instructions,
            "persona_settings": self.persona_settings
        }
    
    def get_response_config(self) -> Dict[str, Any]:
        """Get response customization configuration"""
        return {
            "response_prefix": self.response_prefix,
            "response_suffix": self.response_suffix,
            "custom_prompts": self.custom_prompts,
            "formality_level": self.formality_level,
            "expertise_level": self.expertise_level
        }
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration settings configuration"""
        return {
            "enable_external_tools": self.enable_external_tools,
            "enable_memory_integration": self.enable_memory_integration,
            "enable_multimodal": self.enable_multimodal
        }