"""
Validation utilities using configurable rules.

Provides centralized validation functions that use configuration-driven rules
instead of hardcoded validation logic.
"""

from typing import List, Optional
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO


def validate_user_id(user_id: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate user ID against configuration rules.
    
    Args:
        user_id: User identifier to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not user_id:
        errors.append("User ID cannot be empty")
        return errors
    
    if len(user_id) < config.runtime.min_user_id_length:
        errors.append(f"User ID must be at least {config.runtime.min_user_id_length} characters")
    
    if len(user_id) > config.runtime.max_user_id_length:
        errors.append(f"User ID cannot exceed {config.runtime.max_user_id_length} characters")
    
    # Check for invalid characters (configurable in the future)
    if any(char in user_id for char in ['\0', '\n', '\r', '\t']):
        errors.append("User ID contains invalid control characters")
    
    return errors


def validate_session_name(session_name: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate session name against configuration rules.
    
    Args:
        session_name: Session name to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not session_name:
        errors.append("Session name cannot be empty")
        return errors
    
    if len(session_name) < config.runtime.min_session_name_length:
        errors.append(f"Session name must be at least {config.runtime.min_session_name_length} characters")
    
    if len(session_name) > config.runtime.max_session_name_length:
        errors.append(f"Session name cannot exceed {config.runtime.max_session_name_length} characters")
    
    return errors


def validate_filename(filename: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate filename against configuration rules.
    
    Args:
        filename: Filename to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not filename:
        errors.append("Filename cannot be empty")
        return errors
    
    if len(filename) < config.runtime.min_filename_length:
        errors.append(f"Filename must be at least {config.runtime.min_filename_length} characters")
    
    if len(filename) > config.runtime.max_filename_length:
        errors.append(f"Filename cannot exceed {config.runtime.max_filename_length} characters")
    
    # Check for problematic characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
    for char in invalid_chars:
        if char in filename:
            errors.append(f"Filename contains invalid character: '{char}'")
            break
    
    return errors


def validate_message_content(content: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate message content against configuration rules.
    
    Args:
        content: Message content to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not content and config.runtime.min_message_content_length > 0:
        errors.append("Message content cannot be empty")
        return errors
    
    if len(content) < config.runtime.min_message_content_length:
        errors.append(f"Message content must be at least {config.runtime.min_message_content_length} characters")
    
    return errors


def validate_document_content(content: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate document content against configuration rules.
    
    Args:
        content: Document content to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not content and config.runtime.min_document_content_length > 0:
        errors.append("Document content cannot be empty")
        return errors
    
    if len(content) < config.runtime.min_document_content_length:
        errors.append(f"Document content must be at least {config.runtime.min_document_content_length} characters")
    
    return errors


def validate_panel_name(panel_name: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate panel name against configuration rules.
    
    Args:
        panel_name: Panel name to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not panel_name:
        errors.append("Panel name cannot be empty")
        return errors
    
    if len(panel_name) > config.runtime.max_panel_name_length:
        errors.append(f"Panel name cannot exceed {config.runtime.max_panel_name_length} characters")
    
    return errors


def validate_persona_name(persona_name: str, config: FFConfigurationManagerConfigDTO) -> List[str]:
    """
    Validate persona name against configuration rules.
    
    Args:
        persona_name: Persona name to validate
        config: Configuration containing validation rules
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not persona_name:
        errors.append("Persona name cannot be empty")
        return errors
    
    if len(persona_name) > config.runtime.max_persona_name_length:
        errors.append(f"Persona name cannot exceed {config.runtime.max_persona_name_length} characters")
    
    return errors


def validate_all_inputs(user_id: str, session_name: Optional[str] = None, 
                       filename: Optional[str] = None, content: Optional[str] = None,
                       panel_name: Optional[str] = None, persona_name: Optional[str] = None,
                       config: FFConfigurationManagerConfigDTO = None) -> List[str]:
    """
    Validate multiple inputs at once.
    
    Args:
        user_id: User identifier (required)
        session_name: Optional session name
        filename: Optional filename
        content: Optional content (message or document)
        panel_name: Optional panel name
        persona_name: Optional persona name
        config: Configuration containing validation rules
        
    Returns:
        List of all validation errors (empty if all valid)
    """
    all_errors = []
    
    # Validate user ID (always required)
    all_errors.extend(validate_user_id(user_id, config))
    
    # Validate optional fields
    if session_name is not None:
        all_errors.extend(validate_session_name(session_name, config))
    
    if filename is not None:
        all_errors.extend(validate_filename(filename, config))
    
    if content is not None:
        # Assume message content validation (can be made configurable)
        all_errors.extend(validate_message_content(content, config))
    
    if panel_name is not None:
        all_errors.extend(validate_panel_name(panel_name, config))
    
    if persona_name is not None:
        all_errors.extend(validate_persona_name(persona_name, config))
    
    return all_errors