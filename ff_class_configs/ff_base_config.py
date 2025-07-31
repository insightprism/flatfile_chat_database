"""
Base configuration class with common functionality.

Provides validation, merging, and environment variable support
for all configuration domains.
"""

import os
import json
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Dict, Any, List, Optional, Type, TypeVar, get_type_hints, get_origin, get_args
from pathlib import Path

T = TypeVar('T', bound='FFBaseConfigDTO')


@dataclass
class FFBaseConfigDTO(ABC):
    """
    Base configuration class with common functionality.
    
    All configuration classes should inherit from this base class
    to ensure consistent behavior across the system.
    """
    
    @abstractmethod
    def validate(self) -> List[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    def merge(self: T, overrides: Dict[str, Any]) -> T:
        """
        Create a new configuration with merged values.
        
        Args:
            overrides: Dictionary of values to override
            
        Returns:
            New configuration instance with merged values
        """
        current_values = dataclasses.asdict(self)
        merged_values = {**current_values, **overrides}
        
        # Filter only valid fields
        valid_fields = {f.name for f in dataclasses.fields(self.__class__)}
        filtered_values = {k: v for k, v in merged_values.items() if k in valid_fields}
        
        return self.__class__(**filtered_values)
    
    def apply_environment_overrides(self: T, prefix: str) -> T:
        """
        Apply environment variable overrides.
        
        Args:
            prefix: Environment variable prefix (e.g., "CHATDB_STORAGE")
            
        Returns:
            New configuration instance with environment overrides
        """
        overrides = {}
        
        for field in dataclasses.fields(self.__class__):
            env_var = f"{prefix}_{field.name.upper()}"
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                field_type = field.type
                converted_value = self._convert_env_value(value, field_type)
                overrides[field.name] = converted_value
        
        return self.merge(overrides) if overrides else self
    
    def _convert_env_value(self, value: str, field_type: Type) -> Any:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment
            field_type: Target field type
            
        Returns:
            Converted value
        """
        # Handle basic types
        if field_type == int:
            return int(value)
        elif field_type == float:
            return float(value)
        elif field_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif field_type == str:
            return value
        
        # Handle Optional types
        origin = get_origin(field_type)
        if origin is type(Optional):
            args = get_args(field_type)
            if args:
                return self._convert_env_value(value, args[0])
        
        # Handle List types
        if origin is list:
            if value.startswith('['):
                return json.loads(value)
            else:
                return value.split(',')
        
        # Handle Dict types
        if origin is dict:
            return json.loads(value)
        
        # Default to string
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation
        """
        return dataclasses.asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert configuration to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary of configuration values
            
        Returns:
            Configuration instance
        """
        # Filter only valid fields
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create configuration from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            Configuration instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls: Type[T], file_path: Path) -> T:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


def validate_positive(value: int, field_name: str) -> Optional[str]:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error message
        
    Returns:
        Error message if invalid, None if valid
    """
    if value <= 0:
        return f"{field_name} must be positive, got {value}"
    return None


def validate_range(value: float, min_val: float, max_val: float, field_name: str) -> Optional[str]:
    """
    Validate that a value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field for error message
        
    Returns:
        Error message if invalid, None if valid
    """
    if value < min_val or value > max_val:
        return f"{field_name} must be between {min_val} and {max_val}, got {value}"
    return None


def validate_non_empty(value: str, field_name: str) -> Optional[str]:
    """
    Validate that a string is not empty.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error message
        
    Returns:
        Error message if invalid, None if valid
    """
    if not value or not value.strip():
        return f"{field_name} cannot be empty"
    return None


def validate_path_exists(path: str, field_name: str) -> Optional[str]:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate
        field_name: Name of the field for error message
        
    Returns:
        Error message if invalid, None if valid
    """
    if not Path(path).exists():
        return f"{field_name} path does not exist: {path}"
    return None