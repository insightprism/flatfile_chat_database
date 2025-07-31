"""
Document processing configuration.

Manages document handling, file types, and processing parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .ff_base_config import FFBaseConfigDTO, validate_positive, validate_non_empty


@dataclass
class FFDocumentConfigDTO(FFBaseConfigDTO):
    """
    Document processing configuration.
    
    Controls document storage, processing, and analysis settings.
    """
    
    # Storage settings
    storage_subdirectory: str = "documents"
    analysis_subdirectory: str = "analysis"
    metadata_filename: str = "metadata.json"
    
    # File type settings
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".md", ".json", ".csv",
        ".doc", ".docx", ".rtf", ".odt",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"
    ])
    
    # Processing limits
    max_file_size_bytes: int = 104_857_600  # 100MB
    max_files_per_session: int = 100
    max_total_files: int = 10000
    
    # Text extraction settings
    extract_metadata: bool = True
    extract_text_from_images: bool = True
    ocr_language: str = "eng"
    preserve_formatting: bool = True
    
    # Analysis settings
    enable_content_analysis: bool = True
    extract_entities: bool = True
    extract_keywords: bool = True
    extract_summary: bool = True
    summary_max_length: int = 500
    keywords_max_count: int = 10
    
    # Processing strategies by file type
    file_type_strategies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "text/plain": {
            "encoding": "utf-8",
            "chunk_size": 4096,
            "preserve_line_breaks": True
        },
        "application/pdf": {
            "extract_images": True,
            "extract_tables": True,
            "preserve_layout": False,
            "password_protected_action": "skip"
        },
        "image/*": {
            "resize_large_images": True,
            "max_dimension": 2048,
            "extract_exif": True,
            "generate_thumbnail": True,
            "thumbnail_size": (256, 256)
        },
        "text/csv": {
            "parse_headers": True,
            "delimiter": ",",
            "max_rows_preview": 100
        },
        "application/json": {
            "pretty_print": True,
            "max_depth": 10,
            "validate_schema": False
        }
    })
    
    # Indexing settings
    index_file_content: bool = True
    index_metadata: bool = True
    full_text_index_enabled: bool = True
    
    # Versioning
    enable_versioning: bool = True
    max_versions_per_document: int = 10
    version_comparison_enabled: bool = True
    
    # Security
    scan_for_malware: bool = False
    allowed_mime_types: List[str] = field(default_factory=lambda: [
        "text/*",
        "application/pdf",
        "application/json",
        "application/xml",
        "image/*",
        "application/vnd.openxmlformats-officedocument.*"
    ])
    
    # Performance
    parallel_processing_threshold: int = 5
    processing_timeout_seconds: int = 300
    cache_processed_documents: bool = True
    cache_ttl_hours: int = 24
    
    def validate(self) -> List[str]:
        """
        Validate document configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate directories
        if error := validate_non_empty(self.storage_subdirectory, "storage_subdirectory"):
            errors.append(error)
        if error := validate_non_empty(self.analysis_subdirectory, "analysis_subdirectory"):
            errors.append(error)
        
        # Validate file extensions
        if not self.allowed_extensions:
            errors.append("allowed_extensions cannot be empty")
        
        for ext in self.allowed_extensions:
            if not ext.startswith('.'):
                errors.append(f"Extension '{ext}' must start with a dot")
        
        # Validate limits
        if error := validate_positive(self.max_file_size_bytes, "max_file_size_bytes"):
            errors.append(error)
        if error := validate_positive(self.max_files_per_session, "max_files_per_session"):
            errors.append(error)
        if error := validate_positive(self.max_total_files, "max_total_files"):
            errors.append(error)
        
        # Validate analysis settings
        if self.enable_content_analysis:
            if error := validate_positive(self.summary_max_length, "summary_max_length"):
                errors.append(error)
            if error := validate_positive(self.keywords_max_count, "keywords_max_count"):
                errors.append(error)
        
        # Validate versioning
        if self.enable_versioning:
            if error := validate_positive(self.max_versions_per_document, "max_versions_per_document"):
                errors.append(error)
        
        # Validate performance settings
        if error := validate_positive(self.parallel_processing_threshold, "parallel_processing_threshold"):
            errors.append(error)
        if error := validate_positive(self.processing_timeout_seconds, "processing_timeout_seconds"):
            errors.append(error)
        
        if self.cache_processed_documents:
            if error := validate_positive(self.cache_ttl_hours, "cache_ttl_hours"):
                errors.append(error)
        
        return errors
    
    def get_file_type_config(self, mime_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific file type.
        
        Args:
            mime_type: MIME type of the file
            
        Returns:
            File type specific configuration
        """
        # Check exact match first
        if mime_type in self.file_type_strategies:
            return self.file_type_strategies[mime_type]
        
        # Check wildcard patterns
        for pattern, config in self.file_type_strategies.items():
            if pattern.endswith('/*'):
                prefix = pattern[:-2]
                if mime_type.startswith(prefix):
                    return config
        
        # Return empty config if no match
        return {}
    
    def is_allowed_file(self, extension: str, mime_type: Optional[str] = None) -> bool:
        """
        Check if a file is allowed based on extension and MIME type.
        
        Args:
            extension: File extension (with or without dot)
            mime_type: Optional MIME type to check
            
        Returns:
            True if file is allowed
        """
        # Normalize extension
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        # Check extension
        if extension.lower() not in [ext.lower() for ext in self.allowed_extensions]:
            return False
        
        # Check MIME type if provided
        if mime_type and self.allowed_mime_types:
            allowed = False
            for allowed_type in self.allowed_mime_types:
                if allowed_type.endswith('/*'):
                    prefix = allowed_type[:-2]
                    if mime_type.startswith(prefix):
                        allowed = True
                        break
                elif mime_type == allowed_type:
                    allowed = True
                    break
            
            if not allowed:
                return False
        
        return True