"""
Search configuration for text and advanced search functionality.

Manages search behavior, limits, and filtering options.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .ff_base_config import FFBaseConfig, validate_positive, validate_range


@dataclass
class FFSearchConfig(FFBaseConfig):
    """
    Search-specific configuration.
    
    Controls search behavior, result limits, and filtering options.
    """
    
    # Search behavior
    include_message_content: bool = True
    include_context: bool = True
    include_metadata: bool = False
    case_sensitive: bool = False
    use_regex: bool = False
    
    # Search limits
    min_word_length: int = 3
    default_limit: int = 20
    max_results_per_session: int = 100
    max_total_results: int = 1000
    
    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100
    
    # Relevance scoring
    min_relevance_score: float = 0.0
    relevance_boost_exact_match: float = 2.0
    relevance_boost_title_match: float = 1.5
    relevance_decay_factor: float = 0.95
    
    # Performance
    enable_search_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_entries: int = 1000
    parallel_search_threshold: int = 10  # Number of sessions to search in parallel
    
    # Entity extraction patterns
    extract_urls: bool = True
    extract_emails: bool = True
    extract_mentions: bool = True
    extract_hashtags: bool = True
    extract_code_blocks: bool = True
    
    def validate(self) -> List[str]:
        """
        Validate search configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate word length
        if error := validate_positive(self.min_word_length, "min_word_length"):
            errors.append(error)
        
        # Validate limits
        if error := validate_positive(self.default_limit, "default_limit"):
            errors.append(error)
        if error := validate_positive(self.max_results_per_session, "max_results_per_session"):
            errors.append(error)
        if error := validate_positive(self.max_total_results, "max_total_results"):
            errors.append(error)
        
        # Validate pagination
        if error := validate_positive(self.default_page_size, "default_page_size"):
            errors.append(error)
        if error := validate_positive(self.max_page_size, "max_page_size"):
            errors.append(error)
        
        if self.default_page_size > self.max_page_size:
            errors.append(f"default_page_size ({self.default_page_size}) cannot exceed max_page_size ({self.max_page_size})")
        
        # Validate relevance scoring
        if error := validate_range(self.min_relevance_score, 0.0, 1.0, "min_relevance_score"):
            errors.append(error)
        if error := validate_positive(self.relevance_boost_exact_match, "relevance_boost_exact_match"):
            errors.append(error)
        if error := validate_positive(self.relevance_boost_title_match, "relevance_boost_title_match"):
            errors.append(error)
        if error := validate_range(self.relevance_decay_factor, 0.0, 1.0, "relevance_decay_factor"):
            errors.append(error)
        
        # Validate cache settings
        if self.enable_search_cache:
            if error := validate_positive(self.cache_ttl_seconds, "cache_ttl_seconds"):
                errors.append(error)
            if error := validate_positive(self.max_cache_entries, "max_cache_entries"):
                errors.append(error)
        
        # Validate parallel search threshold
        if error := validate_positive(self.parallel_search_threshold, "parallel_search_threshold"):
            errors.append(error)
        
        return errors