"""
File locking configuration.

Manages concurrent access control and locking strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .base import BaseConfig, validate_positive, validate_range


@dataclass
class LockingConfig(BaseConfig):
    """
    File locking configuration.
    
    Controls concurrent access, retry strategies, and lock management.
    """
    
    # Core settings
    enabled: bool = True
    strategy: str = "file"  # "file", "database", "redis"
    lock_file_suffix: str = ".lock"
    
    # Timeout settings
    timeout_seconds: float = 30.0
    acquire_timeout_seconds: float = 30.0
    hold_timeout_seconds: float = 300.0  # Maximum time a lock can be held
    
    # Retry settings
    retry_enabled: bool = True
    retry_strategy: str = "exponential"  # "fixed", "linear", "exponential"
    retry_initial_delay_ms: int = 10
    retry_max_delay_seconds: float = 1.0
    retry_max_attempts: int = 100
    retry_multiplier: float = 2.0  # For exponential backoff
    
    # Deadlock prevention
    deadlock_detection_enabled: bool = True
    deadlock_timeout_seconds: float = 60.0
    lock_ordering_enforced: bool = True
    
    # Lock modes
    support_shared_locks: bool = True
    upgrade_locks_enabled: bool = True  # Allow upgrading from shared to exclusive
    
    # Performance
    lock_pool_size: int = 100  # Maximum number of concurrent locks
    lock_cache_enabled: bool = True
    lock_metrics_enabled: bool = True
    
    # Cleanup
    auto_cleanup_stale_locks: bool = True
    stale_lock_threshold_seconds: float = 3600.0  # 1 hour
    cleanup_interval_seconds: float = 300.0  # 5 minutes
    
    # Platform-specific settings
    use_fcntl_on_unix: bool = True
    use_msvcrt_on_windows: bool = True
    fallback_to_polling: bool = True
    
    def validate(self) -> List[str]:
        """
        Validate locking configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate strategy
        valid_strategies = ["file", "database", "redis"]
        if self.strategy not in valid_strategies:
            errors.append(f"strategy must be one of {valid_strategies}, got {self.strategy}")
        
        # Validate timeouts
        if error := validate_positive(self.timeout_seconds, "timeout_seconds"):
            errors.append(error)
        if error := validate_positive(self.acquire_timeout_seconds, "acquire_timeout_seconds"):
            errors.append(error)
        if error := validate_positive(self.hold_timeout_seconds, "hold_timeout_seconds"):
            errors.append(error)
        
        # Validate retry settings
        if self.retry_enabled:
            valid_retry_strategies = ["fixed", "linear", "exponential"]
            if self.retry_strategy not in valid_retry_strategies:
                errors.append(f"retry_strategy must be one of {valid_retry_strategies}, got {self.retry_strategy}")
            
            if error := validate_positive(self.retry_initial_delay_ms, "retry_initial_delay_ms"):
                errors.append(error)
            if error := validate_positive(self.retry_max_delay_seconds, "retry_max_delay_seconds"):
                errors.append(error)
            if error := validate_positive(self.retry_max_attempts, "retry_max_attempts"):
                errors.append(error)
            
            if self.retry_strategy == "exponential":
                if error := validate_range(self.retry_multiplier, 1.0, 10.0, "retry_multiplier"):
                    errors.append(error)
        
        # Validate deadlock settings
        if self.deadlock_detection_enabled:
            if error := validate_positive(self.deadlock_timeout_seconds, "deadlock_timeout_seconds"):
                errors.append(error)
        
        # Validate performance settings
        if error := validate_positive(self.lock_pool_size, "lock_pool_size"):
            errors.append(error)
        
        # Validate cleanup settings
        if self.auto_cleanup_stale_locks:
            if error := validate_positive(self.stale_lock_threshold_seconds, "stale_lock_threshold_seconds"):
                errors.append(error)
            if error := validate_positive(self.cleanup_interval_seconds, "cleanup_interval_seconds"):
                errors.append(error)
        
        return errors
    
    def get_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay for a given attempt.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if not self.retry_enabled or attempt >= self.retry_max_attempts:
            return 0.0
        
        initial_delay = self.retry_initial_delay_ms / 1000.0
        
        if self.retry_strategy == "fixed":
            delay = initial_delay
        elif self.retry_strategy == "linear":
            delay = initial_delay * (attempt + 1)
        elif self.retry_strategy == "exponential":
            delay = initial_delay * (self.retry_multiplier ** attempt)
        else:
            delay = initial_delay
        
        # Cap at maximum delay
        return min(delay, self.retry_max_delay_seconds)