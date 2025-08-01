"""
Comprehensive tests for the configuration system.

Tests the modular configuration architecture, environment handling,
validation, and integration with the new system architecture.
"""

import pytest
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ff_class_configs.ff_configuration_manager_config import (
    FFConfigurationManagerConfigDTO, 
    load_config, 
    create_default_config
)
from ff_class_configs.ff_storage_config import FFStorageConfigDTO
from ff_class_configs.ff_search_config import FFSearchConfigDTO
from ff_class_configs.ff_vector_storage_config import FFVectorStorageConfigDTO
from ff_class_configs.ff_document_config import FFDocumentConfigDTO
from ff_class_configs.ff_locking_config import FFLockingConfigDTO
from ff_class_configs.ff_persona_panel_config import FFPersonaPanelConfigDTO


class TestConfigurationCreation:
    """Test configuration object creation and initialization."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = create_default_config("development")
        
        assert config.environment == "development"
        assert config.storage.base_path == "./dev_data"
        assert config.storage.validate_json_on_read is True
        assert config.locking.deadlock_detection_enabled is True
        assert config.search.enable_search_cache is False
        
    def test_production_config_creation(self):
        """Test creating production configuration."""
        config = create_default_config("production")
        
        assert config.environment == "production"
        assert config.storage.base_path == "/var/lib/chatdb/data"
        assert config.storage.validate_json_on_read is False
        assert config.locking.retry_max_attempts == 200
        assert config.search.cache_ttl_seconds == 600
        
    def test_test_config_creation(self):
        """Test creating test configuration."""
        config = create_default_config("test")
        
        assert config.environment == "test"
        assert config.storage.base_path == "./test_data"
        assert config.storage.max_messages_per_session == 1000
        assert config.locking.timeout_seconds == 5.0
        assert config.search.max_total_results == 100
        
    def test_custom_config_creation(self, temp_dir):
        """Test creating configuration with custom parameters."""
        config = FFConfigurationManagerConfigDTO(
            environment="custom",
            validate_on_load=False
        )
        
        # Override storage path
        config.storage.base_path = str(temp_dir)
        
        assert config.environment == "custom"
        assert config.storage.base_path == str(temp_dir)
        assert config.validate_on_load is False


class TestConfigurationLoading:
    """Test configuration loading from files and environment."""
    
    def test_load_config_with_defaults(self):
        """Test loading config with default environment detection."""
        with patch.dict(os.environ, {"CHATDB_ENV": "development"}):
            config = load_config()
            assert config.environment == "development"
    
    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from JSON file."""
        # Create test config file
        config_data = {
            "storage": {
                "base_path": str(temp_dir),
                "max_message_size_bytes": 2048
            },
            "search": {
                "default_limit": 50,
                "enable_search_cache": True
            },
            "vector": {
                "default_embedding_provider": "test-provider"
            }
        }
        
        config_file = temp_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = load_config(config_file, "test")
        
        assert config.storage.base_path == str(temp_dir)
        assert config.storage.max_message_size_bytes == 2048
        assert config.search.default_limit == 50
        assert config.search.enable_search_cache is True
        assert config.vector.default_embedding_provider == "test-provider"
        assert config.environment == "test"
    
    def test_load_config_file_not_found(self, temp_dir):
        """Test loading configuration when file doesn't exist."""
        non_existent_file = temp_dir / "missing.json"
        
        with pytest.raises(FileNotFoundError):
            FFConfigurationManagerConfigDTO.from_file(non_existent_file)
    
    def test_load_config_with_environment_path(self, temp_dir):
        """Test loading config with environment variable path."""
        config_file = temp_dir / "env_config.json"
        config_data = {"storage": {"base_path": str(temp_dir)}}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        with patch.dict(os.environ, {"CHATDB_CONFIG_PATH": str(config_file)}):
            config = load_config()
            assert config.storage.base_path == str(temp_dir)


class TestEnvironmentOverrides:
    """Test environment variable override functionality."""
    
    def test_storage_environment_overrides(self, test_config):
        """Test storage configuration environment overrides."""
        env_vars = {
            "CHATDB_STORAGE_BASE_PATH": "/custom/path",
            "CHATDB_STORAGE_MAX_MESSAGE_SIZE_BYTES": "4096",
            "CHATDB_STORAGE_VALIDATE_JSON_ON_READ": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            config = test_config._apply_environment_overrides()
            
            # Note: The current implementation modifies in place
            assert test_config.storage.base_path == "/custom/path"
            assert test_config.storage.max_message_size_bytes == 4096
            assert test_config.storage.validate_json_on_read is False
    
    def test_search_environment_overrides(self, test_config):
        """Test search configuration environment overrides."""
        env_vars = {
            "CHATDB_SEARCH_DEFAULT_LIMIT": "200",
            "CHATDB_SEARCH_ENABLE_SEARCH_CACHE": "true",
            "CHATDB_SEARCH_CACHE_TTL_SECONDS": "300"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config._apply_environment_overrides()
            
            assert test_config.search.default_limit == 200
            assert test_config.search.enable_search_cache is True
            assert test_config.search.cache_ttl_seconds == 300
    
    def test_general_environment_overrides(self, test_config):
        """Test general environment overrides that affect multiple domains."""
        env_vars = {
            "CHATDB_BASE_PATH": "/global/path",
            "CHATDB_MAX_FILE_SIZE": "5242880",  # 5MB
            "CHATDB_ENABLE_CACHING": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config._apply_general_overrides()
            
            assert test_config.storage.base_path == "/global/path"
            assert test_config.storage.max_document_size_bytes == 5242880
            assert test_config.document.max_file_size_bytes == 5242880
            assert test_config.search.enable_search_cache is True
            assert test_config.vector.cache_enabled is True
            assert test_config.document.cache_processed_documents is True
    
    def test_boolean_environment_conversion(self, test_config):
        """Test proper conversion of boolean environment variables."""
        # Test various boolean representations
        boolean_tests = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("anything_else", False)
        ]
        
        for env_value, expected in boolean_tests:
            with patch.dict(os.environ, {"CHATDB_SEARCH_ENABLE_SEARCH_CACHE": env_value}):
                test_config._apply_environment_overrides()
                assert test_config.search.enable_search_cache == expected


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_valid_configuration(self, test_config):
        """Test validation of a valid configuration."""
        errors = test_config.validate_all()
        assert len(errors) == 0
    
    def test_invalid_storage_configuration(self, test_config):
        """Test validation with invalid storage settings."""
        # Set invalid values
        test_config.storage.base_path = ""  # Empty path
        test_config.storage.max_message_size_bytes = -1  # Negative size
        test_config.storage.jsonl_read_buffer_size = 0  # Zero buffer
        
        errors = test_config.validate_all()
        
        # Should have storage validation errors
        storage_errors = [e for e in errors if e.startswith("storage:")]
        assert len(storage_errors) > 0
    
    def test_invalid_locking_configuration(self, test_config):
        """Test validation with invalid locking settings."""
        test_config.locking.timeout_seconds = -1.0  # Negative timeout
        test_config.locking.retry_max_attempts = 0  # Zero retries
        test_config.locking.retry_initial_delay_ms = -100  # Negative delay
        
        errors = test_config.validate_all()
        
        locking_errors = [e for e in errors if e.startswith("locking:")]
        assert len(locking_errors) > 0
    
    def test_cross_domain_validation(self, test_config):
        """Test cross-domain validation rules."""
        # Set document size larger than storage limit
        test_config.document.max_file_size_bytes = 10_000_000
        test_config.storage.max_document_size_bytes = 5_000_000
        
        # Set search results larger than message limit
        test_config.search.max_results_per_session = 1000
        test_config.storage.max_messages_per_session = 500
        
        errors = test_config.validate_all()
        
        # Should have cross-domain validation errors
        cross_domain_errors = [e for e in errors if "exceeds" in e]
        assert len(cross_domain_errors) >= 2
    
    def test_validation_on_initialization(self, temp_dir):
        """Test that validation runs during initialization when enabled."""
        # Create configuration with invalid values and validation enabled
        with pytest.raises(ValueError, match="Configuration validation failed"):
            FFConfigurationManagerConfigDTO(
                storage=FFStorageConfigDTO(base_path="", max_message_size_bytes=-1),
                validate_on_load=True
            )
    
    def test_skip_validation_on_initialization(self, temp_dir):
        """Test that validation can be skipped during initialization."""
        # Should not raise exception even with invalid values
        config = FFConfigurationManagerConfigDTO(
            storage=FFStorageConfigDTO(base_path="", max_message_size_bytes=-1),
            validate_on_load=False
        )
        
        # But manual validation should still fail
        errors = config.validate_all()
        assert len(errors) > 0


class TestConfigurationPersistence:
    """Test configuration saving and loading."""
    
    def test_save_and_load_config(self, test_config, temp_dir):
        """Test saving configuration to file and loading it back."""
        config_file = temp_dir / "test_save.json"
        
        # Save configuration
        test_config.save_to_file(config_file)
        assert config_file.exists()
        
        # Load and verify
        loaded_config = FFConfigurationManagerConfigDTO()
        loaded_config.load_from_file(config_file)
        
        assert loaded_config.storage.base_path == test_config.storage.base_path
        assert loaded_config.search.default_limit == test_config.search.default_limit
        assert loaded_config.vector.default_embedding_provider == test_config.vector.default_embedding_provider
    
    def test_to_dict_conversion(self, test_config):
        """Test configuration conversion to dictionary."""
        config_dict = test_config.to_dict()
        
        assert "storage" in config_dict
        assert "search" in config_dict
        assert "vector" in config_dict
        assert "document" in config_dict
        assert "locking" in config_dict
        assert "panel" in config_dict
        assert "environment" in config_dict
        
        # Verify nested structure
        assert "base_path" in config_dict["storage"]
        assert "default_limit" in config_dict["search"]
    
    def test_configuration_summary(self, test_config):
        """Test configuration summary generation."""
        summary = test_config.get_summary()
        
        expected_keys = [
            "environment", "base_path", "locking_enabled", 
            "vector_provider", "search_cache_enabled", 
            "document_extensions", "max_file_size_mb", "panel_types"
        ]
        
        for key in expected_keys:
            assert key in summary
        
        # Verify data types
        assert isinstance(summary["locking_enabled"], bool)
        assert isinstance(summary["document_extensions"], int)
        assert isinstance(summary["max_file_size_mb"], (int, float))


class TestDomainConfigurations:
    """Test individual domain configuration classes."""
    
    def test_storage_config_validation(self):
        """Test storage configuration validation."""
        # Valid config
        config = FFStorageConfigDTO(base_path="./test")
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid config
        config = FFStorageConfigDTO(
            base_path="",
            max_message_size_bytes=-1,
            max_document_size_bytes=0
        )
        errors = config.validate()
        assert len(errors) > 0
    
    def test_search_config_validation(self):
        """Test search configuration validation."""
        # Valid config
        config = FFSearchConfigDTO()
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid config
        config = FFSearchConfigDTO(
            default_limit=-1,
            max_total_results=0,
            cache_ttl_seconds=-100
        )
        errors = config.validate()
        assert len(errors) > 0
    
    def test_vector_config_validation(self):
        """Test vector configuration validation."""
        config = FFVectorStorageConfigDTO()
        errors = config.validate()
        assert len(errors) == 0
        
        # Test with invalid provider
        config = FFVectorStorageConfigDTO(
            default_embedding_provider="invalid-provider"
        )
        errors = config.validate()
        # Should have provider validation error if implemented
    
    def test_locking_config_validation(self):
        """Test locking configuration validation."""
        # Valid config
        config = FFLockingConfigDTO()
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid config
        config = FFLockingConfigDTO(
            timeout_seconds=-1.0,
            retry_max_attempts=-1,
            retry_initial_delay_ms=-100
        )
        errors = config.validate()
        assert len(errors) > 0


class TestConfigurationIntegration:
    """Test configuration integration with other system components."""
    
    def test_config_with_dependency_injection(self, di_container, test_config):
        """Test configuration integration with DI container."""
        # Verify config is registered in container
        resolved_config = di_container.resolve(FFConfigurationManagerConfigDTO)
        assert resolved_config == test_config
        assert resolved_config.storage.base_path == test_config.storage.base_path
    
    def test_config_environment_detection(self):
        """Test automatic environment detection."""
        # Test default environment
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            assert config.environment == "development"  # Default
        
        # Test explicit environment
        with patch.dict(os.environ, {"CHATDB_ENV": "production"}):
            config = load_config()
            assert config.environment == "production"
    
    def test_config_backward_compatibility(self, test_config):
        """Test that configuration changes maintain backward compatibility."""
        # Test that old configuration access patterns still work
        assert hasattr(test_config, "storage")
        assert hasattr(test_config.storage, "base_path")
        assert hasattr(test_config.storage, "max_message_size_bytes")
        
        # Test that configuration can be accessed via get_domain_config
        storage_config = test_config.get_domain_config("storage")
        assert storage_config is not None
        assert storage_config == test_config.storage


@pytest.mark.integration
class TestConfigurationWithRealComponents:
    """Integration tests with real system components."""
    
    @pytest.mark.asyncio
    async def test_config_with_storage_manager(self, test_config, temp_dir):
        """Test configuration integration with storage manager."""
        from ff_storage_manager import FFStorageManager
        from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
        
        # Create storage manager with test config
        backend = FFFlatfileStorageBackend(test_config)
        await backend.initialize()
        
        storage_manager = FFStorageManager(test_config, backend)
        await storage_manager.initialize()
        
        # Verify configuration is used correctly
        assert storage_manager.config == test_config
        assert str(storage_manager.base_path) == test_config.storage.base_path
    
    @pytest.mark.asyncio
    async def test_config_with_application_container(self, temp_dir):
        """Test configuration with full application container."""
        from ff_dependency_injection_manager import ff_create_application_container
        
        # Create container with custom config path
        with patch.dict(os.environ, {"CHATDB_ENV": "test"}):
            container = ff_create_application_container()
            
            config = container.resolve(FFConfigurationManagerConfigDTO)
            assert config.environment == "test"
            
            # Verify all services can be resolved with this config
            from ff_protocols import StorageProtocol, BackendProtocol
            
            storage = container.resolve(StorageProtocol)
            backend = container.resolve(BackendProtocol)
            
            assert storage is not None
            assert backend is not None