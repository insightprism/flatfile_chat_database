"""
Unit tests for Chat Application Bridge System components.

Tests individual components in isolation with mocked dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from ff_chat_integration import (
    ChatIntegrationError, ConfigurationError, InitializationError,
    ChatAppStorageConfig, FFChatAppBridge, FFChatDataLayer,
    FFChatConfigFactory, FFIntegrationHealthMonitor
)

from . import MockFactories, PerformanceTester


class TestExceptionHierarchy:
    """Test exception classes and hierarchy."""
    
    def test_base_exception(self):
        """Test ChatIntegrationError base functionality."""
        error = ChatIntegrationError(
            "Test error",
            context={"test": True},
            suggestions=["Fix the test"],
            error_code="TEST_ERROR"
        )
        
        assert str(error) == "Test error (Context: test=True)"
        assert error.context["test"] is True
        assert "Fix the test" in error.suggestions
        assert error.error_code == "TEST_ERROR"
        
        error_dict = error.to_dict()
        assert error_dict["error"] == "Test error (Context: test=True)"
        assert error_dict["error_code"] == "TEST_ERROR"
    
    def test_configuration_error(self):
        """Test ConfigurationError specifics."""
        error = ConfigurationError(
            "Invalid config",
            config_field="test_field",
            config_value="bad_value"
        )
        
        assert error.context["config_field"] == "test_field"
        assert error.context["config_value"] == "bad_value"
        assert error.error_code == "CONFIG_ERROR"
    
    def test_initialization_error(self):
        """Test InitializationError specifics."""
        error = InitializationError(
            "Init failed",
            component="test_component",
            initialization_step="test_step"
        )
        
        assert error.context["component"] == "test_component"
        assert error.context["initialization_step"] == "test_step"
        assert "Verify storage path exists" in error.suggestions


class TestChatAppStorageConfig:
    """Test ChatAppStorageConfig validation and functionality."""
    
    def test_valid_config_creation(self):
        """Test creating valid configuration."""
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced",
            cache_size_mb=100
        )
        
        assert config.storage_path == "./test_data"
        assert config.performance_mode == "balanced"
        assert config.cache_size_mb == 100
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="speed",
            cache_size_mb=50
        )
        
        # Should not raise exception
        issues = config.validate()
        # May have path issues but performance settings should be valid
        performance_issues = [i for i in issues if "performance" in i.lower()]
        assert len(performance_issues) == 0
    
    def test_config_validation_failures(self):
        """Test configuration validation failures."""
        with pytest.raises(ConfigurationError):
            ChatAppStorageConfig(
                storage_path="",  # Empty path
                performance_mode="invalid_mode",  # Invalid mode
                cache_size_mb=5  # Too small
            )
    
    def test_config_serialization(self):
        """Test configuration to_dict method."""
        config = ChatAppStorageConfig(
            storage_path="./test",
            enable_vector_search=True,
            cache_size_mb=100
        )
        
        config_dict = config.to_dict()
        assert config_dict["storage_path"] == "./test"
        assert config_dict["features"]["vector_search"] is True
        assert config_dict["performance"]["cache_size_mb"] == 100


class TestFFChatConfigFactory:
    """Test configuration factory functionality."""
    
    def test_factory_initialization(self):
        """Test factory creates with templates."""
        factory = FFChatConfigFactory()
        templates = factory.list_templates()
        
        assert len(templates) >= 5
        assert "development" in templates
        assert "production" in templates
        assert "high_performance" in templates
    
    def test_template_creation(self):
        """Test creating config from template."""
        factory = FFChatConfigFactory()
        
        config = factory.create_from_template("development", "./test_data")
        assert config.storage_path == "./test_data"
        assert config.environment == "development"
    
    def test_template_with_overrides(self):
        """Test template creation with overrides."""
        factory = FFChatConfigFactory()
        
        config = factory.create_from_template(
            "production",
            "./prod_data",
            {"cache_size_mb": 300, "enable_compression": False}
        )
        
        assert config.cache_size_mb == 300
        assert config.enable_compression is False
    
    def test_invalid_template(self):
        """Test creating from invalid template."""
        factory = FFChatConfigFactory()
        
        with pytest.raises(ConfigurationError):
            factory.create_from_template("nonexistent_template", "./data")
    
    def test_environment_creation(self):
        """Test creating config for environment."""
        factory = FFChatConfigFactory()
        
        config = factory.create_for_environment("production", "./prod_data")
        assert config.environment == "production"
        assert config.storage_path == "./prod_data"
    
    def test_use_case_creation(self):
        """Test creating config for use case."""
        factory = FFChatConfigFactory()
        
        config = factory.create_for_use_case("ai_assistant", "./ai_data")
        assert config.storage_path == "./ai_data"
        # AI assistant should have vector search enabled
        assert config.enable_vector_search is True
    
    def test_config_validation_and_optimization(self):
        """Test configuration validation and optimization scoring."""
        factory = FFChatConfigFactory()
        
        good_config = ChatAppStorageConfig(
            storage_path="./test_data",
            performance_mode="balanced",
            cache_size_mb=100
        )
        
        results = factory.validate_and_optimize(good_config)
        assert results["valid"] is True
        assert "optimization_score" in results
        assert results["optimization_score"] >= 0
    
    def test_migration_from_wrapper(self):
        """Test migrating from wrapper configuration."""
        factory = FFChatConfigFactory()
        
        wrapper_config = {
            "base_path": "./old_data",
            "cache_size_limit": 150,
            "enable_vector_search": True,
            "performance_mode": "speed"
        }
        
        bridge_config = factory.migrate_from_wrapper_config(wrapper_config)
        assert bridge_config.storage_path == "./old_data"
        assert bridge_config.cache_size_mb == 150
        assert bridge_config.enable_vector_search is True


class TestFFChatDataLayer:
    """Test data layer functionality with mocked storage."""
    
    @pytest.fixture
    def mock_data_layer(self):
        """Create data layer with mocked storage."""
        mock_storage = MockFactories.create_mock_storage_manager()
        mock_config = MockFactories.create_mock_config()
        
        return FFChatDataLayer(mock_storage, mock_config)
    
    async def test_standardized_response_format(self, mock_data_layer):
        """Test standardized response format."""
        # Mock successful message storage
        mock_data_layer.storage.add_message.return_value = True
        
        result = await mock_data_layer.store_chat_message(
            "test_user", "test_session",
            {"role": "user", "content": "Test message"}
        )
        
        # Check standardized format
        assert "success" in result
        assert "data" in result
        assert "metadata" in result
        assert "error" in result
        assert "warnings" in result
        
        # Check metadata structure
        metadata = result["metadata"]
        assert "operation" in metadata
        assert "operation_time_ms" in metadata
        assert "records_affected" in metadata
    
    async def test_performance_metrics_tracking(self, mock_data_layer):
        """Test performance metrics tracking."""
        # Mock successful operations
        mock_data_layer.storage.add_message.return_value = True
        
        # Perform operations to generate metrics
        for i in range(3):
            await mock_data_layer.store_chat_message(
                "test_user", "test_session",
                {"role": "user", "content": f"Message {i}"}
            )
        
        metrics = mock_data_layer.get_performance_metrics()
        
        assert "operation_metrics" in metrics
        assert "store_chat_message" in metrics["operation_metrics"]
        
        store_metrics = metrics["operation_metrics"]["store_chat_message"]
        assert store_metrics["total_operations"] == 3
        assert "average_ms" in store_metrics
    
    async def test_cache_functionality(self, mock_data_layer):
        """Test caching functionality."""
        # Mock message return
        mock_messages = [
            {"role": "user", "content": "Test", "timestamp": "2025-01-01T00:00:00Z"}
        ]
        mock_data_layer.storage.get_messages.return_value = mock_messages
        
        # First call - should hit storage
        result1 = await mock_data_layer.get_chat_history("user", "session", limit=10)
        
        # Second call - should hit cache
        result2 = await mock_data_layer.get_chat_history("user", "session", limit=10)
        
        # Both should succeed
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Second call should indicate cache hit
        assert result2["metadata"]["performance_metrics"]["cache_hit"] is True


class TestFFIntegrationHealthMonitor:
    """Test health monitoring functionality."""
    
    @pytest.fixture
    def mock_bridge(self):
        """Create mock bridge for health monitoring."""
        mock_bridge = Mock()
        mock_bridge._initialized = True
        mock_bridge.start_time = 1000000000
        mock_bridge.config = MockFactories.create_mock_config()
        mock_bridge._storage_manager = MockFactories.create_mock_storage_manager()
        
        # Mock get_data_layer
        mock_data_layer = Mock()
        mock_data_layer.get_performance_metrics.return_value = {
            "operation_metrics": {
                "store_chat_message": {
                    "average_ms": 50,
                    "recent_avg_ms": 45,
                    "total_operations": 10
                }
            },
            "cache_stats": {"cache_size": 5}
        }
        mock_bridge.get_data_layer.return_value = mock_data_layer
        
        return mock_bridge
    
    async def test_health_monitor_creation(self, mock_bridge):
        """Test health monitor creation."""
        monitor = FFIntegrationHealthMonitor(mock_bridge)
        
        assert monitor.bridge == mock_bridge
        assert monitor._monitoring_enabled is True
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.Process')
    async def test_comprehensive_health_check(self, mock_process, mock_disk, 
                                            mock_memory, mock_cpu, mock_bridge):
        """Test comprehensive health check."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, available=4000000000)
        mock_disk.return_value = Mock(total=100000000000, used=50000000000, free=50000000000)
        mock_process.return_value.memory_info.return_value.rss = 100000000
        
        monitor = FFIntegrationHealthMonitor(mock_bridge)
        
        health_results = await monitor.comprehensive_health_check()
        
        assert "overall_status" in health_results
        assert "component_health" in health_results
        assert "system_health" in health_results
        assert "optimization_score" in health_results
        assert health_results["overall_status"] in ["healthy", "degraded", "error"]
    
    async def test_issue_diagnosis(self, mock_bridge):
        """Test automated issue diagnosis."""
        monitor = FFIntegrationHealthMonitor(mock_bridge)
        
        diagnosis = await monitor.diagnose_issues()
        
        assert "issues_found" in diagnosis
        assert "diagnostics" in diagnosis
        assert "resolution_plan" in diagnosis
        assert "priority_actions" in diagnosis