"""
Integration tests for Chat Application Bridge System.

Tests component interactions with real Flatfile storage system.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from ff_chat_integration import (
    FFChatAppBridge, FFChatDataLayer, FFIntegrationHealthMonitor,
    create_chat_config_for_development, get_chat_app_presets
)

from . import BridgeTestHelper, PerformanceTester


class TestBridgeIntegration:
    """Test bridge integration with Flatfile storage."""
    
    async def test_bridge_creation_and_initialization(self):
        """Test complete bridge creation process."""
        bridge = await BridgeTestHelper.create_test_bridge()
        
        assert bridge._initialized is True
        assert bridge._storage_manager is not None
        
        # Test basic configuration access
        config = bridge.get_standardized_config()
        assert config["initialized"] is True
        assert "storage_path" in config
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_preset_integration(self):
        """Test bridge creation from presets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "preset_test")
            
            # Test development preset
            bridge = await FFChatAppBridge.create_from_preset("development", storage_path)
            
            assert bridge._initialized is True
            config = bridge.get_standardized_config()
            assert config["environment"] == "development"
            
            await bridge.close()
    
    async def test_use_case_integration(self):
        """Test bridge creation for specific use cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "usecase_test")
            
            # Test AI assistant use case
            bridge = await FFChatAppBridge.create_for_use_case(
                "ai_assistant", storage_path
            )
            
            assert bridge._initialized is True
            config = bridge.get_standardized_config()
            assert config["capabilities"]["vector_search"] is True
            
            await bridge.close()
    
    async def test_data_layer_integration(self):
        """Test data layer integration with bridge."""
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        assert isinstance(data_layer, FFChatDataLayer)
        assert data_layer.storage == bridge._storage_manager
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_health_monitoring_integration(self):
        """Test health monitoring integration."""
        bridge = await BridgeTestHelper.create_test_bridge()
        monitor = FFIntegrationHealthMonitor(bridge)
        
        # Test basic health check
        health = await monitor.comprehensive_health_check()
        
        assert "overall_status" in health
        assert health["overall_status"] in ["healthy", "degraded", "error"]
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestDataLayerIntegration:
    """Test data layer integration with real storage."""
    
    async def test_complete_message_workflow(self):
        """Test complete message storage and retrieval workflow."""
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        # Create test user and session
        user_id = "integration_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Integration Test"})
        session_id = await data_layer.storage.create_session(user_id, "Integration Test Session")
        
        # Store messages
        messages = [
            {"role": "user", "content": "Hello, integration test!"},
            {"role": "assistant", "content": "Hello! Integration test working."},
            {"role": "user", "content": "Can you help with Python integration?"},
            {"role": "assistant", "content": "Yes, integration tests are important!"}
        ]
        
        stored_messages = []
        for msg in messages:
            result = await data_layer.store_chat_message(user_id, session_id, msg)
            assert result["success"] is True
            stored_messages.append(result)
        
        # Retrieve history
        history = await data_layer.get_chat_history(user_id, session_id)
        assert history["success"] is True
        assert len(history["data"]["messages"]) == 4
        
        # Test search
        search_result = await data_layer.search_conversations(
            user_id, "Python", {"search_type": "text"}
        )
        assert search_result["success"] is True
        assert len(search_result["data"]["results"]) > 0
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_streaming_integration(self):
        """Test conversation streaming integration."""
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        # Create test data
        user_id = "stream_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Stream Test"})
        session_id = await data_layer.storage.create_session(user_id, "Stream Test Session")
        
        # Add many messages for streaming
        for i in range(25):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Stream test message {i}"}
            )
        
        # Test streaming
        chunks_received = 0
        total_messages = 0
        
        async for chunk in data_layer.stream_conversation(user_id, session_id, chunk_size=10):
            assert chunk["success"] is True
            chunks_received += 1
            total_messages += len(chunk["data"]["chunk"])
            
            if chunks_received >= 5:  # Safety limit
                break
        
        assert chunks_received >= 3  # Should have multiple chunks
        assert total_messages >= 25
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_analytics_integration(self):
        """Test analytics integration."""
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        # Create test data
        user_id = "analytics_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Analytics Test"})
        
        # Create multiple sessions with messages
        for session_num in range(3):
            session_id = await data_layer.storage.create_session(
                user_id, f"Analytics Session {session_num}"
            )
            
            for msg_num in range(5):
                await data_layer.store_chat_message(
                    user_id, session_id,
                    {"role": "user", "content": f"Analytics message {msg_num}"}
                )
        
        # Get analytics
        analytics = await data_layer.get_analytics_summary(user_id)
        
        assert analytics["success"] is True
        assert analytics["data"]["analytics"]["total_sessions"] >= 3
        assert analytics["data"]["analytics"]["total_messages"] >= 15
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestHealthMonitoringIntegration:
    """Test health monitoring with real bridge components."""
    
    async def test_real_health_check(self):
        """Test health check with real bridge."""
        bridge = await BridgeTestHelper.create_test_bridge()
        monitor = FFIntegrationHealthMonitor(bridge)
        
        health = await monitor.comprehensive_health_check()
        
        # Should be healthy for test setup
        assert health["overall_status"] in ["healthy", "degraded"]
        assert "component_health" in health
        assert "bridge" in health["component_health"]
        assert "storage" in health["component_health"]
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_performance_analytics_real(self):
        """Test performance analytics with real operations."""
        bridge = await BridgeTestHelper.create_test_bridge()
        monitor = FFIntegrationHealthMonitor(bridge)
        data_layer = bridge.get_data_layer()
        
        # Generate some activity
        user_id = "perf_test_user"
        await data_layer.storage.create_user(user_id, {"name": "Perf Test"})
        session_id = await data_layer.storage.create_session(user_id, "Perf Session")
        
        # Perform operations to generate metrics
        for i in range(5):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Performance test {i}"}
            )
        
        # Get analytics
        analytics = await monitor.get_performance_analytics()
        
        assert "performance_trends" in analytics
        assert "recommendations" in analytics
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_issue_diagnosis_integration(self):
        """Test issue diagnosis with real bridge."""
        bridge = await BridgeTestHelper.create_test_bridge()
        monitor = FFIntegrationHealthMonitor(bridge)
        
        diagnosis = await monitor.diagnose_issues()
        
        assert "issues_found" in diagnosis
        assert "diagnostics" in diagnosis
        assert "resolution_plan" in diagnosis
        
        # For healthy test setup, should have minimal issues
        assert diagnosis["issues_found"] <= 2
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    async def test_preset_configurations(self):
        """Test all preset configurations work."""
        presets = get_chat_app_presets()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for preset_name, preset_config in presets.items():
                storage_path = str(Path(temp_dir) / f"preset_{preset_name}")
                
                # Update storage path for test
                preset_config.storage_path = storage_path
                
                # Should be able to create bridge with preset
                bridge = await FFChatAppBridge.create_for_chat_app(
                    storage_path, preset_config.to_dict()
                )
                
                assert bridge._initialized is True
                
                await bridge.close()
    
    async def test_configuration_optimization(self):
        """Test configuration optimization integration."""
        from ff_chat_integration import FFChatConfigFactory
        
        factory = FFChatConfigFactory()
        
        # Test different configurations
        configs_to_test = [
            factory.create_from_template("development", "./test_dev"),
            factory.create_from_template("production", "./test_prod"),
            factory.create_from_template("high_performance", "./test_perf")
        ]
        
        for config in configs_to_test:
            results = factory.validate_and_optimize(config)
            
            assert results["valid"] is True
            assert "optimization_score" in results
            assert results["optimization_score"] >= 0