"""
End-to-end tests for Chat Application Bridge System.

Tests complete workflows and real-world chat application scenarios.
"""

import pytest
import asyncio
from pathlib import Path

from ff_chat_integration import (
    FFChatAppBridge, FFIntegrationHealthMonitor,
    create_chat_config_for_production, diagnose_bridge_issues
)

from . import BridgeTestHelper


class TestCompleteWorkflows:
    """Test complete chat application workflows."""
    
    async def test_simple_chat_app_workflow(self):
        """Test workflow for simple chat application."""
        # Create bridge for simple chat use case
        bridge = await FFChatAppBridge.create_for_use_case(
            "simple_chat",
            "./simple_chat_data"
        )
        
        # Verify initialization
        assert bridge._initialized is True
        config = bridge.get_standardized_config()
        assert config["capabilities"]["vector_search"] is False  # Simple chat doesn't need vector search
        
        # Get data layer
        data_layer = bridge.get_data_layer()
        
        # Create user
        user_id = "simple_chat_user"
        await data_layer.storage.create_user(user_id, {"name": "Simple User"})
        
        # Create session
        session_id = await data_layer.storage.create_session(user_id, "Simple Chat")
        
        # Chat conversation
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to weather data, but I'd be happy to help with other questions!"}
        ]
        
        # Store conversation
        for message in conversation:
            result = await data_layer.store_chat_message(user_id, session_id, message)
            assert result["success"] is True
        
        # Retrieve history
        history = await data_layer.get_chat_history(user_id, session_id)
        assert history["success"] is True
        assert len(history["data"]["messages"]) == 4
        
        # Verify message content
        retrieved_messages = history["data"]["messages"]
        for i, msg in enumerate(retrieved_messages):
            assert msg["content"] == conversation[i]["content"]
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_ai_assistant_workflow(self):
        """Test workflow for AI assistant application."""
        # Create bridge for AI assistant use case
        bridge = await FFChatAppBridge.create_for_use_case(
            "ai_assistant",
            "./ai_assistant_data",
            enable_vector_search=True,
            enable_analytics=True
        )
        
        # Verify capabilities
        config = bridge.get_standardized_config()
        assert config["capabilities"]["vector_search"] is True
        assert config["capabilities"]["analytics"] is True
        
        # Get data layer
        data_layer = bridge.get_data_layer()
        
        # Create user
        user_id = "ai_assistant_user"
        await data_layer.storage.create_user(user_id, {
            "name": "AI Assistant User",
            "preferences": {"model": "advanced", "context_length": "long"}
        })
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session_id = await data_layer.storage.create_session(
                user_id, f"AI Assistant Session {i+1}"
            )
            sessions.append(session_id)
        
        # Have conversations in different sessions
        for session_num, session_id in enumerate(sessions):
            conversation_topics = [
                "Python programming help",
                "Data analysis questions", 
                "Machine learning concepts"
            ]
            
            topic = conversation_topics[session_num]
            
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Can you help me with {topic}?"}
            )
            
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "assistant", "content": f"I'd be happy to help with {topic}! What specific aspects would you like to explore?"}
            )
        
        # Test search across sessions
        search_result = await data_layer.search_conversations(
            user_id, "Python", {"search_type": "text"}
        )
        assert search_result["success"] is True
        assert len(search_result["data"]["results"]) > 0
        
        # Test analytics
        analytics = await data_layer.get_analytics_summary(user_id)
        assert analytics["success"] is True
        assert analytics["data"]["analytics"]["total_sessions"] >= 3
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_high_volume_chat_workflow(self):
        """Test workflow for high volume chat application."""
        # Create bridge optimized for high volume
        bridge = await FFChatAppBridge.create_for_use_case(
            "high_volume_chat",
            "./high_volume_data",
            performance_mode="speed",
            cache_size_mb=200
        )
        
        # Verify performance optimization
        config = bridge.get_standardized_config()
        assert config["performance"]["mode"] == "speed"
        assert config["performance"]["cache_size_mb"] == 200
        
        data_layer = bridge.get_data_layer()
        
        # Simulate high volume scenario
        users = []
        for i in range(5):
            user_id = f"high_volume_user_{i}"
            await data_layer.storage.create_user(user_id, {"name": f"User {i}"})
            users.append(user_id)
        
        # Create sessions for each user
        sessions = {}
        for user_id in users:
            session_id = await data_layer.storage.create_session(user_id, "High Volume Session")
            sessions[user_id] = session_id
        
        # Simulate concurrent message storage
        tasks = []
        for user_id in users:
            for msg_num in range(10):
                task = data_layer.store_chat_message(
                    user_id, sessions[user_id],
                    {"role": "user", "content": f"High volume message {msg_num} from {user_id}"}
                )
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check all operations succeeded
        successful_operations = [r for r in results if not isinstance(r, Exception) and r.get("success")]
        assert len(successful_operations) == 50  # 5 users * 10 messages
        
        # Verify data integrity
        for user_id in users:
            history = await data_layer.get_chat_history(user_id, sessions[user_id])
            assert history["success"] is True
            assert len(history["data"]["messages"]) == 10
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_production_deployment_workflow(self):
        """Test production deployment workflow."""
        # Create production configuration
        prod_config = create_chat_config_for_production(
            "./production_data",
            performance_level="balanced"
        )
        
        # Create bridge with production config
        bridge = await FFChatAppBridge.create_for_chat_app(
            prod_config.storage_path,
            prod_config.to_dict()
        )
        
        # Verify production settings
        config = bridge.get_standardized_config()
        assert config["environment"] == "production"
        assert config["features"]["backup"] is True
        assert config["features"]["compression"] is True
        
        # Test health monitoring
        monitor = FFIntegrationHealthMonitor(bridge)
        health = await monitor.comprehensive_health_check()
        
        # Production deployment should be healthy
        assert health["overall_status"] in ["healthy", "degraded"]
        assert health["optimization_score"] >= 60
        
        # Test performance analytics
        analytics = await monitor.get_performance_analytics()
        assert "performance_trends" in analytics
        
        # Test issue diagnosis
        diagnosis = await diagnose_bridge_issues(bridge)
        assert "resolution_plan" in diagnosis
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestErrorHandlingWorkflows:
    """Test error handling in real scenarios."""
    
    async def test_storage_error_recovery(self):
        """Test recovery from storage errors."""
        bridge = await BridgeTestHelper.create_test_bridge()
        data_layer = bridge.get_data_layer()
        
        # Test with invalid user ID
        result = await data_layer.store_chat_message(
            "nonexistent_user", "nonexistent_session",
            {"role": "user", "content": "Test message"}
        )
        
        # Should handle gracefully
        assert result["success"] is False
        assert "error" in result
        assert result["error"] is not None
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_configuration_error_handling(self):
        """Test configuration error handling."""
        # Test invalid configuration
        try:
            bridge = await FFChatAppBridge.create_for_chat_app(
                "",  # Empty storage path
                {"performance_mode": "invalid_mode"}
            )
            assert False, "Should have raised ConfigurationError"
        except Exception as e:
            assert "Configuration" in str(type(e).__name__)
    
    async def test_health_monitoring_error_detection(self):
        """Test health monitoring error detection."""
        bridge = await BridgeTestHelper.create_test_bridge()
        monitor = FFIntegrationHealthMonitor(bridge)
        
        # Force some issues for testing
        # (In real scenarios, this would detect actual issues)
        health = await monitor.comprehensive_health_check()
        
        # Should complete successfully even with potential issues
        assert "overall_status" in health
        assert health["overall_status"] in ["healthy", "degraded", "error"]
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestMigrationWorkflows:
    """Test migration from existing systems."""
    
    async def test_wrapper_to_bridge_migration(self):
        """Test migration from wrapper-based configuration."""
        from ff_chat_integration import FFChatConfigFactory
        
        # Simulate old wrapper configuration
        old_wrapper_config = {
            "base_path": "./old_system_data",
            "cache_size_limit": 150,
            "enable_vector_search": True,
            "enable_compression": False,
            "performance_mode": "balanced",
            "environment": "production"
        }
        
        # Migrate configuration
        factory = FFChatConfigFactory()
        new_config = factory.migrate_from_wrapper_config(old_wrapper_config)
        
        # Verify migration
        assert new_config.storage_path == "./old_system_data"
        assert new_config.cache_size_mb == 150
        assert new_config.enable_vector_search is True
        assert new_config.enable_compression is False
        
        # Create bridge with migrated config
        bridge = await FFChatAppBridge.create_for_chat_app(
            new_config.storage_path,
            new_config.to_dict()
        )
        
        assert bridge._initialized is True
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)


class TestRealWorldScenarios:
    """Test real-world application scenarios."""
    
    async def test_customer_support_chatbot_scenario(self):
        """Test customer support chatbot workflow."""
        # Create bridge optimized for customer support
        bridge = await FFChatAppBridge.create_for_use_case(
            "customer_support", 
            "./support_data",
            enable_analytics=True
        )
        
        data_layer = bridge.get_data_layer()
        
        # Simulate customer interactions
        customers = ["customer_001", "customer_002", "customer_003"]
        support_sessions = {}
        
        for customer_id in customers:
            # Create customer profile
            await data_layer.storage.create_user(customer_id, {
                "name": f"Customer {customer_id.split('_')[1]}",
                "type": "customer",
                "priority": "standard"
            })
            
            # Create support session
            session_id = await data_layer.storage.create_session(
                customer_id, f"Support Session for {customer_id}"
            )
            support_sessions[customer_id] = session_id
            
            # Simulate support conversation
            support_messages = [
                {"role": "user", "content": "I'm having trouble with my account"},
                {"role": "assistant", "content": "I'd be happy to help with your account issue. Can you describe the specific problem?"},
                {"role": "user", "content": "I can't log in"},
                {"role": "assistant", "content": "Let me help you troubleshoot the login issue. Have you tried resetting your password?"}
            ]
            
            for msg in support_messages:
                result = await data_layer.store_chat_message(customer_id, session_id, msg)
                assert result["success"] is True
        
        # Test analytics for support metrics
        for customer_id in customers:
            analytics = await data_layer.get_analytics_summary(customer_id)
            assert analytics["success"] is True
            assert analytics["data"]["analytics"]["total_messages"] >= 4
        
        # Test search for common issues
        search_result = await data_layer.search_conversations(
            customers[0], "login", {"search_type": "text"}
        )
        assert search_result["success"] is True
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_educational_tutor_scenario(self):
        """Test educational tutoring application workflow."""
        # Create bridge for educational use case
        bridge = await FFChatAppBridge.create_for_use_case(
            "educational_tutor",
            "./tutor_data",
            enable_vector_search=True
        )
        
        data_layer = bridge.get_data_layer()
        
        # Create student profiles
        students = ["student_alice", "student_bob"]
        
        for student_id in students:
            await data_layer.storage.create_user(student_id, {
                "name": student_id.replace("_", " ").title(),
                "type": "student",
                "grade_level": "high_school"
            })
            
            # Create subject-specific sessions
            subjects = ["mathematics", "science", "history"]
            
            for subject in subjects:
                session_id = await data_layer.storage.create_session(
                    student_id, f"{subject.title()} Tutoring"
                )
                
                # Simulate tutoring conversation
                tutoring_messages = [
                    {"role": "user", "content": f"Can you help me with {subject}?"},
                    {"role": "assistant", "content": f"Of course! I'd be happy to help you with {subject}. What specific topic are you working on?"},
                    {"role": "user", "content": "I'm struggling with the homework"},
                    {"role": "assistant", "content": "Let's work through the homework together. Can you show me what you've tried so far?"}
                ]
                
                for msg in tutoring_messages:
                    result = await data_layer.store_chat_message(student_id, session_id, msg)
                    assert result["success"] is True
        
        # Test search across educational content
        search_result = await data_layer.search_conversations(
            "student_alice", "homework", {"search_type": "text"}
        )
        assert search_result["success"] is True
        assert len(search_result["data"]["results"]) >= 3  # Should find in all subjects
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)
    
    async def test_enterprise_knowledge_base_scenario(self):
        """Test enterprise knowledge base workflow."""
        # Create bridge for enterprise knowledge management
        bridge = await FFChatAppBridge.create_for_use_case(
            "knowledge_base",
            "./enterprise_data",
            enable_vector_search=True,
            enable_analytics=True,
            performance_mode="speed"
        )
        
        data_layer = bridge.get_data_layer()
        
        # Create enterprise users with different roles
        enterprise_users = [
            {"id": "emp_john", "role": "developer"},
            {"id": "emp_sarah", "role": "manager"},
            {"id": "emp_mike", "role": "analyst"}
        ]
        
        for user_info in enterprise_users:
            user_id = user_info["id"]
            await data_layer.storage.create_user(user_id, {
                "name": user_id.replace("emp_", "").title(),
                "role": user_info["role"],
                "department": "IT"
            })
            
            # Create knowledge sessions for different topics
            knowledge_topics = ["deployment", "security", "databases"]
            
            for topic in knowledge_topics:
                session_id = await data_layer.storage.create_session(
                    user_id, f"{topic.title()} Knowledge Session"
                )
                
                # Simulate knowledge sharing conversation
                knowledge_messages = [
                    {"role": "user", "content": f"What are the best practices for {topic}?"},
                    {"role": "assistant", "content": f"Here are the key best practices for {topic}..."},
                    {"role": "user", "content": "Can you provide more specific examples?"},
                    {"role": "assistant", "content": "Certainly! Here are some specific examples and case studies..."}
                ]
                
                for msg in knowledge_messages:
                    result = await data_layer.store_chat_message(user_id, session_id, msg)
                    assert result["success"] is True
        
        # Test cross-user knowledge search
        search_result = await data_layer.search_conversations(
            "emp_john", "deployment", {"search_type": "text"}
        )
        assert search_result["success"] is True
        
        # Test enterprise analytics
        for user_info in enterprise_users:
            analytics = await data_layer.get_analytics_summary(user_info["id"])
            assert analytics["success"] is True
            assert analytics["data"]["analytics"]["total_sessions"] >= 3
        
        await BridgeTestHelper.cleanup_test_bridge(bridge)