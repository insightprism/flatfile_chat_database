"""
Phase 5 validation script for Chat Application Bridge System.

Validates health monitoring, diagnostics, and performance analytics.
"""

import asyncio
import sys
import tempfile
import traceback
import time
from pathlib import Path

async def test_health_monitor_creation():
    """Test health monitor creation."""
    try:
        from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "health_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            monitor = FFIntegrationHealthMonitor(bridge)
            
            assert monitor is not None
            assert monitor.bridge == bridge
            assert monitor._monitoring_enabled is True
            assert monitor._background_monitoring is False
            print("✓ Health monitor creation successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Health monitor creation test failed: {e}")
        traceback.print_exc()
        return False

async def test_comprehensive_health_check():
    """Test comprehensive health checking."""
    try:
        from ff_chat_integration import FFChatAppBridge, quick_health_check
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "health_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            
            # Run comprehensive health check
            health_results = await quick_health_check(bridge)
            
            assert "overall_status" in health_results
            assert "component_health" in health_results
            assert "system_health" in health_results
            assert "optimization_score" in health_results
            assert health_results["overall_status"] in ["healthy", "degraded", "error"]
            assert isinstance(health_results["optimization_score"], int)
            assert 0 <= health_results["optimization_score"] <= 100
            
            # Check component health structure
            component_health = health_results["component_health"]
            expected_components = ["bridge", "storage", "data_layer", "configuration", "cache"]
            for component in expected_components:
                assert component in component_health
                assert "status" in component_health[component]
                assert "message" in component_health[component]
                
            # Check system health structure
            system_health = health_results["system_health"]
            expected_resources = ["cpu", "memory", "disk", "process"]
            for resource in expected_resources:
                if resource in system_health:  # May not be available in all environments
                    assert "status" in system_health[resource]
            
            print("✓ Comprehensive health check successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive health check test failed: {e}")
        traceback.print_exc()
        return False

async def test_issue_diagnosis():
    """Test automated issue diagnosis."""
    try:
        from ff_chat_integration import FFChatAppBridge, diagnose_bridge_issues
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "diagnosis_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            
            # Run issue diagnosis
            diagnosis = await diagnose_bridge_issues(bridge)
            
            assert "issues_found" in diagnosis
            assert "diagnostics" in diagnosis
            assert "resolution_plan" in diagnosis
            assert "priority_actions" in diagnosis
            assert isinstance(diagnosis["issues_found"], int)
            assert isinstance(diagnosis["diagnostics"], list)
            assert isinstance(diagnosis["resolution_plan"], list)
            assert isinstance(diagnosis["priority_actions"], list)
            
            # Check diagnostics structure if issues exist
            if diagnosis["issues_found"] > 0:
                for diagnostic in diagnosis["diagnostics"]:
                    assert "issue" in diagnostic
                    assert "probable_causes" in diagnostic
                    assert "resolution_suggestions" in diagnostic
            
            print("✓ Issue diagnosis successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Issue diagnosis test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_analytics():
    """Test performance analytics."""
    try:
        from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "analytics_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            monitor = FFIntegrationHealthMonitor(bridge)
            
            # Generate some activity for analytics
            data_layer = bridge.get_data_layer()
            
            # Store a test message to generate metrics
            try:
                await data_layer.store_chat_message(
                    user_id="test_user",
                    session_id="test_session",
                    message={
                        "role": "user",
                        "content": "Test message for analytics",
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                )
            except Exception:
                # Some storage operations might fail in test environment
                pass
            
            # Get performance analytics
            analytics = await monitor.get_performance_analytics()
            
            assert "performance_trends" in analytics
            assert "optimization_history" in analytics
            assert "recommendations" in analytics
            assert "summary" in analytics
            assert "time_range_hours" in analytics
            assert isinstance(analytics["performance_trends"], dict)
            assert isinstance(analytics["recommendations"], list)
            
            print("✓ Performance analytics successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance analytics test failed: {e}")
        traceback.print_exc()
        return False

async def test_background_monitoring():
    """Test background monitoring functionality."""
    try:
        from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "monitoring_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            monitor = FFIntegrationHealthMonitor(bridge)
            
            # Verify initial state
            assert monitor._background_monitoring is False
            assert monitor._monitoring_thread is None
            
            # Start background monitoring with very short interval for testing
            monitor.start_background_monitoring(interval_minutes=0.02)  # 1.2 seconds
            
            # Verify monitoring started
            assert monitor._background_monitoring is True
            assert monitor._monitoring_thread is not None
            
            # Wait briefly to let monitoring thread run
            time.sleep(2.0)
            
            # Stop background monitoring
            monitor.stop_background_monitoring()
            
            # Verify monitoring stopped
            assert monitor._background_monitoring is False
            
            print("✓ Background monitoring successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Background monitoring test failed: {e}")
        traceback.print_exc()
        return False

async def test_health_check_results_structure():
    """Test health check results structure and data types."""
    try:
        from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "structure_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            monitor = FFIntegrationHealthMonitor(bridge)
            
            # Run health check
            results = await monitor.comprehensive_health_check()
            
            # Verify top-level structure
            required_keys = [
                "overall_status", "timestamp", "check_duration_ms",
                "component_health", "system_health", "performance_health",
                "issues_detected", "recommendations", "optimization_score"
            ]
            
            for key in required_keys:
                assert key in results, f"Missing required key: {key}"
            
            # Verify data types
            assert isinstance(results["overall_status"], str)
            assert isinstance(results["timestamp"], str)
            assert isinstance(results["check_duration_ms"], (int, float))
            assert isinstance(results["component_health"], dict)
            assert isinstance(results["issues_detected"], list)
            assert isinstance(results["recommendations"], list)
            assert isinstance(results["optimization_score"], int)
            
            # Verify optimization score range
            assert 0 <= results["optimization_score"] <= 100
            
            # Verify status values
            assert results["overall_status"] in ["healthy", "degraded", "error", "unknown"]
            
            print("✓ Health check results structure validation successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Health check results structure test failed: {e}")
        traceback.print_exc()
        return False

async def test_convenience_functions():
    """Test convenience functions for health monitoring."""
    try:
        from ff_chat_integration import (
            FFChatAppBridge, create_health_monitor, 
            quick_health_check, diagnose_bridge_issues
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "convenience_test")
            
            bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
            
            # Test create_health_monitor
            monitor = await create_health_monitor(bridge)
            assert monitor is not None
            assert monitor.bridge == bridge
            
            # Test quick_health_check
            health_results = await quick_health_check(bridge)
            assert "overall_status" in health_results
            
            # Test diagnose_bridge_issues
            diagnosis = await diagnose_bridge_issues(bridge)
            assert "issues_found" in diagnosis
            assert "diagnostics" in diagnosis
            
            print("✓ Convenience functions successful")
            
            await bridge.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        traceback.print_exc()
        return False

async def test_health_monitor_integration():
    """Test health monitor integration with all bridge components."""
    try:
        from ff_chat_integration import (
            FFChatAppBridge, FFIntegrationHealthMonitor,
            FFChatConfigFactory
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = str(Path(temp_dir) / "integration_test")
            
            # Test with different configurations
            factory = FFChatConfigFactory()
            
            # Test with production preset
            bridge1 = await FFChatAppBridge.create_from_preset(
                "production", storage_path + "_prod"
            )
            monitor1 = FFIntegrationHealthMonitor(bridge1)
            health1 = await monitor1.comprehensive_health_check()
            
            assert "component_health" in health1
            assert "configuration" in health1["component_health"]
            
            # Test with development preset
            bridge2 = await FFChatAppBridge.create_from_preset(
                "development", storage_path + "_dev"
            )
            monitor2 = FFIntegrationHealthMonitor(bridge2)
            health2 = await monitor2.comprehensive_health_check()
            
            assert "component_health" in health2
            assert "configuration" in health2["component_health"]
            
            # Test with high-performance preset
            bridge3 = await FFChatAppBridge.create_from_preset(
                "high_performance", storage_path + "_hp"
            )
            monitor3 = FFIntegrationHealthMonitor(bridge3)
            health3 = await monitor3.comprehensive_health_check()
            
            assert "component_health" in health3
            assert "configuration" in health3["component_health"]
            
            print("✓ Health monitor integration successful")
            
            await bridge1.close()
            await bridge2.close()
            await bridge3.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Health monitor integration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 5 validation tests."""
    print("Phase 5 Validation - Health Monitoring and Diagnostics")
    print("=" * 55)
    
    tests = [
        ("Health Monitor Creation", test_health_monitor_creation),
        ("Comprehensive Health Check", test_comprehensive_health_check),
        ("Issue Diagnosis", test_issue_diagnosis),  
        ("Performance Analytics", test_performance_analytics),
        ("Background Monitoring", test_background_monitoring),
        ("Health Check Results Structure", test_health_check_results_structure),
        ("Convenience Functions", test_convenience_functions),
        ("Health Monitor Integration", test_health_monitor_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if await test_func():
                passed += 1
            else:
                print(f"Test {test_name} failed!")
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n" + "=" * 55)
    print(f"Phase 5 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Phase 5 implementation is complete and ready!")
        return True
    else:
        print("✗ Phase 5 needs fixes before completion")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)