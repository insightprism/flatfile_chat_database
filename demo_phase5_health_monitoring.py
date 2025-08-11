#!/usr/bin/env python3
"""
Demonstration of Phase 5: Health Monitoring and Diagnostics
Shows comprehensive health monitoring capabilities of the Chat Application Bridge System.
"""

import asyncio
import json
import tempfile
from pathlib import Path

async def main():
    print("🔍 Phase 5 Demo: Health Monitoring and Diagnostics")
    print("=" * 60)
    
    # Import Phase 5 components
    from ff_chat_integration import (
        FFChatAppBridge,
        FFIntegrationHealthMonitor,
        quick_health_check,
        diagnose_bridge_issues,
        create_health_monitor
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = str(Path(temp_dir) / "demo_health")
        
        print(f"\n📂 Setting up bridge with path: {storage_path}")
        
        # Create bridge from production preset
        bridge = await FFChatAppBridge.create_from_preset(
            "production", 
            storage_path,
            {"cache_size_mb": 150}  # Custom override
        )
        
        print("✅ Bridge initialized successfully")
        
        # Demo 1: Quick Health Check
        print("\n🚀 Demo 1: Quick Health Check")
        print("-" * 40)
        
        health_results = await quick_health_check(bridge)
        
        print(f"Overall Status: {health_results['overall_status'].upper()}")
        print(f"Optimization Score: {health_results['optimization_score']}/100")
        print(f"Check Duration: {health_results['check_duration_ms']:.1f}ms")
        
        # Show component health summary
        component_health = health_results.get('component_health', {})
        print(f"\nComponent Health Summary:")
        for component, details in component_health.items():
            status = details.get('status', 'unknown').upper()
            message = details.get('message', 'No details')
            print(f"  • {component}: {status} - {message}")
        
        # Demo 2: Detailed Health Monitor
        print("\n🔬 Demo 2: Comprehensive Health Analysis")
        print("-" * 40)
        
        monitor = await create_health_monitor(bridge)
        detailed_health = await monitor.comprehensive_health_check()
        
        # Show system resource status
        system_health = detailed_health.get('system_health', {})
        print("System Resources:")
        for resource, info in system_health.items():
            if isinstance(info, dict) and 'status' in info:
                status = info['status'].upper()
                if resource == 'cpu':
                    usage = info.get('usage_percent', 0)
                    print(f"  • CPU: {status} ({usage:.1f}% usage)")
                elif resource == 'memory':
                    usage = info.get('usage_percent', 0)
                    available = info.get('available_gb', 0)
                    print(f"  • Memory: {status} ({usage:.1f}% used, {available:.1f}GB available)")
                elif resource == 'disk':
                    usage = info.get('usage_percent', 0)
                    free = info.get('free_gb', 0)
                    print(f"  • Disk: {status} ({usage:.1f}% used, {free:.1f}GB free)")
                elif resource == 'process':
                    memory = info.get('memory_mb', 0)
                    print(f"  • Process: {status} ({memory:.1f}MB memory)")
        
        # Show recommendations if any
        recommendations = detailed_health.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {i}. {rec}")
        
        # Demo 3: Performance Analytics
        print("\n📊 Demo 3: Performance Analytics")
        print("-" * 40)
        
        # Generate some activity first
        data_layer = bridge.get_data_layer()
        
        # Store some test messages to generate metrics
        try:
            await data_layer.store_chat_message(
                user_id="demo_user",
                session_id="demo_session",
                message={
                    "role": "user",
                    "content": "Hello, this is a demo message!",
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            )
            print("✅ Generated test activity for analytics")
        except Exception as e:
            print(f"⚠️  Could not generate test activity: {e}")
        
        analytics = await monitor.get_performance_analytics()
        
        print(f"Analytics Time Range: {analytics['time_range_hours']} hours")
        
        # Show performance trends
        performance_trends = analytics.get('performance_trends', {})
        if performance_trends:
            print("Performance Trends:")
            for operation, stats in performance_trends.items():
                avg_time = stats.get('current_avg_ms', 0)
                operations_count = stats.get('total_operations', 0)
                print(f"  • {operation}: {avg_time:.1f}ms avg ({operations_count} operations)")
        else:
            print("  No performance data available yet")
        
        # Demo 4: Issue Diagnosis
        print("\n🛠️  Demo 4: Automated Issue Diagnosis")
        print("-" * 40)
        
        diagnosis = await diagnose_bridge_issues(bridge)
        
        issues_found = diagnosis.get('issues_found', 0)
        print(f"Issues Detected: {issues_found}")
        
        if issues_found > 0:
            print("Issue Analysis:")
            diagnostics = diagnosis.get('diagnostics', [])
            for i, diagnostic in enumerate(diagnostics[:2], 1):  # Show first 2
                issue = diagnostic.get('issue', {})
                issue_type = issue.get('type', 'unknown')
                severity = issue.get('severity', 'unknown')
                message = issue.get('message', 'No details')
                print(f"  {i}. {issue_type.title()} Issue ({severity}): {message}")
                
                # Show resolution suggestions
                suggestions = diagnostic.get('resolution_suggestions', [])
                if suggestions:
                    print(f"     Resolution: {suggestions[0]}")
        else:
            print("🎉 No issues detected - system is running optimally!")
        
        # Show priority actions if any
        priority_actions = diagnosis.get('priority_actions', [])
        if priority_actions:
            print(f"\n⚡ Priority Actions:")
            for i, action in enumerate(priority_actions[:2], 1):
                print(f"  {i}. {action}")
        
        # Demo 5: Background Monitoring
        print("\n🔄 Demo 5: Background Monitoring")
        print("-" * 40)
        
        print("Starting background monitoring...")
        monitor.start_background_monitoring(interval_minutes=0.05)  # Very short for demo
        
        import time
        print("Monitoring in background for 3 seconds...")
        time.sleep(3)
        
        monitor.stop_background_monitoring()
        print("✅ Background monitoring stopped")
        
        # Final summary
        print(f"\n📋 Health Monitoring Summary")
        print("=" * 40)
        print(f"Overall System Status: {detailed_health['overall_status'].upper()}")
        print(f"Optimization Score: {detailed_health['optimization_score']}/100")
        print(f"Components Monitored: {len(detailed_health.get('component_health', {}))}")
        print(f"Issues Detected: {issues_found}")
        print(f"Recommendations Generated: {len(recommendations)}")
        
        # Cleanup
        await bridge.close()
        print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())