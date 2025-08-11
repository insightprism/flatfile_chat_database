# Troubleshooting Guide - Chat Application Bridge System

## Overview

This guide provides comprehensive troubleshooting information for common issues encountered when using the Chat Application Bridge System. It includes diagnostic procedures, common solutions, and tools for resolving problems quickly.

## Quick Diagnostic Commands

### System Health Check

```python
# health_check.py - Quick system health diagnostic
import asyncio
from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor

async def quick_health_check(storage_path: str):
    """Perform quick health check."""
    try:
        bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
        
        # Basic health check
        health = await bridge.health_check()
        print(f"System Status: {health['status']}")
        
        if health['status'] == 'error':
            print("Errors:")
            for error in health.get('errors', []):
                print(f"  - {error}")
        
        if health.get('warnings'):
            print("Warnings:")
            for warning in health['warnings']:
                print(f"  - {warning}")
        
        # Performance check
        perf = health.get('performance_metrics', {})
        if 'uptime_seconds' in perf:
            print(f"Uptime: {perf['uptime_seconds']:.2f} seconds")
        
        await bridge.close()
        return health['status'] == 'healthy'
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

# Usage: asyncio.run(quick_health_check("./your_data_path"))
```

### Configuration Validation

```python
# config_validation.py - Validate configuration
from ff_chat_integration import ChatAppStorageConfig, FFChatConfigFactory

def validate_configuration(storage_path: str, **config_options):
    """Validate configuration settings."""
    try:
        config = ChatAppStorageConfig(
            storage_path=storage_path,
            **config_options
        )
        
        issues = config.validate()
        
        if not issues:
            print("✅ Configuration is valid")
            return True
        else:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Usage: validate_configuration("./data", performance_mode="balanced")
```

## Common Issues and Solutions

### 1. Bridge Initialization Failures

#### Issue: Bridge fails to initialize

**Symptoms:**
- `InitializationError` when creating bridge
- "Failed to initialize storage" messages
- Bridge not responding to requests

**Diagnostic Steps:**

```python
# diagnose_initialization.py
import asyncio
from pathlib import Path
from ff_chat_integration import FFChatAppBridge, InitializationError

async def diagnose_initialization_failure(storage_path: str):
    """Diagnose bridge initialization failures."""
    
    print("=== Bridge Initialization Diagnostics ===")
    
    # Check 1: Storage path accessibility
    storage_path_obj = Path(storage_path)
    
    print(f"Storage path: {storage_path}")
    print(f"Path exists: {storage_path_obj.exists()}")
    print(f"Path is directory: {storage_path_obj.is_dir()}")
    print(f"Path is writable: {storage_path_obj.parent.exists() and (storage_path_obj.parent.stat().st_mode & 0o200)}")
    
    # Check 2: Parent directory permissions
    if not storage_path_obj.parent.exists():
        print("❌ Parent directory does not exist")
        print(f"   Create with: mkdir -p {storage_path_obj.parent}")
        return
    
    # Check 3: Disk space
    import shutil
    total, used, free = shutil.disk_usage(storage_path_obj.parent)
    free_gb = free // (1024**3)
    
    print(f"Free disk space: {free_gb} GB")
    if free_gb < 1:
        print("❌ Insufficient disk space (need at least 1GB)")
        return
    
    # Check 4: Try minimal configuration
    try:
        print("Testing minimal bridge creation...")
        bridge = await FFChatAppBridge.create_for_chat_app(
            storage_path,
            {"performance_mode": "balanced", "cache_size_mb": 50}
        )
        print("✅ Minimal bridge creation successful")
        await bridge.close()
    except InitializationError as e:
        print(f"❌ Initialization failed: {e}")
        print(f"Component: {e.context.get('component', 'unknown')}")
        print(f"Step: {e.context.get('initialization_step', 'unknown')}")
        
        # Check specific failure points
        if 'storage_initialization' in str(e):
            print("💡 Storage initialization issue:")
            print("   - Check file permissions")
            print("   - Verify storage path is accessible")
            print("   - Check for file locking conflicts")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# Usage: asyncio.run(diagnose_initialization_failure("./your_data"))
```

**Common Solutions:**

1. **Permission Issues**:
   ```bash
   # Fix permissions
   chmod 755 /path/to/storage/directory
   chown -R $USER:$GROUP /path/to/storage/directory
   ```

2. **Missing Directory**:
   ```python
   # Create directory programmatically
   from pathlib import Path
   Path("./chat_data").mkdir(parents=True, exist_ok=True)
   ```

3. **Configuration Issues**:
   ```python
   # Use minimal configuration
   bridge = await FFChatAppBridge.create_for_chat_app(
       "./chat_data",
       {
           "performance_mode": "balanced",
           "cache_size_mb": 50,
           "enable_compression": False
       }
   )
   ```

### 2. Performance Issues

#### Issue: Slow response times

**Symptoms:**
- Operations taking >200ms consistently
- High memory usage
- Cache hit rate <50%

**Diagnostic Steps:**

```python
# diagnose_performance.py
async def diagnose_performance_issues(bridge):
    """Diagnose performance issues."""
    
    print("=== Performance Diagnostics ===")
    
    # Get performance metrics
    data_layer = bridge.get_data_layer()
    metrics = data_layer.get_performance_metrics()
    
    # Check operation metrics
    if "operation_metrics" in metrics:
        for operation, stats in metrics["operation_metrics"].items():
            avg_ms = stats.get("average_ms", 0)
            print(f"{operation}: {avg_ms:.2f}ms average")
            
            if avg_ms > 100:
                print(f"  ⚠️ {operation} is slow")
                
                # Specific recommendations
                if operation == "store_chat_message":
                    print("    💡 Try: Increase cache size or use 'speed' mode")
                elif operation == "get_chat_history":
                    print("    💡 Try: Enable compression or reduce batch size")
                elif "search" in operation:
                    print("    💡 Try: Optimize search indices or reduce result limit")
    
    # Check cache performance
    if "cache_stats" in metrics:
        cache = metrics["cache_stats"]
        hit_rate = cache.get("cache_hit_rate", 0)
        print(f"Cache hit rate: {hit_rate:.2%}")
        
        if hit_rate < 0.5:
            print("  ⚠️ Low cache hit rate")
            print("    💡 Try: Increase cache size or adjust cache TTL")
    
    # Check system resources
    if "system_metrics" in metrics:
        system = metrics["system_metrics"]
        memory_mb = system.get("memory_usage_mb", 0)
        print(f"Memory usage: {memory_mb:.1f} MB")
        
        if memory_mb > 500:
            print("  ⚠️ High memory usage")
            print("    💡 Try: Reduce cache size or enable compression")
    
    # Configuration recommendations
    config = bridge.get_standardized_config()
    perf_mode = config["performance"]["mode"]
    cache_mb = config["performance"]["cache_size_mb"]
    
    print(f"\nCurrent configuration:")
    print(f"  Performance mode: {perf_mode}")
    print(f"  Cache size: {cache_mb} MB")
    
    if perf_mode != "speed" and avg_ms > 100:
        print("  💡 Consider switching to 'speed' mode")
    
    if cache_mb < 100:
        print("  💡 Consider increasing cache size")

# Usage: await diagnose_performance_issues(your_bridge)
```

**Performance Solutions:**

1. **Optimize Configuration**:
   ```python
   # High-performance configuration
   bridge = await FFChatAppBridge.create_for_chat_app(
       "./data",
       {
           "performance_mode": "speed",
           "cache_size_mb": 300,
           "enable_compression": True,
           "message_batch_size": 200
       }
   )
   ```

2. **Reduce Memory Usage**:
   ```python
   # Memory-optimized configuration
   bridge = await FFChatAppBridge.create_for_chat_app(
       "./data",
       {
           "cache_size_mb": 50,
           "enable_compression": True,
           "max_session_size_mb": 25
       }
   )
   ```

### 3. Data Access Issues

#### Issue: Messages not retrieving correctly

**Symptoms:**
- `get_chat_history()` returns empty results
- Search operations fail
- Data appears to be stored but not retrievable

**Diagnostic Steps:**

```python
# diagnose_data_access.py
async def diagnose_data_access_issues(bridge, user_id: str, session_id: str):
    """Diagnose data access issues."""
    
    print("=== Data Access Diagnostics ===")
    
    data_layer = bridge.get_data_layer()
    
    # Check 1: Test basic connectivity
    try:
        health = await bridge.health_check()
        storage_accessible = health.get("storage_accessible", False)
        print(f"Storage accessible: {storage_accessible}")
        
        if not storage_accessible:
            print("❌ Storage not accessible - check initialization")
            return
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Check 2: Verify user/session existence
    try:
        # Try to create a test message
        test_result = await data_layer.store_chat_message(
            user_id, session_id,
            {"role": "system", "content": "Diagnostic test message"}
        )
        
        print(f"Test message storage: {'✅' if test_result['success'] else '❌'}")
        
        if not test_result["success"]:
            print(f"Storage error: {test_result['error']}")
            
            # Check if it's a user/session issue
            if "user" in test_result["error"].lower():
                print("💡 Try creating the user first:")
                print(f"   await data_layer.storage.create_user('{user_id}', {{'name': 'Test User'}})")
            
            if "session" in test_result["error"].lower():
                print("💡 Try creating the session first:")
                print(f"   session_id = await data_layer.storage.create_session('{user_id}', 'Test Session')")
                
    except Exception as e:
        print(f"❌ Test message failed: {e}")
    
    # Check 3: Try retrieving messages
    try:
        history = await data_layer.get_chat_history(user_id, session_id, limit=5)
        
        if history["success"]:
            message_count = len(history["data"]["messages"])
            print(f"✅ Retrieved {message_count} messages")
            
            if message_count == 0:
                print("💡 No messages found - check if messages were stored correctly")
        else:
            print(f"❌ History retrieval failed: {history['error']}")
            
    except Exception as e:
        print(f"❌ History retrieval error: {e}")
    
    # Check 4: Test search functionality
    try:
        search_result = await data_layer.search_conversations(
            user_id, "test", {"search_type": "text", "limit": 5}
        )
        
        if search_result["success"]:
            result_count = len(search_result["data"]["results"])
            print(f"✅ Search returned {result_count} results")
        else:
            print(f"❌ Search failed: {search_result['error']}")
            
    except Exception as e:
        print(f"❌ Search error: {e}")

# Usage: await diagnose_data_access_issues(bridge, "test_user", "test_session")
```

**Data Access Solutions:**

1. **Ensure User and Session Exist**:
   ```python
   # Always create user and session before storing messages
   await data_layer.storage.create_user(user_id, {"name": "User Name"})
   session_id = await data_layer.storage.create_session(user_id, "Session Name")
   ```

2. **Check Message Format**:
   ```python
   # Correct message format
   message = {
       "role": "user",  # Must be "user", "assistant", or "system"
       "content": "Message content",
       "timestamp": "2024-01-15T10:00:00Z"  # Optional but recommended
   }
   ```

### 4. Configuration Issues

#### Issue: Configuration validation errors

**Symptoms:**
- `ConfigurationError` during bridge creation
- Invalid parameter warnings
- Features not working as expected

**Diagnostic Steps:**

```python
# diagnose_config_issues.py
from ff_chat_integration import FFChatConfigFactory, ConfigurationError

def diagnose_configuration_issues(storage_path: str, config_options: dict):
    """Diagnose configuration issues."""
    
    print("=== Configuration Diagnostics ===")
    
    factory = FFChatConfigFactory()
    
    try:
        # Try to create configuration
        from ff_chat_integration import ChatAppStorageConfig
        config = ChatAppStorageConfig(
            storage_path=storage_path,
            **config_options
        )
        
        print("✅ Configuration created successfully")
        
        # Validate configuration
        validation_results = factory.validate_and_optimize(config)
        
        print(f"Configuration valid: {validation_results['valid']}")
        print(f"Optimization score: {validation_results['optimization_score']}/100")
        
        if validation_results["errors"]:
            print("❌ Configuration errors:")
            for error in validation_results["errors"]:
                print(f"    - {error}")
        
        if validation_results["warnings"]:
            print("⚠️ Configuration warnings:")
            for warning in validation_results["warnings"]:
                print(f"    - {warning}")
        
        if validation_results["recommendations"]:
            print("💡 Configuration recommendations:")
            for rec in validation_results["recommendations"]:
                print(f"    - {rec}")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        print(f"Context: {e.context}")
        
        print("💡 Suggestions:")
        for suggestion in e.suggestions:
            print(f"    - {suggestion}")
            
        # Common configuration fixes
        print("\n🔧 Common fixes:")
        
        if "performance_mode" in str(e):
            print("    - Use: 'speed', 'balanced', or 'quality'")
        
        if "cache_size" in str(e):
            print("    - Use cache_size_mb between 10 and 2000")
        
        if "storage_path" in str(e):
            print("    - Ensure storage path is accessible")
            print("    - Use absolute paths for production")
            
    except Exception as e:
        print(f"❌ Unexpected configuration error: {e}")

# Usage: diagnose_configuration_issues("./data", {"performance_mode": "invalid"})
```

**Configuration Solutions:**

1. **Valid Configuration Parameters**:
   ```python
   # Example of valid configuration
   valid_config = {
       "performance_mode": "balanced",  # "speed", "balanced", "quality"
       "cache_size_mb": 100,           # 10-2000
       "enable_vector_search": True,
       "enable_streaming": True,
       "enable_analytics": True,
       "enable_compression": False,
       "backup_enabled": False,
       "environment": "development"     # "development", "production", "test"
   }
   ```

2. **Use Configuration Presets**:
   ```python
   # Use tested presets instead of manual configuration
   bridge = await FFChatAppBridge.create_from_preset(
       "development",  # or "production", "high_performance", etc.
       "./data"
   )
   ```

### 5. Memory and Resource Issues

#### Issue: High memory usage or memory leaks

**Symptoms:**
- Memory usage continuously increasing
- System becoming sluggish
- Out of memory errors

**Diagnostic Steps:**

```python
# diagnose_memory_issues.py
import asyncio
import gc
import psutil
import os

async def diagnose_memory_issues(bridge, duration_minutes: int = 5):
    """Monitor memory usage over time."""
    
    print("=== Memory Usage Diagnostics ===")
    
    process = psutil.Process(os.getpid())
    
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    data_layer = bridge.get_data_layer()
    
    # Perform operations and monitor memory
    for minute in range(duration_minutes):
        # Simulate workload
        await _simulate_workload(data_layer)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - baseline_memory
        
        print(f"Minute {minute + 1}: {current_memory:.1f} MB (+{memory_growth:.1f} MB)")
        
        # Check if memory is growing excessively
        if memory_growth > 100:  # 100MB growth
            print("⚠️ Excessive memory growth detected!")
            
            # Get system metrics
            system_metrics = data_layer.get_performance_metrics().get("system_metrics", {})
            cache_size = system_metrics.get("cache_size", 0)
            
            print(f"Cache size: {cache_size} items")
            
            if cache_size > 1000:
                print("💡 Try: Reduce cache size or enable cache TTL")
        
        await asyncio.sleep(60)  # Wait 1 minute
    
    # Final memory check
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    total_growth = final_memory - baseline_memory
    
    print(f"\nFinal memory: {final_memory:.1f} MB")
    print(f"Total growth: {total_growth:.1f} MB")
    
    if total_growth > 50:
        print("⚠️ Significant memory growth - possible memory leak")
        await _suggest_memory_optimizations(bridge)

async def _simulate_workload(data_layer):
    """Simulate typical workload."""
    user_id = "memory_test_user"
    
    try:
        await data_layer.storage.create_user(user_id, {"name": "Memory Test"})
        session_id = await data_layer.storage.create_session(user_id, "Memory Test Session")
        
        # Store and retrieve messages
        for i in range(10):
            await data_layer.store_chat_message(
                user_id, session_id,
                {"role": "user", "content": f"Memory test message {i}"}
            )
        
        await data_layer.get_chat_history(user_id, session_id, limit=10)
        
    except Exception:
        pass  # Ignore errors in diagnostic workload

async def _suggest_memory_optimizations(bridge):
    """Suggest memory optimizations."""
    config = bridge.get_standardized_config()
    
    print("\n🔧 Memory Optimization Suggestions:")
    
    cache_mb = config["performance"]["cache_size_mb"]
    if cache_mb > 200:
        print(f"    - Reduce cache size from {cache_mb}MB to 100MB")
    
    if not config["features"]["compression"]:
        print("    - Enable compression to reduce memory usage")
    
    print("    - Implement periodic cache cleanup")
    print("    - Monitor for resource leaks in application code")

# Usage: await diagnose_memory_issues(your_bridge, duration_minutes=5)
```

**Memory Solutions:**

1. **Optimize Cache Configuration**:
   ```python
   # Memory-efficient configuration
   bridge = await FFChatAppBridge.create_for_chat_app(
       "./data",
       {
           "cache_size_mb": 50,        # Smaller cache
           "enable_compression": True,  # Compress cached data
           "max_session_size_mb": 25   # Limit session size
       }
   )
   ```

2. **Implement Periodic Cleanup**:
   ```python
   # Periodic resource cleanup
   async def periodic_cleanup(bridge):
       while True:
           # Force garbage collection
           import gc
           gc.collect()
           
           # Check memory usage
           import psutil, os
           process = psutil.Process(os.getpid())
           memory_mb = process.memory_info().rss / 1024 / 1024
           
           if memory_mb > 500:  # 500MB threshold
               print(f"High memory usage: {memory_mb:.1f}MB")
               # Implement cache cleanup logic
           
           await asyncio.sleep(300)  # Check every 5 minutes
   ```

## Advanced Diagnostics

### Comprehensive System Check

```python
# comprehensive_diagnostics.py
import asyncio
from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor

async def comprehensive_system_check(storage_path: str):
    """Comprehensive system diagnostic."""
    
    print("=" * 60)
    print("COMPREHENSIVE SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    try:
        # Phase 1: Bridge Creation
        print("\n1. BRIDGE CREATION TEST")
        print("-" * 30)
        
        bridge = await FFChatAppBridge.create_for_chat_app(storage_path)
        print("✅ Bridge created successfully")
        
        # Phase 2: Health Check
        print("\n2. HEALTH CHECK")
        print("-" * 30)
        
        health = await bridge.health_check()
        print(f"Status: {health['status']}")
        print(f"Bridge initialized: {health['bridge_initialized']}")
        print(f"Storage accessible: {health['storage_accessible']}")
        print(f"Write permissions: {health['write_permissions']}")
        print(f"Disk space sufficient: {health['disk_space_sufficient']}")
        
        if health['errors']:
            print("Errors found:")
            for error in health['errors']:
                print(f"  - {error}")
        
        # Phase 3: Capabilities Check
        print("\n3. CAPABILITIES CHECK")
        print("-" * 30)
        
        capabilities = await bridge.get_capabilities()
        print(f"Vector search: {capabilities['vector_search']}")
        print(f"Streaming: {capabilities['streaming']}")
        print(f"Analytics: {capabilities['analytics']}")
        print(f"Storage features: {', '.join(capabilities['storage_features'])}")
        
        # Phase 4: Data Operations Test
        print("\n4. DATA OPERATIONS TEST")
        print("-" * 30)
        
        data_layer = bridge.get_data_layer()
        
        # Test user creation
        test_user = "diagnostic_user"
        await data_layer.storage.create_user(test_user, {"name": "Diagnostic User"})
        print("✅ User creation successful")
        
        # Test session creation
        session_id = await data_layer.storage.create_session(test_user, "Diagnostic Session")
        print("✅ Session creation successful")
        
        # Test message storage
        result = await data_layer.store_chat_message(
            test_user, session_id,
            {"role": "user", "content": "Diagnostic test message"}
        )
        
        if result["success"]:
            print("✅ Message storage successful")
            storage_time = result["metadata"]["performance_metrics"]["storage_time_ms"]
            print(f"   Storage time: {storage_time:.2f}ms")
        else:
            print(f"❌ Message storage failed: {result['error']}")
        
        # Test message retrieval
        history = await data_layer.get_chat_history(test_user, session_id)
        
        if history["success"]:
            message_count = len(history["data"]["messages"])
            print(f"✅ Message retrieval successful ({message_count} messages)")
        else:
            print(f"❌ Message retrieval failed: {history['error']}")
        
        # Phase 5: Performance Check
        print("\n5. PERFORMANCE CHECK")
        print("-" * 30)
        
        metrics = data_layer.get_performance_metrics()
        
        if "operation_metrics" in metrics:
            for operation, stats in metrics["operation_metrics"].items():
                avg_ms = stats.get("average_ms", 0)
                print(f"{operation}: {avg_ms:.2f}ms average")
        
        if "cache_stats" in metrics:
            cache = metrics["cache_stats"]
            hit_rate = cache.get("cache_hit_rate", 0)
            print(f"Cache hit rate: {hit_rate:.2%}")
        
        # Phase 6: Monitoring Test
        print("\n6. MONITORING TEST")
        print("-" * 30)
        
        monitor = FFIntegrationHealthMonitor(bridge)
        comprehensive_health = await monitor.comprehensive_health_check()
        
        print(f"Overall status: {comprehensive_health['overall_status']}")
        print(f"Optimization score: {comprehensive_health['optimization_score']}/100")
        
        # Phase 7: Final Assessment
        print("\n7. FINAL ASSESSMENT")
        print("-" * 30)
        
        issues_found = 0
        
        if health['status'] == 'error':
            issues_found += 1
            print("❌ System health issues")
        
        if not result.get("success", False):
            issues_found += 1
            print("❌ Data operation issues")
        
        if avg_ms > 200:
            issues_found += 1
            print("❌ Performance issues")
        
        if comprehensive_health['optimization_score'] < 60:
            issues_found += 1
            print("❌ Optimization issues")
        
        if issues_found == 0:
            print("🎉 ALL SYSTEMS OPERATIONAL!")
            print("   No issues found - system is healthy")
        else:
            print(f"⚠️  {issues_found} issue(s) found - see details above")
        
        await bridge.close()
        return issues_found == 0
        
    except Exception as e:
        print(f"❌ Comprehensive diagnostic failed: {e}")
        return False

# Usage: success = await comprehensive_system_check("./your_data_path")
```

### Log Analysis Tools

```python
# log_analysis.py
import re
import json
from datetime import datetime
from collections import defaultdict

def analyze_bridge_logs(log_file_path: str):
    """Analyze bridge system logs for issues."""
    
    print("=== LOG ANALYSIS ===")
    
    error_patterns = [
        r"ERROR.*?(InitializationError|ConfigurationError|StorageError)",
        r"WARNING.*?(performance|cache|memory)",
        r"FAILED.*?(storage|retrieval|search)"
    ]
    
    performance_patterns = [
        r"storage_time_ms.*?(\d+\.?\d*)",
        r"cache_hit.*?(true|false)",
        r"average_ms.*?(\d+\.?\d*)"
    ]
    
    error_counts = defaultdict(int)
    performance_metrics = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check for errors
                for pattern in error_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        error_type = match.group(1) if match.groups() else "Unknown"
                        error_counts[error_type] += 1
                
                # Check for performance metrics
                for pattern in performance_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and match.groups():
                        performance_metrics.append(match.group(1))
    
        # Report findings
        if error_counts:
            print("Errors found:")
            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count} occurrences")
        else:
            print("✅ No errors found in logs")
        
        if performance_metrics:
            avg_perf = sum(float(m) for m in performance_metrics if m.replace('.', '').isdigit()) / len(performance_metrics)
            print(f"Average performance metric: {avg_perf:.2f}ms")
            
            if avg_perf > 200:
                print("⚠️ Performance may be degraded")
        
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
    except Exception as e:
        print(f"❌ Log analysis failed: {e}")

# Usage: analyze_bridge_logs("/path/to/bridge.log")
```

## Error Code Reference

### Configuration Errors (CONFIG_*)

- **CONFIG_ERROR**: General configuration issue
  - **Solution**: Check configuration parameters against API reference
  
- **CONFIG_VALIDATION_FAILED**: Configuration validation failed
  - **Solution**: Use `validate_configuration()` to identify specific issues
  
- **CONFIG_TEMPLATE_NOT_FOUND**: Invalid template name
  - **Solution**: Use `FFChatConfigFactory.list_templates()` to see available templates

### Initialization Errors (INIT_*)

- **INIT_STORAGE_FAILED**: Storage initialization failed
  - **Solution**: Check storage path permissions and disk space
  
- **INIT_BRIDGE_FAILED**: Bridge initialization failed
  - **Solution**: Verify configuration and system requirements
  
- **INIT_TIMEOUT**: Initialization timed out
  - **Solution**: Check system resources and network connectivity

### Storage Errors (STORAGE_*)

- **STORAGE_ACCESS_DENIED**: Storage access denied
  - **Solution**: Check file permissions and user privileges
  
- **STORAGE_DISK_FULL**: Insufficient disk space
  - **Solution**: Free up disk space or move to larger storage
  
- **STORAGE_CORRUPTION**: Data corruption detected
  - **Solution**: Restore from backup or recreate storage

### Performance Errors (PERF_*)

- **PERF_DEGRADED**: Performance below acceptable thresholds
  - **Solution**: Optimize configuration or increase resources
  
- **PERF_MEMORY_HIGH**: High memory usage
  - **Solution**: Reduce cache size or enable compression

## Support Resources

### Getting Help

1. **Check Documentation**:
   - API Reference
   - Integration Examples
   - Migration Guide

2. **Run Diagnostics**:
   ```bash
   python -c "import asyncio; from diagnostics import comprehensive_system_check; asyncio.run(comprehensive_system_check('./your_path'))"
   ```

3. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Collect System Information**:
   ```python
   # system_info.py
   import platform
   import sys
   from ff_chat_integration import __version__
   
   print(f"Python: {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"Bridge Version: {__version__}")
   ```

### Performance Optimization Checklist

- [ ] Use appropriate performance mode for your use case
- [ ] Set optimal cache size (100-200MB for most applications)
- [ ] Enable compression for large datasets
- [ ] Use batching for bulk operations
- [ ] Monitor memory usage and implement cleanup
- [ ] Use streaming for large conversations
- [ ] Implement proper error handling
- [ ] Monitor system health regularly

### Common Configuration Patterns

**Development Environment**:
```python
bridge = await FFChatAppBridge.create_from_preset("development", "./dev_data")
```

**Production Environment**:
```python
bridge = await FFChatAppBridge.create_from_preset(
    "production", 
    "/var/lib/app/data",
    {"cache_size_mb": 300, "backup_enabled": True}
)
```

**High Performance**:
```python
bridge = await FFChatAppBridge.create_from_preset(
    "high_performance",
    "./data",
    {"performance_mode": "speed", "cache_size_mb": 500}
)
```

**Resource Constrained**:
```python
bridge = await FFChatAppBridge.create_from_preset(
    "lightweight",
    "./data",
    {"cache_size_mb": 25, "enable_compression": True}
)
```

This troubleshooting guide provides comprehensive diagnostic tools and solutions for the most common issues encountered when using the Chat Application Bridge System.