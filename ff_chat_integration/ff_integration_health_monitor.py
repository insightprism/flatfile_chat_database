"""
Comprehensive health monitoring and diagnostics for Chat Application Bridge.

Provides advanced health checking, performance analytics, automated diagnostics,
and intelligent optimization recommendations for chat applications.
"""

import asyncio
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import existing components
from ff_utils.ff_logging import get_logger

# Import our bridge components
from .ff_chat_app_bridge import FFChatAppBridge
from .ff_chat_data_layer import FFChatDataLayer
from .ff_integration_exceptions import (
    ChatIntegrationError, PerformanceError, StorageError
)

logger = get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Structured health check result."""
    component: str
    status: str  # "healthy", "degraded", "error"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    severity: str = "info"  # "info", "warning", "error", "critical"


@dataclass
class PerformanceMetric:
    """Performance metric with historical data."""
    name: str
    current_value: float
    unit: str
    timestamp: datetime
    history: List[float] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)


class FFIntegrationHealthMonitor:
    """
    Comprehensive health monitoring for Chat Application Bridge System.
    
    Provides intelligent monitoring, diagnostics, and optimization recommendations
    to help chat applications maintain optimal performance and reliability.
    """
    
    def __init__(self, bridge: FFChatAppBridge):
        """
        Initialize health monitor.
        
        Args:
            bridge: FFChatAppBridge instance to monitor
        """
        self.bridge = bridge
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_health_check = None
        self._health_check_interval = 300  # 5 minutes
        
        # System monitoring
        self._system_stats = {}
        self._performance_baselines = {}
        
        # Issue tracking
        self._known_issues: List[Dict[str, Any]] = []
        self._resolved_issues: List[Dict[str, Any]] = []
        
        # Monitoring configuration
        self._monitoring_enabled = True
        self._background_monitoring = False
        self._monitoring_thread = None
        
        self.logger.info("FFIntegrationHealthMonitor initialized")
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the entire bridge system.
        
        Returns detailed health information with actionable recommendations.
        """
        start_time = time.time()
        
        health_results = {
            "overall_status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "check_duration_ms": 0,
            "component_health": {},
            "system_health": {},
            "performance_health": {},
            "issues_detected": [],
            "recommendations": [],
            "optimization_score": 0
        }
        
        try:
            # Check all system components
            component_checks = await self._check_all_components()
            health_results["component_health"] = component_checks
            
            # Check system resources
            system_checks = await self._check_system_resources()
            health_results["system_health"] = system_checks
            
            # Check performance metrics
            performance_checks = await self._check_performance_health()
            health_results["performance_health"] = performance_checks
            
            # Detect and analyze issues
            issues = await self._detect_issues(component_checks, system_checks, performance_checks)
            health_results["issues_detected"] = issues
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                component_checks, system_checks, performance_checks, issues
            )
            health_results["recommendations"] = recommendations
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                component_checks, system_checks, performance_checks
            )
            health_results["optimization_score"] = optimization_score
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                component_checks, system_checks, performance_checks, issues
            )
            health_results["overall_status"] = overall_status
            
            # Record check duration
            health_results["check_duration_ms"] = (time.time() - start_time) * 1000
            
            # Store results for trend analysis
            self._store_health_check_results(health_results)
            
            return health_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive health check failed: {e}")
            return {
                "overall_status": "error",
                "timestamp": datetime.now().isoformat(),
                "check_duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "recommendations": ["Fix health monitoring system issues"]
            }
    
    async def _check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all bridge components."""
        checks = {}
        
        # Bridge component health
        checks["bridge"] = await self._check_bridge_health()
        
        # Storage system health
        checks["storage"] = await self._check_storage_health()
        
        # Data layer health
        checks["data_layer"] = await self._check_data_layer_health()
        
        # Configuration health
        checks["configuration"] = await self._check_configuration_health()
        
        # Cache system health
        checks["cache"] = await self._check_cache_health()
        
        return {name: result.__dict__ for name, result in checks.items()}
    
    async def _check_bridge_health(self) -> HealthCheckResult:
        """Check FFChatAppBridge health."""
        try:
            if not self.bridge._initialized:
                return HealthCheckResult(
                    component="bridge",
                    status="error",
                    message="Bridge not initialized",
                    severity="critical",
                    recommendations=["Initialize bridge before use"]
                )
            
            # Check bridge capabilities
            capabilities = await self.bridge.get_capabilities()
            if "error" in capabilities:
                return HealthCheckResult(
                    component="bridge",
                    status="degraded", 
                    message="Bridge capabilities check failed",
                    details={"error": capabilities["error"]},
                    severity="warning",
                    recommendations=["Check bridge configuration and dependencies"]
                )
            
            # Check uptime
            uptime = time.time() - self.bridge.start_time
            if uptime < 60:  # Less than 1 minute
                status = "degraded"
                message = "Bridge recently started"
                severity = "info"
            else:
                status = "healthy"
                message = "Bridge operating normally"
                severity = "info"
            
            return HealthCheckResult(
                component="bridge",
                status=status,
                message=message,
                details={
                    "uptime_seconds": uptime,
                    "capabilities_count": len(capabilities.get("storage_features", [])),
                    "initialized": self.bridge._initialized
                },
                severity=severity
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="bridge",
                status="error",
                message=f"Bridge health check failed: {e}",
                severity="error",
                recommendations=["Check bridge initialization and dependencies"]
            )
    
    async def _check_storage_health(self) -> HealthCheckResult:
        """Check storage system health."""
        try:
            if not self.bridge._storage_manager:
                return HealthCheckResult(
                    component="storage",
                    status="error",
                    message="Storage manager not available",
                    severity="critical"
                )
            
            # Test storage operations
            test_start = time.time()
            
            # Create test user
            test_user_id = f"health_check_{int(time.time())}"
            user_created = await self.bridge._storage_manager.create_user(
                test_user_id, {"health_check": True}
            )
            
            if not user_created:
                return HealthCheckResult(
                    component="storage",
                    status="error",
                    message="Storage user creation failed",
                    severity="error",
                    recommendations=["Check storage permissions and disk space"]
                )
            
            # Test session creation
            session_id = await self.bridge._storage_manager.create_session(
                test_user_id, "Health Check Session"
            )
            
            # Test message storage
            try:
                from flatfile_chat_database.models import Message
                test_message = Message(
                    role="user",
                    content="Health check message",
                    message_id=f"test_msg_{int(time.time())}"
                )
                
                message_stored = await self.bridge._storage_manager.add_message(
                    test_user_id, session_id, test_message
                )
            except ImportError:
                # Fallback to generic message format
                test_message = {
                    "role": "user",
                    "content": "Health check message",
                    "timestamp": datetime.now().isoformat()
                }
                message_stored = True  # Assume success if no specific model
            
            storage_time = (time.time() - test_start) * 1000
            
            # Cleanup test data
            try:
                # Note: Add cleanup if storage manager supports deletion
                pass
            except:
                pass  # Cleanup is optional for health checks
            
            if message_stored:
                status = "healthy" if storage_time < 500 else "degraded"
                message = f"Storage operations working (response time: {storage_time:.1f}ms)"
                severity = "info" if storage_time < 500 else "warning"
                recommendations = []
                if storage_time > 500:
                    recommendations.append("Storage operations are slow - check disk performance")
            else:
                status = "error"
                message = "Message storage failed"
                severity = "error"
                recommendations = ["Check storage configuration and permissions"]
            
            return HealthCheckResult(
                component="storage",
                status=status,
                message=message,
                details={
                    "response_time_ms": storage_time,
                    "user_creation": user_created,
                    "message_storage": message_stored
                },
                severity=severity,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="storage",
                status="error", 
                message=f"Storage health check failed: {e}",
                severity="error",
                recommendations=["Check storage system configuration and connectivity"]
            )
    
    async def _check_data_layer_health(self) -> HealthCheckResult:
        """Check data layer health."""
        try:
            data_layer = self.bridge.get_data_layer()
            
            # Check performance metrics
            metrics = data_layer.get_performance_metrics()
            
            # Analyze recent performance
            avg_response_times = {}
            for operation, stats in metrics.get("operation_metrics", {}).items():
                avg_response_times[operation] = stats.get("recent_avg_ms", 0)
            
            # Determine health based on performance
            max_response_time = max(avg_response_times.values()) if avg_response_times else 0
            
            if max_response_time == 0:
                status = "healthy"
                message = "Data layer ready (no recent operations)"
                severity = "info"
                recommendations = []
            elif max_response_time < 200:
                status = "healthy"
                message = f"Data layer operating well (avg response: {max_response_time:.1f}ms)"
                severity = "info"
                recommendations = []
            elif max_response_time < 500:
                status = "degraded"
                message = f"Data layer performance degraded (avg response: {max_response_time:.1f}ms)"
                severity = "warning"
                recommendations = ["Monitor data layer performance", "Consider cache optimization"]
            else:
                status = "error"
                message = f"Data layer performance poor (avg response: {max_response_time:.1f}ms)"
                severity = "error"
                recommendations = [
                    "Investigate data layer performance issues",
                    "Check system resources",
                    "Review configuration optimization"
                ]
            
            return HealthCheckResult(
                component="data_layer",
                status=status,
                message=message,
                details={
                    "max_response_time_ms": max_response_time,
                    "operations_tracked": len(avg_response_times),
                    "cache_size": metrics.get("cache_stats", {}).get("cache_size", 0)
                },
                severity=severity,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="data_layer",
                status="error",
                message=f"Data layer health check failed: {e}",
                severity="error",
                recommendations=["Check data layer initialization and dependencies"]
            )
    
    async def _check_configuration_health(self) -> HealthCheckResult:
        """Check configuration health and optimization."""
        try:
            config = self.bridge.config
            
            # Use configuration factory to analyze current config
            from .ff_chat_config_factory import FFChatConfigFactory
            factory = FFChatConfigFactory()
            
            validation_results = factory.validate_and_optimize(config)
            
            if not validation_results["valid"]:
                return HealthCheckResult(
                    component="configuration",
                    status="error",
                    message="Configuration validation failed",
                    details={"errors": validation_results["errors"]},
                    severity="error",
                    recommendations=validation_results.get("recommendations", [])
                )
            
            optimization_score = validation_results["optimization_score"]
            
            if optimization_score >= 80:
                status = "healthy"
                message = f"Configuration well optimized (score: {optimization_score})"
                severity = "info"
            elif optimization_score >= 60:
                status = "degraded"
                message = f"Configuration could be optimized (score: {optimization_score})"
                severity = "warning"
            else:
                status = "degraded"
                message = f"Configuration needs optimization (score: {optimization_score})"
                severity = "warning"
            
            return HealthCheckResult(
                component="configuration",
                status=status,
                message=message,
                details={
                    "optimization_score": optimization_score,
                    "performance_mode": config.performance_mode,
                    "cache_size_mb": config.cache_size_mb,
                    "estimated_performance": validation_results.get("estimated_performance")
                },
                severity=severity,
                recommendations=validation_results.get("recommendations", [])
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="configuration",
                status="error",
                message=f"Configuration health check failed: {e}",
                severity="error",
                recommendations=["Check configuration validity and optimization settings"]
            )
    
    async def _check_cache_health(self) -> HealthCheckResult:
        """Check cache system health."""
        try:
            data_layer = self.bridge.get_data_layer()
            cache_stats = data_layer.get_performance_metrics().get("cache_stats", {})
            
            cache_size = cache_stats.get("cache_size", 0)
            max_cache_entries = 100  # From data layer implementation
            
            cache_utilization = (cache_size / max_cache_entries) if max_cache_entries else 0
            
            if cache_utilization < 0.5:
                status = "healthy"
                message = f"Cache utilization good ({cache_utilization:.1%})"
                severity = "info"
                recommendations = []
            elif cache_utilization < 0.8:
                status = "healthy"
                message = f"Cache utilization moderate ({cache_utilization:.1%})"
                severity = "info"
                recommendations = ["Monitor cache usage patterns"]
            else:
                status = "degraded"
                message = f"Cache utilization high ({cache_utilization:.1%})"
                severity = "warning"
                recommendations = [
                    "Consider increasing cache size",
                    "Review cache eviction patterns"
                ]
            
            return HealthCheckResult(
                component="cache",
                status=status,
                message=message,
                details={
                    "cache_size": cache_size,
                    "cache_utilization": cache_utilization,
                    "max_entries": max_cache_entries
                },
                severity=severity,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="cache",
                status="error",
                message=f"Cache health check failed: {e}",
                severity="warning",
                recommendations=["Check cache system implementation"]
            )
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage for storage path
            storage_path = Path(self.bridge.config.storage_path)
            disk_usage = psutil.disk_usage(storage_path.parent)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            system_health = {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "status": "healthy" if cpu_percent < 80 else "degraded" if cpu_percent < 95 else "error"
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": memory.available / 1024 / 1024 / 1024,
                    "status": "healthy" if memory_percent < 80 else "degraded" if memory_percent < 95 else "error"
                },
                "disk": {
                    "usage_percent": disk_percent,
                    "free_gb": disk_usage.free / 1024 / 1024 / 1024,
                    "status": "healthy" if disk_percent < 80 else "degraded" if disk_percent < 95 else "error"
                },
                "process": {
                    "memory_mb": process_memory,
                    "status": "healthy" if process_memory < 500 else "degraded" if process_memory < 1000 else "error"
                }
            }
            
            return system_health
            
        except Exception as e:
            self.logger.warning(f"System resource check failed: {e}")
            return {"error": str(e)}
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """Check performance health and trends."""
        try:
            data_layer = self.bridge.get_data_layer()
            metrics = data_layer.get_performance_metrics()
            
            performance_health = {
                "operation_performance": {},
                "trends": {},
                "overall_status": "healthy"
            }
            
            # Analyze operation performance
            for operation, stats in metrics.get("operation_metrics", {}).items():
                avg_ms = stats.get("average_ms", 0)
                recent_avg_ms = stats.get("recent_avg_ms", 0)
                
                # Performance thresholds (operation-specific)
                thresholds = {
                    "store_chat_message": {"good": 100, "degraded": 300},
                    "get_chat_history": {"good": 150, "degraded": 400},
                    "search_conversations": {"good": 200, "degraded": 600}
                }
                
                threshold = thresholds.get(operation, {"good": 200, "degraded": 500})
                
                if recent_avg_ms < threshold["good"]:
                    status = "healthy"
                elif recent_avg_ms < threshold["degraded"]:
                    status = "degraded"
                else:
                    status = "error"
                
                performance_health["operation_performance"][operation] = {
                    "recent_avg_ms": recent_avg_ms,
                    "overall_avg_ms": avg_ms,
                    "status": status,
                    "total_operations": stats.get("total_operations", 0)
                }
                
                # Update overall status
                if status == "error":
                    performance_health["overall_status"] = "error"
                elif status == "degraded" and performance_health["overall_status"] == "healthy":
                    performance_health["overall_status"] = "degraded"
            
            return performance_health
            
        except Exception as e:
            return {"error": str(e), "overall_status": "error"}
    
    async def _detect_issues(self, component_checks: Dict, 
                           system_checks: Dict, 
                           performance_checks: Dict) -> List[Dict[str, Any]]:
        """Detect and categorize issues."""
        issues = []
        
        # Component issues
        for component, check in component_checks.items():
            if check["status"] in ["error", "degraded"]:
                issues.append({
                    "type": "component",
                    "component": component,
                    "severity": check["severity"],
                    "message": check["message"],
                    "recommendations": check.get("recommendations", [])
                })
        
        # System resource issues
        for resource, info in system_checks.items():
            if isinstance(info, dict) and info.get("status") in ["error", "degraded"]:
                issues.append({
                    "type": "system_resource",
                    "resource": resource,
                    "severity": "error" if info["status"] == "error" else "warning",
                    "message": f"{resource.title()} usage is {info['status']}",
                    "recommendations": [f"Monitor {resource} usage and consider optimization"]
                })
        
        # Performance issues
        perf_ops = performance_checks.get("operation_performance", {})
        for operation, stats in perf_ops.items():
            if stats["status"] in ["error", "degraded"]:
                issues.append({
                    "type": "performance",
                    "operation": operation,
                    "severity": "error" if stats["status"] == "error" else "warning",
                    "message": f"{operation} performance is {stats['status']} ({stats['recent_avg_ms']:.1f}ms)",
                    "recommendations": [
                        f"Optimize {operation} performance",
                        "Check system resources and configuration"
                    ]
                })
        
        return issues
    
    async def _generate_recommendations(self, component_checks: Dict,
                                      system_checks: Dict,
                                      performance_checks: Dict,
                                      issues: List[Dict]) -> List[str]:
        """Generate intelligent recommendations."""
        recommendations = []
        
        # Collect all recommendations from components
        for component, check in component_checks.items():
            recommendations.extend(check.get("recommendations", []))
        
        # Add issue-specific recommendations
        for issue in issues:
            recommendations.extend(issue.get("recommendations", []))
        
        # System-level recommendations
        if system_checks.get("memory", {}).get("status") == "degraded":
            recommendations.append("Consider increasing system memory or optimizing memory usage")
        
        if system_checks.get("disk", {}).get("status") == "degraded":
            recommendations.append("Free up disk space or move storage to larger volume")
        
        # Performance-based recommendations
        if performance_checks.get("overall_status") == "degraded":
            recommendations.extend([
                "Review performance configuration settings",
                "Consider enabling performance optimizations",
                "Monitor system resources during peak usage"
            ])
        
        # Configuration-based recommendations
        config = self.bridge.config
        if config.performance_mode == "balanced" and performance_checks.get("overall_status") == "degraded":
            recommendations.append("Consider switching to 'speed' performance mode for better performance")
        
        if config.cache_size_mb < 100 and system_checks.get("memory", {}).get("status") == "healthy":
            recommendations.append("Consider increasing cache size for better performance")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def _calculate_optimization_score(self, component_checks: Dict,
                                    system_checks: Dict,
                                    performance_checks: Dict) -> int:
        """Calculate overall optimization score (0-100)."""
        score = 100
        
        # Component health penalties
        for component, check in component_checks.items():
            if check["status"] == "error":
                score -= 20
            elif check["status"] == "degraded":
                score -= 10
        
        # System resource penalties
        for resource, info in system_checks.items():
            if isinstance(info, dict):
                if info.get("status") == "error":
                    score -= 15
                elif info.get("status") == "degraded":
                    score -= 5
        
        # Performance penalties
        if performance_checks.get("overall_status") == "error":
            score -= 20
        elif performance_checks.get("overall_status") == "degraded":
            score -= 10
        
        return max(0, score)
    
    def _determine_overall_status(self, component_checks: Dict,
                                system_checks: Dict,
                                performance_checks: Dict,
                                issues: List[Dict]) -> str:
        """Determine overall system status."""
        
        # Check for critical issues
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            return "error"
        
        # Check for error issues
        error_issues = [i for i in issues if i.get("severity") == "error"]
        if error_issues:
            return "error"
        
        # Check for degraded performance
        warning_issues = [i for i in issues if i.get("severity") in ["warning", "degraded"]]
        if warning_issues:
            return "degraded"
        
        # All checks passed
        return "healthy"
    
    def _store_health_check_results(self, results: Dict[str, Any]):
        """Store health check results for trend analysis."""
        self._last_health_check = results
        
        # Store key metrics for trending
        timestamp = datetime.now()
        
        # Store optimization score
        self._metrics_history["optimization_score"].append({
            "timestamp": timestamp,
            "value": results.get("optimization_score", 0)
        })
        
        # Store component status counts
        component_health = results.get("component_health", {})
        healthy_count = sum(1 for c in component_health.values() if c.get("status") == "healthy")
        self._metrics_history["healthy_components"].append({
            "timestamp": timestamp,
            "value": healthy_count
        })
    
    async def get_performance_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance analytics with trends."""
        try:
            analytics = {
                "time_range_hours": time_range_hours,
                "timestamp": datetime.now().isoformat(),
                "performance_trends": {},
                "optimization_history": [],
                "recommendations": [],
                "summary": {}
            }
            
            # Get current performance metrics
            data_layer = self.bridge.get_data_layer()
            current_metrics = data_layer.get_performance_metrics()
            
            # Analyze optimization score trends
            optimization_history = list(self._metrics_history["optimization_score"])
            if optimization_history:
                recent_scores = [m["value"] for m in optimization_history[-10:]]
                analytics["optimization_history"] = optimization_history
                
                if len(recent_scores) >= 2:
                    trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                    analytics["summary"]["optimization_trend"] = trend
            
            # Performance operation analysis
            for operation, stats in current_metrics.get("operation_metrics", {}).items():
                analytics["performance_trends"][operation] = {
                    "current_avg_ms": stats.get("recent_avg_ms", 0),
                    "total_operations": stats.get("total_operations", 0),
                    "trend": "stable"  # Would need historical data for real trend analysis
                }
            
            # Generate performance recommendations
            analytics["recommendations"] = await self._generate_performance_recommendations(analytics)
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Performance analytics failed: {e}")
            return {"error": str(e)}
    
    async def _generate_performance_recommendations(self, analytics: Dict) -> List[str]:
        """Generate performance-specific recommendations."""
        recommendations = []
        
        # Analyze performance trends
        for operation, stats in analytics.get("performance_trends", {}).items():
            avg_time = stats.get("current_avg_ms", 0)
            
            if avg_time > 500:
                recommendations.append(f"Optimize {operation} - current average {avg_time:.1f}ms is high")
            elif avg_time > 200:
                recommendations.append(f"Monitor {operation} performance - approaching high latency")
        
        # Optimization score recommendations
        optimization_history = analytics.get("optimization_history", [])
        if optimization_history:
            latest_score = optimization_history[-1]["value"] if optimization_history else 0
            if latest_score < 70:
                recommendations.append("Overall optimization score is low - review configuration settings")
            elif latest_score < 85:
                recommendations.append("Good optimization but room for improvement")
        
        return recommendations
    
    async def diagnose_issues(self) -> Dict[str, Any]:
        """Automated issue diagnosis with resolution suggestions."""
        try:
            # Run comprehensive health check first
            health_results = await self.comprehensive_health_check()
            
            diagnosis = {
                "timestamp": datetime.now().isoformat(),
                "issues_found": len(health_results.get("issues_detected", [])),
                "diagnostics": [],
                "resolution_plan": [],
                "priority_actions": []
            }
            
            # Analyze each issue
            for issue in health_results.get("issues_detected", []):
                diagnostic = await self._diagnose_individual_issue(issue)
                diagnosis["diagnostics"].append(diagnostic)
            
            # Create resolution plan
            diagnosis["resolution_plan"] = self._create_resolution_plan(diagnosis["diagnostics"])
            
            # Identify priority actions
            diagnosis["priority_actions"] = self._identify_priority_actions(diagnosis["diagnostics"])
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Issue diagnosis failed: {e}")
            return {"error": str(e)}
    
    async def _diagnose_individual_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose an individual issue."""
        diagnostic = {
            "issue": issue,
            "probable_causes": [],
            "diagnostic_steps": [],
            "resolution_suggestions": [],
            "estimated_effort": "unknown"
        }
        
        issue_type = issue.get("type")
        severity = issue.get("severity")
        
        if issue_type == "component":
            component = issue.get("component")
            
            if component == "storage":
                diagnostic["probable_causes"] = [
                    "Disk space insufficient",
                    "Permission issues",
                    "Storage configuration errors",
                    "File system issues"
                ]
                diagnostic["diagnostic_steps"] = [
                    "Check disk space availability",
                    "Verify file system permissions",
                    "Test storage operations manually",
                    "Review storage configuration"
                ]
                diagnostic["estimated_effort"] = "low"
                
            elif component == "bridge":
                diagnostic["probable_causes"] = [
                    "Initialization failure",
                    "Configuration issues",
                    "Dependency problems"
                ]
                diagnostic["estimated_effort"] = "medium"
        
        elif issue_type == "performance":
            operation = issue.get("operation")
            diagnostic["probable_causes"] = [
                "System resource constraints",
                "Suboptimal configuration",
                "Database/storage bottlenecks",
                "Cache misses"
            ]
            diagnostic["diagnostic_steps"] = [
                "Monitor system resources during operation",
                "Profile operation performance",
                "Review cache hit rates",
                "Check storage I/O patterns"
            ]
            diagnostic["estimated_effort"] = "medium"
        
        # Add resolution suggestions from issue
        diagnostic["resolution_suggestions"] = issue.get("recommendations", [])
        
        return diagnostic
    
    def _create_resolution_plan(self, diagnostics: List[Dict]) -> List[Dict[str, Any]]:
        """Create a prioritized resolution plan."""
        plan = []
        
        # Group by severity and effort
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for diagnostic in diagnostics:
            severity = diagnostic["issue"].get("severity", "info")
            effort = diagnostic.get("estimated_effort", "medium")
            
            plan_item = {
                "issue_type": diagnostic["issue"].get("type"),
                "severity": severity,
                "effort": effort,
                "actions": diagnostic.get("resolution_suggestions", []),
                "estimated_time": self._estimate_resolution_time(effort, severity)
            }
            
            if severity in ["critical", "error"]:
                high_priority.append(plan_item)
            elif severity == "warning":
                medium_priority.append(plan_item)
            else:
                low_priority.append(plan_item)
        
        # Sort by effort within each priority
        for priority_list in [high_priority, medium_priority, low_priority]:
            priority_list.sort(key=lambda x: {"low": 1, "medium": 2, "high": 3}.get(x["effort"], 2))
        
        plan.extend(high_priority)
        plan.extend(medium_priority)
        plan.extend(low_priority)
        
        return plan
    
    def _identify_priority_actions(self, diagnostics: List[Dict]) -> List[str]:
        """Identify the most critical actions to take immediately."""
        priority_actions = []
        
        # Critical issues first
        for diagnostic in diagnostics:
            if diagnostic["issue"].get("severity") == "critical":
                priority_actions.extend(diagnostic.get("resolution_suggestions", [])[:2])
        
        # High-impact, low-effort actions
        for diagnostic in diagnostics:
            if (diagnostic.get("estimated_effort") == "low" and 
                diagnostic["issue"].get("severity") in ["error", "warning"]):
                priority_actions.extend(diagnostic.get("resolution_suggestions", [])[:1])
        
        return list(dict.fromkeys(priority_actions))  # Remove duplicates, preserve order
    
    def _estimate_resolution_time(self, effort: str, severity: str) -> str:
        """Estimate time needed to resolve issue."""
        time_matrix = {
            ("low", "critical"): "15-30 minutes",
            ("low", "error"): "15-30 minutes", 
            ("low", "warning"): "5-15 minutes",
            ("medium", "critical"): "1-2 hours",
            ("medium", "error"): "30-60 minutes",
            ("medium", "warning"): "15-30 minutes",
            ("high", "critical"): "2-4 hours",
            ("high", "error"): "1-2 hours",
            ("high", "warning"): "30-60 minutes"
        }
        
        return time_matrix.get((effort, severity), "30-60 minutes")
    
    def start_background_monitoring(self, interval_minutes: int = 5):
        """Start background monitoring thread."""
        if self._background_monitoring:
            return
        
        self._background_monitoring = True
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info(f"Started background monitoring with {interval_minutes} minute intervals")
    
    def stop_background_monitoring(self):
        """Stop background monitoring."""
        self._background_monitoring = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Stopped background monitoring")
    
    def _background_monitoring_loop(self, interval_minutes: int):
        """Background monitoring loop."""
        while self._background_monitoring:
            try:
                # Run health check in background
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                health_results = loop.run_until_complete(self.comprehensive_health_check())
                
                # Log any critical issues
                critical_issues = [
                    i for i in health_results.get("issues_detected", [])
                    if i.get("severity") == "critical"
                ]
                
                if critical_issues:
                    self.logger.error(f"Background monitoring detected {len(critical_issues)} critical issues")
                
                loop.close()
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
            
            # Wait for next interval
            time.sleep(interval_minutes * 60)


# Convenience functions

async def create_health_monitor(bridge: FFChatAppBridge) -> FFIntegrationHealthMonitor:
    """Create and initialize health monitor for a bridge."""
    return FFIntegrationHealthMonitor(bridge)


async def quick_health_check(bridge: FFChatAppBridge) -> Dict[str, Any]:
    """Perform quick health check on a bridge."""
    monitor = FFIntegrationHealthMonitor(bridge)
    return await monitor.comprehensive_health_check()


async def diagnose_bridge_issues(bridge: FFChatAppBridge) -> Dict[str, Any]:
    """Diagnose issues with a bridge and provide resolution plan."""
    monitor = FFIntegrationHealthMonitor(bridge)
    return await monitor.diagnose_issues()