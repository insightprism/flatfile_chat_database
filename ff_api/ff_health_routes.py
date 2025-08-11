"""
FF Health Routes - Health check and status endpoints

Provides REST API endpoints for system health monitoring,
readiness checks, and service status information.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status, Response
from pydantic import BaseModel, Field

from ff_chat_application import FFChatApplication
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)

# Pydantic models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    
class DetailedHealthResponse(HealthResponse):
    ff_backend_status: Dict[str, str] = Field(..., description="FF backend service status")
    component_status: Dict[str, str] = Field(..., description="Chat component status")
    dependencies: Dict[str, Dict[str, Any]] = Field(..., description="External dependencies status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")

class ReadinessResponse(BaseModel):
    ready: bool = Field(..., description="System readiness status")
    timestamp: str = Field(..., description="Check timestamp")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")
    startup_time_seconds: Optional[float] = Field(None, description="Time since startup")

class LivenessResponse(BaseModel):
    alive: bool = Field(..., description="System liveness status")
    timestamp: str = Field(..., description="Check timestamp")
    last_activity: Optional[str] = Field(None, description="Last activity timestamp")

# Application start time for uptime calculation
START_TIME = time.time()

# Create router
router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def basic_health_check():
    """
    Basic health check endpoint.
    
    Returns minimal health information suitable for load balancer checks.
    """
    try:
        uptime = time.time() - START_TIME
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - START_TIME
        )

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    chat_app: FFChatApplication = Depends(lambda: None)
):
    """
    Detailed health check with comprehensive system information.
    
    Includes FF backend status, component health, and system metrics.
    """
    try:
        uptime = time.time() - START_TIME
        overall_status = "healthy"
        
        # Check FF backend status
        ff_backend_status = {}
        if chat_app:
            try:
                if hasattr(chat_app, 'ff_storage') and chat_app.ff_storage:
                    ff_backend_status["storage"] = "healthy"
                else:
                    ff_backend_status["storage"] = "unavailable"
                    overall_status = "degraded"
                
                if hasattr(chat_app, 'ff_search') and chat_app.ff_search:
                    ff_backend_status["search"] = "healthy"
                else:
                    ff_backend_status["search"] = "unavailable"
                
                if hasattr(chat_app, 'ff_vector') and chat_app.ff_vector:
                    ff_backend_status["vector"] = "healthy"
                else:
                    ff_backend_status["vector"] = "unavailable"
                    
            except Exception as e:
                logger.error(f"Error checking FF backend: {e}")
                ff_backend_status["error"] = str(e)
                overall_status = "unhealthy"
        else:
            ff_backend_status["chat_app"] = "unavailable"
            overall_status = "degraded"
        
        # Check component status
        component_status = {}
        if chat_app:
            try:
                components_info = await chat_app.get_components_info()
                for name, info in components_info.items():
                    component_status[name] = "active" if info.get("initialized", False) else "inactive"
            except Exception as e:
                logger.error(f"Error checking components: {e}")
                component_status["error"] = str(e)
                overall_status = "unhealthy"
        
        # Check external dependencies
        dependencies = await _check_external_dependencies()
        
        # Get system metrics
        metrics = await _get_system_metrics(chat_app)
        
        return DetailedHealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=uptime,
            ff_backend_status=ff_backend_status,
            component_status=component_status,
            dependencies=dependencies,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/readiness", response_model=ReadinessResponse)
async def readiness_check(
    chat_app: FFChatApplication = Depends(lambda: None),
    response: Response = None
):
    """
    Kubernetes-style readiness probe.
    
    Checks if the application is ready to serve requests.
    """
    checks = {}
    overall_ready = True
    
    try:
        # Check if chat application is initialized
        if chat_app and hasattr(chat_app, '_initialized'):
            checks["chat_app_initialized"] = chat_app._initialized
        else:
            checks["chat_app_initialized"] = False
        
        # Check FF storage availability
        if chat_app and hasattr(chat_app, 'ff_storage') and chat_app.ff_storage:
            checks["ff_storage_available"] = True
        else:
            checks["ff_storage_available"] = False
        
        # Check component readiness
        if chat_app:
            try:
                components_info = await chat_app.get_components_info()
                active_components = sum(1 for info in components_info.values() 
                                      if info.get("initialized", False))
                total_components = len(components_info)
                
                checks["components_ready"] = active_components > 0
                checks["all_components_ready"] = active_components == total_components
            except:
                checks["components_ready"] = False
                checks["all_components_ready"] = False
        else:
            checks["components_ready"] = False
            checks["all_components_ready"] = False
        
        # Check critical dependencies
        try:
            import os
            checks["filesystem_writable"] = os.access(".", os.W_OK)
        except:
            checks["filesystem_writable"] = False
        
        # Determine overall readiness
        critical_checks = ["chat_app_initialized", "ff_storage_available", "filesystem_writable"]
        overall_ready = all(checks.get(check, False) for check in critical_checks)
        
        # Set HTTP status based on readiness
        if response and not overall_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return ReadinessResponse(
            ready=overall_ready,
            timestamp=datetime.now().isoformat(),
            checks=checks,
            startup_time_seconds=time.time() - START_TIME
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        if response:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return ReadinessResponse(
            ready=False,
            timestamp=datetime.now().isoformat(),
            checks={"error": str(e)},
            startup_time_seconds=time.time() - START_TIME
        )

@router.get("/liveness", response_model=LivenessResponse)
async def liveness_check(
    chat_app: FFChatApplication = Depends(lambda: None),
    response: Response = None
):
    """
    Kubernetes-style liveness probe.
    
    Checks if the application is alive and should be restarted if not.
    """
    try:
        # Basic liveness indicators
        alive = True
        last_activity = None
        
        # Check if the application is responsive
        try:
            # Simple responsiveness test
            test_start = time.time()
            await asyncio.sleep(0.001)  # Minimal async operation
            response_time = time.time() - test_start
            
            # If response time is too high, consider unhealthy
            if response_time > 1.0:  # 1 second threshold
                alive = False
        except:
            alive = False
        
        # Check for recent activity if chat app is available
        if chat_app and hasattr(chat_app, 'get_metrics'):
            try:
                metrics = await chat_app.get_metrics()
                last_activity = metrics.get("last_activity")
            except:
                pass
        
        # Set HTTP status based on liveness
        if response and not alive:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return LivenessResponse(
            alive=alive,
            timestamp=datetime.now().isoformat(),
            last_activity=last_activity
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        if response:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return LivenessResponse(
            alive=False,
            timestamp=datetime.now().isoformat(),
            last_activity=None
        )

@router.get("/status")
async def service_status(
    format: str = Query("json", description="Response format (json, prometheus)"),
    chat_app: FFChatApplication = Depends(lambda: None)
):
    """
    Service status endpoint with multiple format support.
    
    Supports JSON and Prometheus metrics formats.
    """
    try:
        if format.lower() == "prometheus":
            return await _get_prometheus_metrics(chat_app)
        else:
            return await _get_json_status(chat_app)
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )

@router.get("/version")
async def get_version():
    """Get application version information."""
    return {
        "version": "1.0.0",
        "build_date": "2024-01-01",
        "git_commit": "unknown",
        "python_version": "3.8+",
        "ff_chat_version": "4.0.0"
    }

# Helper functions
async def _check_external_dependencies() -> Dict[str, Dict[str, Any]]:
    """Check external service dependencies"""
    dependencies = {}
    
    # Check database connectivity (if applicable)
    # Check external APIs (if applicable)
    # Check file system access
    try:
        import os
        dependencies["filesystem"] = {
            "status": "healthy" if os.access(".", os.R_OK | os.W_OK) else "unhealthy",
            "readable": os.access(".", os.R_OK),
            "writable": os.access(".", os.W_OK)
        }
    except Exception as e:
        dependencies["filesystem"] = {
            "status": "error",
            "error": str(e)
        }
    
    return dependencies

async def _get_system_metrics(chat_app: Optional[FFChatApplication]) -> Dict[str, Any]:
    """Get basic system metrics"""
    metrics = {
        "uptime_seconds": time.time() - START_TIME,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add system resource metrics if available
    try:
        import psutil
        metrics.update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('.').percent
        })
    except ImportError:
        metrics["system_resources"] = "psutil not available"
    except Exception as e:
        metrics["system_resources"] = f"error: {str(e)}"
    
    # Add chat application metrics if available
    if chat_app:
        try:
            app_metrics = await chat_app.get_metrics()
            metrics["chat_metrics"] = {
                "active_sessions": app_metrics.get("active_sessions", 0),
                "total_components": app_metrics.get("total_components", 0)
            }
        except:
            metrics["chat_metrics"] = "unavailable"
    
    return metrics

async def _get_json_status(chat_app: Optional[FFChatApplication]) -> Dict[str, Any]:
    """Get status in JSON format"""
    return {
        "service": "ff_chat_api",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - START_TIME,
        "version": "1.0.0",
        "metrics": await _get_system_metrics(chat_app)
    }

async def _get_prometheus_metrics(chat_app: Optional[FFChatApplication]) -> str:
    """Get metrics in Prometheus format"""
    metrics = await _get_system_metrics(chat_app)
    
    prometheus_output = [
        "# HELP ff_chat_uptime_seconds Total uptime in seconds",
        "# TYPE ff_chat_uptime_seconds counter",
        f"ff_chat_uptime_seconds {metrics.get('uptime_seconds', 0)}",
        "",
        "# HELP ff_chat_active_sessions Number of active chat sessions",
        "# TYPE ff_chat_active_sessions gauge",
        f"ff_chat_active_sessions {metrics.get('chat_metrics', {}).get('active_sessions', 0)}",
        ""
    ]
    
    if "cpu_percent" in metrics:
        prometheus_output.extend([
            "# HELP ff_chat_cpu_usage_percent CPU usage percentage",
            "# TYPE ff_chat_cpu_usage_percent gauge",
            f"ff_chat_cpu_usage_percent {metrics['cpu_percent']}",
            ""
        ])
    
    if "memory_percent" in metrics:
        prometheus_output.extend([
            "# HELP ff_chat_memory_usage_percent Memory usage percentage",
            "# TYPE ff_chat_memory_usage_percent gauge",
            f"ff_chat_memory_usage_percent {metrics['memory_percent']}",
            ""
        ])
    
    return "\n".join(prometheus_output)