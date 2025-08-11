"""
FF Chat Routes - Core chat functionality endpoints

Provides REST API endpoints for chat message processing, use case management,
and real-time chat features using the FF Chat system.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, status
from pydantic import BaseModel, Field

from ff_chat_auth import FFChatAuthManager, User, Permission
from ff_chat_application import FFChatApplication
from ff_class_configs.ff_chat_entities_config import MessageRole
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)

# Pydantic models
class ProcessMessageRequest(BaseModel):
    message: str = Field(..., description="Message content")
    role: str = Field(MessageRole.USER.value, description="Message role")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")
    attachments: Optional[List[str]] = Field(None, description="Optional file attachments")
    stream: bool = Field(False, description="Enable streaming response")

class MessageResponse(BaseModel):
    success: bool
    message_id: str
    response_content: Optional[str] = None
    component: Optional[str] = None
    processor: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    session_ids: Optional[List[str]] = Field(None, description="Session IDs to search")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    processing_time: float

class UseCaseTestRequest(BaseModel):
    test_message: str = Field("Hello, this is a test message", description="Test message")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")

class ComponentStatusResponse(BaseModel):
    component_name: str
    status: str
    capabilities: List[str]
    metadata: Dict[str, Any]

# Create router
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def process_message(
    session_id: str,
    request: ProcessMessageRequest,
    background_tasks: BackgroundTasks,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)  # Will be injected by middleware
):
    """
    Process a chat message in a session.
    
    This endpoint processes messages through the FF Chat system,
    applying all configured components and use case logic.
    """
    start_time = time.time()
    
    try:
        # Validate session access
        if user:
            session_info = await chat_app.get_session_info(session_id)
            if session_info["user_id"] != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to session"
                )
        
        # Process message
        result = await chat_app.process_message(
            session_id=session_id,
            message=request.message,
            role=request.role,
            attachments=request.attachments,
            context=request.context or {}
        )
        
        processing_time = time.time() - start_time
        
        # Background analytics (optional)
        if user:
            background_tasks.add_task(
                _record_message_analytics,
                user.user_id,
                session_id,
                request.message,
                result,
                processing_time
            )
        
        return MessageResponse(
            success=result["success"],
            message_id=result.get("message_id", ""),
            response_content=result.get("response_content"),
            component=result.get("component"),
            processor=result.get("processor"),
            metadata=result.get("metadata"),
            error=result.get("error"),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing message in session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Message offset"),
    include_metadata: bool = Query(False, description="Include message metadata"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Get messages from a chat session with pagination support."""
    try:
        # Validate session access
        if user:
            session_info = await chat_app.get_session_info(session_id)
            if session_info["user_id"] != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to session"
                )
        
        messages = await chat_app.get_session_messages(
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            message_data = {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            
            if include_metadata:
                message_data["metadata"] = getattr(msg, 'metadata', {})
                message_data["attachments"] = getattr(msg, 'attachments', [])
            
            formatted_messages.append(message_data)
        
        return {
            "messages": formatted_messages,
            "count": len(formatted_messages),
            "offset": offset,
            "limit": limit,
            "session_id": session_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting messages for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_messages(
    request: SearchRequest,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """
    Search messages across sessions using FF search capabilities.
    
    Supports full-text search with filtering by sessions and pagination.
    """
    start_time = time.time()
    
    try:
        # For non-system users, only search their own messages
        search_user_id = user.user_id if user else None
        
        results = await chat_app.search_messages(
            user_id=search_user_id,
            query=request.query,
            session_ids=request.session_ids,
            limit=request.limit,
            offset=request.offset
        )
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            results=results.get("results", []),
            total_count=results.get("total_count", 0),
            query=request.query,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/use-cases")
async def list_use_cases(
    chat_app: FFChatApplication = Depends()
):
    """List all available use cases with their configurations."""
    try:
        use_cases = await chat_app.list_use_cases()
        
        formatted_use_cases = []
        for name, info in use_cases.items():
            formatted_use_cases.append({
                "name": name,
                "description": info.get("description", f"{name} use case"),
                "components": info.get("components", []),
                "capabilities": info.get("capabilities", []),
                "example_config": info.get("example_config"),
                "complexity": info.get("complexity", "medium"),
                "coverage": info.get("coverage", "unknown")
            })
        
        return {
            "use_cases": formatted_use_cases,
            "total_count": len(formatted_use_cases)
        }
        
    except Exception as e:
        logger.error(f"Error listing use cases: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/use-cases/{use_case_name}")
async def get_use_case_details(
    use_case_name: str,
    chat_app: FFChatApplication = Depends()
):
    """Get detailed information about a specific use case."""
    try:
        use_case_info = await chat_app.get_use_case_info(use_case_name)
        
        return {
            "name": use_case_name,
            "description": use_case_info.get("description"),
            "components": use_case_info.get("components", []),
            "capabilities": use_case_info.get("capabilities", []),
            "configuration": use_case_info.get("configuration", {}),
            "requirements": use_case_info.get("requirements", []),
            "examples": use_case_info.get("examples", []),
            "performance_metrics": use_case_info.get("performance_metrics", {}),
            "supported_features": use_case_info.get("supported_features", [])
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting use case details: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/use-cases/{use_case_name}/test")
async def test_use_case(
    use_case_name: str,
    request: UseCaseTestRequest,
    background_tasks: BackgroundTasks,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """
    Test a use case with a sample message.
    
    Creates a temporary session to test the use case functionality.
    """
    start_time = time.time()
    
    try:
        # Create temporary session for testing
        test_user_id = user.user_id if user else "test_user"
        session_id = await chat_app.create_chat_session(
            user_id=test_user_id,
            use_case=use_case_name,
            title=f"Test: {use_case_name}",
            custom_config=request.config_overrides
        )
        
        try:
            # Process test message
            result = await chat_app.process_message(
                session_id=session_id,
                message=request.test_message,
                role=MessageRole.USER.value
            )
            
            processing_time = time.time() - start_time
            
            # Schedule cleanup
            background_tasks.add_task(_cleanup_test_session, chat_app, session_id)
            
            return {
                "use_case": use_case_name,
                "test_message": request.test_message,
                "result": result,
                "session_id": session_id,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as test_error:
            # Cleanup session on error
            try:
                await chat_app.close_session(session_id)
            except:
                pass
            raise test_error
            
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing use case {use_case_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/components", response_model=List[ComponentStatusResponse])
async def get_components_status(
    chat_app: FFChatApplication = Depends()
):
    """Get status and capabilities of all chat components."""
    try:
        components_info = await chat_app.get_components_info()
        
        components_status = []
        for name, info in components_info.items():
            components_status.append(ComponentStatusResponse(
                component_name=name,
                status="active" if info.get("initialized", False) else "inactive",
                capabilities=info.get("capabilities", []),
                metadata={
                    "version": info.get("version", "unknown"),
                    "dependencies": info.get("dependencies", []),
                    "configuration": info.get("configuration", {}),
                    "performance_metrics": info.get("performance_metrics", {})
                }
            ))
        
        return components_status
        
    except Exception as e:
        logger.error(f"Error getting components status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/components/{component_name}")
async def get_component_details(
    component_name: str,
    chat_app: FFChatApplication = Depends()
):
    """Get detailed information about a specific component."""
    try:
        components_info = await chat_app.get_components_info()
        
        if component_name not in components_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component '{component_name}' not found"
            )
        
        component_info = components_info[component_name]
        
        return {
            "name": component_name,
            "status": "active" if component_info.get("initialized", False) else "inactive",
            "capabilities": component_info.get("capabilities", []),
            "version": component_info.get("version", "unknown"),
            "dependencies": component_info.get("dependencies", []),
            "configuration": component_info.get("configuration", {}),
            "performance_metrics": component_info.get("performance_metrics", {}),
            "supported_use_cases": component_info.get("supported_use_cases", []),
            "resource_usage": component_info.get("resource_usage", {}),
            "last_updated": component_info.get("last_updated", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component details: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/metrics")
async def get_chat_metrics(
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Get chat system metrics and statistics."""
    try:
        # Check permission for detailed metrics
        include_detailed = False
        if user and hasattr(user, 'permissions'):
            include_detailed = Permission.VIEW_METRICS in user.permissions
        
        metrics = await chat_app.get_metrics()
        
        # Filter metrics based on permissions
        if not include_detailed:
            # Public metrics only
            metrics = {
                "system_status": metrics.get("system_status", "unknown"),
                "active_sessions": metrics.get("active_sessions", 0),
                "total_components": metrics.get("total_components", 0),
                "available_use_cases": metrics.get("available_use_cases", 0)
            }
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "detailed": include_detailed
        }
        
    except Exception as e:
        logger.error(f"Error getting chat metrics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Background task functions
async def _record_message_analytics(user_id: str, session_id: str, message: str, 
                                   result: Dict[str, Any], processing_time: float):
    """Record message analytics in background"""
    try:
        # In a real implementation, this would record analytics
        # to a database or analytics service
        logger.debug(f"Analytics: User {user_id} sent message in session {session_id}, "
                    f"processing time: {processing_time:.3f}s")
    except Exception as e:
        logger.error(f"Error recording analytics: {e}")

async def _cleanup_test_session(chat_app: FFChatApplication, session_id: str):
    """Cleanup test session in background"""
    try:
        await asyncio.sleep(300)  # Wait 5 minutes before cleanup
        await chat_app.close_session(session_id)
        logger.debug(f"Cleaned up test session: {session_id}")
    except Exception as e:
        logger.error(f"Error cleaning up test session {session_id}: {e}")

# Import asyncio at the top
import asyncio