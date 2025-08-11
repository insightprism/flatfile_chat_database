"""
FF Session Routes - Session management endpoints

Provides REST API endpoints for creating, managing, and monitoring
chat sessions using the FF Chat system.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field

from ff_chat_auth import FFChatAuthManager, User, Permission
from ff_chat_application import FFChatApplication
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)

# Pydantic models
class CreateSessionRequest(BaseModel):
    use_case: str = Field(..., description="Use case for the session")
    title: Optional[str] = Field(None, description="Optional session title")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")
    tags: Optional[List[str]] = Field(None, description="Optional session tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional session metadata")

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    tags: Optional[List[str]] = Field(None, description="New session tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New session metadata")
    active: Optional[bool] = Field(None, description="Session active status")

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    use_case: str
    title: Optional[str]
    active: bool
    created_at: str
    updated_at: str
    message_count: int
    last_activity: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]

class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str]
    use_case: str
    active: bool
    created_at: str
    message_count: int
    last_activity: Optional[str]

class SessionMetrics(BaseModel):
    session_id: str
    total_messages: int
    avg_response_time: float
    components_used: List[str]
    processing_stats: Dict[str, Any]
    usage_patterns: Dict[str, Any]

# Create router
router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])

@router.post("", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """
    Create a new chat session.
    
    Creates a new session configured for the specified use case
    with optional custom configuration and metadata.
    """
    try:
        user_id = user.user_id if user else "anonymous"
        
        # Create session
        session_id = await chat_app.create_chat_session(
            user_id=user_id,
            use_case=request.use_case,
            title=request.title,
            custom_config=request.config
        )
        
        # Add optional metadata and tags
        if request.metadata or request.tags:
            await chat_app.update_session_metadata(
                session_id=session_id,
                metadata=request.metadata or {},
                tags=request.tags or []
            )
        
        # Get session info for response
        session_info = await chat_app.get_session_info(session_id)
        
        return SessionResponse(
            session_id=session_id,
            user_id=user_id,
            use_case=request.use_case,
            title=request.title,
            active=session_info["active"],
            created_at=session_info["created_at"],
            updated_at=session_info.get("updated_at", session_info["created_at"]),
            message_count=session_info.get("message_count", 0),
            last_activity=session_info.get("last_activity"),
            tags=request.tags or [],
            metadata=request.metadata or {}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("", response_model=List[SessionSummary])
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID (admin only)"),
    use_case: Optional[str] = Query(None, description="Filter by use case"),
    active_only: bool = Query(True, description="Only show active sessions"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Session offset"),
    chat_app: FFChatApplication = Depends(),
    current_user: User = Depends(lambda: None)
):
    """
    List chat sessions with filtering and pagination.
    
    Users can see their own sessions. Admins can see all sessions.
    """
    try:
        # Determine which user's sessions to show
        target_user_id = current_user.user_id if current_user else "anonymous"
        
        # Admin users can view other users' sessions
        if (current_user and 
            hasattr(current_user, 'permissions') and 
            Permission.VIEW_USERS in current_user.permissions and 
            user_id):
            target_user_id = user_id
        
        # Get sessions
        sessions = await chat_app.get_user_sessions(
            user_id=target_user_id,
            use_case_filter=use_case,
            active_only=active_only,
            limit=limit,
            offset=offset
        )
        
        # Format response
        session_summaries = []
        for session in sessions:
            session_summaries.append(SessionSummary(
                session_id=session["session_id"],
                title=session.get("title"),
                use_case=session.get("use_case", "unknown"),
                active=session.get("active", False),
                created_at=session.get("created_at", ""),
                message_count=session.get("message_count", 0),
                last_activity=session.get("last_activity")
            ))
        
        return session_summaries
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    include_stats: bool = Query(False, description="Include detailed statistics"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Get detailed information about a specific session."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            # Check if user has admin permissions
            if not (hasattr(user, 'permissions') and Permission.VIEW_USERS in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to session"
                )
        
        # Get additional statistics if requested
        additional_stats = {}
        if include_stats:
            try:
                additional_stats = await chat_app.get_session_stats(session_id)
            except Exception as e:
                logger.warning(f"Could not get session stats: {e}")
        
        return SessionResponse(
            session_id=session_id,
            user_id=session_info["user_id"],
            use_case=session_info.get("use_case", "unknown"),
            title=session_info.get("title"),
            active=session_info["active"],
            created_at=session_info["created_at"],
            updated_at=session_info.get("updated_at", session_info["created_at"]),
            message_count=session_info.get("message_count", 0),
            last_activity=session_info.get("last_activity"),
            tags=session_info.get("tags", []),
            metadata={
                **session_info.get("metadata", {}),
                **additional_stats
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Update session information and metadata."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            if not (hasattr(user, 'permissions') and Permission.MANAGE_USERS in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to modify session"
                )
        
        # Prepare update data
        update_data = {}
        if request.title is not None:
            update_data["title"] = request.title
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.metadata is not None:
            update_data["metadata"] = request.metadata
        if request.active is not None:
            update_data["active"] = request.active
        
        # Update session
        await chat_app.update_session_metadata(session_id, update_data)
        
        # Get updated session info
        updated_info = await chat_app.get_session_info(session_id)
        
        return SessionResponse(
            session_id=session_id,
            user_id=updated_info["user_id"],
            use_case=updated_info.get("use_case", "unknown"),
            title=updated_info.get("title"),
            active=updated_info["active"],
            created_at=updated_info["created_at"],
            updated_at=updated_info.get("updated_at", datetime.now().isoformat()),
            message_count=updated_info.get("message_count", 0),
            last_activity=updated_info.get("last_activity"),
            tags=updated_info.get("tags", []),
            metadata=updated_info.get("metadata", {})
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.delete("/{session_id}")
async def close_session(
    session_id: str,
    permanently: bool = Query(False, description="Permanently delete session data"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Close or delete a chat session."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            if not (hasattr(user, 'permissions') and Permission.MANAGE_USERS in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to close session"
                )
        
        if permanently:
            # Permanent deletion requires admin permissions
            if not (user and hasattr(user, 'permissions') and Permission.MANAGE_SYSTEM in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin permissions required for permanent deletion"
                )
            
            await chat_app.delete_session(session_id)
            return {"message": "Session permanently deleted", "session_id": session_id}
        else:
            # Just close the session
            await chat_app.close_session(session_id)
            return {"message": "Session closed", "session_id": session_id}
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/{session_id}/metrics", response_model=SessionMetrics)
async def get_session_metrics(
    session_id: str,
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Get detailed metrics and analytics for a session."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            if not (hasattr(user, 'permissions') and Permission.VIEW_METRICS in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to session metrics"
                )
        
        # Get session metrics
        metrics = await chat_app.get_session_metrics(session_id)
        
        return SessionMetrics(
            session_id=session_id,
            total_messages=metrics.get("total_messages", 0),
            avg_response_time=metrics.get("avg_response_time", 0.0),
            components_used=metrics.get("components_used", []),
            processing_stats=metrics.get("processing_stats", {}),
            usage_patterns=metrics.get("usage_patterns", {})
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session metrics {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = Query("json", description="Export format (json, csv, markdown)"),
    include_metadata: bool = Query(True, description="Include message metadata"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Export session data in various formats."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            if not (hasattr(user, 'permissions') and Permission.VIEW_MESSAGES in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to export session"
                )
        
        # Get session messages
        messages = await chat_app.get_session_messages(session_id)
        
        # Export in requested format
        if format.lower() == "json":
            export_data = {
                "session_info": session_info,
                "messages": [
                    {
                        "message_id": msg.message_id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        **({"metadata": getattr(msg, 'metadata', {})} if include_metadata else {})
                    }
                    for msg in messages
                ],
                "exported_at": datetime.now().isoformat()
            }
            return export_data
            
        elif format.lower() == "csv":
            # CSV format would be implemented here
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="CSV export not yet implemented"
            )
            
        elif format.lower() == "markdown":
            # Markdown format would be implemented here
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Markdown export not yet implemented"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format"
            )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/{session_id}/clone")
async def clone_session(
    session_id: str,
    new_title: Optional[str] = Query(None, description="Title for cloned session"),
    include_messages: bool = Query(False, description="Include existing messages"),
    chat_app: FFChatApplication = Depends(),
    user: User = Depends(lambda: None)
):
    """Clone an existing session with its configuration."""
    try:
        session_info = await chat_app.get_session_info(session_id)
        
        # Check access permissions
        if user and session_info["user_id"] != user.user_id:
            if not (hasattr(user, 'permissions') and Permission.CREATE_SESSION in user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to clone session"
                )
        
        user_id = user.user_id if user else session_info["user_id"]
        
        # Create new session with same configuration
        new_session_id = await chat_app.create_chat_session(
            user_id=user_id,
            use_case=session_info.get("use_case", "basic_chat"),
            title=new_title or f"Copy of {session_info.get('title', session_id)}",
            custom_config=session_info.get("config", {})
        )
        
        # Copy metadata
        if session_info.get("metadata") or session_info.get("tags"):
            await chat_app.update_session_metadata(
                session_id=new_session_id,
                metadata=session_info.get("metadata", {}),
                tags=session_info.get("tags", [])
            )
        
        # Copy messages if requested
        if include_messages:
            messages = await chat_app.get_session_messages(session_id)
            for msg in messages:
                await chat_app.process_message(
                    session_id=new_session_id,
                    message=msg.content,
                    role=msg.role,
                    context={"cloned_from": session_id}
                )
        
        return {
            "original_session_id": session_id,
            "new_session_id": new_session_id,
            "messages_copied": len(messages) if include_messages else 0,
            "created_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))