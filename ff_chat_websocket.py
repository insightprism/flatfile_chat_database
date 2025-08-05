"""
FF Chat WebSocket - Advanced WebSocket Support for Real-time Chat

Provides specialized WebSocket handling for real-time chat experiences,
including connection management, message queuing, and event streaming.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

logger = get_logger(__name__)

class WebSocketEventType(Enum):
    """WebSocket event types"""
    MESSAGE = "message"
    MESSAGE_RESPONSE = "message_response"
    STATUS = "status"
    ERROR = "error"
    TYPING = "typing"
    SYSTEM = "system"
    SESSION_UPDATE = "session_update"
    HEARTBEAT = "heartbeat"

class ConnectionStatus(Enum):
    """WebSocket connection status"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: WebSocketEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    connection_id: str
    websocket: WebSocket
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    message_queue: List[WebSocketMessage] = field(default_factory=list)
    subscriptions: Set[str] = field(default_factory=set)

@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    max_connections: int = 1000
    heartbeat_interval: int = 30
    message_queue_size: int = 100
    connection_timeout: int = 300
    enable_message_history: bool = True
    enable_typing_indicators: bool = True
    enable_presence: bool = True
    broadcast_enabled: bool = False

class FFChatWebSocketManager:
    """
    Advanced WebSocket manager for FF Chat system.
    
    Handles real-time chat connections, message broadcasting,
    typing indicators, and connection lifecycle management.
    """
    
    def __init__(self, config: WebSocketConfig = None, chat_app=None):
        """
        Initialize WebSocket manager.
        
        Args:
            config: WebSocket configuration
            chat_app: FF Chat Application instance
        """
        self.config = config or WebSocketConfig()
        self.chat_app = chat_app
        self.logger = get_logger(__name__)
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        self.user_connections: Dict[str, Set[str]] = {}     # user_id -> connection_ids
        
        # Event handlers
        self.event_handlers: Dict[WebSocketEventType, List[Callable]] = {
            event_type: [] for event_type in WebSocketEventType
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "current_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0
        }
        
        self._running = False
    
    async def start(self):
        """Start WebSocket manager background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        if self.config.heartbeat_interval > 0:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("FF Chat WebSocket Manager started")
    
    async def stop(self):
        """Stop WebSocket manager and close all connections"""
        self._running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id, reason="Server shutdown")
        
        self.logger.info("FF Chat WebSocket Manager stopped")
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            session_id: Chat session ID
            user_id: Optional user ID
            
        Returns:
            Connection ID
        """
        # Check connection limits
        if len(self.connections) >= self.config.max_connections:
            await websocket.close(code=1008, reason="Connection limit exceeded")
            raise Exception("Connection limit exceeded")
        
        # Accept connection
        await websocket.accept()
        
        # Create connection
        connection_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            status=ConnectionStatus.CONNECTED
        )
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Track by session
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(connection_id)
        
        # Track by user
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        # Update statistics
        self.stats["total_connections"] += 1
        self.stats["current_connections"] = len(self.connections)
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type=WebSocketEventType.SYSTEM,
            data={
                "message": "Connected to FF Chat WebSocket",
                "connection_id": connection_id,
                "session_id": session_id
            },
            session_id=session_id,
            user_id=user_id
        )
        await self.send_to_connection(connection_id, welcome_msg)
        
        # Trigger event handlers
        await self._trigger_event_handlers(WebSocketEventType.SYSTEM, {
            "event": "connection_established",
            "connection_id": connection_id,
            "session_id": session_id,
            "user_id": user_id
        })
        
        self.logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str, reason: str = "Client disconnect"):
        """
        Handle WebSocket disconnection.
        
        Args:
            connection_id: Connection ID to disconnect
            reason: Disconnection reason
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.status = ConnectionStatus.DISCONNECTING
        
        try:
            # Close WebSocket
            await connection.websocket.close()
        except Exception as e:
            self.logger.error(f"Error closing WebSocket {connection_id}: {e}")
        
        # Remove from tracking
        del self.connections[connection_id]
        
        # Remove from session tracking
        if connection.session_id in self.session_connections:
            self.session_connections[connection.session_id].discard(connection_id)
            if not self.session_connections[connection.session_id]:
                del self.session_connections[connection.session_id]
        
        # Remove from user tracking
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Update statistics
        self.stats["current_connections"] = len(self.connections)
        
        # Trigger event handlers
        await self._trigger_event_handlers(WebSocketEventType.SYSTEM, {
            "event": "connection_closed",
            "connection_id": connection_id,
            "session_id": connection.session_id,
            "user_id": connection.user_id,
            "reason": reason
        })
        
        self.logger.info(f"WebSocket connection closed: {connection_id} ({reason})")
    
    async def handle_connection(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """
        Handle WebSocket connection lifecycle.
        
        Args:
            websocket: WebSocket instance
            session_id: Chat session ID
            user_id: Optional user ID
        """
        connection_id = None
        
        try:
            # Establish connection
            connection_id = await self.connect(websocket, session_id, user_id)
            connection = self.connections[connection_id]
            
            # Message loop
            while connection.status == ConnectionStatus.CONNECTED:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Update activity
                    connection.last_activity = datetime.now()
                    self.stats["messages_received"] += 1
                    
                    # Process message
                    await self._process_incoming_message(connection_id, message_data)
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError as e:
                    # Send error message
                    error_msg = WebSocketMessage(
                        type=WebSocketEventType.ERROR,
                        data={"error": f"Invalid JSON: {str(e)}"},
                        session_id=session_id
                    )
                    await self.send_to_connection(connection_id, error_msg)
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
                    self.stats["connection_errors"] += 1
                    
                    error_msg = WebSocketMessage(
                        type=WebSocketEventType.ERROR,
                        data={"error": str(e)},
                        session_id=session_id
                    )
                    await self.send_to_connection(connection_id, error_msg)
        
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            self.stats["connection_errors"] += 1
        
        finally:
            if connection_id:
                await self.disconnect(connection_id, "Connection ended")
    
    async def _process_incoming_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        message_type = WebSocketEventType(message_data.get("type", "message"))
        
        if message_type == WebSocketEventType.MESSAGE:
            # Process chat message
            await self._handle_chat_message(connection, message_data)
        
        elif message_type == WebSocketEventType.TYPING:
            # Handle typing indicator
            await self._handle_typing_indicator(connection, message_data)
        
        elif message_type == WebSocketEventType.HEARTBEAT:
            # Handle heartbeat
            await self._handle_heartbeat(connection)
        
        else:
            # Trigger custom event handlers
            await self._trigger_event_handlers(message_type, {
                "connection_id": connection_id,
                "data": message_data,
                "connection": connection
            })
    
    async def _handle_chat_message(self, connection: WebSocketConnection, message_data: Dict[str, Any]):
        """Handle chat message processing"""
        try:
            if not self.chat_app:
                raise Exception("Chat application not available")
            
            # Extract message content
            message_content = message_data.get("message", "")
            role = message_data.get("role", MessageRole.USER.value)
            context = message_data.get("context", {})
            attachments = message_data.get("attachments")
            
            start_time = time.time()
            
            # Process through chat application
            result = await self.chat_app.process_message(
                session_id=connection.session_id,
                message=message_content,
                role=role,
                attachments=attachments,
                **context
            )
            
            processing_time = time.time() - start_time
            
            # Send response
            response_msg = WebSocketMessage(
                type=WebSocketEventType.MESSAGE_RESPONSE,
                data={
                    **result,
                    "processing_time": processing_time
                },
                session_id=connection.session_id,
                user_id=connection.user_id
            )
            
            await self.send_to_connection(connection.connection_id, response_msg)
            
            # Broadcast to other connections in session if enabled
            if self.config.broadcast_enabled:
                await self.broadcast_to_session(
                    connection.session_id,
                    response_msg,
                    exclude_connection=connection.connection_id
                )
            
        except Exception as e:
            error_msg = WebSocketMessage(
                type=WebSocketEventType.ERROR,
                data={"error": str(e)},
                session_id=connection.session_id,
                user_id=connection.user_id
            )
            await self.send_to_connection(connection.connection_id, error_msg)
    
    async def _handle_typing_indicator(self, connection: WebSocketConnection, message_data: Dict[str, Any]):
        """Handle typing indicator"""
        if not self.config.enable_typing_indicators:
            return
        
        typing_msg = WebSocketMessage(
            type=WebSocketEventType.TYPING,
            data={
                "user_id": connection.user_id,
                "is_typing": message_data.get("is_typing", False)
            },
            session_id=connection.session_id
        )
        
        # Broadcast to other connections in session
        await self.broadcast_to_session(
            connection.session_id,
            typing_msg,
            exclude_connection=connection.connection_id
        )
    
    async def _handle_heartbeat(self, connection: WebSocketConnection):
        """Handle heartbeat message"""
        heartbeat_msg = WebSocketMessage(
            type=WebSocketEventType.HEARTBEAT,
            data={"status": "alive"},
            session_id=connection.session_id
        )
        await self.send_to_connection(connection.connection_id, heartbeat_msg)
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection"""
        connection = self.connections.get(connection_id)
        if not connection or connection.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            message_json = json.dumps({
                "type": message.type.value,
                "data": message.data,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "session_id": message.session_id,
                "user_id": message.user_id
            })
            
            await connection.websocket.send_text(message_json)
            self.stats["messages_sent"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id, f"Send error: {str(e)}")
            return False
    
    async def broadcast_to_session(self, session_id: str, message: WebSocketMessage, exclude_connection: Optional[str] = None):
        """Broadcast message to all connections in a session"""
        if session_id not in self.session_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.session_connections[session_id].copy():
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage, exclude_connection: Optional[str] = None):
        """Broadcast message to all connections for a user"""
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.user_connections[user_id].copy():
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    def add_event_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Add event handler for WebSocket events"""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Remove event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _trigger_event_handlers(self, event_type: WebSocketEventType, event_data: Dict[str, Any]):
        """Trigger event handlers for an event type"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self._running:
            try:
                current_time = datetime.now()
                
                for connection_id, connection in list(self.connections.items()):
                    # Check for inactive connections
                    if (current_time - connection.last_activity).seconds > self.config.connection_timeout:
                        await self.disconnect(connection_id, "Connection timeout")
                        continue
                    
                    # Send heartbeat if enabled
                    heartbeat_msg = WebSocketMessage(
                        type=WebSocketEventType.HEARTBEAT,
                        data={"timestamp": current_time.isoformat()},
                        session_id=connection.session_id
                    )
                    await self.send_to_connection(connection_id, heartbeat_msg)
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                # Clean up disconnected connections
                disconnected = []
                for connection_id, connection in self.connections.items():
                    if connection.status in [ConnectionStatus.DISCONNECTED, ConnectionStatus.ERROR]:
                        disconnected.append(connection_id)
                
                for connection_id in disconnected:
                    await self.disconnect(connection_id, "Cleanup")
                
                # Clean up empty session/user tracking
                empty_sessions = [sid for sid, cids in self.session_connections.items() if not cids]
                for sid in empty_sessions:
                    del self.session_connections[sid]
                
                empty_users = [uid for uid, cids in self.user_connections.items() if not cids]
                for uid in empty_users:
                    del self.user_connections[uid]
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information"""
        connection = self.connections.get(connection_id)
        if not connection:
            return None
        
        return {
            "connection_id": connection.connection_id,
            "session_id": connection.session_id,
            "user_id": connection.user_id,
            "created_at": connection.created_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "status": connection.status.value,
            "message_queue_size": len(connection.message_queue),
            "subscriptions": list(connection.subscriptions)
        }
    
    def get_session_connections(self, session_id: str) -> List[str]:
        """Get all connection IDs for a session"""
        return list(self.session_connections.get(session_id, set()))
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return list(self.user_connections.get(user_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.session_connections),
            "active_users": len(self.user_connections),
            "running": self._running
        }