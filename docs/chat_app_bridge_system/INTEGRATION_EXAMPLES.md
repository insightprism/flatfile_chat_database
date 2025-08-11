# Chat Application Bridge System - Integration Examples

## Overview

This document provides practical integration examples for different types of chat applications using the Chat Application Bridge System. Each example includes complete working code, configuration options, and best practices.

## Quick Start Examples

### Basic Chat Application

The simplest possible integration for a basic chat application:

```python
import asyncio
from ff_chat_integration import FFChatAppBridge

async def basic_chat_example():
    """Simple chat application example."""
    
    # Create bridge with one line
    bridge = await FFChatAppBridge.create_for_chat_app("./chat_data")
    
    # Get data layer
    data_layer = bridge.get_data_layer()
    
    # Create user and session
    user_id = "user_123"
    await data_layer.storage.create_user(user_id, {"name": "John Doe"})
    session_id = await data_layer.storage.create_session(user_id, "My Chat Session")
    
    # Store messages
    await data_layer.store_chat_message(
        user_id, session_id,
        {"role": "user", "content": "Hello, how are you?"}
    )
    
    await data_layer.store_chat_message(
        user_id, session_id,
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help?"}
    )
    
    # Retrieve conversation history
    history = await data_layer.get_chat_history(user_id, session_id)
    
    if history["success"]:
        for message in history["data"]["messages"]:
            print(f"{message['role']}: {message['content']}")
    
    # Clean up
    await bridge.close()

# Run the example
asyncio.run(basic_chat_example())
```

### Production-Ready Setup

Production setup with proper configuration and error handling:

```python
import asyncio
import logging
from pathlib import Path
from ff_chat_integration import (
    FFChatAppBridge, FFIntegrationHealthMonitor,
    ConfigurationError, InitializationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatApp:
    """Production-ready chat application integration."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.bridge = None
        self.data_layer = None
        self.health_monitor = None
        
    async def initialize(self):
        """Initialize with production settings."""
        try:
            # Create production-optimized bridge
            self.bridge = await FFChatAppBridge.create_from_preset(
                "production",
                self.storage_path,
                {
                    "cache_size_mb": 200,
                    "enable_compression": True,
                    "backup_enabled": True,
                    "max_session_size_mb": 100
                }
            )
            
            # Get data layer
            self.data_layer = self.bridge.get_data_layer()
            
            # Start health monitoring
            self.health_monitor = FFIntegrationHealthMonitor(self.bridge)
            await self.health_monitor.start_monitoring(interval_seconds=30)
            
            logger.info("Production chat app initialized successfully")
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            logger.error(f"Suggestions: {e.suggestions}")
            raise
        except InitializationError as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    async def create_user_session(self, user_id: str, user_name: str, session_name: str):
        """Create user and session with error handling."""
        try:
            # Create user if not exists
            await self.data_layer.storage.create_user(
                user_id, 
                {
                    "name": user_name,
                    "created_at": "2024-01-15T10:00:00Z",
                    "preferences": {"notification": True}
                }
            )
            
            # Create session
            session_id = await self.data_layer.storage.create_session(
                user_id, session_name
            )
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create user session: {e}")
            raise
    
    async def send_message(self, user_id: str, session_id: str, role: str, content: str):
        """Send message with performance monitoring."""
        try:
            result = await self.data_layer.store_chat_message(
                user_id, session_id,
                {
                    "role": role,
                    "content": content,
                    "timestamp": "2024-01-15T10:00:00Z",
                    "metadata": {"source": "production_app"}
                }
            )
            
            if result["success"]:
                # Log performance metrics
                perf = result["metadata"]["performance_metrics"]
                logger.info(f"Message stored in {perf['storage_time_ms']:.2f}ms")
                return result["data"]["message_id"]
            else:
                logger.error(f"Failed to store message: {result['error']}")
                return None
                
        except Exception as e:
            logger.error(f"Send message error: {e}")
            raise
    
    async def get_conversation(self, user_id: str, session_id: str, limit: int = 50):
        """Get conversation with caching."""
        try:
            result = await self.data_layer.get_chat_history(
                user_id, session_id, limit=limit
            )
            
            if result["success"]:
                # Check cache performance
                if result["metadata"]["performance_metrics"]["cache_hit"]:
                    logger.info("Conversation retrieved from cache")
                
                return result["data"]["messages"]
            else:
                logger.error(f"Failed to get conversation: {result['error']}")
                return []
                
        except Exception as e:
            logger.error(f"Get conversation error: {e}")
            return []
    
    async def search_conversations(self, user_id: str, query: str):
        """Search across conversations."""
        try:
            result = await self.data_layer.search_conversations(
                user_id, query,
                {
                    "search_type": "text",
                    "limit": 20,
                    "include_metadata": True
                }
            )
            
            if result["success"]:
                return result["data"]["results"]
            else:
                logger.error(f"Search failed: {result['error']}")
                return []
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def get_health_status(self):
        """Get comprehensive health status."""
        try:
            if self.health_monitor:
                health = await self.health_monitor.comprehensive_health_check()
                return health
            else:
                return {"status": "monitoring_not_started"}
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self):
        """Clean shutdown."""
        try:
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            if self.bridge:
                await self.bridge.close()
                
            logger.info("Production chat app shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Usage example
async def main():
    app = ProductionChatApp("/var/lib/chatapp/data")
    
    try:
        await app.initialize()
        
        # Create user and session
        session_id = await app.create_user_session(
            "prod_user_123", 
            "Production User", 
            "Production Chat Session"
        )
        
        # Send messages
        await app.send_message(
            "prod_user_123", session_id, 
            "user", "Hello, production system!"
        )
        
        await app.send_message(
            "prod_user_123", session_id,
            "assistant", "Hello! I'm ready for production workload."
        )
        
        # Get conversation
        messages = await app.get_conversation("prod_user_123", session_id)
        print(f"Retrieved {len(messages)} messages")
        
        # Check health
        health = await app.get_health_status()
        print(f"System health: {health['overall_status']}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Use Case Specific Examples

### AI Assistant Integration

Example for AI-powered chat applications with vector search:

```python
import asyncio
import json
from ff_chat_integration import FFChatAppBridge

class AIAssistantChat:
    """AI Assistant chat integration example."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.bridge = None
        
    async def initialize(self):
        """Initialize AI assistant optimized bridge."""
        self.bridge = await FFChatAppBridge.create_for_use_case(
            "ai_assistant",
            self.storage_path,
            # AI-specific optimizations
            enable_vector_search=True,
            enable_analytics=True,
            cache_size_mb=300,
            performance_mode="quality"  # Prioritize quality over speed
        )
        
        self.data_layer = self.bridge.get_data_layer()
        
    async def store_ai_conversation(self, user_id: str, session_id: str, 
                                  user_message: str, ai_response: str,
                                  ai_metadata: dict = None):
        """Store AI conversation with metadata."""
        
        # Store user message
        user_result = await self.data_layer.store_chat_message(
            user_id, session_id,
            {
                "role": "user",
                "content": user_message,
                "metadata": {
                    "timestamp": "2024-01-15T10:00:00Z",
                    "client": "ai_assistant_app"
                }
            }
        )
        
        # Store AI response with metadata
        ai_result = await self.data_layer.store_chat_message(
            user_id, session_id,
            {
                "role": "assistant",
                "content": ai_response,
                "metadata": {
                    "timestamp": "2024-01-15T10:00:05Z",
                    "model": ai_metadata.get("model", "gpt-4") if ai_metadata else "gpt-4",
                    "tokens_used": ai_metadata.get("tokens", 0) if ai_metadata else 0,
                    "confidence": ai_metadata.get("confidence", 0.9) if ai_metadata else 0.9
                }
            }
        )
        
        return user_result["success"] and ai_result["success"]
    
    async def search_similar_conversations(self, user_id: str, query: str, limit: int = 10):
        """Search for similar conversations using vector search."""
        
        # Use vector search if available
        capabilities = await self.bridge.get_capabilities()
        
        search_type = "vector" if capabilities.get("vector_search") else "text"
        
        result = await self.data_layer.search_conversations(
            user_id, query,
            {
                "search_type": search_type,
                "limit": limit,
                "include_metadata": True,
                "similarity_threshold": 0.7
            }
        )
        
        if result["success"]:
            return result["data"]["results"]
        else:
            return []
    
    async def get_conversation_analytics(self, user_id: str):
        """Get AI conversation analytics."""
        
        analytics = await self.data_layer.get_analytics_summary(user_id)
        
        if analytics["success"]:
            return {
                "total_conversations": analytics["data"]["analytics"]["total_sessions"],
                "total_messages": analytics["data"]["analytics"]["total_messages"],
                "ai_responses": analytics["data"]["analytics"].get("assistant_messages", 0),
                "average_response_length": analytics["data"]["analytics"].get("avg_response_length", 0)
            }
        else:
            return {}

# Usage
async def ai_assistant_example():
    assistant = AIAssistantChat("./ai_assistant_data")
    
    await assistant.initialize()
    
    user_id = "ai_user_123"
    await assistant.data_layer.storage.create_user(user_id, {"name": "AI User"})
    session_id = await assistant.data_layer.storage.create_session(user_id, "AI Chat")
    
    # Store AI conversation
    await assistant.store_ai_conversation(
        user_id, session_id,
        "Explain quantum computing in simple terms",
        "Quantum computing uses quantum mechanical phenomena...",
        {"model": "gpt-4", "tokens": 150, "confidence": 0.95}
    )
    
    # Search for similar conversations
    similar = await assistant.search_similar_conversations(
        user_id, "quantum physics"
    )
    
    print(f"Found {len(similar)} similar conversations")
    
    await assistant.bridge.close()

asyncio.run(ai_assistant_example())
```

### High-Volume Chat System

Example for high-throughput chat applications:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ff_chat_integration import FFChatAppBridge

class HighVolumeChat:
    """High-volume chat system integration."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.bridge = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """Initialize for high volume."""
        self.bridge = await FFChatAppBridge.create_for_use_case(
            "high_volume_chat",
            self.storage_path,
            # High-volume optimizations
            performance_mode="speed",
            cache_size_mb=500,
            message_batch_size=200,
            enable_analytics=False,  # Disable for performance
            enable_compression=True  # Save storage space
        )
        
        self.data_layer = self.bridge.get_data_layer()
    
    async def batch_store_messages(self, messages_batch: list):
        """Store multiple messages efficiently."""
        
        # Use asyncio.gather for concurrent storage
        tasks = []
        for message_data in messages_batch:
            task = self.data_layer.store_chat_message(
                message_data["user_id"],
                message_data["session_id"],
                {
                    "role": message_data["role"],
                    "content": message_data["content"]
                }
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful operations
        successful = [r for r in results if not isinstance(r, Exception) and r.get("success")]
        
        return len(successful)
    
    async def get_multiple_conversations(self, conversation_requests: list):
        """Get multiple conversations concurrently."""
        
        tasks = []
        for request in conversation_requests:
            task = self.data_layer.get_chat_history(
                request["user_id"],
                request["session_id"],
                limit=request.get("limit", 50)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        conversations = []
        for result in results:
            if not isinstance(result, Exception) and result.get("success"):
                conversations.append(result["data"]["messages"])
            else:
                conversations.append([])
        
        return conversations
    
    async def stream_large_conversation(self, user_id: str, session_id: str, chunk_size: int = 100):
        """Stream large conversation in chunks."""
        
        messages_received = []
        
        async for chunk in self.data_layer.stream_conversation(
            user_id, session_id, chunk_size
        ):
            if chunk["success"]:
                chunk_messages = chunk["data"]["chunk"]
                messages_received.extend(chunk_messages)
                
                # Process chunk (e.g., send to client)
                yield chunk_messages
            else:
                break
    
    async def monitor_performance(self):
        """Monitor system performance."""
        
        while True:
            try:
                # Get performance metrics
                metrics = self.data_layer.get_performance_metrics()
                
                # Check if performance is acceptable
                avg_response = metrics["operation_metrics"]["store_chat_message"]["average_ms"]
                
                if avg_response > 100:  # 100ms threshold
                    print(f"WARNING: Slow response time: {avg_response:.2f}ms")
                
                # Check cache hit rate
                cache_rate = metrics["cache_stats"]["cache_hit_rate"]
                if cache_rate < 0.7:  # 70% threshold
                    print(f"WARNING: Low cache hit rate: {cache_rate:.2%}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)

# High-volume usage example
async def high_volume_example():
    chat_system = HighVolumeChat("./high_volume_data")
    
    await chat_system.initialize()
    
    # Simulate high volume scenario
    users = [f"user_{i}" for i in range(100)]
    sessions = {}
    
    # Create users and sessions
    for user_id in users:
        await chat_system.data_layer.storage.create_user(user_id, {"name": f"User {user_id}"})
        session_id = await chat_system.data_layer.storage.create_session(user_id, "High Volume Session")
        sessions[user_id] = session_id
    
    # Batch message storage
    messages_batch = []
    for i, user_id in enumerate(users):
        messages_batch.extend([
            {
                "user_id": user_id,
                "session_id": sessions[user_id],
                "role": "user",
                "content": f"High volume message {i} from {user_id}"
            },
            {
                "user_id": user_id,
                "session_id": sessions[user_id],
                "role": "assistant",
                "content": f"Response to message {i}"
            }
        ])
    
    # Store messages in batches
    batch_size = 50
    for i in range(0, len(messages_batch), batch_size):
        batch = messages_batch[i:i + batch_size]
        successful = await chat_system.batch_store_messages(batch)
        print(f"Stored {successful}/{len(batch)} messages in batch")
    
    # Concurrent conversation retrieval
    conversation_requests = [
        {"user_id": user_id, "session_id": sessions[user_id], "limit": 10}
        for user_id in users[:10]  # Test with first 10 users
    ]
    
    conversations = await chat_system.get_multiple_conversations(conversation_requests)
    print(f"Retrieved {len(conversations)} conversations")
    
    await chat_system.bridge.close()

asyncio.run(high_volume_example())
```

### Enterprise Chat Integration

Example for enterprise chat systems with security and compliance:

```python
import asyncio
import json
import logging
from datetime import datetime
from ff_chat_integration import FFChatAppBridge, FFIntegrationHealthMonitor

class EnterpriseChatSystem:
    """Enterprise-grade chat system integration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.bridge = None
        self.health_monitor = None
        self.audit_log = []
        
    async def initialize(self):
        """Initialize enterprise system with security features."""
        
        # Create bridge with enterprise settings
        self.bridge = await FFChatAppBridge.create_from_preset(
            "production",
            self.config["storage_path"],
            {
                "backup_enabled": True,
                "enable_compression": True,
                "enable_analytics": True,
                "cache_size_mb": self.config.get("cache_size_mb", 300),
                "max_session_size_mb": 200,
                "environment": "production"
            }
        )
        
        self.data_layer = self.bridge.get_data_layer()
        
        # Start comprehensive monitoring
        self.health_monitor = FFIntegrationHealthMonitor(self.bridge)
        await self.health_monitor.start_monitoring(interval_seconds=30)
        
        # Log initialization
        self._audit_log("SYSTEM_INIT", {"status": "success"})
    
    def _audit_log(self, action: str, details: dict):
        """Add entry to audit log."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        }
        self.audit_log.append(log_entry)
        
        # In real implementation, write to secure audit log
        logging.info(f"AUDIT: {action} - {details}")
    
    async def create_secure_session(self, user_id: str, department: str, classification: str):
        """Create session with enterprise metadata."""
        
        try:
            # Validate user permissions (simplified example)
            if not await self._validate_user_permissions(user_id, department):
                raise PermissionError(f"User {user_id} not authorized for department {department}")
            
            # Create user with enterprise metadata
            user_metadata = {
                "name": f"Enterprise User {user_id}",
                "department": department,
                "classification": classification,
                "created_at": datetime.utcnow().isoformat(),
                "permissions": await self._get_user_permissions(user_id)
            }
            
            await self.data_layer.storage.create_user(user_id, user_metadata)
            
            # Create session with security classification
            session_name = f"Enterprise Session - {classification}"
            session_id = await self.data_layer.storage.create_session(user_id, session_name)
            
            # Audit log
            self._audit_log("SESSION_CREATE", {
                "user_id": user_id,
                "session_id": session_id,
                "department": department,
                "classification": classification
            })
            
            return session_id
            
        except Exception as e:
            self._audit_log("SESSION_CREATE_ERROR", {
                "user_id": user_id,
                "error": str(e)
            })
            raise
    
    async def send_secure_message(self, user_id: str, session_id: str, 
                                content: str, classification: str = "internal"):
        """Send message with security controls."""
        
        try:
            # Content validation (simplified)
            if not await self._validate_message_content(content, classification):
                raise ValueError("Message content violates security policy")
            
            # Add security metadata
            message_data = {
                "role": "user",
                "content": content,
                "metadata": {
                    "classification": classification,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source_ip": self.config.get("source_ip", "unknown"),
                    "client_app": "enterprise_chat_v1.0"
                }
            }
            
            result = await self.data_layer.store_chat_message(
                user_id, session_id, message_data
            )
            
            if result["success"]:
                self._audit_log("MESSAGE_SENT", {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message_id": result["data"]["message_id"],
                    "classification": classification
                })
                
                return result["data"]["message_id"]
            else:
                raise RuntimeError(f"Message storage failed: {result['error']}")
                
        except Exception as e:
            self._audit_log("MESSAGE_SEND_ERROR", {
                "user_id": user_id,
                "session_id": session_id,
                "error": str(e)
            })
            raise
    
    async def get_authorized_conversation(self, user_id: str, session_id: str, 
                                        requester_id: str):
        """Get conversation with authorization checks."""
        
        try:
            # Check if requester has access
            if not await self._check_conversation_access(requester_id, user_id, session_id):
                raise PermissionError(f"User {requester_id} not authorized to access conversation")
            
            result = await self.data_layer.get_chat_history(user_id, session_id)
            
            if result["success"]:
                # Filter messages based on classification level
                filtered_messages = await self._filter_by_classification(
                    result["data"]["messages"], 
                    await self._get_user_clearance(requester_id)
                )
                
                self._audit_log("CONVERSATION_ACCESS", {
                    "requester_id": requester_id,
                    "target_user_id": user_id,
                    "session_id": session_id,
                    "messages_returned": len(filtered_messages)
                })
                
                return filtered_messages
            else:
                raise RuntimeError(f"Conversation retrieval failed: {result['error']}")
                
        except Exception as e:
            self._audit_log("CONVERSATION_ACCESS_ERROR", {
                "requester_id": requester_id,
                "target_user_id": user_id,
                "session_id": session_id,
                "error": str(e)
            })
            raise
    
    async def generate_compliance_report(self, start_date: str, end_date: str):
        """Generate compliance report."""
        
        try:
            # Get system health
            health = await self.health_monitor.comprehensive_health_check()
            
            # Get audit log entries for period
            relevant_logs = [
                log for log in self.audit_log
                if start_date <= log["timestamp"] <= end_date
            ]
            
            # Generate report
            report = {
                "report_period": {"start": start_date, "end": end_date},
                "system_health": {
                    "overall_status": health["overall_status"],
                    "optimization_score": health["optimization_score"],
                    "uptime": health.get("uptime_seconds", 0)
                },
                "audit_summary": {
                    "total_events": len(relevant_logs),
                    "session_creates": len([l for l in relevant_logs if l["action"] == "SESSION_CREATE"]),
                    "messages_sent": len([l for l in relevant_logs if l["action"] == "MESSAGE_SENT"]),
                    "access_attempts": len([l for l in relevant_logs if l["action"] == "CONVERSATION_ACCESS"]),
                    "errors": len([l for l in relevant_logs if "ERROR" in l["action"]])
                },
                "security_events": [
                    log for log in relevant_logs 
                    if "ERROR" in log["action"] or log["action"] == "PERMISSION_DENIED"
                ],
                "performance_metrics": self.data_layer.get_performance_metrics(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            self._audit_log("COMPLIANCE_REPORT", {
                "period": f"{start_date} to {end_date}",
                "events_included": len(relevant_logs)
            })
            
            return report
            
        except Exception as e:
            self._audit_log("COMPLIANCE_REPORT_ERROR", {"error": str(e)})
            raise
    
    # Security helper methods (simplified implementations)
    async def _validate_user_permissions(self, user_id: str, department: str) -> bool:
        """Validate user has permissions for department."""
        # Simplified - in real implementation, check against identity provider
        return True
    
    async def _get_user_permissions(self, user_id: str) -> list:
        """Get user permissions."""
        # Simplified - return basic permissions
        return ["read", "write", "chat"]
    
    async def _validate_message_content(self, content: str, classification: str) -> bool:
        """Validate message content against security policy."""
        # Simplified - check for sensitive keywords
        sensitive_keywords = ["confidential", "secret", "classified"]
        if classification == "internal" and any(keyword in content.lower() for keyword in sensitive_keywords):
            return False
        return True
    
    async def _check_conversation_access(self, requester_id: str, target_user_id: str, session_id: str) -> bool:
        """Check if requester can access target user's conversation."""
        # Simplified - allow access if same user or admin
        return requester_id == target_user_id or requester_id.startswith("admin_")
    
    async def _get_user_clearance(self, user_id: str) -> str:
        """Get user security clearance level."""
        # Simplified - return basic clearance
        return "internal"
    
    async def _filter_by_classification(self, messages: list, clearance: str) -> list:
        """Filter messages by user's clearance level."""
        # Simplified - return all messages for this example
        return messages

# Enterprise usage example
async def enterprise_example():
    config = {
        "storage_path": "/secure/enterprise/chat/data",
        "cache_size_mb": 500,
        "source_ip": "192.168.1.100"
    }
    
    enterprise_chat = EnterpriseChatSystem(config)
    
    try:
        await enterprise_chat.initialize()
        
        # Create secure session
        session_id = await enterprise_chat.create_secure_session(
            "emp_001", "engineering", "internal"
        )
        
        # Send secure messages
        msg_id = await enterprise_chat.send_secure_message(
            "emp_001", session_id,
            "Discussing the new product roadmap", 
            "internal"
        )
        
        # Access conversation (authorized)
        messages = await enterprise_chat.get_authorized_conversation(
            "emp_001", session_id, "emp_001"
        )
        
        print(f"Retrieved {len(messages)} authorized messages")
        
        # Generate compliance report
        report = await enterprise_chat.generate_compliance_report(
            "2024-01-01T00:00:00",
            "2024-01-31T23:59:59"
        )
        
        print(f"Compliance report: {report['audit_summary']['total_events']} events")
        
    except Exception as e:
        logging.error(f"Enterprise chat error: {e}")
    
    finally:
        if enterprise_chat.bridge:
            await enterprise_chat.bridge.close()

# Run enterprise example
asyncio.run(enterprise_example())
```

## Migration Examples

### From Wrapper-Based Configuration

Example of migrating from the old wrapper-based approach:

```python
import asyncio
from ff_chat_integration import FFChatConfigFactory, FFChatAppBridge

async def migration_example():
    """Example of migrating from wrapper-based configuration."""
    
    # Old wrapper configuration (what you had before)
    old_wrapper_config = {
        "base_path": "./legacy_chat_data",
        "cache_size_limit": 150,
        "enable_vector_search": True,
        "enable_compression": False,
        "enable_streaming": True,
        "performance_mode": "balanced",
        "environment": "production",
        "max_session_size_mb": 75,
        "message_batch_size": 100
    }
    
    print("=== BEFORE: Wrapper-Based Approach ===")
    print("# Old approach required complex wrapper classes:")
    print("class ConfigWrapper:")
    print("    def __init__(self, full_config):")
    print("        # 18+ lines of complex attribute copying...")
    print("        pass")
    print()
    
    print("=== AFTER: Bridge System ===")
    
    # Step 1: Migrate configuration
    factory = FFChatConfigFactory()
    new_config = factory.migrate_from_wrapper_config(old_wrapper_config)
    
    print(f"✓ Configuration migrated successfully")
    print(f"  Storage Path: {new_config.storage_path}")
    print(f"  Cache Size: {new_config.cache_size_mb}MB")
    print(f"  Performance Mode: {new_config.performance_mode}")
    
    # Step 2: Create bridge with migrated config
    bridge = await FFChatAppBridge.create_for_chat_app(
        new_config.storage_path,
        new_config.to_dict()
    )
    
    print(f"✓ Bridge created with migrated configuration")
    
    # Step 3: Verify all functionality works
    data_layer = bridge.get_data_layer()
    
    # Test basic operations
    user_id = "migration_test_user"
    await data_layer.storage.create_user(user_id, {"name": "Migration Test"})
    session_id = await data_layer.storage.create_session(user_id, "Migration Test Session")
    
    result = await data_layer.store_chat_message(
        user_id, session_id,
        {"role": "user", "content": "Testing migrated configuration"}
    )
    
    if result["success"]:
        print("✓ Message storage works with migrated configuration")
    
    history = await data_layer.get_chat_history(user_id, session_id)
    if history["success"]:
        print("✓ Message retrieval works with migrated configuration")
    
    # Step 4: Performance comparison
    print(f"\n=== Performance Improvements ===")
    perf_metrics = result["metadata"]["performance_metrics"]
    print(f"Message storage time: {perf_metrics['storage_time_ms']:.2f}ms")
    print(f"Cache enabled: {perf_metrics.get('cache_hit', 'N/A')}")
    
    # Clean up
    await bridge.close()
    
    print(f"\n=== Migration Complete ===")
    print("✓ No wrapper classes needed")
    print("✓ Simplified configuration") 
    print("✓ Improved performance")
    print("✓ Better error handling")
    print("✓ Comprehensive monitoring")

asyncio.run(migration_example())
```

## Performance Optimization Examples

### Caching Strategy

Example of optimizing performance with caching:

```python
import asyncio
from ff_chat_integration import FFChatAppBridge

async def caching_optimization_example():
    """Example of performance optimization with caching."""
    
    # Create bridge with cache optimization
    bridge = await FFChatAppBridge.create_for_chat_app(
        "./cache_optimized_data",
        {
            "performance_mode": "balanced",
            "cache_size_mb": 200,  # Larger cache
            "enable_compression": True,  # Compress cached data
        }
    )
    
    data_layer = bridge.get_data_layer()
    
    # Setup test data
    user_id = "cache_test_user"
    await data_layer.storage.create_user(user_id, {"name": "Cache Test"})
    session_id = await data_layer.storage.create_session(user_id, "Cache Test Session")
    
    # Add messages to cache
    for i in range(20):
        await data_layer.store_chat_message(
            user_id, session_id,
            {"role": "user", "content": f"Cache test message {i}"}
        )
    
    print("=== Cache Performance Test ===")
    
    # First retrieval (cold cache)
    import time
    start_time = time.time()
    result1 = await data_layer.get_chat_history(user_id, session_id, limit=20)
    first_time = (time.time() - start_time) * 1000
    
    print(f"First retrieval (cold cache): {first_time:.2f}ms")
    print(f"Cache hit: {result1['metadata']['performance_metrics'].get('cache_hit', False)}")
    
    # Second retrieval (warm cache)
    start_time = time.time()
    result2 = await data_layer.get_chat_history(user_id, session_id, limit=20)
    second_time = (time.time() - start_time) * 1000
    
    print(f"Second retrieval (warm cache): {second_time:.2f}ms")
    print(f"Cache hit: {result2['metadata']['performance_metrics'].get('cache_hit', False)}")
    
    if result2['metadata']['performance_metrics'].get('cache_hit'):
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"Cache improvement: {improvement:.1f}%")
    
    # Get cache statistics
    performance_metrics = data_layer.get_performance_metrics()
    cache_stats = performance_metrics.get("cache_stats", {})
    
    print(f"\n=== Cache Statistics ===")
    print(f"Cache size: {cache_stats.get('cache_size', 0)} items")
    print(f"Cache hits: {cache_stats.get('cache_hits', 0)}")
    print(f"Cache misses: {cache_stats.get('cache_misses', 0)}")
    print(f"Hit rate: {cache_stats.get('cache_hit_rate', 0):.2%}")
    
    await bridge.close()

asyncio.run(caching_optimization_example())
```

## Error Handling Examples

### Comprehensive Error Handling

Example of proper error handling patterns:

```python
import asyncio
import logging
from ff_chat_integration import (
    FFChatAppBridge, 
    ConfigurationError, 
    InitializationError, 
    StorageError,
    ChatIntegrationError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def error_handling_example():
    """Comprehensive error handling example."""
    
    bridge = None
    
    try:
        # Configuration error handling
        try:
            bridge = await FFChatAppBridge.create_for_chat_app(
                "./error_test_data",
                {
                    "performance_mode": "balanced",
                    "cache_size_mb": 100
                }
            )
            logger.info("Bridge created successfully")
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            logger.error(f"Error code: {e.error_code}")
            logger.error(f"Context: {e.context}")
            for suggestion in e.suggestions:
                logger.info(f"Suggestion: {suggestion}")
            return
            
        except InitializationError as e:
            logger.error(f"Initialization error: {e}")
            logger.error(f"Failed component: {e.context.get('component', 'unknown')}")
            logger.error(f"Failed step: {e.context.get('initialization_step', 'unknown')}")
            return
        
        # Data layer operations with error handling
        data_layer = bridge.get_data_layer()
        
        user_id = "error_test_user"
        
        # User creation with error handling
        try:
            await data_layer.storage.create_user(user_id, {"name": "Error Test"})
            logger.info("User created successfully")
        except StorageError as e:
            logger.error(f"User creation failed: {e}")
            logger.error(f"Storage operation: {e.context.get('operation', 'unknown')}")
            return
        
        # Session creation with error handling
        try:
            session_id = await data_layer.storage.create_session(user_id, "Error Test Session")
            logger.info(f"Session created: {session_id}")
        except StorageError as e:
            logger.error(f"Session creation failed: {e}")
            return
        
        # Message operations with comprehensive error handling
        message_operations = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"},
            {"role": "user", "content": "Test message 2"}
        ]
        
        successful_messages = 0
        for i, message in enumerate(message_operations):
            try:
                result = await data_layer.store_chat_message(user_id, session_id, message)
                
                if result["success"]:
                    successful_messages += 1
                    logger.info(f"Message {i+1} stored successfully")
                else:
                    logger.warning(f"Message {i+1} failed: {result['error']}")
                    if result["warnings"]:
                        for warning in result["warnings"]:
                            logger.warning(f"Warning: {warning}")
                    
            except ChatIntegrationError as e:
                logger.error(f"Message {i+1} integration error: {e}")
                # Continue with next message
                continue
            except Exception as e:
                logger.error(f"Message {i+1} unexpected error: {e}")
                continue
        
        logger.info(f"Successfully stored {successful_messages}/{len(message_operations)} messages")
        
        # Conversation retrieval with error handling
        try:
            history = await data_layer.get_chat_history(user_id, session_id)
            
            if history["success"]:
                messages_count = len(history["data"]["messages"])
                logger.info(f"Retrieved {messages_count} messages successfully")
                
                # Check for warnings
                if history["warnings"]:
                    for warning in history["warnings"]:
                        logger.warning(f"Retrieval warning: {warning}")
            else:
                logger.error(f"History retrieval failed: {history['error']}")
                
        except Exception as e:
            logger.error(f"History retrieval error: {e}")
        
        # Search operations with error handling
        try:
            search_result = await data_layer.search_conversations(
                user_id, "test",
                {"search_type": "text", "limit": 10}
            )
            
            if search_result["success"]:
                results_count = len(search_result["data"]["results"])
                logger.info(f"Search found {results_count} results")
            else:
                logger.error(f"Search failed: {search_result['error']}")
                
        except Exception as e:
            logger.error(f"Search error: {e}")
        
        # Health check with error handling
        try:
            health = await bridge.health_check()
            logger.info(f"System health: {health['status']}")
            
            if health["errors"]:
                logger.warning("Health check found errors:")
                for error in health["errors"]:
                    logger.warning(f"  - {error}")
                    
            if health["warnings"]:
                logger.info("Health check warnings:")
                for warning in health["warnings"]:
                    logger.info(f"  - {warning}")
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        
    finally:
        # Cleanup with error handling
        if bridge:
            try:
                await bridge.close()
                logger.info("Bridge closed successfully")
            except Exception as e:
                logger.error(f"Bridge closure error: {e}")

asyncio.run(error_handling_example())
```

## Best Practices Summary

### 1. Bridge Creation
```python
# ✓ Good: Use factory methods
bridge = await FFChatAppBridge.create_for_chat_app("./data")

# ✓ Better: Use presets for known scenarios
bridge = await FFChatAppBridge.create_from_preset("production", "./data")

# ✓ Best: Use use-case optimization
bridge = await FFChatAppBridge.create_for_use_case("ai_assistant", "./data")
```

### 2. Error Handling
```python
# ✓ Always handle standardized responses
result = await data_layer.store_chat_message(user_id, session_id, message)
if not result["success"]:
    logger.error(f"Operation failed: {result['error']}")

# ✓ Check for warnings
if result["warnings"]:
    for warning in result["warnings"]:
        logger.warning(warning)
```

### 3. Performance Monitoring
```python
# ✓ Monitor performance metrics
metrics = data_layer.get_performance_metrics()
avg_time = metrics["operation_metrics"]["store_chat_message"]["average_ms"]
if avg_time > 100:
    logger.warning(f"Slow performance: {avg_time}ms")
```

### 4. Resource Cleanup
```python
# ✓ Always close resources
try:
    # ... operations ...
    pass
finally:
    await bridge.close()
```

### 5. Health Monitoring
```python
# ✓ Use health monitoring for production systems
monitor = FFIntegrationHealthMonitor(bridge)
await monitor.start_monitoring()

# Check health periodically
health = await monitor.comprehensive_health_check()
if health["overall_status"] != "healthy":
    # Take corrective action
    pass
```

These examples demonstrate the full range of integration patterns available with the Chat Application Bridge System, from simple setups to complex enterprise deployments.