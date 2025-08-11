# Phase 3: Advanced Features Implementation

## Executive Summary

### Objectives
Implement specialized advanced features that enable professional and technical use cases, achieving 95% coverage. This phase adds external system integration, intelligent routing, and advanced debugging capabilities while maintaining the FF architecture patterns established in previous phases.

### Key Deliverables
- FF Tools Component (supports 7/22 use cases - 32%)
- FF Topic Router Component (supports 2/22 use cases - 9%) 
- FF Trace Logger Component (supports 2/22 use cases - 9%)
- Enhanced Persona System using existing FF panel infrastructure
- Advanced RAG integration using existing FF vector storage
- Enhanced Multimodal processing using existing FF document processing

### Prerequisites
- Phase 1-2 completed successfully (19/22 use cases working through FF integration)
- FF Chat Application operational with component system
- All existing FF managers tested and functional
- Component registry integrated with FF dependency injection

### Success Criteria
- ✅ FF Tools component integrates external systems safely using FF document processing
- ✅ FF Topic Router component intelligently delegates using FF search capabilities
- ✅ FF Trace Logger component provides advanced debugging using FF logging system
- ✅ Enhanced features extend existing FF capabilities without breaking compatibility
- ✅ Support for 21/22 use cases (95% coverage) through FF-integrated architecture

## Technical Specifications

### Module Coverage Analysis

Based on FF integration capabilities, Phase 3 components provide:

| FF Component | Use Cases Supported | FF Backend Integration |
|--------------|-------------------|----------------------|
| **FF Tools** | translation_chat, personal_assistant, language_tutor, ai_notetaker, chatops_assistant, cross_team_concierge, auto_task_agent | FFDocumentProcessingManager + FFStorageManager |
| **FF Topic Router** | topic_delegation | FFSearchManager + FFVectorStorageManager |
| **FF Trace Logger** | ai_debate, prompt_sandbox | ff_utils.ff_logging + FFStorageManager |

**Phase 3 adds**: 2 new use cases (topic_delegation, prompt_sandbox)
**Total Coverage**: 21/22 use cases (95%) using enhanced FF infrastructure

### Implementation Details

#### 1. FF Tools Component

**File**: `ff_chat_components/ff_tools_component.py`

```python
"""
FF Tools Component - External System Integration

Provides secure integration with external systems, APIs, and services
using existing FF document processing and storage as backend.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import aiohttp
import subprocess
import tempfile
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import shlex

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_document_processing_manager import FFDocumentProcessingManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFDocumentDTO
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_utils.ff_validation import validate_input
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

class FFToolType(Enum):
    """Types of tools supported by FF Tools component"""
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    WEB_SEARCH = "web_search"
    EMAIL = "email"
    CALENDAR = "calendar"
    DOCUMENT_PROCESSING = "document_processing"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_INFO = "system_info"

class FFToolSecurityLevel(Enum):
    """Security levels for tool execution"""
    READ_ONLY = "read_only"      # No modifications allowed
    RESTRICTED = "restricted"    # Limited safe operations
    SANDBOXED = "sandboxed"     # Isolated execution
    TRUSTED = "trusted"         # Full access (use with caution)

@dataclass
class FFToolDefinition:
    """Definition of a tool available to the FF Tools component"""
    name: str
    tool_type: FFToolType
    description: str
    security_level: FFToolSecurityLevel
    parameters: Dict[str, Any]
    timeout: int = 30
    enabled: bool = True
    
@dataclass
class FFToolsConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Tools component"""
    
    # Tool execution settings
    execution_timeout: int = 30
    max_concurrent_tools: int = 3
    enable_sandboxing: bool = True
    
    # Security settings
    default_security_level: str = "restricted"
    allowed_domains: List[str] = field(default_factory=lambda: ["api.openai.com", "httpbin.org"])
    blocked_domains: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    # FF integration settings
    use_ff_document_processing: bool = True
    store_tool_results: bool = True
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Available tools configuration
    enabled_tools: List[str] = field(default_factory=lambda: [
        "web_search", "document_analysis", "file_info", "api_call", "email_draft"
    ])
    
    def __post_init__(self):
        super().__post_init__()
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
        if self.max_concurrent_tools <= 0:
            raise ValueError("max_concurrent_tools must be positive")

class FFToolsComponent(FFChatComponentProtocol):
    """
    FF Tools component using existing FF document processing backend.
    
    Provides external system integration while leveraging FF document
    processing for file handling and FF storage for result persistence.
    """
    
    def __init__(self, config: FFToolsConfigDTO):
        """
        Initialize FF tools component.
        
        Args:
            config: Tools component configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.ff_storage: Optional[FFStorageManager] = None
        self.ff_document_processor: Optional[FFDocumentProcessingManager] = None
        
        # Tool management
        self.available_tools: Dict[str, FFToolDefinition] = {}
        self.tool_cache: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Component state
        self._initialized = False
        
        # Initialize available tools
        self._setup_default_tools()
    
    def _setup_default_tools(self) -> None:
        """Setup default tool definitions"""
        try:
            # Web search tool
            self.available_tools["web_search"] = FFToolDefinition(
                name="web_search",
                tool_type=FFToolType.WEB_SEARCH,
                description="Search the web for information",
                security_level=FFToolSecurityLevel.RESTRICTED,
                parameters={
                    "query": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "default": 5}
                }
            )
            
            # Document analysis tool using FF document processing
            self.available_tools["document_analysis"] = FFToolDefinition(
                name="document_analysis",
                tool_type=FFToolType.DOCUMENT_PROCESSING,
                description="Analyze documents using FF document processing",
                security_level=FFToolSecurityLevel.READ_ONLY,
                parameters={
                    "document_path": {"type": "string", "required": True},
                    "analysis_type": {"type": "string", "default": "summary"}
                }
            )
            
            # File information tool
            self.available_tools["file_info"] = FFToolDefinition(
                name="file_info",
                tool_type=FFToolType.FILE_OPERATION,
                description="Get information about files",
                security_level=FFToolSecurityLevel.READ_ONLY,
                parameters={
                    "file_path": {"type": "string", "required": True}
                }
            )
            
            # API call tool
            self.available_tools["api_call"] = FFToolDefinition(
                name="api_call",
                tool_type=FFToolType.API_CALL,
                description="Make HTTP API calls",
                security_level=FFToolSecurityLevel.RESTRICTED,
                parameters={
                    "url": {"type": "string", "required": True},
                    "method": {"type": "string", "default": "GET"},
                    "headers": {"type": "object", "default": {}},
                    "data": {"type": "object", "default": {}}
                }
            )
            
            # Email drafting tool
            self.available_tools["email_draft"] = FFToolDefinition(
                name="email_draft",
                tool_type=FFToolType.EMAIL,
                description="Draft email content",
                security_level=FFToolSecurityLevel.READ_ONLY,
                parameters={
                    "recipient": {"type": "string", "required": True},
                    "subject": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True}
                }
            )
            
            self.logger.info(f"Setup {len(self.available_tools)} default tools")
            
        except Exception as e:
            self.logger.error(f"Error setting up default tools: {e}")
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_tools",
            "version": "1.0.0",
            "description": "FF tools component using FF document processing backend",
            "capabilities": ["external_integration", "api_calls", "file_operations", "document_processing"],
            "use_cases": [
                "translation_chat", "personal_assistant", "language_tutor",
                "ai_notetaker", "chatops_assistant", "cross_team_concierge", "auto_task_agent"
            ],
            "ff_dependencies": ["FFStorageManager", "FFDocumentProcessingManager"],
            "available_tools": list(self.available_tools.keys())
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_document_processor = dependencies.get("ff_document_processor")
            
            if not self.ff_storage:
                raise ValueError("FFStorageManager dependency required")
            
            if not self.ff_document_processor and self.config.use_ff_document_processing:
                self.logger.warning("FFDocumentProcessingManager not available - document tools disabled")
                self.config.use_ff_document_processing = False
                # Disable document-related tools
                if "document_analysis" in self.available_tools:
                    self.available_tools["document_analysis"].enabled = False
            
            # Filter enabled tools
            enabled_tools = {
                name: tool for name, tool in self.available_tools.items()
                if tool.enabled and name in self.config.enabled_tools
            }
            self.available_tools = enabled_tools
            
            self._initialized = True
            self.logger.info(f"FF Tools component initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Tools component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message and execute tools as needed using FF backend.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: FF message DTO
            context: Optional context with tool requests
            
        Returns:
            Processing results with tool execution results
        """
        if not self._initialized:
            raise RuntimeError("FF Tools component not initialized")
        
        try:
            self.logger.info(f"Processing tools message in session {session_id}")
            
            # Parse tool requests from message or context
            tool_requests = await self._parse_tool_requests(message, context)
            
            if not tool_requests:
                return {
                    "success": True,
                    "component": "ff_tools",
                    "message": "No tool requests found",
                    "available_tools": list(self.available_tools.keys())
                }
            
            # Execute requested tools
            tool_results = []
            for tool_request in tool_requests:
                result = await self._execute_tool(
                    session_id=session_id,
                    user_id=user_id,
                    tool_request=tool_request
                )
                tool_results.append(result)
            
            # Store results using FF storage if configured
            if self.config.store_tool_results:
                await self._store_tool_results(session_id, user_id, tool_results)
            
            return {
                "success": True,
                "component": "ff_tools",
                "processor": "ff_document_backend",
                "tools_executed": len(tool_results),
                "tool_results": tool_results,
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "execution_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing tools message: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_tools"
            }
    
    async def _parse_tool_requests(self, 
                                   message: FFMessageDTO,
                                   context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse tool requests from message content or context"""
        try:
            tool_requests = []
            
            # Check context for explicit tool requests
            if context and "tool_requests" in context:
                tool_requests.extend(context["tool_requests"])
            
            # Parse message content for tool mentions (simple parsing)
            content = message.content.lower()
            
            # Look for tool keywords
            if "search" in content or "find" in content:
                # Extract search query
                query = message.content
                tool_requests.append({
                    "tool": "web_search",
                    "parameters": {"query": query}
                })
            
            if "analyze" in content and ("document" in content or "file" in content):
                tool_requests.append({
                    "tool": "document_analysis",
                    "parameters": {"analysis_type": "summary"}
                })
            
            if "email" in content and ("draft" in content or "write" in content):
                tool_requests.append({
                    "tool": "email_draft",
                    "parameters": {"content": message.content}
                })
            
            return tool_requests
            
        except Exception as e:
            self.logger.error(f"Error parsing tool requests: {e}")
            return []
    
    async def _execute_tool(self, 
                            session_id: str,
                            user_id: str,
                            tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool request"""
        try:
            tool_name = tool_request.get("tool")
            parameters = tool_request.get("parameters", {})
            
            if tool_name not in self.available_tools:
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' not available"
                }
            
            tool_def = self.available_tools[tool_name]
            
            # Check cache if enabled
            cache_key = f"{tool_name}_{hash(str(parameters))}"
            if self.config.enable_result_caching and cache_key in self.tool_cache:
                cached_result = self.tool_cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < timedelta(seconds=self.config.cache_ttl_seconds):
                    return {
                        "tool": tool_name,
                        "success": True,
                        "result": cached_result["result"],
                        "cached": True,
                        "timestamp": cached_result["timestamp"].isoformat()
                    }
            
            # Execute tool based on type
            if tool_def.tool_type == FFToolType.WEB_SEARCH:
                result = await self._execute_web_search(parameters)
            elif tool_def.tool_type == FFToolType.DOCUMENT_PROCESSING:
                result = await self._execute_document_analysis(parameters)
            elif tool_def.tool_type == FFToolType.FILE_OPERATION:
                result = await self._execute_file_operation(parameters)
            elif tool_def.tool_type == FFToolType.API_CALL:
                result = await self._execute_api_call(parameters)
            elif tool_def.tool_type == FFToolType.EMAIL:
                result = await self._execute_email_draft(parameters)
            else:
                result = {"error": f"Tool type {tool_def.tool_type} not implemented"}
            
            # Cache result if successful
            if self.config.enable_result_caching and "error" not in result:
                self.tool_cache[cache_key] = {
                    "result": result,
                    "timestamp": datetime.now()
                }
            
            return {
                "tool": tool_name,
                "success": "error" not in result,
                "result": result,
                "cached": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_request.get('tool')}: {e}")
            return {
                "tool": tool_request.get("tool"),
                "success": False,
                "error": str(e)
            }
    
    async def _execute_web_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search (mock implementation for security)"""
        try:
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 5)
            
            # Mock web search results for security
            # In production, this would integrate with search APIs
            results = [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com/result1",
                    "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would call a search API."
                },
                {
                    "title": f"Another result for: {query}",
                    "url": "https://example.com/result2", 
                    "snippet": f"Second mock result showing how the FF Tools component would handle search requests."
                }
            ]
            
            return {
                "query": query,
                "results": results[:max_results],
                "total_results": len(results)
            }
            
        except Exception as e:
            return {"error": f"Web search error: {str(e)}"}
    
    async def _execute_document_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document analysis using FF document processing"""
        try:
            if not self.ff_document_processor:
                return {"error": "FF document processing not available"}
            
            document_path = parameters.get("document_path", "")
            analysis_type = parameters.get("analysis_type", "summary")
            
            # Use FF document processing capabilities
            # For now, return mock analysis
            return {
                "document_path": document_path,
                "analysis_type": analysis_type,
                "analysis": f"FF document analysis of {document_path}: This document appears to contain {analysis_type} information processed through the FF document processing system.",
                "processed_by": "ff_document_processor"
            }
            
        except Exception as e:
            return {"error": f"Document analysis error: {str(e)}"}
    
    async def _execute_file_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute safe file operations"""
        try:
            file_path = parameters.get("file_path", "")
            
            # Validate file path for security
            if not validate_input(file_path, "file_path"):
                return {"error": "Invalid file path"}
            
            path = Path(file_path)
            
            if not path.exists():
                return {"error": "File not found"}
            
            # Get safe file information
            file_info = {
                "path": str(path),
                "name": path.name,
                "suffix": path.suffix,
                "size": path.stat().st_size if path.is_file() else 0,
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
            return file_info
            
        except Exception as e:
            return {"error": f"File operation error: {str(e)}"}
    
    async def _execute_api_call(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP API calls with security restrictions"""
        try:
            url = parameters.get("url", "")
            method = parameters.get("method", "GET").upper()
            headers = parameters.get("headers", {})
            data = parameters.get("data", {})
            
            # Security validation
            if not self._validate_url(url):
                return {"error": "URL not allowed by security policy"}
            
            # Mock API call for security (in production, implement with proper restrictions)
            return {
                "url": url,
                "method": method,
                "status_code": 200,
                "response": f"Mock API response for {method} {url}",
                "note": "This is a mock response for security. Real implementation would make actual HTTP calls with proper restrictions."
            }
            
        except Exception as e:
            return {"error": f"API call error: {str(e)}"}
    
    async def _execute_email_draft(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Draft email content"""
        try:
            recipient = parameters.get("recipient", "")
            subject = parameters.get("subject", "")
            content = parameters.get("content", "")
            
            # Generate email draft
            draft = {
                "to": recipient,
                "subject": subject,
                "body": f"Dear {recipient},\n\n{content}\n\nBest regards,\n[Your Name]",
                "created": datetime.now().isoformat(),
                "status": "draft"
            }
            
            return draft
            
        except Exception as e:
            return {"error": f"Email draft error: {str(e)}"}
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL against security policy"""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            
            # Check allowed domains
            if self.config.allowed_domains:
                if parsed.netloc not in self.config.allowed_domains:
                    return False
            
            # Check blocked domains
            if parsed.netloc in self.config.blocked_domains:
                return False
            
            # Only allow HTTP/HTTPS
            if parsed.scheme not in ["http", "https"]:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _store_tool_results(self, 
                                  session_id: str,
                                  user_id: str,
                                  tool_results: List[Dict[str, Any]]) -> None:
        """Store tool results using FF storage"""
        try:
            if not self.ff_storage:
                return
            
            # Create FF message with tool results
            results_content = f"Tool execution results: {len(tool_results)} tools executed"
            
            tool_message = FFMessageDTO(
                role="system",
                content=results_content,
                metadata={
                    "component": "ff_tools",
                    "tool_results": tool_results,
                    "execution_timestamp": datetime.now().isoformat()
                }
            )
            
            # Store using FF storage
            await self.ff_storage.add_message(user_id, session_id, tool_message)
            
        except Exception as e:
            self.logger.error(f"Error storing tool results: {e}")
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List available tools with their capabilities"""
        return [
            {
                "name": tool.name,
                "type": tool.tool_type.value,
                "description": tool.description,
                "security_level": tool.security_level.value,
                "enabled": tool.enabled
            }
            for tool in self.available_tools.values()
        ]
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Tools component")
        
        # Cancel active executions
        for task in self.active_executions.values():
            if not task.done():
                task.cancel()
        
        self.active_executions.clear()
        self.tool_cache.clear()
        self._initialized = False
```

#### 2. FF Topic Router Component

**File**: `ff_chat_components/ff_topic_router_component.py`

```python
"""
FF Topic Router Component - Intelligent Topic Routing

Provides intelligent topic detection and routing using existing
FF search and vector storage capabilities as backend.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime
from dataclasses import dataclass, field

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager 
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

@dataclass
class FFTopicRoute:
    """Definition of a topic routing rule"""
    name: str
    keywords: List[str]
    patterns: List[str]
    target_agent: str
    confidence_threshold: float
    priority: int = 100
    enabled: bool = True

@dataclass
class FFTopicRouterConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Topic Router component"""
    
    # Routing settings
    confidence_threshold: float = 0.7
    enable_learning: bool = True
    enable_vector_matching: bool = True
    
    # Topic detection settings
    min_topic_confidence: float = 0.5
    max_topics_per_message: int = 3
    topic_cache_ttl: int = 300  # seconds
    
    # FF integration settings
    use_ff_search: bool = True
    use_ff_vector_storage: bool = True
    vector_similarity_threshold: float = 0.8
    
    # Fallback settings
    default_agent: str = "general"
    enable_fallback_routing: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.min_topic_confidence <= 1:
            raise ValueError("min_topic_confidence must be between 0 and 1")

class FFTopicRouterComponent(FFChatComponentProtocol):
    """
    FF Topic Router component using existing FF search and vector backend.
    
    Provides intelligent topic detection and routing while leveraging
    FF search for pattern matching and FF vector storage for semantic similarity.
    """
    
    def __init__(self, config: FFTopicRouterConfigDTO):
        """
        Initialize FF topic router component.
        
        Args:
            config: Topic router configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.ff_storage: Optional[FFStorageManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        self.ff_vector: Optional[FFVectorStorageManager] = None
        
        # Routing configuration
        self.topic_routes: Dict[str, FFTopicRoute] = {}
        self.topic_cache: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        
        # Component state  
        self._initialized = False
        
        # Setup default routes
        self._setup_default_routes()
    
    def _setup_default_routes(self) -> None:
        """Setup default topic routing rules"""
        try:
            # Technical topics
            self.topic_routes["technical"] = FFTopicRoute(
                name="technical",
                keywords=["code", "programming", "software", "development", "algorithm", "database", "api"],
                patterns=["how to code", "programming help", "technical issue", "software problem"],
                target_agent="technical_expert",
                confidence_threshold=0.8
            )
            
            # Creative topics
            self.topic_routes["creative"] = FFTopicRoute(
                name="creative",
                keywords=["creative", "design", "art", "story", "writing", "brainstorm", "idea"],
                patterns=["creative writing", "design help", "brainstorming", "artistic"],
                target_agent="creative_assistant",
                confidence_threshold=0.7
            )
            
            # Business topics
            self.topic_routes["business"] = FFTopicRoute(
                name="business",
                keywords=["business", "marketing", "sales", "strategy", "finance", "management"],
                patterns=["business plan", "marketing strategy", "financial analysis"],
                target_agent="business_advisor",
                confidence_threshold=0.75
            )
            
            # Educational topics
            self.topic_routes["educational"] = FFTopicRoute(
                name="educational",
                keywords=["learn", "education", "tutorial", "explain", "teach", "study"],
                patterns=["how to learn", "explain this", "tutorial on", "teach me"],
                target_agent="tutor",
                confidence_threshold=0.7
            )
            
            self.logger.info(f"Setup {len(self.topic_routes)} default topic routes")
            
        except Exception as e:
            self.logger.error(f"Error setting up default routes: {e}")
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_topic_router",
            "version": "1.0.0",
            "description": "FF topic router component using FF search and vector backend",
            "capabilities": ["topic_detection", "intelligent_routing", "agent_delegation", "pattern_matching"],
            "use_cases": ["topic_delegation"],
            "ff_dependencies": ["FFStorageManager", "FFSearchManager", "FFVectorStorageManager"],
            "available_routes": list(self.topic_routes.keys())
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_search = dependencies.get("ff_search")
            self.ff_vector = dependencies.get("ff_vector")
            
            if not self.ff_storage:
                raise ValueError("FFStorageManager dependency required")
            
            if not self.ff_search and self.config.use_ff_search:
                self.logger.warning("FFSearchManager not available - search-based routing disabled")
                self.config.use_ff_search = False
            
            if not self.ff_vector and self.config.use_ff_vector_storage:
                self.logger.warning("FFVectorStorageManager not available - vector routing disabled")
                self.config.use_ff_vector_storage = False
            
            self._initialized = True
            self.logger.info("FF Topic Router component initialized with FF backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Topic Router component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message and determine routing using FF backend.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: FF message DTO
            context: Optional context information
            
        Returns:
            Processing results with routing decisions
        """
        if not self._initialized:
            raise RuntimeError("FF Topic Router component not initialized")
        
        try:
            self.logger.info(f"Processing topic routing for session {session_id}")
            
            # Detect topics using multiple methods
            detected_topics = await self._detect_topics(message.content, user_id, session_id)
            
            # Determine routing based on detected topics
            routing_decision = await self._determine_routing(detected_topics, message, context)
            
            # Store routing decision using FF storage
            await self._store_routing_decision(session_id, user_id, message, routing_decision)
            
            # Update routing history for learning
            if self.config.enable_learning:
                self.routing_history.append({
                    "message_content": message.content,
                    "detected_topics": detected_topics,
                    "routing_decision": routing_decision,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "component": "ff_topic_router",
                "processor": "ff_search_vector_backend",
                "detected_topics": detected_topics,
                "routing_decision": routing_decision,
                "target_agent": routing_decision.get("target_agent"),
                "confidence": routing_decision.get("confidence"),
                "metadata": {
                    "session_id": session_id,
                    "routing_method": routing_decision.get("method"),
                    "fallback_used": routing_decision.get("fallback_used", False)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing topic routing: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_topic_router"
            }
    
    async def _detect_topics(self, 
                             content: str,
                             user_id: str,
                             session_id: str) -> List[Dict[str, Any]]:
        """Detect topics using multiple FF backend methods"""
        detected_topics = []
        
        try:
            content_lower = content.lower()
            
            # Method 1: Keyword-based detection
            keyword_topics = await self._detect_keywords(content_lower)
            detected_topics.extend(keyword_topics)
            
            # Method 2: Pattern-based detection
            pattern_topics = await self._detect_patterns(content_lower)
            detected_topics.extend(pattern_topics)
            
            # Method 3: FF search-based detection
            if self.config.use_ff_search and self.ff_search:
                search_topics = await self._detect_with_ff_search(content, user_id)
                detected_topics.extend(search_topics)
            
            # Method 4: FF vector-based detection
            if self.config.use_ff_vector_storage and self.ff_vector:
                vector_topics = await self._detect_with_ff_vector(content, user_id)
                detected_topics.extend(vector_topics)
            
            # Consolidate and rank topics
            consolidated_topics = self._consolidate_topics(detected_topics)
            
            # Filter by confidence threshold
            filtered_topics = [
                topic for topic in consolidated_topics
                if topic["confidence"] >= self.config.min_topic_confidence
            ]
            
            # Limit number of topics
            return filtered_topics[:self.config.max_topics_per_message]
            
        except Exception as e:
            self.logger.error(f"Error detecting topics: {e}")
            return []
    
    async def _detect_keywords(self, content: str) -> List[Dict[str, Any]]:
        """Detect topics based on keyword matching"""
        keyword_topics = []
        
        for route_name, route in self.topic_routes.items():
            if not route.enabled:
                continue
                
            keyword_matches = 0
            total_keywords = len(route.keywords)
            
            for keyword in route.keywords:
                if keyword.lower() in content:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                confidence = keyword_matches / total_keywords
                keyword_topics.append({
                    "topic": route_name,
                    "confidence": confidence,
                    "method": "keyword",
                    "matches": keyword_matches,
                    "total_keywords": total_keywords
                })
        
        return keyword_topics
    
    async def _detect_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Detect topics based on pattern matching"""
        pattern_topics = []
        
        for route_name, route in self.topic_routes.items():
            if not route.enabled:
                continue
                
            pattern_matches = 0
            total_patterns = len(route.patterns)
            
            for pattern in route.patterns:
                if pattern.lower() in content:
                    pattern_matches += 1
            
            if pattern_matches > 0:
                confidence = pattern_matches / total_patterns
                pattern_topics.append({
                    "topic": route_name,
                    "confidence": confidence,
                    "method": "pattern",
                    "matches": pattern_matches,
                    "total_patterns": total_patterns
                })
        
        return pattern_topics
    
    async def _detect_with_ff_search(self, content: str, user_id: str) -> List[Dict[str, Any]]:
        """Detect topics using FF search capabilities"""
        try:
            if not self.ff_search:
                return []
            
            # Search for similar messages using FF search
            search_results = await self.ff_search.search_messages(
                user_id=user_id,
                query=content,
                limit=5
            )
            
            # Analyze search results for topic patterns
            search_topics = []
            if search_results:
                # Simple topic extraction from search results
                # In production, this would use more sophisticated analysis
                search_topics.append({
                    "topic": "contextual",
                    "confidence": 0.6,
                    "method": "ff_search",
                    "search_results": len(search_results)
                })
            
            return search_topics
            
        except Exception as e:
            self.logger.error(f"Error in FF search-based topic detection: {e}")
            return []
    
    async def _detect_with_ff_vector(self, content: str, user_id: str) -> List[Dict[str, Any]]:
        """Detect topics using FF vector storage capabilities"""
        try:
            if not self.ff_vector:
                return []
            
            # Use FF vector storage for semantic similarity
            # This would involve generating embeddings and finding similar content
            vector_topics = []
            
            # Placeholder for vector-based topic detection
            # In production, this would use FF vector storage to find semantically similar content
            vector_topics.append({
                "topic": "semantic",
                "confidence": 0.5,
                "method": "ff_vector",
                "note": "Vector-based topic detection placeholder"
            })
            
            return vector_topics
            
        except Exception as e:
            self.logger.error(f"Error in FF vector-based topic detection: {e}")
            return []
    
    def _consolidate_topics(self, detected_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate and rank detected topics"""
        try:
            # Group topics by name
            topic_groups = {}
            for topic in detected_topics:
                topic_name = topic["topic"]
                if topic_name not in topic_groups:
                    topic_groups[topic_name] = []
                topic_groups[topic_name].append(topic)
            
            # Consolidate each group
            consolidated = []
            for topic_name, group in topic_groups.items():
                # Calculate weighted confidence
                total_confidence = sum(t["confidence"] for t in group)
                avg_confidence = total_confidence / len(group)
                
                # Boost confidence for multiple detection methods
                method_boost = min(len(group) * 0.1, 0.3)
                final_confidence = min(avg_confidence + method_boost, 1.0)
                
                consolidated.append({
                    "topic": topic_name,
                    "confidence": final_confidence,
                    "methods": [t["method"] for t in group],
                    "detection_count": len(group)
                })
            
            # Sort by confidence
            consolidated.sort(key=lambda x: x["confidence"], reverse=True)
            
            return consolidated
            
        except Exception as e:
            self.logger.error(f"Error consolidating topics: {e}")
            return detected_topics
    
    async def _determine_routing(self, 
                                 detected_topics: List[Dict[str, Any]],
                                 message: FFMessageDTO,
                                 context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine routing based on detected topics"""
        try:
            # Check for explicit routing in context
            if context and "target_agent" in context:
                return {
                    "target_agent": context["target_agent"],
                    "confidence": 1.0,
                    "method": "explicit",
                    "reason": "Explicitly specified in context"
                }
            
            # Find best matching route
            best_route = None
            best_confidence = 0
            
            for topic in detected_topics:
                topic_name = topic["topic"]
                topic_confidence = topic["confidence"]
                
                if topic_name in self.topic_routes:
                    route = self.topic_routes[topic_name]
                    if route.enabled and topic_confidence >= route.confidence_threshold:
                        if topic_confidence > best_confidence:
                            best_route = route
                            best_confidence = topic_confidence
            
            if best_route:
                return {
                    "target_agent": best_route.target_agent,
                    "confidence": best_confidence,
                    "method": "topic_matching",
                    "matched_topic": best_route.name,
                    "reason": f"Matched topic {best_route.name} with confidence {best_confidence:.2f}"
                }
            
            # Fallback routing
            if self.config.enable_fallback_routing:
                return {
                    "target_agent": self.config.default_agent,
                    "confidence": 0.5,
                    "method": "fallback",
                    "fallback_used": True,
                    "reason": "No confident topic match found, using default agent"
                }
            
            # No routing decision
            return {
                "target_agent": None,
                "confidence": 0,
                "method": "none",
                "reason": "No routing decision could be made"
            }
            
        except Exception as e:
            self.logger.error(f"Error determining routing: {e}")
            return {
                "target_agent": self.config.default_agent,
                "confidence": 0,
                "method": "error",
                "error": str(e)
            }
    
    async def _store_routing_decision(self, 
                                      session_id: str,
                                      user_id: str,
                                      message: FFMessageDTO,
                                      routing_decision: Dict[str, Any]) -> None:
        """Store routing decision using FF storage"""
        try:
            if not self.ff_storage:
                return
            
            # Create routing metadata message
            routing_content = f"Topic routing: {routing_decision.get('target_agent', 'none')}"
            
            routing_message = FFMessageDTO(
                role="system",
                content=routing_content,
                metadata={
                    "component": "ff_topic_router",
                    "routing_decision": routing_decision,
                    "original_message_id": message.message_id,
                    "routing_timestamp": datetime.now().isoformat()
                }
            )
            
            await self.ff_storage.add_message(user_id, session_id, routing_message)
            
        except Exception as e:
            self.logger.error(f"Error storing routing decision: {e}")
    
    async def add_topic_route(self, route: FFTopicRoute) -> bool:
        """Add a new topic route"""
        try:
            self.topic_routes[route.name] = route
            self.logger.info(f"Added topic route: {route.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding topic route: {e}")
            return False
    
    async def get_routing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent routing history"""
        return self.routing_history[-limit:] if self.routing_history else []
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Topic Router component")
        self.topic_cache.clear()
        self.routing_history.clear()
        self._initialized = False
```

#### 3. FF Trace Logger Component

**File**: `ff_chat_components/ff_trace_logger_component.py`

```python
"""
FF Trace Logger Component - Advanced Debugging and Analysis

Provides advanced conversation logging, tracing, and analysis using
existing FF logging system and storage as backend.
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_class_configs.ff_base_config import FFBaseConfigDTO
from ff_utils.ff_logging import get_logger
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol

logger = get_logger(__name__)

class FFTraceLevel(Enum):
    """Trace logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FFTraceEvent(Enum):
    """Types of trace events"""
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    COMPONENT_CALLED = "component_called"
    COMPONENT_RESPONSE = "component_response"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"

@dataclass
class FFTraceEntry:
    """Individual trace log entry"""
    timestamp: str
    session_id: str
    user_id: str
    event_type: FFTraceEvent
    level: FFTraceLevel
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None

@dataclass
class FFTraceLoggerConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Trace Logger component"""
    
    # Logging settings
    log_level: str = "info"
    enable_performance_tracing: bool = True
    enable_component_tracing: bool = True
    enable_error_tracing: bool = True
    
    # Storage settings
    store_traces_in_ff: bool = True
    trace_retention_days: int = 7
    max_trace_entries: int = 10000
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    enable_real_time_export: bool = False
    export_batch_size: int = 100
    
    # Analysis settings
    enable_conversation_analysis: bool = True
    enable_performance_analysis: bool = True
    analysis_window_minutes: int = 60
    
    def __post_init__(self):
        super().__post_init__()
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")

class FFTraceLoggerComponent(FFChatComponentProtocol):
    """
    FF Trace Logger component using existing FF logging and storage.
    
    Provides advanced tracing, debugging, and analysis capabilities
    while leveraging FF logging system and FF storage for persistence.
    """
    
    def __init__(self, config: FFTraceLoggerConfigDTO):
        """
        Initialize FF trace logger component.
        
        Args:
            config: Trace logger configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.ff_storage: Optional[FFStorageManager] = None
        
        # Trace storage
        self.trace_entries: List[FFTraceEntry] = []
        self.session_traces: Dict[str, List[FFTraceEntry]] = {}
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # Analysis data
        self.conversation_stats: Dict[str, Dict[str, Any]] = {}
        self.error_patterns: List[Dict[str, Any]] = []
        
        # Component state
        self._initialized = False
        
        # Setup trace logger
        self.trace_logger = get_logger(f"{__name__}.tracer")
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Component metadata and capabilities"""
        return {
            "name": "ff_trace_logger",
            "version": "1.0.0",
            "description": "FF trace logger component using FF logging and storage backend",
            "capabilities": ["conversation_logging", "performance_monitoring", "error_tracking", "debug_analysis"],
            "use_cases": ["ai_debate", "prompt_sandbox"],
            "ff_dependencies": ["FFStorageManager"],
            "trace_levels": [level.value for level in FFTraceLevel],
            "event_types": [event.value for event in FFTraceEvent]
        }
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            # Get FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            
            if not self.ff_storage and self.config.store_traces_in_ff:
                self.logger.warning("FFStorageManager not available - FF storage disabled for traces")
                self.config.store_traces_in_ff = False
            
            # Initialize trace storage
            self.trace_entries = []
            self.session_traces = {}
            self.performance_metrics = {}
            
            # Start background cleanup task
            asyncio.create_task(self._cleanup_old_traces())
            
            self._initialized = True
            self.logger.info("FF Trace Logger component initialized with FF backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Trace Logger component: {e}")
            return False
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message and generate trace logs using FF backend.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: FF message DTO
            context: Optional context information
            
        Returns:
            Processing results with trace information
        """
        if not self._initialized:
            raise RuntimeError("FF Trace Logger component not initialized")
        
        try:
            start_time = datetime.now()
            trace_id = f"trace_{int(start_time.timestamp() * 1000)}"
            
            # Log message received
            await self._log_trace(
                session_id=session_id,
                user_id=user_id,
                event_type=FFTraceEvent.MESSAGE_RECEIVED,
                level=FFTraceLevel.INFO,
                component="ff_trace_logger",
                message="Message received for tracing",
                data={
                    "message_id": message.message_id,
                    "message_role": message.role,
                    "message_length": len(message.content),
                    "attachments": len(message.attachments) if message.attachments else 0
                },
                trace_id=trace_id
            )
            
            # Process tracing based on context
            trace_config = context.get("trace_config", {}) if context else {}
            
            # Perform conversation analysis if enabled
            analysis_results = {}
            if self.config.enable_conversation_analysis:
                analysis_results = await self._analyze_conversation(session_id, user_id, message)
            
            # Perform performance analysis if enabled
            performance_results = {}
            if self.config.enable_performance_analysis:
                performance_results = await self._analyze_performance(session_id)
            
            # Calculate processing time
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Log message processed
            await self._log_trace(
                session_id=session_id,
                user_id=user_id,
                event_type=FFTraceEvent.MESSAGE_PROCESSED,
                level=FFTraceLevel.INFO,
                component="ff_trace_logger",
                message="Message processing completed",
                data={
                    "processing_duration_ms": duration_ms,
                    "analysis_performed": bool(analysis_results),
                    "performance_analyzed": bool(performance_results)
                },
                duration_ms=duration_ms,
                trace_id=trace_id
            )
            
            return {
                "success": True,
                "component": "ff_trace_logger",
                "processor": "ff_logging_backend",
                "trace_id": trace_id,
                "processing_duration_ms": duration_ms,
                "traces_logged": len(self.session_traces.get(session_id, [])),
                "analysis_results": analysis_results,
                "performance_results": performance_results,
                "metadata": {
                    "session_id": session_id,
                    "trace_level": self.config.log_level,
                    "tracing_enabled": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing trace logging: {e}")
            
            # Log the error
            await self._log_trace(
                session_id=session_id,
                user_id=user_id,
                event_type=FFTraceEvent.ERROR_OCCURRED,
                level=FFTraceLevel.ERROR,
                component="ff_trace_logger",
                message=f"Error in trace processing: {str(e)}",
                data={"error_type": type(e).__name__}
            )
            
            return {
                "success": False,
                "error": str(e),
                "component": "ff_trace_logger"
            }
    
    async def _log_trace(self,
                         session_id: str,
                         user_id: str,
                         event_type: FFTraceEvent,
                         level: FFTraceLevel,
                         component: str,
                         message: str,
                         data: Optional[Dict[str, Any]] = None,
                         duration_ms: Optional[float] = None,
                         trace_id: Optional[str] = None) -> None:
        """Log a trace entry using FF logging system"""
        try:
            # Create trace entry
            trace_entry = FFTraceEntry(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                user_id=user_id,
                event_type=event_type,
                level=level,
                component=component,
                message=message,
                data=data or {},
                duration_ms=duration_ms,
                trace_id=trace_id
            )
            
            # Add to local storage
            self.trace_entries.append(trace_entry)
            
            if session_id not in self.session_traces:
                self.session_traces[session_id] = []
            self.session_traces[session_id].append(trace_entry)
            
            # Log using FF logging system
            log_message = f"[{event_type.value}] {message}"
            log_data = {
                "session_id": session_id,
                "user_id": user_id,
                "component": component,
                "data": data,
                "duration_ms": duration_ms,
                "trace_id": trace_id
            }
            
            if level == FFTraceLevel.DEBUG:
                self.trace_logger.debug(log_message, extra=log_data)
            elif level == FFTraceLevel.INFO:
                self.trace_logger.info(log_message, extra=log_data)
            elif level == FFTraceLevel.WARNING:
                self.trace_logger.warning(log_message, extra=log_data)
            elif level == FFTraceLevel.ERROR:
                self.trace_logger.error(log_message, extra=log_data)
            elif level == FFTraceLevel.CRITICAL:
                self.trace_logger.critical(log_message, extra=log_data)
            
            # Store in FF storage if enabled
            if self.config.store_traces_in_ff and self.ff_storage:
                await self._store_trace_in_ff(trace_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging trace: {e}")
    
    async def _store_trace_in_ff(self, trace_entry: FFTraceEntry) -> None:
        """Store trace entry in FF storage"""
        try:
            # Create FF message for trace entry
            trace_content = f"[TRACE] {trace_entry.event_type.value}: {trace_entry.message}"
            
            trace_message = FFMessageDTO(
                role="system",
                content=trace_content,
                metadata={
                    "component": "ff_trace_logger",
                    "trace_entry": {
                        "timestamp": trace_entry.timestamp,
                        "event_type": trace_entry.event_type.value,
                        "level": trace_entry.level.value,
                        "component": trace_entry.component,
                        "message": trace_entry.message,
                        "data": trace_entry.data,
                        "duration_ms": trace_entry.duration_ms,
                        "trace_id": trace_entry.trace_id
                    }
                }
            )
            
            await self.ff_storage.add_message(
                trace_entry.user_id,
                trace_entry.session_id,
                trace_message
            )
            
        except Exception as e:
            self.logger.error(f"Error storing trace in FF storage: {e}")
    
    async def _analyze_conversation(self, 
                                    session_id: str,
                                    user_id: str,
                                    message: FFMessageDTO) -> Dict[str, Any]:
        """Analyze conversation patterns and metrics"""
        try:
            session_traces = self.session_traces.get(session_id, [])
            
            if not session_traces:
                return {"message": "No trace history for analysis"}
            
            # Calculate conversation metrics
            total_messages = len([t for t in session_traces if t.event_type == FFTraceEvent.MESSAGE_RECEIVED])
            total_errors = len([t for t in session_traces if t.event_type == FFTraceEvent.ERROR_OCCURRED])
            
            # Calculate average response time
            processing_times = [
                t.duration_ms for t in session_traces
                if t.duration_ms is not None and t.event_type == FFTraceEvent.MESSAGE_PROCESSED
            ]
            avg_response_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Analyze component usage
            component_usage = {}
            for trace in session_traces:
                comp = trace.component
                if comp not in component_usage:
                    component_usage[comp] = 0
                component_usage[comp] += 1
            
            analysis = {
                "session_id": session_id,
                "total_messages": total_messages,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_messages, 1),
                "average_response_time_ms": avg_response_time,
                "component_usage": component_usage,
                "conversation_length": len(session_traces),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Store conversation stats
            self.conversation_stats[session_id] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation: {e}")
            return {"error": str(e)}
    
    async def _analyze_performance(self, session_id: str) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            session_traces = self.session_traces.get(session_id, [])
            
            if not session_traces:
                return {"message": "No performance data for analysis"}
            
            # Filter performance-related traces
            perf_traces = [
                t for t in session_traces
                if t.duration_ms is not None and t.event_type in [
                    FFTraceEvent.MESSAGE_PROCESSED,
                    FFTraceEvent.COMPONENT_RESPONSE
                ]
            ]
            
            if not perf_traces:
                return {"message": "No performance traces available"}
            
            # Calculate performance metrics
            durations = [t.duration_ms for t in perf_traces]
            
            performance_analysis = {
                "session_id": session_id,
                "total_operations": len(perf_traces),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "total_duration_ms": sum(durations),
                "performance_trend": "stable",  # Simplified analysis
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Store performance metrics
            if session_id not in self.performance_metrics:
                self.performance_metrics[session_id] = []
            self.performance_metrics[session_id].append(performance_analysis)
            
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    async def get_session_traces(self, 
                                 session_id: str,
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trace entries for a specific session"""
        try:
            session_traces = self.session_traces.get(session_id, [])
            
            if limit:
                session_traces = session_traces[-limit:]
            
            return [
                {
                    "timestamp": trace.timestamp,
                    "event_type": trace.event_type.value,
                    "level": trace.level.value,
                    "component": trace.component,
                    "message": trace.message,
                    "data": trace.data,
                    "duration_ms": trace.duration_ms,
                    "trace_id": trace.trace_id
                }
                for trace in session_traces
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting session traces: {e}")
            return []
    
    async def export_traces(self, 
                            format: str = "json",
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export trace data in specified format"""
        try:
            # Select traces to export
            if session_id:
                traces = self.session_traces.get(session_id, [])
            else:
                traces = self.trace_entries
            
            if format.lower() == "json":
                exported_data = [
                    {
                        "timestamp": trace.timestamp,
                        "session_id": trace.session_id,
                        "user_id": trace.user_id,
                        "event_type": trace.event_type.value,
                        "level": trace.level.value,
                        "component": trace.component,
                        "message": trace.message,
                        "data": trace.data,
                        "duration_ms": trace.duration_ms,
                        "trace_id": trace.trace_id
                    }
                    for trace in traces
                ]
                
                return {
                    "format": "json",
                    "count": len(exported_data),
                    "data": exported_data,
                    "exported_at": datetime.now().isoformat()
                }
            
            elif format.lower() == "csv":
                # Simple CSV export
                csv_lines = ["timestamp,session_id,user_id,event_type,level,component,message,duration_ms"]
                for trace in traces:
                    csv_lines.append(
                        f"{trace.timestamp},{trace.session_id},{trace.user_id},"
                        f"{trace.event_type.value},{trace.level.value},{trace.component},"
                        f'"{trace.message}",{trace.duration_ms or ""}'
                    )
                
                return {
                    "format": "csv",
                    "count": len(traces),
                    "data": "\n".join(csv_lines),
                    "exported_at": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"Unsupported export format: {format}"}
                
        except Exception as e:
            self.logger.error(f"Error exporting traces: {e}")
            return {"error": str(e)}
    
    async def _cleanup_old_traces(self) -> None:
        """Background task to cleanup old trace entries"""
        while self._initialized:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(days=self.config.trace_retention_days)
                
                # Remove old traces
                old_count = len(self.trace_entries)
                self.trace_entries = [
                    trace for trace in self.trace_entries
                    if datetime.fromisoformat(trace.timestamp) > cutoff_time
                ]
                new_count = len(self.trace_entries)
                
                if old_count != new_count:
                    self.logger.info(f"Cleaned up {old_count - new_count} old trace entries")
                
                # Limit total trace entries
                if len(self.trace_entries) > self.config.max_trace_entries:
                    excess = len(self.trace_entries) - self.config.max_trace_entries
                    self.trace_entries = self.trace_entries[excess:]
                    self.logger.info(f"Trimmed {excess} excess trace entries")
                
                # Wait before next cleanup
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in trace cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.logger.info("Cleaning up FF Trace Logger component")
        self._initialized = False
        self.trace_entries.clear()
        self.session_traces.clear()
        self.performance_metrics.clear()
        self.conversation_stats.clear()
```

### Testing and Validation

#### 4. Phase 3 Integration Tests

**File**: `tests/test_ff_chat_phase3.py`

```python
"""
Phase 3 FF Advanced Features Tests

Tests the advanced FF chat components and their integration with existing FF infrastructure.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Import existing FF infrastructure
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

# Import Phase 3 FF chat components
from ff_chat_components.ff_tools_component import FFToolsComponent, FFToolsConfigDTO
from ff_chat_components.ff_topic_router_component import FFTopicRouterComponent, FFTopicRouterConfigDTO
from ff_chat_components.ff_trace_logger_component import FFTraceLoggerComponent, FFTraceLoggerConfigDTO

class TestFFAdvancedComponents:
    """Test Phase 3 advanced FF chat components"""
    
    @pytest.fixture
    async def ff_dependencies(self):
        """Create FF backend services for testing"""
        ff_config = load_config()
        ff_storage = FFStorageManager(ff_config)
        await ff_storage.initialize()
        
        dependencies = {
            "ff_storage": ff_storage,
            "ff_search": ff_storage.search_engine,
            "ff_vector": ff_storage.vector_storage,
            "ff_document_processor": ff_storage.document_processor if hasattr(ff_storage, 'document_processor') else None
        }
        
        return dependencies
    
    @pytest.mark.asyncio
    async def test_ff_tools_component(self, ff_dependencies):
        """Test FF tools component with FF document processing backend"""
        
        # Create and initialize component
        config = FFToolsConfigDTO()
        component = FFToolsComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test available tools
        tools = await component.list_available_tools()
        assert len(tools) > 0
        assert any(tool["name"] == "web_search" for tool in tools)
        assert any(tool["name"] == "document_analysis" for tool in tools)
        
        # Test tool execution through message processing
        message = FFMessageDTO(
            role=MessageRole.USER.value,
            content="Search for information about Python programming"
        )
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message
        )
        
        assert result["success"] == True
        assert result["component"] == "ff_tools"
        assert result["tools_executed"] > 0
        assert len(result["tool_results"]) > 0
        
        # Verify tool result structure
        tool_result = result["tool_results"][0]
        assert "tool" in tool_result
        assert "success" in tool_result
        assert "result" in tool_result
        
        # Test component info
        info = component.component_info
        assert info["name"] == "ff_tools"
        assert "external_integration" in info["capabilities"]
        assert "personal_assistant" in info["use_cases"]
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_topic_router_component(self, ff_dependencies):
        """Test FF topic router component with FF search backend"""
        
        # Create and initialize component
        config = FFTopicRouterConfigDTO()
        component = FFTopicRouterComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test topic routing for technical content
        technical_message = FFMessageDTO(
            role=MessageRole.USER.value,
            content="I need help with programming and software development algorithms"
        )
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=technical_message
        )
        
        assert result["success"] == True
        assert result["component"] == "ff_topic_router"
        assert len(result["detected_topics"]) > 0
        assert result["routing_decision"] is not None
        
        # Check that technical topic was detected
        detected_topics = result["detected_topics"]
        topic_names = [topic["topic"] for topic in detected_topics]
        assert "technical" in topic_names
        
        # Check routing decision
        routing = result["routing_decision"]
        assert routing["target_agent"] is not None
        assert routing["confidence"] > 0
        
        # Test creative content routing
        creative_message = FFMessageDTO(
            role=MessageRole.USER.value,
            content="I want to write a creative story and need design inspiration"
        )
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=creative_message
        )
        
        assert result["success"] == True
        detected_topics = result["detected_topics"]
        topic_names = [topic["topic"] for topic in detected_topics]
        assert "creative" in topic_names
        
        # Test component info
        info = component.component_info
        assert info["name"] == "ff_topic_router"
        assert "topic_detection" in info["capabilities"]
        assert "topic_delegation" in info["use_cases"]
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_ff_trace_logger_component(self, ff_dependencies):
        """Test FF trace logger component with FF logging backend"""
        
        # Create and initialize component
        config = FFTraceLoggerConfigDTO()
        component = FFTraceLoggerComponent(config)
        
        success = await component.initialize(ff_dependencies)
        assert success
        
        # Test trace logging
        message = FFMessageDTO(
            role=MessageRole.USER.value,
            content="This is a test message for trace logging"
        )
        
        result = await component.process_message(
            session_id="test_session",
            user_id="test_user",
            message=message,
            context={"trace_config": {"enable_detailed_tracing": True}}
        )
        
        assert result["success"] == True
        assert result["component"] == "ff_trace_logger"
        assert result["trace_id"] is not None
        assert result["processing_duration_ms"] > 0
        assert result["traces_logged"] > 0
        
        # Test trace retrieval
        session_traces = await component.get_session_traces("test_session")
        assert len(session_traces) > 0
        
        # Verify trace structure
        trace = session_traces[0]
        assert "timestamp" in trace
        assert "event_type" in trace
        assert "level" in trace
        assert "component" in trace
        assert "message" in trace
        
        # Test trace export
        export_result = await component.export_traces(format="json", session_id="test_session")
        assert export_result["format"] == "json"
        assert export_result["count"] > 0
        assert "data" in export_result
        
        # Test CSV export
        csv_export = await component.export_traces(format="csv", session_id="test_session")
        assert csv_export["format"] == "csv"
        assert "timestamp,session_id" in csv_export["data"]
        
        # Test component info
        info = component.component_info
        assert info["name"] == "ff_trace_logger"
        assert "conversation_logging" in info["capabilities"]
        assert "ai_debate" in info["use_cases"]
        
        # Cleanup
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_advanced_components_integration(self, ff_dependencies):
        """Test integration between advanced components"""
        
        # Initialize all advanced components
        tools_component = FFToolsComponent(FFToolsConfigDTO())
        router_component = FFTopicRouterComponent(FFTopicRouterConfigDTO())
        logger_component = FFTraceLoggerComponent(FFTraceLoggerConfigDTO())
        
        await tools_component.initialize(ff_dependencies)
        await router_component.initialize(ff_dependencies)
        await logger_component.initialize(ff_dependencies)
        
        try:
            # Test workflow: Route -> Log -> Execute Tools
            message = FFMessageDTO(
                role=MessageRole.USER.value,
                content="I need technical help with API calls and want to search for documentation"
            )
            
            # Step 1: Route the topic
            routing_result = await router_component.process_message(
                session_id="integration_test",
                user_id="test_user",
                message=message
            )
            
            assert routing_result["success"]
            target_agent = routing_result["routing_decision"]["target_agent"]
            
            # Step 2: Log the interaction
            trace_result = await logger_component.process_message(
                session_id="integration_test",
                user_id="test_user",
                message=message,
                context={"routed_to": target_agent}
            )
            
            assert trace_result["success"]
            
            # Step 3: Execute tools based on routing
            tools_result = await tools_component.process_message(
                session_id="integration_test",
                user_id="test_user",
                message=message,
                context={"target_agent": target_agent}
            )
            
            assert tools_result["success"]
            assert tools_result["tools_executed"] > 0
            
            # Verify the integration worked
            session_traces = await logger_component.get_session_traces("integration_test")
            assert len(session_traces) > 0
            
        finally:
            # Cleanup all components
            await tools_component.cleanup()
            await router_component.cleanup()
            await logger_component.cleanup()
    
    @pytest.mark.asyncio
    async def test_phase3_with_chat_application(self, ff_dependencies):
        """Test Phase 3 components integration with FF Chat Application"""
        
        from ff_chat_application import FFChatApplication
        from ff_chat_components.ff_component_registry import get_ff_chat_component_registry
        
        # Register Phase 3 components
        registry = get_ff_chat_component_registry()
        
        # Add Phase 3 components to registry
        registry.register_component(
            name="ff_tools",
            component_class=FFToolsComponent,
            config_class=FFToolsConfigDTO,
            config=FFToolsConfigDTO(),
            dependencies=["ff_storage", "ff_document_processor"]
        )
        
        registry.register_component(
            name="ff_topic_router",
            component_class=FFTopicRouterComponent,
            config_class=FFTopicRouterConfigDTO,
            config=FFTopicRouterConfigDTO(),
            dependencies=["ff_storage", "ff_search", "ff_vector"]
        )
        
        registry.register_component(
            name="ff_trace_logger",
            component_class=FFTraceLoggerComponent,
            config_class=FFTraceLoggerConfigDTO,
            config=FFTraceLoggerConfigDTO(),
            dependencies=["ff_storage"]
        )
        
        # Create chat application
        chat_app = FFChatApplication()
        await chat_app.initialize()
        
        try:
            # Test use case that requires advanced components
            session_id = await chat_app.create_chat_session(
                user_id="test_user",
                use_case="personal_assistant"  # Should use tools component
            )
            
            result = await chat_app.process_message(
                session_id,
                "I need help with API documentation and want to analyze some files"
            )
            
            assert result["success"] == True
            
        finally:
            await chat_app.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Validation Checklist

#### Phase 3 Completion Requirements

- ✅ **FF Advanced Components**
  - [ ] FF Tools component integrates external systems using FF document processing
  - [ ] FF Topic Router component routes intelligently using FF search/vector
  - [ ] FF Trace Logger component provides debugging using FF logging system
  - [ ] Components integrate with existing FF Chat Application

- ✅ **FF Backend Integration**
  - [ ] Tools component uses FF document processing for file operations
  - [ ] Router component uses FF search and vector storage for topic detection
  - [ ] Logger component uses FF logging system and storage for persistence
  - [ ] All components follow FF architectural patterns

- ✅ **Use Case Coverage**
  - [ ] 21/22 use cases supported (95% coverage)
  - [ ] Professional use cases handled by FF Tools component
  - [ ] Intelligent routing handled by FF Topic Router component
  - [ ] Debugging use cases handled by FF Trace Logger component

- ✅ **Testing Coverage**
  - [ ] Unit tests for all Phase 3 FF chat components
  - [ ] Integration tests with existing FF infrastructure
  - [ ] End-to-end workflow tests combining multiple components
  - [ ] Integration tests with FF Chat Application

### Success Metrics

#### Functional Requirements
- Can execute external tools safely using FF document processing backend
- Can route topics intelligently using FF search and vector capabilities
- Can provide advanced debugging using FF logging and storage systems
- Can integrate advanced features seamlessly with existing components

#### Non-Functional Requirements
- Zero breaking changes to existing FF system
- Performance within 15% of Phase 2 baseline
- Memory usage increase less than 40%
- 100% test coverage for new Phase 3 components

#### Integration Requirements
- Seamless integration with Phase 1-2 FF Chat infrastructure
- Compatible with existing FF managers and protocols
- Uses existing FF configuration patterns consistently
- Maintains existing FF security and error handling standards

---

**Phase 3 delivers advanced professional capabilities using existing FF infrastructure as backend services, achieving 95% use case coverage while maintaining complete compatibility with the established FF architecture.**