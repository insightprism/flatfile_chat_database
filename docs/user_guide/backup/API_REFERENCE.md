# API Reference

## Core APIs

### Message Bus

The message bus provides event-driven communication between modules.

#### Functions

##### `emit(event_type: str, data: Dict[str, Any]) -> None`
Emit an event to all registered handlers.

**Parameters:**
- `event_type` (str): The type of event to emit
- `data` (Dict[str, Any]): Event data payload

**Example:**
```python
await message_bus.emit("chat.message.received", {
    "session_id": "sess_123",
    "message": {
        "content": "Hello",
        "role": "user"
    }
})
```

##### `on(event_type: str) -> Callable`
Decorator to register an event handler.

**Parameters:**
- `event_type` (str): The event type to listen for

**Example:**
```python
@message_bus.on("chat.message.received")
async def handle_message(data: Dict[str, Any]):
    print(f"Received message: {data['message']['content']}")
```

### Module Context

The module context provides access to system resources and other modules.

#### Class: `ModuleContext`

##### Methods

###### `async get_module(name: str) -> Optional[Module]`
Retrieve a loaded module by name.

**Parameters:**
- `name` (str): Module name

**Returns:**
- Module instance or None if not loaded

**Example:**
```python
memory_module = await context.get_module("memory")
if memory_module:
    await memory_module.store("key", "value")
```

###### `async emit(event: str, data: Dict) -> None`
Emit an event through the message bus.

**Parameters:**
- `event` (str): Event type
- `data` (Dict): Event data

###### `get_config(key: str, default: Any = None) -> Any`
Retrieve configuration value.

**Parameters:**
- `key` (str): Configuration key (supports dot notation)
- `default` (Any): Default value if key not found

**Example:**
```python
max_tokens = context.get_config("text_chat.max_tokens", 4096)
```

###### `async get_storage() -> StorageBackend`
Get the configured storage backend.

**Returns:**
- Storage backend instance

## Module APIs

### 1. Text Chat Module

#### Functions

##### `async send_message(session_id: str, content: str, role: str = "user", metadata: Optional[Dict] = None, context: ModuleContext = None) -> Message`
Send a message in a chat session.

**Parameters:**
- `session_id` (str): The session identifier
- `content` (str): Message content
- `role` (str): Message role (user/assistant/system)
- `metadata` (Optional[Dict]): Additional metadata
- `context` (ModuleContext): Module context

**Returns:**
- `Message`: The created message object

**Example:**
```python
message = await send_message(
    session_id="sess_123",
    content="What's the weather like?",
    role="user",
    metadata={"client": "mobile"}
)
```

##### `async create_session(user_id: str, title: Optional[str] = None, metadata: Optional[Dict] = None, context: ModuleContext = None) -> Session`
Create a new chat session.

**Parameters:**
- `user_id` (str): User identifier
- `title` (Optional[str]): Session title
- `metadata` (Optional[Dict]): Session metadata
- `context` (ModuleContext): Module context

**Returns:**
- `Session`: The created session object

##### `async get_messages(session_id: str, limit: int = 100, offset: int = 0, context: ModuleContext = None) -> List[Message]`
Retrieve messages from a session.

**Parameters:**
- `session_id` (str): Session identifier
- `limit` (int): Maximum messages to retrieve
- `offset` (int): Number of messages to skip
- `context` (ModuleContext): Module context

**Returns:**
- `List[Message]`: List of messages

##### `async stream_response(session_id: str, prompt: str, context: ModuleContext = None) -> AsyncIterator[str]`
Stream AI response chunks.

**Parameters:**
- `session_id` (str): Session identifier
- `prompt` (str): User prompt
- `context` (ModuleContext): Module context

**Yields:**
- `str`: Response chunks

**Example:**
```python
async for chunk in stream_response("sess_123", "Tell me a story"):
    print(chunk, end="", flush=True)
```

### 2. Memory Module

#### Functions

##### `async store_memory(key: str, content: Any, memory_type: str = "short_term", ttl: Optional[int] = None, metadata: Optional[Dict] = None, context: ModuleContext = None) -> bool`
Store information in memory.

**Parameters:**
- `key` (str): Memory key
- `content` (Any): Content to store
- `memory_type` (str): Type of memory (short_term/long_term/working)
- `ttl` (Optional[int]): Time to live in seconds
- `metadata` (Optional[Dict]): Additional metadata
- `context` (ModuleContext): Module context

**Returns:**
- `bool`: Success status

##### `async recall_memory(key: str, memory_type: str = "short_term", context: ModuleContext = None) -> Optional[Any]`
Retrieve specific memory.

**Parameters:**
- `key` (str): Memory key
- `memory_type` (str): Type of memory
- `context` (ModuleContext): Module context

**Returns:**
- Content or None if not found

##### `async search_memories(query: str, memory_type: str = "all", limit: int = 10, context: ModuleContext = None) -> List[MemoryItem]`
Search through memories.

**Parameters:**
- `query` (str): Search query
- `memory_type` (str): Filter by memory type
- `limit` (int): Maximum results
- `context` (ModuleContext): Module context

**Returns:**
- `List[MemoryItem]`: Matching memories

##### `async get_context_window(session_id: str, max_tokens: int = 4000, context: ModuleContext = None) -> List[Dict]`
Get relevant context for session.

**Parameters:**
- `session_id` (str): Session identifier
- `max_tokens` (int): Maximum context size in tokens
- `context` (ModuleContext): Module context

**Returns:**
- `List[Dict]`: Context items

### 3. RAG Module

#### Functions

##### `async add_document(document_path: str, metadata: Optional[Dict] = None, chunking_strategy: str = "semantic", context: ModuleContext = None) -> DocumentInfo`
Add document to knowledge base.

**Parameters:**
- `document_path` (str): Path to document
- `metadata` (Optional[Dict]): Document metadata
- `chunking_strategy` (str): How to split document
- `context` (ModuleContext): Module context

**Returns:**
- `DocumentInfo`: Document information

##### `async search_knowledge(query: str, top_k: int = 5, filters: Optional[Dict] = None, context: ModuleContext = None) -> List[SearchResult]`
Search knowledge base.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results
- `filters` (Optional[Dict]): Search filters
- `context` (ModuleContext): Module context

**Returns:**
- `List[SearchResult]`: Search results

##### `async get_relevant_context(query: str, max_tokens: int = 2000, context: ModuleContext = None) -> RetrievalContext`
Get relevant context for query.

**Parameters:**
- `query` (str): Query text
- `max_tokens` (int): Maximum context size
- `context` (ModuleContext): Module context

**Returns:**
- `RetrievalContext`: Retrieved context

### 4. Multi-Agent Module

#### Functions

##### `async register_agent(agent_id: str, agent_type: str, capabilities: List[str], metadata: Optional[Dict] = None, context: ModuleContext = None) -> AgentInfo`
Register a new agent.

**Parameters:**
- `agent_id` (str): Unique agent identifier
- `agent_type` (str): Type of agent
- `capabilities` (List[str]): Agent capabilities
- `metadata` (Optional[Dict]): Agent metadata
- `context` (ModuleContext): Module context

**Returns:**
- `AgentInfo`: Agent information

##### `async start_conversation(topic: str, agent_ids: List[str], max_turns: Optional[int] = None, context: ModuleContext = None) -> ConversationInfo`
Start multi-agent conversation.

**Parameters:**
- `topic` (str): Conversation topic
- `agent_ids` (List[str]): Participating agents
- `max_turns` (Optional[int]): Turn limit
- `context` (ModuleContext): Module context

**Returns:**
- `ConversationInfo`: Conversation details

##### `async send_to_agents(message: str, agent_ids: List[str], wait_for_all: bool = True, context: ModuleContext = None) -> List[AgentResponse]`
Send message to multiple agents.

**Parameters:**
- `message` (str): Message content
- `agent_ids` (List[str]): Target agents
- `wait_for_all` (bool): Wait for all responses
- `context` (ModuleContext): Module context

**Returns:**
- `List[AgentResponse]`: Agent responses

### 5. Tool Use Module

#### Functions

##### `async register_tool(tool_name: str, endpoint: str, input_schema: Dict, output_schema: Dict, metadata: Optional[Dict] = None, context: ModuleContext = None) -> ToolInfo`
Register an external tool.

**Parameters:**
- `tool_name` (str): Tool name
- `endpoint` (str): Tool endpoint URL
- `input_schema` (Dict): Input JSON schema
- `output_schema` (Dict): Output JSON schema
- `metadata` (Optional[Dict]): Tool metadata
- `context` (ModuleContext): Module context

**Returns:**
- `ToolInfo`: Tool information

##### `async execute_tool(tool_name: str, parameters: Dict, timeout: Optional[int] = None, context: ModuleContext = None) -> ToolResult`
Execute a registered tool.

**Parameters:**
- `tool_name` (str): Tool to execute
- `parameters` (Dict): Tool parameters
- `timeout` (Optional[int]): Execution timeout
- `context` (ModuleContext): Module context

**Returns:**
- `ToolResult`: Execution result

**Example:**
```python
result = await execute_tool(
    "weather_api",
    {"location": "San Francisco", "units": "metric"},
    timeout=5
)
```

### 6. Persona Module

#### Functions

##### `async create_persona(name: str, traits: Dict[str, Any], backstory: Optional[str] = None, examples: Optional[List[Dict]] = None, context: ModuleContext = None) -> PersonaInfo`
Create a new persona.

**Parameters:**
- `name` (str): Persona name
- `traits` (Dict[str, Any]): Personality traits
- `backstory` (Optional[str]): Character backstory
- `examples` (Optional[List[Dict]]): Example interactions
- `context` (ModuleContext): Module context

**Returns:**
- `PersonaInfo`: Persona information

##### `async activate_persona(persona_id: str, session_id: Optional[str] = None, context: ModuleContext = None) -> bool`
Activate a persona for use.

**Parameters:**
- `persona_id` (str): Persona to activate
- `session_id` (Optional[str]): Session scope
- `context` (ModuleContext): Module context

**Returns:**
- `bool`: Success status

##### `async inject_personality(message: str, persona_id: str, strength: float = 1.0, context: ModuleContext = None) -> str`
Inject persona traits into response.

**Parameters:**
- `message` (str): Original message
- `persona_id` (str): Persona to apply
- `strength` (float): Injection strength (0-1)
- `context` (ModuleContext): Module context

**Returns:**
- `str`: Modified message

### 7. Multimodal Module

#### Functions

##### `async process_image(image_path: str, operations: List[str], context: ModuleContext = None) -> ImageResult`
Process image file.

**Parameters:**
- `image_path` (str): Path to image
- `operations` (List[str]): Operations to perform
- `context` (ModuleContext): Module context

**Returns:**
- `ImageResult`: Processing result

##### `async transcribe_audio(audio_path: str, language: Optional[str] = None, context: ModuleContext = None) -> TranscriptionResult`
Transcribe audio to text.

**Parameters:**
- `audio_path` (str): Path to audio file
- `language` (Optional[str]): Audio language
- `context` (ModuleContext): Module context

**Returns:**
- `TranscriptionResult`: Transcription

##### `async analyze_video(video_path: str, analysis_types: List[str], context: ModuleContext = None) -> VideoAnalysis`
Analyze video content.

**Parameters:**
- `video_path` (str): Path to video
- `analysis_types` (List[str]): Types of analysis
- `context` (ModuleContext): Module context

**Returns:**
- `VideoAnalysis`: Analysis results

##### `async generate_image(prompt: str, style: Optional[str] = None, size: str = "1024x1024", context: ModuleContext = None) -> GeneratedImage`
Generate image from text.

**Parameters:**
- `prompt` (str): Image description
- `style` (Optional[str]): Art style
- `size` (str): Image dimensions
- `context` (ModuleContext): Module context

**Returns:**
- `GeneratedImage`: Generated image

### 8. Topic Router Module

#### Functions

##### `async classify_content(content: str, context_history: Optional[List[str]] = None, context: ModuleContext = None) -> Classification`
Classify content into topics.

**Parameters:**
- `content` (str): Content to classify
- `context_history` (Optional[List[str]]): Previous context
- `context` (ModuleContext): Module context

**Returns:**
- `Classification`: Topic classification

##### `async route_to_specialist(content: str, classification: Classification, available_specialists: List[str], context: ModuleContext = None) -> RoutingDecision`
Route to appropriate specialist.

**Parameters:**
- `content` (str): Content to route
- `classification` (Classification): Topic classification
- `available_specialists` (List[str]): Available specialists
- `context` (ModuleContext): Module context

**Returns:**
- `RoutingDecision`: Routing decision

### 9. Trace Logger Module

#### Functions

##### `async start_trace(operation: str, metadata: Optional[Dict] = None, context: ModuleContext = None) -> TraceSpan`
Start a new trace span.

**Parameters:**
- `operation` (str): Operation name
- `metadata` (Optional[Dict]): Span metadata
- `context` (ModuleContext): Module context

**Returns:**
- `TraceSpan`: Trace span

##### `async log_event(level: str, message: str, data: Optional[Dict] = None, context: ModuleContext = None) -> bool`
Log an event.

**Parameters:**
- `level` (str): Log level (DEBUG/INFO/WARNING/ERROR)
- `message` (str): Log message
- `data` (Optional[Dict]): Additional data
- `context` (ModuleContext): Module context

**Returns:**
- `bool`: Success status

##### `async create_audit_entry(action: str, user_id: str, resource: str, details: Dict, context: ModuleContext = None) -> AuditEntry`
Create audit trail entry.

**Parameters:**
- `action` (str): Action performed
- `user_id` (str): User who performed action
- `resource` (str): Resource affected
- `details` (Dict): Action details
- `context` (ModuleContext): Module context

**Returns:**
- `AuditEntry`: Audit entry

## Event Specifications

### Event Naming Convention
Events follow the pattern: `module.entity.action`

Examples:
- `chat.message.send`
- `memory.store.complete`
- `agent.response.ready`

### Common Events

#### Chat Events
- `chat.message.send` - User sends message
- `chat.message.received` - Message processed
- `chat.response.ready` - AI response generated
- `chat.session.created` - New session created
- `chat.session.ended` - Session ended
- `chat.error` - Chat error occurred

#### Memory Events
- `memory.store` - Store memory request
- `memory.stored` - Memory stored successfully
- `memory.recall` - Recall memory request
- `memory.retrieved` - Memory retrieved
- `memory.consolidated` - Memory consolidation complete

#### RAG Events
- `rag.document.add` - Add document request
- `rag.document.indexed` - Document indexed
- `rag.query` - Search request
- `rag.results.ready` - Search results available

#### Multi-Agent Events
- `agent.register` - Register agent
- `agent.message` - Inter-agent message
- `agent.response` - Agent response
- `conversation.start` - Start conversation
- `conversation.complete` - Conversation ended

### Event Data Structures

#### Base Event Structure
```python
{
    "event_id": "evt_123456",
    "timestamp": "2024-01-15T10:30:00Z",
    "source_module": "text_chat",
    "data": {
        # Event-specific data
    }
}
```

#### Message Event Data
```python
{
    "session_id": "sess_123",
    "message": {
        "id": "msg_456",
        "content": "Hello",
        "role": "user",
        "metadata": {}
    },
    "user_id": "user_789"
}
```

## Storage Backend Interface

All storage backends must implement this interface:

### Interface: `StorageBackend`

#### Methods

##### `async read(key: str) -> Optional[bytes]`
Read data by key.

**Parameters:**
- `key` (str): Storage key

**Returns:**
- Data as bytes or None

##### `async write(key: str, data: bytes) -> bool`
Write data with key.

**Parameters:**
- `key` (str): Storage key
- `data` (bytes): Data to store

**Returns:**
- Success status

##### `async delete(key: str) -> bool`
Delete data by key.

**Parameters:**
- `key` (str): Storage key

**Returns:**
- Success status

##### `async list(prefix: str) -> List[str]`
List keys with prefix.

**Parameters:**
- `prefix` (str): Key prefix

**Returns:**
- List of matching keys

##### `async exists(key: str) -> bool`
Check if key exists.

**Parameters:**
- `key` (str): Storage key

**Returns:**
- Existence status

## Error Handling

### Error Types

#### `ModuleNotFoundError`
Raised when a requested module is not available.

```python
try:
    module = await context.get_module("unknown_module")
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
```

#### `ConfigurationError`
Raised when module configuration is invalid.

```python
try:
    await module.initialize(invalid_config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### `StorageError`
Raised when storage operations fail.

```python
try:
    data = await storage.read("invalid/key")
except StorageError as e:
    print(f"Storage error: {e}")
```

### Error Response Format
```python
{
    "error": {
        "type": "ValidationError",
        "message": "Invalid message format",
        "details": {
            "field": "content",
            "reason": "exceeds maximum length"
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Rate Limiting

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318200
```

### Rate Limit Response
```python
{
    "error": {
        "type": "RateLimitError",
        "message": "Rate limit exceeded",
        "retry_after": 60
    }
}
```

## Versioning

### API Version Header
```
X-API-Version: 1.0.0
```

### Module Version Info
```python
module_info = {
    "name": "text_chat",
    "version": "1.0.0",
    "api_version": "1.0",
    "min_compatible_version": "0.9.0"
}
```

## Examples

### Complete Chat Flow
```python
# Initialize context
context = ModuleContext()

# Create session
session = await create_session(
    user_id="user_123",
    title="Morning Chat",
    context=context
)

# Send message
message = await send_message(
    session_id=session.id,
    content="What's the weather today?",
    context=context
)

# Add to memory
memory_module = await context.get_module("memory")
await memory_module.store_memory(
    f"weather_query_{session.id}",
    {"query": "weather", "timestamp": datetime.now()},
    memory_type="short_term"
)

# Use RAG for context
rag_module = await context.get_module("rag")
context_data = await rag_module.get_relevant_context(
    "weather today",
    max_tokens=500
)

# Generate response
response = await generate_response(
    prompt=message.content,
    context=context_data,
    session_id=session.id
)
```

### Multi-Agent Conversation
```python
# Register agents
finance_agent = await register_agent(
    agent_id="finance_expert",
    agent_type="specialist",
    capabilities=["financial_analysis", "budgeting"]
)

legal_agent = await register_agent(
    agent_id="legal_expert",
    agent_type="specialist",
    capabilities=["compliance", "contracts"]
)

# Start conversation
conversation = await start_conversation(
    topic="Should we proceed with the acquisition?",
    agent_ids=["finance_expert", "legal_expert"],
    max_turns=10
)

# Get responses
responses = await send_to_agents(
    "What are the risks?",
    agent_ids=["finance_expert", "legal_expert"],
    wait_for_all=True
)

for response in responses:
    print(f"{response.agent_id}: {response.content}")
```