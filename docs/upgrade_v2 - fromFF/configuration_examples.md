# FF Chat System - Configuration Examples and Use Case Definitions

## Overview

This document provides comprehensive configuration examples and use case definitions for the FF Chat System. All configurations extend existing FF patterns while adding chat-specific capabilities.

## 1. Configuration Examples

### Basic Chat Configuration

```yaml
# basic_chat_config.yaml - Minimal FF chat configuration
# Extends existing FF configuration with basic chat capabilities

# Existing FF Storage Configuration (unchanged)
ff_storage_config:
  enabled: true
  storage_base_path: "./ff_chat_data"
  session_id_prefix: "chat_"
  user_id_prefix: "user_"
  enable_file_locking: true
  max_file_size_mb: 100

# Existing FF Search Configuration (unchanged)
ff_search_config:
  enabled: true
  max_results: 50
  relevance_threshold: 0.5

# Existing FF Vector Storage Configuration (unchanged)  
ff_vector_storage_config:
  enabled: true
  dimension: 384
  similarity_metric: "cosine"
  max_vectors: 100000

# New Chat Application Settings
chat_enabled: true
chat_api_enabled: false
chat_websocket_enabled: false

# Chat Session Settings
chat_session:
  max_messages_per_session: 1000
  session_timeout_minutes: 60
  auto_save_interval_seconds: 30
  enable_message_threading: true
  max_concurrent_sessions_per_user: 5

# Chat Component Settings
chat_components:
  text_chat_enabled: true
  text_chat_max_tokens: 4000
  text_chat_temperature: 0.7
  
  memory_enabled: false
  multi_agent_enabled: false
  tools_enabled: false

# Use Case Settings
chat_use_cases:
  default_use_case: "basic_chat"
  use_case_components:
    basic_chat: ["text_chat"]
```

### Memory Chat Configuration

```yaml
# memory_chat_config.yaml - FF chat with memory capabilities
# Uses existing FF vector storage for memory functionality

# Inherit basic configuration and extend
ff_storage_config:
  enabled: true
  storage_base_path: "./ff_memory_chat_data"
  session_id_prefix: "mem_chat_"

ff_vector_storage_config:
  enabled: true
  dimension: 384
  similarity_metric: "cosine"
  max_vectors: 50000
  memory_cleanup_threshold: 0.8

chat_enabled: true

chat_components:
  text_chat_enabled: true
  memory_enabled: true
  memory_max_entries: 5000
  memory_similarity_threshold: 0.75
  memory_context_window: 10

chat_use_cases:
  default_use_case: "memory_chat"
  use_case_components:
    basic_chat: ["text_chat"]
    memory_chat: ["text_chat", "memory"]
    thought_partner: ["text_chat", "memory"]
    story_world_chat: ["text_chat", "memory", "persona"]
```

### Multi-Agent Configuration

```yaml
# multi_agent_config.yaml - FF chat with multi-agent capabilities
# Uses existing FF panel manager for agent coordination

# Existing FF Panel Configuration (leverage existing)
ff_persona_panel_config:
  enabled: true
  max_personas: 5
  context_sharing_enabled: true
  coordination_model: "round_robin"

ff_storage_config:
  enabled: true
  storage_base_path: "./ff_multi_agent_data"
  session_id_prefix: "multi_"

chat_enabled: true

chat_components:
  text_chat_enabled: true
  memory_enabled: true
  multi_agent_enabled: true
  multi_agent_max_agents: 5
  multi_agent_coordination_model: "dynamic"

chat_use_cases:
  default_use_case: "multi_ai_panel"
  use_case_components:
    basic_chat: ["text_chat"]
    multi_ai_panel: ["multi_agent", "memory", "persona"]
    ai_debate: ["multi_agent", "persona", "trace"]
    topic_delegation: ["text_chat", "memory", "search", "multi_agent", "router"]
    ai_game_master: ["text_chat", "multi_agent", "memory"]
    auto_task_agent: ["tools", "multi_agent", "memory"]
```

### Production Configuration

```yaml
# production_config.yaml - Full production FF chat configuration
# Comprehensive setup for all 22 use cases

# Production FF Storage Configuration
ff_storage_config:
  enabled: true
  storage_base_path: "/var/lib/ff_chat/data"
  session_id_prefix: "prod_chat_"
  user_id_prefix: "user_"
  enable_file_locking: true
  max_file_size_mb: 500
  backup_enabled: true
  backup_interval_hours: 6

# Production FF Search Configuration
ff_search_config:
  enabled: true
  max_results: 100
  relevance_threshold: 0.6
  index_refresh_interval: 300

# Production FF Vector Storage Configuration
ff_vector_storage_config:
  enabled: true
  dimension: 768
  similarity_metric: "cosine"
  max_vectors: 1000000
  batch_size: 1000
  memory_cleanup_threshold: 0.9

# Production FF Panel Configuration
ff_persona_panel_config:
  enabled: true
  max_personas: 10
  context_sharing_enabled: true
  coordination_model: "intelligent"

# Production FF Document Processing Configuration
ff_document_processing_config:
  enabled: true
  max_file_size_mb: 100
  supported_formats: ["txt", "md", "pdf", "docx", "html"]
  processing_timeout_seconds: 120

# Chat Application Settings
chat_enabled: true
chat_api_enabled: true
chat_websocket_enabled: true

# Production Chat Session Settings
chat_session:
  max_messages_per_session: 10000
  session_timeout_minutes: 240
  auto_save_interval_seconds: 15
  enable_message_threading: true
  max_concurrent_sessions_per_user: 20
  session_cleanup_interval_hours: 24

# Production Chat Component Settings
chat_components:
  # Text chat settings
  text_chat_enabled: true
  text_chat_max_tokens: 8000
  text_chat_temperature: 0.7
  
  # Memory settings
  memory_enabled: true
  memory_max_entries: 100000
  memory_similarity_threshold: 0.8
  memory_context_window: 20
  
  # Multi-agent settings
  multi_agent_enabled: true
  multi_agent_max_agents: 10
  multi_agent_coordination_model: "intelligent"
  multi_agent_timeout_seconds: 300
  
  # Tools settings
  tools_enabled: true
  tools_sandbox_enabled: true
  tools_max_execution_time: 60
  tools_max_concurrent: 5
  
  # Advanced features
  router_enabled: true
  trace_enabled: true
  persona_enabled: true
  multimodal_enabled: true

# Production Use Case Settings
chat_use_cases:
  default_use_case: "basic_chat"
  
  # All 22 use cases configured
  use_case_components:
    # Basic patterns (4)
    basic_chat: ["text_chat"]
    multimodal_chat: ["text_chat", "multimodal"]
    rag_chat: ["text_chat", "memory", "search"]
    multimodal_rag: ["text_chat", "multimodal", "memory", "search"]
    
    # Specialized modes (9)
    translation_chat: ["text_chat", "multimodal", "tools", "memory", "persona"]
    personal_assistant: ["text_chat", "tools", "memory", "persona"]
    interactive_tutor: ["text_chat", "persona"]
    language_tutor: ["text_chat", "tools", "persona"]
    exam_assistant: ["text_chat", "memory", "search"]
    ai_notetaker: ["multimodal", "memory", "search", "tools"]
    chatops_assistant: ["text_chat", "tools"]
    cross_team_concierge: ["text_chat", "memory", "search", "tools"]
    scene_critic: ["multimodal", "persona"]
    
    # Multi-participant (5)
    multi_ai_panel: ["multi_agent", "memory", "persona"]
    ai_debate: ["multi_agent", "persona", "trace"]
    topic_delegation: ["text_chat", "memory", "search", "multi_agent", "router"]
    ai_game_master: ["text_chat", "multi_agent", "memory"]
    auto_task_agent: ["tools", "multi_agent", "memory"]
    
    # Context & memory (3)
    memory_chat: ["text_chat", "memory"]
    thought_partner: ["text_chat", "memory"]
    story_world_chat: ["text_chat", "memory", "persona"]
    
    # Development (1)
    prompt_sandbox: ["text_chat", "trace"]
  
  # Use case specific settings
  use_case_settings:
    translation_chat:
      supported_languages: ["en", "es", "fr", "de", "zh", "ja"]
      translation_service: "external_api"
    
    personal_assistant:
      calendar_integration: true
      email_integration: true
      task_management: true
    
    interactive_tutor:
      subject_areas: ["math", "science", "language", "history"]
      difficulty_adaptation: true
    
    ai_notetaker:
      auto_transcription: true
      summary_generation: true
      action_item_extraction: true
    
    multi_ai_panel:
      min_agents: 2
      max_agents: 5
      voting_enabled: true
    
    ai_debate:
      topic_categories: ["ethics", "technology", "philosophy", "current_events"]
      argument_validation: true
      
    prompt_sandbox:
      version_control: true
      export_formats: ["json", "yaml", "markdown"]

# API Configuration
chat_api:
  host: "0.0.0.0"
  port: 8000
  cors_enabled: true
  rate_limiting:
    requests_per_minute: 100
    requests_per_hour: 1000
  
# WebSocket Configuration  
chat_websocket:
  host: "0.0.0.0"
  port: 8001
  max_connections: 1000
  heartbeat_interval: 30

# Logging Configuration (extends FF logging)
logging:
  level: "INFO"
  format: "detailed"
  file_rotation: true
  max_file_size_mb: 50
  backup_count: 10
  
# Monitoring Configuration
monitoring:
  metrics_enabled: true
  health_check_enabled: true
  performance_tracking: true
  error_reporting: true
```

## 2. Use Case Definitions

### Basic Patterns (4 Use Cases)

#### 1. Basic 1:1 Chat
**Description**: Simple text-based conversation between user and AI assistant.

**Components**: `text_chat`

**FF Integration**: Uses `FFStorageManager` for message persistence.

**Configuration**:
```yaml
basic_chat:
  components: ["text_chat"]
  settings:
    max_context_messages: 20
    response_timeout_seconds: 30
    personality: "helpful_assistant"
```

#### 2. Multimodal Chat  
**Description**: Chat supporting text, images, audio, and other media types.

**Components**: `text_chat`, `multimodal`

**FF Integration**: Uses `FFDocumentProcessingManager` for media handling.

**Configuration**:
```yaml
multimodal_chat:
  components: ["text_chat", "multimodal"]
  settings:
    supported_media: ["image", "audio", "video", "document"]
    max_file_size_mb: 50
    auto_transcription: true
```

#### 3. RAG Chat
**Description**: Retrieval-Augmented Generation with document knowledge.

**Components**: `text_chat`, `memory`, `search`

**FF Integration**: Uses `FFSearchManager` and `FFVectorStorageManager`.

**Configuration**:
```yaml
rag_chat:
  components: ["text_chat", "memory", "search"]
  settings:
    knowledge_base_enabled: true
    retrieval_threshold: 0.7
    max_retrieved_documents: 5
```

#### 4. Multimodal + RAG
**Description**: Combined multimodal capabilities with knowledge retrieval.

**Components**: `text_chat`, `multimodal`, `memory`, `search`

**FF Integration**: Full FF manager integration.

### Specialized Modes (9 Use Cases)

#### 5. Translation Chat
**Description**: Real-time language translation with cultural context.

**Components**: `text_chat`, `multimodal`, `tools`, `memory`, `persona`

**FF Integration**: Uses FF tools for translation APIs, memory for context.

**Configuration**:
```yaml
translation_chat:
  components: ["text_chat", "multimodal", "tools", "memory", "persona"]
  settings:
    source_language: "auto_detect"
    target_languages: ["en", "es", "fr", "de", "zh", "ja"]
    cultural_context: true
    formality_level: "adaptive"
```

#### 6. Personal Assistant
**Description**: AI assistant with task management and calendar integration.

**Components**: `text_chat`, `tools`, `memory`, `persona`

**FF Integration**: Uses FF tools for external integrations.

**Configuration**:
```yaml
personal_assistant:
  components: ["text_chat", "tools", "memory", "persona"]
  settings:
    calendar_integration: true
    email_integration: true
    task_management: true
    reminder_system: true
    scheduling_enabled: true
```

#### 7. Interactive Tutor
**Description**: Educational AI that adapts to learning style and pace.

**Components**: `text_chat`, `persona`

**FF Integration**: Uses FF persona system for educational personalities.

**Configuration**:
```yaml
interactive_tutor:
  components: ["text_chat", "persona"]
  settings:
    subject_areas: ["math", "science", "language", "history"]
    difficulty_adaptation: true
    learning_style_detection: true
    progress_tracking: true
```

#### 8. Language Tutor
**Description**: Specialized language learning with pronunciation and grammar.

**Components**: `text_chat`, `tools`, `persona`

**FF Integration**: Uses FF tools for language processing APIs.

**Configuration**:
```yaml
language_tutor:
  components: ["text_chat", "tools", "persona"]
  settings:
    target_language: "configurable"
    proficiency_level: "adaptive"
    pronunciation_feedback: true
    grammar_correction: true
    vocabulary_building: true
```

#### 9. Exam Assistant
**Description**: Study aid with knowledge base and practice questions.

**Components**: `text_chat`, `memory`, `search`

**FF Integration**: Uses FF search for exam material retrieval.

**Configuration**:
```yaml
exam_assistant:
  components: ["text_chat", "memory", "search"]
  settings:
    exam_type: "configurable"
    practice_mode: true
    knowledge_testing: true
    weak_area_identification: true
```

#### 10. AI Notetaker
**Description**: Automated meeting transcription and summary generation.

**Components**: `multimodal`, `memory`, `search`, `tools`

**FF Integration**: Uses FF document processing for audio/video.

**Configuration**:
```yaml
ai_notetaker:
  components: ["multimodal", "memory", "search", "tools"]
  settings:
    auto_transcription: true
    speaker_identification: true
    summary_generation: true
    action_item_extraction: true
    meeting_insights: true
```

#### 11. ChatOps Assistant
**Description**: DevOps automation through chat interface.

**Components**: `text_chat`, `tools`

**FF Integration**: Uses FF tools for infrastructure APIs.

**Configuration**:
```yaml
chatops_assistant:
  components: ["text_chat", "tools"]
  settings:
    supported_platforms: ["aws", "kubernetes", "docker", "github"]
    security_validation: true
    audit_logging: true
    rollback_capabilities: true
```

#### 12. Cross-Team Concierge
**Description**: Inter-team communication and knowledge sharing.

**Components**: `text_chat`, `memory`, `search`, `tools`

**FF Integration**: Full FF integration for team knowledge bases.

**Configuration**:
```yaml
cross_team_concierge:
  components: ["text_chat", "memory", "search", "tools"]
  settings:
    team_contexts: "configurable"
    knowledge_routing: true
    escalation_rules: true
    expertise_matching: true
```

#### 13. Scene Critic
**Description**: Creative feedback on visual and narrative content.

**Components**: `multimodal`, `persona`

**FF Integration**: Uses FF document processing and persona system.

**Configuration**:
```yaml
scene_critic:
  components: ["multimodal", "persona"]
  settings:
    critique_style: "constructive"
    artistic_domains: ["visual", "narrative", "cinematic"]
    feedback_depth: "detailed"
```

### Multi-Participant (5 Use Cases)

#### 14. Multi-AI Panel
**Description**: Multiple AI personas collaborating on complex problems.

**Components**: `multi_agent`, `memory`, `persona`

**FF Integration**: Uses `FFPanelManager` for agent coordination.

**Configuration**:
```yaml
multi_ai_panel:
  components: ["multi_agent", "memory", "persona"]
  settings:
    min_agents: 2
    max_agents: 5
    agent_specializations: "configurable"
    consensus_mechanism: "voting"
    conflict_resolution: "moderated"
```

#### 15. AI Debate
**Description**: Structured debates between AI agents on complex topics.

**Components**: `multi_agent`, `persona`, `trace`

**FF Integration**: Uses FF panel system with trace logging.

**Configuration**:
```yaml
ai_debate:
  components: ["multi_agent", "persona", "trace"]
  settings:
    debate_format: "structured"
    topic_categories: ["ethics", "technology", "philosophy", "current_events"]
    argument_validation: true
    fact_checking: true
    moderation_enabled: true
```

#### 16. Topic Delegation
**Description**: Intelligent routing of queries to specialized AI agents.

**Components**: `text_chat`, `memory`, `search`, `multi_agent`, `router`

**FF Integration**: Uses FF search for topic classification.

**Configuration**:
```yaml
topic_delegation:
  components: ["text_chat", "memory", "search", "multi_agent", "router"]
  settings:
    routing_strategy: "expertise_based"
    fallback_agent: "generalist"
    delegation_threshold: 0.8
    load_balancing: true
```

#### 17. AI Game Master
**Description**: Interactive storytelling and game management.

**Components**: `text_chat`, `multi_agent`, `memory`

**FF Integration**: Uses FF memory for game state persistence.

**Configuration**:
```yaml
ai_game_master:
  components: ["text_chat", "multi_agent", "memory"]
  settings:
    game_types: ["rpg", "mystery", "adventure", "puzzle"]
    npc_management: true
    world_state_tracking: true
    dynamic_story_adaptation: true
```

#### 18. Auto Task Agent
**Description**: Autonomous task execution with multi-agent coordination.

**Components**: `tools`, `multi_agent`, `memory`

**FF Integration**: Uses FF tools for task execution APIs.

**Configuration**:
```yaml
auto_task_agent:
  components: ["tools", "multi_agent", "memory"]
  settings:
    task_decomposition: true
    parallel_execution: true
    error_recovery: true
    progress_reporting: true
    human_approval_required: true
```

### Context & Memory (3 Use Cases)

#### 19. Memory Chat
**Description**: Persistent memory across conversations with context awareness.

**Components**: `text_chat`, `memory`

**FF Integration**: Uses `FFVectorStorageManager` for memory embeddings.

**Configuration**:
```yaml
memory_chat:
  components: ["text_chat", "memory"]
  settings:
    memory_types: ["episodic", "semantic", "procedural"]
    retention_period: "permanent"
    context_window: 50
    memory_consolidation: true
```

#### 20. Thought Partner
**Description**: Deep thinking companion for complex problem-solving.

**Components**: `text_chat`, `memory`

**FF Integration**: Advanced memory patterns for thought processes.

**Configuration**:
```yaml
thought_partner:
  components: ["text_chat", "memory"]
  settings:
    thinking_frameworks: ["systems", "design", "critical", "creative"]
    reflection_prompts: true
    idea_development: true
    decision_support: true
```

#### 21. Story World Chat
**Description**: Immersive fictional world with persistent characters and lore.

**Components**: `text_chat`, `memory`, `persona`

**FF Integration**: Uses FF memory for world state and FF persona for characters.

**Configuration**:
```yaml
story_world_chat:
  components: ["text_chat", "memory", "persona"]
  settings:
    world_building: true
    character_consistency: true
    lore_tracking: true
    timeline_management: true
    immersion_level: "high"
```

### Development (1 Use Case)

#### 22. Prompt Sandbox
**Description**: Prompt engineering and testing environment with version control.

**Components**: `text_chat`, `trace`

**FF Integration**: Uses FF trace logging for prompt analysis.

**Configuration**:
```yaml
prompt_sandbox:
  components: ["text_chat", "trace"]
  settings:
    version_control: true
    A_B_testing: true
    performance_metrics: true
    export_formats: ["json", "yaml", "markdown"]
    collaboration_features: true
```

## 3. Configuration Loading Examples

### Python Configuration Loading

```python
"""
Example of loading FF chat configurations following FF patterns
"""

import yaml
from pathlib import Path
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_chat_application_manager import FFChatApplicationConfigDTO

def load_ff_chat_config(config_path: str) -> FFChatApplicationConfigDTO:
    """Load FF chat configuration from YAML file"""
    
    # Load YAML configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Create FF chat configuration from data
    config = FFChatApplicationConfigDTO()
    
    # Load existing FF configurations (unchanged pattern)
    if 'ff_storage_config' in config_data:
        config.ff_storage_config = FFStorageConfigDTO(**config_data['ff_storage_config'])
    
    if 'ff_search_config' in config_data:
        config.ff_search_config = FFSearchConfigDTO(**config_data['ff_search_config'])
    
    # Load chat-specific configurations
    if 'chat_session' in config_data:
        config.chat_session = FFChatSessionConfigDTO(**config_data['chat_session'])
    
    if 'chat_components' in config_data:
        config.chat_components = FFChatComponentsConfigDTO(**config_data['chat_components'])
    
    if 'chat_use_cases' in config_data:
        config.chat_use_cases = FFChatUseCasesConfigDTO(**config_data['chat_use_cases'])
    
    return config

# Usage example
async def main():
    # Load production configuration
    config = load_ff_chat_config("production_config.yaml")
    
    # Initialize FF chat application
    app = FFChatApplicationManager(config)
    await app.initialize()
    
    # Use application...
    
    await app.cleanup()
```

### Environment-Based Configuration

```python
"""
Environment-based configuration for FF chat system
"""

import os
from ff_chat_application_manager import FFChatApplicationConfigDTO

def create_environment_config() -> FFChatApplicationConfigDTO:
    """Create configuration from environment variables"""
    
    config = FFChatApplicationConfigDTO()
    
    # FF Storage settings from environment
    config.ff_storage_config.storage_base_path = os.getenv(
        'FF_STORAGE_PATH', './ff_chat_data'
    )
    config.ff_storage_config.session_id_prefix = os.getenv(
        'FF_SESSION_PREFIX', 'chat_'
    )
    
    # Chat settings from environment
    config.chat_enabled = os.getenv('CHAT_ENABLED', 'true').lower() == 'true'
    config.chat_api_enabled = os.getenv('CHAT_API_ENABLED', 'false').lower() == 'true'
    
    # Component settings from environment
    config.chat_components.text_chat_enabled = os.getenv(
        'TEXT_CHAT_ENABLED', 'true'
    ).lower() == 'true'
    
    config.chat_components.memory_enabled = os.getenv(
        'MEMORY_ENABLED', 'false'
    ).lower() == 'true'
    
    config.chat_components.multi_agent_enabled = os.getenv(
        'MULTI_AGENT_ENABLED', 'false'
    ).lower() == 'true'
    
    # Use case settings
    config.chat_use_cases.default_use_case = os.getenv(
        'DEFAULT_USE_CASE', 'basic_chat'
    )
    
    return config
```

## 4. Docker Configuration

### Docker Compose for FF Chat System

```yaml
# docker-compose.yml - FF Chat System deployment
version: '3.8'

services:
  ff_chat_app:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - FF_STORAGE_PATH=/app/data
      - CHAT_ENABLED=true
      - CHAT_API_ENABLED=true
      - CHAT_WEBSOCKET_ENABLED=true
      - TEXT_CHAT_ENABLED=true
      - MEMORY_ENABLED=true
      - MULTI_AGENT_ENABLED=true
      - DEFAULT_USE_CASE=basic_chat
    volumes:
      - ff_chat_data:/app/data
      - ./config/production_config.yaml:/app/config.yaml
    depends_on:
      - redis
      - postgres
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=ff_chat
      - POSTGRES_USER=ff_chat_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  ff_chat_data:
  redis_data:
  postgres_data:
```

This comprehensive configuration guide provides all necessary examples for implementing and deploying the FF Chat System while maintaining integration with existing FF infrastructure patterns.