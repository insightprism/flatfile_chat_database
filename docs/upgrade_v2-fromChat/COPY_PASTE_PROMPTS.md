# Copy-Paste Implementation Prompts

**IMPORTANT**: For each prompt, you must attach these two files:
1. `IMPLEMENTATION_CONTEXT.md`
2. `PHASE_X_[COMPONENT].md` (the specific phase file)

---

## 🧠 Phase 1: Multi-Layered Memory System

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_1_MEMORY_LAYERS.md`

**Prompt to Copy:**

```
I need you to implement Phase 1: Multi-Layered Memory System for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_1_MEMORY_LAYERS.md - Phase 1 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_memory_layer_config.py - Memory layer configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add memory_layer field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add memory context DTOs
4. ff_memory_layer_manager.py - Multi-tiered memory management system
5. ff_protocols/ff_memory_layer_protocol.py - Protocol interface
6. tests/test_memory_layer_manager.py - Comprehensive unit tests

Requirements:
- Implement 5 memory layers: immediate, short-term, medium-term, long-term, permanent
- Automatic archival based on retention policies and relevance scores
- Memory compression with summarization capabilities
- Integration with existing FFContextManager
- User-based memory isolation following established patterns
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: Multi-tiered memory system operational, existing system unchanged, comprehensive tests passing.
```

---

## 👥 Phase 2: Panel Session Enhancement

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_2_PANEL_SESSIONS.md`

**Prompt to Copy:**

```
I need you to implement Phase 2: Panel Session Enhancement for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_2_PANEL_SESSIONS.md - Phase 2 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_panel_session_config.py - Enhanced panel configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add panel_session field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add panel session DTOs
4. ff_panel_session_manager.py - Enhanced panel session management system
5. ff_protocols/ff_panel_session_protocol.py - Protocol interface
6. tests/test_panel_session_manager.py - Comprehensive unit tests

Requirements:
- Enhanced participant management with roles and capabilities
- Panel session coordination with state management
- Participant insights and decision tracking
- Analytics for panel effectiveness and consensus
- Integration with FFMemoryLayerManager from Phase 1
- Use memory layers to store panel session insights and context
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: Enhanced panel collaboration operational, memory integration working, existing panel functionality enhanced.
```

---

## 🔍 Phase 3: RAG Integration

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_3_RAG_INTEGRATION.md`

**Prompt to Copy:**

```
I need you to implement Phase 3: RAG Integration for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_3_RAG_INTEGRATION.md - Phase 3 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_knowledge_base_config.py - Knowledge base configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add knowledge_base field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add RAG-related DTOs
4. ff_knowledge_base_manager.py - Knowledge base lifecycle management
5. ff_rag_context_manager.py - RAG context management and retrieval
6. ff_protocols/ff_knowledge_base_protocol.py - Protocol interface
7. tests/test_knowledge_base_manager.py - Knowledge base tests
8. tests/test_rag_context_manager.py - RAG context tests

Requirements:
- Personal knowledge base creation and management
- Document indexing with vector storage (NumPy arrays + JSONL)
- Conversation-specific RAG context management
- Knowledge base lifecycle and maintenance
- Integration with FFMemoryLayerManager and FFPanelSessionManager
- Store knowledge base context in memory layers
- Support RAG in panel sessions
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: Knowledge base management operational, RAG context retrieval working, integration with memory and panels seamless.
```

---

## 🛠️ Phase 4: Tool Execution Framework

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_4_TOOL_EXECUTION.md`

**Prompt to Copy:**

```
I need you to implement Phase 4: Tool Execution Framework for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_4_TOOL_EXECUTION.md - Phase 4 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_tool_execution_config.py - Tool execution configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add tool_execution field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add tool execution DTOs
4. ff_tool_registry_manager.py - Tool registry and capability management
5. ff_tool_execution_manager.py - Secure tool execution and orchestration
6. ff_protocols/ff_tool_execution_protocol.py - Protocol interface
7. tests/test_tool_registry_manager.py - Tool registry tests
8. tests/test_tool_execution_manager.py - Tool execution tests

Requirements:
- Dynamic tool registry with discovery and validation
- Secure tool execution with sandboxing and resource limits
- Security policies with permission management and rate limiting
- Comprehensive audit logging for all tool executions
- Performance monitoring and analytics
- Integration with FFMemoryLayerManager for tool context
- Tool results integrate seamlessly with conversations
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: Tool registry operational, secure execution working, security policies enforced, tool results integrated with conversations.
```

---

## 📊 Phase 5: Analytics & Monitoring

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_5_ANALYTICS_MONITORING.md`

**Prompt to Copy:**

```
I need you to implement Phase 5: Analytics & Monitoring System for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_5_ANALYTICS_MONITORING.md - Phase 5 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_analytics_config.py - Analytics configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add analytics field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add analytics DTOs
4. ff_metrics_collection_manager.py - Comprehensive metrics collection system
5. ff_analytics_dashboard_manager.py - Analytics dashboard and reporting
6. ff_protocols/ff_analytics_protocol.py - Protocol interface
7. tests/test_metrics_collection_manager.py - Metrics collection tests
8. tests/test_analytics_dashboard_manager.py - Dashboard tests

Requirements:
- System performance monitoring with alerting
- User behavior analytics with engagement metrics and privacy controls
- Business intelligence with usage trends and insights
- Real-time monitoring with anomaly detection
- Integration with ALL previous phase components for monitoring
- User data anonymization and privacy controls
- Configurable data retention policies
- GDPR and privacy regulation compliance
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: System metrics collected automatically, user analytics captured with privacy protection, business intelligence operational, real-time monitoring working.
```

---

## ✅ Phase 6: Integration & Testing

**Required Files to Attach:**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_6_INTEGRATION_TESTING.md`

**Prompt to Copy:**

```
I need you to implement Phase 6: Integration & Testing for my flat file chat database upgrade.

CRITICAL: Read both attached files before starting:
1. IMPLEMENTATION_CONTEXT.md - Project context and architectural standards
2. PHASE_6_INTEGRATION_TESTING.md - Phase 6 detailed specification

Create these files following the exact patterns from IMPLEMENTATION_CONTEXT.md:

1. ff_class_configs/ff_integration_testing_config.py - Testing configuration DTOs
2. Extend ff_class_configs/ff_configuration_manager_config.py - Add integration_testing field
3. Extend ff_class_configs/ff_chat_entities_config.py - Add testing result DTOs
4. ff_integration_test_manager.py - Comprehensive integration testing system
5. ff_migration_validator.py - Data migration and validation tools
6. ff_protocols/ff_integration_testing_protocol.py - Protocol interface
7. tests/test_integration_test_manager.py - Integration testing validation
8. tests/test_complete_system_integration.py - End-to-end system testing

Requirements:
- End-to-end integration testing for all components
- All 22 PrismMind use case validation
- Performance testing meets enterprise requirements
- Security validation confirms system hardening
- Migration procedures tested with rollback capability
- Component integration tests for all phase interactions
- Integration with ALL previous phase components for testing
- Comprehensive test suite for entire system
- Integration reporting with actionable insights
- Follow ALL architectural patterns from IMPLEMENTATION_CONTEXT.md
- Maintain 100% backward compatibility

Success criteria: All 22 PrismMind use cases validated, system integration tests pass, performance and security requirements met, production readiness confirmed.
```

---

## 📋 Quick Reference

**For each phase:**
1. Copy the prompt above
2. Attach the two required files to Claude Code
3. Paste the prompt
4. Wait for implementation
5. Test before moving to next phase

**File attachment checklist:**
- [ ] `IMPLEMENTATION_CONTEXT.md` ✅
- [ ] `PHASE_X_[COMPONENT].md` ✅
- [ ] Prompt copied correctly ✅