# Ready-to-Use Implementation Prompts

Copy and paste these prompts directly into Claude Code with the required attached files.

---

## 🧠 **Phase 1: Multi-Layered Memory System**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_1_MEMORY_LAYERS.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 1: Multi-Layered Memory System for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_1_MEMORY_LAYERS.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand:
  - Existing system architecture (what NOT to break)
  - Mandatory architectural principles and patterns
  - Required code standards and conventions
  - Integration guidelines and dependencies

Step 2: Phase Specification Review
- Read PHASE_1_MEMORY_LAYERS.md completely to understand:
  - Multi-tiered memory system requirements (5 retention layers)
  - Data models and DTOs to implement
  - Memory layer manager functionality
  - Protocol interfaces required
  - Integration with existing FFContextManager
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_memory_layer_config.py - Memory configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add memory_layer config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add memory context DTOs

Manager Implementation:
- [ ] ff_memory_layer_manager.py - Multi-tiered memory management system
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md
- [ ] Include comprehensive error handling and logging
- [ ] Implement all methods specified in the phase specification

Protocol Interface:
- [ ] ff_protocols/ff_memory_layer_protocol.py - Abstract interface for dependency injection

Unit Tests:
- [ ] tests/test_memory_layer_manager.py - Comprehensive unit test suite
- [ ] Follow testing patterns from IMPLEMENTATION_CONTEXT.md
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

Backward Compatibility:
- ✅ NEVER modify existing files except at designated extension points
- ✅ NEVER change existing function signatures or data structures
- ✅ ALWAYS ensure existing functionality continues working unchanged
- ✅ Test that current system features still work after your changes

Architecture Compliance:
- ✅ Follow EXACT DTO patterns from IMPLEMENTATION_CONTEXT.md
- ✅ Use established manager implementation pattern
- ✅ Implement protocol interfaces for dependency injection
- ✅ Follow atomic file operations using ff_file_ops utilities
- ✅ Use proper async/await patterns throughout

Memory System Requirements:
- ✅ Implement 5 memory layers: immediate, short-term, medium-term, long-term, permanent
- ✅ Automatic archival based on retention policies and relevance scores
- ✅ Memory compression with summarization capabilities
- ✅ Integration with existing FFContextManager
- ✅ User-based memory isolation following established patterns

✅ Success Criteria
Your implementation is successful when:
- [ ] All existing system functionality works unchanged
- [ ] Multi-tiered memory system operational with all 5 layers
- [ ] Memory compression and archival working automatically
- [ ] Integration with existing context manager seamless
- [ ] Unit tests pass with >90% coverage
- [ ] Performance meets established benchmarks

Ready to implement Phase 1? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## 👥 **Phase 2: Panel Session Enhancement**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_2_PANEL_SESSIONS.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 2: Panel Session Enhancement for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_2_PANEL_SESSIONS.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand existing architecture and patterns
- Note dependencies on Phase 1 (Memory Layers) - you will integrate with FFMemoryLayerManager

Step 2: Phase Specification Review
- Read PHASE_2_PANEL_SESSIONS.md completely to understand:
  - Enhanced panel session management requirements
  - Multi-agent collaboration capabilities
  - Panel participant coordination and insights
  - Integration with memory system from Phase 1
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_panel_session_config.py - Enhanced panel configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add panel_session config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add panel session DTOs

Manager Implementation:
- [ ] ff_panel_session_manager.py - Enhanced panel session management system
- [ ] Integration with FFMemoryLayerManager from Phase 1
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md
- [ ] Include comprehensive error handling and logging

Protocol Interface:
- [ ] ff_protocols/ff_panel_session_protocol.py - Abstract interface for dependency injection

Unit Tests:
- [ ] tests/test_panel_session_manager.py - Comprehensive unit test suite
- [ ] Test integration with memory layer manager
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

Phase Dependencies:
- ✅ Must integrate with FFMemoryLayerManager from Phase 1
- ✅ Use memory layers to store panel session insights and context
- ✅ Follow established integration patterns

Panel Enhancement Requirements:
- ✅ Enhanced participant management with roles and capabilities
- ✅ Panel session coordination with state management
- ✅ Participant insights and decision tracking
- ✅ Analytics for panel effectiveness and consensus
- ✅ Integration with existing panel system (extend, don't replace)

✅ Success Criteria
Your implementation is successful when:
- [ ] All existing panel functionality enhanced, not replaced
- [ ] Multi-agent collaboration capabilities operational
- [ ] Integration with memory layers working seamlessly
- [ ] Panel insights captured and analyzed
- [ ] Unit tests pass with >90% coverage

Ready to implement Phase 2? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## 🔍 **Phase 3: RAG Integration**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_3_RAG_INTEGRATION.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 3: RAG Integration for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_3_RAG_INTEGRATION.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand existing architecture and patterns
- Note dependencies on Phase 1 (Memory) and Phase 2 (Panels) - you will integrate with both

Step 2: Phase Specification Review
- Read PHASE_3_RAG_INTEGRATION.md completely to understand:
  - Knowledge base lifecycle management requirements
  - RAG context management and retrieval optimization
  - Document indexing and vector storage patterns
  - Integration with memory and panel systems
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_knowledge_base_config.py - Knowledge base configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add knowledge_base config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add RAG-related DTOs

Manager Implementation:
- [ ] ff_knowledge_base_manager.py - Knowledge base lifecycle management
- [ ] ff_rag_context_manager.py - RAG context management and retrieval
- [ ] Integration with FFMemoryLayerManager and FFPanelSessionManager
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md

Protocol Interface:
- [ ] ff_protocols/ff_knowledge_base_protocol.py - Abstract interface for dependency injection

Unit Tests:
- [ ] tests/test_knowledge_base_manager.py - Comprehensive unit test suite
- [ ] tests/test_rag_context_manager.py - RAG context testing
- [ ] Test integration with memory and panel systems
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

Phase Dependencies:
- ✅ Must integrate with FFMemoryLayerManager from Phase 1
- ✅ Must integrate with FFPanelSessionManager from Phase 2
- ✅ Store knowledge base context in memory layers
- ✅ Support RAG in panel sessions

RAG System Requirements:
- ✅ Personal knowledge base creation and management
- ✅ Document indexing with vector storage (NumPy arrays + JSONL)
- ✅ Conversation-specific RAG context management
- ✅ Knowledge base lifecycle and maintenance
- ✅ Integration with existing document processing capabilities

✅ Success Criteria
Your implementation is successful when:
- [ ] Knowledge base creation and management operational
- [ ] RAG context retrieval working in conversations
- [ ] Integration with memory and panel systems seamless
- [ ] Document indexing and search performing efficiently
- [ ] Unit tests pass with >90% coverage

Ready to implement Phase 3? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## 🛠️ **Phase 4: Tool Execution Framework**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_4_TOOL_EXECUTION.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 4: Tool Execution Framework for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_4_TOOL_EXECUTION.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand existing architecture and patterns
- Note integration with Phase 1 (Memory) for tool context storage

Step 2: Phase Specification Review
- Read PHASE_4_TOOL_EXECUTION.md completely to understand:
  - Tool registry and capability management requirements
  - Secure execution environment with sandboxing
  - Security policies and performance monitoring
  - Tool integration with conversation flow
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_tool_execution_config.py - Tool execution configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add tool_execution config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add tool execution DTOs

Manager Implementation:
- [ ] ff_tool_registry_manager.py - Tool registry and capability management
- [ ] ff_tool_execution_manager.py - Secure tool execution and orchestration
- [ ] Integration with FFMemoryLayerManager for tool context
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md

Protocol Interface:
- [ ] ff_protocols/ff_tool_execution_protocol.py - Abstract interface for dependency injection

Unit Tests:
- [ ] tests/test_tool_registry_manager.py - Tool registry testing
- [ ] tests/test_tool_execution_manager.py - Tool execution testing
- [ ] Test security policies and sandboxing
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

Security Requirements:
- ✅ Secure tool execution with sandboxing and resource limits
- ✅ Security policies with permission management and rate limiting
- ✅ Comprehensive audit logging for all tool executions
- ✅ Input validation and security scanning

Tool System Requirements:
- ✅ Dynamic tool registry with discovery and validation
- ✅ Tool execution engine with multiple security modes
- ✅ Performance monitoring and analytics
- ✅ Integration with conversation context and memory
- ✅ Built-in tool library with essential capabilities

✅ Success Criteria
Your implementation is successful when:
- [ ] Tool registry supports dynamic tool management
- [ ] Secure execution environment prevents system compromise
- [ ] Security policies enforce proper access controls
- [ ] Tool results integrate seamlessly with conversations
- [ ] Unit tests pass with >90% coverage

Ready to implement Phase 4? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## 📊 **Phase 5: Analytics & Monitoring**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_5_ANALYTICS_MONITORING.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 5: Analytics & Monitoring System for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_5_ANALYTICS_MONITORING.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand existing architecture and patterns
- Note this phase monitors ALL previous phases - integration with all components

Step 2: Phase Specification Review
- Read PHASE_5_ANALYTICS_MONITORING.md completely to understand:
  - System-wide metrics collection requirements
  - User behavior analytics with privacy controls
  - Business intelligence and usage insights
  - Real-time monitoring with alerting
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_analytics_config.py - Analytics configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add analytics config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add analytics DTOs

Manager Implementation:
- [ ] ff_metrics_collection_manager.py - Comprehensive metrics collection system
- [ ] ff_analytics_dashboard_manager.py - Analytics dashboard and reporting
- [ ] Integration with ALL previous phase components for monitoring
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md

Protocol Interface:
- [ ] ff_protocols/ff_analytics_protocol.py - Abstract interface for dependency injection

Unit Tests:
- [ ] tests/test_metrics_collection_manager.py - Metrics collection testing
- [ ] tests/test_analytics_dashboard_manager.py - Dashboard testing
- [ ] Test privacy controls and data anonymization
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

Privacy and Compliance:
- ✅ User data anonymization and privacy controls
- ✅ Configurable data retention policies
- ✅ GDPR and privacy regulation compliance
- ✅ Audit logging for compliance requirements

Analytics Requirements:
- ✅ System performance monitoring with alerting
- ✅ User behavior analytics with engagement metrics
- ✅ Business intelligence with usage trends
- ✅ Real-time monitoring with anomaly detection
- ✅ Integration with all system components

✅ Success Criteria
Your implementation is successful when:
- [ ] System metrics collected automatically across all components
- [ ] User analytics captured with privacy protection
- [ ] Business intelligence provides actionable insights
- [ ] Real-time monitoring detects issues proactively
- [ ] Unit tests pass with >90% coverage

Ready to implement Phase 5? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## ✅ **Phase 6: Integration & Testing**

### **📎 Required Attachments**
- `IMPLEMENTATION_CONTEXT.md`
- `PHASE_6_INTEGRATION_TESTING.md`

### **📋 Copy This Prompt:**

```
I need you to implement Phase 6: Integration & Testing for my flat file chat database upgrade to support advanced PrismMind capabilities.

📎 Required Files (Must Read Both)
CRITICAL: You must read both attached files before starting implementation:

1. IMPLEMENTATION_CONTEXT.md - Complete project context, architectural standards, and implementation patterns
2. PHASE_6_INTEGRATION_TESTING.md - Detailed specification for this specific phase

🎯 Your Implementation Task

Step 1: Context Understanding
- Read IMPLEMENTATION_CONTEXT.md thoroughly to understand existing architecture and patterns
- Note this phase validates ALL previous phases - comprehensive system testing

Step 2: Phase Specification Review
- Read PHASE_6_INTEGRATION_TESTING.md completely to understand:
  - End-to-end integration testing requirements
  - All 22 PrismMind use case validation
  - Performance, security, and migration testing
  - System integration reporting
  - Testing requirements

Step 3: Complete Implementation
Create ALL specified files with complete, production-ready implementations:

Configuration Files:
- [ ] ff_class_configs/ff_integration_testing_config.py - Testing configuration DTOs
- [ ] Extend ff_class_configs/ff_configuration_manager_config.py - Add integration_testing config
- [ ] Extend ff_class_configs/ff_chat_entities_config.py - Add testing result DTOs

Manager Implementation:
- [ ] ff_integration_test_manager.py - Comprehensive integration testing system
- [ ] ff_migration_validator.py - Data migration and validation tools
- [ ] Integration with ALL previous phase components for testing
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md

Protocol Interface:
- [ ] ff_protocols/ff_integration_testing_protocol.py - Abstract interface for dependency injection

Comprehensive Testing:
- [ ] tests/test_integration_test_manager.py - Integration testing validation
- [ ] tests/test_complete_system_integration.py - End-to-end system testing
- [ ] All 22 PrismMind use case validation tests
- [ ] Achieve >90% test coverage

🚨 CRITICAL Requirements (MUST FOLLOW)

System Validation:
- ✅ All 22 PrismMind use cases validated end-to-end
- ✅ Component integration tests for all phase interactions
- ✅ Performance testing meets enterprise requirements
- ✅ Security validation confirms system hardening
- ✅ Migration procedures tested with rollback capability

Testing Requirements:
- ✅ Comprehensive test suite for entire system
- ✅ Integration reporting with actionable insights
- ✅ Performance benchmarks for ongoing monitoring
- ✅ Security baseline validation
- ✅ Production readiness assessment

✅ Success Criteria
Your implementation is successful when:
- [ ] All 22 PrismMind use cases pass validation
- [ ] System integration tests achieve 100% success
- [ ] Performance tests meet established benchmarks
- [ ] Security validation passes all critical requirements
- [ ] Migration procedures validated with rollback capability
- [ ] Production readiness score exceeds 95%

Ready to implement Phase 6? Please confirm you've read both attached files and understand the requirements before beginning.
```

---

## 📋 **Usage Instructions**

### **For Each Phase:**

1. **Copy the appropriate prompt** from above
2. **Attach the two required files:**
   - `IMPLEMENTATION_CONTEXT.md` (always required)
   - `PHASE_X_[COMPONENT].md` (specific to the phase)
3. **Paste the prompt** into Claude Code
4. **Wait for complete implementation**
5. **Validate results** before proceeding to next phase

### **File Attachment Checklist:**
- [ ] `IMPLEMENTATION_CONTEXT.md` ✅ Attached
- [ ] `PHASE_X_[COMPONENT].md` ✅ Attached  
- [ ] Prompt customized with correct phase information ✅
- [ ] Claude Code session ready ✅

These prompts ensure consistent, comprehensive implementation across all phases while maintaining full architectural compliance and system integrity.