# Phase 2 Implementation Prompt

Copy and paste this entire prompt to Claude Code to implement Phase 2:

---

## Context and Objective

I need you to implement **Phase 2: Chat Capabilities** for the FF Chat System. This phase builds on the Phase 1 foundation to implement core chat components that provide 19/22 use cases (86% coverage) while using existing FF managers as backend services.

**Prerequisites**: Phase 1 must be completed successfully before starting Phase 2.

## Implementation Context

Please read and understand the implementation context from this file:
`/home/markly2/claude_code/flatfile_chat_database_v2/docs/upgrade_pm_integration/IMPLEMENTATION_CONTEXT.md`

## Phase Specification

Please read the detailed Phase 2 specification from this file:
`/home/markly2/claude_code/flatfile_chat_database_v2/docs/upgrade_pm_integration/phase_2_chat_capabilities.md`

## Code Templates and Patterns  

For implementation guidance, refer to the code templates:
`/home/markly2/claude_code/flatfile_chat_database_v2/docs/upgrade_pm_integration/code_templates.md`

## Configuration Examples

For configuration patterns, refer to:
`/home/markly2/claude_code/flatfile_chat_database_v2/docs/upgrade_pm_integration/configuration_examples.md`

## Test Templates

For testing patterns, refer to:
`/home/markly2/claude_code/flatfile_chat_database_v2/docs/upgrade_pm_integration/test_templates.md`

## Key Requirements for Phase 2

1. **Create FF Text Chat Component** (`ff_text_chat_component.py`)
   - Handles all text-based chat interactions
   - Uses existing `FFStorageManager` for message persistence
   - Supports 17/22 use cases (77% coverage)
   - Implements conversation management and response generation

2. **Create FF Memory Component** (`ff_memory_component.py`)
   - Provides persistent memory across conversations
   - Uses existing `FFVectorStorageManager` for embeddings and similarity search
   - Supports context-aware responses and information retrieval
   - Covers 7/22 use cases (32% coverage)

3. **Create FF Multi-Agent Component** (`ff_multi_agent_component.py`)
   - Coordinates multiple AI agents for complex discussions
   - Uses existing `FFPanelManager` for agent coordination
   - Supports consensus building and specialized expertise
   - Covers 5/22 use cases (23% coverage)

4. **Create Component Registry System** (`ff_chat_component_registry.py`)
   - Manages component lifecycle and dependencies
   - Integrates with existing `ff_dependency_injection_manager`
   - Provides dynamic component loading and configuration
   - Handles component initialization and cleanup

5. **Enhance Phase 1 Managers**
   - Update `ff_chat_application.py` to register and use new components
   - Extend `ff_chat_use_case_manager.py` to route to specific components
   - Add component configuration to existing config classes

## Implementation Guidelines

- **Build on Phase 1 Foundation**: Use the chat application and session managers from Phase 1
- **Follow FF Integration Patterns**: Each component must use existing FF managers as backend
- **Maintain FF Conventions**: Continue using FF naming, async patterns, and configuration approaches
- **Component Protocol Compliance**: All components must implement the FF chat component protocol from Phase 1
- **Backward Compatibility**: Existing FF functionality must remain unchanged
- **Performance**: New components should not degrade existing FF manager performance

## Success Criteria

Phase 2 is complete when:

1. ✅ **Text Chat Component** successfully processes messages using FF storage
2. ✅ **Memory Component** stores and retrieves context using FF vector storage
3. ✅ **Multi-Agent Component** coordinates agents using FF panel manager
4. ✅ **Component Registry** manages component lifecycle with FF dependency injection
5. ✅ **Use Case Coverage** - 19/22 use cases work end-to-end
6. ✅ **Integration Tests** verify components work with existing FF managers
7. ✅ **Performance Tests** show no degradation to existing FF operations
8. ✅ **Documentation** includes component usage examples and integration patterns

## File Structure Expected

After Phase 2 implementation, these new files should exist:

```
# Core Chat Components
ff_text_chat_component.py               # Text chat processing component
ff_memory_component.py                  # Memory and context component  
ff_multi_agent_component.py             # Multi-agent coordination component
ff_chat_component_registry.py           # Component management system

# Component Configurations
ff_class_configs/
├── ff_text_chat_config.py              # Text chat component config
├── ff_memory_config.py                 # Memory component config
├── ff_multi_agent_config.py            # Multi-agent component config
└── ff_component_registry_config.py     # Registry configuration

# Component Protocols (if needed beyond Phase 1)
ff_protocols/
├── ff_text_chat_protocol.py            # Text chat interface
├── ff_memory_protocol.py               # Memory interface
└── ff_multi_agent_protocol.py          # Multi-agent interface

# Tests
tests/
├── unit/
│   ├── test_ff_text_chat_component.py  # Text chat component tests
│   ├── test_ff_memory_component.py     # Memory component tests
│   ├── test_ff_multi_agent_component.py # Multi-agent component tests
│   └── test_ff_component_registry.py   # Registry tests
├── integration/
│   ├── test_ff_chat_storage_integration.py    # Storage integration
│   ├── test_ff_chat_vector_integration.py     # Vector storage integration
│   ├── test_ff_chat_panel_integration.py      # Panel manager integration
│   └── test_ff_chat_phase2_integration.py     # Phase 2 integration tests
└── system/
    └── test_ff_chat_use_cases_phase2.py       # End-to-end use case tests
```

## Use Cases to Implement and Test

Phase 2 should support these 19 use cases:

### Basic Patterns (4/4)
- ✅ Basic 1:1 Chat (text_chat)
- ✅ Multimodal Chat (text_chat + multimodal - partial support)
- ✅ RAG Chat (text_chat + memory)  
- ✅ Multimodal + RAG (text_chat + multimodal + memory - partial support)

### Specialized Modes (7/9) 
- ✅ Personal Assistant (text_chat + memory)
- ✅ Interactive Tutor (text_chat)
- ✅ Language Tutor (text_chat) 
- ✅ Exam Assistant (text_chat + memory)
- ✅ ChatOps Assistant (text_chat)
- ✅ Cross-Team Concierge (text_chat + memory)
- ✅ Scene Critic (text_chat - partial support)

### Multi-Participant (5/5)
- ✅ Multi-AI Panel (multi_agent + memory)
- ✅ AI Debate (multi_agent)
- ✅ Topic Delegation (text_chat + multi_agent)
- ✅ AI Game Master (text_chat + multi_agent + memory)
- ✅ Auto Task Agent (multi_agent + memory)

### Context & Memory (3/3)
- ✅ Memory Chat (text_chat + memory)
- ✅ Thought Partner (text_chat + memory)
- ✅ Story World Chat (text_chat + memory)

### Development (0/1)
- ⏸️ Prompt Sandbox (Phase 3)

**Total: 19/22 use cases (86% coverage)**

## Implementation Sequence

1. **Text Chat Component First**
   - Implement basic text processing and response generation
   - Integrate with FF storage for message persistence
   - Test with basic chat use cases

2. **Memory Component Second**
   - Implement context storage and retrieval using FF vector storage
   - Add similarity search and context awareness
   - Test with memory-based use cases

3. **Multi-Agent Component Third** 
   - Implement agent coordination using FF panel manager
   - Add consensus mechanisms and role specialization
   - Test with multi-participant use cases

4. **Component Registry Last**
   - Implement component management and lifecycle
   - Integrate with existing FF dependency injection
   - Test dynamic component loading and configuration

5. **Integration and System Testing**
   - Test all 19 use cases end-to-end
   - Verify integration with existing FF managers
   - Performance and regression testing

## Testing Requirements

Create comprehensive tests that verify:
- Each component works independently with its respective FF manager backend
- Components integrate correctly through the Phase 1 application manager
- All 19 target use cases work end-to-end
- No performance degradation to existing FF functionality
- Memory components properly use FF vector storage for embeddings
- Multi-agent components properly use FF panel manager coordination
- Component registry manages lifecycle without interfering with existing FF dependency injection

## Integration Points with Existing FF Managers

### Text Chat → FF Storage Manager
- Message persistence and retrieval
- Session management and history
- User profile integration

### Memory Component → FF Vector Storage Manager  
- Embedding storage and similarity search
- Context retrieval and relevance scoring
- Memory consolidation and cleanup

### Multi-Agent → FF Panel Manager
- Agent persona management
- Conversation coordination
- Consensus building and decision making

### All Components → FF Search Manager
- Full-text search across conversations
- Content indexing and retrieval
- Query processing and ranking

## Notes

- This is Phase 2 of 4 - focus on core chat components only
- Phase 1 foundation must be working before starting Phase 2
- Don't implement advanced features yet (Phase 3) or API layer (Phase 4)
- Each component must demonstrate clear integration with existing FF managers
- Maintain backward compatibility - existing FF users should see no changes
- Test coverage should be comprehensive, following existing FF test patterns

Please implement Phase 2 following these specifications. Focus on one component at a time, testing integration with existing FF managers before moving to the next component. Let me know when Phase 2 is complete so I can verify the implementation before proceeding to Phase 3.