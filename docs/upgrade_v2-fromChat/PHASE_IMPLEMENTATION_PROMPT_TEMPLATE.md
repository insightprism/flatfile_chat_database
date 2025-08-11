# Phase Implementation Prompt Template

## 🎯 Universal Prompt for All Phases

Copy and customize this prompt for each phase implementation. Replace `{PHASE_NUMBER}`, `{PHASE_NAME}`, and `{COMPONENT_NAME}` with the appropriate values.

---

## 📋 **Phase {PHASE_NUMBER}: {PHASE_NAME} Implementation**

I need you to implement **Phase {PHASE_NUMBER}: {PHASE_NAME}** for my flat file chat database upgrade to support advanced PrismMind capabilities.

### **📎 Required Files (Must Read Both)**
**CRITICAL**: You must read both attached files before starting implementation:

1. **`IMPLEMENTATION_CONTEXT.md`** - Complete project context, architectural standards, and implementation patterns
2. **`PHASE_{PHASE_NUMBER}_{COMPONENT_NAME}.md`** - Detailed specification for this specific phase

### **🎯 Your Implementation Task**

**Step 1: Context Understanding**
- Read `IMPLEMENTATION_CONTEXT.md` thoroughly to understand:
  - Existing system architecture (what NOT to break)
  - Mandatory architectural principles and patterns
  - Required code standards and conventions
  - Integration guidelines and dependencies

**Step 2: Phase Specification Review**
- Read `PHASE_{PHASE_NUMBER}_{COMPONENT_NAME}.md` completely to understand:
  - Specific requirements for this phase
  - Data models and DTOs to implement
  - Manager classes and their functionality
  - Protocol interfaces required
  - Integration specifications
  - Testing requirements

**Step 3: Complete Implementation**
Create ALL specified files with complete, production-ready implementations:

**Configuration Files:**
- [ ] `ff_class_configs/ff_{component}_config.py` - Configuration DTOs following established patterns
- [ ] Extend `ff_class_configs/ff_configuration_manager_config.py` - Add new component config
- [ ] Extend `ff_class_configs/ff_chat_entities_config.py` - Add new entity DTOs

**Manager Implementation:**
- [ ] `ff_{component}_manager.py` - Main manager class following flatfile patterns
- [ ] Follow exact patterns from IMPLEMENTATION_CONTEXT.md
- [ ] Include comprehensive error handling and logging
- [ ] Implement all methods specified in the phase specification

**Protocol Interface:**
- [ ] `ff_protocols/ff_{component}_protocol.py` - Abstract interface for dependency injection
- [ ] Define all required operations as abstract methods

**Unit Tests:**
- [ ] `tests/test_{component}_manager.py` - Comprehensive unit test suite
- [ ] Follow testing patterns from IMPLEMENTATION_CONTEXT.md
- [ ] Achieve >90% test coverage
- [ ] Include both positive and negative test cases

### **🚨 CRITICAL Requirements (MUST FOLLOW)**

**Backward Compatibility:**
- ✅ NEVER modify existing files except at designated extension points
- ✅ NEVER change existing function signatures or data structures
- ✅ ALWAYS ensure existing functionality continues working unchanged
- ✅ Test that current system features still work after your changes

**Architecture Compliance:**
- ✅ Follow EXACT DTO patterns from IMPLEMENTATION_CONTEXT.md
- ✅ Use established manager implementation pattern
- ✅ Implement protocol interfaces for dependency injection
- ✅ Follow atomic file operations using ff_file_ops utilities
- ✅ Use proper async/await patterns throughout

**Code Quality Standards:**
- ✅ Comprehensive docstrings for ALL functions, classes, and methods
- ✅ Full type hints for all function signatures
- ✅ Structured error handling with proper logging
- ✅ Resource cleanup in finally blocks
- ✅ Follow the exact naming conventions shown in context

**Integration Requirements:**
- ✅ Integrate with existing components as specified
- ✅ Use existing utilities (ff_file_ops, ff_json_utils, ff_logging)
- ✅ Follow established directory structure patterns
- ✅ Maintain user-based data isolation

### **📋 Implementation Checklist**

**Before Starting:**
- [ ] Read IMPLEMENTATION_CONTEXT.md completely
- [ ] Read phase specification completely
- [ ] Understand integration points with existing system
- [ ] Plan file structure and naming

**During Implementation:**
- [ ] Create configuration DTOs first (data contracts)
- [ ] Implement protocol interfaces (operation contracts)
- [ ] Build manager classes (core functionality)
- [ ] Extend existing configuration files
- [ ] Add comprehensive error handling and logging
- [ ] Write detailed docstrings and type hints

**After Implementation:**
- [ ] Create comprehensive unit tests
- [ ] Verify existing functionality still works
- [ ] Test all new capabilities
- [ ] Validate integration points
- [ ] Ensure performance requirements met

### **🔧 File Creation Order**

Implement files in this specific order for best results:

1. **Configuration DTOs** (`ff_class_configs/ff_{component}_config.py`)
2. **Entity DTOs** (extend `ff_class_configs/ff_chat_entities_config.py`)
3. **Protocol Interface** (`ff_protocols/ff_{component}_protocol.py`)
4. **Manager Implementation** (`ff_{component}_manager.py`)
5. **Configuration Extension** (extend `ff_configuration_manager_config.py`)
6. **Unit Tests** (`tests/test_{component}_manager.py`)

### **✅ Success Criteria**

Your implementation is successful when:
- [ ] All existing system functionality works unchanged (backward compatibility)
- [ ] All new functionality meets phase specification requirements
- [ ] Code follows established architectural patterns exactly
- [ ] Unit tests pass with >90% coverage
- [ ] Integration with existing components works seamlessly
- [ ] Performance meets or exceeds current system benchmarks
- [ ] All files follow naming and organization conventions

### **🎯 Expected Deliverables**

When complete, provide:
1. **Summary of implemented files** with brief description of each
2. **Integration points** identified and implemented
3. **Test results** showing all tests pass
4. **Backward compatibility confirmation** that existing features work
5. **Next steps** for integrating with subsequent phases

### **📞 Implementation Notes**

- Use TodoWrite tool to track your progress through the implementation
- If you encounter any ambiguity, refer back to the IMPLEMENTATION_CONTEXT.md patterns
- Focus on one file at a time for thorough, quality implementation
- Test each component as you build it
- Ask for clarification if any requirements are unclear

**Ready to implement Phase {PHASE_NUMBER}? Please confirm you've read both attached files and understand the requirements before beginning.**

---

## 🔄 **Phase-Specific Customizations**

### **Phase 1: Multi-Layered Memory System**
```
Replace placeholders with:
- {PHASE_NUMBER} = 1
- {PHASE_NAME} = Multi-Layered Memory System
- {COMPONENT_NAME} = MEMORY_LAYERS
```

### **Phase 2: Panel Session Enhancement**
```
Replace placeholders with:
- {PHASE_NUMBER} = 2  
- {PHASE_NAME} = Panel Session Enhancement
- {COMPONENT_NAME} = PANEL_SESSIONS
```

### **Phase 3: RAG Integration**
```
Replace placeholders with:
- {PHASE_NUMBER} = 3
- {PHASE_NAME} = RAG Integration
- {COMPONENT_NAME} = RAG_INTEGRATION
```

### **Phase 4: Tool Execution Framework**
```
Replace placeholders with:
- {PHASE_NUMBER} = 4
- {PHASE_NAME} = Tool Execution Framework
- {COMPONENT_NAME} = TOOL_EXECUTION
```

### **Phase 5: Analytics & Monitoring**
```
Replace placeholders with:
- {PHASE_NUMBER} = 5
- {PHASE_NAME} = Analytics & Monitoring System
- {COMPONENT_NAME} = ANALYTICS_MONITORING
```

### **Phase 6: Integration & Testing**
```
Replace placeholders with:
- {PHASE_NUMBER} = 6
- {PHASE_NAME} = Integration & Testing
- {COMPONENT_NAME} = INTEGRATION_TESTING
```

## 📋 **Usage Instructions**

1. **Copy the universal prompt** from the top section
2. **Replace the placeholders** with values from the appropriate phase section
3. **Attach the two required files:**
   - `IMPLEMENTATION_CONTEXT.md`
   - `PHASE_{NUMBER}_{COMPONENT}.md`
4. **Paste the customized prompt** into Claude Code
5. **Wait for implementation** to complete
6. **Validate results** before moving to next phase

This template ensures consistent, high-quality implementation across all phases while maintaining full context and architectural compliance.