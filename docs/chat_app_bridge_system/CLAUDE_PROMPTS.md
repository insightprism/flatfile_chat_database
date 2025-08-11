# Claude Code Prompts - Chat Application Bridge System

## How to Use These Prompts

For each phase implementation:
1. **Copy the prompt** for the specific phase below
2. **Provide these files** to Claude Code:
   - `CLAUDE_CONTEXT.md` (always include)
   - The specific `PHASE_*.md` file for that phase
   - Any additional files mentioned in the prompt
3. **Paste the prompt** and let Claude Code implement the phase

---

## Phase 1 Prompt: Bridge Infrastructure

```
You are implementing Phase 1 of the Chat Application Bridge System for the Flatfile Database project. 

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- PHASE_1_BRIDGE_INFRASTRUCTURE.md (detailed implementation specification)

**Your Task:**
Implement Phase 1: Bridge Infrastructure and Exception Handling according to the specification. This includes:

1. Create the complete exception hierarchy with ChatIntegrationError as the base class
2. Set up the ff_chat_integration module structure with proper __init__.py
3. Create all base classes and interfaces defined in the specification
4. Implement the module exports and version management
5. Add comprehensive docstrings and type hints throughout
6. Create the validation scripts specified in the phase documentation

**Key Requirements:**
- Follow the exact module structure specified in PHASE_1_BRIDGE_INFRASTRUCTURE.md
- Implement all exception classes with proper context and suggestions
- Create the base interfaces that other phases will build upon
- Ensure all code follows Python best practices and includes comprehensive error handling
- Add validation scripts to verify the phase implementation

**Success Criteria:**
- All exception classes created with proper inheritance
- Module structure matches specification exactly
- Validation scripts pass all tests
- Code is well-documented with docstrings and type hints
- Ready for Phase 2 to build upon this foundation

Please implement this phase completely and run the validation scripts to confirm successful implementation.
```

---

## Phase 2 Prompt: Bridge Implementation

```
You are implementing Phase 2 of the Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context) 
- PHASE_2_BRIDGE_IMPLEMENTATION.md (detailed implementation specification)

**Prerequisites:**
Phase 1 must be completed first. If not completed, please implement Phase 1 before proceeding with Phase 2.

**Your Task:**
Implement Phase 2: Bridge Implementation and Factory Methods according to the specification. This includes:

1. Create the FFChatAppBridge main class with factory methods
2. Implement ChatAppStorageConfig with validation and performance presets
3. Add direct integration with existing StorageManager (no wrapper classes)
4. Create factory methods: create_for_chat_app() and create_from_preset()
5. Implement bridge lifecycle management (initialization, health checks, cleanup)
6. Add comprehensive error handling and performance tracking

**Key Requirements:**
- Use existing StorageManager directly - do not create wrapper classes
- Implement one-line bridge creation as specified
- Add performance presets (speed, balanced, quality modes)
- Include comprehensive validation for all configuration parameters
- Ensure proper async/await patterns throughout
- Add standardized response format for all operations

**Integration Points:**
- Import and use existing StorageManager, Session, Message, UserProfile classes
- Extend StorageConfig to create ChatAppStorageConfig
- Maintain full backward compatibility with existing code

**Success Criteria:**
- Bridge can be created with single-line factory methods
- All configuration presets work correctly
- Integration with existing Flatfile storage is seamless
- Performance metrics are tracked for all operations
- Validation scripts pass all tests
- Ready for Phase 3 data layer implementation

Please implement this phase completely, ensuring proper integration with the existing codebase.
```

---

## Phase 3 Prompt: Data Layer Implementation

```
You are implementing Phase 3 of the Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- PHASE_3_DATA_LAYER.md (detailed implementation specification)

**Prerequisites:**
Phases 1 and 2 must be completed first. If not completed, please implement them before proceeding with Phase 3.

**Your Task:**
Implement Phase 3: Chat-Optimized Data Access Layer according to the specification. This includes:

1. Create the FFChatDataLayer class with chat-optimized operations
2. Implement standardized response format with performance metrics
3. Add caching, streaming, and performance optimization features
4. Create methods: store_chat_message, get_chat_history, search_conversations, stream_conversation
5. Implement intelligent caching with configurable cache sizes
6. Add performance analytics and optimization recommendations

**Key Requirements:**
- All operations must return standardized response format with success/error/data/metadata
- Include performance metrics (execution time, cache hits, etc.) in all responses
- Implement efficient caching with TTL and size limits
- Add streaming support for large conversations
- Ensure all operations are async and properly handle errors
- Meet performance targets: <50ms for storage, <75ms for retrieval

**Performance Targets:**
- Message storage: <50ms average, >500 ops/sec throughput
- Message retrieval: <75ms average, >300 ops/sec throughput  
- Search operations: <150ms average, >100 ops/sec throughput
- Cache hit rate: >70% for repeated operations

**Success Criteria:**
- All chat operations work with standardized response format
- Performance targets are met or exceeded
- Caching system works efficiently with configurable parameters
- Streaming works for large conversation histories
- Integration with Phase 2 bridge is seamless
- Validation scripts pass all performance tests

Please implement this phase with focus on performance optimization and standardized interfaces.
```

---

## Phase 4 Prompt: Configuration Factory

```
You are implementing Phase 4 of the Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- PHASE_4_CONFIG_FACTORY.md (detailed implementation specification)

**Prerequisites:**
Phases 1, 2, and 3 must be completed first. If not completed, please implement them before proceeding with Phase 4.

**Your Task:**
Implement Phase 4: Configuration Factory and Presets according to the specification. This includes:

1. Create the FFChatConfigFactory class with template system
2. Implement configuration presets (development, production, high_performance, lightweight)
3. Create JSON template files for each preset configuration
4. Add migration utilities from wrapper-based configurations
5. Implement configuration validation, optimization, and analysis tools
6. Create environment-specific configuration management

**Key Requirements:**
- Create complete template system with JSON configuration files
- Implement preset validation with detailed error reporting
- Add configuration optimization recommendations
- Create migration tools for existing wrapper-based setups
- Support environment-specific configurations (dev, staging, prod)
- Include comprehensive configuration analysis and suggestions

**Configuration Presets to Implement:**
- **development**: Fast iteration with debugging features
- **production**: Optimized for stability and performance
- **high_performance**: Maximum speed and throughput
- **lightweight**: Minimal resource usage
- **testing**: Optimized for test environments

**Success Criteria:**
- All configuration presets work correctly and meet their design goals
- Template system allows easy customization and extension
- Migration utilities successfully convert wrapper-based configurations
- Configuration validation provides helpful error messages and suggestions
- Performance optimization recommendations are accurate
- Integration with Phases 1-3 is seamless

Please implement this phase with focus on ease of use and comprehensive preset coverage.
```

---

## Phase 5 Prompt: Health Monitoring

```
You are implementing Phase 5 of the Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- PHASE_5_HEALTH_MONITORING.md (detailed implementation specification)

**Prerequisites:**
Phases 1, 2, 3, and 4 must be completed first. If not completed, please implement them before proceeding with Phase 5.

**Your Task:**
Implement Phase 5: Health Monitoring and Diagnostics according to the specification. This includes:

1. Create the FFIntegrationHealthMonitor class with comprehensive diagnostics
2. Implement health check system with detailed status reporting
3. Add performance analytics with optimization recommendations
4. Create proactive issue detection and alerting
5. Implement background monitoring capabilities
6. Add comprehensive diagnostic tools for troubleshooting

**Key Requirements:**
- Implement multi-level health checks: basic, comprehensive, and continuous
- Add performance analytics with trend analysis and recommendations
- Create proactive issue detection for common problems
- Include system resource monitoring (memory, disk, performance)
- Implement alerting system for threshold violations
- Add diagnostic tools for troubleshooting integration issues

**Health Check Levels:**
- **Basic**: Quick status check (bridge, storage, permissions)
- **Comprehensive**: Detailed analysis with performance metrics
- **Continuous**: Background monitoring with trend analysis

**Monitoring Capabilities:**
- Performance metrics tracking and analysis
- Resource usage monitoring (memory, disk, CPU)
- Cache efficiency and optimization recommendations
- Error rate monitoring and alerting
- Configuration analysis and optimization suggestions

**Success Criteria:**
- Health monitoring provides accurate system status
- Performance analytics identify optimization opportunities
- Proactive issue detection prevents problems before they occur
- Background monitoring works efficiently without impacting performance
- Diagnostic tools provide actionable troubleshooting information
- Integration with all previous phases works seamlessly

Please implement this phase with focus on proactive monitoring and actionable diagnostics.
```

---

## Phase 6 Prompt: Testing and Validation

```
You are implementing Phase 6 of the Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- PHASE_6_TESTING_VALIDATION.md (detailed implementation specification)
- PERFORMANCE_REQUIREMENTS.md (detailed performance benchmarking requirements)

**Prerequisites:**
Phases 1, 2, 3, 4, and 5 must be completed first. If not completed, please implement them before proceeding with Phase 6.

**Your Task:**
Implement Phase 6: Testing, Documentation, and Validation according to the specification. This includes:

1. Create comprehensive test suite (unit, integration, performance, end-to-end)
2. Implement performance benchmarks validating 30% improvement claims
3. Add production readiness validation
4. Create migration testing and validation tools
5. Implement continuous integration test pipeline
6. Add comprehensive documentation and examples

**Key Requirements:**
- **Unit Tests**: Test each component in isolation with >90% code coverage
- **Integration Tests**: Test complete workflows with real Flatfile storage
- **Performance Tests**: Validate 30% improvement over wrapper-based approaches
- **End-to-End Tests**: Complete user scenarios and production workflows
- **Migration Tests**: Validate smooth migration from wrapper-based setups

**Performance Validation:**
- Benchmark against wrapper-based implementations
- Prove 30% performance improvement across all operations
- Validate sub-100ms response times for 95% of operations
- Confirm memory usage stays below 200MB for typical workloads
- Achieve >70% cache hit rates in performance tests

**Success Criteria:**
- All tests pass with >90% code coverage
- Performance benchmarks demonstrate 30% improvement
- Migration from wrapper-based configurations works seamlessly
- Production readiness validation confirms system is deployment-ready
- Documentation is comprehensive and includes working examples
- CI/CD pipeline is functional and catches regressions

**Deliverables:**
- Complete test suite with all test types
- Performance benchmark results proving 30% improvement
- Production readiness checklist and validation
- Migration guides and automated tools
- Comprehensive documentation and examples

Please implement this phase with focus on thorough validation and production readiness.
```

---

## Complete System Validation Prompt

```
You are performing final validation of the complete Chat Application Bridge System for the Flatfile Database project.

**Context Files Provided:**
- CLAUDE_CONTEXT.md (project overview and context)
- All PHASE_*.md files (for reference)
- PERFORMANCE_REQUIREMENTS.md (benchmarking requirements)

**Prerequisites:**
All phases (1-6) must be completed. This is the final validation step.

**Your Task:**
Perform comprehensive system validation to ensure the entire Chat Application Bridge System meets all requirements:

1. **Functional Validation**: Test all components work together correctly
2. **Performance Validation**: Confirm 30% improvement over wrapper-based approaches
3. **Integration Validation**: Ensure seamless integration with existing Flatfile codebase
4. **Production Readiness**: Validate system is ready for production deployment
5. **Documentation Validation**: Ensure all documentation is complete and accurate

**Key Validation Points:**
- One-line bridge creation works: `bridge = await FFChatAppBridge.create_for_chat_app("./data")`
- All preset configurations work correctly
- Performance targets are met (30% improvement, <100ms response times)
- Memory usage stays below 200MB for typical workloads
- Cache hit rates exceed 70%
- Health monitoring provides accurate diagnostics
- Migration from wrapper-based configurations works smoothly

**Success Criteria:**
- ✅ 30% performance improvement demonstrated
- ✅ 95% integration success rate achieved
- ✅ Sub-100ms response times for core operations
- ✅ Memory efficiency below 200MB for typical workloads
- ✅ All tests pass with >90% coverage
- ✅ Production deployment readiness confirmed

**Final Deliverable:**
A comprehensive validation report showing all success criteria have been met and the system is ready for production use.

Please perform this final validation and provide a detailed report on system readiness.
```

---

## Quick Reference

### Standard Files to Always Include:
1. **CLAUDE_CONTEXT.md** - Always provide this for project context
2. **Specific PHASE_*.md** - The phase being implemented
3. **Additional files** as mentioned in each prompt

### Typical Implementation Flow:
1. Copy the appropriate phase prompt
2. Provide the context file and phase specification
3. Let Claude Code implement the phase
4. Run validation scripts to confirm success
5. Move to next phase

### If Previous Phases Are Missing:
Each prompt includes prerequisite checks. Claude Code will implement missing phases before proceeding with the current phase.

These prompts ensure consistent, complete implementation across all context window restarts while maintaining focus on the specific phase requirements and success criteria.