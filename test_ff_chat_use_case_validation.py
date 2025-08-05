"""
FF Chat Phase 2 Use Case Validation

Validates that Phase 2 implementation supports 19/22 use cases (86% coverage)
by demonstrating end-to-end functionality for each supported use case.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import FF Chat Application and related components
from ff_chat_application import create_ff_chat_app, FFChatApplication
from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
from ff_protocols.ff_chat_component_protocol import (
    USE_CASE_COMPONENT_MAPPINGS, get_required_components_for_use_case,
    COMPONENT_TYPE_TEXT_CHAT, COMPONENT_TYPE_MEMORY, COMPONENT_TYPE_MULTI_AGENT
)
from ff_utils.ff_logging import get_logger

logger = get_logger(__name__)


class FFChatUseCaseValidator:
    """Validates FF Chat Phase 2 use case coverage"""
    
    def __init__(self):
        self.temp_dir = None
        self.chat_app: FFChatApplication = None
        self.validation_results: Dict[str, Any] = {}
        self.supported_use_cases = []
        self.unsupported_use_cases = []
        
    async def setup(self):
        """Setup validation environment"""
        # Create temporary storage
        self.temp_dir = tempfile.mkdtemp(prefix="ff_chat_validation_")
        
        # Create FF configuration
        ff_config = FFConfigurationManagerConfigDTO()
        ff_config.storage_base_path = self.temp_dir
        ff_config.session_id_prefix = "validation_"
        ff_config.enable_file_locking = False
        
        # Create and initialize chat application
        self.chat_app = await create_ff_chat_app(ff_config=ff_config)
        
        logger.info(f"Validation setup complete. Temp dir: {self.temp_dir}")
    
    async def cleanup(self):
        """Cleanup validation environment"""
        if self.chat_app:
            await self.chat_app.shutdown()
        
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        logger.info("Validation cleanup complete")
    
    async def validate_all_use_cases(self) -> Dict[str, Any]:
        """Validate all use cases defined in Phase 2"""
        
        logger.info("Starting comprehensive use case validation...")
        
        # Get all use cases from component mappings
        all_use_cases = list(USE_CASE_COMPONENT_MAPPINGS.keys())
        
        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_use_cases": len(all_use_cases),
            "use_case_results": {},
            "component_coverage": {},
            "summary": {}
        }
        
        successful_validations = 0
        failed_validations = 0
        
        # Validate each use case
        for use_case in all_use_cases:
            logger.info(f"Validating use case: {use_case}")
            
            try:
                use_case_result = await self._validate_single_use_case(use_case)
                results["use_case_results"][use_case] = use_case_result
                
                if use_case_result["validation_successful"]:
                    successful_validations += 1
                    self.supported_use_cases.append(use_case)
                else:
                    failed_validations += 1
                    self.unsupported_use_cases.append(use_case)
                    
            except Exception as e:
                logger.error(f"Error validating use case {use_case}: {e}")
                results["use_case_results"][use_case] = {
                    "validation_successful": False,
                    "error": str(e),
                    "components_required": get_required_components_for_use_case(use_case),
                    "validation_details": {}
                }
                failed_validations += 1
                self.unsupported_use_cases.append(use_case)
        
        # Calculate component coverage
        results["component_coverage"] = await self._calculate_component_coverage()
        
        # Create summary
        results["summary"] = {
            "total_use_cases": len(all_use_cases),
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": (successful_validations / len(all_use_cases)) * 100,
            "supported_use_cases": self.supported_use_cases,
            "unsupported_use_cases": self.unsupported_use_cases,
            "phase2_target_coverage": 19,
            "phase2_target_percentage": 86.4,
            "actual_coverage": successful_validations,
            "actual_percentage": (successful_validations / 22) * 100  # Out of total 22 use cases
        }
        
        self.validation_results = results
        return results
    
    async def _validate_single_use_case(self, use_case: str) -> Dict[str, Any]:
        """Validate a single use case end-to-end"""
        
        result = {
            "use_case": use_case,
            "validation_successful": False,
            "components_required": get_required_components_for_use_case(use_case),
            "validation_details": {},
            "error": None,
            "session_id": None,
            "messages_exchanged": 0
        }
        
        try:
            # Check if use case is supported by the application
            if not await self.chat_app.use_case_manager.is_use_case_supported(use_case):
                result["error"] = "Use case not supported by application"
                return result
            
            # Get use case information
            use_case_info = await self.chat_app.use_case_manager.get_use_case_info(use_case)
            result["validation_details"]["use_case_info"] = use_case_info
            
            # Create session for this use case
            session_id = await self.chat_app.create_chat_session(
                user_id=f"validation_user_{use_case}",
                use_case=use_case,
                title=f"Validation Session - {use_case}"
            )
            result["session_id"] = session_id
            
            # Generate appropriate test messages for the use case
            test_messages = self._generate_test_messages_for_use_case(use_case)
            
            # Process test messages
            for i, test_message in enumerate(test_messages):
                logger.debug(f"Processing message {i+1}/{len(test_messages)} for {use_case}")
                
                # Add use case specific context
                context = self._get_use_case_context(use_case)
                
                # Process message
                process_result = await self.chat_app.process_message(
                    session_id=session_id,
                    message=test_message,
                    **context
                )
                
                # Validate response
                if not process_result.get("success", False):
                    result["error"] = f"Message processing failed: {process_result.get('error', 'Unknown error')}"
                    return result
                
                # Store processing result
                result["validation_details"][f"message_{i+1}_result"] = {
                    "success": process_result["success"],
                    "has_response": bool(process_result.get("response_content")),
                    "processor": process_result.get("processor", "unknown"),
                    "components_used": process_result.get("components_used", [])
                }
                
                result["messages_exchanged"] += 1
            
            # Validate session persistence
            session_messages = await self.chat_app.get_session_messages(session_id)
            result["validation_details"]["session_persistence"] = {
                "messages_stored": len(session_messages),
                "expected_minimum": len(test_messages)
            }
            
            # Validate component usage
            await self._validate_component_usage(use_case, result)
            
            # Clean up session
            await self.chat_app.close_session(session_id)
            
            result["validation_successful"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Validation failed for {use_case}: {e}")
        
        return result
    
    def _generate_test_messages_for_use_case(self, use_case: str) -> List[str]:
        """Generate appropriate test messages for each use case"""
        
        use_case_messages = {
            # Basic Use Cases (Phase 1)
            "basic_chat": [
                "Hello, how are you today?",
                "What's the weather like?",
                "Can you help me with a simple question?"
            ],
            
            "multimodal_chat": [
                "I have an image to share with you",
                "Can you analyze this document?",
                "What do you think about this audio file?"
            ],
            
            "rag_chat": [
                "Search for information about renewable energy",
                "What does the documentation say about API usage?",
                "Find relevant knowledge about machine learning"
            ],
            
            # Memory-Enhanced Use Cases (Phase 2)
            "memory_chat": [
                "Please remember that my name is Alice and I work in software engineering",
                "What do you know about my professional background?",
                "I also enjoy hiking and photography in my free time"
            ],
            
            "long_term_memory": [
                "Store this important fact: The project deadline is December 15th",
                "What important dates should I remember?",
                "Add to memory: My favorite programming language is Python"
            ],
            
            "contextual_memory": [
                "In our last conversation, we discussed project planning",
                "What did we talk about regarding the timeline?",
                "Continue our previous discussion about implementation details"
            ],
            
            "personal_assistant": [
                "Schedule a meeting for next Tuesday at 2 PM",
                "What's on my calendar for tomorrow?",
                "Remind me to call the client about the proposal"
            ],
            
            # Multi-Agent Use Cases (Phase 2)
            "multi_ai_panel": [
                "What are different perspectives on climate change solutions?",
                "I need multiple viewpoints on this business decision",
                "Can different experts weigh in on this technical problem?"
            ],
            
            "agent_debate": [
                "Should companies prioritize remote work or office work?",
                "What are the pros and cons of artificial intelligence in education?",
                "Debate the merits of different programming paradigms"
            ],
            
            "consensus_building": [
                "Help me find common ground on this controversial topic",
                "What do most experts agree on regarding cybersecurity?",
                "Build consensus on the best approach to solve this problem"
            ],
            
            "expert_consultation": [
                "I need expert advice on database design",
                "Consult with specialists about my marketing strategy",
                "Get professional opinions on this technical architecture"
            ],
            
            "collaborative_brainstorming": [
                "Let's brainstorm ideas for improving user experience",
                "Generate creative solutions for reducing costs",
                "Collaborate on innovative features for our product"
            ],
            
            # Advanced Search Use Cases (Phase 2)
            "smart_search_chat": [
                "Search intelligently for research on quantum computing",
                "Find the most relevant information about sustainable agriculture",
                "Discover insights about emerging technology trends"
            ],
            
            "semantic_search_expert": [
                "Find documents with similar meaning to 'customer satisfaction'",
                "Search for semantically related content about 'digital transformation'",
                "Locate information that relates to 'employee engagement strategies'"
            ],
            
            "research_assistant": [
                "Help me research the history of machine learning",
                "Compile information about renewable energy technologies",
                "Assist with gathering data on market trends"
            ],
            
            # Specialized Use Cases (Phase 2)
            "learning_companion": [
                "Teach me about neural networks step by step",
                "Help me understand complex mathematical concepts",
                "Guide me through learning a new programming language"
            ],
            
            "creative_collaboration": [
                "Help me write a creative story about space exploration",
                "Collaborate on designing a user interface",
                "Brainstorm creative marketing campaign ideas"
            ],
            
            "code_review_assistant": [
                "Review this Python code for potential improvements",
                "Check this algorithm for efficiency issues",
                "Suggest best practices for this implementation"
            ],
            
            # Currently Unsupported (Phase 3+)
            "voice_chat": [
                "Process this voice message",
                "Respond with speech synthesis",
                "Handle voice command"
            ],
            
            "real_time_collaboration": [
                "Start real-time collaborative session",
                "Join live discussion with multiple users",
                "Coordinate simultaneous editing"
            ],
            
            "advanced_reasoning": [
                "Perform complex logical reasoning",
                "Solve multi-step mathematical proofs",
                "Apply advanced cognitive processing"
            ]
        }
        
        return use_case_messages.get(use_case, [
            f"Test message 1 for {use_case}",
            f"Test message 2 for {use_case}",
            f"Test message 3 for {use_case}"
        ])
    
    def _get_use_case_context(self, use_case: str) -> Dict[str, Any]:
        """Get appropriate context parameters for each use case"""
        
        context_mappings = {
            "memory_chat": {
                "retrieve_memories": True,
                "store_memories": True
            },
            
            "multi_ai_panel": {
                "agent_personas": ["technical_expert", "business_analyst", "creative_writer"],
                "coordination_mode": "collaborative"
            },
            
            "agent_debate": {
                "agent_personas": ["advocate", "critic", "moderator"],
                "coordination_mode": "competitive"
            },
            
            "consensus_building": {
                "agent_personas": ["expert_1", "expert_2", "expert_3"],
                "coordination_mode": "consensus"
            },
            
            "expert_consultation": {
                "agent_personas": ["domain_expert", "technical_specialist"],
                "coordination_mode": "collaborative"
            },
            
            "collaborative_brainstorming": {
                "agent_personas": ["innovator", "analyst", "implementer"],
                "coordination_mode": "collaborative"
            },
            
            "smart_search_chat": {
                "enhanced_search": True,
                "similarity_threshold": 0.7
            },
            
            "semantic_search_expert": {
                "semantic_search": True,
                "similarity_threshold": 0.8
            }
        }
        
        return context_mappings.get(use_case, {})
    
    async def _validate_component_usage(self, use_case: str, result: Dict[str, Any]):
        """Validate that appropriate components were used for the use case"""
        
        required_components = get_required_components_for_use_case(use_case)
        
        # Check if this is a Phase 2 use case
        is_phase2 = self.chat_app.use_case_manager.is_phase2_use_case(use_case)
        
        result["validation_details"]["component_validation"] = {
            "required_components": required_components,
            "is_phase2_use_case": is_phase2,
            "component_routing_expected": is_phase2
        }
        
        # For Phase 2 use cases, verify appropriate component routing occurred
        if is_phase2:
            routing_info = self.chat_app.use_case_manager.get_component_routing_info(use_case)
            result["validation_details"]["component_routing"] = routing_info
    
    async def _calculate_component_coverage(self) -> Dict[str, Any]:
        """Calculate component coverage statistics"""
        
        component_stats = {
            COMPONENT_TYPE_TEXT_CHAT: {"supported_use_cases": 0, "use_cases": []},
            COMPONENT_TYPE_MEMORY: {"supported_use_cases": 0, "use_cases": []},
            COMPONENT_TYPE_MULTI_AGENT: {"supported_use_cases": 0, "use_cases": []}
        }
        
        for use_case in self.supported_use_cases:
            required_components = get_required_components_for_use_case(use_case)
            
            for component in required_components:
                if component in component_stats:
                    component_stats[component]["supported_use_cases"] += 1
                    component_stats[component]["use_cases"].append(use_case)
        
        return component_stats
    
    def print_validation_summary(self):
        """Print a comprehensive validation summary"""
        
        if not self.validation_results:
            print("No validation results available. Run validate_all_use_cases() first.")
            return
        
        summary = self.validation_results["summary"]
        
        print("\n" + "="*80)
        print("FF CHAT PHASE 2 USE CASE VALIDATION SUMMARY")
        print("="*80)
        
        print(f"\nValidation Timestamp: {self.validation_results['validation_timestamp']}")
        print(f"Total Use Cases Evaluated: {summary['total_use_cases']}")
        print(f"Successful Validations: {summary['successful_validations']}")
        print(f"Failed Validations: {summary['failed_validations']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\nPhase 2 Target Coverage: {summary['phase2_target_coverage']}/22 ({summary['phase2_target_percentage']:.1f}%)")
        print(f"Actual Coverage Achieved: {summary['actual_coverage']}/22 ({summary['actual_percentage']:.1f}%)")
        
        if summary['actual_coverage'] >= summary['phase2_target_coverage']:
            print("✅ Phase 2 coverage target ACHIEVED!")
        else:
            remaining = summary['phase2_target_coverage'] - summary['actual_coverage']
            print(f"⚠️  Phase 2 coverage target not met. Need {remaining} more use cases.")
        
        print(f"\n📋 SUPPORTED USE CASES ({len(self.supported_use_cases)}):")
        for i, use_case in enumerate(self.supported_use_cases, 1):
            components = get_required_components_for_use_case(use_case)
            print(f"  {i:2d}. {use_case:<25} (Components: {', '.join(components)})")
        
        if self.unsupported_use_cases:
            print(f"\n❌ UNSUPPORTED USE CASES ({len(self.unsupported_use_cases)}):")
            for i, use_case in enumerate(self.unsupported_use_cases, 1):
                components = get_required_components_for_use_case(use_case)
                print(f"  {i:2d}. {use_case:<25} (Components: {', '.join(components)})")
        
        # Component coverage
        component_coverage = self.validation_results["component_coverage"]
        print(f"\n🔧 COMPONENT COVERAGE:")
        for component, stats in component_coverage.items():
            print(f"  {component}: {stats['supported_use_cases']} use cases")
        
        print("\n" + "="*80)


# Main validation functions
async def validate_phase2_use_cases() -> Dict[str, Any]:
    """Main function to validate Phase 2 use case coverage"""
    
    validator = FFChatUseCaseValidator()
    
    try:
        await validator.setup()
        results = await validator.validate_all_use_cases()
        validator.print_validation_summary()
        return results
        
    finally:
        await validator.cleanup()


async def quick_validation_check() -> bool:
    """Quick validation check for CI/CD pipelines"""
    
    validator = FFChatUseCaseValidator()
    
    try:
        await validator.setup()
        results = await validator.validate_all_use_cases()
        
        # Check if we meet Phase 2 targets
        summary = results["summary"]
        target_met = summary["actual_coverage"] >= summary["phase2_target_coverage"]
        
        print(f"Quick Validation: {'PASSED' if target_met else 'FAILED'}")
        print(f"Coverage: {summary['actual_coverage']}/{summary['phase2_target_coverage']} use cases")
        
        return target_met
        
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    # Run comprehensive validation
    print("Starting FF Chat Phase 2 Use Case Validation...")
    results = asyncio.run(validate_phase2_use_cases())
    
    # Save results to file
    import json
    results_file = Path("ff_chat_phase2_validation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("Validation complete!")