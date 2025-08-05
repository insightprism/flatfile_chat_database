"""
FF Multi-Agent Component - Phase 2 Implementation

Coordinates multiple AI agents using existing FF panel manager for
complex discussions and collaborative responses. Supports 5/22 use cases (23% coverage).
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import existing FF infrastructure
from ff_core_storage_manager import FFCoreStorageManager
from ff_panel_manager import FFPanelManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_class_configs.ff_multi_agent_config import (
    FFMultiAgentConfigDTO, FFMultiAgentUseCaseConfigDTO,
    FFCoordinationMode, FFAgentSelectionStrategy, FFConflictResolution
)
from ff_class_configs.ff_chat_entities_config import FFMessageDTO
from ff_protocols.ff_chat_component_protocol import (
    FFMultiAgentComponentProtocol, FFComponentInfo, FFComponentCapability,
    COMPONENT_TYPE_MULTI_AGENT, get_use_cases_for_component
)
from ff_utils.ff_logging import get_logger


@dataclass
class FFAgentResponse:
    """Represents a response from an individual agent"""
    agent_id: str
    agent_persona: str
    response_content: str
    confidence_score: float
    response_time: float
    metadata: Dict[str, Any]


@dataclass
class FFAgentPanel:
    """Represents a multi-agent panel"""
    panel_id: str
    session_id: str
    user_id: str
    agent_personas: List[str]
    coordination_mode: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    active_agents: List[str]


class FFMultiAgentComponent(FFMultiAgentComponentProtocol):
    """
    FF Multi-Agent Component providing multi-agent coordination.
    
    Uses existing FF panel manager for agent coordination and supports
    5/22 use cases requiring multi-agent functionality.
    """
    
    def __init__(self, config: FFMultiAgentConfigDTO):
        """
        Initialize FF Multi-Agent Component.
        
        Args:
            config: Multi-agent component configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services (initialized via dependencies)
        self.ff_storage: Optional[FFCoreStorageManager] = None
        self.ff_panel: Optional[FFPanelManager] = None
        self.ff_search: Optional[FFSearchManager] = None
        self.ff_vector: Optional[FFVectorStorageManager] = None
        
        # Component state
        self._initialized = False
        self._component_info = self._create_component_info()
        
        # Panel management
        self._active_panels: Dict[str, FFAgentPanel] = {}
        self._agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Response coordination
        self._response_cache: Dict[str, Any] = {}
        self._coordination_locks: Dict[str, asyncio.Lock] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._performance_tracking_task: Optional[asyncio.Task] = None
        
        # Processing statistics
        self._multi_agent_stats = {
            "total_panels_created": 0,
            "total_messages_processed": 0,
            "total_agent_responses": 0,
            "consensus_achieved": 0,
            "conflicts_resolved": 0,
            "average_response_time": 0.0,
            "average_agents_per_panel": 0.0
        }
    
    @property
    def component_info(self) -> Dict[str, Any]:
        """Get component metadata and capabilities"""
        return self._component_info.to_dict()
    
    def _create_component_info(self) -> FFComponentInfo:
        """Create component information structure"""
        capabilities = [
            FFComponentCapability(
                name="agent_panel_coordination",
                description="Create and manage multi-agent panels using FF panel manager",
                parameters={
                    "max_agents": self.config.max_agents,
                    "coordination_modes": [mode.value for mode in FFCoordinationMode],
                    "conflict_resolution": [res.value for res in FFConflictResolution]
                },
                ff_dependencies=["ff_panel", "ff_storage"]
            ),
            FFComponentCapability(
                name="multi_agent_processing",
                description="Process messages through multiple agents with coordination",
                parameters={
                    "response_timeout": self.config.response_timeout,
                    "max_response_rounds": self.config.max_response_rounds,
                    "consensus_threshold": self.config.consensus_threshold
                },
                ff_dependencies=["ff_panel", "ff_storage"]
            ),
            FFComponentCapability(
                name="agent_performance_tracking",
                description="Track and optimize individual agent performance",
                parameters={
                    "performance_metrics": self.config.performance_metrics,
                    "performance_window": self.config.agent_performance_window
                },
                ff_dependencies=["ff_storage"]
            ),
            FFComponentCapability(
                name="consensus_building",
                description="Build consensus among multiple agents",
                parameters={
                    "voting_strategies": ["equal", "expertise", "performance"],
                    "conflict_resolution_methods": [res.value for res in FFConflictResolution]
                },
                ff_dependencies=["ff_panel"]
            )
        ]
        
        supported_use_cases = get_use_cases_for_component(COMPONENT_TYPE_MULTI_AGENT)
        
        return FFComponentInfo(
            name="ff_multi_agent",
            version="2.0.0",
            description="FF Multi-Agent Component for agent coordination using FF panel manager backend",
            capabilities=capabilities,
            use_cases=supported_use_cases,
            ff_dependencies=["ff_panel", "ff_storage", "ff_search", "ff_vector"],
            priority=80
        )
    
    async def initialize(self, dependencies: Dict[str, Any]) -> bool:
        """
        Initialize component with FF backend services.
        
        Args:
            dependencies: Dictionary containing FF manager instances
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing FF Multi-Agent Component...")
            
            # Extract FF backend services
            self.ff_storage = dependencies.get("ff_storage")
            self.ff_panel = dependencies.get("ff_panel")
            self.ff_search = dependencies.get("ff_search")
            self.ff_vector = dependencies.get("ff_vector")
            
            # Validate required dependencies
            if not self.ff_storage:
                raise ValueError("ff_storage dependency is required")
            
            if self.config.use_ff_panel_manager and not self.ff_panel:
                raise ValueError("ff_panel dependency is required when use_ff_panel_manager is enabled")
            
            # Test FF backend connections
            if not await self._test_ff_backend_connections():
                raise RuntimeError("Failed to connect to FF backend services")
            
            # Initialize performance tracking
            if self.config.enable_agent_performance_tracking:
                await self._initialize_performance_tracking()
            
            # Initialize coordination systems
            await self._initialize_coordination_systems()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            self.logger.info("FF Multi-Agent Component initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FF Multi-Agent Component: {e}")
            return False
    
    async def _test_ff_backend_connections(self) -> bool:
        """Test connections to FF backend services"""
        try:
            # Test FF storage
            test_user_id = "ff_multi_agent_test_user"
            test_session_name = "FF Multi-Agent Component Test"
            session_id = await self.ff_storage.create_session(test_user_id, test_session_name)
            if not session_id:
                return False
            
            # Test FF panel manager if enabled
            if self.config.use_ff_panel_manager and self.ff_panel:
                # Test panel creation functionality
                test_panel_id = await self.ff_panel.create_panel(
                    user_id=test_user_id,
                    panel_name="FF Multi-Agent Test Panel",
                    personas=["test_agent_1", "test_agent_2"]
                )
                if not test_panel_id:
                    return False
            
            self.logger.debug("FF backend connections test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"FF backend connections test failed: {e}")
            return False
    
    async def _initialize_performance_tracking(self) -> None:
        """Initialize agent performance tracking"""
        self._agent_performance = {}
        self.logger.debug("Agent performance tracking initialized")
    
    async def _initialize_coordination_systems(self) -> None:
        """Initialize coordination systems"""
        self._coordination_locks = {}
        self._response_cache = {}
        self.logger.debug("Coordination systems initialized")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_task_loop())
        
        # Start performance tracking task if enabled
        if self.config.enable_agent_performance_tracking:
            self._performance_tracking_task = asyncio.create_task(self._performance_tracking_loop())
        
        self.logger.debug("Background tasks started")
    
    async def _cleanup_task_loop(self) -> None:
        """Background cleanup task loop"""
        while self._initialized:
            try:
                await self._cleanup_inactive_panels()
                await self._cleanup_coordination_locks()
                
                await asyncio.sleep(self.config.panel_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self) -> None:
        """Background performance tracking task loop"""
        while self._initialized:
            try:
                await self._update_agent_performance_metrics()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking task: {e}")
                await asyncio.sleep(60)
    
    async def process_message(self, 
                              session_id: str,
                              user_id: str,
                              message: FFMessageDTO,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process chat message through multi-agent coordination.
        
        Args:
            session_id: FF storage session identifier
            user_id: User identifier
            message: FF message DTO with content and metadata
            context: Optional processing context and parameters
            
        Returns:
            Processing results dictionary
        """
        if not self._initialized:
            return {
                "success": False,
                "error": "Component not initialized",
                "component": "ff_multi_agent"
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.debug(f"Processing multi-agent message in session {session_id}")
            
            # Get or create agent panel
            panel_id = context.get("panel_id")
            if not panel_id:
                panel_id = await self.create_agent_panel(
                    session_id=session_id,
                    user_id=user_id,
                    agent_personas=context.get("agent_personas", ["general_assistant", "specialist"]),
                    coordination_mode=context.get("coordination_mode", self.config.default_coordination_mode)
                )
            
            # Process message through multiple agents
            agent_responses = await self.process_multi_agent_message(
                panel_id=panel_id,
                message=message,
                agent_config=context.get("agent_config", {})
            )
            
            # Coordinate responses
            final_response = await self.coordinate_agents(
                agents=[resp.agent_id for resp in agent_responses],
                coordination_mode=self._active_panels[panel_id].coordination_mode,
                context={
                    "agent_responses": agent_responses,
                    "message": message,
                    "session_id": session_id
                }
            )
            
            # Store interaction in FF storage
            await self._store_multi_agent_interaction(session_id, user_id, message, agent_responses, final_response)
            
            processing_time = time.time() - start_time
            self._multi_agent_stats["total_messages_processed"] += 1
            self._multi_agent_stats["total_agent_responses"] += len(agent_responses)
            
            # Update average response time
            current_avg = self._multi_agent_stats["average_response_time"]
            total_processed = self._multi_agent_stats["total_messages_processed"]
            self._multi_agent_stats["average_response_time"] = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
            
            result = {
                "success": True,
                "response_content": final_response.get("final_response", ""),
                "component": "ff_multi_agent",
                "processor": "ff_panel_backend",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "panel_id": panel_id,
                    "agent_responses_count": len(agent_responses),
                    "coordination_mode": self._active_panels[panel_id].coordination_mode,
                    "consensus_achieved": final_response.get("consensus_achieved", False),
                    "processing_time": processing_time,
                    "individual_responses": [
                        {
                            "agent_id": resp.agent_id,
                            "agent_persona": resp.agent_persona,
                            "response_preview": resp.response_content[:100] + "..." if len(resp.response_content) > 100 else resp.response_content,
                            "confidence_score": resp.confidence_score
                        }
                        for resp in agent_responses
                    ]
                }
            }
            
            self.logger.debug(f"Successfully processed multi-agent message in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Error processing multi-agent message: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_multi_agent",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": processing_time
                }
            }
    
    async def create_agent_panel(self,
                                 session_id: str,
                                 user_id: str,
                                 agent_personas: List[str],
                                 coordination_mode: str = "round_robin") -> str:
        """
        Create agent panel using FF panel manager.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            agent_personas: List of agent personas to create
            coordination_mode: Agent coordination mode
            
        Returns:
            Panel identifier
        """
        try:
            # Validate agent count
            if len(agent_personas) < self.config.min_agents:
                raise ValueError(f"Minimum {self.config.min_agents} agents required")
            
            if len(agent_personas) > self.config.max_agents:
                agent_personas = agent_personas[:self.config.max_agents]
                self.logger.warning(f"Trimmed agent list to maximum {self.config.max_agents} agents")
            
            # Create panel using FF panel manager
            panel_name = f"Chat Panel - {session_id}"
            ff_panel_id = await self.ff_panel.create_panel(
                user_id=user_id,
                panel_name=panel_name,
                personas=agent_personas
            )
            
            if not ff_panel_id:
                raise RuntimeError("Failed to create panel with FF panel manager")
            
            # Create our panel tracking structure
            panel_id = f"multi_agent_panel_{uuid.uuid4().hex[:8]}"
            
            panel = FFAgentPanel(
                panel_id=panel_id,
                session_id=session_id,
                user_id=user_id,
                agent_personas=agent_personas,
                coordination_mode=coordination_mode,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                message_count=0,
                active_agents=agent_personas.copy()
            )
            
            self._active_panels[panel_id] = panel
            self._coordination_locks[panel_id] = asyncio.Lock()
            
            # Initialize agent performance tracking for this panel
            for persona in agent_personas:
                if persona not in self._agent_performance:
                    self._agent_performance[persona] = {
                        "total_responses": 0,
                        "average_response_time": 0.0,
                        "average_confidence": 0.0,
                        "consensus_contributions": 0,
                        "quality_score": 0.5
                    }
            
            self._multi_agent_stats["total_panels_created"] += 1
            
            # Update average agents per panel
            total_panels = self._multi_agent_stats["total_panels_created"]
            current_avg = self._multi_agent_stats["average_agents_per_panel"]
            self._multi_agent_stats["average_agents_per_panel"] = ((current_avg * (total_panels - 1)) + len(agent_personas)) / total_panels
            
            self.logger.info(f"Created agent panel {panel_id} with {len(agent_personas)} agents")
            return panel_id
            
        except Exception as e:
            self.logger.error(f"Failed to create agent panel: {e}")
            raise
    
    async def process_multi_agent_message(self,
                                          panel_id: str,
                                          message: FFMessageDTO,
                                          agent_config: Dict[str, Any]) -> List[FFAgentResponse]:
        """
        Process message through multiple agents.
        
        Args:
            panel_id: Panel identifier
            message: Message to process
            agent_config: Agent configuration parameters
            
        Returns:
            List of agent responses
        """
        if panel_id not in self._active_panels:
            raise ValueError(f"Panel {panel_id} not found")
        
        panel = self._active_panels[panel_id]
        agent_responses = []
        
        try:
            # Acquire coordination lock
            async with self._coordination_locks[panel_id]:
                
                # Process message through each active agent
                response_tasks = []
                for agent_persona in panel.active_agents:
                    task = self._process_message_with_single_agent(
                        agent_persona=agent_persona,
                        message=message,
                        panel_context={
                            "panel_id": panel_id,
                            "session_id": panel.session_id,
                            "coordination_mode": panel.coordination_mode,
                            **agent_config
                        }
                    )
                    response_tasks.append(task)
                
                # Wait for all agents to respond (with timeout)
                try:
                    responses = await asyncio.wait_for(
                        asyncio.gather(*response_tasks, return_exceptions=True),
                        timeout=self.config.response_timeout
                    )
                    
                    # Process responses
                    for i, response in enumerate(responses):
                        if isinstance(response, Exception):
                            self.logger.error(f"Agent {panel.active_agents[i]} failed: {response}")
                            continue
                        
                        if response:
                            agent_responses.append(response)
                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Agent responses timed out for panel {panel_id}")
                    # Return any responses that completed
                
                # Update panel activity
                panel.last_activity = datetime.now()
                panel.message_count += 1
            
            return agent_responses
            
        except Exception as e:
            self.logger.error(f"Failed to process multi-agent message: {e}")
            return []
    
    async def _process_message_with_single_agent(self,
                                                 agent_persona: str,
                                                 message: FFMessageDTO,
                                                 panel_context: Dict[str, Any]) -> Optional[FFAgentResponse]:
        """Process message with a single agent"""
        start_time = time.time()
        
        try:
            # This is a placeholder for actual agent processing
            # In a full implementation, this would integrate with the FF panel manager
            # to process the message through the specific agent persona
            
            # Get agent-specific context
            session_id = panel_context.get("session_id", "")
            coordination_mode = panel_context.get("coordination_mode", "sequential")
            
            # Generate agent response (placeholder)
            response_content = await self._generate_agent_response(
                agent_persona=agent_persona,
                message=message,
                context=panel_context
            )
            
            # Calculate confidence score (placeholder)
            confidence_score = await self._calculate_agent_confidence(
                agent_persona=agent_persona,
                response_content=response_content,
                message=message
            )
            
            response_time = time.time() - start_time
            
            # Create agent response
            agent_response = FFAgentResponse(
                agent_id=f"agent_{agent_persona}_{int(time.time() * 1000)}",
                agent_persona=agent_persona,
                response_content=response_content,
                confidence_score=confidence_score,
                response_time=response_time,
                metadata={
                    "coordination_mode": coordination_mode,
                    "session_id": session_id,
                    "message_length": len(message.content),
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            # Update agent performance tracking
            await self._update_agent_performance(agent_persona, agent_response)
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Failed to process message with agent {agent_persona}: {e}")
            return None
    
    async def _generate_agent_response(self,
                                       agent_persona: str,
                                       message: FFMessageDTO,
                                       context: Dict[str, Any]) -> str:
        """Generate response from specific agent persona (placeholder)"""
        # This is a Phase 2 placeholder - would integrate with actual agent/LLM in full implementation
        
        persona_styles = {
            "general_assistant": "I'm here to help with your general questions.",
            "technical_expert": "From a technical perspective, I can provide detailed analysis.",
            "creative_writer": "Let me approach this with creativity and imagination.",
            "analyst": "Based on the data and evidence available, my assessment is:",
            "moderator": "Let me help facilitate this discussion and find common ground.",
            "specialist": "As a specialist in this area, I can offer specific insights.",
            "critic": "I'll provide a critical analysis of this topic.",
            "supporter": "I see the positive aspects and potential in this approach."
        }
        
        base_response = persona_styles.get(agent_persona, "As an AI assistant, I'll help you with this.")
        user_message = message.content
        
        return f"{base_response} Regarding your message '{user_message}', I think this requires careful consideration. [Agent: {agent_persona}]"
    
    async def _calculate_agent_confidence(self,
                                          agent_persona: str,
                                          response_content: str,
                                          message: FFMessageDTO) -> float:
        """Calculate confidence score for agent response (placeholder)"""
        # This is a placeholder - would use more sophisticated confidence calculation
        
        # Base confidence varies by agent type
        base_confidence = {
            "general_assistant": 0.7,
            "technical_expert": 0.8,
            "creative_writer": 0.6,
            "analyst": 0.8,
            "moderator": 0.7,
            "specialist": 0.9,
            "critic": 0.7,
            "supporter": 0.6
        }.get(agent_persona, 0.7)
        
        # Adjust based on response length (longer responses might be more confident)
        length_factor = min(len(response_content) / 500, 1.0) * 0.1
        
        return min(base_confidence + length_factor, 1.0)
    
    async def coordinate_agents(self,
                                agents: List[str],
                                coordination_mode: str,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple agents for response generation.
        
        Args:
            agents: List of agent identifiers
            coordination_mode: Mode of coordination
            context: Coordination context
            
        Returns:
            Coordination results
        """
        try:
            agent_responses = context.get("agent_responses", [])
            
            if not agent_responses:
                return {
                    "success": False,
                    "error": "No agent responses to coordinate",
                    "final_response": "I apologize, but I couldn't generate a response."
                }
            
            # Coordinate based on mode
            if coordination_mode == FFCoordinationMode.CONSENSUS.value:
                result = await self._coordinate_by_consensus(agent_responses, context)
            elif coordination_mode == FFCoordinationMode.COMPETITIVE.value:
                result = await self._coordinate_by_competition(agent_responses, context)
            elif coordination_mode == FFCoordinationMode.COLLABORATIVE.value:
                result = await self._coordinate_by_collaboration(agent_responses, context)
            elif coordination_mode == FFCoordinationMode.ROUND_ROBIN.value:
                result = await self._coordinate_by_round_robin(agent_responses, context)
            else:
                # Default to sequential coordination
                result = await self._coordinate_sequentially(agent_responses, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate agents: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_response": "I encountered an error coordinating the agent responses."
            }
    
    async def _coordinate_by_consensus(self, agent_responses: List[FFAgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents by building consensus"""
        if not agent_responses:
            return {"success": False, "final_response": "No responses to coordinate"}
        
        # For demonstration, use the response with highest confidence as consensus
        best_response = max(agent_responses, key=lambda r: r.confidence_score)
        
        # Check if consensus threshold is met
        avg_confidence = sum(r.confidence_score for r in agent_responses) / len(agent_responses)
        consensus_achieved = avg_confidence >= self.config.consensus_threshold
        
        if consensus_achieved:
            self._multi_agent_stats["consensus_achieved"] += 1
        
        final_response = f"Based on consensus among {len(agent_responses)} agents: {best_response.response_content}"
        
        return {
            "success": True,
            "final_response": final_response,
            "coordination_method": "consensus",
            "consensus_achieved": consensus_achieved,
            "participating_agents": len(agent_responses),
            "selected_agent": best_response.agent_persona,
            "confidence_score": best_response.confidence_score
        }
    
    async def _coordinate_by_competition(self, agent_responses: List[FFAgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through competition"""
        if not agent_responses:
            return {"success": False, "final_response": "No responses to coordinate"}
        
        # Select the response with highest confidence score
        winner = max(agent_responses, key=lambda r: r.confidence_score)
        
        return {
            "success": True,
            "final_response": f"Winning response from {winner.agent_persona}: {winner.response_content}",
            "coordination_method": "competitive", 
            "winning_agent": winner.agent_persona,
            "confidence_score": winner.confidence_score,
            "competing_agents": len(agent_responses)
        }
    
    async def _coordinate_by_collaboration(self, agent_responses: List[FFAgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through collaboration"""
        if not agent_responses:
            return {"success": False, "final_response": "No responses to coordinate"}
        
        # Combine insights from all agents
        collaborative_response = "Based on collaborative input from multiple agents:\n\n"
        
        for i, response in enumerate(agent_responses, 1):
            collaborative_response += f"{i}. {response.agent_persona}: {response.response_content}\n\n"
        
        collaborative_response += "Synthesized response: This represents the combined wisdom of our multi-agent panel."
        
        avg_confidence = sum(r.confidence_score for r in agent_responses) / len(agent_responses)
        
        return {
            "success": True,
            "final_response": collaborative_response,
            "coordination_method": "collaborative",
            "participating_agents": len(agent_responses),
            "average_confidence": avg_confidence
        }
    
    async def _coordinate_by_round_robin(self, agent_responses: List[FFAgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents in round-robin fashion"""
        if not agent_responses:
            return {"success": False, "final_response": "No responses to coordinate"}
        
        # For round-robin, we can select the first response or rotate based on some state
        # For simplicity, just use the first response
        selected_response = agent_responses[0]
        
        return {
            "success": True,
            "final_response": f"Response from {selected_response.agent_persona}: {selected_response.response_content}",
            "coordination_method": "round_robin",
            "selected_agent": selected_response.agent_persona,
            "confidence_score": selected_response.confidence_score,
            "available_agents": len(agent_responses)
        }
    
    async def _coordinate_sequentially(self, agent_responses: List[FFAgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents sequentially"""
        if not agent_responses:
            return {"success": False, "final_response": "No responses to coordinate"}
        
        # Process responses in order of agent personas
        sequential_response = "Sequential multi-agent response:\n\n"
        
        for response in agent_responses:
            sequential_response += f"• {response.agent_persona}: {response.response_content}\n\n"
        
        return {
            "success": True,
            "final_response": sequential_response,
            "coordination_method": "sequential",
            "agents_count": len(agent_responses)
        }
    
    async def get_panel_history(self, panel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get panel conversation history using FF panel manager.
        
        Args:
            panel_id: Panel identifier
            
        Returns:
            Panel history information or None if not found
        """
        try:
            if panel_id not in self._active_panels:
                return None
            
            panel = self._active_panels[panel_id]
            
            # Get conversation history from FF storage
            messages = await self.ff_storage.get_messages(
                user_id=panel.user_id,
                session_id=panel.session_id,
                limit=50
            )
            
            panel_history = {
                "panel_id": panel_id,
                "session_id": panel.session_id,
                "user_id": panel.user_id,
                "agent_personas": panel.agent_personas,
                "coordination_mode": panel.coordination_mode,
                "created_at": panel.created_at.isoformat(),
                "last_activity": panel.last_activity.isoformat(),
                "message_count": panel.message_count,
                "active_agents": panel.active_agents,
                "conversation_history": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    }
                    for msg in messages
                ]
            }
            
            return panel_history
            
        except Exception as e:
            self.logger.error(f"Failed to get panel history: {e}")
            return None
    
    # Helper methods
    
    async def _store_multi_agent_interaction(self, session_id: str, user_id: str, 
                                             message: FFMessageDTO, agent_responses: List[FFAgentResponse],
                                             final_response: Dict[str, Any]) -> None:
        """Store multi-agent interaction in FF storage"""
        try:
            # Store the final coordinated response
            await self.ff_storage.add_message(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=final_response.get("final_response", ""),
                metadata={
                    "multi_agent": True,
                    "coordination_method": final_response.get("coordination_method", "unknown"),
                    "agent_count": len(agent_responses),
                    "consensus_achieved": final_response.get("consensus_achieved", False),
                    "individual_agents": [resp.agent_persona for resp in agent_responses]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store multi-agent interaction: {e}")
    
    async def _update_agent_performance(self, agent_persona: str, response: FFAgentResponse) -> None:
        """Update performance metrics for an agent"""
        try:
            if agent_persona not in self._agent_performance:
                self._agent_performance[agent_persona] = {
                    "total_responses": 0,
                    "average_response_time": 0.0,
                    "average_confidence": 0.0,
                    "consensus_contributions": 0,
                    "quality_score": 0.5
                }
            
            perf = self._agent_performance[agent_persona]
            perf["total_responses"] += 1
            
            # Update average response time
            current_avg_time = perf["average_response_time"]
            total_responses = perf["total_responses"]
            perf["average_response_time"] = ((current_avg_time * (total_responses - 1)) + response.response_time) / total_responses
            
            # Update average confidence
            current_avg_conf = perf["average_confidence"]
            perf["average_confidence"] = ((current_avg_conf * (total_responses - 1)) + response.confidence_score) / total_responses
            
        except Exception as e:
            self.logger.error(f"Failed to update agent performance: {e}")
    
    async def _cleanup_inactive_panels(self) -> None:
        """Clean up inactive panels"""
        try:
            current_time = datetime.now()
            timeout_seconds = self.config.panel_session_timeout
            
            inactive_panels = []
            for panel_id, panel in self._active_panels.items():
                if (current_time - panel.last_activity).total_seconds() > timeout_seconds:
                    inactive_panels.append(panel_id)
            
            for panel_id in inactive_panels:
                self._active_panels.pop(panel_id, None)
                self._coordination_locks.pop(panel_id, None)
            
            if inactive_panels:
                self.logger.debug(f"Cleaned up {len(inactive_panels)} inactive panels")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up inactive panels: {e}")
    
    async def _cleanup_coordination_locks(self) -> None:
        """Clean up unused coordination locks"""
        try:
            # Remove locks for panels that no longer exist
            lock_keys = list(self._coordination_locks.keys())
            for panel_id in lock_keys:
                if panel_id not in self._active_panels:
                    self._coordination_locks.pop(panel_id, None)
        except Exception as e:
            self.logger.error(f"Error cleaning up coordination locks: {e}")
    
    async def _update_agent_performance_metrics(self) -> None:
        """Update aggregate agent performance metrics"""
        try:
            # This could include more sophisticated performance analysis
            # For now, just log the current statistics
            total_agents = len(self._agent_performance)
            if total_agents > 0:
                self.logger.debug(f"Tracking performance for {total_agents} agents")
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def get_capabilities(self) -> List[str]:
        """Get list of component capabilities"""
        return [cap.name for cap in self._component_info.capabilities]
    
    async def supports_use_case(self, use_case: str) -> bool:
        """Check if component supports a specific use case"""
        return use_case in self._component_info.use_cases
    
    def get_multi_agent_statistics(self) -> Dict[str, Any]:
        """Get multi-agent processing statistics"""
        return {
            **self._multi_agent_stats,
            "active_panels": len(self._active_panels),
            "tracked_agents": len(self._agent_performance),
            "agent_performance": self._agent_performance.copy()
        }
    
    async def cleanup(self) -> None:
        """Cleanup component resources following FF patterns"""
        try:
            self.logger.info("Cleaning up FF Multi-Agent Component...")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._performance_tracking_task:
                self._performance_tracking_task.cancel()
            
            # Clear panel structures
            self._active_panels.clear()
            self._agent_performance.clear()
            self._coordination_locks.clear()
            self._response_cache.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("FF Multi-Agent Component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during FF Multi-Agent Component cleanup: {e}")
            raise