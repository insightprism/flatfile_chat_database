"""
FF Multi-Agent Component Configuration

Configuration for FF multi-agent component following existing FF configuration 
patterns and integrating with FF panel manager for agent coordination.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from .ff_base_config import FFBaseConfigDTO, validate_positive, validate_range


class FFCoordinationMode(str, Enum):
    """Agent coordination modes"""
    SEQUENTIAL = "sequential"          # Agents respond in sequence
    PARALLEL = "parallel"              # Agents respond simultaneously
    ROUND_ROBIN = "round_robin"        # Agents take turns
    DYNAMIC = "dynamic"                # Smart routing based on context
    CONSENSUS = "consensus"            # Agents reach consensus
    COMPETITIVE = "competitive"        # Agents compete for best response
    COLLABORATIVE = "collaborative"    # Agents work together on response


class FFAgentSelectionStrategy(str, Enum):
    """Agent selection strategies"""
    ALL = "all"                       # Use all available agents
    RANDOM = "random"                 # Select agents randomly
    EXPERTISE_BASED = "expertise"     # Select based on expertise
    PERFORMANCE_BASED = "performance" # Select based on past performance
    LOAD_BALANCED = "load_balanced"   # Balance load across agents
    USER_PREFERENCE = "user_preference" # User-specified agents


class FFConflictResolution(str, Enum):
    """Conflict resolution strategies"""
    MAJORITY_VOTE = "majority_vote"   # Use majority decision
    WEIGHTED_VOTE = "weighted_vote"   # Weight votes by agent expertise
    MODERATOR = "moderator"           # Use designated moderator agent
    USER_CHOICE = "user_choice"       # Let user choose
    MERGE_RESPONSES = "merge"         # Merge all responses
    FIRST_RESPONSE = "first"          # Use first response


@dataclass
class FFMultiAgentConfigDTO(FFBaseConfigDTO):
    """FF Multi-Agent component configuration following FF patterns"""
    
    # Agent management settings
    max_agents: int = 5
    min_agents: int = 2
    default_coordination_mode: str = FFCoordinationMode.ROUND_ROBIN.value
    
    # FF panel manager integration settings
    use_ff_panel_manager: bool = True
    panel_session_timeout: int = 3600  # seconds
    panel_cleanup_interval: int = 600   # seconds
    
    # Agent selection settings
    agent_selection_strategy: str = FFAgentSelectionStrategy.EXPERTISE_BASED.value
    enable_dynamic_agent_selection: bool = True
    agent_load_balancing: bool = True
    
    # Response coordination settings
    response_timeout: int = 30  # seconds per agent
    max_response_rounds: int = 3
    enable_response_validation: bool = True
    
    # Consensus and voting settings
    consensus_threshold: float = 0.7  # Agreement level for consensus
    voting_weight_strategy: str = "equal"  # equal, expertise, performance
    conflict_resolution: str = FFConflictResolution.MAJORITY_VOTE.value
    
    # Agent personas and specialization
    enable_agent_personas: bool = True
    persona_consistency_check: bool = True
    allow_persona_switching: bool = False
    
    # Performance and monitoring
    enable_agent_performance_tracking: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: ["response_time", "quality", "relevance"])
    agent_performance_window: int = 100  # number of interactions to track
    
    # Communication and collaboration
    enable_inter_agent_communication: bool = False
    agent_discussion_rounds: int = 1
    enable_agent_memory_sharing: bool = False
    
    # Quality control
    enable_response_filtering: bool = True
    min_response_quality_score: float = 0.6
    enable_redundancy_removal: bool = True
    
    # Advanced features
    enable_agent_learning: bool = False
    enable_adaptive_coordination: bool = False
    context_sharing_level: str = "session"  # none, session, user, global
    
    def validate(self) -> List[str]:
        """Validate multi-agent configuration"""
        errors = []
        
        # Validate agent count settings
        error = validate_positive(self.max_agents, "max_agents")
        if error:
            errors.append(error)
            
        error = validate_positive(self.min_agents, "min_agents")
        if error:
            errors.append(error)
            
        if self.min_agents > self.max_agents:
            errors.append("min_agents must be less than or equal to max_agents")
        
        if self.min_agents < 2:
            errors.append("min_agents must be at least 2 for multi-agent coordination")
        
        # Validate coordination mode
        valid_modes = [mode.value for mode in FFCoordinationMode]
        if self.default_coordination_mode not in valid_modes:
            errors.append(f"default_coordination_mode must be one of {valid_modes}")
        
        # Validate panel manager settings
        if not self.use_ff_panel_manager:
            errors.append("FF panel manager integration is required for multi-agent functionality")
            
        error = validate_positive(self.panel_session_timeout, "panel_session_timeout")
        if error:
            errors.append(error)
            
        error = validate_positive(self.panel_cleanup_interval, "panel_cleanup_interval")
        if error:
            errors.append(error)
        
        # Validate agent selection strategy
        valid_strategies = [strategy.value for strategy in FFAgentSelectionStrategy]
        if self.agent_selection_strategy not in valid_strategies:
            errors.append(f"agent_selection_strategy must be one of {valid_strategies}")
        
        # Validate response settings
        error = validate_positive(self.response_timeout, "response_timeout")
        if error:
            errors.append(error)
            
        error = validate_positive(self.max_response_rounds, "max_response_rounds")
        if error:
            errors.append(error)
        
        # Validate consensus settings
        error = validate_range(self.consensus_threshold, 0.0, 1.0, "consensus_threshold")
        if error:
            errors.append(error)
        
        valid_weight_strategies = ["equal", "expertise", "performance"]
        if self.voting_weight_strategy not in valid_weight_strategies:
            errors.append(f"voting_weight_strategy must be one of {valid_weight_strategies}")
        
        valid_conflict_resolution = [resolution.value for resolution in FFConflictResolution]
        if self.conflict_resolution not in valid_conflict_resolution:
            errors.append(f"conflict_resolution must be one of {valid_conflict_resolution}")
        
        # Validate performance tracking settings
        if self.enable_agent_performance_tracking:
            error = validate_positive(self.agent_performance_window, "agent_performance_window")
            if error:
                errors.append(error)
                
            valid_metrics = ["response_time", "quality", "relevance", "accuracy", "user_satisfaction"]
            for metric in self.performance_metrics:
                if metric not in valid_metrics:
                    errors.append(f"Invalid performance metric '{metric}'. Must be one of {valid_metrics}")
        
        # Validate communication settings
        if self.enable_inter_agent_communication:
            error = validate_positive(self.agent_discussion_rounds, "agent_discussion_rounds")
            if error:
                errors.append(error)
        
        # Validate quality control settings
        if self.enable_response_filtering:
            error = validate_range(self.min_response_quality_score, 0.0, 1.0, "min_response_quality_score")
            if error:
                errors.append(error)
        
        # Validate context sharing level
        valid_context_levels = ["none", "session", "user", "global"]
        if self.context_sharing_level not in valid_context_levels:
            errors.append(f"context_sharing_level must be one of {valid_context_levels}")
        
        return errors
    
    def get_agent_management_config(self) -> Dict[str, Any]:
        """Get agent management configuration"""
        return {
            "max_agents": self.max_agents,
            "min_agents": self.min_agents,
            "coordination_mode": self.default_coordination_mode,
            "selection_strategy": self.agent_selection_strategy,
            "dynamic_selection": self.enable_dynamic_agent_selection,
            "load_balancing": self.agent_load_balancing
        }
    
    def get_panel_manager_config(self) -> Dict[str, Any]:
        """Get FF panel manager configuration"""
        return {
            "enabled": self.use_ff_panel_manager,
            "session_timeout": self.panel_session_timeout,
            "cleanup_interval": self.panel_cleanup_interval
        }
    
    def get_response_coordination_config(self) -> Dict[str, Any]:
        """Get response coordination configuration"""
        return {
            "timeout": self.response_timeout,
            "max_rounds": self.max_response_rounds,
            "validation_enabled": self.enable_response_validation,
            "consensus_threshold": self.consensus_threshold,
            "voting_weight_strategy": self.voting_weight_strategy,
            "conflict_resolution": self.conflict_resolution
        }
    
    def get_persona_config(self) -> Dict[str, Any]:
        """Get agent persona configuration"""
        return {
            "personas_enabled": self.enable_agent_personas,
            "consistency_check": self.persona_consistency_check,
            "persona_switching": self.allow_persona_switching
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance tracking configuration"""
        return {
            "tracking_enabled": self.enable_agent_performance_tracking,
            "metrics": self.performance_metrics,
            "performance_window": self.agent_performance_window
        }
    
    def get_communication_config(self) -> Dict[str, Any]:
        """Get inter-agent communication configuration"""
        return {
            "inter_agent_communication": self.enable_inter_agent_communication,
            "discussion_rounds": self.agent_discussion_rounds,
            "memory_sharing": self.enable_agent_memory_sharing,
            "context_sharing_level": self.context_sharing_level
        }
    
    def get_quality_control_config(self) -> Dict[str, Any]:
        """Get quality control configuration"""
        return {
            "response_filtering": self.enable_response_filtering,
            "min_quality_score": self.min_response_quality_score,
            "redundancy_removal": self.enable_redundancy_removal
        }
    
    def get_advanced_features_config(self) -> Dict[str, Any]:
        """Get advanced features configuration"""
        return {
            "agent_learning": self.enable_agent_learning,
            "adaptive_coordination": self.enable_adaptive_coordination
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()
        
        # Additional post-initialization validation
        if hasattr(self, '_post_init_done'):
            return
        self._post_init_done = True
        
        # Ensure minimum agents for multi-agent scenarios
        if self.min_agents < 2:
            self.min_agents = 2
        
        # Disable advanced features that require additional infrastructure
        if not self.use_ff_panel_manager:
            self.enable_inter_agent_communication = False
            self.enable_agent_memory_sharing = False
        
        # Set reasonable defaults for performance metrics
        if not self.performance_metrics:
            self.performance_metrics = ["response_time", "quality", "relevance"]


@dataclass
class FFMultiAgentUseCaseConfigDTO(FFBaseConfigDTO):
    """Configuration for multi-agent usage in specific use cases"""
    
    # Use case specific settings
    use_case_name: str = "multi_ai_panel"
    coordination_style: str = "collaborative"  # competitive, collaborative, debate
    
    # Agent role definitions
    agent_roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_expertise_areas: List[str] = field(default_factory=list)
    
    # Use case specific coordination
    custom_coordination_mode: Optional[str] = None
    custom_selection_strategy: Optional[str] = None
    custom_conflict_resolution: Optional[str] = None
    
    # Response synthesis settings
    response_synthesis_method: str = "merge"  # merge, vote, delegate, moderate
    enable_cross_agent_validation: bool = True
    
    # Use case specific thresholds
    custom_consensus_threshold: Optional[float] = None
    custom_quality_threshold: Optional[float] = None
    custom_timeout: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate multi-agent use case configuration"""
        errors = []
        
        # Validate use case name
        if not self.use_case_name or not self.use_case_name.strip():
            errors.append("use_case_name cannot be empty")
        
        # Validate coordination style
        valid_styles = ["competitive", "collaborative", "debate", "discussion", "panel"]
        if self.coordination_style not in valid_styles:
            errors.append(f"coordination_style must be one of {valid_styles}")
        
        # Validate custom coordination mode if specified
        if self.custom_coordination_mode:
            valid_modes = [mode.value for mode in FFCoordinationMode]
            if self.custom_coordination_mode not in valid_modes:
                errors.append(f"custom_coordination_mode must be one of {valid_modes}")
        
        # Validate custom selection strategy if specified
        if self.custom_selection_strategy:
            valid_strategies = [strategy.value for strategy in FFAgentSelectionStrategy]
            if self.custom_selection_strategy not in valid_strategies:
                errors.append(f"custom_selection_strategy must be one of {valid_strategies}")
        
        # Validate custom conflict resolution if specified
        if self.custom_conflict_resolution:
            valid_resolutions = [resolution.value for resolution in FFConflictResolution]
            if self.custom_conflict_resolution not in valid_resolutions:
                errors.append(f"custom_conflict_resolution must be one of {valid_resolutions}")
        
        # Validate response synthesis method
        valid_synthesis = ["merge", "vote", "delegate", "moderate", "consensus"]
        if self.response_synthesis_method not in valid_synthesis:
            errors.append(f"response_synthesis_method must be one of {valid_synthesis}")
        
        # Validate custom thresholds
        if self.custom_consensus_threshold is not None:
            error = validate_range(self.custom_consensus_threshold, 0.0, 1.0, "custom_consensus_threshold")
            if error:
                errors.append(error)
        
        if self.custom_quality_threshold is not None:
            error = validate_range(self.custom_quality_threshold, 0.0, 1.0, "custom_quality_threshold")
            if error:
                errors.append(error)
        
        if self.custom_timeout is not None:
            error = validate_positive(self.custom_timeout, "custom_timeout")
            if error:
                errors.append(error)
        
        return errors
    
    def get_coordination_config(self) -> Dict[str, Any]:
        """Get coordination configuration for this use case"""
        return {
            "coordination_style": self.coordination_style,
            "custom_mode": self.custom_coordination_mode,
            "custom_selection": self.custom_selection_strategy,
            "custom_conflict_resolution": self.custom_conflict_resolution
        }
    
    def get_agent_role_config(self) -> Dict[str, Any]:
        """Get agent role and expertise configuration"""
        return {
            "agent_roles": self.agent_roles,
            "required_expertise": self.required_expertise_areas
        }
    
    def get_response_synthesis_config(self) -> Dict[str, Any]:
        """Get response synthesis configuration"""
        return {
            "synthesis_method": self.response_synthesis_method,
            "cross_validation": self.enable_cross_agent_validation,
            "custom_thresholds": {
                "consensus": self.custom_consensus_threshold,
                "quality": self.custom_quality_threshold,
                "timeout": self.custom_timeout
            }
        }