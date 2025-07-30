"""
Panel and persona configuration.

Manages multi-persona panels, insights, and collaboration settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .base import BaseConfig, validate_positive, validate_range


@dataclass
class PanelConfig(BaseConfig):
    """
    Panel and persona configuration.
    
    Controls panel sessions, personas, and insight generation.
    """
    
    # Directory settings
    panel_sessions_directory: str = "panel_sessions"
    global_personas_directory: str = "personas_global"
    insights_subdirectory: str = "insights"
    personas_subdirectory: str = "personas"
    
    # File names
    panel_metadata_filename: str = "panel.json"
    panel_id_prefix: str = "panel"
    
    # Panel limits
    max_personas_per_panel: int = 10
    min_personas_per_panel: int = 2
    max_messages_per_panel: int = 10000
    max_panels_per_user: int = 100
    
    # Persona settings
    global_persona_limit: int = 50
    user_persona_limit: int = 20
    allow_custom_personas: bool = True
    require_persona_validation: bool = True
    
    # Insight settings
    insight_generation_enabled: bool = True
    insight_retention_days: int = 90
    insight_min_messages: int = 5  # Minimum messages before generating insights
    insight_update_frequency: str = "on_demand"  # "on_demand", "periodic", "real_time"
    
    # Threading settings
    message_threading_enabled: bool = True
    max_thread_depth: int = 10
    thread_branching_allowed: bool = True
    
    # Collaboration features
    voting_enabled: bool = True
    consensus_tracking: bool = True
    disagreement_highlighting: bool = True
    
    # Panel types configuration
    panel_types: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "multi_persona": {
            "description": "Multiple AI personas discussing a topic",
            "min_personas": 2,
            "max_personas": 10,
            "features": ["voting", "threading", "insights"]
        },
        "focus_group": {
            "description": "Structured feedback from diverse perspectives",
            "min_personas": 3,
            "max_personas": 8,
            "features": ["voting", "consensus", "structured_feedback"]
        },
        "expert_panel": {
            "description": "Domain experts providing specialized input",
            "min_personas": 2,
            "max_personas": 5,
            "features": ["expertise_validation", "citations", "technical_depth"]
        },
        "brainstorm": {
            "description": "Creative ideation with multiple perspectives",
            "min_personas": 2,
            "max_personas": 10,
            "features": ["idea_tracking", "building_on_ideas", "divergent_thinking"]
        }
    })
    
    # Persona templates
    default_personas: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "id": "analyst",
            "name": "Analytical Thinker",
            "description": "Focuses on data, logic, and systematic analysis",
            "traits": ["logical", "data-driven", "systematic"]
        },
        {
            "id": "creative",
            "name": "Creative Innovator",
            "description": "Brings creative solutions and out-of-the-box thinking",
            "traits": ["creative", "innovative", "imaginative"]
        },
        {
            "id": "pragmatist",
            "name": "Practical Pragmatist",
            "description": "Focuses on practical implementation and real-world constraints",
            "traits": ["practical", "realistic", "solution-oriented"]
        }
    ])
    
    # Performance settings
    lazy_load_insights: bool = True
    cache_panel_state: bool = True
    panel_state_cache_ttl: int = 3600  # 1 hour
    
    # Context management
    maintain_persona_context: bool = True
    context_window_messages: int = 50
    context_summary_enabled: bool = True
    
    def validate(self) -> List[str]:
        """
        Validate panel configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate directories
        if not self.panel_sessions_directory:
            errors.append("panel_sessions_directory cannot be empty")
        if not self.global_personas_directory:
            errors.append("global_personas_directory cannot be empty")
        
        # Validate panel limits
        if error := validate_positive(self.max_personas_per_panel, "max_personas_per_panel"):
            errors.append(error)
        if error := validate_positive(self.min_personas_per_panel, "min_personas_per_panel"):
            errors.append(error)
        
        if self.min_personas_per_panel > self.max_personas_per_panel:
            errors.append(f"min_personas_per_panel ({self.min_personas_per_panel}) cannot exceed max_personas_per_panel ({self.max_personas_per_panel})")
        
        if error := validate_positive(self.max_messages_per_panel, "max_messages_per_panel"):
            errors.append(error)
        if error := validate_positive(self.max_panels_per_user, "max_panels_per_user"):
            errors.append(error)
        
        # Validate persona limits
        if error := validate_positive(self.global_persona_limit, "global_persona_limit"):
            errors.append(error)
        if error := validate_positive(self.user_persona_limit, "user_persona_limit"):
            errors.append(error)
        
        # Validate insight settings
        if self.insight_generation_enabled:
            if error := validate_positive(self.insight_retention_days, "insight_retention_days"):
                errors.append(error)
            if error := validate_positive(self.insight_min_messages, "insight_min_messages"):
                errors.append(error)
            
            valid_frequencies = ["on_demand", "periodic", "real_time"]
            if self.insight_update_frequency not in valid_frequencies:
                errors.append(f"insight_update_frequency must be one of {valid_frequencies}, got {self.insight_update_frequency}")
        
        # Validate threading settings
        if self.message_threading_enabled:
            if error := validate_positive(self.max_thread_depth, "max_thread_depth"):
                errors.append(error)
        
        # Validate panel types
        for panel_type, config in self.panel_types.items():
            if "min_personas" in config and "max_personas" in config:
                if config["min_personas"] > config["max_personas"]:
                    errors.append(f"Panel type '{panel_type}': min_personas cannot exceed max_personas")
        
        # Validate performance settings
        if self.cache_panel_state:
            if error := validate_positive(self.panel_state_cache_ttl, "panel_state_cache_ttl"):
                errors.append(error)
        
        # Validate context settings
        if self.maintain_persona_context:
            if error := validate_positive(self.context_window_messages, "context_window_messages"):
                errors.append(error)
        
        return errors
    
    def get_panel_type_config(self, panel_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific panel type.
        
        Args:
            panel_type: Type of panel
            
        Returns:
            Panel type configuration
        """
        return self.panel_types.get(panel_type, {})
    
    def get_default_persona(self, persona_id: str) -> Optional[Dict[str, str]]:
        """
        Get a default persona by ID.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            Persona configuration or None
        """
        for persona in self.default_personas:
            if persona.get("id") == persona_id:
                return persona
        return None