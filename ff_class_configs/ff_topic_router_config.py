"""
FF Topic Router Component Configuration

Configuration classes for the FF Topic Router component that provides
intelligent topic detection and routing using existing FF search and vector storage.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ff_class_configs.ff_base_config import FFBaseConfigDTO


class FFTopicDetectionMethod(Enum):
    """Methods for detecting topics in messages"""
    KEYWORD_MATCHING = "keyword_matching"
    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ML_CLASSIFICATION = "ml_classification"
    HYBRID = "hybrid"


class FFRoutingStrategy(Enum):
    """Strategies for routing based on detected topics"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"
    ROUND_ROBIN = "round_robin"


class FFTopicCategory(Enum):
    """Categories of topics for organization"""
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"
    EDUCATIONAL = "educational"
    PERSONAL = "personal"
    ENTERTAINMENT = "entertainment"
    SCIENTIFIC = "scientific"
    MEDICAL = "medical"
    LEGAL = "legal"
    GENERAL = "general"


@dataclass
class FFTopicRoute:
    """Definition of a topic routing rule"""
    name: str
    category: FFTopicCategory
    keywords: List[str]
    patterns: List[str]
    target_agent: str
    confidence_threshold: float
    priority: int = 100
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced routing features
    fallback_agents: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    time_based_routing: Optional[Dict[str, Any]] = None
    user_preference_weight: float = 1.0


@dataclass
class FFTopicDetectionConfigDTO:
    """Configuration for topic detection methods"""
    
    # Detection methods
    enabled_methods: List[str] = field(default_factory=lambda: [
        "keyword_matching", "pattern_matching", "semantic_similarity"
    ])
    primary_method: str = "hybrid"
    
    # Keyword matching settings
    keyword_matching_config: Dict[str, Any] = field(default_factory=lambda: {
        "case_sensitive": False,
        "partial_match": True,
        "stemming_enabled": True,
        "synonym_expansion": True,
        "weight": 1.0
    })
    
    # Pattern matching settings
    pattern_matching_config: Dict[str, Any] = field(default_factory=lambda: {
        "regex_enabled": True,
        "fuzzy_matching": True,
        "edit_distance_threshold": 2,
        "weight": 1.2
    })
    
    # Semantic similarity settings
    semantic_similarity_config: Dict[str, Any] = field(default_factory=lambda: {
        "embedding_model": "sentence-transformers",
        "similarity_threshold": 0.7,
        "use_ff_vector_storage": True,
        "weight": 1.5
    })
    
    # ML classification settings
    ml_classification_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_type": "naive_bayes",
        "training_data_required": True,
        "auto_retrain": False,
        "confidence_calibration": True,
        "weight": 2.0
    })
    
    # Hybrid approach settings
    hybrid_config: Dict[str, Any] = field(default_factory=lambda: {
        "method_weights": {
            "keyword_matching": 0.3,
            "pattern_matching": 0.3,
            "semantic_similarity": 0.4
        },
        "consensus_threshold": 0.6,
        "enable_method_selection": True
    })


@dataclass
class FFTopicRouterConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Topic Router component"""
    
    # Core routing settings
    confidence_threshold: float = 0.7
    min_topic_confidence: float = 0.5
    max_topics_per_message: int = 3
    enable_multi_topic_routing: bool = True
    
    # Detection configuration
    detection: FFTopicDetectionConfigDTO = field(default_factory=FFTopicDetectionConfigDTO)
    routing_strategy: str = "highest_confidence"
    
    # FF integration settings
    use_ff_search: bool = True
    use_ff_vector_storage: bool = True
    use_ff_storage_history: bool = True
    vector_similarity_threshold: float = 0.8
    search_history_limit: int = 100
    
    # Caching and performance
    enable_topic_caching: bool = True
    topic_cache_ttl: int = 300  # seconds
    enable_route_caching: bool = True
    route_cache_size: int = 1000
    
    # Learning and adaptation
    enable_learning: bool = True
    learning_rate: float = 0.1
    adaptation_window: int = 100  # messages
    enable_feedback_learning: bool = True
    feedback_weight: float = 2.0
    
    # Fallback settings
    default_agent: str = "general"
    enable_fallback_routing: bool = True
    fallback_confidence_threshold: float = 0.3
    enable_escalation: bool = True
    escalation_agents: List[str] = field(default_factory=lambda: ["supervisor", "expert"])
    
    # Context awareness
    enable_context_awareness: bool = True
    context_window_size: int = 5  # previous messages
    user_preference_weight: float = 1.2
    session_context_weight: float = 1.1
    
    # Topic management
    max_routes: int = 100
    enable_dynamic_routes: bool = True
    auto_disable_low_confidence: bool = True
    low_confidence_threshold: float = 0.2
    route_usage_tracking: bool = True
    
    # Performance monitoring
    enable_routing_metrics: bool = True
    enable_performance_monitoring: bool = True
    performance_logging_interval: int = 3600  # seconds
    
    # Security and validation
    enable_input_validation: bool = True
    max_message_length: int = 10000
    blocked_keywords: List[str] = field(default_factory=list)
    sensitive_topic_handling: str = "escalate"  # escalate, block, sanitize
    
    # Pre-defined topic routes
    default_routes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "technical",
            "category": "technical",
            "keywords": ["code", "programming", "software", "development", "algorithm", 
                        "database", "api", "bug", "error", "debug", "technical"],
            "patterns": ["how to code", "programming help", "technical issue", 
                        "software problem", "debug this", "fix error"],
            "target_agent": "technical_expert",
            "confidence_threshold": 0.8,
            "priority": 90
        },
        {
            "name": "creative",
            "category": "creative", 
            "keywords": ["creative", "design", "art", "story", "writing", "brainstorm",
                        "idea", "artistic", "imagination", "creative writing"],
            "patterns": ["creative writing", "design help", "brainstorming", 
                        "artistic project", "story idea", "creative process"],
            "target_agent": "creative_assistant",
            "confidence_threshold": 0.7,
            "priority": 80
        },
        {
            "name": "business",
            "category": "business",
            "keywords": ["business", "marketing", "sales", "strategy", "finance",
                        "management", "revenue", "profit", "customer", "market"],
            "patterns": ["business plan", "marketing strategy", "financial analysis",
                        "sales process", "market research", "business strategy"],
            "target_agent": "business_advisor",
            "confidence_threshold": 0.75,
            "priority": 85
        },
        {
            "name": "educational",
            "category": "educational",
            "keywords": ["learn", "education", "tutorial", "explain", "teach", "study",
                        "lesson", "course", "knowledge", "understanding"],
            "patterns": ["how to learn", "explain this", "tutorial on", "teach me",
                        "help me understand", "educational content"],
            "target_agent": "tutor",
            "confidence_threshold": 0.7,
            "priority": 75
        },
        {
            "name": "personal",
            "category": "personal",
            "keywords": ["personal", "advice", "help", "guidance", "support",
                        "recommendation", "suggestion", "opinion", "assistant"],
            "patterns": ["personal advice", "help me", "what should I", "recommend",
                        "personal assistance", "guidance on"],
            "target_agent": "personal_assistant",
            "confidence_threshold": 0.6,
            "priority": 70
        }
    ])
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate thresholds
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.min_topic_confidence <= 1:
            raise ValueError("min_topic_confidence must be between 0 and 1")
        if not 0 <= self.vector_similarity_threshold <= 1:
            raise ValueError("vector_similarity_threshold must be between 0 and 1")
        
        # Validate limits
        if self.max_topics_per_message <= 0:
            raise ValueError("max_topics_per_message must be positive")
        if self.topic_cache_ttl < 0:
            raise ValueError("topic_cache_ttl must be non-negative")
        if self.max_routes <= 0:
            raise ValueError("max_routes must be positive")
        
        # Validate detection configuration
        if not isinstance(self.detection, FFTopicDetectionConfigDTO):
            self.detection = FFTopicDetectionConfigDTO()
        
        # Validate routing strategy
        valid_strategies = [strategy.value for strategy in FFRoutingStrategy]
        if self.routing_strategy not in valid_strategies:
            raise ValueError(f"routing_strategy must be one of {valid_strategies}")
        
        # Initialize default routes if empty
        if not hasattr(self, '_routes_initialized'):
            self._initialize_default_routes()
            self._routes_initialized = True
    
    def _initialize_default_routes(self):
        """Initialize default topic routes"""
        # Convert default routes to FFTopicRoute objects
        self.topic_routes = []
        for route_data in self.default_routes:
            try:
                category = FFTopicCategory(route_data.get("category", "general"))
            except ValueError:
                category = FFTopicCategory.GENERAL
            
            route = FFTopicRoute(
                name=route_data["name"],
                category=category,
                keywords=route_data["keywords"],
                patterns=route_data["patterns"],
                target_agent=route_data["target_agent"],
                confidence_threshold=route_data["confidence_threshold"],
                priority=route_data.get("priority", 100),
                enabled=route_data.get("enabled", True),
                description=route_data.get("description", "")
            )
            self.topic_routes.append(route)
    
    def get_routes_by_category(self, category: FFTopicCategory) -> List[FFTopicRoute]:
        """Get routes for a specific category"""
        return [route for route in self.topic_routes if route.category == category]
    
    def get_enabled_routes(self) -> List[FFTopicRoute]:
        """Get all enabled routes"""
        return [route for route in self.topic_routes if route.enabled]
    
    def get_route_by_name(self, name: str) -> Optional[FFTopicRoute]:
        """Get a specific route by name"""
        for route in self.topic_routes:
            if route.name == name:
                return route
        return None
    
    def add_route(self, route: FFTopicRoute) -> bool:
        """Add a new topic route"""
        if len(self.topic_routes) >= self.max_routes:
            return False
        
        # Check for duplicate names
        if self.get_route_by_name(route.name):
            return False
        
        self.topic_routes.append(route)
        return True
    
    def remove_route(self, name: str) -> bool:
        """Remove a topic route by name"""
        for i, route in enumerate(self.topic_routes):
            if route.name == name:
                del self.topic_routes[i]
                return True
        return False
    
    def update_route(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update a topic route"""
        route = self.get_route_by_name(name)
        if not route:
            return False
        
        # Update route attributes
        for key, value in updates.items():
            if hasattr(route, key):
                setattr(route, key, value)
        
        return True


@dataclass
class FFTopicRoutingResult:
    """Result of topic routing analysis"""
    detected_topics: List[Dict[str, Any]]
    routing_decision: Dict[str, Any]
    target_agent: Optional[str]
    confidence: float
    method_used: str
    fallback_used: bool = False
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFTopicAnalysisContext:
    """Context for topic analysis"""
    session_id: str
    user_id: str
    message_content: str
    previous_messages: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Use case specific configurations
@dataclass
class FFTopicDelegationConfig:
    """Configuration for topic delegation use case"""
    enable_multi_agent_routing: bool = True
    delegation_confidence_threshold: float = 0.8
    max_delegation_depth: int = 3
    enable_delegation_tracking: bool = True
    
    # Agent capabilities mapping
    agent_capabilities: Dict[str, List[str]] = field(default_factory=lambda: {
        "technical_expert": ["programming", "software", "debugging", "architecture"],
        "creative_assistant": ["writing", "design", "brainstorming", "artistic"],
        "business_advisor": ["strategy", "finance", "marketing", "management"],
        "tutor": ["education", "explanation", "learning", "knowledge"],
        "personal_assistant": ["scheduling", "organization", "recommendations", "support"]
    })
    
    # Delegation rules
    delegation_rules: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "condition": "high_technical_complexity",
            "action": "delegate_to_technical_expert",
            "threshold": 0.9
        },
        {
            "condition": "creative_request",
            "action": "delegate_to_creative_assistant", 
            "threshold": 0.8
        },
        {
            "condition": "business_analysis_needed",
            "action": "delegate_to_business_advisor",
            "threshold": 0.85
        }
    ])


@dataclass
class FFIntelligentRoutingConfig:
    """Configuration for intelligent routing features"""
    enable_adaptive_routing: bool = True
    enable_load_balancing: bool = True
    enable_performance_based_routing: bool = True
    
    # Load balancing settings
    max_agent_load: int = 10
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, weighted
    
    # Performance-based routing
    track_agent_performance: bool = True
    performance_window_minutes: int = 60
    performance_weight: float = 0.3
    
    # Adaptive routing
    adaptation_learning_rate: float = 0.1
    min_samples_for_adaptation: int = 10
    enable_a_b_testing: bool = False