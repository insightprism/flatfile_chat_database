# Phase 2: Enhanced Panel Session Management

## 🎯 Phase Overview

Enhance your existing panel system to support PrismMind's sophisticated multi-agent collaboration capabilities. This includes advanced panel session coordination, multi-persona conversation tracking, participant insights and decision logging, and comprehensive panel analytics while maintaining full backward compatibility with your current `FFPanelManager`.

## 📋 Requirements Analysis

### **Current State Assessment**
Your system already has:
- ✅ `FFPanelManager` - Basic panel coordination
- ✅ `FFPersonaPanelConfigDTO` - Panel configuration
- ✅ `FFPanelDTO` and `FFPersonaDTO` - Panel data models
- ✅ Panel session storage structure
- ✅ Multi-persona message handling

### **Enhanced Panel Requirements**
Based on PrismMind's multi-agent use cases:
1. **Advanced Panel Coordination** - Session state management, participant tracking
2. **Multi-Persona Conversations** - Role-based messaging, conversation threading
3. **Panel Insights & Decisions** - Decision tracking, consensus analysis
4. **Participant Analytics** - Contribution analysis, engagement metrics
5. **Real-time Collaboration** - Synchronized messaging, conflict resolution

## 🏗️ Architecture Design

### **Enhanced Panel Storage Structure**
```
panel_sessions/{panel_id}/
├── panel_config.json           # Panel setup and configuration
├── panel_messages.jsonl        # Multi-persona conversation log
├── insights/                   # Panel insights and decisions
│   ├── insights.jsonl          # Decision points and conclusions
│   ├── consensus_tracking.json # Agreement/disagreement tracking
│   └── decision_history.json   # Major decisions with timestamps
├── participants/               # Individual agent contexts
│   ├── {agent_id}_context.json # Agent-specific context
│   ├── {agent_id}_memory.jsonl # Agent conversation memory
│   └── {agent_id}_analytics.json # Agent performance metrics
├── analytics/                  # Session-wide analytics
│   ├── participation_stats.json # Speaking time, contribution metrics
│   ├── topic_evolution.json    # How topics evolved during session
│   └── engagement_metrics.json # Engagement and interaction patterns
└── session_metadata.json      # Session state and coordination info
```

### **Panel Session Flow**
```
Panel Creation
      ↓
[Initialize Participants] → [Load Agent Contexts] → [Set Panel Rules]
      ↓                           ↓                        ↓
[Start Session] → [Message Exchange] → [Track Insights] → [Update Analytics]
      ↓                 ↓                    ↓                 ↓
[Decision Points] → [Consensus Check] → [Record Decisions] → [Session Summary]
      ↓
[Panel Conclusion] → [Generate Reports] → [Archive Session]
```

## 📊 Data Models

### **1. Enhanced Panel Session Configuration**

```python
# ff_class_configs/ff_panel_session_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class PanelSessionType(str, Enum):
    """Types of panel sessions."""
    MULTI_PERSONA = "multi_persona"
    AI_DEBATE = "ai_debate"
    FOCUS_GROUP = "focus_group"
    BRAINSTORM = "brainstorm"
    EXPERT_CONSULTATION = "expert_consultation"
    DECISION_MAKING = "decision_making"

class PanelParticipantRole(str, Enum):
    """Roles for panel participants."""
    MODERATOR = "moderator"
    EXPERT = "expert"
    ADVOCATE = "advocate"
    CRITIC = "critic"
    OBSERVER = "observer"
    FACILITATOR = "facilitator"

class ConsensusLevel(str, Enum):
    """Levels of consensus in panel decisions."""
    UNANIMOUS = "unanimous"
    STRONG_MAJORITY = "strong_majority"
    SIMPLE_MAJORITY = "simple_majority"
    NO_CONSENSUS = "no_consensus"
    CONFLICT = "conflict"

@dataclass
class FFPanelSessionConfigDTO:
    """Configuration for enhanced panel sessions."""
    
    # Session basic settings
    session_type: PanelSessionType = PanelSessionType.MULTI_PERSONA
    max_participants: int = 8
    min_participants: int = 2
    session_timeout_minutes: int = 120
    
    # Conversation flow settings
    enable_turn_taking: bool = True
    max_consecutive_messages_per_participant: int = 3
    require_moderator: bool = False
    enable_real_time_collaboration: bool = True
    
    # Decision tracking settings
    enable_decision_tracking: bool = True
    consensus_threshold: float = 0.7  # 70% agreement for consensus
    enable_voting: bool = False
    enable_anonymous_voting: bool = False
    
    # Analytics settings
    track_participation_metrics: bool = True
    track_topic_evolution: bool = True
    track_sentiment_analysis: bool = False
    generate_session_summaries: bool = True
    
    # Message handling
    max_message_length: int = 2000
    enable_message_threading: bool = True
    enable_message_reactions: bool = True
    
    # Insights and decision settings
    auto_capture_insights: bool = True
    insight_detection_keywords: List[str] = field(default_factory=lambda: [
        "decision", "conclusion", "consensus", "agreement", "resolution",
        "action item", "next step", "recommendation", "finding"
    ])
    
    # Performance settings
    enable_real_time_updates: bool = True
    max_concurrent_sessions: int = 10
    session_archive_days: int = 30
    
    # Integration settings
    integrate_with_memory_layers: bool = True
    memory_layer_for_insights: str = "medium_term"
    enable_cross_session_learning: bool = True
```

### **2. Enhanced Panel Data Models**

```python
# ff_class_configs/ff_chat_entities_config.py (extend existing)

@dataclass
class FFPanelParticipantDTO:
    """Enhanced panel participant with role and analytics."""
    
    # Basic participant info
    participant_id: str
    display_name: str
    participant_type: str  # "human", "ai_agent", "persona"
    role: PanelParticipantRole = PanelParticipantRole.EXPERT
    
    # Participation tracking
    joined_at: str = field(default_factory=current_timestamp)
    last_active: str = field(default_factory=current_timestamp)
    is_active: bool = True
    
    # Performance metrics
    message_count: int = 0
    total_words: int = 0
    influence_score: float = 0.5  # How much others respond to this participant
    engagement_score: float = 0.5  # How engaged this participant is
    
    # Role-specific settings
    can_moderate: bool = False
    can_make_decisions: bool = True
    can_vote: bool = True
    expertise_areas: List[str] = field(default_factory=list)
    
    # Context and memory
    participant_context: Dict[str, Any] = field(default_factory=dict)
    conversation_memory: List[str] = field(default_factory=list)  # Recent message IDs
    
    def update_activity(self, message_word_count: int = 0) -> None:
        """Update participant activity metrics."""
        self.last_active = current_timestamp()
        self.message_count += 1
        self.total_words += message_word_count

@dataclass
class FFPanelMessageDTO:
    """Enhanced panel message with threading and reactions."""
    
    # Message basics
    message_id: str = field(default_factory=generate_message_id)
    panel_id: str = ""
    participant_id: str = ""
    content: str = ""
    timestamp: str = field(default_factory=current_timestamp)
    
    # Threading and conversation flow
    thread_id: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    conversation_turn: int = 0
    
    # Message classification
    message_type: str = "conversation"  # "conversation", "decision", "insight", "question", "answer"
    topic_tags: List[str] = field(default_factory=list)
    urgency_level: str = "normal"  # "low", "normal", "high", "urgent"
    
    # Reactions and feedback
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> participant_ids
    mentions: List[str] = field(default_factory=list)  # @participant_id mentions
    
    # Analytics data
    sentiment_score: Optional[float] = None  # -1.0 (negative) to 1.0 (positive)
    influence_metrics: Dict[str, float] = field(default_factory=dict)
    response_count: int = 0  # How many messages responded to this one
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_reaction(self, emoji: str, participant_id: str) -> None:
        """Add reaction to message."""
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        if participant_id not in self.reactions[emoji]:
            self.reactions[emoji].append(participant_id)

@dataclass
class FFPanelInsightDTO:
    """Panel insight or decision point."""
    
    # Insight identification
    insight_id: str = field(default_factory=generate_insight_id)
    panel_id: str = ""
    insight_type: str = "insight"  # "insight", "decision", "consensus", "conflict", "action_item"
    
    # Content
    content: str = ""
    summary: str = ""
    context_messages: List[str] = field(default_factory=list)  # Related message IDs
    
    # Timing
    timestamp: str = field(default_factory=current_timestamp)
    discussion_duration_minutes: Optional[int] = None
    
    # Consensus tracking
    consensus_level: ConsensusLevel = ConsensusLevel.NO_CONSENSUS
    participant_positions: Dict[str, str] = field(default_factory=dict)  # participant_id -> position
    agreement_score: float = 0.0  # 0.0 to 1.0
    
    # Decision details (if applicable)
    is_decision: bool = False
    decision_text: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    responsible_parties: List[str] = field(default_factory=list)
    deadline: Optional[str] = None
    
    # Impact and follow-up
    impact_level: str = "medium"  # "low", "medium", "high", "critical"
    follow_up_required: bool = False
    related_insights: List[str] = field(default_factory=list)  # Other insight IDs
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FFPanelAnalyticsDTO:
    """Panel session analytics and metrics."""
    
    # Session info
    panel_id: str = ""
    session_start: str = field(default_factory=current_timestamp)
    session_end: Optional[str] = None
    total_duration_minutes: int = 0
    
    # Participation metrics
    participant_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    message_distribution: Dict[str, int] = field(default_factory=dict)  # participant_id -> message_count
    word_count_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Conversation flow
    total_messages: int = 0
    average_message_length: float = 0.0
    conversation_threads: int = 0
    topic_changes: int = 0
    
    # Decision and consensus metrics
    total_insights: int = 0
    total_decisions: int = 0
    consensus_achieved_count: int = 0
    average_consensus_time_minutes: float = 0.0
    
    # Engagement metrics
    reaction_counts: Dict[str, int] = field(default_factory=dict)  # emoji -> count
    mention_network: Dict[str, List[str]] = field(default_factory=dict)  # who mentioned whom
    response_patterns: Dict[str, List[str]] = field(default_factory=dict)  # who responded to whom
    
    # Topic evolution
    topic_timeline: List[Dict[str, Any]] = field(default_factory=list)
    dominant_topics: List[str] = field(default_factory=list)
    topic_transition_map: Dict[str, List[str]] = field(default_factory=dict)
    
    # Performance indicators
    panel_effectiveness_score: float = 0.0  # Overall panel performance
    decision_quality_score: float = 0.0     # Quality of decisions made
    participant_satisfaction: Dict[str, float] = field(default_factory=dict)
    
    def calculate_effectiveness_score(self) -> float:
        """Calculate overall panel effectiveness."""
        # Combine various metrics into effectiveness score
        decision_rate = self.total_decisions / max(1, self.total_duration_minutes / 60)
        consensus_rate = self.consensus_achieved_count / max(1, self.total_decisions)
        participation_balance = 1.0 - self._calculate_participation_inequality()
        
        effectiveness = (decision_rate * 0.4 + consensus_rate * 0.3 + participation_balance * 0.3)
        self.panel_effectiveness_score = min(1.0, effectiveness)
        return self.panel_effectiveness_score
    
    def _calculate_participation_inequality(self) -> float:
        """Calculate inequality in participation (Gini coefficient)."""
        if not self.message_distribution:
            return 0.0
        
        message_counts = list(self.message_distribution.values())
        if len(message_counts) <= 1:
            return 0.0
        
        # Simple inequality measure
        total_messages = sum(message_counts)
        expected_per_participant = total_messages / len(message_counts)
        
        inequality = sum(abs(count - expected_per_participant) for count in message_counts)
        return inequality / (2 * total_messages)
```

## 🔧 Implementation Specifications

### **1. Enhanced Panel Session Manager**

```python
# ff_panel_session_manager.py

"""
Enhanced panel session management system.

Provides sophisticated multi-agent collaboration capabilities including
session coordination, participant tracking, insight capture, and analytics.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_panel_session_config import (
    FFPanelSessionConfigDTO, PanelSessionType, PanelParticipantRole, ConsensusLevel
)
from ff_class_configs.ff_chat_entities_config import (
    FFPanelParticipantDTO, FFPanelMessageDTO, FFPanelInsightDTO, FFPanelAnalyticsDTO
)
from ff_utils.ff_file_ops import ff_atomic_write, ff_ensure_directory
from ff_utils.ff_json_utils import ff_read_jsonl, ff_append_jsonl, ff_write_json, ff_read_json
from ff_utils.ff_logging import get_logger

class FFPanelSessionManager:
    """
    Enhanced panel session management following flatfile patterns.
    
    Manages sophisticated multi-agent collaboration including:
    - Advanced session coordination and state management
    - Multi-persona conversation tracking with threading
    - Real-time insight capture and decision logging
    - Comprehensive analytics and participant metrics
    - Integration with memory layers for long-term learning
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO):
        """Initialize enhanced panel session manager."""
        self.config = config
        self.panel_config = getattr(config, 'panel_session', FFPanelSessionConfigDTO())
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Active session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Analytics cache
        self._analytics_cache: Dict[str, FFPanelAnalyticsDTO] = {}
        
    def _get_panel_session_path(self, panel_id: str) -> Path:
        """Get panel session directory path."""
        return self.base_path / "panel_sessions" / panel_id
    
    async def create_panel_session(
        self,
        panel_id: str,
        session_type: PanelSessionType,
        participants: List[FFPanelParticipantDTO],
        session_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create new enhanced panel session."""
        try:
            panel_path = self._get_panel_session_path(panel_id)
            await ff_ensure_directory(panel_path)
            
            # Create session directories
            for subdir in ["insights", "participants", "analytics"]:
                await ff_ensure_directory(panel_path / subdir)
            
            # Create panel configuration
            panel_config = {
                "panel_id": panel_id,
                "session_type": session_type.value,
                "created_at": datetime.now().isoformat(),
                "status": "initialized",
                "participants": [p.to_dict() for p in participants],
                "config": session_config or {},
                "session_rules": self._generate_session_rules(session_type, len(participants))
            }
            
            await ff_write_json(panel_path / "panel_config.json", panel_config, self.config)
            
            # Initialize participant contexts
            for participant in participants:
                participant_file = panel_path / "participants" / f"{participant.participant_id}_context.json"
                participant_context = {
                    "participant_id": participant.participant_id,
                    "context": participant.participant_context,
                    "conversation_history": [],
                    "performance_metrics": {
                        "messages_sent": 0,
                        "words_contributed": 0,
                        "insights_generated": 0,
                        "decisions_influenced": 0
                    }
                }
                await ff_write_json(participant_file, participant_context, self.config)
            
            # Initialize session metadata
            session_metadata = {
                "panel_id": panel_id,
                "session_state": "created",
                "current_turn": 0,
                "active_participants": [p.participant_id for p in participants],
                "conversation_threads": [],
                "pending_decisions": [],
                "session_timeline": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "session_created",
                        "details": {"participant_count": len(participants)}
                    }
                ]
            }
            
            await ff_write_json(panel_path / "session_metadata.json", session_metadata, self.config)
            
            # Initialize analytics
            analytics = FFPanelAnalyticsDTO(
                panel_id=panel_id,
                participant_stats={p.participant_id: {
                    "role": p.role.value,
                    "messages": 0,
                    "words": 0,
                    "influence": 0.5,
                    "engagement": 0.5
                } for p in participants}
            )
            
            await ff_write_json(panel_path / "analytics" / "participation_stats.json", 
                              analytics.to_dict(), self.config)
            
            # Track active session
            self.active_sessions[panel_id] = {
                "status": "created",
                "participants": {p.participant_id: p for p in participants},
                "created_at": datetime.now(),
                "message_count": 0
            }
            
            self.session_locks[panel_id] = asyncio.Lock()
            
            self.logger.info(f"Created panel session {panel_id} with {len(participants)} participants")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create panel session {panel_id}: {e}")
            return False
    
    async def add_panel_message(
        self,
        panel_id: str,
        participant_id: str,
        content: str,
        message_type: str = "conversation",
        reply_to_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[FFPanelMessageDTO]:
        """Add message to panel session with threading and analytics."""
        try:
            if panel_id not in self.session_locks:
                self.logger.error(f"Panel session {panel_id} not found")
                return None
            
            async with self.session_locks[panel_id]:
                # Create message
                message = FFPanelMessageDTO(
                    panel_id=panel_id,
                    participant_id=participant_id,
                    content=content,
                    message_type=message_type,
                    reply_to_message_id=reply_to_message_id,
                    metadata=metadata or {}
                )
                
                # Set conversation turn
                session_metadata = await self._load_session_metadata(panel_id)
                message.conversation_turn = session_metadata.get("current_turn", 0) + 1
                
                # Handle threading
                if reply_to_message_id:
                    # Find the thread this message belongs to
                    thread_id = await self._find_or_create_thread(panel_id, reply_to_message_id)
                    message.thread_id = thread_id
                
                # Add to messages file
                panel_path = self._get_panel_session_path(panel_id)
                await ff_append_jsonl(panel_path / "panel_messages.jsonl", 
                                    [message.to_dict()], self.config)
                
                # Update session metadata
                await self._update_session_metadata(panel_id, message)
                
                # Update participant analytics
                await self._update_participant_analytics(panel_id, participant_id, message)
                
                # Check for insights
                if self.panel_config.auto_capture_insights:
                    await self._check_for_insights(panel_id, message)
                
                # Update active session tracking
                if panel_id in self.active_sessions:
                    self.active_sessions[panel_id]["message_count"] += 1
                
                self.logger.debug(f"Added message to panel {panel_id} from {participant_id}")
                return message
                
        except Exception as e:
            self.logger.error(f"Failed to add message to panel {panel_id}: {e}")
            return None
    
    async def capture_panel_insight(
        self,
        panel_id: str,
        insight_content: str,
        insight_type: str = "insight",
        context_message_ids: Optional[List[str]] = None,
        participant_positions: Optional[Dict[str, str]] = None
    ) -> Optional[FFPanelInsightDTO]:
        """Capture panel insight or decision point."""
        try:
            insight = FFPanelInsightDTO(
                panel_id=panel_id,
                insight_type=insight_type,
                content=insight_content,
                context_messages=context_message_ids or [],
                participant_positions=participant_positions or {}
            )
            
            # Calculate consensus if participant positions provided
            if participant_positions:
                insight.consensus_level, insight.agreement_score = self._calculate_consensus(
                    participant_positions
                )
            
            # Determine if this is a decision
            insight.is_decision = insight_type in ["decision", "consensus"]
            
            # Extract action items if it's a decision
            if insight.is_decision:
                insight.action_items = self._extract_action_items(insight_content)
            
            # Save insight
            panel_path = self._get_panel_session_path(panel_id)
            await ff_append_jsonl(panel_path / "insights" / "insights.jsonl",
                                [insight.to_dict()], self.config)
            
            # Update analytics
            await self._update_insight_analytics(panel_id, insight)
            
            self.logger.info(f"Captured {insight_type} for panel {panel_id}: {insight.insight_id}")
            return insight
            
        except Exception as e:
            self.logger.error(f"Failed to capture insight for panel {panel_id}: {e}")
            return None
    
    async def get_panel_messages(
        self,
        panel_id: str,
        limit: Optional[int] = None,
        participant_filter: Optional[List[str]] = None,
        message_type_filter: Optional[List[str]] = None,
        thread_id: Optional[str] = None
    ) -> List[FFPanelMessageDTO]:
        """Get panel messages with filtering options."""
        try:
            panel_path = self._get_panel_session_path(panel_id)
            messages_file = panel_path / "panel_messages.jsonl"
            
            if not messages_file.exists():
                return []
            
            message_data = await ff_read_jsonl(messages_file, self.config)
            messages = [FFPanelMessageDTO.from_dict(data) for data in message_data]
            
            # Apply filters
            filtered_messages = messages
            
            if participant_filter:
                filtered_messages = [m for m in filtered_messages 
                                   if m.participant_id in participant_filter]
            
            if message_type_filter:
                filtered_messages = [m for m in filtered_messages 
                                   if m.message_type in message_type_filter]
            
            if thread_id:
                filtered_messages = [m for m in filtered_messages 
                                   if m.thread_id == thread_id]
            
            # Sort by timestamp
            filtered_messages.sort(key=lambda x: x.timestamp)
            
            # Apply limit
            if limit:
                filtered_messages = filtered_messages[-limit:]
            
            return filtered_messages
            
        except Exception as e:
            self.logger.error(f"Failed to get messages for panel {panel_id}: {e}")
            return []
    
    async def get_panel_insights(
        self,
        panel_id: str,
        insight_type_filter: Optional[List[str]] = None,
        include_decisions_only: bool = False
    ) -> List[FFPanelInsightDTO]:
        """Get panel insights and decisions."""
        try:
            panel_path = self._get_panel_session_path(panel_id)
            insights_file = panel_path / "insights" / "insights.jsonl"
            
            if not insights_file.exists():
                return []
            
            insight_data = await ff_read_jsonl(insights_file, self.config)
            insights = [FFPanelInsightDTO.from_dict(data) for data in insight_data]
            
            # Apply filters
            filtered_insights = insights
            
            if insight_type_filter:
                filtered_insights = [i for i in filtered_insights 
                                   if i.insight_type in insight_type_filter]
            
            if include_decisions_only:
                filtered_insights = [i for i in filtered_insights if i.is_decision]
            
            # Sort by timestamp
            filtered_insights.sort(key=lambda x: x.timestamp)
            
            return filtered_insights
            
        except Exception as e:
            self.logger.error(f"Failed to get insights for panel {panel_id}: {e}")
            return []
    
    async def get_panel_analytics(self, panel_id: str) -> Optional[FFPanelAnalyticsDTO]:
        """Get comprehensive panel analytics."""
        try:
            # Check cache first
            if panel_id in self._analytics_cache:
                return self._analytics_cache[panel_id]
            
            # Load or generate analytics
            analytics = await self._generate_panel_analytics(panel_id)
            
            if analytics:
                self._analytics_cache[panel_id] = analytics
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics for panel {panel_id}: {e}")
            return None
    
    async def end_panel_session(self, panel_id: str) -> Dict[str, Any]:
        """End panel session and generate final analytics."""
        try:
            # Update session status
            session_metadata = await self._load_session_metadata(panel_id)
            session_metadata["session_state"] = "ended"
            session_metadata["ended_at"] = datetime.now().isoformat()
            
            panel_path = self._get_panel_session_path(panel_id)
            await ff_write_json(panel_path / "session_metadata.json", session_metadata, self.config)
            
            # Generate final analytics
            final_analytics = await self._generate_panel_analytics(panel_id)
            if final_analytics:
                final_analytics.session_end = datetime.now().isoformat()
                final_analytics.calculate_effectiveness_score()
                
                await ff_write_json(panel_path / "analytics" / "final_analytics.json",
                                  final_analytics.to_dict(), self.config)
            
            # Clean up active session tracking
            if panel_id in self.active_sessions:
                del self.active_sessions[panel_id]
            
            if panel_id in self.session_locks:
                del self.session_locks[panel_id]
            
            # Generate session summary
            summary = await self._generate_session_summary(panel_id)
            
            self.logger.info(f"Ended panel session {panel_id}")
            return {
                "success": True,
                "panel_id": panel_id,
                "ended_at": datetime.now().isoformat(),
                "final_analytics": final_analytics.to_dict() if final_analytics else None,
                "session_summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to end panel session {panel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    # Private helper methods
    
    async def _load_session_metadata(self, panel_id: str) -> Dict[str, Any]:
        """Load session metadata from file."""
        panel_path = self._get_panel_session_path(panel_id)
        metadata_file = panel_path / "session_metadata.json"
        
        if metadata_file.exists():
            return await ff_read_json(metadata_file, self.config)
        return {}
    
    async def _update_session_metadata(self, panel_id: str, message: FFPanelMessageDTO) -> None:
        """Update session metadata with new message."""
        metadata = await self._load_session_metadata(panel_id)
        
        metadata["current_turn"] = message.conversation_turn
        metadata["last_message_at"] = message.timestamp
        metadata["last_participant"] = message.participant_id
        
        # Add to timeline
        if "session_timeline" not in metadata:
            metadata["session_timeline"] = []
        
        metadata["session_timeline"].append({
            "timestamp": message.timestamp,
            "event": "message_added",
            "participant": message.participant_id,
            "message_type": message.message_type
        })
        
        panel_path = self._get_panel_session_path(panel_id)
        await ff_write_json(panel_path / "session_metadata.json", metadata, self.config)
    
    def _generate_session_rules(self, session_type: PanelSessionType, participant_count: int) -> Dict[str, Any]:
        """Generate session rules based on type and participant count."""
        base_rules = {
            "max_message_length": self.panel_config.max_message_length,
            "enable_reactions": self.panel_config.enable_message_reactions,
            "enable_threading": self.panel_config.enable_message_threading
        }
        
        if session_type == PanelSessionType.AI_DEBATE:
            base_rules.update({
                "require_alternating_turns": True,
                "max_consecutive_messages": 2,
                "enable_position_tracking": True
            })
        elif session_type == PanelSessionType.BRAINSTORM:
            base_rules.update({
                "encourage_rapid_fire": True,
                "max_consecutive_messages": 1,
                "enable_idea_building": True
            })
        elif session_type == PanelSessionType.DECISION_MAKING:
            base_rules.update({
                "require_consensus_check": True,
                "enable_voting": True,
                "track_decision_process": True
            })
        
        return base_rules
    
    def _calculate_consensus(self, participant_positions: Dict[str, str]) -> Tuple[ConsensusLevel, float]:
        """Calculate consensus level and agreement score."""
        if not participant_positions:
            return ConsensusLevel.NO_CONSENSUS, 0.0
        
        # Count position agreements
        position_counts = Counter(participant_positions.values())
        total_participants = len(participant_positions)
        
        if len(position_counts) == 1:
            return ConsensusLevel.UNANIMOUS, 1.0
        
        max_agreement = max(position_counts.values())
        agreement_ratio = max_agreement / total_participants
        
        if agreement_ratio >= 0.8:
            return ConsensusLevel.STRONG_MAJORITY, agreement_ratio
        elif agreement_ratio >= 0.6:
            return ConsensusLevel.SIMPLE_MAJORITY, agreement_ratio
        else:
            return ConsensusLevel.NO_CONSENSUS, agreement_ratio
    
    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from decision content."""
        # Simple extraction based on common patterns
        action_indicators = ["action item:", "todo:", "task:", "follow up:", "next step:"]
        
        action_items = []
        lines = content.lower().split('\n')
        
        for line in lines:
            for indicator in action_indicators:
                if indicator in line:
                    action_item = line.split(indicator, 1)[1].strip()
                    if action_item:
                        action_items.append(action_item)
        
        return action_items
```

### **2. Panel Session Protocol Interface**

```python
# ff_protocols/ff_panel_session_protocol.py

"""Protocol interface for enhanced panel session management."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ff_class_configs.ff_panel_session_config import PanelSessionType
from ff_class_configs.ff_chat_entities_config import (
    FFPanelParticipantDTO, FFPanelMessageDTO, FFPanelInsightDTO, FFPanelAnalyticsDTO
)

class PanelSessionProtocol(ABC):
    """Protocol interface for panel session management operations."""
    
    @abstractmethod
    async def create_panel_session(
        self,
        panel_id: str,
        session_type: PanelSessionType,
        participants: List[FFPanelParticipantDTO],
        session_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create new enhanced panel session."""
        pass
    
    @abstractmethod
    async def add_panel_message(
        self,
        panel_id: str,
        participant_id: str,
        content: str,
        message_type: str = "conversation",
        reply_to_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[FFPanelMessageDTO]:
        """Add message to panel session with threading and analytics."""
        pass
    
    @abstractmethod
    async def capture_panel_insight(
        self,
        panel_id: str,
        insight_content: str,
        insight_type: str = "insight",
        context_message_ids: Optional[List[str]] = None,
        participant_positions: Optional[Dict[str, str]] = None
    ) -> Optional[FFPanelInsightDTO]:
        """Capture panel insight or decision point."""
        pass
    
    @abstractmethod
    async def get_panel_messages(
        self,
        panel_id: str,
        limit: Optional[int] = None,
        participant_filter: Optional[List[str]] = None,
        message_type_filter: Optional[List[str]] = None,
        thread_id: Optional[str] = None
    ) -> List[FFPanelMessageDTO]:
        """Get panel messages with filtering options."""
        pass
    
    @abstractmethod
    async def get_panel_insights(
        self,
        panel_id: str,
        insight_type_filter: Optional[List[str]] = None,
        include_decisions_only: bool = False
    ) -> List[FFPanelInsightDTO]:
        """Get panel insights and decisions."""
        pass
    
    @abstractmethod
    async def get_panel_analytics(self, panel_id: str) -> Optional[FFPanelAnalyticsDTO]:
        """Get comprehensive panel analytics."""
        pass
    
    @abstractmethod
    async def end_panel_session(self, panel_id: str) -> Dict[str, Any]:
        """End panel session and generate final analytics."""
        pass
```

### **3. Integration with Configuration Manager**

```python
# ff_class_configs/ff_configuration_manager_config.py (extend existing)

# Add to existing FFConfigurationManagerConfigDTO class:

@dataclass
class FFConfigurationManagerConfigDTO:
    # ... existing fields ...
    
    panel_session: FFPanelSessionConfigDTO = field(default_factory=FFPanelSessionConfigDTO)
    
    # ... rest of existing implementation ...
```

## 🧪 Testing Specifications

### **Unit Tests**

```python
# tests/test_panel_session_manager.py

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ff_panel_session_manager import FFPanelSessionManager
from ff_class_configs.ff_panel_session_config import FFPanelSessionConfigDTO, PanelSessionType, PanelParticipantRole
from ff_class_configs.ff_chat_entities_config import FFPanelParticipantDTO, FFPanelMessageDTO

class TestPanelSessionManager:
    
    @pytest.fixture
    async def panel_manager(self):
        """Create panel session manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FFConfigurationManagerConfigDTO()
            config.storage.base_path = temp_dir
            config.panel_session = FFPanelSessionConfigDTO()
            
            manager = FFPanelSessionManager(config)
            yield manager
    
    @pytest.fixture
    def sample_participants(self):
        """Create sample participants for testing."""
        return [
            FFPanelParticipantDTO(
                participant_id="expert_1",
                display_name="AI Ethics Expert",
                participant_type="ai_agent",
                role=PanelParticipantRole.EXPERT,
                expertise_areas=["ethics", "ai_safety"]
            ),
            FFPanelParticipantDTO(
                participant_id="critic_1", 
                display_name="Technology Critic",
                participant_type="ai_agent",
                role=PanelParticipantRole.CRITIC,
                expertise_areas=["technology", "society"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_create_panel_session(self, panel_manager, sample_participants):
        """Test panel session creation."""
        panel_id = "test_panel_ethics_debate"
        
        success = await panel_manager.create_panel_session(
            panel_id=panel_id,
            session_type=PanelSessionType.AI_DEBATE,
            participants=sample_participants
        )
        
        assert success
        assert panel_id in panel_manager.active_sessions
        
        # Check directory structure
        panel_path = panel_manager._get_panel_session_path(panel_id)
        assert panel_path.exists()
        assert (panel_path / "panel_config.json").exists()
        assert (panel_path / "session_metadata.json").exists()
        assert (panel_path / "insights").exists()
        assert (panel_path / "participants").exists()
        assert (panel_path / "analytics").exists()
    
    @pytest.mark.asyncio
    async def test_add_panel_message(self, panel_manager, sample_participants):
        """Test adding messages to panel session."""
        panel_id = "test_panel_conversation"
        
        # Create session
        await panel_manager.create_panel_session(
            panel_id, PanelSessionType.MULTI_PERSONA, sample_participants
        )
        
        # Add message
        message = await panel_manager.add_panel_message(
            panel_id=panel_id,
            participant_id="expert_1",
            content="I believe AI ethics requires careful consideration of bias.",
            message_type="conversation"
        )
        
        assert message is not None
        assert message.participant_id == "expert_1"
        assert message.panel_id == panel_id
        assert message.conversation_turn == 1
        
        # Add reply message
        reply_message = await panel_manager.add_panel_message(
            panel_id=panel_id,
            participant_id="critic_1",
            content="That's an important point, but we also need to consider economic impacts.",
            message_type="conversation",
            reply_to_message_id=message.message_id
        )
        
        assert reply_message is not None
        assert reply_message.reply_to_message_id == message.message_id
        assert reply_message.conversation_turn == 2
    
    @pytest.mark.asyncio
    async def test_capture_panel_insight(self, panel_manager, sample_participants):
        """Test capturing panel insights and decisions."""
        panel_id = "test_panel_insights"
        
        await panel_manager.create_panel_session(
            panel_id, PanelSessionType.DECISION_MAKING, sample_participants
        )
        
        # Capture insight
        insight = await panel_manager.capture_panel_insight(
            panel_id=panel_id,
            insight_content="We need to establish ethical guidelines before deployment.",
            insight_type="decision",
            participant_positions={
                "expert_1": "strongly_agree",
                "critic_1": "agree_with_conditions"
            }
        )
        
        assert insight is not None
        assert insight.insight_type == "decision"
        assert insight.is_decision is True
        assert insight.consensus_level is not None
        assert insight.agreement_score > 0.0
    
    @pytest.mark.asyncio
    async def test_get_panel_messages(self, panel_manager, sample_participants):
        """Test retrieving panel messages with filters."""
        panel_id = "test_panel_retrieval"
        
        await panel_manager.create_panel_session(
            panel_id, PanelSessionType.MULTI_PERSONA, sample_participants
        )
        
        # Add multiple messages
        messages = []
        for i in range(5):
            message = await panel_manager.add_panel_message(
                panel_id=panel_id,
                participant_id=sample_participants[i % 2].participant_id,
                content=f"Message {i+1} from {sample_participants[i % 2].display_name}",
                message_type="conversation"
            )
            messages.append(message)
        
        # Test retrieval
        all_messages = await panel_manager.get_panel_messages(panel_id)
        assert len(all_messages) == 5
        
        # Test participant filter
        expert_messages = await panel_manager.get_panel_messages(
            panel_id, participant_filter=["expert_1"]
        )
        assert len(expert_messages) == 3  # Messages 1, 3, 5
        
        # Test limit
        limited_messages = await panel_manager.get_panel_messages(panel_id, limit=3)
        assert len(limited_messages) == 3
    
    @pytest.mark.asyncio
    async def test_panel_analytics(self, panel_manager, sample_participants):
        """Test panel analytics generation."""
        panel_id = "test_panel_analytics"
        
        await panel_manager.create_panel_session(
            panel_id, PanelSessionType.BRAINSTORM, sample_participants
        )
        
        # Add messages and insights to generate analytics data
        for i in range(10):
            await panel_manager.add_panel_message(
                panel_id=panel_id,
                participant_id=sample_participants[i % 2].participant_id,
                content=f"Analytics test message {i+1}",
                message_type="conversation"
            )
        
        await panel_manager.capture_panel_insight(
            panel_id=panel_id,
            insight_content="Key insight from brainstorm session",
            insight_type="insight"
        )
        
        # Get analytics
        analytics = await panel_manager.get_panel_analytics(panel_id)
        
        assert analytics is not None
        assert analytics.panel_id == panel_id
        assert analytics.total_messages == 10
        assert analytics.total_insights == 1
        assert len(analytics.participant_stats) == 2
    
    @pytest.mark.asyncio
    async def test_end_panel_session(self, panel_manager, sample_participants):
        """Test ending panel session."""
        panel_id = "test_panel_ending"
        
        await panel_manager.create_panel_session(
            panel_id, PanelSessionType.FOCUS_GROUP, sample_participants
        )
        
        # Add some activity
        await panel_manager.add_panel_message(
            panel_id=panel_id,
            participant_id="expert_1",
            content="Final thoughts on our discussion",
            message_type="conversation"
        )
        
        # End session
        result = await panel_manager.end_panel_session(panel_id)
        
        assert result["success"] is True
        assert result["panel_id"] == panel_id
        assert "final_analytics" in result
        assert "session_summary" in result
        assert panel_id not in panel_manager.active_sessions
```

### **Integration Tests**

```python
# tests/test_panel_integration.py

class TestPanelIntegration:
    
    @pytest.mark.asyncio
    async def test_panel_memory_integration(self):
        """Test integration between panel sessions and memory layers."""
        # Test that panel insights automatically promote to memory layers
        pass
    
    @pytest.mark.asyncio
    async def test_multi_session_analytics(self):
        """Test analytics across multiple panel sessions."""
        # Test cross-session learning and pattern recognition
        pass
    
    @pytest.mark.asyncio
    async def test_real_time_collaboration(self):
        """Test real-time collaboration features."""
        # Test concurrent message handling and conflict resolution
        pass
```

## 📈Success Criteria

### **Functional Requirements**
- ✅ Enhanced panel sessions support all PrismMind multi-agent use cases
- ✅ Message threading and conversation flow management operational
- ✅ Insight capture and decision tracking with consensus analysis
- ✅ Comprehensive analytics provide actionable participant and session metrics
- ✅ Integration with existing panel system maintains backward compatibility

### **Performance Requirements**
- ✅ Panel message operations complete within 100ms
- ✅ Analytics generation completes within 5 seconds for typical sessions
- ✅ Concurrent panel sessions supported (up to configured limit)
- ✅ Real-time updates delivered within 500ms

### **Integration Requirements**
- ✅ Existing `FFPanelManager` functionality enhanced, not replaced
- ✅ Panel sessions integrate with memory layers for long-term learning
- ✅ All operations follow existing async patterns and error handling
- ✅ Configuration-driven behavior with comprehensive settings

### **Testing Requirements**
- ✅ Unit test coverage > 90% for all new components
- ✅ Integration tests validate panel session workflows
- ✅ Performance tests validate concurrent session handling
- ✅ End-to-end tests validate complete multi-agent collaboration scenarios

## 🚀 Implementation Checklist

### **Phase 2A: Enhanced Data Models**
- [ ] Create `ff_panel_session_config.py` with comprehensive configuration DTOs
- [ ] Extend `ff_chat_entities_config.py` with enhanced panel entities
- [ ] Create `ff_panel_session_protocol.py` with abstract interface
- [ ] Update configuration manager to include panel session config

### **Phase 2B: Panel Session Manager**
- [ ] Implement `FFPanelSessionManager` with enhanced coordination
- [ ] Add message threading and conversation flow management
- [ ] Implement insight capture and decision tracking
- [ ] Add comprehensive analytics and metrics collection

### **Phase 2C: Advanced Features**
- [ ] Implement consensus tracking and decision analysis
- [ ] Add participant performance analytics
- [ ] Create session summary and report generation
- [ ] Add real-time collaboration features

### **Phase 2D: Integration & Testing**
- [ ] Integrate with existing panel management system
- [ ] Update dependency injection container registration
- [ ] Create comprehensive unit test suite
- [ ] Create integration tests with memory layers
- [ ] Performance test concurrent panel sessions
- [ ] Validate backward compatibility

### **Phase 2E: Documentation & Validation**
- [ ] Update panel system documentation
- [ ] Create multi-agent collaboration usage examples
- [ ] Validate all success criteria met
- [ ] Performance benchmark panel operations
- [ ] Create migration guide for existing panel data

This specification provides comprehensive guidance for enhancing your panel system to support sophisticated multi-agent collaboration while maintaining your excellent architectural standards.