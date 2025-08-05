"""
FF Topic Router Component - Intelligent Topic Detection and Routing

Provides intelligent topic detection and routing capabilities for the FF Chat System,
using FF search manager for topic analysis and content classification.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_topic_router_config import FFTopicRouterConfigDTO, FFTopicDetectionMethod, FFRoutingStrategy
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_protocols.ff_message_dto import FFMessageDTO
from ff_managers.ff_search_manager import FFSearchManager
from ff_managers.ff_storage_manager import FFStorageManager
from ff_managers.ff_document_manager import FFDocumentManager

logger = get_logger(__name__)


class FFTopicConfidence(Enum):
    """Topic detection confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FFDetectedTopic:
    """Detected topic information"""
    topic_id: str
    topic_name: str
    confidence: float
    detection_method: FFTopicDetectionMethod
    keywords: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    routing_target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFRoutingDecision:
    """Routing decision result"""
    route_to: str
    confidence: float
    reasoning: str
    alternative_routes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFTopicAnalysis:
    """Complete topic analysis result"""
    message_id: str
    detected_topics: List[FFDetectedTopic]
    primary_topic: Optional[FFDetectedTopic]
    routing_decision: Optional[FFRoutingDecision]
    analysis_time_ms: float
    context: Dict[str, Any] = field(default_factory=dict)


class FFTopicRouterComponent(FFChatComponentProtocol):
    """
    FF Topic Router Component for intelligent topic detection and routing.
    
    Provides topic analysis, classification, and intelligent routing decisions
    using FF search capabilities for semantic understanding.
    """
    
    def __init__(self, 
                 config: FFTopicRouterConfigDTO,
                 search_manager: FFSearchManager,
                 storage_manager: FFStorageManager,
                 document_manager: FFDocumentManager):
        """
        Initialize FF Topic Router Component.
        
        Args:
            config: Topic router configuration
            search_manager: FF search manager for semantic analysis
            storage_manager: FF storage manager for data persistence
            document_manager: FF document manager for content processing
        """
        self.config = config
        self.search_manager = search_manager
        self.storage_manager = storage_manager
        self.document_manager = document_manager
        self.logger = get_logger(__name__)
        
        # Topic management
        self.topic_definitions: Dict[str, Dict[str, Any]] = {}
        self.routing_rules: Dict[str, Dict[str, Any]] = {}
        self.topic_cache: Dict[str, FFTopicAnalysis] = {}
        self.cache_ttl_seconds = config.cache_ttl_seconds
        
        # Performance tracking
        self.analysis_metrics: Dict[str, Any] = {
            "total_analyses": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_analysis_time_ms": 0.0,
            "topic_confidence_distribution": {confidence.value: 0 for confidence in FFTopicConfidence}
        }
        
        # Learning system (if enabled)
        self.learning_data: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        
        # Initialize built-in topics and routing rules
        self._initialize_built_in_topics()
        self._initialize_routing_rules()
    
    async def process_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message: FFMessageDTO, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message for topic detection and routing.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: Message to analyze
            context: Additional context information
            
        Returns:
            Dict containing topic analysis and routing decision
        """
        start_time = time.time()
        
        try:
            # Extract message content
            content = message.content if hasattr(message, 'content') else str(message)
            
            # Generate message hash for caching
            message_hash = self._generate_message_hash(content, context)
            
            # Check cache first
            cached_analysis = self._get_cached_analysis(message_hash)
            if cached_analysis:
                self.logger.debug(f"Using cached topic analysis for message hash {message_hash}")
                return self._format_analysis_result(cached_analysis)
            
            # Perform topic detection
            detected_topics = await self._detect_topics(content, context)
            
            # Determine primary topic
            primary_topic = self._determine_primary_topic(detected_topics)
            
            # Make routing decision
            routing_decision = await self._make_routing_decision(primary_topic, detected_topics, context)
            
            # Create analysis result
            analysis_time_ms = (time.time() - start_time) * 1000
            analysis = FFTopicAnalysis(
                message_id=getattr(message, 'message_id', f"msg_{int(time.time())}"),
                detected_topics=detected_topics,
                primary_topic=primary_topic,
                routing_decision=routing_decision,
                analysis_time_ms=analysis_time_ms,
                context=context or {}
            )
            
            # Cache analysis
            self._cache_analysis(message_hash, analysis)
            
            # Update metrics
            self._update_metrics(analysis)
            
            # Store analysis for learning (if enabled)
            if self.config.enable_learning:
                await self._store_analysis_for_learning(session_id, user_id, analysis)
            
            self.logger.info(f"Topic analysis completed in {analysis_time_ms:.2f}ms. Primary topic: {primary_topic.topic_name if primary_topic else 'None'}")
            
            return self._format_analysis_result(analysis)
            
        except Exception as e:
            self.logger.error(f"Topic analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detected_topics": [],
                "primary_topic": None,
                "routing_decision": None,
                "analysis_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _detect_topics(self, content: str, context: Optional[Dict[str, Any]]) -> List[FFDetectedTopic]:
        """Detect topics in the given content using multiple methods"""
        detected_topics = []
        
        try:
            # Apply each enabled detection method
            for method in self.config.detection.detection_methods:
                method_topics = await self._apply_detection_method(method, content, context)
                detected_topics.extend(method_topics)
            
            # Remove duplicates and merge similar topics
            detected_topics = self._merge_similar_topics(detected_topics)
            
            # Filter by confidence threshold
            detected_topics = [
                topic for topic in detected_topics 
                if topic.confidence >= self.config.confidence_threshold
            ]
            
            # Sort by confidence
            detected_topics.sort(key=lambda t: t.confidence, reverse=True)
            
            return detected_topics
            
        except Exception as e:
            self.logger.error(f"Topic detection failed: {e}")
            return []
    
    async def _apply_detection_method(self, 
                                    method: FFTopicDetectionMethod, 
                                    content: str, 
                                    context: Optional[Dict[str, Any]]) -> List[FFDetectedTopic]:
        """Apply a specific detection method to content"""
        try:
            if method == FFTopicDetectionMethod.KEYWORD:
                return await self._detect_topics_by_keywords(content)
            
            elif method == FFTopicDetectionMethod.PATTERN:
                return await self._detect_topics_by_patterns(content)
            
            elif method == FFTopicDetectionMethod.SEMANTIC:
                return await self._detect_topics_by_semantic_search(content, context)
            
            elif method == FFTopicDetectionMethod.ML_CLASSIFICATION:
                return await self._detect_topics_by_ml_classification(content, context)
            
            elif method == FFTopicDetectionMethod.HYBRID:
                return await self._detect_topics_by_hybrid_approach(content, context)
            
            else:
                self.logger.warning(f"Unknown detection method: {method}")
                return []
                
        except Exception as e:
            self.logger.error(f"Detection method {method} failed: {e}")
            return []
    
    async def _detect_topics_by_keywords(self, content: str) -> List[FFDetectedTopic]:
        """Detect topics using keyword matching"""
        detected_topics = []
        content_lower = content.lower()
        
        for topic_id, topic_def in self.topic_definitions.items():
            keywords = topic_def.get("keywords", [])
            matched_keywords = []
            total_matches = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                matches = content_lower.count(keyword_lower)
                if matches > 0:
                    matched_keywords.append(keyword)
                    total_matches += matches
            
            if matched_keywords:
                # Calculate confidence based on keyword matches
                confidence = min(0.95, (len(matched_keywords) / len(keywords)) * 0.7 + (total_matches / len(content.split())) * 0.3)
                
                detected_topics.append(FFDetectedTopic(
                    topic_id=topic_id,
                    topic_name=topic_def["name"],
                    confidence=confidence,
                    detection_method=FFTopicDetectionMethod.KEYWORD,
                    keywords=matched_keywords,
                    context={"total_matches": total_matches},
                    routing_target=topic_def.get("default_route")
                ))
        
        return detected_topics
    
    async def _detect_topics_by_patterns(self, content: str) -> List[FFDetectedTopic]:
        """Detect topics using regex patterns"""
        import re
        detected_topics = []
        
        for topic_id, topic_def in self.topic_definitions.items():
            patterns = topic_def.get("patterns", [])
            matched_patterns = []
            
            for pattern_str in patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    matches = pattern.findall(content)
                    if matches:
                        matched_patterns.append({
                            "pattern": pattern_str,
                            "matches": matches
                        })
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern {pattern_str}: {e}")
                    continue
            
            if matched_patterns:
                # Calculate confidence based on pattern matches
                confidence = min(0.90, len(matched_patterns) / max(1, len(patterns)) * 0.8)
                
                detected_topics.append(FFDetectedTopic(
                    topic_id=topic_id,
                    topic_name=topic_def["name"],
                    confidence=confidence,
                    detection_method=FFTopicDetectionMethod.PATTERN,
                    keywords=[],
                    context={"matched_patterns": matched_patterns},
                    routing_target=topic_def.get("default_route")
                ))
        
        return detected_topics
    
    async def _detect_topics_by_semantic_search(self, content: str, context: Optional[Dict[str, Any]]) -> List[FFDetectedTopic]:
        """Detect topics using semantic search with FF search manager"""
        detected_topics = []
        
        try:
            # Use FF search manager for semantic analysis
            search_query = content[:500]  # Limit query length
            
            # Search for similar content in topic examples
            search_results = await self.search_manager.search_messages(
                user_id="system",  # Use system user for topic examples
                query=search_query,
                limit=10,
                include_metadata=True
            )
            
            # Analyze search results for topic patterns
            topic_scores = {}
            
            for result in search_results:
                result_metadata = result.get("metadata", {})
                result_topics = result_metadata.get("topics", [])
                similarity_score = result.get("similarity_score", 0.0)
                
                for topic in result_topics:
                    topic_id = topic.get("topic_id")
                    if topic_id:
                        if topic_id not in topic_scores:
                            topic_scores[topic_id] = []
                        topic_scores[topic_id].append(similarity_score)
            
            # Calculate topic confidences
            for topic_id, scores in topic_scores.items():
                if topic_id in self.topic_definitions:
                    avg_score = sum(scores) / len(scores)
                    confidence = min(0.85, avg_score * 0.9)  # Semantic detection is generally reliable
                    
                    detected_topics.append(FFDetectedTopic(
                        topic_id=topic_id,
                        topic_name=self.topic_definitions[topic_id]["name"],
                        confidence=confidence,
                        detection_method=FFTopicDetectionMethod.SEMANTIC,
                        keywords=[],
                        context={"semantic_scores": scores, "average_score": avg_score},
                        routing_target=self.topic_definitions[topic_id].get("default_route")
                    ))
            
            return detected_topics
            
        except Exception as e:
            self.logger.error(f"Semantic topic detection failed: {e}")
            return []
    
    async def _detect_topics_by_ml_classification(self, content: str, context: Optional[Dict[str, Any]]) -> List[FFDetectedTopic]:
        """Detect topics using ML classification (simplified implementation)"""
        # This would typically use a trained ML model
        # For now, we'll use a simplified heuristic approach
        detected_topics = []
        
        try:
            # Simple classification based on content characteristics
            content_lower = content.lower()
            word_count = len(content.split())
            
            # Technical support classification
            if any(word in content_lower for word in ["error", "bug", "issue", "problem", "help", "support"]):
                confidence = 0.75 if word_count > 10 else 0.60
                detected_topics.append(FFDetectedTopic(
                    topic_id="technical_support",
                    topic_name="Technical Support",
                    confidence=confidence,
                    detection_method=FFTopicDetectionMethod.ML_CLASSIFICATION,
                    keywords=[],
                    context={"classification_features": ["error_keywords"]},
                    routing_target="support_agent"
                ))
            
            # General inquiry classification
            if any(word in content_lower for word in ["what", "how", "when", "where", "why", "?"]):
                confidence = 0.65 if word_count > 5 else 0.50
                detected_topics.append(FFDetectedTopic(
                    topic_id="general_inquiry",
                    topic_name="General Inquiry",
                    confidence=confidence,
                    detection_method=FFTopicDetectionMethod.ML_CLASSIFICATION,
                    keywords=[],
                    context={"classification_features": ["question_keywords"]},
                    routing_target="general_agent"
                ))
            
            return detected_topics
            
        except Exception as e:
            self.logger.error(f"ML classification failed: {e}")
            return []
    
    async def _detect_topics_by_hybrid_approach(self, content: str, context: Optional[Dict[str, Any]]) -> List[FFDetectedTopic]:
        """Detect topics using a hybrid approach combining multiple methods"""
        try:
            # Apply multiple methods
            keyword_topics = await self._detect_topics_by_keywords(content)
            semantic_topics = await self._detect_topics_by_semantic_search(content, context)
            ml_topics = await self._detect_topics_by_ml_classification(content, context)
            
            # Combine and weight results
            combined_topics = {}
            
            # Weight different methods
            method_weights = {
                FFTopicDetectionMethod.KEYWORD: 0.3,
                FFTopicDetectionMethod.SEMANTIC: 0.5,
                FFTopicDetectionMethod.ML_CLASSIFICATION: 0.2
            }
            
            for topics, method in [(keyword_topics, FFTopicDetectionMethod.KEYWORD),
                                 (semantic_topics, FFTopicDetectionMethod.SEMANTIC),
                                 (ml_topics, FFTopicDetectionMethod.ML_CLASSIFICATION)]:
                for topic in topics:
                    topic_id = topic.topic_id
                    weighted_confidence = topic.confidence * method_weights[method]
                    
                    if topic_id not in combined_topics:
                        combined_topics[topic_id] = {
                            "topic": topic,
                            "total_confidence": 0.0,
                            "methods": []
                        }
                    
                    combined_topics[topic_id]["total_confidence"] += weighted_confidence
                    combined_topics[topic_id]["methods"].append(method)
            
            # Create final hybrid topics
            hybrid_topics = []
            for topic_data in combined_topics.values():
                topic = topic_data["topic"]
                topic.confidence = min(0.95, topic_data["total_confidence"])
                topic.detection_method = FFTopicDetectionMethod.HYBRID
                topic.context["hybrid_methods"] = [m.value for m in topic_data["methods"]]
                hybrid_topics.append(topic)
            
            return hybrid_topics
            
        except Exception as e:
            self.logger.error(f"Hybrid topic detection failed: {e}")
            return []
    
    def _merge_similar_topics(self, topics: List[FFDetectedTopic]) -> List[FFDetectedTopic]:
        """Merge similar topics to avoid duplicates"""
        if not topics:
            return topics
        
        merged_topics = {}
        
        for topic in topics:
            if topic.topic_id in merged_topics:
                # Merge with existing topic
                existing = merged_topics[topic.topic_id]
                # Use higher confidence
                if topic.confidence > existing.confidence:
                    existing.confidence = topic.confidence
                    existing.detection_method = topic.detection_method
                # Merge keywords
                existing.keywords = list(set(existing.keywords + topic.keywords))
                # Merge context
                existing.context.update(topic.context)
            else:
                merged_topics[topic.topic_id] = topic
        
        return list(merged_topics.values())
    
    def _determine_primary_topic(self, topics: List[FFDetectedTopic]) -> Optional[FFDetectedTopic]:
        """Determine the primary topic from detected topics"""
        if not topics:
            return None
        
        # Sort by confidence and return the highest
        topics.sort(key=lambda t: t.confidence, reverse=True)
        return topics[0]
    
    async def _make_routing_decision(self, 
                                   primary_topic: Optional[FFDetectedTopic],
                                   all_topics: List[FFDetectedTopic],
                                   context: Optional[Dict[str, Any]]) -> Optional[FFRoutingDecision]:
        """Make routing decision based on detected topics"""
        try:
            if not primary_topic:
                return None
            
            # Apply routing strategy
            if self.config.routing_strategy == FFRoutingStrategy.HIGHEST_CONFIDENCE:
                return self._route_by_highest_confidence(primary_topic, all_topics)
            
            elif self.config.routing_strategy == FFRoutingStrategy.WEIGHTED_AVERAGE:
                return self._route_by_weighted_average(all_topics)
            
            elif self.config.routing_strategy == FFRoutingStrategy.MULTI_TOPIC:
                return self._route_by_multi_topic(all_topics)
            
            elif self.config.routing_strategy == FFRoutingStrategy.CONTEXTUAL:
                return await self._route_by_context(primary_topic, all_topics, context)
            
            else:
                # Default to highest confidence
                return self._route_by_highest_confidence(primary_topic, all_topics)
                
        except Exception as e:
            self.logger.error(f"Routing decision failed: {e}")
            return None
    
    def _route_by_highest_confidence(self, primary_topic: FFDetectedTopic, all_topics: List[FFDetectedTopic]) -> FFRoutingDecision:
        """Route based on the highest confidence topic"""
        route_to = primary_topic.routing_target or "default_agent"
        
        alternative_routes = []
        for topic in all_topics[1:3]:  # Take top 2 alternatives
            if topic.routing_target and topic.routing_target != route_to:
                alternative_routes.append(topic.routing_target)
        
        return FFRoutingDecision(
            route_to=route_to,
            confidence=primary_topic.confidence,
            reasoning=f"Routed to {route_to} based on highest confidence topic: {primary_topic.topic_name}",
            alternative_routes=alternative_routes,
            metadata={"primary_topic": primary_topic.topic_id}
        )
    
    def _route_by_weighted_average(self, topics: List[FFDetectedTopic]) -> FFRoutingDecision:
        """Route based on weighted average of all topics"""
        if not topics:
            return FFRoutingDecision(
                route_to="default_agent",
                confidence=0.0,
                reasoning="No topics detected, routing to default"
            )
        
        # Calculate weighted scores for each routing target
        route_scores = {}
        total_confidence = sum(t.confidence for t in topics)
        
        for topic in topics:
            route = topic.routing_target or "default_agent"
            weight = topic.confidence / total_confidence if total_confidence > 0 else 0
            
            if route not in route_scores:
                route_scores[route] = 0.0
            route_scores[route] += weight
        
        # Choose route with highest weighted score
        best_route = max(route_scores.items(), key=lambda x: x[1])
        route_to = best_route[0]
        confidence = best_route[1]
        
        return FFRoutingDecision(
            route_to=route_to,
            confidence=confidence,
            reasoning=f"Routed to {route_to} based on weighted average of {len(topics)} topics",
            alternative_routes=[r for r, _ in sorted(route_scores.items(), key=lambda x: x[1], reverse=True)[1:3]],
            metadata={"route_scores": route_scores}
        )
    
    def _route_by_multi_topic(self, topics: List[FFDetectedTopic]) -> FFRoutingDecision:
        """Route considering multiple topics simultaneously"""
        if not topics:
            return FFRoutingDecision(
                route_to="default_agent",
                confidence=0.0,
                reasoning="No topics detected, routing to default"
            )
        
        # For multi-topic routing, we might need a specialized multi-topic agent
        # or route to the most general agent that can handle multiple topics
        
        high_confidence_topics = [t for t in topics if t.confidence > 0.7]
        
        if len(high_confidence_topics) > 1:
            # Multiple high-confidence topics - route to multi-topic handler
            return FFRoutingDecision(
                route_to="multi_topic_agent",
                confidence=min(t.confidence for t in high_confidence_topics),
                reasoning=f"Multiple high-confidence topics detected: {[t.topic_name for t in high_confidence_topics]}",
                alternative_routes=[t.routing_target for t in high_confidence_topics[:2] if t.routing_target],
                metadata={"multi_topics": [t.topic_id for t in high_confidence_topics]}
            )
        else:
            # Single high-confidence topic or low confidence - route normally
            return self._route_by_highest_confidence(topics[0], topics)
    
    async def _route_by_context(self, 
                              primary_topic: FFDetectedTopic,
                              all_topics: List[FFDetectedTopic],
                              context: Optional[Dict[str, Any]]) -> FFRoutingDecision:
        """Route based on contextual information"""
        # Consider context factors like user history, session state, etc.
        route_to = primary_topic.routing_target or "default_agent"
        confidence = primary_topic.confidence
        reasoning = f"Routed to {route_to} based on topic {primary_topic.topic_name}"
        
        # Adjust routing based on context
        if context:
            user_preference = context.get("preferred_agent")
            if user_preference:
                route_to = user_preference
                reasoning += f" (adjusted for user preference: {user_preference})"
            
            session_context = context.get("session_context", {})
            if session_context.get("escalation_level") == "high":
                route_to = "senior_agent"
                reasoning += " (escalated to senior agent)"
        
        return FFRoutingDecision(
            route_to=route_to,
            confidence=confidence,
            reasoning=reasoning,
            alternative_routes=[primary_topic.routing_target] if primary_topic.routing_target != route_to else [],
            metadata={"context_factors": list(context.keys()) if context else []}
        )
    
    def _initialize_built_in_topics(self) -> None:
        """Initialize built-in topic definitions"""
        self.topic_definitions = {
            "technical_support": {
                "name": "Technical Support",
                "keywords": ["error", "bug", "issue", "problem", "crash", "broken", "not working", "help", "support"],
                "patterns": [r"error\s+\d+", r"exception:\s+\w+", r"failed\s+to\s+\w+"],
                "default_route": "support_agent",
                "priority": 8
            },
            "general_inquiry": {
                "name": "General Inquiry",
                "keywords": ["what", "how", "when", "where", "why", "question", "inquiry", "ask"],
                "patterns": [r"\?\s*$", r"^(how|what|when|where|why)\s+"],
                "default_route": "general_agent",
                "priority": 3
            },
            "account_management": {
                "name": "Account Management",
                "keywords": ["account", "profile", "settings", "password", "login", "signup", "billing", "subscription"],
                "patterns": [r"change\s+password", r"update\s+profile", r"billing\s+info"],
                "default_route": "account_agent",
                "priority": 7
            },
            "product_information": {
                "name": "Product Information",
                "keywords": ["product", "feature", "specification", "details", "information", "about"],
                "patterns": [r"tell\s+me\s+about", r"information\s+on", r"details\s+of"],
                "default_route": "product_agent",
                "priority": 5
            },
            "sales_inquiry": {
                "name": "Sales Inquiry",
                "keywords": ["price", "cost", "buy", "purchase", "order", "sales", "quote", "pricing"],
                "patterns": [r"how\s+much", r"price\s+for", r"cost\s+of"],
                "default_route": "sales_agent",
                "priority": 6
            }
        }
    
    def _initialize_routing_rules(self) -> None:
        """Initialize routing rules and mappings"""
        self.routing_rules = {
            "escalation_rules": {
                "high_priority_keywords": ["urgent", "critical", "emergency", "asap"],
                "escalation_route": "senior_agent"
            },
            "time_based_rules": {
                "business_hours": {
                    "start": 9,
                    "end": 17,
                    "route": "human_agent"
                },
                "after_hours": {
                    "route": "ai_agent"
                }
            },
            "fallback_rules": {
                "low_confidence_threshold": 0.3,
                "fallback_route": "general_agent"
            }
        }
    
    def _generate_message_hash(self, content: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate hash for message caching"""
        hash_input = content
        if context:
            hash_input += json.dumps(context, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cached_analysis(self, message_hash: str) -> Optional[FFTopicAnalysis]:
        """Get cached topic analysis if available and valid"""
        if message_hash not in self.topic_cache:
            return None
        
        cached_analysis, cached_time = self.topic_cache[message_hash]
        
        # Check if cache is still valid
        if time.time() - cached_time > self.cache_ttl_seconds:
            del self.topic_cache[message_hash]
            return None
        
        return cached_analysis
    
    def _cache_analysis(self, message_hash: str, analysis: FFTopicAnalysis) -> None:
        """Cache topic analysis result"""
        self.topic_cache[message_hash] = (analysis, time.time())
        
        # Clean old cache entries if cache is getting too large
        if len(self.topic_cache) > 1000:
            # Remove oldest 25% of entries
            oldest_entries = sorted(self.topic_cache.items(), key=lambda x: x[1][1])[:250]
            for hash_key, _ in oldest_entries:
                del self.topic_cache[hash_key]
    
    def _update_metrics(self, analysis: FFTopicAnalysis) -> None:
        """Update performance metrics"""
        self.analysis_metrics["total_analyses"] += 1
        
        # Update average analysis time
        total_time = (self.analysis_metrics["average_analysis_time_ms"] * 
                     (self.analysis_metrics["total_analyses"] - 1) + 
                     analysis.analysis_time_ms)
        self.analysis_metrics["average_analysis_time_ms"] = total_time / self.analysis_metrics["total_analyses"]
        
        # Update routing success/failure
        if analysis.routing_decision:
            self.analysis_metrics["successful_routes"] += 1
        else:
            self.analysis_metrics["failed_routes"] += 1
        
        # Update confidence distribution
        if analysis.primary_topic:
            confidence = analysis.primary_topic.confidence
            if confidence >= 0.8:
                confidence_level = FFTopicConfidence.VERY_HIGH
            elif confidence >= 0.6:
                confidence_level = FFTopicConfidence.HIGH
            elif confidence >= 0.4:
                confidence_level = FFTopicConfidence.MEDIUM
            elif confidence >= 0.2:
                confidence_level = FFTopicConfidence.LOW
            else:
                confidence_level = FFTopicConfidence.VERY_LOW
            
            self.analysis_metrics["topic_confidence_distribution"][confidence_level.value] += 1
    
    async def _store_analysis_for_learning(self, session_id: str, user_id: str, analysis: FFTopicAnalysis) -> None:
        """Store analysis data for learning system"""
        if not self.config.enable_learning:
            return
        
        try:
            learning_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "message_id": analysis.message_id,
                "detected_topics": [
                    {
                        "topic_id": t.topic_id,
                        "confidence": t.confidence,
                        "detection_method": t.detection_method.value
                    } for t in analysis.detected_topics
                ],
                "primary_topic": {
                    "topic_id": analysis.primary_topic.topic_id,
                    "confidence": analysis.primary_topic.confidence
                } if analysis.primary_topic else None,
                "routing_decision": {
                    "route_to": analysis.routing_decision.route_to,
                    "confidence": analysis.routing_decision.confidence
                } if analysis.routing_decision else None
            }
            
            self.learning_data.append(learning_entry)
            
            # Keep learning data size manageable
            if len(self.learning_data) > 10000:
                self.learning_data = self.learning_data[-5000:]
                
        except Exception as e:
            self.logger.error(f"Failed to store learning data: {e}")
    
    def _format_analysis_result(self, analysis: FFTopicAnalysis) -> Dict[str, Any]:
        """Format analysis result for return"""
        return {
            "success": True,
            "message_id": analysis.message_id,
            "detected_topics": [
                {
                    "topic_id": t.topic_id,
                    "topic_name": t.topic_name,
                    "confidence": t.confidence,
                    "detection_method": t.detection_method.value,
                    "keywords": t.keywords,
                    "routing_target": t.routing_target
                } for t in analysis.detected_topics
            ],
            "primary_topic": {
                "topic_id": analysis.primary_topic.topic_id,
                "topic_name": analysis.primary_topic.topic_name,
                "confidence": analysis.primary_topic.confidence,
                "routing_target": analysis.primary_topic.routing_target
            } if analysis.primary_topic else None,
            "routing_decision": {
                "route_to": analysis.routing_decision.route_to,
                "confidence": analysis.routing_decision.confidence,
                "reasoning": analysis.routing_decision.reasoning,
                "alternative_routes": analysis.routing_decision.alternative_routes
            } if analysis.routing_decision else None,
            "analysis_time_ms": analysis.analysis_time_ms,
            "context": analysis.context
        }
    
    async def add_topic_definition(self, topic_id: str, topic_definition: Dict[str, Any]) -> bool:
        """Add a new topic definition"""
        try:
            self.topic_definitions[topic_id] = topic_definition
            self.logger.info(f"Added topic definition: {topic_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add topic definition: {e}")
            return False
    
    def remove_topic_definition(self, topic_id: str) -> bool:
        """Remove a topic definition"""
        try:
            if topic_id in self.topic_definitions:
                del self.topic_definitions[topic_id]
                self.logger.info(f"Removed topic definition: {topic_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove topic definition: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            "analysis_metrics": self.analysis_metrics.copy(),
            "cache_stats": {
                "cache_size": len(self.topic_cache),
                "cache_ttl_seconds": self.cache_ttl_seconds
            },
            "topic_definitions_count": len(self.topic_definitions),
            "learning_data_size": len(self.learning_data) if self.config.enable_learning else 0
        }
    
    async def provide_feedback(self, message_id: str, actual_route: str, feedback_type: str) -> bool:
        """Provide feedback for learning system"""
        try:
            if not self.config.enable_learning:
                return False
            
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id,
                "actual_route": actual_route,
                "feedback_type": feedback_type  # "correct", "incorrect", "partial"
            }
            
            self.feedback_history.append(feedback_entry)
            
            # Keep feedback history manageable
            if len(self.feedback_history) > 5000:
                self.feedback_history = self.feedback_history[-2500:]
            
            self.logger.info(f"Recorded feedback for message {message_id}: {feedback_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record feedback: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the topic analysis cache"""
        self.topic_cache.clear()
        self.logger.info("Topic analysis cache cleared")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "component_name": "FF Topic Router Component",
            "version": "1.0.0",
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "routing_strategy": self.config.routing_strategy.value,
                "detection_methods": [m.value for m in self.config.detection.detection_methods],
                "enable_learning": self.config.enable_learning,
                "cache_ttl_seconds": self.config.cache_ttl_seconds
            },
            "status": "active",
            "metrics": self.get_metrics()
        }