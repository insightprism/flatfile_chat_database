"""
FF Memory Component - Phase 3 Enhanced Implementation

Advanced memory management with RAG capabilities, knowledge graphs,
and contextual retrieval using existing FF infrastructure as backend.
Supports 9/22 use cases (41% coverage) with Phase 3 enhancements.
"""

import asyncio
import time
import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid

from ff_utils.ff_logging import get_logger
from ff_managers.ff_storage_manager import FFStorageManager
from ff_managers.ff_vector_storage_manager import FFVectorStorageManager
from ff_managers.ff_search_manager import FFSearchManager
from ff_managers.ff_document_manager import FFDocumentManager
from ff_protocols.ff_message_dto import FFMessageDTO
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_class_configs.ff_memory_config import FFMemoryConfigDTO

logger = get_logger(__name__)


class FFRAGStrategy(Enum):
    """Advanced RAG retrieval strategies"""
    SIMILARITY_ONLY = "similarity_only"
    HYBRID_SEARCH = "hybrid_search"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"
    TEMPORAL_WEIGHTED = "temporal_weighted"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    KNOWLEDGE_GRAPH_TRAVERSAL = "knowledge_graph_traversal"
    CLUSTER_BASED_RETRIEVAL = "cluster_based_retrieval"


class FFMemoryType(Enum):
    """Enhanced memory types for Phase 3"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CONTEXTUAL = "contextual"
    AUTOBIOGRAPHICAL = "autobiographical"
    DECLARATIVE = "declarative"
    WORKING = "working"
    LONG_TERM = "long_term"


class FFMemoryQuality(Enum):
    """Memory quality levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FFEnhancedMemoryEntry:
    """Enhanced memory entry with advanced RAG capabilities"""
    memory_id: str
    user_id: str
    session_id: str
    memory_type: FFMemoryType
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance_score: float
    
    # Phase 3 RAG enhancements
    context_embedding: Optional[List[float]] = None
    related_memories: List[str] = field(default_factory=list)
    knowledge_graph_nodes: List[str] = field(default_factory=list)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_contexts: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    semantic_concepts: List[str] = field(default_factory=list)
    memory_clusters: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    update_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FFRAGResult:
    """Advanced RAG retrieval result"""
    memories: List[FFEnhancedMemoryEntry]
    context_summary: str
    retrieval_strategy: FFRAGStrategy
    confidence_score: float
    processing_time_ms: float
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph_paths: List[List[str]] = field(default_factory=list)
    cluster_contributions: Dict[str, float] = field(default_factory=dict)
    temporal_distribution: Dict[str, int] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFKnowledgeGraphNode:
    """Knowledge graph node for memory connections"""
    node_id: str
    content: str
    node_type: str
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    update_count: int = 0


@dataclass
class FFMemoryCluster:
    """Memory cluster for related memories"""
    cluster_id: str
    memory_ids: List[str]
    centroid_embedding: List[float]
    cluster_theme: str
    confidence: float
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryRanker:
    """Advanced memory ranking with multiple criteria"""
    
    def __init__(self, config: FFMemoryConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def rank_memories(self, 
                          memories: List[FFEnhancedMemoryEntry], 
                          query: str, 
                          context: Dict[str, Any],
                          strategy: FFRAGStrategy = FFRAGStrategy.HYBRID_SEARCH) -> List[FFEnhancedMemoryEntry]:
        """Rank memories using advanced scoring"""
        try:
            scored_memories = []
            
            for memory in memories:
                score = await self._calculate_comprehensive_score(memory, query, context, strategy)
                memory.metadata['retrieval_score'] = score
                scored_memories.append(memory)
            
            # Sort by score
            scored_memories.sort(key=lambda m: m.metadata.get('retrieval_score', 0.0), reverse=True)
            
            return scored_memories
            
        except Exception as e:
            self.logger.error(f"Memory ranking failed: {e}")
            return memories
    
    async def _calculate_comprehensive_score(self, 
                                           memory: FFEnhancedMemoryEntry, 
                                           query: str, 
                                           context: Dict[str, Any],
                                           strategy: FFRAGStrategy) -> float:
        """Calculate comprehensive relevance score"""
        scores = {}
        
        # Semantic similarity score
        scores['semantic'] = memory.metadata.get('similarity_score', 0.0)
        
        # Importance score
        scores['importance'] = memory.importance_score
        
        # Recency score (decay over time)
        time_diff = (datetime.now() - memory.created_at).total_seconds()
        scores['recency'] = np.exp(-time_diff / (30 * 24 * 3600))  # 30-day half-life
        
        # Access frequency score
        scores['frequency'] = min(memory.access_count / 100.0, 1.0)  # Normalize to [0,1]
        
        # Context relevance score
        scores['context'] = await self._calculate_context_relevance(memory, context)
        
        # Quality score
        scores['quality'] = np.mean(list(memory.quality_metrics.values())) if memory.quality_metrics else 0.5
        
        # Strategy-specific weighting
        weights = self._get_strategy_weights(strategy)
        
        # Calculate weighted sum
        final_score = sum(scores[key] * weights.get(key, 0.0) for key in scores)
        
        return min(final_score, 1.0)
    
    async def _calculate_context_relevance(self, memory: FFEnhancedMemoryEntry, context: Dict[str, Any]) -> float:
        """Calculate relevance to current context"""
        relevance = 0.0
        
        # Check session continuity
        if context.get('session_id') == memory.session_id:
            relevance += 0.3
        
        # Check temporal context
        if 'time_window' in context:
            time_window = context['time_window']
            if time_window[0] <= memory.created_at <= time_window[1]:
                relevance += 0.2
        
        # Check topic continuity
        current_topics = context.get('topics', [])
        memory_concepts = memory.semantic_concepts
        topic_overlap = len(set(current_topics) & set(memory_concepts))
        if current_topics:
            relevance += (topic_overlap / len(current_topics)) * 0.3
        
        # Check user context
        if context.get('user_id') == memory.user_id:
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _get_strategy_weights(self, strategy: FFRAGStrategy) -> Dict[str, float]:
        """Get weighting scheme for different strategies"""
        weight_schemes = {
            FFRAGStrategy.SIMILARITY_ONLY: {
                'semantic': 1.0, 'importance': 0.0, 'recency': 0.0, 
                'frequency': 0.0, 'context': 0.0, 'quality': 0.0
            },
            FFRAGStrategy.HYBRID_SEARCH: {
                'semantic': 0.4, 'importance': 0.2, 'recency': 0.15, 
                'frequency': 0.1, 'context': 0.1, 'quality': 0.05
            },
            FFRAGStrategy.CONTEXTUAL_RETRIEVAL: {
                'semantic': 0.2, 'importance': 0.1, 'recency': 0.1, 
                'frequency': 0.1, 'context': 0.4, 'quality': 0.1
            },
            FFRAGStrategy.TEMPORAL_WEIGHTED: {
                'semantic': 0.3, 'importance': 0.1, 'recency': 0.4, 
                'frequency': 0.1, 'context': 0.05, 'quality': 0.05
            },
            FFRAGStrategy.IMPORTANCE_WEIGHTED: {
                'semantic': 0.2, 'importance': 0.5, 'recency': 0.1, 
                'frequency': 0.1, 'context': 0.05, 'quality': 0.05
            }
        }
        
        return weight_schemes.get(strategy, weight_schemes[FFRAGStrategy.HYBRID_SEARCH])


class ContextAugmenter:
    """Context augmentation for better retrieval"""
    
    def __init__(self, config: FFMemoryConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def augment_query_context(self, 
                                  query: str, 
                                  session_context: Dict[str, Any],
                                  user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Augment query with contextual information"""
        try:
            augmented_context = {
                'original_query': query,
                'expanded_query': await self._expand_query(query, session_context),
                'contextual_keywords': await self._extract_contextual_keywords(session_context),
                'temporal_context': self._build_temporal_context(user_history),
                'topic_context': await self._extract_topic_context(query, session_context),
                'user_intent': await self._infer_user_intent(query, session_context)
            }
            
            return augmented_context
            
        except Exception as e:
            self.logger.error(f"Context augmentation failed: {e}")
            return {'original_query': query}
    
    async def _expand_query(self, query: str, context: Dict[str, Any]) -> str:
        """Expand query with contextual terms"""
        expanded_terms = [query]
        
        # Add related terms from context
        if 'previous_queries' in context:
            related_terms = self._extract_related_terms(context['previous_queries'])
            expanded_terms.extend(related_terms[:3])  # Limit expansion
        
        # Add domain-specific terms
        if 'domain' in context:
            domain_terms = self._get_domain_terms(context['domain'])
            expanded_terms.extend(domain_terms[:2])
        
        return ' '.join(expanded_terms)
    
    async def _extract_contextual_keywords(self, context: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords from session context"""
        keywords = []
        
        # Extract from recent messages
        if 'recent_messages' in context:
            for message in context['recent_messages']:
                message_keywords = self._extract_keywords_from_text(message.get('content', ''))
                keywords.extend(message_keywords)
        
        # Remove duplicates and rank by frequency
        keyword_counts = defaultdict(int)
        for keyword in keywords:
            keyword_counts[keyword] += 1
        
        # Return top keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:10]]
    
    def _build_temporal_context(self, user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build temporal context from user history"""
        if not user_history:
            return {}
        
        # Analyze temporal patterns
        recent_activity = [event for event in user_history if 
                          (datetime.now() - datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))).days <= 7]
        
        return {
            'recent_activity_count': len(recent_activity),
            'activity_timespan_days': (datetime.now() - datetime.fromisoformat(user_history[0].get('timestamp', datetime.now().isoformat()))).days,
            'peak_activity_hours': self._find_peak_activity_hours(user_history),
            'session_patterns': self._analyze_session_patterns(user_history)
        }
    
    async def _extract_topic_context(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Extract topical context from query and session"""
        topics = []
        
        # Basic topic extraction (would use NLP in production)
        query_topics = self._extract_topics_from_text(query)
        topics.extend(query_topics)
        
        # Context topics
        if 'current_topics' in context:
            topics.extend(context['current_topics'])
        
        return list(set(topics))
    
    async def _infer_user_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Infer user intent from query and context"""
        # Simplified intent classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['remember', 'recall', 'what did', 'when did']):
            return 'memory_retrieval'
        elif any(word in query_lower for word in ['how to', 'explain', 'what is']):
            return 'information_seeking'
        elif any(word in query_lower for word in ['help', 'assist', 'support']):
            return 'assistance_request'
        else:
            return 'general_query'
    
    def _extract_related_terms(self, previous_queries: List[str]) -> List[str]:
        """Extract related terms from previous queries"""
        # Simplified implementation - would use word embeddings in production
        all_terms = []
        for query in previous_queries[-5:]:  # Last 5 queries
            terms = [word.lower() for word in query.split() if len(word) > 3]
            all_terms.extend(terms)
        
        # Return most frequent terms
        term_counts = defaultdict(int)
        for term in all_terms:
            term_counts[term] += 1
        
        return [term for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _get_domain_terms(self, domain: str) -> List[str]:
        """Get domain-specific terms"""
        domain_terms = {
            'technology': ['software', 'hardware', 'programming', 'development', 'system'],
            'science': ['research', 'experiment', 'hypothesis', 'data', 'analysis'],
            'business': ['strategy', 'market', 'revenue', 'customer', 'product'],
            'health': ['treatment', 'symptoms', 'diagnosis', 'patient', 'medical']
        }
        
        return domain_terms.get(domain.lower(), [])
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text (simplified)"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
        words = [word.lower().strip('.,!?;:"()[]{}') for word in text.split()]
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:10]  # Limit to top 10
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        # Simplified topic extraction - would use proper NLP in production
        topic_indicators = {
            'technology': ['computer', 'software', 'programming', 'code', 'system', 'app', 'website'],
            'science': ['research', 'study', 'experiment', 'theory', 'data', 'analysis'],
            'business': ['company', 'market', 'sales', 'revenue', 'customer', 'product'],
            'health': ['health', 'medical', 'doctor', 'treatment', 'medicine', 'patient']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, indicators in topic_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _find_peak_activity_hours(self, history: List[Dict[str, Any]]) -> List[int]:
        """Find peak activity hours from history"""
        hour_counts = defaultdict(int)
        
        for event in history:
            try:
                timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
                hour_counts[timestamp.hour] += 1
            except:
                continue
        
        # Return top 3 hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]
    
    def _analyze_session_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze session patterns"""
        if not history:
            return {}
        
        session_lengths = []
        session_gaps = []
        
        current_session_start = None
        last_event_time = None
        
        for event in history:
            try:
                event_time = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
                
                if current_session_start is None:
                    current_session_start = event_time
                elif last_event_time and (event_time - last_event_time).seconds > 1800:  # 30 min gap = new session
                    session_lengths.append((last_event_time - current_session_start).seconds)
                    session_gaps.append((event_time - last_event_time).seconds)
                    current_session_start = event_time
                
                last_event_time = event_time
            except:
                continue
        
        return {
            'average_session_length_minutes': np.mean(session_lengths) / 60 if session_lengths else 0,
            'average_session_gap_hours': np.mean(session_gaps) / 3600 if session_gaps else 0,
            'total_sessions': len(session_lengths) + 1
        }


class KnowledgeExtractor:
    """Extract and structure knowledge from memories"""
    
    def __init__(self, config: FFMemoryConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def extract_knowledge_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract knowledge entities from content"""
        try:
            entities = []
            
            # Named entity extraction (simplified)
            entities.extend(self._extract_named_entities(content))
            
            # Concept extraction
            entities.extend(self._extract_concepts(content))
            
            # Relationship extraction
            relationships = self._extract_relationships(content)
            
            return {
                'entities': entities,
                'relationships': relationships,
                'concepts': self._extract_key_concepts(content)
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'concepts': []}
    
    def _extract_named_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities (simplified implementation)"""
        # This would use proper NER in production (spaCy, NLTK, etc.)
        entities = []
        
        # Simple pattern-based extraction
        import re
        
        # Extract potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        persons = re.findall(person_pattern, content)
        for person in persons:
            entities.append({'text': person, 'type': 'PERSON', 'confidence': 0.7})
        
        # Extract potential organizations
        org_indicators = ['Inc', 'LLC', 'Corp', 'Company', 'Organization']
        for indicator in org_indicators:
            pattern = rf'\b\w+\s+{indicator}\b'
            orgs = re.findall(pattern, content)
            for org in orgs:
                entities.append({'text': org, 'type': 'ORG', 'confidence': 0.8})
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(date_pattern, content)
        for date in dates:
            entities.append({'text': date, 'type': 'DATE', 'confidence': 0.9})
        
        return entities
    
    def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract conceptual terms"""
        concepts = []
        
        # Domain-specific concept patterns
        concept_patterns = {
            'TECHNOLOGY': [r'\b\w*software\w*\b', r'\b\w*algorithm\w*\b', r'\b\w*database\w*\b'],
            'SCIENCE': [r'\b\w*research\w*\b', r'\b\w*hypothesis\w*\b', r'\b\w*experiment\w*\b'],
            'BUSINESS': [r'\b\w*strategy\w*\b', r'\b\w*market\w*\b', r'\b\w*revenue\w*\b']
        }
        
        import re
        for concept_type, patterns in concept_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    concepts.append({'text': match, 'type': concept_type, 'confidence': 0.6})
        
        return concepts
    
    def _extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship patterns
        import re
        
        # "X is Y" relationships
        is_pattern = r'(\w+(?:\s+\w+)*)\s+is\s+(\w+(?:\s+\w+)*)'
        is_matches = re.findall(is_pattern, content, re.IGNORECASE)
        for subject, obj in is_matches:
            relationships.append({
                'subject': subject.strip(),
                'predicate': 'is',
                'object': obj.strip(),
                'confidence': 0.7
            })
        
        # "X has Y" relationships
        has_pattern = r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)'
        has_matches = re.findall(has_pattern, content, re.IGNORECASE)
        for subject, obj in has_matches:
            relationships.append({
                'subject': subject.strip(),
                'predicate': 'has',
                'object': obj.strip(),
                'confidence': 0.6
            })
        
        return relationships
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key conceptual terms"""
        # Simple TF-IDF-like approach
        words = content.lower().split()
        word_freq = defaultdict(int)
        
        # Count word frequencies
        for word in words:
            if len(word) > 4:  # Focus on longer, more meaningful words
                word_freq[word] += 1
        
        # Return most frequent concepts
        sorted_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in sorted_concepts[:10] if freq > 1]


class MemoryQualityAssessor:
    """Assess and improve memory quality"""
    
    def __init__(self, config: FFMemoryConfigDTO):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def assess_memory_quality(self, memory: FFEnhancedMemoryEntry) -> Dict[str, float]:
        """Assess quality of a memory entry"""
        try:
            quality_metrics = {}
            
            # Content quality
            quality_metrics['content_quality'] = self._assess_content_quality(memory.content)
            
            # Contextual completeness
            quality_metrics['contextual_completeness'] = self._assess_contextual_completeness(memory)
            
            # Temporal relevance
            quality_metrics['temporal_relevance'] = self._assess_temporal_relevance(memory)
            
            # Retrieval utility
            quality_metrics['retrieval_utility'] = self._assess_retrieval_utility(memory)
            
            # Connection richness
            quality_metrics['connection_richness'] = self._assess_connection_richness(memory)
            
            # Overall quality score
            quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {'overall_quality': 0.5}
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess quality of memory content"""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length score (not too short, not too long)
        length = len(content)
        if 50 <= length <= 500:
            score += 0.3
        elif 20 <= length < 50 or 500 < length <= 1000:
            score += 0.2
        elif length > 1000:
            score += 0.1
        
        # Information density (presence of meaningful content)
        words = content.split()
        unique_words = set(word.lower() for word in words)
        if words:
            uniqueness_ratio = len(unique_words) / len(words)
            score += uniqueness_ratio * 0.3
        
        # Structural indicators (punctuation, capitalization)
        has_punctuation = any(char in content for char in '.,!?;:')
        has_proper_capitalization = content[0].isupper() if content else False
        
        if has_punctuation:
            score += 0.2
        if has_proper_capitalization:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_contextual_completeness(self, memory: FFEnhancedMemoryEntry) -> float:
        """Assess completeness of contextual information"""
        score = 0.0
        
        # Basic metadata presence
        required_fields = ['session_id', 'user_id', 'created_at']
        present_fields = sum(1 for field in required_fields if getattr(memory, field, None))
        score += (present_fields / len(required_fields)) * 0.3
        
        # Metadata richness
        metadata_score = len(memory.metadata) / 10  # Normalize by expected max
        score += min(metadata_score, 0.3)
        
        # Temporal context
        if memory.temporal_context:
            score += 0.2
        
        # Related memories
        if memory.related_memories:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_temporal_relevance(self, memory: FFEnhancedMemoryEntry) -> float:
        """Assess temporal relevance of memory"""
        if not memory.created_at:
            return 0.5
        
        # Decay based on age
        age_days = (datetime.now() - memory.created_at).days
        
        # Different decay rates for different memory types
        if memory.memory_type in [FFMemoryType.SEMANTIC, FFMemoryType.DECLARATIVE]:
            # Slower decay for factual information
            decay_rate = 0.01
        else:
            # Faster decay for episodic memories
            decay_rate = 0.02
        
        temporal_score = np.exp(-decay_rate * age_days)
        
        # Boost for recently accessed memories
        if memory.last_accessed:
            days_since_access = (datetime.now() - memory.last_accessed).days
            access_boost = np.exp(-0.05 * days_since_access) * 0.2
            temporal_score += access_boost
        
        return min(temporal_score, 1.0)
    
    def _assess_retrieval_utility(self, memory: FFEnhancedMemoryEntry) -> float:
        """Assess how useful this memory is for retrieval"""
        score = 0.0
        
        # Access frequency
        access_score = min(memory.access_count / 10, 0.4)  # Normalize, max 0.4
        score += access_score
        
        # Importance score
        score += memory.importance_score * 0.3
        
        # Confidence score
        score += memory.confidence_score * 0.3
        
        return min(score, 1.0)
    
    def _assess_connection_richness(self, memory: FFEnhancedMemoryEntry) -> float:
        """Assess richness of connections to other memories"""
        score = 0.0
        
        # Related memories
        related_score = min(len(memory.related_memories) / 5, 0.3)  # Max 0.3
        score += related_score
        
        # Knowledge graph connections
        kg_score = min(len(memory.knowledge_graph_nodes) / 3, 0.3)  # Max 0.3
        score += kg_score
        
        # Cross-references
        ref_score = min(len(memory.cross_references) / 3, 0.2)  # Max 0.2
        score += ref_score
        
        # Cluster memberships
        cluster_score = min(len(memory.memory_clusters) / 2, 0.2)  # Max 0.2
        score += cluster_score
        
        return min(score, 1.0)


class FFEnhancedMemoryComponent(FFChatComponentProtocol):
    """
    Enhanced FF Memory Component with advanced RAG capabilities.
    
    Provides sophisticated memory management including knowledge graphs,
    contextual retrieval, and quality assessment using FF infrastructure.
    """
    
    def __init__(self, 
                 config: FFMemoryConfigDTO,
                 storage_manager: FFStorageManager,
                 vector_manager: FFVectorStorageManager,
                 search_manager: FFSearchManager,
                 document_manager: FFDocumentManager):
        """
        Initialize Enhanced FF Memory Component.
        
        Args:
            config: Memory component configuration
            storage_manager: FF storage manager
            vector_manager: FF vector storage manager  
            search_manager: FF search manager
            document_manager: FF document manager for RAG processing
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # FF backend services
        self.storage_manager = storage_manager
        self.vector_manager = vector_manager
        self.search_manager = search_manager
        self.document_manager = document_manager
        
        # Component state
        self._initialized = False
        
        # Enhanced memory management
        self._working_memory: Dict[str, Dict[str, Any]] = {}
        self._memory_cache: Dict[str, FFEnhancedMemoryEntry] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Phase 3 RAG enhancements
        self._knowledge_graph: Dict[str, FFKnowledgeGraphNode] = {}
        self._memory_clusters: Dict[str, FFMemoryCluster] = {}
        self._retrieval_history: deque = deque(maxlen=1000)
        self._context_embeddings: Dict[str, List[float]] = {}
        self._rag_strategies: Dict[str, FFRAGStrategy] = {}
        
        # Advanced RAG components
        self._memory_ranker = MemoryRanker(config)
        self._context_augmenter = ContextAugmenter(config)
        self._knowledge_extractor = KnowledgeExtractor(config)
        self._quality_assessor = MemoryQualityAssessor(config)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None
        self._quality_assessment_task: Optional[asyncio.Task] = None
        
        # Enhanced processing statistics
        self._memory_stats = {
            "total_memories_stored": 0,
            "total_memories_retrieved": 0,
            "working_memory_updates": 0,
            "consolidation_runs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_retrieval_time": 0.0,
            # Phase 3 metrics
            "rag_retrievals": 0,
            "context_augmentations": 0,
            "knowledge_extractions": 0,
            "memory_quality_assessments": 0,
            "multi_hop_queries": 0,
            "cluster_formations": 0,
            "graph_updates": 0,
            "strategy_adaptations": 0
        }
    
    async def process_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message: FFMessageDTO, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process chat message with enhanced RAG capabilities.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: Message to process
            context: Optional processing context
            
        Returns:
            Enhanced processing results with RAG information
        """
        if not self._initialized:
            return {
                "success": False,
                "error": "Component not initialized",
                "component": "ff_enhanced_memory"
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.debug(f"Processing enhanced memory for message in session {session_id}")
            
            # Update working memory
            await self._update_working_memory(session_id, message)
            
            # Extract knowledge from message
            knowledge_extracted = False
            extracted_knowledge = {}
            if await self._should_extract_knowledge(message, context):
                extracted_knowledge = await self._knowledge_extractor.extract_knowledge_entities(message.content)
                knowledge_extracted = True
                self._memory_stats["knowledge_extractions"] += 1
            
            # Store important messages with enhanced metadata
            memory_stored = False
            if await self._should_store_as_memory(message, context):
                enhanced_memory = await self._create_enhanced_memory_entry(
                    user_id=user_id,
                    session_id=session_id,
                    message=message,
                    context=context,
                    extracted_knowledge=extracted_knowledge
                )
                
                memory_stored = await self._store_enhanced_memory(enhanced_memory)
                
                # Update knowledge graph
                if memory_stored and extracted_knowledge:
                    await self._update_knowledge_graph(enhanced_memory, extracted_knowledge)
                    self._memory_stats["graph_updates"] += 1
            
            # Enhanced RAG retrieval
            rag_result = None
            if context.get("retrieve_memories", True):
                # Augment context for better retrieval
                augmented_context = await self._context_augmenter.augment_query_context(
                    query=message.content,
                    session_context=context,
                    user_history=await self._get_user_history(user_id)
                )
                
                self._memory_stats["context_augmentations"] += 1
                
                # Determine optimal RAG strategy
                rag_strategy = await self._determine_rag_strategy(user_id, message.content, context)
                
                # Perform advanced RAG retrieval
                rag_result = await self._advanced_rag_retrieval(
                    user_id=user_id,
                    query=message.content,
                    augmented_context=augmented_context,
                    strategy=rag_strategy,
                    limit=context.get("memory_limit", 5)
                )
                
                self._memory_stats["rag_retrievals"] += 1
            
            # Memory clustering and organization
            if memory_stored:
                await self._update_memory_clusters(user_id)
                self._memory_stats["cluster_formations"] += 1
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "component": "ff_enhanced_memory",
                "processor": "enhanced_rag_backend",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_stored": memory_stored,
                    "knowledge_extracted": knowledge_extracted,
                    "rag_retrieval_performed": rag_result is not None,
                    "processing_time": processing_time
                }
            }
            
            # Add enhanced RAG results
            if rag_result:
                result["rag_results"] = {
                    "memories_count": len(rag_result.memories),
                    "context_summary": rag_result.context_summary,
                    "retrieval_strategy": rag_result.retrieval_strategy.value,
                    "confidence_score": rag_result.confidence_score,
                    "processing_time_ms": rag_result.processing_time_ms,
                    "memory_context": self._build_enhanced_memory_context(rag_result.memories)
                }
                
                # Include detailed memory information if requested
                if context.get("include_detailed_memories", False):
                    result["rag_results"]["detailed_memories"] = [
                        self._format_memory_for_response(memory) for memory in rag_result.memories
                    ]
            
            # Add knowledge extraction results
            if extracted_knowledge:
                result["knowledge_extraction"] = {
                    "entities_count": len(extracted_knowledge.get('entities', [])),
                    "relationships_count": len(extracted_knowledge.get('relationships', [])),
                    "concepts_count": len(extracted_knowledge.get('concepts', []))
                }
                
                if context.get("include_extracted_knowledge", False):
                    result["knowledge_extraction"]["details"] = extracted_knowledge
            
            # Add working memory status
            working_memory_info = await self._get_session_memory(session_id)
            result["working_memory"] = {
                "message_count": working_memory_info.get("message_count", 0),
                "last_updated": working_memory_info.get("last_updated"),
                "context_size": len(str(working_memory_info.get("working_memory", {})))
            }
            
            self.logger.debug(f"Enhanced memory processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Enhanced memory processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "component": "ff_enhanced_memory",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": processing_time
                }
            }
    
    async def _advanced_rag_retrieval(self, 
                                    user_id: str,
                                    query: str,
                                    augmented_context: Dict[str, Any],
                                    strategy: FFRAGStrategy,
                                    limit: int = 5) -> FFRAGResult:
        """Perform advanced RAG retrieval with multiple strategies"""
        start_time = time.time()
        
        try:
            memories = []
            retrieval_metadata = {}
            
            # Strategy-specific retrieval
            if strategy == FFRAGStrategy.SIMILARITY_ONLY:
                memories = await self._similarity_based_retrieval(user_id, query, limit)
            
            elif strategy == FFRAGStrategy.HYBRID_SEARCH:
                memories = await self._hybrid_retrieval(user_id, query, augmented_context, limit)
            
            elif strategy == FFRAGStrategy.CONTEXTUAL_RETRIEVAL:
                memories = await self._contextual_retrieval(user_id, query, augmented_context, limit)
            
            elif strategy == FFRAGStrategy.TEMPORAL_WEIGHTED:
                memories = await self._temporal_weighted_retrieval(user_id, query, augmented_context, limit)
            
            elif strategy == FFRAGStrategy.KNOWLEDGE_GRAPH_TRAVERSAL:
                memories = await self._knowledge_graph_retrieval(user_id, query, limit)
                
            elif strategy == FFRAGStrategy.MULTI_HOP_REASONING:
                memories = await self._multi_hop_retrieval(user_id, query, augmented_context, limit)
                self._memory_stats["multi_hop_queries"] += 1
            
            else:
                # Default to hybrid search
                memories = await self._hybrid_retrieval(user_id, query, augmented_context, limit)
            
            # Rank memories using advanced ranking
            ranked_memories = await self._memory_ranker.rank_memories(
                memories, query, augmented_context, strategy
            )
            
            # Generate context summary
            context_summary = await self._generate_context_summary(ranked_memories, query)
            
            # Calculate confidence score
            confidence_score = await self._calculate_retrieval_confidence(ranked_memories, query)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create RAG result
            rag_result = FFRAGResult(
                memories=ranked_memories,
                context_summary=context_summary,
                retrieval_strategy=strategy,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                retrieval_metadata=retrieval_metadata
            )
            
            # Store retrieval in history for learning
            self._retrieval_history.append({
                'timestamp': datetime.now(),
                'user_id': user_id,
                'query': query,
                'strategy': strategy,
                'results_count': len(memories),
                'confidence': confidence_score
            })
            
            return rag_result
            
        except Exception as e:
            self.logger.error(f"Advanced RAG retrieval failed: {e}")
            return FFRAGResult(
                memories=[],
                context_summary="",
                retrieval_strategy=strategy,
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                retrieval_metadata={"error": str(e)}
            )
    
    async def _similarity_based_retrieval(self, user_id: str, query: str, limit: int) -> List[FFEnhancedMemoryEntry]:
        """Pure similarity-based retrieval"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Search in vector storage
            results = await self.vector_manager.search_similar_vectors(
                query_vector=query_embedding,
                limit=limit * 2,  # Get more for filtering
                similarity_threshold=0.5
            )
            
            memories = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("user_id") == user_id:
                    memory = await self._result_to_enhanced_memory(result)
                    if memory:
                        memories.append(memory)
            
            return memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Similarity-based retrieval failed: {e}")
            return []
    
    async def _hybrid_retrieval(self, user_id: str, query: str, context: Dict[str, Any], limit: int) -> List[FFEnhancedMemoryEntry]:
        """Hybrid retrieval combining multiple approaches"""
        try:
            all_memories = []
            
            # Vector similarity search
            vector_memories = await self._similarity_based_retrieval(user_id, query, limit)
            all_memories.extend(vector_memories)
            
            # Text-based search
            text_memories = await self._text_based_retrieval(user_id, query, limit)
            all_memories.extend(text_memories)
            
            # Context-aware retrieval
            if context.get('current_topics'):
                topic_memories = await self._topic_based_retrieval(user_id, context['current_topics'], limit//2)
                all_memories.extend(topic_memories)
            
            # Remove duplicates
            unique_memories = self._remove_duplicate_memories(all_memories)
            
            return unique_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    async def _contextual_retrieval(self, user_id: str, query: str, context: Dict[str, Any], limit: int) -> List[FFEnhancedMemoryEntry]:
        """Context-aware retrieval"""
        try:
            memories = []
            
            # Session continuity
            session_id = context.get('session_id')
            if session_id:
                session_memories = await self._get_session_memories(user_id, session_id, limit//2)
                memories.extend(session_memories)
            
            # Temporal context
            if 'time_window' in context:
                temporal_memories = await self._get_temporal_memories(user_id, context['time_window'], limit//2)
                memories.extend(temporal_memories)
            
            # Topic context
            if context.get('current_topics'):
                topic_memories = await self._topic_based_retrieval(user_id, context['current_topics'], limit//2)
                memories.extend(topic_memories)
            
            # Remove duplicates and rank by contextual relevance
            unique_memories = self._remove_duplicate_memories(memories)
            
            return unique_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Contextual retrieval failed: {e}")
            return []
    
    async def _temporal_weighted_retrieval(self, user_id: str, query: str, context: Dict[str, Any], limit: int) -> List[FFEnhancedMemoryEntry]:
        """Temporal-weighted retrieval"""
        try:
            # Get base similarity results
            base_memories = await self._similarity_based_retrieval(user_id, query, limit * 2)
            
            # Apply temporal weighting
            current_time = datetime.now()
            for memory in base_memories:
                age_days = (current_time - memory.created_at).days
                
                # Different decay rates for different memory types
                if memory.memory_type in [FFMemoryType.SEMANTIC, FFMemoryType.DECLARATIVE]:
                    decay = np.exp(-0.01 * age_days)  # Slower decay
                else:
                    decay = np.exp(-0.02 * age_days)  # Faster decay
                
                # Boost recently accessed memories
                access_boost = 1.0
                if memory.last_accessed:
                    days_since_access = (current_time - memory.last_accessed).days
                    access_boost = 1.0 + np.exp(-0.05 * days_since_access)
                
                # Update retrieval score
                base_score = memory.metadata.get('similarity_score', 0.5)
                temporal_score = base_score * decay * access_boost
                memory.metadata['temporal_weighted_score'] = temporal_score
            
            # Sort by temporal-weighted score
            base_memories.sort(key=lambda m: m.metadata.get('temporal_weighted_score', 0.0), reverse=True)
            
            return base_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Temporal-weighted retrieval failed: {e}")
            return []
    
    async def _knowledge_graph_retrieval(self, user_id: str, query: str, limit: int) -> List[FFEnhancedMemoryEntry]:
        """Knowledge graph traversal retrieval"""
        try:
            memories = []
            
            # Find relevant knowledge graph nodes
            relevant_nodes = await self._find_relevant_kg_nodes(query)
            
            for node in relevant_nodes:
                # Get memories connected to this node
                node_memories = await self._get_memories_for_kg_node(user_id, node.node_id)
                memories.extend(node_memories)
                
                # Traverse connections
                for connected_node_id in node.connections:
                    if connected_node_id in self._knowledge_graph:
                        connected_memories = await self._get_memories_for_kg_node(user_id, connected_node_id)
                        memories.extend(connected_memories)
            
            # Remove duplicates and rank by graph relevance
            unique_memories = self._remove_duplicate_memories(memories)
            
            return unique_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Knowledge graph retrieval failed: {e}")
            return []
    
    async def _multi_hop_retrieval(self, user_id: str, query: str, context: Dict[str, Any], limit: int) -> List[FFEnhancedMemoryEntry]:
        """Multi-hop reasoning retrieval"""
        try:
            all_memories = []
            
            # First hop: Direct similarity
            first_hop_memories = await self._similarity_based_retrieval(user_id, query, limit)
            all_memories.extend(first_hop_memories)
            
            # Second hop: Related memories
            for memory in first_hop_memories:
                related_memories = await self._get_related_memories(user_id, memory.memory_id, limit//4)
                all_memories.extend(related_memories)
            
            # Third hop: Knowledge graph connections
            for memory in first_hop_memories:
                kg_memories = await self._get_kg_connected_memories(user_id, memory.memory_id, limit//4)
                all_memories.extend(kg_memories)
            
            # Remove duplicates and rank by multi-hop relevance
            unique_memories = self._remove_duplicate_memories(all_memories)
            
            return unique_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Multi-hop retrieval failed: {e}")
            return []
    
    # Helper methods for enhanced functionality
    
    async def _create_enhanced_memory_entry(self, 
                                           user_id: str,
                                           session_id: str,
                                           message: FFMessageDTO,
                                           context: Dict[str, Any],
                                           extracted_knowledge: Dict[str, Any]) -> FFEnhancedMemoryEntry:
        """Create enhanced memory entry with additional metadata"""
        memory_id = self._generate_memory_id(user_id, session_id, message.content)
        
        # Generate embeddings
        content_embedding = await self._generate_embedding(message.content)
        context_embedding = await self._generate_context_embedding(context)
        
        # Extract semantic concepts
        semantic_concepts = extracted_knowledge.get('concepts', [])
        
        # Determine memory type
        memory_type = self._determine_enhanced_memory_type(message, context, extracted_knowledge)
        
        # Calculate importance score
        importance_score = await self._calculate_enhanced_importance_score(
            message.content, context, extracted_knowledge
        )
        
        enhanced_memory = FFEnhancedMemoryEntry(
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            content=message.content,
            embedding=content_embedding,
            metadata={
                "message_id": getattr(message, 'message_id', str(uuid.uuid4())),
                "timestamp": getattr(message, 'timestamp', datetime.now()),
                "role": getattr(message, 'role', 'user'),
                "context": context,
                "extracted_knowledge": extracted_knowledge
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            importance_score=importance_score,
            context_embedding=context_embedding,
            semantic_concepts=semantic_concepts,
            confidence_score=1.0
        )
        
        return enhanced_memory
    
    async def _store_enhanced_memory(self, memory: FFEnhancedMemoryEntry) -> bool:
        """Store enhanced memory with all metadata"""
        try:
            # Check if memory already exists
            if await self._memory_exists(memory.memory_id):
                await self._update_memory_access(memory.memory_id)
                return True
            
            # Store in vector storage with enhanced metadata
            if self.vector_manager and memory.embedding:
                vector_metadata = {
                    "user_id": memory.user_id,
                    "session_id": memory.session_id,
                    "memory_type": memory.memory_type.value,
                    "content": memory.content,
                    "created_at": memory.created_at.isoformat(),
                    "importance_score": memory.importance_score,
                    "confidence_score": memory.confidence_score,
                    "semantic_concepts": memory.semantic_concepts,
                    **memory.metadata
                }
                
                success = await self.vector_manager.store_vector(
                    doc_id=memory.memory_id,
                    vector=memory.embedding,
                    metadata=vector_metadata
                )
                
                if not success:
                    self.logger.error(f"Failed to store memory vector {memory.memory_id}")
                    return False
            
            # Store in text storage for search
            message_id = await self.storage_manager.add_message(
                user_id=memory.user_id,
                session_id=f"memory_{memory.memory_type.value}_{memory.user_id}",
                role="memory",
                content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type.value,
                    "original_session_id": memory.session_id,
                    "importance_score": memory.importance_score,
                    "confidence_score": memory.confidence_score,
                    "semantic_concepts": memory.semantic_concepts,
                    **memory.metadata
                }
            )
            
            if not message_id:
                self.logger.error(f"Failed to store memory in text storage")
                return False
            
            # Cache the memory
            self._memory_cache[memory.memory_id] = memory
            self._cache_timestamps[memory.memory_id] = time.time()
            
            # Update statistics
            self._memory_stats["total_memories_stored"] += 1
            
            self.logger.debug(f"Successfully stored enhanced memory {memory.memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced memory: {e}")
            return False
    
    async def _update_knowledge_graph(self, memory: FFEnhancedMemoryEntry, knowledge: Dict[str, Any]) -> None:
        """Update knowledge graph with extracted knowledge"""
        try:
            entities = knowledge.get('entities', [])
            relationships = knowledge.get('relationships', [])
            
            # Create or update nodes for entities
            for entity in entities:
                node_id = self._generate_kg_node_id(entity['text'], entity['type'])
                
                if node_id not in self._knowledge_graph:
                    # Create new node
                    node = FFKnowledgeGraphNode(
                        node_id=node_id,
                        content=entity['text'],
                        node_type=entity['type'],
                        embedding=await self._generate_embedding(entity['text']),
                        importance_score=entity.get('confidence', 0.5)
                    )
                    self._knowledge_graph[node_id] = node
                else:
                    # Update existing node
                    node = self._knowledge_graph[node_id]
                    node.update_count += 1
                    node.importance_score = max(node.importance_score, entity.get('confidence', 0.5))
                
                # Link memory to knowledge graph node
                if node_id not in memory.knowledge_graph_nodes:
                    memory.knowledge_graph_nodes.append(node_id)
            
            # Create connections based on relationships
            for relationship in relationships:
                subject_id = self._generate_kg_node_id(relationship['subject'], 'ENTITY')
                object_id = self._generate_kg_node_id(relationship['object'], 'ENTITY')
                
                # Add bidirectional connections
                if subject_id in self._knowledge_graph and object_id in self._knowledge_graph:
                    if object_id not in self._knowledge_graph[subject_id].connections:
                        self._knowledge_graph[subject_id].connections.append(object_id)
                    if subject_id not in self._knowledge_graph[object_id].connections:
                        self._knowledge_graph[object_id].connections.append(subject_id)
            
        except Exception as e:
            self.logger.error(f"Knowledge graph update failed: {e}")
    
    async def _update_memory_clusters(self, user_id: str) -> None:
        """Update memory clusters for improved organization"""
        try:
            # Get user's memories
            user_memories = await self._get_user_memories(user_id)
            
            if len(user_memories) < 3:  # Need minimum memories for clustering
                return
            
            # Simple clustering based on semantic similarity
            embeddings = []
            memory_ids = []
            
            for memory in user_memories:
                if memory.embedding:
                    embeddings.append(memory.embedding)
                    memory_ids.append(memory.memory_id)
            
            if len(embeddings) < 3:
                return
            
            # Perform clustering (simplified k-means-like approach)
            clusters = await self._perform_memory_clustering(embeddings, memory_ids)
            
            # Update memory cluster assignments
            for cluster_id, cluster_data in clusters.items():
                cluster = FFMemoryCluster(
                    cluster_id=cluster_id,
                    memory_ids=cluster_data['memory_ids'],
                    centroid_embedding=cluster_data['centroid'],
                    cluster_theme=cluster_data.get('theme', 'General'),
                    confidence=cluster_data.get('confidence', 0.5),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                self._memory_clusters[cluster_id] = cluster
                
                # Update individual memories with cluster assignment
                for memory_id in cluster_data['memory_ids']:
                    if memory_id in self._memory_cache:
                        memory = self._memory_cache[memory_id]
                        if cluster_id not in memory.memory_clusters:
                            memory.memory_clusters.append(cluster_id)
        
        except Exception as e:
            self.logger.error(f"Memory clustering failed: {e}")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get enhanced component information"""
        return {
            "component_name": "FF Enhanced Memory Component",
            "version": "3.0.0",
            "description": "Advanced memory management with RAG capabilities, knowledge graphs, and contextual retrieval",
            "capabilities": [
                "persistent_memory",
                "working_memory",
                "advanced_rag_retrieval",
                "knowledge_graph_management",
                "memory_clustering",
                "quality_assessment",
                "contextual_augmentation",
                "multi_hop_reasoning"
            ],
            "supported_strategies": [strategy.value for strategy in FFRAGStrategy],
            "supported_memory_types": [memory_type.value for memory_type in FFMemoryType],
            "status": "active" if self._initialized else "inactive",
            "statistics": self.get_enhanced_statistics()
        }
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced component statistics"""
        stats = self._memory_stats.copy()
        
        # Add component-specific metrics
        stats.update({
            "knowledge_graph_nodes": len(self._knowledge_graph),
            "memory_clusters": len(self._memory_clusters),
            "cached_memories": len(self._memory_cache),
            "working_memory_sessions": len(self._working_memory),
            "retrieval_history_size": len(self._retrieval_history),
            "context_embeddings": len(self._context_embeddings)
        })
        
        return stats
    
    # Additional helper methods (abbreviated for space)
    
    def _generate_memory_id(self, user_id: str, session_id: str, content: str) -> str:
        """Generate unique memory ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"enhanced_mem_{user_id}_{session_id}_{content_hash}_{timestamp}"
    
    def _generate_kg_node_id(self, content: str, node_type: str) -> str:
        """Generate knowledge graph node ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"kg_{node_type.lower()}_{content_hash}"
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        # Placeholder - would integrate with actual embedding service
        import random
        return [random.random() for _ in range(384)]  # Common embedding dimension
    
    async def _generate_context_embedding(self, context: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for context"""
        # Convert context to text and generate embedding
        context_text = json.dumps(context, default=str)
        return await self._generate_embedding(context_text)
    
    # ... (Additional helper methods would be implemented here)
    
    async def cleanup(self) -> None:
        """Enhanced cleanup with Phase 3 resources"""
        try:
            self.logger.info("Cleaning up Enhanced FF Memory Component...")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._consolidation_task:
                self._consolidation_task.cancel()
            if self._quality_assessment_task:
                self._quality_assessment_task.cancel()
            
            # Clear enhanced memory structures
            self._working_memory.clear()
            self._memory_cache.clear()
            self._cache_timestamps.clear()
            self._knowledge_graph.clear()
            self._memory_clusters.clear()
            self._retrieval_history.clear()
            self._context_embeddings.clear()
            self._rag_strategies.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("Enhanced FF Memory Component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Enhanced memory component cleanup failed: {e}")
            raise