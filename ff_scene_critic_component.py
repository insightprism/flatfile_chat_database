"""
FF Scene Critic Component - Specialized Creative Feedback System

Provides advanced creative feedback on visual and narrative content using
enhanced multimodal processing combined with expert persona systems.
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_multimodal_config import FFMultimodalConfigDTO, FFMediaType
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_protocols.ff_message_dto import FFMessageDTO
from ff_multimodal_component import FFMultimodalComponent, FFProcessingResult
from ff_multi_agent_component import FFMultiAgentComponent
from ff_managers.ff_storage_manager import FFStorageManager
from ff_managers.ff_document_manager import FFDocumentManager

logger = get_logger(__name__)

class CritiqueStyle(Enum):
    """Available critique styles"""
    CONSTRUCTIVE = "constructive"
    ANALYTICAL = "analytical"
    ENCOURAGING = "encouraging"
    PROFESSIONAL = "professional"
    ARTISTIC = "artistic"
    TECHNICAL = "technical"

class CritiqueDomain(Enum):
    """Critique domains"""
    VISUAL = "visual"
    NARRATIVE = "narrative"
    CINEMATIC = "cinematic"
    DESIGN = "design"
    PHOTOGRAPHY = "photography"
    ANIMATION = "animation"
    WRITING = "writing"
    MIXED_MEDIA = "mixed_media"

class FeedbackAspect(Enum):
    """Aspects to analyze"""
    COMPOSITION = "composition"
    COLOR_THEORY = "color_theory"
    LIGHTING = "lighting"
    NARRATIVE_STRUCTURE = "narrative_structure"
    CHARACTER_DEVELOPMENT = "character_development"
    PACING = "pacing"
    TECHNICAL_EXECUTION = "technical_execution"
    EMOTIONAL_IMPACT = "emotional_impact"
    ORIGINALITY = "originality"
    COHERENCE = "coherence"

@dataclass
class FFSceneCriticConfig:
    """Configuration for Scene Critic component"""
    critique_style: CritiqueStyle = CritiqueStyle.CONSTRUCTIVE
    primary_domains: List[CritiqueDomain] = field(default_factory=lambda: [CritiqueDomain.VISUAL, CritiqueDomain.NARRATIVE])
    feedback_aspects: List[FeedbackAspect] = field(default_factory=lambda: [
        FeedbackAspect.COMPOSITION,
        FeedbackAspect.NARRATIVE_STRUCTURE,
        FeedbackAspect.EMOTIONAL_IMPACT
    ])
    feedback_depth: str = "detailed"  # basic, detailed, comprehensive
    include_suggestions: bool = True
    include_examples: bool = True
    expert_personas: List[str] = field(default_factory=lambda: [
        "visual_artist", "narrative_expert", "technical_critic"
    ])
    max_analysis_time_seconds: int = 300
    enable_collaborative_critique: bool = True

@dataclass
class FFCritiqueAnalysis:
    """Analysis result for a specific aspect"""
    aspect: FeedbackAspect
    score: float  # 0.0 to 1.0
    feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_suggestions: List[str]
    examples: List[str]
    confidence: float

@dataclass
class FFSceneCritique:
    """Complete scene critique result"""
    critique_id: str
    session_id: str
    user_id: str
    content_type: str
    content_description: str
    overall_score: float
    domain_analyses: Dict[CritiqueDomain, Dict[str, Any]]
    aspect_analyses: Dict[FeedbackAspect, FFCritiqueAnalysis]
    expert_opinions: Dict[str, Dict[str, Any]]
    summary: str
    key_recommendations: List[str]
    processing_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class FFSceneCriticComponent(FFChatComponentProtocol):
    """
    FF Scene Critic Component for advanced creative feedback.
    
    Combines multimodal processing with expert persona systems to provide
    comprehensive, nuanced feedback on creative content.
    """
    
    def __init__(self, 
                 config: FFSceneCriticConfig,
                 multimodal_component: FFMultimodalComponent,
                 multi_agent_component: FFMultiAgentComponent,
                 storage_manager: FFStorageManager,
                 document_manager: FFDocumentManager):
        """
        Initialize Scene Critic Component.
        
        Args:
            config: Scene critic configuration
            multimodal_component: Multimodal processing component
            multi_agent_component: Multi-agent system for expert personas
            storage_manager: FF storage manager
            document_manager: FF document manager
        """
        self.config = config
        self.multimodal_component = multimodal_component
        self.multi_agent_component = multi_agent_component
        self.storage_manager = storage_manager
        self.document_manager = document_manager
        self.logger = get_logger(__name__)
        
        # Expert persona definitions
        self.expert_personas = self._initialize_expert_personas()
        
        # Analysis cache
        self.critique_cache: Dict[str, FFSceneCritique] = {}
        
        # Performance metrics
        self.metrics = {
            "total_critiques": 0,
            "avg_processing_time": 0.0,
            "domain_analysis_count": {domain.value: 0 for domain in CritiqueDomain},
            "aspect_analysis_count": {aspect.value: 0 for aspect in FeedbackAspect},
            "satisfaction_scores": []
        }
        
        self._initialized = False
    
    def _initialize_expert_personas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize expert persona configurations"""
        return {
            "visual_artist": {
                "name": "Visual Art Expert",
                "expertise": ["composition", "color_theory", "visual_design", "artistic_techniques"],
                "style": "Passionate about visual storytelling and aesthetic excellence",
                "focus_aspects": [FeedbackAspect.COMPOSITION, FeedbackAspect.COLOR_THEORY, FeedbackAspect.EMOTIONAL_IMPACT],
                "personality": "encouraging yet discerning, with deep appreciation for artistic vision"
            },
            "narrative_expert": {
                "name": "Narrative Specialist",
                "expertise": ["storytelling", "character_development", "plot_structure", "dialogue"],
                "style": "Focused on story coherence and narrative impact",
                "focus_aspects": [FeedbackAspect.NARRATIVE_STRUCTURE, FeedbackAspect.CHARACTER_DEVELOPMENT, FeedbackAspect.PACING],
                "personality": "analytical yet supportive, with expertise in story craft"
            },
            "technical_critic": {
                "name": "Technical Production Expert",
                "expertise": ["technical_execution", "production_quality", "cinematography", "editing"],
                "style": "Detail-oriented analysis of technical aspects and production values",
                "focus_aspects": [FeedbackAspect.TECHNICAL_EXECUTION, FeedbackAspect.LIGHTING, FeedbackAspect.COHERENCE],
                "personality": "precise and constructive, focused on craft excellence"
            },
            "cinematic_analyst": {
                "name": "Cinema Studies Expert",
                "expertise": ["film_theory", "visual_language", "genre_conventions", "cultural_context"],
                "style": "Academic approach to visual media analysis",
                "focus_aspects": [FeedbackAspect.COMPOSITION, FeedbackAspect.NARRATIVE_STRUCTURE, FeedbackAspect.ORIGINALITY],
                "personality": "scholarly yet accessible, connecting theory to practice"
            },
            "design_critic": {
                "name": "Design Professional",
                "expertise": ["graphic_design", "user_experience", "visual_hierarchy", "brand_consistency"],
                "style": "User-centered design perspective with commercial awareness",
                "focus_aspects": [FeedbackAspect.COMPOSITION, FeedbackAspect.TECHNICAL_EXECUTION, FeedbackAspect.COHERENCE],
                "personality": "practical and solution-oriented, balancing aesthetics with function"
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the Scene Critic component"""
        try:
            self.logger.info("Initializing FF Scene Critic Component...")
            
            # Verify dependencies are initialized
            if not getattr(self.multimodal_component, '_initialized', False):
                raise RuntimeError("Multimodal component not initialized")
            
            if not getattr(self.multi_agent_component, '_initialized', False):
                raise RuntimeError("Multi-agent component not initialized")
            
            # Initialize expert personas in multi-agent system
            for persona_id, persona_config in self.expert_personas.items():
                if persona_id in self.config.expert_personas:
                    await self.multi_agent_component.add_agent_persona(persona_id, persona_config)
            
            self._initialized = True
            self.logger.info("FF Scene Critic Component initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Scene Critic component: {e}")
            return False
    
    async def process_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message: FFMessageDTO, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message for scene critique analysis.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: Message containing content to critique
            context: Additional context and parameters
            
        Returns:
            Dict containing comprehensive critique analysis
        """
        if not self._initialized:
            return {
                "success": False,
                "error": "Scene Critic component not initialized",
                "component": "ff_scene_critic"
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.info(f"Processing scene critique for session {session_id}")
            
            # Extract content for analysis
            content_info = await self._extract_content_info(message, context)
            
            if not content_info:
                return {
                    "success": False,
                    "error": "No analyzable content found in message",
                    "component": "ff_scene_critic"
                }
            
            # Perform multimodal analysis first
            multimodal_results = await self._perform_multimodal_analysis(
                session_id, user_id, content_info
            )
            
            # Generate comprehensive critique
            critique = await self._generate_comprehensive_critique(
                session_id, user_id, message.message_id, content_info, multimodal_results, context
            )
            
            # Store critique for future reference
            await self._store_critique(critique)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(critique, processing_time)
            
            result = {
                "success": True,
                "component": "ff_scene_critic",
                "processor": "expert_multimodal_analysis",
                "critique_id": critique.critique_id,
                "response_content": self._format_critique_response(critique),
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "content_type": critique.content_type,
                    "overall_score": critique.overall_score,
                    "processing_time": processing_time,
                    "expert_opinions_count": len(critique.expert_opinions),
                    "aspects_analyzed": len(critique.aspect_analyses),
                    "key_recommendations_count": len(critique.key_recommendations)
                },
                "detailed_analysis": {
                    "overall_score": critique.overall_score,
                    "domain_analyses": critique.domain_analyses,
                    "aspect_scores": {
                        aspect.value: analysis.score 
                        for aspect, analysis in critique.aspect_analyses.items()
                    },
                    "expert_opinions": critique.expert_opinions,
                    "key_recommendations": critique.key_recommendations
                }
            }
            
            self.logger.info(f"Scene critique completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in scene critique processing: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "component": "ff_scene_critic",
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": processing_time
                }
            }
    
    async def _extract_content_info(self, message: FFMessageDTO, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract content information from message"""
        content_info = {
            "text_content": message.content,
            "media_attachments": getattr(message, 'attachments', []),
            "content_type": "text",
            "analysis_domains": []
        }
        
        # Check for media attachments
        if content_info["media_attachments"]:
            content_info["content_type"] = "mixed_media"
            
            # Determine primary domains based on media types
            for attachment in content_info["media_attachments"]:
                media_type = self._detect_media_type(attachment)
                if media_type == FFMediaType.IMAGE:
                    content_info["analysis_domains"].extend([CritiqueDomain.VISUAL, CritiqueDomain.DESIGN])
                elif media_type == FFMediaType.VIDEO:
                    content_info["analysis_domains"].extend([CritiqueDomain.CINEMATIC, CritiqueDomain.VISUAL])
                elif media_type == FFMediaType.AUDIO:
                    content_info["analysis_domains"].append(CritiqueDomain.NARRATIVE)
        
        # Check for narrative content
        if len(message.content) > 100:  # Substantial text content
            content_info["analysis_domains"].append(CritiqueDomain.NARRATIVE)
            if not content_info["media_attachments"]:
                content_info["content_type"] = "narrative"
        
        # Remove duplicates and limit to configured domains
        content_info["analysis_domains"] = list(set(content_info["analysis_domains"]))
        content_info["analysis_domains"] = [
            domain for domain in content_info["analysis_domains"]
            if domain in self.config.primary_domains
        ]
        
        return content_info if content_info["analysis_domains"] else None
    
    def _detect_media_type(self, attachment: str) -> Optional[FFMediaType]:
        """Detect media type from attachment"""
        try:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(attachment)
            
            if mime_type:
                if mime_type.startswith('image/'):
                    return FFMediaType.IMAGE
                elif mime_type.startswith('video/'):
                    return FFMediaType.VIDEO
                elif mime_type.startswith('audio/'):
                    return FFMediaType.AUDIO
                elif mime_type in ['application/pdf', 'text/plain']:
                    return FFMediaType.DOCUMENT
            
            return None
        except Exception:
            return None
    
    async def _perform_multimodal_analysis(self, session_id: str, user_id: str, 
                                         content_info: Dict[str, Any]) -> List[FFProcessingResult]:
        """Perform multimodal analysis on content"""
        results = []
        
        # Process media attachments
        for attachment in content_info["media_attachments"]:
            try:
                # Create dummy message for multimodal processing
                media_message = FFMessageDTO(
                    message_id=f"critique_{uuid.uuid4().hex[:8]}",
                    role="user",
                    content="Scene critique analysis",
                    attachments=[attachment]
                )
                
                multimodal_result = await self.multimodal_component.process_message(
                    session_id, user_id, media_message, {"analysis_mode": "comprehensive"}
                )
                
                if multimodal_result.get("success"):
                    results.extend(multimodal_result.get("processing_results", []))
                    
            except Exception as e:
                self.logger.error(f"Error processing attachment {attachment}: {e}")
        
        return results
    
    async def _generate_comprehensive_critique(self, session_id: str, user_id: str, message_id: str,
                                             content_info: Dict[str, Any], 
                                             multimodal_results: List[FFProcessingResult],
                                             context: Dict[str, Any]) -> FFSceneCritique:
        """Generate comprehensive critique using expert personas"""
        critique_id = f"critique_{uuid.uuid4().hex}"
        
        # Initialize critique
        critique = FFSceneCritique(
            critique_id=critique_id,
            session_id=session_id,
            user_id=user_id,
            content_type=content_info["content_type"],
            content_description=content_info["text_content"][:200] + "..." if len(content_info["text_content"]) > 200 else content_info["text_content"],
            overall_score=0.0,
            domain_analyses={},
            aspect_analyses={},
            expert_opinions={},
            summary="",
            key_recommendations=[],
            processing_time_ms=0.0,
            timestamp=datetime.now()
        )
        
        # Analyze each domain
        for domain in content_info["analysis_domains"]:
            domain_analysis = await self._analyze_domain(domain, content_info, multimodal_results)
            critique.domain_analyses[domain] = domain_analysis
        
        # Analyze specific aspects
        for aspect in self.config.feedback_aspects:
            aspect_analysis = await self._analyze_aspect(aspect, content_info, multimodal_results, critique.domain_analyses)
            critique.aspect_analyses[aspect] = aspect_analysis
        
        # Get expert opinions
        if self.config.enable_collaborative_critique:
            expert_opinions = await self._get_expert_opinions(content_info, multimodal_results, context)
            critique.expert_opinions = expert_opinions
        
        # Calculate overall score
        critique.overall_score = self._calculate_overall_score(critique.aspect_analyses)
        
        # Generate summary and recommendations
        critique.summary = await self._generate_critique_summary(critique)
        critique.key_recommendations = await self._generate_key_recommendations(critique)
        
        return critique
    
    async def _analyze_domain(self, domain: CritiqueDomain, content_info: Dict[str, Any], 
                            multimodal_results: List[FFProcessingResult]) -> Dict[str, Any]:
        """Analyze specific domain aspects"""
        analysis = {
            "domain": domain.value,
            "applicable": True,
            "confidence": 0.8,
            "findings": [],
            "score": 0.0
        }
        
        if domain == CritiqueDomain.VISUAL:
            analysis.update(await self._analyze_visual_domain(content_info, multimodal_results))
        elif domain == CritiqueDomain.NARRATIVE:
            analysis.update(await self._analyze_narrative_domain(content_info))
        elif domain == CritiqueDomain.CINEMATIC:
            analysis.update(await self._analyze_cinematic_domain(content_info, multimodal_results))
        elif domain == CritiqueDomain.DESIGN:
            analysis.update(await self._analyze_design_domain(content_info, multimodal_results))
        
        return analysis
    
    async def _analyze_visual_domain(self, content_info: Dict[str, Any], 
                                   multimodal_results: List[FFProcessingResult]) -> Dict[str, Any]:
        """Analyze visual domain aspects"""
        findings = []
        scores = []
        
        # Analyze composition from multimodal results
        for result in multimodal_results:
            if result.media_type == FFMediaType.IMAGE:
                visual_analysis = result.results.get("visual_analysis", {})
                
                # Composition analysis
                composition_score = visual_analysis.get("composition_score", 0.5)
                scores.append(composition_score)
                findings.append(f"Composition analysis: {visual_analysis.get('composition_notes', 'Standard composition detected')}")
                
                # Color analysis
                color_analysis = visual_analysis.get("color_analysis", {})
                if color_analysis:
                    findings.append(f"Color scheme: {color_analysis.get('dominant_colors', 'Mixed palette')}")
                    findings.append(f"Color harmony: {color_analysis.get('harmony_assessment', 'Balanced')}")
        
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        return {
            "score": avg_score,
            "findings": findings,
            "confidence": 0.8 if findings else 0.3
        }
    
    async def _analyze_narrative_domain(self, content_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze narrative domain aspects"""
        text_content = content_info["text_content"]
        findings = []
        
        # Basic narrative analysis
        word_count = len(text_content.split())
        sentence_count = len([s for s in text_content.split('.') if s.strip()])
        
        if word_count > 50:
            findings.append(f"Substantial narrative content: {word_count} words")
            
            # Check for narrative elements
            if any(word in text_content.lower() for word in ["character", "story", "plot", "dialogue"]):
                findings.append("Narrative elements detected")
                score = 0.7
            else:
                findings.append("Descriptive content without clear narrative structure")
                score = 0.5
        else:
            findings.append("Limited narrative content")
            score = 0.3
        
        return {
            "score": score,
            "findings": findings,
            "confidence": 0.6
        }
    
    async def _analyze_cinematic_domain(self, content_info: Dict[str, Any], 
                                      multimodal_results: List[FFProcessingResult]) -> Dict[str, Any]:
        """Analyze cinematic domain aspects"""
        findings = []
        scores = []
        
        for result in multimodal_results:
            if result.media_type == FFMediaType.VIDEO:
                video_analysis = result.results.get("video_analysis", {})
                
                # Cinematic techniques
                techniques = video_analysis.get("cinematic_techniques", [])
                if techniques:
                    findings.append(f"Cinematic techniques: {', '.join(techniques)}")
                    scores.append(0.8)
                
                # Pacing analysis
                pacing = video_analysis.get("pacing_analysis", {})
                if pacing:
                    findings.append(f"Pacing: {pacing.get('assessment', 'Standard')}")
                    scores.append(pacing.get("score", 0.6))
        
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        return {
            "score": avg_score,
            "findings": findings if findings else ["Limited cinematic content for analysis"],
            "confidence": 0.7 if findings else 0.2
        }
    
    async def _analyze_design_domain(self, content_info: Dict[str, Any], 
                                   multimodal_results: List[FFProcessingResult]) -> Dict[str, Any]:
        """Analyze design domain aspects"""
        findings = []
        score = 0.5
        
        # Design principles analysis
        for result in multimodal_results:
            if result.media_type in [FFMediaType.IMAGE, FFMediaType.DOCUMENT]:
                design_analysis = result.results.get("design_analysis", {})
                
                if design_analysis:
                    findings.append(f"Design coherence: {design_analysis.get('coherence', 'Acceptable')}")
                    findings.append(f"Visual hierarchy: {design_analysis.get('hierarchy', 'Clear')}")
                    score = design_analysis.get("overall_score", 0.6)
        
        return {
            "score": score,
            "findings": findings if findings else ["Limited design elements for analysis"],
            "confidence": 0.6 if findings else 0.3
        }
    
    async def _analyze_aspect(self, aspect: FeedbackAspect, content_info: Dict[str, Any],
                            multimodal_results: List[FFProcessingResult],
                            domain_analyses: Dict[CritiqueDomain, Dict[str, Any]]) -> FFCritiqueAnalysis:
        """Analyze specific feedback aspect"""
        
        # Base analysis structure
        analysis = FFCritiqueAnalysis(
            aspect=aspect,
            score=0.5,
            feedback="",
            strengths=[],
            areas_for_improvement=[],
            specific_suggestions=[],
            examples=[],
            confidence=0.5
        )
        
        # Aspect-specific analysis
        if aspect == FeedbackAspect.COMPOSITION:
            analysis = await self._analyze_composition_aspect(analysis, multimodal_results)
        elif aspect == FeedbackAspect.COLOR_THEORY:
            analysis = await self._analyze_color_theory_aspect(analysis, multimodal_results)
        elif aspect == FeedbackAspect.NARRATIVE_STRUCTURE:
            analysis = await self._analyze_narrative_structure_aspect(analysis, content_info)
        elif aspect == FeedbackAspect.EMOTIONAL_IMPACT:
            analysis = await self._analyze_emotional_impact_aspect(analysis, content_info, multimodal_results)
        elif aspect == FeedbackAspect.TECHNICAL_EXECUTION:
            analysis = await self._analyze_technical_execution_aspect(analysis, multimodal_results)
        # Add more aspect analyzers as needed
        
        return analysis
    
    async def _analyze_composition_aspect(self, analysis: FFCritiqueAnalysis, 
                                        multimodal_results: List[FFProcessingResult]) -> FFCritiqueAnalysis:
        """Analyze composition aspect"""
        composition_scores = []
        composition_notes = []
        
        for result in multimodal_results:
            if result.media_type == FFMediaType.IMAGE:
                visual_data = result.results.get("visual_analysis", {})
                comp_score = visual_data.get("composition_score", 0.5)
                composition_scores.append(comp_score)
                
                comp_notes = visual_data.get("composition_notes", "")
                if comp_notes:
                    composition_notes.append(comp_notes)
        
        if composition_scores:
            analysis.score = sum(composition_scores) / len(composition_scores)
            analysis.confidence = 0.8
            
            if analysis.score > 0.7:
                analysis.strengths.append("Strong compositional elements")
                analysis.feedback = "The composition demonstrates good understanding of visual balance and focal points."
            elif analysis.score > 0.5:
                analysis.feedback = "The composition is adequate with some effective elements."
                analysis.areas_for_improvement.append("Consider strengthening the focal hierarchy")
            else:
                analysis.areas_for_improvement.append("Composition could benefit from clearer focal points")
                analysis.specific_suggestions.append("Try applying the rule of thirds for better balance")
                analysis.feedback = "The composition has potential but could be strengthened with more attention to visual flow."
        else:
            analysis.feedback = "Limited visual content available for composition analysis."
            analysis.confidence = 0.2
        
        return analysis
    
    async def _analyze_color_theory_aspect(self, analysis: FFCritiqueAnalysis,
                                         multimodal_results: List[FFProcessingResult]) -> FFCritiqueAnalysis:
        """Analyze color theory aspect"""
        # Implementation for color theory analysis
        analysis.feedback = "Color analysis requires visual content for comprehensive evaluation."
        return analysis
    
    async def _analyze_narrative_structure_aspect(self, analysis: FFCritiqueAnalysis,
                                                content_info: Dict[str, Any]) -> FFCritiqueAnalysis:
        """Analyze narrative structure aspect"""
        text_content = content_info["text_content"]
        
        if len(text_content) > 100:
            analysis.confidence = 0.7
            
            # Basic narrative structure analysis
            has_beginning = any(word in text_content.lower()[:100] for word in ["once", "began", "start", "first"])
            has_conflict = any(word in text_content.lower() for word in ["but", "however", "conflict", "problem", "challenge"])
            has_resolution = any(word in text_content.lower()[-100:] for word in ["finally", "end", "resolved", "conclusion"])
            
            structure_score = (has_beginning + has_conflict + has_resolution) / 3
            analysis.score = 0.4 + (structure_score * 0.4)  # Base score + structure bonus
            
            if structure_score > 0.6:
                analysis.strengths.append("Clear narrative progression")
                analysis.feedback = "The narrative demonstrates good structural elements with identifiable progression."
            else:
                analysis.areas_for_improvement.append("Consider strengthening narrative arc")
                analysis.specific_suggestions.append("Ensure clear beginning, middle, and end structure")
                analysis.feedback = "The narrative could benefit from stronger structural elements."
        else:
            analysis.feedback = "Insufficient narrative content for structural analysis."
            analysis.confidence = 0.2
        
        return analysis
    
    async def _analyze_emotional_impact_aspect(self, analysis: FFCritiqueAnalysis,
                                             content_info: Dict[str, Any],
                                             multimodal_results: List[FFProcessingResult]) -> FFCritiqueAnalysis:
        """Analyze emotional impact aspect"""
        # Combine text and visual emotional analysis
        emotional_indicators = []
        
        # Text emotional content
        text_content = content_info["text_content"].lower()
        emotional_words = ["love", "fear", "joy", "anger", "sadness", "hope", "despair", "excitement", "calm", "tension"]
        found_emotions = [word for word in emotional_words if word in text_content]
        
        if found_emotions:
            emotional_indicators.append(f"Emotional vocabulary: {', '.join(found_emotions)}")
        
        # Visual emotional content
        for result in multimodal_results:
            if result.media_type == FFMediaType.IMAGE:
                emotion_data = result.results.get("emotion_analysis", {})
                if emotion_data:
                    emotional_indicators.append(f"Visual mood: {emotion_data.get('dominant_emotion', 'neutral')}")
        
        if emotional_indicators:
            analysis.score = 0.6 + (len(emotional_indicators) * 0.1)
            analysis.confidence = 0.7
            analysis.strengths.append("Demonstrates emotional awareness")
            analysis.feedback = f"The content shows emotional depth with {len(emotional_indicators)} emotional elements identified."
        else:
            analysis.score = 0.4
            analysis.areas_for_improvement.append("Consider adding more emotional resonance")
            analysis.feedback = "The content could benefit from stronger emotional elements to increase audience engagement."
        
        return analysis
    
    async def _analyze_technical_execution_aspect(self, analysis: FFCritiqueAnalysis,
                                                multimodal_results: List[FFProcessingResult]) -> FFCritiqueAnalysis:
        """Analyze technical execution aspect"""
        technical_scores = []
        technical_notes = []
        
        for result in multimodal_results:
            quality_metrics = result.quality_metrics
            if quality_metrics:
                tech_score = quality_metrics.get("technical_quality", 0.5)
                technical_scores.append(tech_score)
                
                if tech_score > 0.8:
                    technical_notes.append("High technical quality")
                elif tech_score > 0.6:
                    technical_notes.append("Good technical execution")
                else:
                    technical_notes.append("Technical aspects could be improved")
        
        if technical_scores:
            analysis.score = sum(technical_scores) / len(technical_scores)
            analysis.confidence = 0.8
            analysis.feedback = f"Technical execution assessment: {'; '.join(technical_notes)}"
        else:
            analysis.feedback = "Limited technical content available for evaluation."
            analysis.confidence = 0.3
        
        return analysis
    
    async def _get_expert_opinions(self, content_info: Dict[str, Any], 
                                 multimodal_results: List[FFProcessingResult],
                                 context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get opinions from expert personas using multi-agent system"""
        expert_opinions = {}
        
        for persona_id in self.config.expert_personas:
            if persona_id in self.expert_personas:
                try:
                    # Create context for expert analysis
                    expert_context = {
                        "persona_id": persona_id,
                        "analysis_focus": self.expert_personas[persona_id]["focus_aspects"],
                        "critique_style": self.config.critique_style.value,
                        "content_summary": content_info["text_content"][:500],
                        "multimodal_findings": [
                            result.results for result in multimodal_results
                        ]
                    }
                    
                    # Get expert opinion through multi-agent system
                    expert_response = await self.multi_agent_component.get_agent_response(
                        persona_id, 
                        "Please provide your expert critique of this creative content.",
                        expert_context
                    )
                    
                    if expert_response.get("success"):
                        expert_opinions[persona_id] = {
                            "expert_name": self.expert_personas[persona_id]["name"],
                            "opinion": expert_response.get("response", ""),
                            "confidence": expert_response.get("confidence", 0.7),
                            "focus_areas": self.expert_personas[persona_id]["focus_aspects"],
                            "expertise": self.expert_personas[persona_id]["expertise"]
                        }
                
                except Exception as e:
                    self.logger.error(f"Error getting opinion from expert {persona_id}: {e}")
        
        return expert_opinions
    
    def _calculate_overall_score(self, aspect_analyses: Dict[FeedbackAspect, FFCritiqueAnalysis]) -> float:
        """Calculate overall critique score"""
        if not aspect_analyses:
            return 0.5
        
        # Weight different aspects
        aspect_weights = {
            FeedbackAspect.COMPOSITION: 0.2,
            FeedbackAspect.NARRATIVE_STRUCTURE: 0.2,
            FeedbackAspect.EMOTIONAL_IMPACT: 0.15,
            FeedbackAspect.TECHNICAL_EXECUTION: 0.15,
            FeedbackAspect.COLOR_THEORY: 0.1,
            FeedbackAspect.ORIGINALITY: 0.1,
            FeedbackAspect.COHERENCE: 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for aspect, analysis in aspect_analyses.items():
            weight = aspect_weights.get(aspect, 0.05)
            weighted_score += analysis.score * weight * analysis.confidence
            total_weight += weight * analysis.confidence
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    async def _generate_critique_summary(self, critique: FFSceneCritique) -> str:
        """Generate overall critique summary"""
        summary_parts = []
        
        # Overall assessment
        if critique.overall_score > 0.8:
            summary_parts.append("This is an excellent piece of creative work with strong execution across multiple areas.")
        elif critique.overall_score > 0.6:
            summary_parts.append("This work demonstrates solid creative skills with some standout elements.")
        elif critique.overall_score > 0.4:
            summary_parts.append("This work shows creative potential with opportunities for refinement.")
        else:
            summary_parts.append("This work has creative foundations that can be developed further.")
        
        # Highlight top strengths
        all_strengths = []
        for analysis in critique.aspect_analyses.values():
            all_strengths.extend(analysis.strengths)
        
        if all_strengths:
            summary_parts.append(f"Key strengths include: {', '.join(all_strengths[:3])}")
        
        # Main areas for development
        all_improvements = []
        for analysis in critique.aspect_analyses.values():
            all_improvements.extend(analysis.areas_for_improvement)
        
        if all_improvements:
            summary_parts.append(f"Primary areas for development: {', '.join(all_improvements[:2])}")
        
        return " ".join(summary_parts)
    
    async def _generate_key_recommendations(self, critique: FFSceneCritique) -> List[str]:
        """Generate key recommendations based on analysis"""
        recommendations = []
        
        # Collect suggestions from all analyses
        all_suggestions = []
        for analysis in critique.aspect_analyses.values():
            all_suggestions.extend(analysis.specific_suggestions)
        
        # Prioritize recommendations based on impact and feasibility
        if all_suggestions:
            # Remove duplicates and select top recommendations
            unique_suggestions = list(set(all_suggestions))
            recommendations = unique_suggestions[:5]  # Top 5 recommendations
        
        # Add domain-specific recommendations
        for domain, domain_analysis in critique.domain_analyses.items():
            if domain_analysis["score"] < 0.6:
                if domain == CritiqueDomain.VISUAL:
                    recommendations.append("Focus on strengthening visual composition and design principles")
                elif domain == CritiqueDomain.NARRATIVE:
                    recommendations.append("Develop clearer narrative structure and character development")
        
        return recommendations[:5]  # Limit to 5 key recommendations
    
    def _format_critique_response(self, critique: FFSceneCritique) -> str:
        """Format critique into readable response"""
        response_parts = []
        
        # Header
        response_parts.append(f"# Scene Critique Analysis")
        response_parts.append(f"**Overall Score:** {critique.overall_score:.1f}/1.0")
        response_parts.append("")
        
        # Summary
        response_parts.append("## Summary")
        response_parts.append(critique.summary)
        response_parts.append("")
        
        # Expert Opinions
        if critique.expert_opinions:
            response_parts.append("## Expert Perspectives")
            for expert_id, opinion in critique.expert_opinions.items():
                response_parts.append(f"### {opinion['expert_name']}")
                response_parts.append(opinion['opinion'])
                response_parts.append("")
        
        # Detailed Analysis
        response_parts.append("## Detailed Analysis")
        for aspect, analysis in critique.aspect_analyses.items():
            response_parts.append(f"### {aspect.value.replace('_', ' ').title()}")
            response_parts.append(f"**Score:** {analysis.score:.1f}/1.0")
            response_parts.append(analysis.feedback)
            
            if analysis.strengths:
                response_parts.append("**Strengths:**")
                for strength in analysis.strengths:
                    response_parts.append(f"- {strength}")
            
            if analysis.areas_for_improvement:
                response_parts.append("**Areas for Improvement:**")
                for improvement in analysis.areas_for_improvement:
                    response_parts.append(f"- {improvement}")
            
            response_parts.append("")
        
        # Key Recommendations
        if critique.key_recommendations:
            response_parts.append("## Key Recommendations")
            for i, recommendation in enumerate(critique.key_recommendations, 1):
                response_parts.append(f"{i}. {recommendation}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    async def _store_critique(self, critique: FFSceneCritique):
        """Store critique for future reference"""
        try:
            # Cache in memory
            self.critique_cache[critique.critique_id] = critique
            
            # Store in FF storage (if available)
            if self.storage_manager:
                critique_data = {
                    "critique_id": critique.critique_id,
                    "session_id": critique.session_id,
                    "user_id": critique.user_id,
                    "overall_score": critique.overall_score,
                    "summary": critique.summary,
                    "recommendations": critique.key_recommendations,
                    "timestamp": critique.timestamp.isoformat(),
                    "metadata": critique.metadata
                }
                
                await self.storage_manager.add_message(
                    user_id=critique.user_id,
                    session_id=f"critiques_{critique.user_id}",
                    role="system",
                    content=json.dumps(critique_data),
                    metadata={"type": "scene_critique", "critique_id": critique.critique_id}
                )
        
        except Exception as e:
            self.logger.error(f"Error storing critique: {e}")
    
    def _update_metrics(self, critique: FFSceneCritique, processing_time: float):
        """Update component metrics"""
        self.metrics["total_critiques"] += 1
        
        # Update average processing time
        total_critiques = self.metrics["total_critiques"]
        current_avg = self.metrics["avg_processing_time"]
        self.metrics["avg_processing_time"] = ((current_avg * (total_critiques - 1)) + processing_time) / total_critiques
        
        # Update domain analysis counts
        for domain in critique.domain_analyses:
            self.metrics["domain_analysis_count"][domain.value] += 1
        
        # Update aspect analysis counts
        for aspect in critique.aspect_analyses:
            self.metrics["aspect_analysis_count"][aspect.value] += 1
        
        # Record satisfaction score (overall score as proxy)
        self.metrics["satisfaction_scores"].append(critique.overall_score)
        
        # Keep only recent satisfaction scores
        if len(self.metrics["satisfaction_scores"]) > 100:
            self.metrics["satisfaction_scores"] = self.metrics["satisfaction_scores"][-100:]
    
    async def get_component_info(self) -> Dict[str, Any]:
        """Get component information and capabilities"""
        return {
            "component_name": "ff_scene_critic",
            "version": "1.0.0",
            "description": "Advanced creative feedback system combining multimodal analysis with expert personas",
            "capabilities": [
                "visual_critique",
                "narrative_analysis", 
                "cinematic_analysis",
                "design_critique",
                "expert_collaboration",
                "comprehensive_feedback"
            ],
            "supported_domains": [domain.value for domain in CritiqueDomain],
            "feedback_aspects": [aspect.value for aspect in FeedbackAspect],
            "expert_personas": list(self.expert_personas.keys()),
            "initialized": self._initialized,
            "metrics": self.metrics
        }
    
    async def cleanup(self):
        """Cleanup component resources"""
        try:
            self.logger.info("Cleaning up FF Scene Critic Component...")
            
            # Clear caches
            self.critique_cache.clear()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("FF Scene Critic Component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during Scene Critic cleanup: {e}")
            raise