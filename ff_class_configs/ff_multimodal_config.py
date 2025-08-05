"""
FF Enhanced Multimodal Component Configuration

Configuration classes for the enhanced FF Multimodal component that provides
advanced multimedia processing using existing FF document processing as backend.
"""

from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from ff_class_configs.ff_base_config import FFBaseConfigDTO


class FFMediaType(Enum):
    """Supported media types for multimodal processing"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    ARCHIVE = "archive"
    CODE = "code"


class FFProcessingMode(Enum):
    """Processing modes for multimodal content"""
    EXTRACT_ONLY = "extract_only"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    CLASSIFY = "classify"
    TRANSFORM = "transform"
    ENHANCE = "enhance"
    FULL_ANALYSIS = "full_analysis"


class FFMultimodalCapability(Enum):
    """Capabilities for multimodal processing"""
    OCR = "ocr"  # Optical Character Recognition
    ASR = "asr"  # Automatic Speech Recognition
    OBJECT_DETECTION = "object_detection"
    FACE_RECOGNITION = "face_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    LANGUAGE_DETECTION = "language_detection"
    CONTENT_MODERATION = "content_moderation"
    METADATA_EXTRACTION = "metadata_extraction"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    FORMAT_CONVERSION = "format_conversion"


@dataclass
class FFMediaProcessingRule:
    """Rule for processing specific media types"""
    media_type: FFMediaType
    file_extensions: List[str]
    max_file_size_mb: int
    processing_mode: FFProcessingMode
    capabilities: List[FFMultimodalCapability]
    enabled: bool = True
    priority: int = 100
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FFImageProcessingConfigDTO:
    """Configuration for image processing"""
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"
    ])
    
    # Size limits
    max_file_size_mb: int = 10
    max_width_pixels: int = 4096
    max_height_pixels: int = 4096
    
    # Processing capabilities
    enable_ocr: bool = True
    enable_object_detection: bool = False  # Requires advanced AI models
    enable_face_detection: bool = False    # Privacy concerns
    enable_content_analysis: bool = True
    
    # OCR settings
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    ocr_confidence_threshold: float = 0.7
    ocr_preserve_layout: bool = True
    
    # Image enhancement
    enable_auto_enhancement: bool = False
    enable_noise_reduction: bool = False
    enable_upscaling: bool = False
    
    # Output settings
    generate_thumbnails: bool = True
    thumbnail_sizes: List[tuple] = field(default_factory=lambda: [(150, 150), (300, 300)])
    extract_exif_data: bool = True
    preserve_original: bool = True


@dataclass
class FFAudioProcessingConfigDTO:
    """Configuration for audio processing"""
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "mp3", "wav", "ogg", "m4a", "flac", "aac"
    ])
    
    # Size and duration limits
    max_file_size_mb: int = 50
    max_duration_minutes: int = 60
    
    # Processing capabilities
    enable_transcription: bool = True
    enable_speaker_identification: bool = False
    enable_sentiment_analysis: bool = True
    enable_language_detection: bool = True
    
    # Transcription settings
    transcription_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr"])
    transcription_accuracy: str = "balanced"  # fast, balanced, accurate
    include_timestamps: bool = True
    include_confidence_scores: bool = True
    
    # Audio analysis
    analyze_audio_quality: bool = True
    detect_silence: bool = True
    extract_audio_features: bool = False
    
    # Output settings
    generate_waveform: bool = False
    extract_audio_metadata: bool = True


@dataclass
class FFVideoProcessingConfigDTO:
    """Configuration for video processing"""
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "mp4", "avi", "mov", "mkv", "webm", "flv"
    ])
    
    # Size and duration limits
    max_file_size_mb: int = 100
    max_duration_minutes: int = 30
    
    # Processing capabilities
    enable_frame_extraction: bool = True
    enable_audio_extraction: bool = True
    enable_scene_detection: bool = False
    enable_object_tracking: bool = False
    
    # Frame extraction settings
    frames_per_second: float = 1.0  # For analysis
    max_frames: int = 100
    extract_keyframes_only: bool = True
    
    # Video analysis
    analyze_video_quality: bool = True
    detect_motion: bool = False
    extract_video_metadata: bool = True
    
    # Output settings
    generate_preview_images: bool = True
    preview_timestamps: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])  # As fraction of duration


@dataclass
class FFDocumentProcessingConfigDTO:
    """Configuration for document processing (enhanced from FF base)"""
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "doc", "txt", "rtf", "odt", "html", "md", "xlsx", "xls", "pptx", "ppt"
    ])
    
    # Size limits
    max_file_size_mb: int = 20
    max_pages: int = 100
    
    # Processing capabilities
    enable_text_extraction: bool = True
    enable_ocr_fallback: bool = True
    enable_structure_analysis: bool = True
    enable_table_extraction: bool = True
    enable_image_extraction: bool = True
    
    # Text processing
    preserve_formatting: bool = True
    extract_metadata: bool = True
    detect_language: bool = True
    perform_spell_check: bool = False
    
    # Advanced features
    enable_semantic_analysis: bool = True
    generate_summary: bool = True
    extract_key_phrases: bool = True
    classify_document_type: bool = True
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["text", "markdown", "html"])
    include_page_numbers: bool = True
    include_section_headers: bool = True


@dataclass
class FFMultimodalConfigDTO(FFBaseConfigDTO):
    """Configuration for FF Enhanced Multimodal component"""
    
    # Core processing settings
    enable_multimodal_processing: bool = True
    max_concurrent_files: int = 3
    processing_timeout_seconds: int = 60
    default_processing_mode: str = "analyze"
    
    # FF integration settings
    use_ff_document_processing: bool = True
    use_ff_storage_backend: bool = True
    store_processed_results: bool = True
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Media type configurations
    image_processing: FFImageProcessingConfigDTO = field(default_factory=FFImageProcessingConfigDTO)
    audio_processing: FFAudioProcessingConfigDTO = field(default_factory=FFAudioProcessingConfigDTO)
    video_processing: FFVideoProcessingConfigDTO = field(default_factory=FFVideoProcessingConfigDTO)
    document_processing: FFDocumentProcessingConfigDTO = field(default_factory=FFDocumentProcessingConfigDTO)
    
    # Processing rules
    processing_rules: List[FFMediaProcessingRule] = field(default_factory=list)
    
    # Content safety and moderation
    enable_content_moderation: bool = True
    content_safety_threshold: float = 0.8
    blocked_content_types: List[str] = field(default_factory=lambda: [
        "adult", "violence", "hate", "harassment"
    ])
    
    # Performance and optimization
    enable_parallel_processing: bool = True
    max_memory_usage_mb: int = 500
    enable_gpu_acceleration: bool = False
    temp_storage_path: str = "/tmp/ff_multimodal"
    cleanup_temp_files: bool = True
    
    # Quality and validation
    enable_quality_checks: bool = True
    min_quality_threshold: float = 0.6
    enable_format_validation: bool = True
    validate_file_integrity: bool = True
    
    # Privacy and security
    enable_metadata_stripping: bool = True
    preserve_user_privacy: bool = True
    log_processing_details: bool = False  # Don't log sensitive content
    anonymize_extracted_text: bool = False
    
    # Error handling and fallbacks
    enable_fallback_processing: bool = True
    max_retry_attempts: int = 2
    retry_delay_seconds: int = 5
    graceful_degradation: bool = True
    
    # Use case specific settings
    ai_notetaker_config: Dict[str, Any] = field(default_factory=lambda: {
        "auto_transcribe_audio": True,
        "extract_action_items": True,
        "generate_meeting_summary": True,
        "identify_speakers": False,  # Privacy
        "create_searchable_index": True
    })
    
    translation_chat_config: Dict[str, Any] = field(default_factory=lambda: {
        "auto_detect_language": True,
        "translate_images_ocr": True,
        "translate_audio_transcripts": True,
        "preserve_document_formatting": True,
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    })
    
    scene_critic_config: Dict[str, Any] = field(default_factory=lambda: {
        "analyze_composition": True,
        "evaluate_lighting": True,
        "assess_color_palette": True,
        "critique_subject_matter": True,
        "provide_improvement_suggestions": True
    })
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate basic settings
        if self.max_concurrent_files <= 0:
            raise ValueError("max_concurrent_files must be positive")
        if self.processing_timeout_seconds <= 0:
            raise ValueError("processing_timeout_seconds must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        
        # Validate thresholds
        if not 0 <= self.content_safety_threshold <= 1:
            raise ValueError("content_safety_threshold must be between 0 and 1")
        if not 0 <= self.min_quality_threshold <= 1:
            raise ValueError("min_quality_threshold must be between 0 and 1")
        
        # Validate processing mode
        valid_modes = [mode.value for mode in FFProcessingMode]
        if self.default_processing_mode not in valid_modes:
            raise ValueError(f"default_processing_mode must be one of {valid_modes}")
        
        # Initialize sub-configurations
        if not isinstance(self.image_processing, FFImageProcessingConfigDTO):
            self.image_processing = FFImageProcessingConfigDTO()
        if not isinstance(self.audio_processing, FFAudioProcessingConfigDTO):
            self.audio_processing = FFAudioProcessingConfigDTO()
        if not isinstance(self.video_processing, FFVideoProcessingConfigDTO):
            self.video_processing = FFVideoProcessingConfigDTO()
        if not isinstance(self.document_processing, FFDocumentProcessingConfigDTO):
            self.document_processing = FFDocumentProcessingConfigDTO()
        
        # Initialize default processing rules if empty
        if not self.processing_rules:
            self._initialize_default_processing_rules()
    
    def _initialize_default_processing_rules(self):
        """Initialize default processing rules for different media types"""
        
        # Image processing rule
        self.processing_rules.append(FFMediaProcessingRule(
            media_type=FFMediaType.IMAGE,
            file_extensions=self.image_processing.supported_formats,
            max_file_size_mb=self.image_processing.max_file_size_mb,
            processing_mode=FFProcessingMode.ANALYZE,
            capabilities=[
                FFMultimodalCapability.OCR,
                FFMultimodalCapability.METADATA_EXTRACTION,
                FFMultimodalCapability.THUMBNAIL_GENERATION
            ],
            timeout_seconds=30
        ))
        
        # Audio processing rule
        self.processing_rules.append(FFMediaProcessingRule(
            media_type=FFMediaType.AUDIO,
            file_extensions=self.audio_processing.supported_formats,
            max_file_size_mb=self.audio_processing.max_file_size_mb,
            processing_mode=FFProcessingMode.FULL_ANALYSIS,
            capabilities=[
                FFMultimodalCapability.ASR,
                FFMultimodalCapability.SENTIMENT_ANALYSIS,
                FFMultimodalCapability.LANGUAGE_DETECTION,
                FFMultimodalCapability.METADATA_EXTRACTION
            ],
            timeout_seconds=120
        ))
        
        # Video processing rule
        self.processing_rules.append(FFMediaProcessingRule(
            media_type=FFMediaType.VIDEO,
            file_extensions=self.video_processing.supported_formats,
            max_file_size_mb=self.video_processing.max_file_size_mb,
            processing_mode=FFProcessingMode.ANALYZE,
            capabilities=[
                FFMultimodalCapability.METADATA_EXTRACTION,
                FFMultimodalCapability.THUMBNAIL_GENERATION
            ],
            timeout_seconds=180
        ))
        
        # Document processing rule
        self.processing_rules.append(FFMediaProcessingRule(
            media_type=FFMediaType.DOCUMENT,
            file_extensions=self.document_processing.supported_formats,
            max_file_size_mb=self.document_processing.max_file_size_mb,
            processing_mode=FFProcessingMode.FULL_ANALYSIS,
            capabilities=[
                FFMultimodalCapability.OCR,
                FFMultimodalCapability.LANGUAGE_DETECTION,
                FFMultimodalCapability.METADATA_EXTRACTION,
                FFMultimodalCapability.FORMAT_CONVERSION
            ],
            timeout_seconds=60
        ))
    
    def get_processing_rule(self, file_extension: str) -> Optional[FFMediaProcessingRule]:
        """Get processing rule for a file extension"""
        for rule in self.processing_rules:
            if rule.enabled and file_extension.lower() in rule.file_extensions:
                return rule
        return None
    
    def is_supported_format(self, file_extension: str) -> bool:
        """Check if a file format is supported"""
        return self.get_processing_rule(file_extension) is not None
    
    def get_media_type(self, file_extension: str) -> Optional[FFMediaType]:
        """Get media type for a file extension"""
        rule = self.get_processing_rule(file_extension)
        return rule.media_type if rule else None
    
    def get_max_file_size(self, file_extension: str) -> int:
        """Get maximum file size for a file extension"""
        rule = self.get_processing_rule(file_extension)
        return rule.max_file_size_mb if rule else 0
    
    def get_processing_capabilities(self, file_extension: str) -> List[FFMultimodalCapability]:
        """Get processing capabilities for a file extension"""
        rule = self.get_processing_rule(file_extension)
        return rule.capabilities if rule else []
    
    def add_processing_rule(self, rule: FFMediaProcessingRule) -> bool:
        """Add a new processing rule"""
        # Check for conflicts with existing rules
        for existing_rule in self.processing_rules:
            common_extensions = set(rule.file_extensions) & set(existing_rule.file_extensions)
            if common_extensions:
                # Handle conflicts by priority
                if rule.priority > existing_rule.priority:
                    # Remove conflicting extensions from existing rule
                    existing_rule.file_extensions = [
                        ext for ext in existing_rule.file_extensions 
                        if ext not in common_extensions
                    ]
                else:
                    # Remove conflicting extensions from new rule
                    rule.file_extensions = [
                        ext for ext in rule.file_extensions 
                        if ext not in common_extensions
                    ]
        
        self.processing_rules.append(rule)
        return True
    
    def remove_processing_rule(self, media_type: FFMediaType) -> bool:
        """Remove processing rule for a media type"""
        for i, rule in enumerate(self.processing_rules):
            if rule.media_type == media_type:
                del self.processing_rules[i]
                return True
        return False


@dataclass
class FFMultimodalProcessingResult:
    """Result of multimodal processing"""
    file_path: str
    media_type: FFMediaType
    processing_mode: FFProcessingMode
    success: bool
    extracted_content: Dict[str, Any]
    metadata: Dict[str, Any]
    capabilities_used: List[FFMultimodalCapability]
    processing_time_ms: float
    quality_score: float
    error: Optional[str] = None
    thumbnails: List[str] = field(default_factory=list)
    derived_files: List[str] = field(default_factory=list)


@dataclass
class FFMultimodalAnalysisContext:
    """Context for multimodal analysis"""
    session_id: str
    user_id: str
    files: List[str]
    processing_preferences: Dict[str, Any] = field(default_factory=dict)
    use_case_specific_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)