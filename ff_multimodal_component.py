"""
FF Multimodal Component - Enhanced Multimodal Processing

Provides advanced multimodal processing capabilities for the FF Chat System,
supporting image, audio, video, and document processing using FF document processing pipeline.
"""

import asyncio
import time
import json
import hashlib
import mimetypes
import base64
from typing import Dict, Any, List, Optional, Union, BinaryIO, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import tempfile
import os

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_multimodal_config import FFMultimodalConfigDTO, FFMediaType, FFProcessingCapability
from ff_protocols.ff_chat_component_protocol import FFChatComponentProtocol
from ff_protocols.ff_message_dto import FFMessageDTO
from ff_managers.ff_document_manager import FFDocumentManager
from ff_managers.ff_storage_manager import FFStorageManager
from ff_managers.ff_search_manager import FFSearchManager

logger = get_logger(__name__)


class FFProcessingStatus(Enum):
    """Status of multimodal processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FFQualityLevel(Enum):
    """Quality levels for processing"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class FFMediaMetadata:
    """Metadata for media files"""
    file_name: str
    file_size: int
    mime_type: str
    media_type: FFMediaType
    dimensions: Optional[Tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    format_info: Dict[str, Any] = field(default_factory=dict)
    technical_metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_text: Optional[str] = None
    thumbnail_path: Optional[str] = None


@dataclass
class FFProcessingResult:
    """Result of multimodal processing"""
    media_id: str
    processing_id: str
    status: FFProcessingStatus
    media_type: FFMediaType
    capabilities_applied: List[FFProcessingCapability]
    processing_time_ms: float
    results: Dict[str, Any] = field(default_factory=dict)
    extracted_content: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: FFMediaMetadata = None


@dataclass
class FFMultimodalAnalysis:
    """Complete multimodal analysis result"""
    analysis_id: str
    session_id: str
    user_id: str
    message_id: str
    media_items: List[FFProcessingResult]
    combined_analysis: Dict[str, Any] = field(default_factory=dict)
    content_summary: str = ""
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class FFMultimodalComponent(FFChatComponentProtocol):
    """
    FF Multimodal Component for advanced media processing and analysis.
    
    Provides comprehensive multimodal processing including image analysis,
    audio transcription, video processing, and document extraction.
    """
    
    def __init__(self, 
                 config: FFMultimodalConfigDTO,
                 document_manager: FFDocumentManager,
                 storage_manager: FFStorageManager,
                 search_manager: FFSearchManager):
        """
        Initialize FF Multimodal Component.
        
        Args:
            config: Multimodal processing configuration
            document_manager: FF document manager for content processing
            storage_manager: FF storage manager for data persistence
            search_manager: FF search manager for content search
        """
        self.config = config
        self.document_manager = document_manager
        self.storage_manager = storage_manager
        self.search_manager = search_manager
        self.logger = get_logger(__name__)
        
        # Processing state
        self.active_processing: Dict[str, FFProcessingResult] = {}
        self.processing_cache: Dict[str, FFProcessingResult] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
        
        # Performance metrics
        self.processing_metrics: Dict[str, Any] = {
            "total_processed": 0,
            "processing_times": {},
            "success_rates": {},
            "error_counts": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Media handlers registry
        self.media_handlers: Dict[FFMediaType, List] = {
            FFMediaType.IMAGE: [],
            FFMediaType.AUDIO: [],
            FFMediaType.VIDEO: [],
            FFMediaType.DOCUMENT: []
        }
        
        # Initialize built-in handlers
        self._initialize_media_handlers()
        
        # Temporary storage for processing
        self.temp_dir = tempfile.mkdtemp(prefix="ff_multimodal_")
    
    async def process_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message: FFMessageDTO, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process message for multimodal content.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            message: Message potentially containing multimodal content
            context: Additional context information
            
        Returns:
            Dict containing multimodal processing results
        """
        start_time = time.time()
        
        try:
            # Extract media items from message
            media_items = await self._extract_media_from_message(message, context)
            
            if not media_items:
                return {
                    "success": True,
                    "has_multimodal_content": False,
                    "message": "No multimodal content detected",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            # Generate analysis ID
            analysis_id = self._generate_analysis_id(session_id, user_id, message)
            
            # Process each media item
            processing_results = []
            for media_item in media_items:
                result = await self._process_media_item(media_item, session_id, user_id, context)
                processing_results.append(result)
            
            # Perform combined analysis
            combined_analysis = await self._perform_combined_analysis(processing_results, context)
            
            # Create multimodal analysis result
            analysis = FFMultimodalAnalysis(
                analysis_id=analysis_id,
                session_id=session_id,
                user_id=user_id,
                message_id=getattr(message, 'message_id', f"msg_{int(time.time())}"),
                media_items=processing_results,
                combined_analysis=combined_analysis,
                content_summary=combined_analysis.get("summary", ""),
                confidence_score=combined_analysis.get("confidence", 0.0),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Store analysis results
            await self._store_analysis_results(analysis)
            
            # Update metrics
            self._update_processing_metrics(analysis)
            
            self.logger.info(f"Multimodal analysis completed: {analysis_id} with {len(processing_results)} media items")
            
            return self._format_analysis_response(analysis)
            
        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "has_multimodal_content": False,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _extract_media_from_message(self, message: FFMessageDTO, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract media items from message"""
        media_items = []
        
        try:
            # Check message content for media references
            content = getattr(message, 'content', str(message))
            
            # Look for file attachments
            attachments = getattr(message, 'attachments', [])
            for attachment in attachments:
                media_item = await self._process_attachment(attachment)
                if media_item:
                    media_items.append(media_item)
            
            # Look for embedded media in content
            embedded_media = self._extract_embedded_media(content)
            media_items.extend(embedded_media)
            
            # Check context for media references
            if context:
                context_media = context.get('media_files', [])
                for media_ref in context_media:
                    media_item = await self._process_media_reference(media_ref)
                    if media_item:
                        media_items.append(media_item)
            
            return media_items
            
        except Exception as e:
            self.logger.error(f"Media extraction failed: {e}")
            return []
    
    async def _process_attachment(self, attachment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process message attachment"""
        try:
            file_path = attachment.get('file_path')
            file_url = attachment.get('url')
            file_data = attachment.get('data')
            
            if not any([file_path, file_url, file_data]):
                return None
            
            # Determine media type
            mime_type = attachment.get('mime_type') or mimetypes.guess_type(file_path or file_url or '')[0]
            media_type = self._determine_media_type(mime_type)
            
            if media_type == FFMediaType.UNKNOWN:
                return None
            
            # Create media item
            media_item = {
                'media_id': self._generate_media_id(attachment),
                'media_type': media_type,
                'mime_type': mime_type,
                'source_type': 'attachment',
                'file_path': file_path,
                'file_url': file_url,
                'file_data': file_data,
                'file_name': attachment.get('file_name', 'unnamed'),
                'file_size': attachment.get('file_size', 0)
            }
            
            return media_item
            
        except Exception as e:
            self.logger.error(f"Attachment processing failed: {e}")
            return None
    
    def _extract_embedded_media(self, content: str) -> List[Dict[str, Any]]:
        """Extract embedded media from content"""
        media_items = []
        
        try:
            # Look for data URLs (base64 encoded media)
            import re
            
            data_url_pattern = r'data:([^;]+);base64,([A-Za-z0-9+/=]+)'
            matches = re.findall(data_url_pattern, content)
            
            for mime_type, data in matches:
                media_type = self._determine_media_type(mime_type)
                if media_type != FFMediaType.UNKNOWN:
                    media_item = {
                        'media_id': self._generate_media_id({'mime_type': mime_type, 'data': data[:50]}),
                        'media_type': media_type,
                        'mime_type': mime_type,
                        'source_type': 'embedded',
                        'file_data': data,
                        'file_name': f'embedded_{media_type.value}',
                        'file_size': len(data) * 3 // 4  # Approximate size from base64
                    }
                    media_items.append(media_item)
            
            # Look for URLs pointing to media files
            url_pattern = r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|svg|mp4|avi|mov|mp3|wav|pdf|doc|docx)'
            url_matches = re.findall(url_pattern, content, re.IGNORECASE)
            
            for url in url_matches:
                mime_type = mimetypes.guess_type(url)[0]
                media_type = self._determine_media_type(mime_type)
                if media_type != FFMediaType.UNKNOWN:
                    media_item = {
                        'media_id': self._generate_media_id({'url': url}),
                        'media_type': media_type,
                        'mime_type': mime_type,
                        'source_type': 'url',
                        'file_url': url,
                        'file_name': os.path.basename(url),
                        'file_size': 0  # Unknown
                    }
                    media_items.append(media_item)
            
            return media_items
            
        except Exception as e:
            self.logger.error(f"Embedded media extraction failed: {e}")
            return []
    
    async def _process_media_reference(self, media_ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process media reference from context"""
        try:
            # Similar to attachment processing but for context references
            return await self._process_attachment(media_ref)
        except Exception as e:
            self.logger.error(f"Media reference processing failed: {e}")
            return None
    
    def _determine_media_type(self, mime_type: Optional[str]) -> FFMediaType:
        """Determine media type from MIME type"""
        if not mime_type:
            return FFMediaType.UNKNOWN
        
        mime_type = mime_type.lower()
        
        if mime_type.startswith('image/'):
            return FFMediaType.IMAGE
        elif mime_type.startswith('audio/'):
            return FFMediaType.AUDIO
        elif mime_type.startswith('video/'):
            return FFMediaType.VIDEO
        elif mime_type in ['application/pdf', 'text/plain', 'application/msword', 
                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return FFMediaType.DOCUMENT
        else:
            return FFMediaType.UNKNOWN
    
    def _generate_media_id(self, media_data: Dict[str, Any]) -> str:
        """Generate unique media ID"""
        hash_input = json.dumps(media_data, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _generate_analysis_id(self, session_id: str, user_id: str, message: FFMessageDTO) -> str:
        """Generate unique analysis ID"""
        timestamp = int(time.time())
        hash_input = f"{session_id}_{user_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _process_media_item(self, 
                                media_item: Dict[str, Any], 
                                session_id: str, 
                                user_id: str, 
                                context: Optional[Dict[str, Any]]) -> FFProcessingResult:
        """Process individual media item"""
        start_time = time.time()
        processing_id = f"proc_{int(time.time())}_{media_item['media_id']}"
        
        # Check cache first
        cache_key = self._generate_cache_key(media_item)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.processing_metrics["cache_hits"] += 1
            return cached_result
        
        self.processing_metrics["cache_misses"] += 1
        
        # Create processing result
        result = FFProcessingResult(
            media_id=media_item['media_id'],
            processing_id=processing_id,
            status=FFProcessingStatus.PROCESSING,
            media_type=FFMediaType(media_item['media_type']),
            capabilities_applied=[],
            processing_time_ms=0.0
        )
        
        # Register active processing
        self.active_processing[processing_id] = result
        
        try:
            # Load media file
            media_file_path = await self._load_media_file(media_item)
            
            # Extract metadata
            metadata = await self._extract_media_metadata(media_file_path, media_item)
            result.metadata = metadata
            
            # Apply processing capabilities based on media type and configuration
            if media_item['media_type'] == FFMediaType.IMAGE:
                await self._process_image(result, media_file_path, context)
            elif media_item['media_type'] == FFMediaType.AUDIO:
                await self._process_audio(result, media_file_path, context)
            elif media_item['media_type'] == FFMediaType.VIDEO:
                await self._process_video(result, media_file_path, context)
            elif media_item['media_type'] == FFMediaType.DOCUMENT:
                await self._process_document(result, media_file_path, context)
            
            # Calculate quality metrics
            result.quality_metrics = await self._calculate_quality_metrics(result, media_file_path)
            
            result.status = FFProcessingStatus.COMPLETED
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self._cache_result(cache_key, result)
            
        except Exception as e:
            result.status = FFProcessingStatus.FAILED
            result.error_message = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Media processing failed for {media_item['media_id']}: {e}")
        
        finally:
            # Clean up temporary files
            if 'temp_file_path' in locals():
                try:
                    os.remove(media_file_path)
                except Exception:
                    pass
            
            # Remove from active processing
            if processing_id in self.active_processing:
                del self.active_processing[processing_id]
        
        return result
    
    async def _load_media_file(self, media_item: Dict[str, Any]) -> str:
        """Load media file to temporary location"""
        try:
            temp_file_path = os.path.join(self.temp_dir, f"{media_item['media_id']}.tmp")
            
            if media_item.get('file_path'):
                # Copy from local file
                import shutil
                shutil.copy2(media_item['file_path'], temp_file_path)
            
            elif media_item.get('file_url'):
                # Download from URL
                await self._download_file(media_item['file_url'], temp_file_path)
            
            elif media_item.get('file_data'):
                # Decode base64 data
                if isinstance(media_item['file_data'], str):
                    file_data = base64.b64decode(media_item['file_data'])
                else:
                    file_data = media_item['file_data']
                
                with open(temp_file_path, 'wb') as f:
                    f.write(file_data)
            
            else:
                raise ValueError("No valid media source found")
            
            return temp_file_path
            
        except Exception as e:
            raise Exception(f"Failed to load media file: {e}")
    
    async def _download_file(self, url: str, output_path: str) -> None:
        """Download file from URL"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                    else:
                        raise Exception(f"HTTP {response.status} downloading {url}")
        
        except ImportError:
            # Fallback to requests if aiohttp not available
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    async def _extract_media_metadata(self, file_path: str, media_item: Dict[str, Any]) -> FFMediaMetadata:
        """Extract metadata from media file"""
        try:
            file_stat = os.stat(file_path)
            
            metadata = FFMediaMetadata(
                file_name=media_item.get('file_name', os.path.basename(file_path)),
                file_size=file_stat.st_size,
                mime_type=media_item['mime_type'],
                media_type=FFMediaType(media_item['media_type'])
            )
            
            # Extract type-specific metadata
            if metadata.media_type == FFMediaType.IMAGE:
                metadata = await self._extract_image_metadata(file_path, metadata)
            elif metadata.media_type == FFMediaType.AUDIO:
                metadata = await self._extract_audio_metadata(file_path, metadata)
            elif metadata.media_type == FFMediaType.VIDEO:
                metadata = await self._extract_video_metadata(file_path, metadata)
            elif metadata.media_type == FFMediaType.DOCUMENT:
                metadata = await self._extract_document_metadata(file_path, metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            # Return basic metadata
            return FFMediaMetadata(
                file_name=media_item.get('file_name', 'unknown'),
                file_size=0,
                mime_type=media_item['mime_type'],
                media_type=FFMediaType(media_item['media_type'])
            )
    
    async def _extract_image_metadata(self, file_path: str, metadata: FFMediaMetadata) -> FFMediaMetadata:
        """Extract image-specific metadata"""
        try:
            from PIL import Image
            
            with Image.open(file_path) as img:
                metadata.dimensions = img.size
                metadata.format_info = {
                    'format': img.format,
                    'mode': img.mode,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata.technical_metadata = {'exif': dict(img._getexif())}
        
        except ImportError:
            # Fallback without PIL
            metadata.format_info = {'format': 'unknown'}
        except Exception as e:
            self.logger.warning(f"Image metadata extraction failed: {e}")
        
        return metadata
    
    async def _extract_audio_metadata(self, file_path: str, metadata: FFMediaMetadata) -> FFMediaMetadata:
        """Extract audio-specific metadata"""
        try:
            # Would integrate with audio processing library like librosa or ffmpeg
            # For now, provide basic metadata
            metadata.format_info = {'format': 'audio'}
            metadata.duration_seconds = 0.0  # Would extract actual duration
            
        except Exception as e:
            self.logger.warning(f"Audio metadata extraction failed: {e}")
        
        return metadata
    
    async def _extract_video_metadata(self, file_path: str, metadata: FFMediaMetadata) -> FFMediaMetadata:
        """Extract video-specific metadata"""
        try:
            # Would integrate with video processing library like ffmpeg-python
            # For now, provide basic metadata
            metadata.format_info = {'format': 'video'}
            metadata.duration_seconds = 0.0  # Would extract actual duration
            metadata.dimensions = (0, 0)  # Would extract actual dimensions
            
        except Exception as e:
            self.logger.warning(f"Video metadata extraction failed: {e}")
        
        return metadata
    
    async def _extract_document_metadata(self, file_path: str, metadata: FFMediaMetadata) -> FFMediaMetadata:
        """Extract document-specific metadata"""
        try:
            # Basic text extraction for documents
            if metadata.mime_type == 'text/plain':
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata.extracted_text = f.read()
            elif metadata.mime_type == 'application/pdf':
                # Would integrate with PDF processing library
                metadata.extracted_text = "PDF text extraction not implemented"
            
            metadata.format_info = {'format': 'document', 'pages': 1}
            
        except Exception as e:
            self.logger.warning(f"Document metadata extraction failed: {e}")
        
        return metadata
    
    async def _process_image(self, result: FFProcessingResult, file_path: str, context: Optional[Dict[str, Any]]) -> None:
        """Process image file"""
        try:
            processing_results = {}
            
            # Apply configured image processing capabilities
            if FFProcessingCapability.IMAGE_ANALYSIS in self.config.image_processing.capabilities:
                analysis = await self._analyze_image(file_path)
                processing_results['analysis'] = analysis
                result.capabilities_applied.append(FFProcessingCapability.IMAGE_ANALYSIS)
            
            if FFProcessingCapability.OCR in self.config.image_processing.capabilities:
                ocr_text = await self._perform_ocr(file_path)
                processing_results['ocr'] = ocr_text
                result.capabilities_applied.append(FFProcessingCapability.OCR)
            
            if FFProcessingCapability.OBJECT_DETECTION in self.config.image_processing.capabilities:
                objects = await self._detect_objects(file_path)
                processing_results['objects'] = objects
                result.capabilities_applied.append(FFProcessingCapability.OBJECT_DETECTION)
            
            if FFProcessingCapability.FACE_DETECTION in self.config.image_processing.capabilities:
                faces = await self._detect_faces(file_path)
                processing_results['faces'] = faces
                result.capabilities_applied.append(FFProcessingCapability.FACE_DETECTION)
            
            # Generate thumbnail
            thumbnail_path = await self._generate_thumbnail(file_path, 'image')
            if thumbnail_path:
                result.metadata.thumbnail_path = thumbnail_path
            
            result.results = processing_results
            
            # Extract content for search indexing
            content_for_search = []
            if 'ocr' in processing_results:
                content_for_search.append(processing_results['ocr'])
            if 'analysis' in processing_results:
                content_for_search.append(processing_results['analysis'].get('description', ''))
            
            result.extracted_content = {
                'text': ' '.join(content_for_search),
                'type': 'image_analysis'
            }
            
        except Exception as e:
            raise Exception(f"Image processing failed: {e}")
    
    async def _analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze image content"""
        try:
            # This would integrate with computer vision services or models
            # For now, return mock analysis
            analysis = {
                'description': 'Image analysis not fully implemented',
                'categories': ['general'],
                'confidence': 0.5,
                'colors': ['unknown'],
                'scene_type': 'unknown'
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {'error': str(e)}
    
    async def _perform_ocr(self, file_path: str) -> str:
        """Perform OCR on image"""
        try:
            # This would integrate with OCR service like Tesseract
            # For now, return placeholder
            return "OCR text extraction not fully implemented"
            
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return f"OCR error: {str(e)}"
    
    async def _detect_objects(self, file_path: str) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        try:
            # This would integrate with object detection models
            # For now, return placeholder
            return [{'object': 'unknown', 'confidence': 0.5, 'bbox': [0, 0, 100, 100]}]
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return []
    
    async def _detect_faces(self, file_path: str) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            # This would integrate with face detection models
            # For now, return placeholder
            return []
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    async def _process_audio(self, result: FFProcessingResult, file_path: str, context: Optional[Dict[str, Any]]) -> None:
        """Process audio file"""
        try:
            processing_results = {}
            
            # Apply configured audio processing capabilities
            if FFProcessingCapability.SPEECH_TO_TEXT in self.config.audio_processing.capabilities:
                transcript = await self._transcribe_audio(file_path)
                processing_results['transcript'] = transcript
                result.capabilities_applied.append(FFProcessingCapability.SPEECH_TO_TEXT)
            
            if FFProcessingCapability.AUDIO_ANALYSIS in self.config.audio_processing.capabilities:
                analysis = await self._analyze_audio(file_path)
                processing_results['analysis'] = analysis
                result.capabilities_applied.append(FFProcessingCapability.AUDIO_ANALYSIS)
            
            # Generate waveform thumbnail
            thumbnail_path = await self._generate_thumbnail(file_path, 'audio')
            if thumbnail_path:
                result.metadata.thumbnail_path = thumbnail_path
            
            result.results = processing_results
            
            # Extract content for search indexing
            result.extracted_content = {
                'text': processing_results.get('transcript', ''),
                'type': 'audio_transcript'
            }
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {e}")
    
    async def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio to text"""
        try:
            # This would integrate with speech recognition service
            # For now, return placeholder
            return "Audio transcription not fully implemented"
            
        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            return f"Transcription error: {str(e)}"
    
    async def _analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio content"""
        try:
            # This would integrate with audio analysis libraries
            analysis = {
                'duration': 0.0,
                'sample_rate': 44100,
                'channels': 2,
                'format': 'unknown',
                'quality': 'unknown'
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {'error': str(e)}
    
    async def _process_video(self, result: FFProcessingResult, file_path: str, context: Optional[Dict[str, Any]]) -> None:
        """Process video file"""
        try:
            processing_results = {}
            
            # Apply configured video processing capabilities
            if FFProcessingCapability.VIDEO_ANALYSIS in self.config.video_processing.capabilities:
                analysis = await self._analyze_video(file_path)
                processing_results['analysis'] = analysis
                result.capabilities_applied.append(FFProcessingCapability.VIDEO_ANALYSIS)
            
            if FFProcessingCapability.FRAME_EXTRACTION in self.config.video_processing.capabilities:
                frames = await self._extract_frames(file_path)
                processing_results['frames'] = frames
                result.capabilities_applied.append(FFProcessingCapability.FRAME_EXTRACTION)
            
            if FFProcessingCapability.SPEECH_TO_TEXT in self.config.video_processing.capabilities:
                transcript = await self._extract_video_audio_transcript(file_path)
                processing_results['transcript'] = transcript
                result.capabilities_applied.append(FFProcessingCapability.SPEECH_TO_TEXT)
            
            # Generate video thumbnail
            thumbnail_path = await self._generate_thumbnail(file_path, 'video')
            if thumbnail_path:
                result.metadata.thumbnail_path = thumbnail_path
            
            result.results = processing_results
            
            # Extract content for search indexing
            content_for_search = []
            if 'transcript' in processing_results:
                content_for_search.append(processing_results['transcript'])
            if 'analysis' in processing_results:
                content_for_search.append(processing_results['analysis'].get('description', ''))
            
            result.extracted_content = {
                'text': ' '.join(content_for_search),
                'type': 'video_analysis'
            }
            
        except Exception as e:
            raise Exception(f"Video processing failed: {e}")
    
    async def _analyze_video(self, file_path: str) -> Dict[str, Any]:
        """Analyze video content"""
        try:
            # This would integrate with video analysis tools
            analysis = {
                'duration': 0.0,
                'resolution': [0, 0],
                'fps': 30,
                'format': 'unknown',
                'scenes': [],
                'quality': 'unknown'
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_frames(self, file_path: str) -> List[str]:
        """Extract key frames from video"""
        try:
            # This would use ffmpeg or similar to extract frames
            # For now, return placeholder
            return []
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return []
    
    async def _extract_video_audio_transcript(self, file_path: str) -> str:
        """Extract and transcribe audio from video"""
        try:
            # This would extract audio track and transcribe it
            return "Video audio transcription not fully implemented"
            
        except Exception as e:
            self.logger.error(f"Video audio transcription failed: {e}")
            return f"Transcription error: {str(e)}"
    
    async def _process_document(self, result: FFProcessingResult, file_path: str, context: Optional[Dict[str, Any]]) -> None:
        """Process document file"""
        try:
            processing_results = {}
            
            # Apply configured document processing capabilities
            if FFProcessingCapability.TEXT_EXTRACTION in self.config.document_processing.capabilities:
                text = await self._extract_text_from_document(file_path, result.metadata.mime_type)
                processing_results['text'] = text
                result.capabilities_applied.append(FFProcessingCapability.TEXT_EXTRACTION)
            
            if FFProcessingCapability.DOCUMENT_ANALYSIS in self.config.document_processing.capabilities:
                analysis = await self._analyze_document(file_path, processing_results.get('text', ''))
                processing_results['analysis'] = analysis
                result.capabilities_applied.append(FFProcessingCapability.DOCUMENT_ANALYSIS)
            
            # Generate document thumbnail (first page as image)
            thumbnail_path = await self._generate_thumbnail(file_path, 'document')
            if thumbnail_path:
                result.metadata.thumbnail_path = thumbnail_path
            
            result.results = processing_results
            
            # Extract content for search indexing
            result.extracted_content = {
                'text': processing_results.get('text', ''),
                'type': 'document_text'
            }
            
        except Exception as e:
            raise Exception(f"Document processing failed: {e}")
    
    async def _extract_text_from_document(self, file_path: str, mime_type: str) -> str:
        """Extract text from document"""
        try:
            if mime_type == 'text/plain':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif mime_type == 'application/pdf':
                # Would integrate with PDF processing library like PyPDF2 or pdfplumber
                return "PDF text extraction not fully implemented"
            
            elif 'word' in mime_type:
                # Would integrate with python-docx or similar
                return "Word document text extraction not fully implemented"
            
            else:
                return "Unsupported document type for text extraction"
                
        except Exception as e:
            self.logger.error(f"Document text extraction failed: {e}")
            return f"Text extraction error: {str(e)}"
    
    async def _analyze_document(self, file_path: str, text: str) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            analysis = {
                'word_count': len(text.split()) if text else 0,
                'character_count': len(text) if text else 0,
                'language': 'unknown',
                'structure': 'unknown',
                'topics': [],
                'summary': text[:200] + '...' if len(text) > 200 else text
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return {'error': str(e)}
    
    async def _generate_thumbnail(self, file_path: str, media_type: str) -> Optional[str]:
        """Generate thumbnail for media file"""
        try:
            thumbnail_dir = os.path.join(self.temp_dir, 'thumbnails')
            os.makedirs(thumbnail_dir, exist_ok=True)
            
            thumbnail_path = os.path.join(thumbnail_dir, f"{os.path.basename(file_path)}_thumb.jpg")
            
            if media_type == 'image':
                # Would generate image thumbnail
                return None  # Placeholder
            elif media_type == 'video':
                # Would extract frame and create thumbnail
                return None  # Placeholder
            elif media_type == 'document':
                # Would render first page as image
                return None  # Placeholder
            elif media_type == 'audio':
                # Would generate waveform visualization
                return None  # Placeholder
            
            return None
            
        except Exception as e:
            self.logger.error(f"Thumbnail generation failed: {e}")
            return None
    
    async def _calculate_quality_metrics(self, result: FFProcessingResult, file_path: str) -> Dict[str, float]:
        """Calculate quality metrics for processed media"""
        try:
            metrics = {}
            
            # File-based metrics
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            metrics['file_size_mb'] = file_size_mb
            
            # Processing success rate
            success_count = len([cap for cap in result.capabilities_applied])
            total_capabilities = len(self._get_applicable_capabilities(result.media_type))
            metrics['processing_coverage'] = success_count / max(1, total_capabilities)
            
            # Confidence score based on results
            if result.results:
                confidence_scores = []
                for capability_result in result.results.values():
                    if isinstance(capability_result, dict) and 'confidence' in capability_result:
                        confidence_scores.append(capability_result['confidence'])
                
                if confidence_scores:
                    metrics['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
                else:
                    metrics['average_confidence'] = 0.5
            else:
                metrics['average_confidence'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {}
    
    def _get_applicable_capabilities(self, media_type: FFMediaType) -> List[FFProcessingCapability]:
        """Get applicable capabilities for media type"""
        if media_type == FFMediaType.IMAGE:
            return self.config.image_processing.capabilities
        elif media_type == FFMediaType.AUDIO:
            return self.config.audio_processing.capabilities
        elif media_type == FFMediaType.VIDEO:
            return self.config.video_processing.capabilities
        elif media_type == FFMediaType.DOCUMENT:
            return self.config.document_processing.capabilities
        else:
            return []
    
    async def _perform_combined_analysis(self, processing_results: List[FFProcessingResult], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform combined analysis of all processed media"""
        try:
            combined_analysis = {
                'media_count': len(processing_results),
                'media_types': list(set(result.media_type.value for result in processing_results)),
                'total_processing_time_ms': sum(result.processing_time_ms for result in processing_results),
                'success_rate': len([r for r in processing_results if r.status == FFProcessingStatus.COMPLETED]) / max(1, len(processing_results)),
                'capabilities_used': list(set(cap.value for result in processing_results for cap in result.capabilities_applied)),
                'extracted_text_length': sum(len(result.extracted_content.get('text', '')) for result in processing_results),
                'quality_score': 0.0,
                'summary': '',
                'confidence': 0.0
            }
            
            # Calculate overall quality score
            quality_scores = []
            for result in processing_results:
                if result.quality_metrics:
                    avg_quality = sum(result.quality_metrics.values()) / len(result.quality_metrics)
                    quality_scores.append(avg_quality)
            
            if quality_scores:
                combined_analysis['quality_score'] = sum(quality_scores) / len(quality_scores)
            
            # Generate combined summary
            text_contents = []
            for result in processing_results:
                if result.extracted_content.get('text'):
                    text_contents.append(result.extracted_content['text'])
            
            if text_contents:
                combined_text = ' '.join(text_contents)
                combined_analysis['summary'] = combined_text[:300] + '...' if len(combined_text) > 300 else combined_text
                combined_analysis['confidence'] = min(1.0, len(combined_text) / 1000)  # Simple confidence based on content length
            
            # Add context-based analysis
            if context:
                combined_analysis['context_integration'] = await self._integrate_context_analysis(processing_results, context)
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Combined analysis failed: {e}")
            return {'error': str(e)}
    
    async def _integrate_context_analysis(self, processing_results: List[FFProcessingResult], context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate context information with processing results"""
        try:
            integration = {
                'context_relevance': 0.0,
                'related_content': [],
                'suggestions': []
            }
            
            # Use FF search manager to find related content
            if self.search_manager:
                for result in processing_results:
                    if result.extracted_content.get('text'):
                        search_results = await self.search_manager.search_messages(
                            user_id=context.get('user_id', 'unknown'),
                            query=result.extracted_content['text'][:100],
                            limit=5
                        )
                        
                        integration['related_content'].extend([
                            {
                                'content': res.get('content', '')[:100],
                                'similarity': res.get('similarity_score', 0.0)
                            }
                            for res in search_results
                        ])
            
            return integration
            
        except Exception as e:
            self.logger.error(f"Context integration failed: {e}")
            return {'error': str(e)}
    
    async def _store_analysis_results(self, analysis: FFMultimodalAnalysis) -> None:
        """Store analysis results for future reference"""
        try:
            # Store analysis using document manager
            analysis_doc = {
                'analysis_id': analysis.analysis_id,
                'session_id': analysis.session_id,
                'user_id': analysis.user_id,
                'message_id': analysis.message_id,
                'timestamp': analysis.timestamp.isoformat(),
                'media_count': len(analysis.media_items),
                'processing_time_ms': analysis.processing_time_ms,
                'content_summary': analysis.content_summary,
                'confidence_score': analysis.confidence_score,
                'media_items': [
                    {
                        'media_id': item.media_id,
                        'media_type': item.media_type.value,
                        'status': item.status.value,
                        'capabilities': [cap.value for cap in item.capabilities_applied],
                        'extracted_text': item.extracted_content.get('text', ''),
                        'processing_time_ms': item.processing_time_ms
                    }
                    for item in analysis.media_items
                ],
                'combined_analysis': analysis.combined_analysis
            }
            
            await self.document_manager.store_document(
                document_id=f"multimodal_analysis_{analysis.analysis_id}",
                content=json.dumps(analysis_doc),
                metadata={
                    'type': 'multimodal_analysis',
                    'session_id': analysis.session_id,
                    'user_id': analysis.user_id,
                    'media_count': len(analysis.media_items),
                    'timestamp': analysis.timestamp.isoformat()
                }
            )
            
            self.logger.debug(f"Stored multimodal analysis {analysis.analysis_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis results: {e}")
    
    def _update_processing_metrics(self, analysis: FFMultimodalAnalysis) -> None:
        """Update processing metrics"""
        self.processing_metrics['total_processed'] += len(analysis.media_items)
        
        for item in analysis.media_items:
            media_type = item.media_type.value
            
            # Update processing times
            if media_type not in self.processing_metrics['processing_times']:
                self.processing_metrics['processing_times'][media_type] = []
            self.processing_metrics['processing_times'][media_type].append(item.processing_time_ms)
            
            # Update success rates
            if media_type not in self.processing_metrics['success_rates']:
                self.processing_metrics['success_rates'][media_type] = {'success': 0, 'total': 0}
            
            self.processing_metrics['success_rates'][media_type]['total'] += 1
            if item.status == FFProcessingStatus.COMPLETED:
                self.processing_metrics['success_rates'][media_type]['success'] += 1
            
            # Update error counts
            if item.status == FFProcessingStatus.FAILED:
                if media_type not in self.processing_metrics['error_counts']:
                    self.processing_metrics['error_counts'][media_type] = 0
                self.processing_metrics['error_counts'][media_type] += 1
    
    def _generate_cache_key(self, media_item: Dict[str, Any]) -> str:
        """Generate cache key for media item"""
        # Include relevant properties that affect processing
        cache_data = {
            'media_id': media_item['media_id'],
            'media_type': media_item['media_type'],
            'file_size': media_item.get('file_size', 0),
            'capabilities': sorted([cap.value for cap in self._get_applicable_capabilities(FFMediaType(media_item['media_type']))])
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[FFProcessingResult]:
        """Get cached processing result"""
        if cache_key not in self.processing_cache:
            return None
        
        cached_result, cached_time = self.processing_cache[cache_key]
        
        # Check if cache is still valid
        if time.time() - cached_time > self.cache_ttl_seconds:
            del self.processing_cache[cache_key]
            return None
        
        return cached_result
    
    def _cache_result(self, cache_key: str, result: FFProcessingResult) -> None:
        """Cache processing result"""
        self.processing_cache[cache_key] = (result, time.time())
        
        # Clean old cache entries if cache is getting too large
        if len(self.processing_cache) > 500:
            # Remove oldest 25% of entries
            oldest_entries = sorted(self.processing_cache.items(), key=lambda x: x[1][1])[:125]
            for cache_key, _ in oldest_entries:
                del self.processing_cache[cache_key]
    
    def _format_analysis_response(self, analysis: FFMultimodalAnalysis) -> Dict[str, Any]:
        """Format analysis response for return"""
        return {
            'success': True,
            'has_multimodal_content': True,
            'analysis_id': analysis.analysis_id,
            'media_count': len(analysis.media_items),
            'processing_time_ms': analysis.processing_time_ms,
            'content_summary': analysis.content_summary,
            'confidence_score': analysis.confidence_score,
            'media_items': [
                {
                    'media_id': item.media_id,
                    'media_type': item.media_type.value,
                    'status': item.status.value,
                    'capabilities_applied': [cap.value for cap in item.capabilities_applied],
                    'processing_time_ms': item.processing_time_ms,
                    'extracted_content': item.extracted_content,
                    'error_message': item.error_message
                }
                for item in analysis.media_items
            ],
            'combined_analysis': analysis.combined_analysis,
            'timestamp': analysis.timestamp.isoformat()
        }
    
    def _initialize_media_handlers(self) -> None:
        """Initialize built-in media handlers"""
        # This would register various media processing handlers
        # For now, we're using the built-in processing methods
        pass
    
    # Public API methods
    
    async def process_media_file(self, 
                               file_path: str, 
                               session_id: str, 
                               user_id: str,
                               processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single media file directly"""
        try:
            # Create media item from file path
            mime_type = mimetypes.guess_type(file_path)[0]
            media_type = self._determine_media_type(mime_type)
            
            if media_type == FFMediaType.UNKNOWN:
                return {
                    'success': False,
                    'error': f'Unsupported media type: {mime_type}'
                }
            
            media_item = {
                'media_id': self._generate_media_id({'file_path': file_path}),
                'media_type': media_type,
                'mime_type': mime_type,
                'source_type': 'direct',
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
            
            # Process the media item
            result = await self._process_media_item(media_item, session_id, user_id, processing_options)
            
            return {
                'success': result.status == FFProcessingStatus.COMPLETED,
                'processing_result': {
                    'media_id': result.media_id,
                    'media_type': result.media_type.value,
                    'status': result.status.value,
                    'capabilities_applied': [cap.value for cap in result.capabilities_applied],
                    'processing_time_ms': result.processing_time_ms,
                    'results': result.results,
                    'extracted_content': result.extracted_content,
                    'error_message': result.error_message
                }
            }
            
        except Exception as e:
            self.logger.error(f"Direct media processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_processing_status(self, processing_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active processing"""
        if processing_id in self.active_processing:
            result = self.active_processing[processing_id]
            return {
                'processing_id': processing_id,
                'media_id': result.media_id,
                'status': result.status.value,
                'media_type': result.media_type.value,
                'progress': 0.5 if result.status == FFProcessingStatus.PROCESSING else 1.0
            }
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        # Calculate average processing times
        avg_times = {}
        for media_type, times in self.processing_metrics['processing_times'].items():
            if times:
                avg_times[media_type] = sum(times) / len(times)
        
        # Calculate success rates
        success_rates = {}
        for media_type, counts in self.processing_metrics['success_rates'].items():
            if counts['total'] > 0:
                success_rates[media_type] = counts['success'] / counts['total']
        
        return {
            'processing_metrics': {
                'total_processed': self.processing_metrics['total_processed'],
                'average_processing_times_ms': avg_times,
                'success_rates': success_rates,
                'error_counts': self.processing_metrics['error_counts'],
                'cache_hit_rate': (
                    self.processing_metrics['cache_hits'] / 
                    max(1, self.processing_metrics['cache_hits'] + self.processing_metrics['cache_misses'])
                )
            },
            'system_stats': {
                'active_processing': len(self.active_processing),
                'cache_size': len(self.processing_cache),
                'temp_dir_size_mb': self._get_temp_dir_size()
            }
        }
    
    def _get_temp_dir_size(self) -> float:
        """Get temporary directory size in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.temp_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def cleanup_temp_files(self) -> int:
        """Clean up temporary files"""
        try:
            import shutil
            removed_count = 0
            
            for item in os.listdir(self.temp_dir):
                item_path = os.path.join(self.temp_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        removed_count += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        removed_count += 1
                except Exception:
                    pass
            
            self.logger.info(f"Cleaned up {removed_count} temporary files")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Temp file cleanup failed: {e}")
            return 0
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            'component_name': 'FF Multimodal Component',
            'version': '1.0.0',
            'config': {
                'enable_multimodal_processing': self.config.enable_multimodal_processing,
                'supported_media_types': [mt.value for mt in FFMediaType if mt != FFMediaType.UNKNOWN],
                'image_capabilities': [cap.value for cap in self.config.image_processing.capabilities],
                'audio_capabilities': [cap.value for cap in self.config.audio_processing.capabilities],
                'video_capabilities': [cap.value for cap in self.config.video_processing.capabilities],
                'document_capabilities': [cap.value for cap in self.config.document_processing.capabilities]
            },
            'status': 'active',
            'metrics': self.get_metrics()
        }