"""
Document management functionality extracted from FFStorageManager.

Handles document storage, retrieval, and document-specific operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
from ff_class_configs.ff_chat_entities_config import FFDocumentDTO
from backends import StorageBackend
from ff_utils import (
    ff_get_documents_path, ff_sanitize_filename, ff_write_json, ff_read_json
)
from ff_utils.ff_logging import get_logger
from ff_utils.ff_validation import validate_user_id, validate_filename, validate_document_content


class FFDocumentManager:
    """
    Manages document storage and retrieval.
    
    Single responsibility: Document data management and operations.
    """
    
    def __init__(self, config: FFConfigurationManagerConfigDTO, backend: StorageBackend):
        """
        Initialize document manager.
        
        Args:
            config: Storage configuration
            backend: Storage backend for data operations
        """
        self.config = config
        self.backend = backend
        self.base_path = Path(config.storage.base_path)
        self.logger = get_logger(__name__)
    
    async def store_document(self, user_id: str, session_id: str, filename: str, 
                           content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store document content and metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            filename: Original filename
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            Document ID or empty string if failed
        """
        # Validate inputs
        user_errors = validate_user_id(user_id, self.config)
        filename_errors = validate_filename(filename, self.config)
        content_errors = validate_document_content(content, self.config)
        
        if user_errors or filename_errors or content_errors:
            all_errors = user_errors + filename_errors + content_errors
            self.logger.warning(f"Invalid document storage inputs: {'; '.join(all_errors)}")
            return ""
        
        # Check document size
        if len(content) > self.config.storage.max_document_size_bytes:
            self.logger.warning(f"Document size {len(content)} exceeds limit {self.config.storage.max_document_size_bytes}")
            return ""
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in self.config.document.allowed_extensions:
            self.logger.warning(f"File extension {ext} not allowed")
            return ""
        
        # Generate safe filename
        safe_filename = ff_sanitize_filename(filename)
        doc_id = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
        
        # Create document object
        doc = FFDocumentDTO(
            document_id=doc_id,
            filename=filename,
            content=content,
            size=len(content.encode('utf-8')),
            content_type=ext,
            uploaded_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Get document paths  
        docs_base_path = ff_get_documents_path(self.base_path, user_id, session_id, self.config)
        
        # Store document content
        content_extension = self.config.runtime.document_content_extension
        content_path = docs_base_path / f"{doc_id}{content_extension}"
        content_key = str(content_path.relative_to(self.base_path))
        
        content_data = content.encode('utf-8')
        if not await self.backend.write(content_key, content_data):
            return ""
        
        # Store document metadata
        metadata_path = docs_base_path / "documents.json"  
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        # Load existing metadata
        existing_metadata = await self._read_json(metadata_key) or {}
        existing_metadata[doc_id] = doc.to_dict()
        
        # Save updated metadata
        if await self._write_json(metadata_key, existing_metadata):
            return doc_id
        return ""
    
    async def get_document(self, user_id: str, session_id: str, doc_id: str) -> Optional[FFDocumentDTO]:
        """
        Retrieve document by ID.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            doc_id: Document ID
            
        Returns:
            Document object or None
        """
        # Get metadata
        docs_base_path = ff_get_documents_path(self.base_path, user_id, session_id, self.config)
        metadata_path = docs_base_path / "documents.json"
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        metadata = await self._read_json(metadata_key)
        if not metadata or doc_id not in metadata:
            return None
        
        # Get document content
        content_extension = self.config.runtime.document_content_extension
        content_path = docs_base_path / f"{doc_id}{content_extension}"
        content_key = str(content_path.relative_to(self.base_path))
        
        content_data = await self.backend.read(content_key)
        if not content_data:
            return None
        
        # Create document object with content
        doc_data = metadata[doc_id].copy()
        doc_data['content'] = content_data.decode('utf-8')
        
        return FFDocumentDTO.from_dict(doc_data)
    
    async def list_documents(self, user_id: str, session_id: str) -> List[FFDocumentDTO]:
        """
        List all documents in session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of documents (without content)
        """
        docs_base_path = ff_get_documents_path(self.base_path, user_id, session_id, self.config)
        metadata_path = docs_base_path / "documents.json"
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        docs_metadata = await self._read_json(metadata_key) or {}
        
        documents = []
        for doc_data in docs_metadata.values():
            try:
                # Create document without content for listing
                doc_data_copy = doc_data.copy()
                doc_data_copy['content'] = ""  # Don't load content for listing
                documents.append(FFDocumentDTO.from_dict(doc_data_copy))
            except Exception as e:
                self.logger.error(f"Failed to parse document metadata: {e}", exc_info=True)
                continue
        
        return documents
    
    async def update_document_analysis(self, user_id: str, session_id: str,
                                     doc_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Update document with analysis results.
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            doc_id: Document ID
            analysis: Analysis results to store
            
        Returns:
            True if successful
        """
        docs_base_path = ff_get_documents_path(self.base_path, user_id, session_id, self.config)
        metadata_path = docs_base_path / "documents.json"
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        # Load existing metadata
        metadata = await self._read_json(metadata_key)
        if not metadata or doc_id not in metadata:
            return False
        
        # Update analysis
        if 'analysis' not in metadata[doc_id]:
            metadata[doc_id]['analysis'] = {}
        
        metadata[doc_id]['analysis'].update(analysis)
        metadata[doc_id]['analyzed_at'] = datetime.now().isoformat()
        
        # Save updated metadata
        return await self._write_json(metadata_key, metadata)
    
    async def delete_document(self, user_id: str, session_id: str, doc_id: str) -> bool:
        """
        Delete document and its metadata.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        docs_base_path = ff_get_documents_path(self.base_path, user_id, session_id, self.config)
        
        # Delete content file
        content_extension = self.config.runtime.document_content_extension
        content_path = docs_base_path / f"{doc_id}{content_extension}"
        content_key = str(content_path.relative_to(self.base_path))
        await self.backend.delete(content_key)
        
        # Remove from metadata
        metadata_path = docs_base_path / "documents.json"
        metadata_key = str(metadata_path.relative_to(self.base_path))
        
        metadata = await self._read_json(metadata_key) or {}
        if doc_id in metadata:
            del metadata[doc_id]
            return await self._write_json(metadata_key, metadata)
        
        return True
    
    async def get_document_statistics(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get document statistics for session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Statistics dictionary
        """
        documents = await self.list_documents(user_id, session_id)
        
        stats = {
            'document_count': len(documents),
            'total_size_bytes': 0,
            'file_types': {},
            'analyzed_count': 0
        }
        
        for doc in documents:
            if hasattr(doc, 'size') and doc.size:
                stats['total_size_bytes'] += doc.size
            
            if hasattr(doc, 'content_type') and doc.content_type:
                content_type = doc.content_type
                stats['file_types'][content_type] = stats['file_types'].get(content_type, 0) + 1
            
            if hasattr(doc, 'metadata') and doc.metadata and 'analysis' in doc.metadata:
                stats['analyzed_count'] += 1
        
        return stats
    
    # === Helper Methods ===
    
    async def _write_json(self, key: str, data: Dict[str, Any]) -> bool:
        """Write JSON data using backend"""
        json_path = self.base_path / key
        return await ff_write_json(json_path, data, self.config)
    
    async def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON data using backend"""
        json_path = self.base_path / key
        return await ff_read_json(json_path, self.config)