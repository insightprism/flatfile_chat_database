"""
Compression utilities for the flatfile chat database.

Provides transparent compression/decompression for stored data to save disk space.
"""

import gzip
import zlib
import asyncio
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
from enum import Enum

from flatfile_chat_database.config import StorageConfig


class CompressionType(Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"


class CompressionConfig:
    """Configuration for compression settings"""
    
    def __init__(self,
                 enabled: bool = True,
                 type: CompressionType = CompressionType.GZIP,
                 level: int = 6,
                 min_size_bytes: int = 1024,
                 compress_messages: bool = True,
                 compress_documents: bool = True,
                 compress_context: bool = True):
        """
        Initialize compression configuration.
        
        Args:
            enabled: Whether compression is enabled
            type: Compression algorithm to use
            level: Compression level (1-9, where 9 is max compression)
            min_size_bytes: Minimum size before compression is applied
            compress_messages: Whether to compress message files
            compress_documents: Whether to compress documents
            compress_context: Whether to compress context data
        """
        self.enabled = enabled
        self.type = type
        self.level = min(max(level, 1), 9)  # Clamp to 1-9
        self.min_size_bytes = min_size_bytes
        self.compress_messages = compress_messages
        self.compress_documents = compress_documents
        self.compress_context = compress_context


class CompressionManager:
    """
    Manages compression and decompression of stored data.
    
    Provides transparent compression to reduce storage requirements.
    """
    
    def __init__(self, config: StorageConfig, 
                 compression_config: Optional[CompressionConfig] = None):
        """
        Initialize compression manager.
        
        Args:
            config: Storage configuration
            compression_config: Compression configuration
        """
        self.config = config
        self.compression_config = compression_config or CompressionConfig()
    
    async def compress_data(self, data: Union[str, bytes],
                          force: bool = False) -> bytes:
        """
        Compress data if it meets criteria.
        
        Args:
            data: Data to compress
            force: Force compression regardless of size
            
        Returns:
            Compressed data or original if compression not beneficial
        """
        if not self.compression_config.enabled:
            return data if isinstance(data, bytes) else data.encode('utf-8')
        
        # Convert to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Check size threshold
        if not force and len(data) < self.compression_config.min_size_bytes:
            return data
        
        # Compress based on type
        if self.compression_config.type == CompressionType.GZIP:
            compressed = await self._gzip_compress(data)
        elif self.compression_config.type == CompressionType.ZLIB:
            compressed = await self._zlib_compress(data)
        else:
            return data
        
        # Only use compression if it's beneficial
        if len(compressed) < len(data) * 0.9:  # At least 10% reduction
            return compressed
        else:
            return data
    
    async def decompress_data(self, data: bytes,
                            compression_type: Optional[CompressionType] = None) -> bytes:
        """
        Decompress data.
        
        Args:
            data: Compressed data
            compression_type: Type of compression used
            
        Returns:
            Decompressed data
        """
        if not compression_type:
            compression_type = await self._detect_compression(data)
        
        if compression_type == CompressionType.GZIP:
            return await self._gzip_decompress(data)
        elif compression_type == CompressionType.ZLIB:
            return await self._zlib_decompress(data)
        else:
            return data
    
    async def compress_file(self, file_path: Path, 
                          output_path: Optional[Path] = None) -> Path:
        """
        Compress a file.
        
        Args:
            file_path: Path to file to compress
            output_path: Optional output path (defaults to adding .gz)
            
        Returns:
            Path to compressed file
        """
        if not self.compression_config.enabled:
            return file_path
        
        if not output_path:
            output_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        # Read and compress in chunks for large files
        chunk_size = 1024 * 1024  # 1MB chunks
        
        if self.compression_config.type == CompressionType.GZIP:
            with open(file_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', 
                             compresslevel=self.compression_config.level) as f_out:
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
        else:
            # For zlib, we need to handle it differently
            compressor = zlib.compressobj(level=self.compression_config.level)
            with open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        compressed_chunk = compressor.compress(chunk)
                        if compressed_chunk:
                            f_out.write(compressed_chunk)
                    # Write any remaining data
                    final_chunk = compressor.flush()
                    if final_chunk:
                        f_out.write(final_chunk)
        
        return output_path
    
    async def decompress_file(self, file_path: Path,
                            output_path: Optional[Path] = None) -> Path:
        """
        Decompress a file.
        
        Args:
            file_path: Path to compressed file
            output_path: Optional output path
            
        Returns:
            Path to decompressed file
        """
        if not output_path:
            # Remove compression extension
            if file_path.suffix == '.gz':
                output_path = file_path.with_suffix('')
            else:
                output_path = file_path.with_suffix('.decompressed')
        
        compression_type = await self._detect_file_compression(file_path)
        
        if compression_type == CompressionType.GZIP:
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        elif compression_type == CompressionType.ZLIB:
            with open(file_path, 'rb') as f_in:
                compressed_data = f_in.read()
                decompressed = zlib.decompress(compressed_data)
                with open(output_path, 'wb') as f_out:
                    f_out.write(decompressed)
        else:
            # Just copy if not compressed
            import shutil
            shutil.copy2(file_path, output_path)
        
        return output_path
    
    async def compress_json(self, data: Dict[str, Any]) -> bytes:
        """
        Compress JSON data efficiently.
        
        Args:
            data: Dictionary to compress
            
        Returns:
            Compressed JSON data
        """
        # Convert to compact JSON first
        json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return await self.compress_data(json_str)
    
    async def decompress_json(self, data: bytes) -> Dict[str, Any]:
        """
        Decompress and parse JSON data.
        
        Args:
            data: Compressed JSON data
            
        Returns:
            Parsed dictionary
        """
        decompressed = await self.decompress_data(data)
        json_str = decompressed.decode('utf-8')
        return json.loads(json_str)
    
    def should_compress_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be compressed based on type.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file should be compressed
        """
        if not self.compression_config.enabled:
            return False
        
        # Check file type
        if file_path.name == self.config.messages_filename:
            return self.compression_config.compress_messages
        elif file_path.name == self.config.situational_context_filename:
            return self.compression_config.compress_context
        elif file_path.parent.name == self.config.document_storage_subdirectory_name:
            return self.compression_config.compress_documents
        
        # Default to True for other files
        return True
    
    async def get_compression_stats(self, file_path: Path) -> Dict[str, Any]:
        """
        Get compression statistics for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with compression stats
        """
        original_size = file_path.stat().st_size
        
        # Try compressing to see potential savings
        with open(file_path, 'rb') as f:
            data = f.read()
        
        compressed = await self.compress_data(data, force=True)
        compressed_size = len(compressed)
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compressed_size / original_size if original_size > 0 else 0,
            "space_saved": original_size - compressed_size,
            "space_saved_percent": ((original_size - compressed_size) / original_size * 100) 
                                  if original_size > 0 else 0
        }
    
    # === Private Methods ===
    
    async def _gzip_compress(self, data: bytes) -> bytes:
        """Compress using gzip"""
        return await asyncio.get_event_loop().run_in_executor(
            None, gzip.compress, data, self.compression_config.level
        )
    
    async def _gzip_decompress(self, data: bytes) -> bytes:
        """Decompress using gzip"""
        return await asyncio.get_event_loop().run_in_executor(
            None, gzip.decompress, data
        )
    
    async def _zlib_compress(self, data: bytes) -> bytes:
        """Compress using zlib"""
        return await asyncio.get_event_loop().run_in_executor(
            None, zlib.compress, data, self.compression_config.level
        )
    
    async def _zlib_decompress(self, data: bytes) -> bytes:
        """Decompress using zlib"""
        return await asyncio.get_event_loop().run_in_executor(
            None, zlib.decompress, data
        )
    
    async def _detect_compression(self, data: bytes) -> CompressionType:
        """Detect compression type from data"""
        if len(data) < 2:
            return CompressionType.NONE
        
        # Check magic numbers
        if data[:2] == b'\x1f\x8b':  # gzip magic number
            return CompressionType.GZIP
        elif data[:2] == b'\x78\x9c' or data[:2] == b'\x78\x01':  # zlib magic numbers
            return CompressionType.ZLIB
        
        return CompressionType.NONE
    
    async def _detect_file_compression(self, file_path: Path) -> CompressionType:
        """Detect compression type from file"""
        with open(file_path, 'rb') as f:
            header = f.read(2)
        
        return await self._detect_compression(header)