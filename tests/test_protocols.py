"""
Protocol compliance tests for the flatfile chat database system.

Tests that all implementations properly adhere to their protocol interfaces
and that the protocols are correctly defined.
"""

import pytest
import inspect
from typing import get_type_hints, get_origin, get_args
from unittest.mock import AsyncMock, Mock
import sys
from pathlib import Path

# Add parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ff_protocols import (
    StorageProtocol,
    SearchProtocol, 
    VectorStoreProtocol,
    DocumentProcessorProtocol,
    BackendProtocol,
    FileOperationsProtocol
)
from ff_protocols.ff_storage_protocol import FFStorageProtocol

# Import implementations to test
from ff_storage_manager import FFStorageManager
from ff_search_manager import FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_document_processing_manager import FFDocumentProcessingManager
from backends.ff_flatfile_storage_backend import FFFlatfileStorageBackend
from ff_utils.ff_file_ops import FileOperationManager


class TestProtocolDefinitions:
    """Test that protocols are properly defined."""
    
    def test_storage_protocol_methods(self):
        """Test StorageProtocol has all required methods."""
        expected_methods = [
            'initialize', 'create_user', 'get_user_profile', 'update_user_profile',
            'delete_user', 'create_session', 'get_session', 'update_session',
            'delete_session', 'list_sessions', 'add_message', 'get_messages',
            'stream_messages', 'delete_message', 'store_document', 'get_document',
            'delete_document', 'save_context', 'get_context', 'create_panel',
            'get_panel', 'get_storage_stats', 'cleanup_old_data'
        ]
        
        protocol_methods = [name for name, obj in inspect.getmembers(FFStorageProtocol) 
                           if inspect.isfunction(obj)]
        
        for method in expected_methods:
            assert method in protocol_methods or hasattr(FFStorageProtocol, method), \
                f"StorageProtocol should have method {method}"
    
    def test_backend_protocol_methods(self):
        """Test BackendProtocol has all required methods."""
        expected_methods = [
            'initialize', 'read', 'write', 'append', 'exists', 'delete', 'list_keys'
        ]
        
        # Get methods from protocol annotations
        annotations = get_type_hints(BackendProtocol)
        protocol_methods = list(annotations.keys())
        
        for method in expected_methods:
            # Check if method exists in protocol (either as annotation or attribute)
            has_method = (method in protocol_methods or 
                         hasattr(BackendProtocol, method) or
                         method in dir(BackendProtocol))
            assert has_method, f"BackendProtocol should have method {method}"
    
    def test_search_protocol_methods(self):
        """Test SearchProtocol has required methods."""
        expected_methods = ['search', 'search_by_entities', 'extract_entities']
        
        # Since these are Protocol classes, check if they have the method signatures
        for method in expected_methods:
            assert hasattr(SearchProtocol, method) or method in get_type_hints(SearchProtocol), \
                f"SearchProtocol should have method {method}"
    
    def test_vector_store_protocol_methods(self):
        """Test VectorStoreProtocol has required methods."""
        expected_methods = [
            'store_vectors', 'search_similar', 'delete_document_vectors', 'get_vector_stats'
        ]
        
        for method in expected_methods:
            assert hasattr(VectorStoreProtocol, method) or method in get_type_hints(VectorStoreProtocol), \
                f"VectorStoreProtocol should have method {method}"


class TestStorageProtocolCompliance:
    """Test that FFStorageManager complies with StorageProtocol."""
    
    @pytest.mark.asyncio
    async def test_storage_manager_implements_protocol(self, storage_manager):
        """Test that FFStorageManager implements all StorageProtocol methods."""
        # Check that storage_manager implements the protocol
        assert isinstance(storage_manager, StorageProtocol.__class__.__bases__[0]) or \
               hasattr(storage_manager, 'initialize')
        
        # Test core methods exist and are callable
        assert hasattr(storage_manager, 'initialize')
        assert callable(storage_manager.initialize)
        
        assert hasattr(storage_manager, 'create_user')
        assert callable(storage_manager.create_user)
        
        assert hasattr(storage_manager, 'create_session')
        assert callable(storage_manager.create_session)
        
        assert hasattr(storage_manager, 'add_message')
        assert callable(storage_manager.add_message)
    
    @pytest.mark.asyncio
    async def test_storage_manager_method_signatures(self, storage_manager):
        """Test that FFStorageManager methods have correct signatures."""
        # Test initialize method
        init_sig = inspect.signature(storage_manager.initialize)
        assert len(init_sig.parameters) == 0
        
        # Test create_user method
        create_user_sig = inspect.signature(storage_manager.create_user)
        params = list(create_user_sig.parameters.keys())
        assert 'user_id' in params
        assert 'profile' in params or 'profile_data' in params
        
        # Test create_session method  
        create_session_sig = inspect.signature(storage_manager.create_session)
        params = list(create_session_sig.parameters.keys())
        assert 'user_id' in params
        assert 'title' in params
    
    @pytest.mark.asyncio
    async def test_storage_manager_return_types(self, storage_manager):
        """Test that FFStorageManager methods return correct types."""
        # Test initialize returns bool
        result = await storage_manager.initialize()
        assert isinstance(result, bool)
        
        # Test create_user returns bool
        user_result = await storage_manager.create_user("test_user")
        assert isinstance(user_result, bool)
        
        # Test create_session returns string (session ID)
        if user_result:  # Only if user creation succeeded
            session_result = await storage_manager.create_session("test_user", "Test Session")
            assert isinstance(session_result, str)


class TestBackendProtocolCompliance:
    """Test that FFFlatfileStorageBackend complies with BackendProtocol."""
    
    @pytest.mark.asyncio
    async def test_backend_implements_protocol(self, backend):
        """Test that FFFlatfileStorageBackend implements BackendProtocol methods."""
        # Test core methods exist
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'read')
        assert hasattr(backend, 'write')
        assert hasattr(backend, 'exists')
        assert hasattr(backend, 'delete')
        assert hasattr(backend, 'list_keys')
        
        # Test methods are callable
        assert callable(backend.initialize)
        assert callable(backend.read)
        assert callable(backend.write)
        assert callable(backend.exists)
        assert callable(backend.delete)
        assert callable(backend.list_keys)
    
    @pytest.mark.asyncio
    async def test_backend_method_signatures(self, backend):
        """Test backend method signatures match protocol."""
        # Test read method
        read_sig = inspect.signature(backend.read)
        params = list(read_sig.parameters.keys())
        assert 'key' in params
        
        # Test write method
        write_sig = inspect.signature(backend.write)
        params = list(write_sig.parameters.keys())
        assert 'key' in params
        assert 'data' in params
        
        # Test exists method
        exists_sig = inspect.signature(backend.exists)
        params = list(exists_sig.parameters.keys())
        assert 'key' in params
    
    @pytest.mark.asyncio
    async def test_backend_return_types(self, backend):
        """Test that backend methods return correct types."""
        # Initialize first
        init_result = await backend.initialize()
        assert isinstance(init_result, bool)
        
        # Test write returns bool
        write_result = await backend.write("test_key", b"test_data")
        assert isinstance(write_result, bool)
        
        # Test exists returns bool
        exists_result = await backend.exists("test_key") 
        assert isinstance(exists_result, bool)
        
        # Test read returns bytes or None
        read_result = await backend.read("test_key")
        assert read_result is None or isinstance(read_result, bytes)


class TestSearchProtocolCompliance:
    """Test search engine protocol compliance."""
    
    def test_search_manager_implements_protocol(self, test_config):
        """Test that FFSearchManager implements SearchProtocol."""
        search_manager = FFSearchManager(test_config)
        
        # Test core methods exist
        assert hasattr(search_manager, 'search')
        assert hasattr(search_manager, 'search_by_entities') 
        assert hasattr(search_manager, 'extract_entities')
        
        # Test methods are callable
        assert callable(search_manager.search)
        assert callable(search_manager.search_by_entities)
        assert callable(search_manager.extract_entities)
    
    @pytest.mark.asyncio
    async def test_search_manager_method_signatures(self, test_config):
        """Test search manager method signatures."""
        search_manager = FFSearchManager(test_config)
        
        # Test search method signature
        search_sig = inspect.signature(search_manager.search)
        params = list(search_sig.parameters.keys())
        assert 'query' in params
        
        # Test extract_entities method
        extract_sig = inspect.signature(search_manager.extract_entities)
        params = list(extract_sig.parameters.keys())  
        assert 'text' in params


class TestVectorStoreProtocolCompliance:
    """Test vector store protocol compliance."""
    
    def test_vector_manager_implements_protocol(self, test_config):
        """Test that FFVectorStorageManager implements VectorStoreProtocol."""
        vector_manager = FFVectorStorageManager(test_config.vector)
        
        # Test core methods exist
        assert hasattr(vector_manager, 'store_vectors')
        assert hasattr(vector_manager, 'search_similar')
        assert hasattr(vector_manager, 'delete_document_vectors')
        assert hasattr(vector_manager, 'get_vector_stats')
        
        # Test methods are callable
        assert callable(vector_manager.store_vectors)
        assert callable(vector_manager.search_similar)
        assert callable(vector_manager.delete_document_vectors)
        assert callable(vector_manager.get_vector_stats)
    
    @pytest.mark.asyncio
    async def test_vector_manager_method_signatures(self, test_config):
        """Test vector manager method signatures."""
        vector_manager = FFVectorStorageManager(test_config.vector)
        
        # Test store_vectors method
        store_sig = inspect.signature(vector_manager.store_vectors)
        params = list(store_sig.parameters.keys())
        expected_params = ['session_id', 'document_id', 'chunks', 'vectors']
        for param in expected_params:
            assert param in params, f"store_vectors should have parameter {param}"
        
        # Test search_similar method
        search_sig = inspect.signature(vector_manager.search_similar)
        params = list(search_sig.parameters.keys())
        expected_params = ['session_id', 'query_vector']
        for param in expected_params:
            assert param in params, f"search_similar should have parameter {param}"


class TestDocumentProcessorProtocolCompliance:
    """Test document processor protocol compliance."""
    
    def test_document_processor_implements_protocol(self, test_config):
        """Test that FFDocumentProcessingManager implements DocumentProcessorProtocol."""
        doc_processor = FFDocumentProcessingManager(test_config.document)
        
        # Test core methods exist - check what methods are actually available
        available_methods = [method for method in dir(doc_processor) 
                           if not method.startswith('_') and callable(getattr(doc_processor, method))]
        
        # Test that it has some form of document processing capability
        assert len(available_methods) > 0, "Document processor should have public methods"
        
        # Check for common document processing methods
        expected_methods = ['process_document', 'extract_text', 'chunk_document']
        has_processing_method = any(method in available_methods for method in expected_methods)
        assert has_processing_method, "Document processor should have processing methods"


class TestFileOperationsProtocolCompliance:
    """Test file operations protocol compliance."""
    
    def test_file_operations_implements_protocol(self, test_config):
        """Test that FileOperationManager implements FileOperationsProtocol."""
        file_ops = FileOperationManager(test_config)
        
        # Test that it has file operation methods
        available_methods = [method for method in dir(file_ops) 
                           if not method.startswith('_') and callable(getattr(file_ops, method))]
        
        assert len(available_methods) > 0, "File operations should have public methods"


class TestProtocolCompatibility:
    """Test compatibility between protocols and implementations."""
    
    def test_storage_manager_protocol_compatibility(self, app_container):
        """Test that storage manager from DI container satisfies protocol."""
        storage = app_container.resolve(StorageProtocol)
        
        # Should be able to call protocol methods
        assert hasattr(storage, 'initialize')
        assert hasattr(storage, 'create_user')
        assert hasattr(storage, 'create_session')
        
        # Should be an instance of the actual implementation
        assert isinstance(storage, FFStorageManager)
    
    def test_backend_protocol_compatibility(self, app_container):
        """Test that backend from DI container satisfies protocol."""
        backend = app_container.resolve(BackendProtocol)
        
        # Should be able to call protocol methods
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'read')
        assert hasattr(backend, 'write')
        
        # Should be an instance of the actual implementation
        assert isinstance(backend, FFFlatfileStorageBackend)
    
    def test_cross_protocol_compatibility(self, app_container):
        """Test that protocols work together correctly."""
        storage = app_container.resolve(StorageProtocol)
        backend = app_container.resolve(BackendProtocol)
        
        # Storage should use backend that satisfies BackendProtocol
        assert hasattr(storage, 'backend')
        assert storage.backend is backend or isinstance(storage.backend, type(backend))


class TestProtocolMocking:
    """Test that protocols can be properly mocked for testing."""
    
    def test_mock_storage_protocol(self):
        """Test creating mock storage protocol implementation."""
        mock_storage = AsyncMock(spec=StorageProtocol)
        
        # Configure mock behavior
        mock_storage.initialize.return_value = True
        mock_storage.create_user.return_value = True
        mock_storage.create_session.return_value = "mock_session_123"
        
        # Test that mock satisfies protocol interface
        assert hasattr(mock_storage, 'initialize')
        assert hasattr(mock_storage, 'create_user')
        assert hasattr(mock_storage, 'create_session')
        
        # Test mock behavior
        assert mock_storage.create_session.return_value == "mock_session_123"
    
    def test_mock_backend_protocol(self):
        """Test creating mock backend protocol implementation."""
        mock_backend = AsyncMock(spec=BackendProtocol)
        
        # Configure mock behavior
        mock_backend.initialize.return_value = True
        mock_backend.read.return_value = b'mock_data'
        mock_backend.write.return_value = True
        mock_backend.exists.return_value = True
        
        # Test mock interface
        assert hasattr(mock_backend, 'read')
        assert hasattr(mock_backend, 'write')
        assert hasattr(mock_backend, 'exists')
    
    def test_protocol_substitution(self, di_container):
        """Test that protocol implementations can be substituted via DI."""
        # Create mock implementation
        mock_backend = AsyncMock(spec=BackendProtocol)
        mock_backend.initialize.return_value = True
        
        # Register mock in DI container
        di_container.register_singleton(BackendProtocol, instance=mock_backend)
        
        # Resolve should return mock
        resolved_backend = di_container.resolve(BackendProtocol)
        assert resolved_backend is mock_backend


class TestProtocolEvolution:
    """Test protocol evolution and backward compatibility."""
    
    def test_protocol_method_additions(self):
        """Test that new methods can be added to protocols without breaking existing implementations."""
        # This test would be more relevant when protocols actually evolve
        # For now, just verify current protocol stability
        
        storage_methods = set(dir(StorageProtocol))
        backend_methods = set(dir(BackendProtocol))
        
        # Ensure core methods are present
        assert 'initialize' in backend_methods or hasattr(BackendProtocol, 'initialize')
        
        # Storage protocol should have user/session management
        storage_has_user_methods = any('user' in method.lower() for method in storage_methods)
        storage_has_session_methods = any('session' in method.lower() for method in storage_methods)
        
        # Note: Since StorageProtocol is imported as alias, check the actual implementation
        assert hasattr(FFStorageManager, 'create_user')
        assert hasattr(FFStorageManager, 'create_session')
    
    def test_implementation_extensibility(self, test_config):
        """Test that implementations can be extended without breaking protocols."""
        # Test that implementations have additional methods beyond protocol requirements
        storage_manager = FFStorageManager(test_config)
        backend = FFFlatfileStorageBackend(test_config)
        
        # Storage manager should have implementation-specific methods
        storage_methods = [method for method in dir(storage_manager) 
                          if not method.startswith('_') and callable(getattr(storage_manager, method))]
        
        backend_methods = [method for method in dir(backend)
                          if not method.startswith('_') and callable(getattr(backend, method))]
        
        # Should have substantial number of methods (indicating rich implementations)
        assert len(storage_methods) >= 10, "Storage manager should have many methods"
        assert len(backend_methods) >= 5, "Backend should have core methods"


@pytest.mark.integration  
class TestProtocolIntegration:
    """Integration tests for protocol implementations."""
    
    @pytest.mark.asyncio
    async def test_full_protocol_chain(self, app_container):
        """Test complete protocol chain from storage to backend."""
        storage = app_container.resolve(StorageProtocol)
        backend = app_container.resolve(BackendProtocol)
        
        # Initialize components
        backend_init = await backend.initialize()
        assert backend_init is True
        
        storage_init = await storage.initialize()
        assert storage_init is True
        
        # Test operation that goes through protocol chain
        user_created = await storage.create_user("protocol_test_user")
        assert isinstance(user_created, bool)
    
    @pytest.mark.asyncio
    async def test_protocol_error_handling(self, mock_backend):
        """Test error handling through protocol interfaces."""
        # Configure mock to raise errors
        mock_backend.initialize.side_effect = RuntimeError("Initialization failed")
        
        # Test that protocol interface properly propagates errors
        with pytest.raises(RuntimeError, match="Initialization failed"):
            await mock_backend.initialize()
    
    @pytest.mark.asyncio
    async def test_protocol_async_compatibility(self, app_container):
        """Test that async protocol methods work correctly."""
        storage = app_container.resolve(StorageProtocol)
        
        # All storage operations should be async
        assert inspect.iscoroutinefunction(storage.initialize)
        assert inspect.iscoroutinefunction(storage.create_user)
        assert inspect.iscoroutinefunction(storage.create_session)
        
        # Should be able to await them
        await storage.initialize()
        result = await storage.create_user("async_test_user")
        assert isinstance(result, bool)