"""
Search protocol interfaces.

Defines contracts for search functionality including queries, results,
and search engines.
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class SearchType(str, Enum):
    """Search operation types."""
    TEXT = "text"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchQueryProtocol(Protocol):
    """
    Search query interface.
    
    Defines the structure of search queries.
    """
    
    query: str
    user_id: Optional[str]
    session_ids: Optional[List[str]]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    message_roles: Optional[List[str]]
    entities: Optional[Dict[str, List[str]]]
    include_documents: bool
    include_context: bool
    max_results: int
    min_relevance_score: float
    
    # Vector search parameters
    use_vector_search: bool
    similarity_threshold: float
    embedding_provider: str
    chunking_strategy: str
    hybrid_search: bool
    vector_weight: float


class SearchResultProtocol(Protocol):
    """
    Search result interface.
    
    Defines the structure of search results.
    """
    
    id: str
    type: str  # "message", "document", "context"
    content: str
    session_id: str
    user_id: str
    timestamp: str
    relevance_score: float
    highlights: List[Tuple[int, int]]  # (start, end) positions
    metadata: Dict[str, Any]
    
    def __lt__(self, other: 'SearchResultProtocol') -> bool:
        """Compare results by relevance score."""
        ...


class SearchProtocol(Protocol):
    """
    Search engine interface.
    
    All search implementations must follow this protocol.
    """
    
    async def search(self, query: SearchQueryProtocol) -> List[SearchResultProtocol]:
        """
        Execute a search query.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of search results sorted by relevance
        """
        ...
    
    async def search_by_text(self, text: str, user_id: Optional[str] = None,
                           limit: int = 100) -> List[SearchResultProtocol]:
        """
        Simple text search.
        
        Args:
            text: Search text
            user_id: Optional user scope
            limit: Maximum results
            
        Returns:
            List of search results
        """
        ...
    
    async def search_by_vector(self, embedding: List[float], 
                             user_id: Optional[str] = None,
                             limit: int = 100,
                             threshold: float = 0.7) -> List[SearchResultProtocol]:
        """
        Vector similarity search.
        
        Args:
            embedding: Query embedding vector
            user_id: Optional user scope
            limit: Maximum results
            threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        ...
    
    async def hybrid_search(self, text: str, embedding: List[float],
                          user_id: Optional[str] = None,
                          limit: int = 100,
                          text_weight: float = 0.5) -> List[SearchResultProtocol]:
        """
        Hybrid text and vector search.
        
        Args:
            text: Search text
            embedding: Query embedding vector
            user_id: Optional user scope
            limit: Maximum results
            text_weight: Weight for text search (0-1)
            
        Returns:
            List of search results
        """
        ...
    
    async def search_by_entities(self, entities: Dict[str, List[str]], 
                               user_id: Optional[str] = None,
                               limit: int = 100) -> List[SearchResultProtocol]:
        """
        Search for messages containing specific entities.
        
        Args:
            entities: Entity types and values to search for
            user_id: Optional user scope
            limit: Maximum results
            
        Returns:
            List of search results
        """
        ...
    
    async def search_by_time_range(self, start_date: datetime, end_date: datetime,
                                 user_id: Optional[str] = None,
                                 query_text: Optional[str] = None,
                                 limit: int = 100) -> List[SearchResultProtocol]:
        """
        Search within a specific time range.
        
        Args:
            start_date: Start of time range
            end_date: End of time range
            user_id: Optional user scope
            query_text: Optional text filter
            limit: Maximum results
            
        Returns:
            List of search results
        """
        ...
    
    async def index_document(self, doc_id: str, content: str, 
                           metadata: Dict[str, Any],
                           embedding: Optional[List[float]] = None) -> bool:
        """
        Index a document for search.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Document metadata
            embedding: Optional pre-computed embedding
            
        Returns:
            True if successful
        """
        ...
    
    async def remove_from_index(self, doc_id: str) -> bool:
        """
        Remove a document from search index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful
        """
        ...
    
    async def update_index(self, doc_id: str, content: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         embedding: Optional[List[float]] = None) -> bool:
        """
        Update a document in the search index.
        
        Args:
            doc_id: Document identifier
            content: Updated content (if provided)
            metadata: Updated metadata (if provided)
            embedding: Updated embedding (if provided)
            
        Returns:
            True if successful
        """
        ...
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get search index statistics.
        
        Returns:
            Statistics dictionary with metrics like document count, index size, etc.
        """
        ...
    
    async def rebuild_index(self) -> bool:
        """
        Rebuild the entire search index.
        
        Returns:
            True if successful
        """
        ...
    
    async def optimize_index(self) -> bool:
        """
        Optimize search index for better performance.
        
        Returns:
            True if successful
        """
        ...