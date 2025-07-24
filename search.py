"""
Advanced search functionality for the flatfile chat database.

Provides cross-session search, entity-based search, time-range queries,
and optimized full-text search with ranking.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re
from collections import defaultdict, Counter

from flatfile_chat_database.config import StorageConfig
from flatfile_chat_database.models import Message, Session, Document, SituationalContext, SearchType
from flatfile_chat_database.utils import read_json, read_jsonl, get_user_path


@dataclass
class SearchQuery:
    """Advanced search query parameters"""
    query: str
    user_id: Optional[str] = None
    session_ids: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    message_roles: Optional[List[str]] = None  # e.g., ["user", "assistant"]
    entities: Optional[Dict[str, List[str]]] = None  # e.g., {"languages": ["Python"]}
    include_documents: bool = False
    include_context: bool = False
    max_results: int = 100
    min_relevance_score: float = 0.0
    
    # Vector search parameters
    use_vector_search: bool = False
    similarity_threshold: float = 0.7
    embedding_provider: str = "nomic-ai"
    chunking_strategy: str = "optimized_summary"
    hybrid_search: bool = False
    vector_weight: float = 0.5


@dataclass
class SearchResult:
    """Search result with metadata and scoring"""
    id: str
    type: str  # "message", "document", "context"
    content: str
    session_id: str
    user_id: str
    timestamp: str
    relevance_score: float
    highlights: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) positions
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Sort by relevance score descending"""
        return self.relevance_score > other.relevance_score


class AdvancedSearchEngine:
    """
    Advanced search engine for the flatfile chat database.
    
    Provides sophisticated search capabilities including entity extraction,
    time-based filtering, and relevance ranking.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize search engine.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.base_path = Path(config.storage_base_path)
        
        # Compile regex patterns for entity extraction
        self.entity_patterns = {
            "urls": re.compile(r'https?://[^\s]+'),
            "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "code_blocks": re.compile(r'```[\s\S]*?```'),
            "mentions": re.compile(r'@\w+'),
            "hashtags": re.compile(r'#\w+'),
            "numbers": re.compile(r'\b\d+\.?\d*\b'),
        }
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute advanced search across the database.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of search results sorted by relevance
        """
        results = []
        
        # Get user scope
        if query.user_id:
            user_ids = [query.user_id]
        else:
            user_ids = await self._get_all_users()
        
        # Search across users
        for user_id in user_ids:
            user_results = await self._search_user(user_id, query)
            results.extend(user_results)
        
        # Sort by relevance and apply limit
        results.sort()
        
        # Apply minimum score filter
        results = [r for r in results if r.relevance_score >= query.min_relevance_score]
        
        return results[:query.max_results]
    
    async def search_by_entities(self, entities: Dict[str, List[str]], 
                                user_id: Optional[str] = None,
                                limit: int = 100) -> List[SearchResult]:
        """
        Search for messages containing specific entities.
        
        Args:
            entities: Entity types and values to search for
            user_id: Optional user scope
            limit: Maximum results
            
        Returns:
            List of search results
        """
        query = SearchQuery(
            query="",  # No text query
            user_id=user_id,
            entities=entities,
            max_results=limit
        )
        
        return await self.search(query)
    
    async def search_by_time_range(self, start_date: datetime, end_date: datetime,
                                  user_id: Optional[str] = None,
                                  query_text: Optional[str] = None,
                                  limit: int = 100) -> List[SearchResult]:
        """
        Search within a specific time range.
        
        Args:
            start_date: Start of time range
            end_date: End of time range
            user_id: Optional user scope
            query_text: Optional text to search for
            limit: Maximum results
            
        Returns:
            List of search results
        """
        query = SearchQuery(
            query=query_text or "",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            max_results=limit
        )
        
        return await self.search(query)
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types to values
        """
        entities = defaultdict(list)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type].extend(matches)
        
        # Extract programming languages (simple heuristic)
        lang_keywords = {
            "python": ["python", "py", "pip", "django", "flask"],
            "javascript": ["javascript", "js", "node", "npm", "react", "vue"],
            "java": ["java", "springframework", "maven", "gradle"],
            "go": ["golang", "go ", "goroutine"],
            "rust": ["rust", "cargo", "rustc"],
        }
        
        text_lower = text.lower()
        for lang, keywords in lang_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                entities["languages"].append(lang)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return dict(entities)
    
    async def build_search_index(self, user_id: str) -> Dict[str, Any]:
        """
        Build a search index for a user (for optimization).
        
        Args:
            user_id: User to index
            
        Returns:
            Index metadata
        """
        index = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "sessions": {},
            "entities": defaultdict(set),
            "word_frequencies": Counter()
        }
        
        # Get all sessions for user
        user_path = get_user_path(self.base_path, user_id, self.config)
        if not user_path.exists():
            return index
        
        # Index each session
        for session_dir in user_path.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith(self.config.session_id_prefix):
                session_data = await self._index_session(user_id, session_dir.name)
                if session_data:
                    index["sessions"][session_dir.name] = session_data
                    
                    # Aggregate entities
                    for entity_type, values in session_data.get("entities", {}).items():
                        index["entities"][entity_type].update(values)
                    
                    # Aggregate word frequencies
                    index["word_frequencies"].update(session_data.get("word_frequencies", {}))
        
        # Convert sets to lists for JSON serialization
        index["entities"] = {k: list(v) for k, v in index["entities"].items()}
        
        return index
    
    # === Private Methods ===
    
    async def _get_all_users(self) -> List[str]:
        """Get all user IDs in the system"""
        users = []
        
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                # Skip system directories
                if path.name not in [
                    self.config.panel_sessions_directory_name,
                    self.config.global_personas_directory_name,
                    self.config.system_config_directory_name
                ]:
                    users.append(path.name)
        
        return users
    
    async def _search_user(self, user_id: str, query: SearchQuery) -> List[SearchResult]:
        """Search within a specific user's data"""
        results = []
        
        user_path = get_user_path(self.base_path, user_id, self.config)
        if not user_path.exists():
            return results
        
        # Get sessions to search
        if query.session_ids:
            session_ids = query.session_ids
        else:
            session_ids = await self._get_user_sessions(user_id)
        
        # Search each session
        for session_id in session_ids:
            session_results = await self._search_session(user_id, session_id, query)
            results.extend(session_results)
        
        return results
    
    async def _get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        sessions = []
        user_path = get_user_path(self.base_path, user_id, self.config)
        
        if user_path.exists():
            for path in user_path.iterdir():
                if path.is_dir() and path.name.startswith(self.config.session_id_prefix):
                    sessions.append(path.name)
        
        return sessions
    
    async def _search_session(self, user_id: str, session_id: str, 
                            query: SearchQuery) -> List[SearchResult]:
        """Search within a specific session"""
        results = []
        
        session_path = self.base_path / user_id / session_id
        if not session_path.exists():
            return results
        
        # Search messages
        messages_file = session_path / self.config.messages_filename
        if messages_file.exists():
            message_results = await self._search_messages(
                user_id, session_id, messages_file, query
            )
            results.extend(message_results)
        
        # Search documents if requested
        if query.include_documents:
            docs_dir = session_path / self.config.document_storage_subdirectory_name
            if docs_dir.exists():
                doc_results = await self._search_documents(
                    user_id, session_id, docs_dir, query
                )
                results.extend(doc_results)
        
        # Search context if requested
        if query.include_context:
            context_results = await self._search_context(
                user_id, session_id, session_path, query
            )
            results.extend(context_results)
        
        return results
    
    async def _search_messages(self, user_id: str, session_id: str,
                             messages_file: Path, query: SearchQuery) -> List[SearchResult]:
        """Search messages in a session"""
        results = []
        
        # Read messages
        messages_data = await read_jsonl(messages_file, self.config)
        
        for msg_data in messages_data:
            try:
                msg = Message.from_dict(msg_data)
                
                # Apply filters
                if not self._message_matches_filters(msg, query):
                    continue
                
                # Calculate relevance
                score, highlights = self._calculate_relevance(msg.content, query)
                
                if score > 0 or not query.query:  # Include all if no text query
                    result = SearchResult(
                        id=msg.message_id,
                        type="message",
                        content=msg.content,
                        session_id=session_id,
                        user_id=user_id,
                        timestamp=msg.timestamp,
                        relevance_score=score,
                        highlights=highlights,
                        metadata={"role": msg.role}
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
        
        return results
    
    async def _search_documents(self, user_id: str, session_id: str,
                              docs_dir: Path, query: SearchQuery) -> List[SearchResult]:
        """Search documents in a session"""
        results = []
        
        # Read document metadata
        metadata_file = docs_dir / self.config.document_metadata_filename
        if not metadata_file.exists():
            return results
        
        docs_metadata = await read_json(metadata_file, self.config) or {}
        
        for doc_id, doc_data in docs_metadata.items():
            try:
                doc = Document.from_dict(doc_data)
                
                # Check time filter
                if not self._matches_time_filter(doc.uploaded_at, query):
                    continue
                
                # Search in document metadata and analysis
                search_text = f"{doc.original_name} {doc.metadata.get('description', '')}"
                
                # Include analysis results if available
                if doc.analysis:
                    for analysis_data in doc.analysis.values():
                        if "results" in analysis_data:
                            results_text = str(analysis_data["results"])
                            search_text += f" {results_text}"
                
                # Calculate relevance
                score, highlights = self._calculate_relevance(search_text, query)
                
                if score > 0 or not query.query:
                    result = SearchResult(
                        id=doc.filename,
                        type="document",
                        content=search_text[:500],  # Limit content size
                        session_id=session_id,
                        user_id=user_id,
                        timestamp=doc.uploaded_at,
                        relevance_score=score,
                        highlights=highlights,
                        metadata={
                            "original_name": doc.original_name,
                            "mime_type": doc.mime_type,
                            "size": doc.size
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing document: {e}")
                continue
        
        return results
    
    async def _search_context(self, user_id: str, session_id: str,
                            session_path: Path, query: SearchQuery) -> List[SearchResult]:
        """Search situational context"""
        results = []
        
        # Search current context
        context_file = session_path / self.config.situational_context_filename
        if context_file.exists():
            context_data = await read_json(context_file, self.config)
            if context_data:
                try:
                    context = SituationalContext.from_dict(context_data)
                    result = await self._search_single_context(
                        user_id, session_id, context, "current", query
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing context: {e}")
        
        # Search context history
        history_dir = session_path / self.config.context_history_subdirectory_name
        if history_dir.exists():
            for snapshot_file in history_dir.glob("*.json"):
                context_data = await read_json(snapshot_file, self.config)
                if context_data:
                    try:
                        context = SituationalContext.from_dict(context_data)
                        result = await self._search_single_context(
                            user_id, session_id, context, 
                            snapshot_file.stem, query
                        )
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing context snapshot: {e}")
        
        return results
    
    async def _search_single_context(self, user_id: str, session_id: str,
                                   context: SituationalContext, context_id: str,
                                   query: SearchQuery) -> Optional[SearchResult]:
        """Search a single context object"""
        # Check time filter
        if not self._matches_time_filter(context.timestamp, query):
            return None
        
        # Build search text
        search_text = f"{context.summary} "
        search_text += " ".join(context.key_points)
        
        # Add entities
        for entity_type, values in context.entities.items():
            search_text += f" {' '.join(values)}"
        
        # Check entity filter
        if query.entities:
            if not self._matches_entity_filter(context.entities, query.entities):
                return None
        
        # Calculate relevance
        score, highlights = self._calculate_relevance(search_text, query)
        
        if score > 0 or not query.query:
            return SearchResult(
                id=context_id,
                type="context",
                content=context.summary,
                session_id=session_id,
                user_id=user_id,
                timestamp=context.timestamp,
                relevance_score=score * context.confidence,  # Weight by confidence
                highlights=highlights,
                metadata={
                    "key_points": context.key_points,
                    "entities": context.entities,
                    "confidence": context.confidence
                }
            )
        
        return None
    
    async def _index_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Build index data for a session"""
        session_data = {
            "message_count": 0,
            "entities": defaultdict(set),
            "word_frequencies": Counter(),
            "first_message": None,
            "last_message": None
        }
        
        session_path = self.base_path / user_id / session_id
        messages_file = session_path / self.config.messages_filename
        
        if messages_file.exists():
            messages = await read_jsonl(messages_file, self.config)
            session_data["message_count"] = len(messages)
            
            for msg_data in messages:
                try:
                    msg = Message.from_dict(msg_data)
                    
                    # Extract entities
                    entities = await self.extract_entities(msg.content)
                    for entity_type, values in entities.items():
                        session_data["entities"][entity_type].update(values)
                    
                    # Count words
                    words = self._tokenize(msg.content.lower())
                    session_data["word_frequencies"].update(words)
                    
                    # Track first/last messages
                    if not session_data["first_message"]:
                        session_data["first_message"] = msg.timestamp
                    session_data["last_message"] = msg.timestamp
                    
                except Exception:
                    continue
        
        # Convert sets to lists
        session_data["entities"] = {
            k: list(v) for k, v in session_data["entities"].items()
        }
        
        return session_data
    
    def _message_matches_filters(self, message: Message, query: SearchQuery) -> bool:
        """Check if message matches query filters"""
        # Role filter
        if query.message_roles and message.role not in query.message_roles:
            return False
        
        # Time filter
        if not self._matches_time_filter(message.timestamp, query):
            return False
        
        return True
    
    def _matches_time_filter(self, timestamp: str, query: SearchQuery) -> bool:
        """Check if timestamp matches time range filter"""
        if not query.start_date and not query.end_date:
            return True
        
        try:
            dt = datetime.fromisoformat(timestamp)
            
            if query.start_date and dt < query.start_date:
                return False
            
            if query.end_date and dt > query.end_date:
                return False
            
            return True
            
        except Exception:
            return True  # Include if can't parse timestamp
    
    def _matches_entity_filter(self, entities: Dict[str, List[str]], 
                             filter_entities: Dict[str, List[str]]) -> bool:
        """Check if entities match filter"""
        for entity_type, required_values in filter_entities.items():
            if entity_type not in entities:
                return False
            
            entity_values = set(entities[entity_type])
            if not any(val in entity_values for val in required_values):
                return False
        
        return True
    
    def _calculate_relevance(self, text: str, query: SearchQuery) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate relevance score and find highlight positions.
        
        Returns:
            Tuple of (relevance_score, highlight_positions)
        """
        if not query.query:
            return 1.0, []  # Perfect score if no text query
        
        text_lower = text.lower()
        query_lower = query.query.lower()
        
        # Simple scoring based on matches
        score = 0.0
        highlights = []
        
        # Exact phrase match
        if query_lower in text_lower:
            score += 2.0
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(query_lower, start)
                if pos == -1:
                    break
                highlights.append((pos, pos + len(query_lower)))
                start = pos + 1
        
        # Word-based matching
        query_words = self._tokenize(query_lower)
        text_words = self._tokenize(text_lower)
        text_word_set = set(text_words)
        
        # Calculate word overlap
        matching_words = sum(1 for word in query_words if word in text_word_set)
        if query_words:
            word_score = matching_words / len(query_words)
            score += word_score
        
        # Boost score for recent content (simple recency bias)
        try:
            # This is a simple heuristic - could be improved
            score *= 1.0  # No recency bias for now
        except Exception:
            pass
        
        return score, highlights
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words
        return [w for w in words if len(w) > 2]
    
    async def vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute vector-based semantic search.
        
        Args:
            query: Search query with vector parameters
            
        Returns:
            List of search results based on semantic similarity
        """
        if not query.use_vector_search:
            return []
        
        # Import StorageManager here to avoid circular imports
        from .storage import StorageManager
        
        # Use StorageManager for vector search
        storage = StorageManager(self.config)
        
        results = await storage.vector_search(
            user_id=query.user_id,
            query=query.query,
            session_ids=query.session_ids,
            top_k=query.max_results,
            threshold=query.similarity_threshold,
            embedding_provider=query.embedding_provider
        )
        
        return results
    
    async def search_enhanced(self, query: SearchQuery) -> List[SearchResult]:
        """
        Enhanced search with vector support.
        
        This method routes to the appropriate search implementation based on
        query parameters (text, vector, or hybrid).
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of search results
        """
        if query.hybrid_search:
            # Use hybrid search from StorageManager
            from .storage import StorageManager
            storage = StorageManager(self.config)
            
            return await storage.hybrid_search(
                user_id=query.user_id,
                query=query.query,
                session_ids=query.session_ids,
                top_k=query.max_results,
                vector_weight=query.vector_weight
            )
        elif query.use_vector_search:
            # Use pure vector search
            return await self.vector_search(query)
        else:
            # Use traditional text search
            return await self.search(query)