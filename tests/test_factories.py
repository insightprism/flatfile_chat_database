"""
Enhanced test factories for creating test objects with realistic data patterns.

Provides factories for creating various test objects with configurable parameters,
realistic data patterns, and different scenarios for comprehensive testing.
"""

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ff_class_configs.ff_chat_entities_config import (
    FFMessageDTO, FFSessionDTO, FFDocumentDTO, FFSituationalContextDTO,
    FFUserProfileDTO, FFPersonaDTO, FFPanelDTO, FFPanelMessageDTO,
    FFSearchQueryDTO, FFSearchResultDTO
)


@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    seed: Optional[int] = None                    # Random seed for reproducible tests
    realistic_data: bool = True                   # Generate realistic vs simple data
    include_metadata: bool = True                 # Include metadata fields
    use_timestamps: bool = True                   # Generate realistic timestamps
    content_variety: bool = True                  # Vary content patterns


class TestDataFactory:
    """Factory for creating test data objects with various patterns."""
    
    def __init__(self, config: Optional[TestDataConfig] = None):
        """Initialize factory with configuration."""
        self.config = config or TestDataConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    # === User Factories ===
    
    def create_user_profile(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        **kwargs
    ) -> FFUserProfileDTO:
        """Create a user profile with configurable properties."""
        user_id = user_id or self._generate_user_id()
        username = username or f"user_{user_id}"
        
        profile_data = {
            "user_id": user_id,
            "username": username,
            "email": f"{username}@example.com" if self.config.realistic_data else None,
            "display_name": username.title() if self.config.realistic_data else username,
            "preferences": self._generate_user_preferences() if self.config.realistic_data else {},
            "metadata": self._generate_metadata("user") if self.config.include_metadata else {}
        }
        
        profile_data.update(kwargs)
        return FFUserProfileDTO(**profile_data)
    
    def create_user_batch(self, count: int, **kwargs) -> List[FFUserProfileDTO]:
        """Create multiple users with unique IDs."""
        return [
            self.create_user_profile(user_id=f"user_{i:03d}", **kwargs)
            for i in range(count)
        ]
    
    # === Session Factories ===
    
    def create_session(
        self,
        user_id: str = "test_user",
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> FFSessionDTO:
        """Create a session with configurable properties."""
        session_id = session_id or self._generate_session_id()
        title = title or self._generate_session_title() if self.config.realistic_data else "Test Session"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "updated_at": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "metadata": self._generate_metadata("session") if self.config.include_metadata else {}
        }
        
        session_data.update(kwargs)
        return FFSessionDTO(**session_data)
    
    def create_session_with_messages(
        self,
        user_id: str = "test_user",
        message_count: int = 5,
        **kwargs
    ) -> tuple[FFSessionDTO, List[FFMessageDTO]]:
        """Create a session with associated messages."""
        session = self.create_session(user_id=user_id, **kwargs)
        messages = self.create_message_batch(message_count)
        return session, messages
    
    # === Message Factories ===
    
    def create_message(
        self,
        role: str = "user",
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs
    ) -> FFMessageDTO:
        """Create a message with configurable properties."""
        message_id = message_id or f"msg_{uuid.uuid4().hex[:8]}"
        content = content or self._generate_message_content(role) if self.config.realistic_data else f"Test message from {role}"
        
        message_data = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "metadata": self._generate_metadata("message") if self.config.include_metadata else {}
        }
        
        message_data.update(kwargs)
        return FFMessageDTO(**message_data)
    
    def create_message_batch(
        self,
        count: int,
        alternate_roles: bool = True,
        **kwargs
    ) -> List[FFMessageDTO]:
        """Create multiple messages with optional role alternation."""
        messages = []
        for i in range(count):
            if alternate_roles:
                role = "user" if i % 2 == 0 else "assistant"
            else:
                role = kwargs.get("role", "user")
            
            message = self.create_message(
                role=role,
                message_id=f"msg_{i:03d}",
                **{k: v for k, v in kwargs.items() if k != "role"}
            )
            messages.append(message)
        
        return messages
    
    def create_conversation(
        self,
        turns: int = 5,
        topic: Optional[str] = None
    ) -> List[FFMessageDTO]:
        """Create a realistic conversation with multiple turns."""
        topic = topic or random.choice([
            "technology", "cooking", "travel", "books", "science", "music"
        ]) if self.config.realistic_data else "general"
        
        messages = []
        for i in range(turns * 2):  # Each turn has user + assistant message
            role = "user" if i % 2 == 0 else "assistant"
            content = self._generate_conversation_message(role, topic, i // 2)
            
            message = self.create_message(
                role=role,
                content=content,
                message_id=f"conv_{i:03d}"
            )
            messages.append(message)
        
        return messages
    
    # === Document Factories ===
    
    def create_document(
        self,
        filename: Optional[str] = None,
        content: Optional[str] = None,
        document_id: Optional[str] = None,
        **kwargs
    ) -> FFDocumentDTO:
        """Create a document with configurable properties."""
        filename = filename or self._generate_filename()
        content = content or self._generate_document_content(filename) if self.config.realistic_data else "Test document content"
        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        
        document_data = {
            "document_id": document_id,
            "filename": filename,
            "content": content,
            "size": len(content.encode('utf-8')),
            "content_type": self._get_content_type(filename),
            "uploaded_at": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "metadata": self._generate_metadata("document") if self.config.include_metadata else {}
        }
        
        document_data.update(kwargs)
        return FFDocumentDTO(**document_data)
    
    def create_document_batch(self, count: int, **kwargs) -> List[FFDocumentDTO]:
        """Create multiple documents with different types."""
        file_types = [".txt", ".md", ".json", ".csv", ".log"] if self.config.realistic_data else [".txt"]
        documents = []
        
        for i in range(count):
            ext = file_types[i % len(file_types)]
            filename = f"test_doc_{i:03d}{ext}"
            document = self.create_document(filename=filename, **kwargs)
            documents.append(document)
        
        return documents
    
    # === Search Factories ===
    
    def create_search_query(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> FFSearchQueryDTO:
        """Create a search query with configurable parameters."""
        query = query or self._generate_search_query() if self.config.realistic_data else "test query"
        
        query_data = {
            "query": query,
            "user_id": user_id,
            "max_results": kwargs.get("max_results", 100),
            "min_relevance_score": kwargs.get("min_relevance_score", 0.0),
            "include_documents": kwargs.get("include_documents", False),
            "include_context": kwargs.get("include_context", False)
        }
        
        query_data.update(kwargs)
        return FFSearchQueryDTO(**query_data)
    
    def create_search_result(
        self,
        result_type: str = "message",
        content: Optional[str] = None,
        relevance_score: Optional[float] = None,
        **kwargs
    ) -> FFSearchResultDTO:
        """Create a search result with configurable properties."""
        content = content or f"Sample {result_type} content for search result"
        relevance_score = relevance_score or random.uniform(0.5, 1.0) if self.config.realistic_data else 0.8
        
        result_data = {
            "id": f"result_{uuid.uuid4().hex[:8]}",
            "type": result_type,
            "content": content,
            "user_id": kwargs.get("user_id", "test_user"),
            "session_id": kwargs.get("session_id", "test_session"),
            "relevance_score": relevance_score,
            "highlights": kwargs.get("highlights", []),
            "metadata": self._generate_metadata("search_result") if self.config.include_metadata else {}
        }
        
        result_data.update(kwargs)
        return FFSearchResultDTO(**result_data)
    
    # === Context Factories ===
    
    def create_context(
        self,
        summary: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        **kwargs
    ) -> FFSituationalContextDTO:
        """Create situational context with configurable properties."""
        summary = summary or self._generate_context_summary() if self.config.realistic_data else "Test context summary"
        key_points = key_points or self._generate_key_points() if self.config.realistic_data else ["Test point 1", "Test point 2"]
        
        context_data = {
            "summary": summary,
            "key_points": key_points,
            "entities": self._generate_entities() if self.config.realistic_data else {"topics": ["test"]},
            "confidence": kwargs.get("confidence", random.uniform(0.7, 0.95) if self.config.realistic_data else 0.8),
            "metadata": self._generate_metadata("context") if self.config.include_metadata else {}
        }
        
        context_data.update(kwargs)
        return FFSituationalContextDTO(**context_data)
    
    # === Persona Factories ===
    
    def create_persona(
        self,
        persona_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> FFPersonaDTO:
        """Create a persona with configurable properties."""
        persona_id = persona_id or f"persona_{uuid.uuid4().hex[:8]}"
        name = name or self._generate_persona_name() if self.config.realistic_data else "Test Persona"
        
        persona_data = {
            "persona_id": persona_id,
            "name": name,
            "description": self._generate_persona_description(name) if self.config.realistic_data else "Test persona description",
            "user_id": kwargs.get("user_id"),  # Optional for global personas
            "metadata": self._generate_metadata("persona") if self.config.include_metadata else {}
        }
        
        persona_data.update(kwargs)
        return FFPersonaDTO(**persona_data)
    
    # === Panel Factories ===
    
    def create_panel(
        self,
        panel_id: Optional[str] = None,
        user_id: str = "test_user",
        personas: Optional[List[str]] = None,
        **kwargs
    ) -> FFPanelDTO:
        """Create a panel with configurable properties."""
        panel_id = panel_id or f"panel_{uuid.uuid4().hex[:8]}"
        personas = personas or [f"persona_{i}" for i in range(3)] if self.config.realistic_data else ["test_persona"]
        
        panel_data = {
            "panel_id": panel_id,
            "user_id": user_id,
            "personas": personas,
            "title": self._generate_panel_title() if self.config.realistic_data else "Test Panel",
            "created_at": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "updated_at": self._generate_timestamp() if self.config.use_timestamps else datetime.now().isoformat(),
            "metadata": self._generate_metadata("panel") if self.config.include_metadata else {}
        }
        
        panel_data.update(kwargs)
        return FFPanelDTO(**panel_data)
    
    # === Size-Based Factories ===
    
    def create_large_message(self, size_kb: int = 100) -> FFMessageDTO:
        """Create a large message for testing size limits."""
        content = "A" * (size_kb * 1024)
        return self.create_message(content=content, message_id="large_message")
    
    def create_large_document(self, size_mb: int = 10) -> FFDocumentDTO:
        """Create a large document for testing size limits."""
        content = "Large document content. " * (size_mb * 1024 * 1024 // 25)
        return self.create_document(
            filename="large_document.txt",
            content=content,
            document_id="large_document"
        )
    
    # === Edge Case Factories ===
    
    def create_edge_case_user(self, case: str) -> FFUserProfileDTO:
        """Create users for edge case testing."""
        cases = {
            "min_length": FFUserProfileDTO(user_id="a", username="a"),
            "max_length": FFUserProfileDTO(user_id="x" * 50, username="x" * 50),
            "special_chars": FFUserProfileDTO(user_id="user_123-test", username="user@test"),
            "unicode": FFUserProfileDTO(user_id="用户测试", username="тест пользователя"),
            "empty_metadata": FFUserProfileDTO(user_id="empty_user", username="empty", metadata={})
        }
        return cases.get(case, self.create_user_profile())
    
    def create_edge_case_message(self, case: str) -> FFMessageDTO:
        """Create messages for edge case testing."""
        cases = {
            "empty_content": FFMessageDTO(message_id="empty", role="user", content=""),
            "only_whitespace": FFMessageDTO(message_id="whitespace", role="user", content="   \n\t  "),
            "unicode_content": FFMessageDTO(message_id="unicode", role="user", content="Hello 世界! 🚀 Тест"),
            "json_content": FFMessageDTO(message_id="json", role="user", content='{"test": "value", "nested": {"key": 123}}'),
            "markdown_content": FFMessageDTO(message_id="markdown", role="user", content="# Title\n\n**Bold** and *italic* text\n\n```python\ncode_block()\n```")
        }
        return cases.get(case, self.create_message())
    
    # === Private Helper Methods ===
    
    def _generate_user_id(self) -> str:
        """Generate a realistic user ID."""
        if self.config.realistic_data:
            return f"user_{random.randint(1000, 9999)}"
        return f"test_user_{random.randint(1, 100)}"
    
    def _generate_session_id(self) -> str:
        """Generate a realistic session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(100000, 999999)
        return f"chat_session_{timestamp}_{random_suffix}"
    
    def _generate_session_title(self) -> str:
        """Generate a realistic session title."""
        topics = [
            "Morning Chat", "Project Discussion", "Help with Code", "Random Thoughts",
            "Book Recommendations", "Travel Planning", "Recipe Ideas", "Tech Support"
        ]
        return random.choice(topics)
    
    def _generate_message_content(self, role: str) -> str:
        """Generate realistic message content based on role."""
        if role == "user":
            templates = [
                "Can you help me with {}?",
                "I'm wondering about {}.",
                "What do you think about {}?",
                "I need assistance with {}.",
                "Could you explain {}?"
            ]
            topics = ["programming", "cooking", "travel", "science", "music", "books"]
            return random.choice(templates).format(random.choice(topics))
        else:  # assistant
            templates = [
                "I'd be happy to help you with that!",
                "Here's what I think about this topic...",
                "Let me explain this concept to you.",
                "That's a great question! Here's my perspective...",
                "I can definitely assist you with that."
            ]
            return random.choice(templates)
    
    def _generate_conversation_message(self, role: str, topic: str, turn: int) -> str:
        """Generate contextual conversation messages."""
        if role == "user":
            if turn == 0:
                return f"I'd like to learn about {topic}. Can you help?"
            else:
                return f"That's interesting! Can you tell me more about the {topic} aspects?"
        else:
            return f"I'd be happy to explain {topic}. Here are some key points to consider..."
    
    def _generate_filename(self) -> str:
        """Generate a realistic filename."""
        names = ["readme", "guide", "notes", "data", "config", "report", "analysis"]
        extensions = [".txt", ".md", ".json", ".csv", ".log"]
        return f"{random.choice(names)}_{random.randint(1, 100)}{random.choice(extensions)}"
    
    def _generate_document_content(self, filename: str) -> str:
        """Generate realistic document content based on filename."""
        ext = Path(filename).suffix.lower()
        if ext == ".json":
            return '{"title": "Test Document", "version": "1.0", "data": [1, 2, 3]}'
        elif ext == ".md":
            return "# Test Document\n\nThis is a **test** document with *markdown* formatting.\n\n## Section\n\nSome content here."
        elif ext == ".csv":
            return "name,age,city\nAlice,30,New York\nBob,25,London\nCharlie,35,Tokyo"
        else:
            return f"This is a test document with filename: {filename}\n\nIt contains some sample content for testing purposes."
    
    def _generate_search_query(self) -> str:
        """Generate a realistic search query."""
        queries = [
            "help with programming",
            "recipe for pasta",
            "travel to Japan",
            "explain machine learning",
            "book recommendations",
            "Python tutorial",
            "cooking tips"
        ]
        return random.choice(queries)
    
    def _generate_context_summary(self) -> str:
        """Generate a realistic context summary."""
        summaries = [
            "Discussion about programming concepts and best practices.",
            "Conversation about travel plans and recommendations.",
            "Help session for cooking and recipe suggestions.",
            "Educational dialogue about science and technology.",
            "Planning session for project management."
        ]
        return random.choice(summaries)
    
    def _generate_key_points(self) -> List[str]:
        """Generate realistic key points."""
        points = [
            ["Main topic discussed", "Key insights shared", "Action items identified"],
            ["Problem definition", "Solution approaches", "Next steps planned"],
            ["Background context", "Current situation", "Future considerations"],
            ["User questions", "Detailed explanations", "Follow-up topics"]
        ]
        return random.choice(points)
    
    def _generate_entities(self) -> Dict[str, List[str]]:
        """Generate realistic entities."""
        return {
            "topics": random.sample(["programming", "travel", "cooking", "science", "music"], 2),
            "technologies": random.sample(["Python", "JavaScript", "React", "Docker"], 1),
            "locations": random.sample(["New York", "London", "Tokyo", "Paris"], 1)
        }
    
    def _generate_persona_name(self) -> str:
        """Generate a realistic persona name."""
        names = [
            "The Helper", "Code Mentor", "Travel Guide", "Chef Assistant",
            "Science Educator", "Creative Writer", "Data Analyst", "Problem Solver"
        ]
        return random.choice(names)
    
    def _generate_persona_description(self, name: str) -> str:
        """Generate persona description based on name."""
        return f"A helpful {name.lower()} persona that specializes in providing assistance and guidance."
    
    def _generate_panel_title(self) -> str:
        """Generate a realistic panel title."""
        titles = [
            "Expert Panel Discussion",
            "Multi-Perspective Analysis",
            "Collaborative Problem Solving",
            "Team Consultation Session",
            "Advisory Board Meeting"
        ]
        return random.choice(titles)
    
    def _generate_user_preferences(self) -> Dict[str, Any]:
        """Generate realistic user preferences."""
        return {
            "theme": random.choice(["light", "dark", "auto"]),
            "language": random.choice(["en", "es", "fr", "de", "ja"]),
            "timezone": random.choice(["UTC", "EST", "PST", "GMT"]),
            "notifications": random.choice([True, False]),
            "auto_save": True
        }
    
    def _generate_metadata(self, object_type: str) -> Dict[str, Any]:
        """Generate realistic metadata for different object types."""
        base_metadata = {
            "created_by": "test_factory",
            "test_run": True,
            "object_type": object_type
        }
        
        if object_type == "message":
            base_metadata.update({
                "word_count": random.randint(5, 50),
                "language": "en",
                "sentiment": random.choice(["positive", "neutral", "negative"])
            })
        elif object_type == "document":
            base_metadata.update({
                "file_hash": f"hash_{uuid.uuid4().hex[:16]}",
                "processed": True,
                "encoding": "utf-8"
            })
        elif object_type == "session":
            base_metadata.update({
                "duration_minutes": random.randint(5, 120),
                "message_count": random.randint(1, 50),
                "last_active": self._generate_timestamp()
            })
        
        return base_metadata
    
    def _generate_timestamp(self, days_ago: int = 30) -> str:
        """Generate a timestamp within the last N days."""
        base_time = datetime.now()
        random_delta = timedelta(
            days=random.randint(0, days_ago),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        return (base_time - random_delta).isoformat()
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on filename."""
        ext = Path(filename).suffix.lower()
        type_map = {
            ".txt": "text/plain",
            ".md": "text/markdown", 
            ".json": "application/json",
            ".csv": "text/csv",
            ".log": "text/plain"
        }
        return type_map.get(ext, "application/octet-stream")


# === Convenience Factory Instances ===

# Default factory for general use
factory = TestDataFactory()

# Factory with reproducible seed for consistent tests
seeded_factory = TestDataFactory(TestDataConfig(seed=42))

# Factory for simple, fast test data
simple_factory = TestDataFactory(TestDataConfig(
    realistic_data=False,
    include_metadata=False,
    use_timestamps=False,
    content_variety=False
))

# Factory for realistic, complex test data
realistic_factory = TestDataFactory(TestDataConfig(
    realistic_data=True,
    include_metadata=True,
    use_timestamps=True,
    content_variety=True
))