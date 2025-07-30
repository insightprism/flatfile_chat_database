"""
Migration utilities for exporting from flatfile to database and vice versa.

Provides tools to migrate data between the flatfile storage and traditional databases.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from abc import ABC, abstractmethod

from config import StorageConfig
from storage import StorageManager
from models import Message, Session, Document, SituationalContext, UserProfile
from streaming import ExportStreamer, StreamConfig
from compression import CompressionManager, CompressionConfig, CompressionType


@dataclass
class MigrationStats:
    """Statistics for migration operations"""
    total_users: int = 0
    total_sessions: int = 0
    total_messages: int = 0
    total_documents: int = 0
    total_contexts: int = 0
    errors: List[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def add_error(self, error_type: str, details: str, item_id: str = ""):
        """Add an error to the stats"""
        self.errors.append({
            "type": error_type,
            "details": details,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def finalize(self):
        """Mark migration as complete"""
        self.end_time = datetime.now()
    
    @property
    def duration_seconds(self) -> float:
        """Get migration duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_users": self.total_users,
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "total_documents": self.total_documents,
            "total_contexts": self.total_contexts,
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # First 10 errors
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds
        }


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize database connection and schema"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close database connection"""
        pass
    
    @abstractmethod
    async def save_user(self, user_data: Dict[str, Any]) -> bool:
        """Save user data"""
        pass
    
    @abstractmethod
    async def save_session(self, session_data: Dict[str, Any]) -> bool:
        """Save session data"""
        pass
    
    @abstractmethod
    async def save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> int:
        """Save messages batch"""
        pass
    
    @abstractmethod
    async def save_document(self, document_data: Dict[str, Any]) -> bool:
        """Save document metadata"""
        pass
    
    @abstractmethod
    async def save_context(self, context_data: Dict[str, Any]) -> bool:
        """Save context data"""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter for migration"""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite adapter.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
    
    async def initialize(self) -> bool:
        """Initialize database and create schema"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create schema
            await self._create_schema()
            
            return True
        except Exception as e:
            print(f"Failed to initialize SQLite: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    async def _create_schema(self):
        """Create database schema"""
        schema = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            created_at TEXT,
            updated_at TEXT,
            preferences TEXT,
            metadata TEXT
        );
        
        -- Sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            message_count INTEGER DEFAULT 0,
            metadata TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        
        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT,
            mime_type TEXT,
            size INTEGER,
            uploaded_at TEXT,
            uploaded_by TEXT,
            analysis TEXT,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        
        -- Contexts table
        CREATE TABLE IF NOT EXISTS contexts (
            context_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT,
            key_points TEXT,
            entities TEXT,
            confidence REAL,
            timestamp TEXT,
            is_current BOOLEAN DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
        CREATE INDEX IF NOT EXISTS idx_contexts_session ON contexts(session_id);
        """
        
        self.conn.executescript(schema)
        self.conn.commit()
    
    async def save_user(self, user_data: Dict[str, Any]) -> bool:
        """Save user data"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO users 
                (user_id, username, created_at, updated_at, preferences, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_data.get("user_id"),
                user_data.get("username"),
                user_data.get("created_at"),
                user_data.get("updated_at"),
                json.dumps(user_data.get("preferences", {})),
                json.dumps(user_data.get("metadata", {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False
    
    async def save_session(self, session_data: Dict[str, Any]) -> bool:
        """Save session data"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, title, created_at, updated_at, 
                 message_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_data.get("id"),
                session_data.get("user_id"),
                session_data.get("title"),
                session_data.get("created_at"),
                session_data.get("updated_at"),
                session_data.get("message_count", 0),
                json.dumps(session_data.get("metadata", {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    async def save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> int:
        """Save messages batch"""
        saved = 0
        try:
            for msg in messages:
                self.conn.execute("""
                    INSERT OR REPLACE INTO messages
                    (message_id, session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    msg.get("id"),
                    session_id,
                    msg.get("role"),
                    msg.get("content"),
                    msg.get("timestamp"),
                    json.dumps(msg.get("metadata", {}))
                ))
                saved += 1
            
            self.conn.commit()
        except Exception as e:
            print(f"Error saving messages: {e}")
            self.conn.rollback()
        
        return saved
    
    async def save_document(self, document_data: Dict[str, Any]) -> bool:
        """Save document metadata"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO documents
                (document_id, session_id, filename, original_name, mime_type,
                 size, uploaded_at, uploaded_by, analysis, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_data.get("filename"),  # Using filename as ID
                document_data.get("session_id"),
                document_data.get("filename"),
                document_data.get("original_name"),
                document_data.get("mime_type"),
                document_data.get("size"),
                document_data.get("uploaded_at"),
                document_data.get("uploaded_by"),
                json.dumps(document_data.get("analysis", {})),
                json.dumps(document_data.get("metadata", {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False
    
    async def save_context(self, context_data: Dict[str, Any]) -> bool:
        """Save context data"""
        try:
            # Mark existing contexts as not current
            if context_data.get("is_current", False):
                self.conn.execute("""
                    UPDATE contexts SET is_current = 0 
                    WHERE session_id = ?
                """, (context_data.get("session_id"),))
            
            self.conn.execute("""
                INSERT INTO contexts
                (context_id, session_id, summary, key_points, entities,
                 confidence, timestamp, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context_data.get("id", f"ctx_{datetime.now().timestamp()}"),
                context_data.get("session_id"),
                context_data.get("summary"),
                json.dumps(context_data.get("key_points", [])),
                json.dumps(context_data.get("entities", {})),
                context_data.get("confidence", 0.0),
                context_data.get("timestamp"),
                context_data.get("is_current", False)
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving context: {e}")
            return False


class FlatfileExporter:
    """
    Exports data from flatfile storage to various formats.
    """
    
    def __init__(self, storage_manager: StorageManager,
                 compression_config: Optional[CompressionConfig] = None):
        """
        Initialize exporter.
        
        Args:
            storage_manager: Storage manager instance
            compression_config: Optional compression configuration
        """
        self.storage = storage_manager
        self.config = storage_manager.config
        self.streamer = ExportStreamer(self.config)
        self.compression = CompressionManager(self.config, compression_config)
    
    async def export_to_database(self, adapter: DatabaseAdapter,
                               user_filter: Optional[List[str]] = None,
                               progress_callback: Optional[Callable[[str, int, int], None]] = None) -> MigrationStats:
        """
        Export all data to a database.
        
        Args:
            adapter: Database adapter to use
            user_filter: Optional list of user IDs to export
            progress_callback: Optional callback for progress updates
            
        Returns:
            Migration statistics
        """
        stats = MigrationStats()
        
        try:
            # Initialize database
            if not await adapter.initialize():
                stats.add_error("initialization", "Failed to initialize database")
                return stats
            
            # Get users to export
            if user_filter:
                users = user_filter
            else:
                users = await self.storage.list_users()
            
            stats.total_users = len(users)
            
            # Export each user
            for i, user_id in enumerate(users):
                if progress_callback:
                    progress_callback("users", i + 1, stats.total_users)
                
                try:
                    await self._export_user(user_id, adapter, stats)
                except Exception as e:
                    stats.add_error("user_export", str(e), user_id)
            
            stats.finalize()
            
        finally:
            await adapter.close()
        
        return stats
    
    async def export_to_json(self, output_path: Path,
                           user_filter: Optional[List[str]] = None,
                           compress: bool = True) -> MigrationStats:
        """
        Export all data to JSON format.
        
        Args:
            output_path: Output file path
            user_filter: Optional list of user IDs to export
            compress: Whether to compress the output
            
        Returns:
            Migration statistics
        """
        stats = MigrationStats()
        
        # Prepare output
        output_data = {
            "export_version": "1.0",
            "export_date": datetime.now().isoformat(),
            "users": {}
        }
        
        # Get users to export
        if user_filter:
            users = user_filter
        else:
            users = await self.storage.list_users()
        
        stats.total_users = len(users)
        
        # Export each user
        for user_id in users:
            try:
                user_data = await self._export_user_to_dict(user_id, stats)
                if user_data:
                    output_data["users"][user_id] = user_data
            except Exception as e:
                stats.add_error("user_export", str(e), user_id)
        
        # Write output
        try:
            if compress:
                json_data = await self.compression.compress_json(output_data)
                output_path = output_path.with_suffix('.json.gz')
                with open(output_path, 'wb') as f:
                    f.write(json_data)
            else:
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            stats.add_error("write_output", str(e))
        
        stats.finalize()
        return stats
    
    async def _export_user(self, user_id: str, adapter: DatabaseAdapter,
                         stats: MigrationStats):
        """Export a single user to database"""
        # Export user profile
        profile = await self.storage.get_user_profile(user_id)
        if profile:
            await adapter.save_user(profile)
        
        # Export sessions
        sessions = await self.storage.list_sessions(user_id, limit=None)
        stats.total_sessions += len(sessions)
        
        for session in sessions:
            try:
                # Save session metadata
                await adapter.save_session(session.to_dict())
                
                # Export messages using streaming
                async for chunk in self.streamer.stream_session_export(
                    user_id, session.session_id, include_documents=True, include_context=True
                ):
                    
                    if chunk["type"] == "messages":
                        saved = await adapter.save_messages(session.session_id, chunk["data"])
                        stats.total_messages += saved
                    
                    elif chunk["type"] == "documents":
                        for doc_id, doc_data in chunk["data"].items():
                            doc_data["session_id"] = session.session_id
                            if await adapter.save_document(doc_data):
                                stats.total_documents += 1
                    
                    elif chunk["type"] == "context":
                        context_data = chunk["data"]
                        context_data["session_id"] = session.session_id
                        context_data["is_current"] = True
                        if await adapter.save_context(context_data):
                            stats.total_contexts += 1
                    
                    elif chunk["type"] == "context_history":
                        context_data = chunk["data"]["context"]
                        context_data["session_id"] = session.session_id
                        context_data["id"] = chunk["data"]["snapshot_id"]
                        if await adapter.save_context(context_data):
                            stats.total_contexts += 1
                            
            except Exception as e:
                stats.add_error("session_export", str(e), session.session_id)
    
    async def _export_user_to_dict(self, user_id: str, 
                                 stats: MigrationStats) -> Optional[Dict[str, Any]]:
        """Export a single user to dictionary format"""
        user_data = {
            "profile": None,
            "sessions": {}
        }
        
        # Get user profile
        profile = await self.storage.get_user_profile(user_id)
        if profile:
            user_data["profile"] = profile
        
        # Get sessions
        sessions = await self.storage.list_sessions(user_id, limit=None)
        stats.total_sessions += len(sessions)
        
        for session in sessions:
            session_data = {
                "metadata": session.to_dict(),
                "messages": [],
                "documents": {},
                "context": None,
                "context_history": []
            }
            
            # Stream session data
            async for chunk in self.streamer.stream_session_export(
                user_id, session.session_id, include_documents=True, include_context=True
            ):
                if chunk["type"] == "messages":
                    session_data["messages"].extend(chunk["data"])
                    stats.total_messages += len(chunk["data"])
                elif chunk["type"] == "documents":
                    session_data["documents"].update(chunk["data"])
                    stats.total_documents += len(chunk["data"])
                elif chunk["type"] == "context":
                    session_data["context"] = chunk["data"]
                    stats.total_contexts += 1
                elif chunk["type"] == "context_history":
                    session_data["context_history"].append(chunk["data"])
                    stats.total_contexts += 1
            
            user_data["sessions"][session.session_id] = session_data
        
        return user_data


class DatabaseImporter:
    """
    Imports data from database to flatfile storage.
    """
    
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize importer.
        
        Args:
            storage_manager: Storage manager instance
        """
        self.storage = storage_manager
        self.config = storage_manager.config
    
    async def import_from_json(self, json_path: Path,
                             user_filter: Optional[List[str]] = None) -> MigrationStats:
        """
        Import data from JSON export.
        
        Args:
            json_path: Path to JSON file
            user_filter: Optional list of user IDs to import
            
        Returns:
            Migration statistics
        """
        stats = MigrationStats()
        
        try:
            # Load JSON data
            if json_path.suffix == '.gz':
                compression = CompressionManager(self.config)
                with open(json_path, 'rb') as f:
                    compressed_data = f.read()
                data = await compression.decompress_json(compressed_data)
            else:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            
            # Import users
            for user_id, user_data in data.get("users", {}).items():
                if user_filter and user_id not in user_filter:
                    continue
                
                try:
                    await self._import_user_from_dict(user_id, user_data, stats)
                    stats.total_users += 1
                except Exception as e:
                    stats.add_error("user_import", str(e), user_id)
            
        except Exception as e:
            stats.add_error("load_data", str(e))
        
        stats.finalize()
        return stats
    
    async def _import_user_from_dict(self, user_id: str, user_data: Dict[str, Any],
                                   stats: MigrationStats):
        """Import a single user from dictionary"""
        # Create user
        profile = user_data.get("profile", {})
        await self.storage.create_user(user_id, profile)
        
        # Import sessions
        for session_id, session_data in user_data.get("sessions", {}).items():
            try:
                # Create session
                metadata = session_data.get("metadata", {})
                title = metadata.get("title", "Imported Session")
                
                # We need to use the specific session ID
                # This is a bit hacky but necessary for preserving IDs
                created_session_id = await self.storage.create_session(user_id, title)
                
                # Import messages
                for msg_data in session_data.get("messages", []):
                    msg = Message.from_dict(msg_data)
                    await self.storage.add_message(user_id, created_session_id, msg)
                    stats.total_messages += 1
                
                # Import documents
                for doc_id, doc_data in session_data.get("documents", {}).items():
                    # Note: We can't import actual document content from metadata
                    # This would need to be handled separately
                    stats.total_documents += 1
                
                # Import context
                if session_data.get("context"):
                    context = SituationalContext.from_dict(session_data["context"])
                    await self.storage.update_context(user_id, created_session_id, context)
                    stats.total_contexts += 1
                
                # Import context history
                for snapshot_data in session_data.get("context_history", []):
                    context = SituationalContext.from_dict(snapshot_data["context"])
                    await self.storage.save_context_snapshot(user_id, created_session_id, context)
                    stats.total_contexts += 1
                
                stats.total_sessions += 1
                
            except Exception as e:
                stats.add_error("session_import", str(e), session_id)