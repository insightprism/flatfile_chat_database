#!/usr/bin/env python3
"""
CLI Interactive Demo for Flatfile Chat Database

A menu-driven interactive demonstration of the flatfile chat database system.
Run this script to explore features through a command-line interface.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional, Dict, Any

# Add parent directory to path
sys.path.append('..')

# Add PrismMind directory to path (if available)
prismmind_path = '/home/markly2/prismmind'
if os.path.exists(prismmind_path):
    sys.path.append(prismmind_path)
    print(f"✅ Added PrismMind path: {prismmind_path}")
else:
    print(f"⚠️ PrismMind not found at: {prismmind_path} - will use legacy document processing")

from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, FFSession, FFDocument, FFUserProfile, MessageRole
from ff_search_manager import FFSearchQueryDTO, FFSearchManager
from ff_vector_storage_manager import FFVectorStorageManager
from ff_document_processing_manager import FFDocumentProcessingManager


class InteractiveDemo:
    """Interactive CLI demo for the flatfile chat database."""
    
    def __init__(self):
        self.demo_data_path = Path("./demo_data_cli")
        self.config = None
        self.storage_manager = None
        self.search_engine = None
        self.vector_storage = None
        self.current_user = None
        self.current_session = None
        
    async def initialize(self):
        """Initialize the demo environment."""
        print("🚀 Initializing Flatfile Chat Database Demo...")
        
        # Create demo data directory
        self.demo_data_path.mkdir(exist_ok=True)
        
        # Setup configuration
        self.config = load_config()
        self.config.storage.base_path = str(self.demo_data_path)
        self.config.storage.enable_compression = False
        self.config.locking.enabled = True
        
        # Initialize components
        self.storage_manager = FFStorageManager(self.config)
        self.search_engine = FFSearchManager(self.config)
        self.vector_storage = FFVectorStorageManager(self.config)
        
        print(f"✅ Demo initialized! Data will be stored in: {self.demo_data_path}")
        
    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*60)
        print("🏠 FLATFILE CHAT DATABASE - INTERACTIVE DEMO")
        print("="*60)
        
        status = f"User: {self.current_user or 'None'}"
        if self.current_session:
            status += f" | Session: {self.current_session.title}"
        print(f"📊 Status: {status}")
        
        print("\n📋 MENU OPTIONS:")
        print("1.  👤 User Management")
        print("2.  💬 Chat Sessions")  
        print("3.  📄 Document Management")
        print("4.  🔍 Search & Retrieval")
        print("5.  📊 Statistics & Info")
        print("6.  ⚙️  Configuration")
        print("7.  📁 File Explorer")
        print("8.  🧪 Run Quick Test")
        print("9.  🧹 Cleanup Demo Data")
        print("0.  🚪 Exit")
        print("-" * 60)
        
    async def user_management_menu(self):
        """Handle user management operations."""
        while True:
            print("\n👤 USER MANAGEMENT")
            print("1. Create new user")
            print("2. List all users")
            print("3. Select current user")
            print("4. View user profile")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                await self.create_user()
            elif choice == "2":
                await self.list_users()
            elif choice == "3":
                await self.select_user()
            elif choice == "4":
                await self.view_user_profile()
            elif choice == "0":
                break
            else:
                print("❌ Invalid choice!")
                
    async def create_user(self):
        """Create a new user."""
        print("\n➕ CREATE NEW USER")
        
        user_id = input("Enter user ID: ").strip()
        if not user_id:
            print("❌ User ID cannot be empty!")
            return
            
        display_name = input("Enter display name: ").strip()
        if not display_name:
            display_name = user_id
            
        # Optional metadata
        department = input("Enter department (optional): ").strip()
        role = input("Enter role (optional): ").strip()
        
        metadata = {}
        if department:
            metadata["department"] = department
        if role:
            metadata["role"] = role
            
        # Create user profile
        profile = FFUserProfile(
            user_id=user_id,
            username=display_name,
            preferences={"theme": "dark", "language": "en"},
            metadata=metadata
        )
        
        try:
            await self.storage_manager.store_user_profile(profile)
            print(f"✅ User '{display_name}' created successfully!")
            self.current_user = user_id
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            
    async def list_users(self):
        """List all users."""
        print("\n📋 ALL USERS:")
        try:
            users_dir = self.demo_data_path / "users"
            if not users_dir.exists():
                print("📭 No users found.")
                return
                
            user_dirs = [d for d in users_dir.iterdir() if d.is_dir()]
            if not user_dirs:
                print("📭 No users found.")
                return
                
            for i, user_dir in enumerate(user_dirs, 1):
                user_id = user_dir.name
                profile_file = user_dir / "profile.json"
                
                if profile_file.exists():
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                    display_name = profile_data.get('display_name', user_id)
                    metadata = profile_data.get('metadata', {})
                    
                    print(f"{i}. {display_name} ({user_id})")
                    if metadata:
                        print(f"   Metadata: {metadata}")
                else:
                    print(f"{i}. {user_id} (no profile)")
                    
        except Exception as e:
            print(f"❌ Error listing users: {e}")
            
    async def select_user(self):
        """Select current user."""
        await self.list_users()
        user_id = input("\nEnter user ID to select: ").strip()
        
        if user_id:
            # Check if user exists
            user_dir = self.demo_data_path / "users" / user_id
            if user_dir.exists():
                self.current_user = user_id
                print(f"✅ Selected user: {user_id}")
            else:
                print(f"❌ User '{user_id}' not found!")
                
    async def view_user_profile(self):
        """View current user profile."""
        if not self.current_user:
            print("❌ No user selected!")
            return
            
        try:
            profile_file = self.demo_data_path / "users" / self.current_user / "profile.json"
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    
                print(f"\n👤 PROFILE: {profile_data.get('display_name', self.current_user)}")
                print(f"User ID: {profile_data.get('user_id')}")
                print(f"Created: {profile_data.get('created_at')}")
                print(f"Preferences: {profile_data.get('preferences', {})}")
                print(f"Metadata: {profile_data.get('metadata', {})}")
            else:
                print("❌ Profile not found!")
        except Exception as e:
            print(f"❌ Error viewing profile: {e}")
            
    async def chat_sessions_menu(self):
        """Handle chat session operations."""
        if not self.current_user:
            print("❌ Please select a user first!")
            return
            
        while True:
            print(f"\n💬 CHAT SESSIONS (User: {self.current_user})")
            print("1. Create new session")
            print("2. List sessions")
            print("3. Select session")
            print("4. Add message to session")
            print("5. View session messages")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                await self.create_session()
            elif choice == "2":
                await self.list_sessions()
            elif choice == "3":
                await self.select_session()
            elif choice == "4":
                await self.add_message()
            elif choice == "5":
                await self.view_messages()
            elif choice == "0":
                break
            else:
                print("❌ Invalid choice!")
                
    async def create_session(self):
        """Create a new chat session."""
        print("\n➕ CREATE NEW SESSION")
        
        title = input("Enter session title: ").strip()
        if not title:
            title = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
        try:
            session = await self.storage_manager.create_session(
                user_id=self.current_user,
                title=title
            )
            self.current_session = session
            print(f"✅ Session '{title}' created!")
            print(f"📋 Session ID: {session.session_id}")
        except Exception as e:
            print(f"❌ Error creating session: {e}")
            
    async def list_sessions(self):
        """List all sessions for current user."""
        try:
            user_dir = self.demo_data_path / "users" / self.current_user
            if not user_dir.exists():
                print("📭 No sessions found.")
                return
                
            session_dirs = [d for d in user_dir.iterdir() if d.is_dir() and d.name != "profile.json"]
            if not session_dirs:
                print("📭 No sessions found.")
                return
                
            print(f"\n📋 SESSIONS FOR {self.current_user}:")
            for i, session_dir in enumerate(session_dirs, 1):
                session_file = session_dir / "session.json"
                if session_file.exists():
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    title = session_data.get('title', 'Untitled')
                    created_at = session_data.get('created_at', 'Unknown')
                    message_count = session_data.get('message_count', 0)
                    
                    print(f"{i}. {title}")
                    print(f"   ID: {session_dir.name}")
                    print(f"   Created: {created_at}")
                    print(f"   Messages: {message_count}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            
    async def select_session(self):
        """Select current session."""
        await self.list_sessions()
        session_id = input("\nEnter session ID to select: ").strip()
        
        if session_id:
            try:
                session_file = self.demo_data_path / "users" / self.current_user / session_id / "session.json"
                if session_file.exists():
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Create session object
                    self.current_session = FFSession(
                        session_id=session_id,
                        user_id=self.current_user,
                        title=session_data.get('title', 'Untitled'),
                        created_at=session_data.get('created_at'),
                        updated_at=session_data.get('updated_at'),
                        message_count=session_data.get('message_count', 0)
                    )
                    print(f"✅ Selected session: {self.current_session.title}")
                else:
                    print(f"❌ Session '{session_id}' not found!")
            except Exception as e:
                print(f"❌ Error selecting session: {e}")
                
    async def add_message(self):
        """Add a message to current session."""
        if not self.current_session:
            print("❌ No session selected!")
            return
            
        print("\n➕ ADD MESSAGE")
        print("1. User message")
        print("2. Assistant message")
        print("3. System message")
        
        role_choice = input("Select role: ").strip()
        
        role_map = {"1": MessageRole.USER, "2": MessageRole.ASSISTANT, "3": MessageRole.SYSTEM}
        role = role_map.get(role_choice)
        
        if not role:
            print("❌ Invalid role selection!")
            return
            
        print(f"\nEnter message content (role: {role}):")
        print("(Press Enter twice to finish)")
        
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
            
        content = "\n".join(lines).strip()
        if not content:
            print("❌ Message content cannot be empty!")
            return
            
        try:
            message = FFMessageDTO(role=role, content=content)
            await self.storage_manager.add_message(
                self.current_user,
                self.current_session.session_id,
                message
            )
            print("✅ Message added successfully!")
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            
    async def view_messages(self):
        """View messages in current session."""
        if not self.current_session:
            print("❌ No session selected!")
            return
            
        try:
            messages_file = (self.demo_data_path / "users" / self.current_user / 
                           self.current_session.session_id / "messages.jsonl")
            
            if not messages_file.exists():
                print("📭 No messages found in this session.")
                return
                
            print(f"\n💬 MESSAGES IN '{self.current_session.title}':")
            print("-" * 60)
            
            with open(messages_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    message_data = json.loads(line)
                    role = message_data.get('role', 'unknown')
                    content = message_data.get('content', '')
                    timestamp = message_data.get('timestamp', '')
                    
                    print(f"[{i}] {role.upper()} ({timestamp})")
                    print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error viewing messages: {e}")
            
    async def document_management_menu(self):
        """Handle document operations."""
        if not self.current_user or not self.current_session:
            print("❌ Please select a user and session first!")
            return
            
        while True:
            print(f"\n📄 DOCUMENT MANAGEMENT")
            print(f"Session: {self.current_session.title}")
            print("1. Add document from text")
            print("2. Add document from file")
            print("3. List documents")
            print("4. View document")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                await self.add_text_document()
            elif choice == "2":
                await self.add_file_document()
            elif choice == "3":
                await self.list_documents()
            elif choice == "4":
                await self.view_document()
            elif choice == "0":
                break
            else:
                print("❌ Invalid choice!")
                
    async def add_text_document(self):
        """Add a document from text input."""
        print("\n➕ ADD TEXT DOCUMENT")
        
        filename = input("Enter filename: ").strip()
        if not filename:
            filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        print("Enter document content (Press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
            
        content = "\n".join(lines).strip()
        if not content:
            print("❌ Document content cannot be empty!")
            return
            
        try:
            document = FFDocument(
                filename=filename,
                content=content,
                metadata={"source": "manual_input", "created_by": self.current_user}
            )
            
            doc_id = await self.storage_manager.store_document(
                self.current_session.session_id,
                self.current_user,
                document
            )
            
            print(f"✅ Document '{filename}' added successfully!")
            print(f"📋 Document ID: {doc_id}")
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            
    async def add_file_document(self):
        """Add a document from file."""
        print("\n📁 ADD DOCUMENT FROM FILE")
        
        file_path = input("Enter file path: ").strip()
        if not file_path:
            print("❌ File path cannot be empty!")
            return
            
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            document = FFDocument(
                filename=file_path.name,
                content=content,
                metadata={"source": "file_upload", "original_path": str(file_path)}
            )
            
            doc_id = await self.storage_manager.store_document(
                self.current_session.session_id,
                self.current_user,
                document
            )
            
            print(f"✅ Document '{file_path.name}' added successfully!")
            print(f"📋 Document ID: {doc_id}")
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            
    async def list_documents(self):
        """List documents in current session."""
        try:
            docs_dir = (self.demo_data_path / "users" / self.current_user / 
                       self.current_session.session_id / "documents")
            
            if not docs_dir.exists():
                print("📭 No documents found.")
                return
                
            # Read metadata file
            metadata_file = docs_dir / "metadata.json"
            if not metadata_file.exists():
                print("📭 No documents found.")
                return
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            print(f"\n📋 DOCUMENTS IN '{self.current_session.title}':")
            
            for doc_id, doc_info in metadata.items():
                filename = doc_info.get('filename', 'Unknown')
                created_at = doc_info.get('created_at', 'Unknown')
                size = doc_info.get('size_bytes', 0)
                
                print(f"• {filename}")
                print(f"  ID: {doc_id}")
                print(f"  Created: {created_at}")
                print(f"  Size: {size} bytes")
                print()
                
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            
    async def view_document(self):
        """View a specific document."""
        await self.list_documents()
        
        doc_id = input("\nEnter document ID to view: ").strip()
        if not doc_id:
            return
            
        try:
            doc_file = (self.demo_data_path / "users" / self.current_user / 
                       self.current_session.session_id / "documents" / f"{doc_id}.txt")
            
            if not doc_file.exists():
                print(f"❌ Document '{doc_id}' not found!")
                return
                
            with open(doc_file, 'r') as f:
                content = f.read()
                
            print(f"\n📄 DOCUMENT CONTENT:")
            print("-" * 60)
            print(content[:1000])
            if len(content) > 1000:
                print("\n... (truncated)")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error viewing document: {e}")
            
    async def search_menu(self):
        """Handle search operations."""
        if not self.current_user:
            print("❌ Please select a user first!")
            return
            
        while True:
            print(f"\n🔍 SEARCH & RETRIEVAL")
            print("1. Search messages")
            print("2. Search documents")
            print("3. Search all content")
            print("4. Vector similarity search")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                await self.search_messages()
            elif choice == "2":
                await self.search_documents()
            elif choice == "3":
                await self.search_all()
            elif choice == "4":
                await self.vector_search()
            elif choice == "0":
                break
            else:
                print("❌ Invalid choice!")
                
    async def search_messages(self):
        """Search in messages."""
        query_text = input("\n🔍 Enter search query: ").strip()
        if not query_text:
            return
            
        try:
            search_query = FFSearchQueryDTO(
                query_text=query_text,
                user_id=self.current_user,
                session_id=self.current_session.session_id if self.current_session else None,
                include_messages=True,
                include_documents=False
            )
            
            results = await self.search_engine.search(search_query)
            
            print(f"\n📝 SEARCH RESULTS: Found {len(results.results)} matches")
            print("-" * 60)
            
            for i, result in enumerate(results.results[:5], 1):  # Show top 5
                print(f"[{i}] Score: {result.score:.3f}")  
                print(f"    Type: {result.result_type}")
                print(f"    Content: {result.content[:150]}...")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
                print()
                
        except Exception as e:
            print(f"❌ Error searching: {e}")
            
    async def search_documents(self):
        """Search in documents."""
        query_text = input("\n🔍 Enter search query: ").strip()
        if not query_text:
            return
            
        try:
            search_query = FFSearchQueryDTO(
                query_text=query_text,
                user_id=self.current_user,
                session_id=self.current_session.session_id if self.current_session else None,
                include_messages=False,
                include_documents=True
            )
            
            results = await self.search_engine.search(search_query)
            
            print(f"\n📄 DOCUMENT SEARCH RESULTS: Found {len(results.results)} matches")
            print("-" * 60)
            
            for i, result in enumerate(results.results[:5], 1):  # Show top 5
                print(f"[{i}] Score: {result.score:.3f}")
                print(f"    Type: {result.result_type}")
                print(f"    Content: {result.content[:150]}...")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
                print()
                
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            
    async def search_all(self):
        """Search all content."""
        query_text = input("\n🔍 Enter search query: ").strip()
        if not query_text:
            return
            
        try:
            search_query = FFSearchQueryDTO(
                query_text=query_text,
                user_id=self.current_user,
                session_id=self.current_session.session_id if self.current_session else None,
                include_messages=True,
                include_documents=True
            )
            
            results = await self.search_engine.search(search_query)
            
            print(f"\n🔍 ALL CONTENT SEARCH: Found {len(results.results)} matches")
            print("-" * 60)
            
            for i, result in enumerate(results.results[:10], 1):  # Show top 10
                print(f"[{i}] Score: {result.score:.3f} | Type: {result.result_type}")
                print(f"    Content: {result.content[:120]}...")
                print()
                
        except Exception as e:
            print(f"❌ Error searching: {e}")
            
    async def vector_search(self):
        """Perform vector similarity search.""" 
        if not self.current_session:
            print("❌ Please select a session first!")
            return
            
        query_text = input("\n🔢 Enter text for similarity search: ").strip()
        if not query_text:
            return
            
        print("ℹ️  Note: This demo uses mock embeddings for illustration.")
        
        try:
            # Create mock embedding (in real usage, you'd use a proper embedding model)
            import numpy as np
            hash_val = hash(query_text) % (2**31)
            np.random.seed(hash_val)
            query_vector = np.random.normal(0, 1, 384).tolist()
            
            results = await self.vector_storage.search_similar(
                session_id=self.current_session.session_id,
                query_vector=query_vector,
                top_k=5
            )
            
            print(f"\n🔢 VECTOR SEARCH RESULTS: Found {len(results)} similar items")
            print("-" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] Similarity: {result.similarity:.3f}")
                print(f"    Text: {result.text[:120]}...")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
                print()
                
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            
    async def statistics_menu(self):
        """Show statistics and info."""
        print(f"\n📊 STATISTICS & INFORMATION")
        
        try:
            # Basic stats
            if self.current_user and self.current_session:
                stats = await self.storage_manager.get_session_stats(
                    self.current_session.session_id, 
                    self.current_user
                )
                
                print(f"Current Session: {self.current_session.title}")
                print(f"  Messages: {stats.get('message_count', 0)}")
                print(f"  Documents: {stats.get('document_count', 0)}")
                print(f"  Total Size: {stats.get('total_size_bytes', 0)} bytes")
                print()
                
            # Directory size
            total_size = self.get_directory_size(self.demo_data_path)
            file_count = sum([len(files) for r, d, files in os.walk(self.demo_data_path)])
            
            print(f"Storage Statistics:")
            print(f"  Total Size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
            print(f"  Total Files: {file_count}")
            print(f"  Storage Path: {self.demo_data_path}")
            
            # Configuration info
            print(f"\nConfiguration:")
            print(f"  Compression: {self.config.enable_compression}")
            print(f"  File Locking: {self.config.enable_file_locking}")
            print(f"  Max Message Size: {self.config.max_message_size_bytes} bytes")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            
        input("\nPress Enter to continue...")
        
    def get_directory_size(self, path):
        """Calculate total size of directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
        
    def configuration_menu(self):
        """Show configuration information."""
        print(f"\n⚙️ CONFIGURATION SYSTEM")
        
        # Legacy config
        print("Legacy Configuration:")
        print(f"  Base Path: {self.config.storage_base_path}")
        print(f"  Compression: {self.config.enable_compression}")
        print(f"  File Locking: {self.config.enable_file_locking}")
        print(f"  Max Message Size: {self.config.max_message_size_bytes} bytes")
        
        # Try new config system
        try:
            from ff_class_configs.ff_configuration_manager_config import FFConfigurationManagerConfigDTO
            new_config = FFConfigurationManagerConfigDTO.from_environment("development")
            
            print(f"\nNew Modular Configuration:")
            print(f"  Environment: {new_config.environment}")
            print(f"  Storage Path: {new_config.storage.base_path}")
            print(f"  Search Limit: {new_config.search.default_limit}")
            print(f"  Vector Provider: {new_config.vector.default_embedding_provider}")
            
        except ImportError:
            print(f"\nNew configuration system not available.")
            
        input("\nPress Enter to continue...")
        
    def file_explorer(self):
        """Simple file explorer."""
        print(f"\n📁 FILE EXPLORER")
        print(f"Base Path: {self.demo_data_path}")
        
        self.print_directory_tree(self.demo_data_path)
        input("\nPress Enter to continue...")
        
    def print_directory_tree(self, path, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure.""" 
        if current_depth > max_depth:
            return
            
        path = Path(path)
        if not path.exists():
            return
            
        items = list(path.iterdir())
        items.sort(key=lambda x: (x.is_file(), x.name))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            
            if item.is_file():
                size = item.stat().st_size
                print(f"{prefix}{current_prefix}{item.name} ({size} bytes)")
            else:
                print(f"{prefix}{current_prefix}{item.name}/")
                
                if current_depth < max_depth:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    self.print_directory_tree(item, next_prefix, max_depth, current_depth + 1)
                    
    async def run_quick_test(self):
        """Run a quick test of all major features."""
        print("\n🧪 RUNNING QUICK TEST...")
        
        try:
            # Create test user
            test_user = FFUserProfile(
                user_id="test_user",
                username="Test User",
                metadata={"role": "tester"}
            )
            await self.storage_manager.store_user_profile(test_user)
            print("✅ Created test user")
            
            # Create test session
            session_id = await self.storage_manager.create_session(
                user_id="test_user",
                title="Quick Test Session"
            )
            print("✅ Created test session")
            
            # Add test message
            message = FFMessageDTO(
                role=MessageRole.USER,
                content="This is a test message for the quick test."
            )
            await self.storage_manager.add_message("test_user", session_id, message)
            print("✅ Added test message")
            
            # Add test document
            doc_content = "This is a test document containing important information about testing."
            doc_id = await self.storage_manager.save_document(
                user_id="test_user",
                session_id=session_id,
                filename="test_doc.txt",
                content=doc_content.encode('utf-8'),
                metadata={"type": "test"}
            )
            print("✅ Added test document")
            
            # Test search
            search_query = FFSearchQueryDTO(
                query="test",
                user_id="test_user",
                session_ids=[session_id],
                include_documents=True
            )
            results = await self.search_engine.search(search_query)
            print(f"✅ Search test: Found {len(results)} results")
            
            print("\n🎉 Quick test completed successfully!")
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            
        input("Press Enter to continue...")
        
    def cleanup_demo_data(self):
        """Clean up demo data."""
        import shutil
        
        confirm = input(f"\n🧹 Delete all demo data in {self.demo_data_path}? (y/N): ").strip().lower()
        
        if confirm == 'y':
            try:
                shutil.rmtree(self.demo_data_path, ignore_errors=True)
                print("✅ Demo data cleaned up!")
                
                # Reset state
                self.current_user = None
                self.current_session = None
                
                # Recreate directory
                self.demo_data_path.mkdir(exist_ok=True)
                
            except Exception as e:
                print(f"❌ Error cleaning up: {e}")
        else:
            print("Cleanup cancelled.")
            
        input("Press Enter to continue...")
        
    async def run(self):
        """Run the interactive demo."""
        await self.initialize()
        
        while True:
            try:
                self.display_menu()
                choice = input("Enter choice: ").strip()
                
                if choice == "1":
                    await self.user_management_menu()
                elif choice == "2":
                    await self.chat_sessions_menu()
                elif choice == "3":
                    await self.document_management_menu()
                elif choice == "4":
                    await self.search_menu()
                elif choice == "5":
                    await self.statistics_menu()
                elif choice == "6":
                    self.configuration_menu()
                elif choice == "7":
                    self.file_explorer()
                elif choice == "8":
                    await self.run_quick_test()
                elif choice == "9":
                    self.cleanup_demo_data()
                elif choice == "0":
                    print("\n👋 Thanks for trying Flatfile Chat Database!")
                    break
                else:
                    print("❌ Invalid choice! Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                input("Press Enter to continue...")


async def main():
    """Main entry point."""
    print("🚀 Welcome to Flatfile Chat Database Interactive Demo!")
    print("This demo will let you explore all features through a menu interface.")
    print()
    
    demo = InteractiveDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())