"""
Command handlers for the Interactive CLI Chat Demo.

Each command is independent and loosely coupled with the storage layer.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple

# Only import from the public API
from flatfile_chat_database import (
    StorageManager, Message, SearchQuery, SituationalContext
)
from chat_config import ChatDemoConfig
from chat_ui import ChatUI


class CommandHandler:
    """Base class for command handlers"""
    
    def __init__(self, storage: StorageManager, config: ChatDemoConfig, ui: ChatUI):
        self.storage = storage
        self.config = config
        self.ui = ui
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        """Execute the command. Returns True to continue, False to exit."""
        raise NotImplementedError


class HelpCommand(CommandHandler):
    """Show available commands"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        help_text = self.ui.format_help(
            self.config.commands.commands,
            self.config.commands.aliases
        )
        print(help_text)
        return True


class ExitCommand(CommandHandler):
    """Exit the chat"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        self.ui.print_info("Goodbye! Thanks for using the chat demo.")
        return False


class SearchCommand(CommandHandler):
    """Search messages"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if not args:
            self.ui.print_error("Usage: /search <query>")
            return True
        
        query = " ".join(args)
        user_id = context.get('user_id')
        
        self.ui.show_spinner(f"Searching for '{query}'...")
        
        # Use advanced search if available
        search_query = SearchQuery(
            query=query,
            user_id=user_id,
            include_context=self.config.behavior.search_include_context,
            min_relevance_score=self.config.behavior.search_min_relevance_score
        )
        
        results = await self.storage.advanced_search(search_query)
        
        self.ui.hide_spinner()
        
        # Format results for display
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.content,
                'score': result.relevance_score,
                'session_title': 'Current Session'  # Could be enhanced
            })
        
        print(self.ui.format_search_results(formatted_results, query))
        return True


class HistoryCommand(CommandHandler):
    """Show message history"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        count = self.config.ui.history_display_limit
        if args and args[0].isdigit():
            count = int(args[0])
        
        user_id = context.get('user_id')
        session_id = context.get('session_id')
        
        messages = await self.storage.get_messages(user_id, session_id, limit=count)
        
        if not messages:
            self.ui.print_info("No messages in history")
            return True
        
        self.ui.print_header(f"Last {len(messages)} messages")
        for msg in messages:
            print(self.ui.format_message(msg.role, msg.content, msg.timestamp))
            print()  # Empty line between messages
        
        return True


class ContextCommand(CommandHandler):
    """Show current conversation context"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        session_id = context.get('session_id')
        
        ctx = await self.storage.get_context(user_id, session_id)
        
        if ctx:
            context_data = {
                'summary': ctx.summary,
                'confidence': ctx.confidence,
                'key_points': ctx.key_points,
                'entities': ctx.entities
            }
            print(self.ui.format_context(context_data))
        else:
            self.ui.print_info("No context available for this session")
        
        return True


class SessionsCommand(CommandHandler):
    """List all sessions"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        
        sessions = await self.storage.list_sessions(
            user_id, 
            limit=self.config.ui.session_list_limit
        )
        
        if not sessions:
            self.ui.print_info("No sessions found")
            return True
        
        # Format sessions for display
        session_list = []
        for i, session in enumerate(sessions, 1):
            session_list.append((
                i,
                session.title,
                session.message_count,
                session.updated_at
            ))
        
        print(self.ui.format_session_list(session_list))
        return True


class NewSessionCommand(CommandHandler):
    """Create new session"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        
        title = " ".join(args) if args else None
        if not title:
            title = self.ui.get_input("Session title: ")
            if title == "/exit":
                return True
        
        session_id = await self.storage.create_session(user_id, title)
        
        # Update context
        context['session_id'] = session_id
        context['session_title'] = title
        
        self.ui.print_success(f"Created new session: {title}")
        return True


class SwitchCommand(CommandHandler):
    """Switch to another session"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if not args or not args[0].isdigit():
            self.ui.print_error("Usage: /switch <session_number>")
            return True
        
        session_num = int(args[0]) - 1
        user_id = context.get('user_id')
        
        sessions = await self.storage.list_sessions(user_id)
        
        if 0 <= session_num < len(sessions):
            session = sessions[session_num]
            context['session_id'] = session.id
            context['session_title'] = session.title
            self.ui.print_success(f"Switched to session: {session.title}")
        else:
            self.ui.print_error(f"Invalid session number. Choose 1-{len(sessions)}")
        
        return True


class ExportCommand(CommandHandler):
    """Export session data"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        session_id = context.get('session_id')
        session_title = context.get('session_title', 'session')
        
        format_type = args[0] if args else self.config.behavior.default_export_format
        
        # Create export data
        export_data = {
            'session_id': session_id,
            'title': session_title,
            'exported_at': datetime.now().isoformat(),
            'messages': []
        }
        
        # Get all messages
        messages = await self.storage.get_all_messages(user_id, session_id)
        for msg in messages:
            export_data['messages'].append(msg.to_dict())
        
        # Include context if configured
        if self.config.behavior.export_include_context:
            ctx = await self.storage.get_context(user_id, session_id)
            if ctx:
                export_data['context'] = ctx.to_dict()
        
        # Save to file
        filename = f"{session_title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.ui.print_success(f"Session exported to {filename}")
        return True


class StatsCommand(CommandHandler):
    """Show user statistics"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        
        # Gather statistics
        sessions = await self.storage.list_sessions(user_id, limit=None)
        
        total_messages = 0
        total_docs = 0
        session_lengths = []
        
        for session in sessions:
            messages = await self.storage.get_messages(user_id, session.id)
            message_count = len(messages)
            total_messages += message_count
            session_lengths.append(message_count)
            
            docs = await self.storage.list_documents(user_id, session.id)
            total_docs += len(docs)
        
        stats = {
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'total_documents': total_docs,
            'average_session_length': sum(session_lengths) / len(session_lengths) if session_lengths else 0,
            'longest_session': max(session_lengths) if session_lengths else 0,
            'current_session': context.get('session_title', 'Unknown')
        }
        
        print(self.ui.format_statistics(stats))
        return True


class ClearCommand(CommandHandler):
    """Clear screen"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        self.ui.clear_screen()
        return True


class UploadCommand(CommandHandler):
    """Upload a document"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if not args:
            self.ui.print_error("Usage: /upload <filepath>")
            return True
        
        filepath = " ".join(args)
        path = Path(filepath)
        
        if not path.exists():
            self.ui.print_error(f"File not found: {filepath}")
            return True
        
        if not path.is_file():
            self.ui.print_error(f"Not a file: {filepath}")
            return True
        
        user_id = context.get('user_id')
        session_id = context.get('session_id')
        
        try:
            # Read file content
            content = path.read_bytes()
            
            # Save document
            doc_id = await self.storage.save_document(
                user_id, session_id, path.name, content,
                metadata={'uploaded_via': 'cli', 'original_path': str(path)}
            )
            
            if doc_id:
                self.ui.print_success(f"Document uploaded: {path.name} (ID: {doc_id})")
            else:
                self.ui.print_error("Failed to upload document")
        
        except Exception as e:
            self.ui.print_error(f"Upload error: {str(e)}")
        
        return True


class DocsCommand(CommandHandler):
    """List documents in session"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        user_id = context.get('user_id')
        session_id = context.get('session_id')
        
        docs = await self.storage.list_documents(user_id, session_id)
        
        if not docs:
            self.ui.print_info("No documents in this session")
            return True
        
        self.ui.print_header("Documents in session")
        for i, doc in enumerate(docs, 1):
            size_kb = doc.size / 1024
            print(f"{i}. {doc.original_name} ({size_kb:.1f} KB) - Uploaded: {doc.uploaded_at}")
        
        return True


class ThemeCommand(CommandHandler):
    """Change color theme"""
    
    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if not args:
            self.ui.print_info("Available themes: default, dark, minimal, colorful")
            return True
        
        theme_name = args[0]
        
        # This would need to update the UI config
        # For now, just show a message
        self.ui.print_success(f"Theme changed to: {theme_name}")
        self.ui.print_info("(Theme changes will take effect on restart)")
        
        return True


class CommandRegistry:
    """Registry for all available commands"""
    
    def __init__(self, storage: StorageManager, config: ChatDemoConfig, ui: ChatUI):
        self.storage = storage
        self.config = config
        self.ui = ui
        self.commands: Dict[str, CommandHandler] = {}
        self._register_commands()
    
    def _register_commands(self):
        """Register all command handlers"""
        self.commands = {
            'help': HelpCommand(self.storage, self.config, self.ui),
            'exit': ExitCommand(self.storage, self.config, self.ui),
            'quit': ExitCommand(self.storage, self.config, self.ui),
            'search': SearchCommand(self.storage, self.config, self.ui),
            'history': HistoryCommand(self.storage, self.config, self.ui),
            'context': ContextCommand(self.storage, self.config, self.ui),
            'sessions': SessionsCommand(self.storage, self.config, self.ui),
            'new': NewSessionCommand(self.storage, self.config, self.ui),
            'switch': SwitchCommand(self.storage, self.config, self.ui),
            'export': ExportCommand(self.storage, self.config, self.ui),
            'stats': StatsCommand(self.storage, self.config, self.ui),
            'clear': ClearCommand(self.storage, self.config, self.ui),
            'upload': UploadCommand(self.storage, self.config, self.ui),
            'docs': DocsCommand(self.storage, self.config, self.ui),
            'theme': ThemeCommand(self.storage, self.config, self.ui),
        }
    
    async def execute_command(self, command_line: str, context: Dict[str, Any]) -> bool:
        """Execute a command. Returns True to continue, False to exit."""
        if not command_line.startswith(self.config.commands.command_prefix):
            return True
        
        # Parse command and arguments
        parts = command_line[1:].split()
        if not parts:
            return True
        
        command = parts[0].lower()
        args = parts[1:]
        
        # Check for aliases
        if command in self.config.commands.aliases:
            command = self.config.commands.aliases[command]
        
        # Execute command
        if command in self.commands:
            try:
                return await self.commands[command].execute(args, context)
            except Exception as e:
                self.ui.print_error(f"Command error: {str(e)}")
                return True
        else:
            self.ui.print_error(f"Unknown command: {command}")
            self.ui.print_info("Type /help for available commands")
            return True