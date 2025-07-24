#!/usr/bin/env python3
"""
Interactive CLI Chat Demo for Flatfile Chat Database.

This demonstrates the flatfile chat database with a fully-featured
command-line interface while maintaining loose coupling with the storage layer.
"""

import asyncio
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only import from the public API - demonstrates loose coupling
from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, SituationalContext
)

# Import our CLI-specific modules
from chat_config import ChatDemoConfig, get_config_with_theme
from chat_ui import ChatUI, create_ui
from chat_commands import CommandRegistry


class InteractiveChatDemo:
    """Main interactive chat application"""
    
    def __init__(self, config: ChatDemoConfig):
        """Initialize with configuration"""
        self.config = config
        self.ui = create_ui(config.ui)
        
        # Initialize storage with loose coupling
        storage_config = StorageConfig(
            storage_base_path=config.storage.storage_base_path,
            max_messages_per_session=config.storage.max_messages_per_session,
            session_list_default_limit=config.storage.session_list_default_limit,
            search_results_default_limit=config.storage.search_results_default_limit
        )
        
        self.storage = StorageManager(storage_config)
        self.command_registry = None
        
        # Session context
        self.context: Dict[str, Any] = {
            'user_id': None,
            'session_id': None,
            'session_title': None,
            'message_count': 0
        }
        
        # AI simulation responses (if enabled)
        self.ai_responses = [
            "That's an interesting point! Could you tell me more?",
            "I understand. Let me help you with that.",
            "Great question! Here's what I think...",
            "Thanks for sharing that. Have you considered...",
            "I see what you mean. Another way to look at it is...",
            "That makes sense. Would you like to explore...",
            "Excellent observation! This relates to...",
            "I appreciate your perspective. Let me add...",
        ]
    
    async def initialize(self):
        """Initialize the chat system"""
        await self.storage.initialize()
        self.command_registry = CommandRegistry(self.storage, self.config, self.ui)
    
    async def run(self):
        """Main application loop"""
        # Show welcome
        self._show_welcome()
        
        # User login
        await self._handle_login()
        
        # Session selection
        await self._handle_session_selection()
        
        # Main chat loop
        await self._chat_loop()
    
    def _show_welcome(self):
        """Display welcome message"""
        self.ui.clear_screen()
        self.ui.print_header(f"🗨️  {self.config.app_name} v{self.config.app_version}")
        print(f"\n{self.config.welcome_message}")
        print(f"Storage location: {self.config.storage.storage_base_path}")
        print("\nType /help for available commands\n")
    
    async def _handle_login(self):
        """Handle user login/creation"""
        username = self.ui.get_input("Username: ")
        
        if username == "/exit":
            sys.exit(0)
        
        # Check if user exists
        profile = await self.storage.get_user_profile(username)
        
        if profile:
            self.ui.print_success(f"Welcome back, {profile.get('username', username)}!")
            
            # Show user stats
            sessions = await self.storage.list_sessions(username)
            if sessions:
                print(f"You have {len(sessions)} session(s)")
        else:
            # Create new user
            if self.config.behavior.auto_create_user:
                display_name = self.ui.get_input("Display name (or press Enter to use username): ")
                if not display_name or display_name == "/exit":
                    display_name = username
                
                await self.storage.create_user(username, {
                    'username': display_name,
                    'created_via': 'cli_demo',
                    'preferences': {
                        'theme': 'default',
                        'notifications': True
                    }
                })
                self.ui.print_success(f"Created new user: {display_name}")
            else:
                self.ui.print_error("User not found and auto-create is disabled")
                sys.exit(1)
        
        self.context['user_id'] = username
    
    async def _handle_session_selection(self):
        """Handle session selection or creation"""
        sessions = await self.storage.list_sessions(
            self.context['user_id'],
            limit=self.config.ui.session_list_limit
        )
        
        if sessions:
            # Show existing sessions
            session_list = []
            for i, session in enumerate(sessions, 1):
                # Get last message preview
                messages = await self.storage.get_messages(
                    self.context['user_id'], 
                    session.id, 
                    limit=1
                )
                
                session_list.append((
                    i,
                    session.title,
                    session.message_count,
                    session.updated_at
                ))
            
            print(self.ui.format_session_list(session_list))
            
            choice = self.ui.get_input("\nSelect session (number) or 'n' for new: ")
            
            if choice.lower() == 'n':
                await self._create_new_session()
            elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
                selected = sessions[int(choice) - 1]
                self.context['session_id'] = selected.id
                self.context['session_title'] = selected.title
                self.ui.print_success(f"Resumed session: {selected.title}")
                
                # Show recent messages
                print("\n--- Recent messages ---")
                recent = await self.storage.get_messages(
                    self.context['user_id'],
                    self.context['session_id'],
                    limit=5
                )
                for msg in recent:
                    print(self.ui.format_message(msg.role, msg.content, msg.timestamp))
                print("--- End of recent messages ---\n")
            else:
                self.ui.print_error("Invalid choice")
                await self._handle_session_selection()
        else:
            # No sessions, create new
            await self._create_new_session()
    
    async def _create_new_session(self):
        """Create a new chat session"""
        title = self.ui.get_input("Session title: ")
        
        if title == "/exit":
            sys.exit(0)
        
        if not title:
            title = self.config.behavior.default_session_title
        
        session_id = await self.storage.create_session(
            self.context['user_id'],
            title
        )
        
        self.context['session_id'] = session_id
        self.context['session_title'] = title
        
        self.ui.print_success(f"Created new session: {title}")
    
    async def _chat_loop(self):
        """Main chat interaction loop"""
        self.ui.print_header(f"💬 {self.context['session_title']}")
        print("Type your message or use /help for commands\n")
        
        while True:
            # Get user input
            if self.config.behavior.multi_line_mode:
                user_input = self.ui.get_multiline_input()
            else:
                user_input = self.ui.get_input()
            
            # Handle commands
            if user_input.startswith(self.config.commands.command_prefix):
                continue_chat = await self.command_registry.execute_command(
                    user_input, 
                    self.context
                )
                if not continue_chat:
                    break
                continue
            
            # Handle regular messages
            await self._handle_message(user_input)
    
    async def _handle_message(self, user_input: str):
        """Handle a regular chat message"""
        # Save user message
        user_msg = Message(role="user", content=user_input)
        await self.storage.add_message(
            self.context['user_id'],
            self.context['session_id'],
            user_msg
        )
        
        self.context['message_count'] += 1
        
        # Show user message (already displayed via input)
        
        # Generate and save assistant response (if AI simulation enabled)
        if self.config.enable_ai_simulation:
            # Simulate thinking time
            self.ui.show_spinner("Thinking...")
            await asyncio.sleep(random.uniform(0.5, 1.5))
            self.ui.hide_spinner()
            
            # Generate response
            response = await self._generate_response(user_input)
            
            # Save assistant message
            assistant_msg = Message(role="assistant", content=response)
            await self.storage.add_message(
                self.context['user_id'],
                self.context['session_id'],
                assistant_msg
            )
            
            self.context['message_count'] += 1
            
            # Display assistant response
            print(self.ui.format_message("assistant", response))
        
        # Update context periodically
        if self.context['message_count'] % 10 == 0:
            await self._update_context()
    
    async def _generate_response(self, user_input: str) -> str:
        """Generate a simulated AI response"""
        # Simple response selection based on input
        if "?" in user_input:
            responses = [
                "That's a great question! Let me think about that...",
                "Interesting question. Here's my perspective...",
                "I'm glad you asked! The answer depends on...",
            ]
        elif any(word in user_input.lower() for word in ['help', 'how', 'what', 'why']):
            responses = [
                "I'd be happy to help with that!",
                "Let me explain how that works...",
                "Here's what you need to know...",
            ]
        else:
            responses = self.ai_responses
        
        base_response = random.choice(responses)
        
        # Add context-aware elements
        if "python" in user_input.lower():
            base_response += " Python is a versatile language with many applications."
        elif "code" in user_input.lower():
            base_response += " Would you like me to show you a code example?"
        
        return base_response
    
    async def _update_context(self):
        """Update conversation context"""
        # Get recent messages for context
        recent_messages = await self.storage.get_messages(
            self.context['user_id'],
            self.context['session_id'],
            limit=20
        )
        
        if not recent_messages:
            return
        
        # Simple context generation
        topics = set()
        for msg in recent_messages:
            # Extract potential topics (simplified)
            words = msg.content.lower().split()
            for word in words:
                if len(word) > 5 and word.isalpha():
                    topics.add(word)
        
        # Create context
        context = SituationalContext(
            summary=f"Conversation about {', '.join(list(topics)[:3])}",
            key_points=[f"Discussed {len(topics)} topics"],
            entities={'topics': list(topics)[:10]},
            confidence=0.7
        )
        
        await self.storage.update_context(
            self.context['user_id'],
            self.context['session_id'],
            context
        )


async def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Chat Demo")
    parser.add_argument("--theme", choices=["default", "dark", "minimal", "colorful"], 
                       default="default", help="UI theme")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--storage-path", help="Override storage path")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI simulation")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ChatDemoConfig.load_from_file(args.config)
    else:
        config = get_config_with_theme(args.theme)
    
    # Apply command line overrides
    if args.storage_path:
        config.storage.storage_base_path = args.storage_path
    
    if args.no_ai:
        config.enable_ai_simulation = False
    
    # Create and run application
    app = InteractiveChatDemo(config)
    await app.initialize()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThank you for using the Flatfile Chat Demo!")


if __name__ == "__main__":
    asyncio.run(main())