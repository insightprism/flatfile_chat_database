"""
UI utilities for the Interactive CLI Chat Demo.

Provides formatting, colors, and display helpers while maintaining
independence from the storage layer.
"""

import os
import sys
import textwrap
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from chat_config import ChatUIConfig


class ChatUI:
    """UI helper for the chat interface"""
    
    def __init__(self, config: ChatUIConfig):
        self.config = config
        self.terminal_width = shutil.get_terminal_size().columns
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, text: str, char: str = "="):
        """Print a formatted header"""
        print(f"\n{self.config.color_bold}{text}{self.config.color_reset}")
        print(char * min(len(text), self.terminal_width))
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{self.config.color_success}✓ {text}{self.config.color_reset}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{self.config.color_error}✗ {text}{self.config.color_reset}")
    
    def print_info(self, text: str):
        """Print info message"""
        print(f"{self.config.color_info}→ {text}{self.config.color_reset}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{self.config.color_system}⚠ {text}{self.config.color_reset}")
    
    def format_message(self, role: str, content: str, timestamp: Optional[str] = None) -> str:
        """Format a message for display"""
        # Get color based on role
        if role == "user":
            color = self.config.color_user
            prefix = self.config.user_prompt
        elif role == "assistant":
            color = self.config.color_assistant
            prefix = self.config.assistant_prefix
        else:
            color = self.config.color_system
            prefix = self.config.system_prefix
        
        # Format timestamp if needed
        time_str = ""
        if self.config.show_timestamps and timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = f"{self.config.color_dim}[{dt.strftime('%H:%M')}]{self.config.color_reset} "
            except:
                pass
        
        # Wrap content if needed
        if len(content) > self.config.message_wrap_width:
            lines = textwrap.wrap(content, width=self.config.message_wrap_width)
            formatted_lines = [f"{time_str}{color}{prefix}{lines[0]}{self.config.color_reset}"]
            indent = " " * len(prefix)
            for line in lines[1:]:
                formatted_lines.append(f"{indent}{color}{line}{self.config.color_reset}")
            return "\n".join(formatted_lines)
        else:
            return f"{time_str}{color}{prefix}{content}{self.config.color_reset}"
    
    def format_session_list(self, sessions: List[Tuple[int, str, int, str]]) -> str:
        """Format a list of sessions for display"""
        lines = []
        lines.append(f"\n{self.config.color_bold}Available Sessions:{self.config.color_reset}")
        lines.append("─" * 60)
        
        for idx, title, msg_count, last_active in sessions:
            # Format last active time
            try:
                dt = datetime.fromisoformat(last_active)
                time_ago = self._format_time_ago(dt)
            except:
                time_ago = "Unknown"
            
            line = f"{self.config.color_info}{idx}.{self.config.color_reset} "
            line += f"{title:<30} "
            line += f"{self.config.color_dim}({msg_count} messages, {time_ago}){self.config.color_reset}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        lines = []
        lines.append(f"\n{self.config.search_prefix}Found {len(results)} results for '{query}':")
        lines.append("─" * 60)
        
        for i, result in enumerate(results[:self.config.max_search_results], 1):
            # Highlight query in content
            content = result.get('content', '')
            highlighted = self._highlight_text(content, query)
            
            # Truncate if needed
            if len(highlighted) > self.config.truncate_preview_length:
                highlighted = highlighted[:self.config.truncate_preview_length] + "..."
            
            score = result.get('score', 0)
            session = result.get('session_title', 'Unknown Session')
            
            lines.append(f"{self.config.color_info}{i}.{self.config.color_reset} "
                        f"[{session}] {highlighted} "
                        f"{self.config.color_dim}(score: {score:.2f}){self.config.color_reset}")
        
        if len(results) > self.config.max_search_results:
            lines.append(f"{self.config.color_dim}... and {len(results) - self.config.max_search_results} more results{self.config.color_reset}")
        
        return "\n".join(lines)
    
    def format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format user statistics for display"""
        lines = []
        lines.append(f"\n{self.config.color_bold}📊 User Statistics{self.config.color_reset}")
        lines.append("─" * 40)
        
        # Format each stat
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"{self.config.color_info}• {formatted_key}:{self.config.color_reset} {value}")
        
        return "\n".join(lines)
    
    def format_context(self, context: Dict[str, Any]) -> str:
        """Format conversation context for display"""
        lines = []
        lines.append(f"\n{self.config.color_bold}📋 Current Context{self.config.color_reset}")
        lines.append("─" * 50)
        
        summary = context.get('summary', 'No context available')
        confidence = context.get('confidence', 0)
        key_points = context.get('key_points', [])
        entities = context.get('entities', {})
        
        lines.append(f"{self.config.color_info}Summary:{self.config.color_reset} {summary}")
        lines.append(f"{self.config.color_info}Confidence:{self.config.color_reset} {confidence:.0%}")
        
        if key_points:
            lines.append(f"\n{self.config.color_info}Key Points:{self.config.color_reset}")
            for point in key_points:
                lines.append(f"  • {point}")
        
        if entities:
            lines.append(f"\n{self.config.color_info}Entities:{self.config.color_reset}")
            for entity_type, values in entities.items():
                lines.append(f"  • {entity_type}: {', '.join(values)}")
        
        return "\n".join(lines)
    
    def format_help(self, commands: Dict[str, str], aliases: Dict[str, str]) -> str:
        """Format help text"""
        lines = []
        lines.append(f"\n{self.config.color_bold}Available Commands:{self.config.color_reset}")
        lines.append("─" * 60)
        
        # Group commands by category
        categories = {
            "Navigation": ["sessions", "new", "switch", "exit", "quit"],
            "Messages": ["history", "search", "context", "multi"],
            "Data": ["export", "import", "upload", "docs"],
            "Utility": ["help", "clear", "stats", "config", "theme"]
        }
        
        for category, cmd_list in categories.items():
            lines.append(f"\n{self.config.color_info}{category}:{self.config.color_reset}")
            for cmd in cmd_list:
                if cmd in commands:
                    # Check if there's an alias
                    alias_text = ""
                    for alias, target in aliases.items():
                        if target == cmd and alias != cmd:
                            alias_text = f" (alias: /{alias})"
                            break
                    
                    lines.append(f"  /{cmd:<12} - {commands[cmd]}{alias_text}")
        
        return "\n".join(lines)
    
    def get_input(self, prompt: str = None, color: str = None) -> str:
        """Get user input with optional colored prompt"""
        if prompt is None:
            prompt = self.config.user_prompt
        if color is None:
            color = self.config.color_user
        
        try:
            return input(f"{color}{prompt}{self.config.color_reset}")
        except KeyboardInterrupt:
            return "/exit"
        except EOFError:
            return "/exit"
    
    def get_multiline_input(self, prompt: str = None) -> str:
        """Get multi-line input from user"""
        if prompt:
            print(f"{self.config.color_info}{prompt}{self.config.color_reset}")
        
        print(f"{self.config.color_dim}(Enter '{self.config.behavior.multi_line_delimiter}' on a new line to finish){self.config.color_reset}")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == self.config.behavior.multi_line_delimiter:
                    break
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                return "/exit"
        
        return "\n".join(lines)
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format a datetime as 'X ago' string"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 7:
            return dt.strftime("%Y-%m-%d")
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    
    def _highlight_text(self, text: str, query: str) -> str:
        """Highlight query terms in text"""
        # Simple case-insensitive highlighting
        import re
        
        def highlight_match(match):
            return f"{self.config.color_bold}{match.group()}{self.config.color_reset}"
        
        # Escape special regex characters in query
        escaped_query = re.escape(query)
        return re.sub(escaped_query, highlight_match, text, flags=re.IGNORECASE)
    
    def show_spinner(self, message: str = "Loading..."):
        """Show a simple spinner (would need threading for real spinner)"""
        print(f"{self.config.color_info}{message}{self.config.color_reset}", end='', flush=True)
    
    def hide_spinner(self):
        """Hide the spinner"""
        print("\r" + " " * 50 + "\r", end='', flush=True)


class ProgressBar:
    """Simple progress bar for long operations"""
    
    def __init__(self, ui: ChatUI, total: int, description: str = "Progress"):
        self.ui = ui
        self.total = total
        self.current = 0
        self.description = description
        self.width = 40
    
    def update(self, current: int):
        """Update progress bar"""
        self.current = current
        percentage = (current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * current // self.total) if self.total > 0 else 0
        
        bar = "█" * filled + "░" * (self.width - filled)
        print(f"\r{self.description}: [{bar}] {percentage:.1f}%", end='', flush=True)
    
    def finish(self):
        """Finish progress bar"""
        print()  # New line


def create_ui(config: ChatUIConfig) -> ChatUI:
    """Factory function to create UI instance"""
    return ChatUI(config)