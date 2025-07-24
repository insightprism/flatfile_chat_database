"""
Configuration for the Interactive CLI Chat Demo.

This configuration is separate from the storage configuration,
demonstrating loose coupling and modularity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChatUIConfig:
    """Configuration for chat UI appearance and behavior"""
    
    # Display settings
    terminal_width: int = 80
    message_wrap_width: int = 70
    show_timestamps: bool = True
    show_message_ids: bool = False
    
    # Colors (ANSI codes)
    color_user: str = '\033[94m'      # Blue
    color_assistant: str = '\033[92m'  # Green
    color_system: str = '\033[93m'     # Yellow
    color_error: str = '\033[91m'      # Red
    color_success: str = '\033[92m'    # Green
    color_info: str = '\033[96m'       # Cyan
    color_reset: str = '\033[0m'
    color_bold: str = '\033[1m'
    color_dim: str = '\033[2m'
    
    # UI elements
    user_prompt: str = "You: "
    assistant_prefix: str = "🤖 Assistant: "
    system_prefix: str = "📋 System: "
    search_prefix: str = "🔍 Search: "
    
    # Display options
    max_search_results: int = 10
    history_display_limit: int = 20
    session_list_limit: int = 10
    truncate_preview_length: int = 50


@dataclass
class ChatCommandConfig:
    """Configuration for chat commands"""
    
    # Command prefix
    command_prefix: str = "/"
    
    # Available commands
    commands: Dict[str, str] = field(default_factory=lambda: {
        "help": "Show available commands",
        "exit": "Exit the chat",
        "quit": "Exit the chat (alias for exit)",
        "search": "Search messages: /search <query>",
        "history": "Show message history: /history [count]",
        "context": "Show current conversation context",
        "sessions": "List all sessions",
        "new": "Create new session",
        "switch": "Switch session: /switch <session_number>",
        "delete": "Delete current session",
        "export": "Export session: /export [format]",
        "import": "Import session: /import <file>",
        "stats": "Show user statistics",
        "clear": "Clear screen",
        "upload": "Upload document: /upload <filepath>",
        "docs": "List documents in session",
        "config": "Show current configuration",
        "multi": "Toggle multi-line mode",
        "theme": "Change color theme: /theme <name>"
    })
    
    # Command aliases
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "quit": "exit",
        "ls": "sessions",
        "?": "help",
        "h": "history",
        "s": "search",
        "ctx": "context",
        "cls": "clear"
    })


@dataclass
class ChatBehaviorConfig:
    """Configuration for chat behavior"""
    
    # Session management
    auto_create_user: bool = True
    default_session_title: str = "New Chat Session"
    confirm_session_delete: bool = True
    auto_save_interval_seconds: int = 30
    
    # Message handling
    multi_line_mode: bool = False
    multi_line_delimiter: str = "```"
    max_message_length: int = 10000
    auto_detect_code_blocks: bool = True
    
    # Search behavior
    search_include_context: bool = True
    search_include_all_sessions: bool = True
    search_min_relevance_score: float = 0.3
    
    # Export/Import
    default_export_format: str = "json"
    export_include_context: bool = True
    export_include_documents: bool = True
    export_compress: bool = True
    
    # Performance
    message_fetch_batch_size: int = 50
    enable_message_streaming: bool = True
    cache_recent_messages: int = 100


@dataclass
class ChatStorageConfig:
    """Configuration for storage integration"""
    
    # Storage settings (passed to StorageConfig)
    storage_base_path: str = "./chat_data"
    
    # These are examples of storage config overrides
    # that the CLI might want to set
    max_messages_per_session: int = 10000
    session_list_default_limit: int = 50
    search_results_default_limit: int = 100


@dataclass
class ChatDemoConfig:
    """Main configuration container for the chat demo"""
    
    # Sub-configurations
    ui: ChatUIConfig = field(default_factory=ChatUIConfig)
    commands: ChatCommandConfig = field(default_factory=ChatCommandConfig)
    behavior: ChatBehaviorConfig = field(default_factory=ChatBehaviorConfig)
    storage: ChatStorageConfig = field(default_factory=ChatStorageConfig)
    
    # Application settings
    app_name: str = "Flatfile Chat Demo"
    app_version: str = "1.0.0"
    welcome_message: str = "Welcome to the Interactive Chat Demo!"
    
    # Feature flags
    enable_ai_simulation: bool = True
    enable_auto_complete: bool = True
    enable_command_history: bool = True
    enable_spell_check: bool = False
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'ChatDemoConfig':
        """Load configuration from JSON file"""
        import json
        from pathlib import Path
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # This would need proper deserialization logic
                return cls()  # Simplified for now
        return cls()
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        import json
        from pathlib import Path
        
        # This would need proper serialization logic
        config_data = {
            "app_name": self.app_name,
            "app_version": self.app_version,
            # ... etc
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Preset themes
THEMES = {
    "default": ChatUIConfig(),
    "dark": ChatUIConfig(
        color_user='\033[36m',      # Cyan
        color_assistant='\033[35m',  # Magenta
        color_system='\033[33m',     # Yellow
    ),
    "minimal": ChatUIConfig(
        user_prompt="> ",
        assistant_prefix="< ",
        system_prefix="* ",
        show_timestamps=False
    ),
    "colorful": ChatUIConfig(
        color_user='\033[95m',      # Bright Magenta
        color_assistant='\033[96m',  # Bright Cyan
        color_system='\033[93m',     # Bright Yellow
        user_prompt="🗣️  ",
        assistant_prefix="🤖 ",
        system_prefix="⚙️  "
    )
}


def get_default_config() -> ChatDemoConfig:
    """Get default configuration"""
    return ChatDemoConfig()


def get_config_with_theme(theme_name: str = "default") -> ChatDemoConfig:
    """Get configuration with specific theme"""
    config = ChatDemoConfig()
    if theme_name in THEMES:
        config.ui = THEMES[theme_name]
    return config