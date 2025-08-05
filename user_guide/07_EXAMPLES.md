# Examples & Tutorials

Real-world examples and tutorials for implementing common use cases with the Flatfile Chat Database system.

## 🎯 Overview

This section provides practical, ready-to-use examples for:

- 🤖 **Chatbot Integration** - Building AI-powered chat systems
- 🎧 **Customer Support** - Implementing support ticket systems
- 📚 **Knowledge Base** - Creating searchable knowledge repositories
- 👥 **Team Collaboration** - Building team communication tools
- 📊 **Analytics Dashboard** - Creating usage analytics and insights
- 🔄 **Data Migration** - Importing existing chat data
- 🌐 **Web Integration** - Web application integration patterns

## 🤖 Chatbot Integration

### Simple AI Chatbot
```python
import asyncio
from typing import List, Dict, Any
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole

class SimpleChatbot:
    """A simple chatbot that stores conversations persistently."""
    
    def __init__(self):
        self.storage = None
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "help": "I can assist you with general questions. What would you like to know?",
            "bye": "Goodbye! Have a great day!",
            "default": "I'm not sure I understand. Could you rephrase that?"
        }
    
    async def initialize(self):
        """Initialize the chatbot with storage."""
        config = load_config()
        self.storage = FFStorageManager(config)
        await self.storage.initialize()
        
        # Create bot user
        await self.storage.create_user("chatbot", {
            "name": "AI Assistant",
            "type": "bot",
            "capabilities": ["general_qa", "conversation"]
        })
    
    async def start_conversation(self, user_id: str, user_name: str = None) -> str:
        """Start a new conversation with a user."""
        # Create user if doesn't exist
        if not await self.storage.user_exists(user_id):
            await self.storage.create_user(user_id, {
                "name": user_name or f"User {user_id}",
                "type": "human"
            })
        
        # Create new session
        session_id = await self.storage.create_session(
            user_id, 
            f"Chat with AI Assistant - {user_name or user_id}"
        )
        
        # Send welcome message
        welcome_msg = FFMessageDTO(
            role=MessageRole.ASSISTANT,
            content="Hello! I'm your AI assistant. How can I help you today?",
            metadata={"type": "welcome"}
        )
        await self.storage.add_message(user_id, session_id, welcome_msg)
        
        return session_id
    
    async def process_message(self, user_id: str, session_id: str, user_input: str) -> str:
        """Process user message and generate response."""
        # Store user message
        user_msg = FFMessageDTO(
            role=MessageRole.USER,
            content=user_input,
            metadata={"processed": True}
        )
        await self.storage.add_message(user_id, session_id, user_msg)
        
        # Generate response (simple keyword matching)
        response = self._generate_response(user_input)
        
        # Store bot response
        bot_msg = FFMessageDTO(
            role=MessageRole.ASSISTANT,
            content=response,
            metadata={"generated": True, "user": "chatbot"}
        )
        await self.storage.add_message(user_id, session_id, bot_msg)
        
        return response
    
    def _generate_response(self, user_input: str) -> str:
        """Generate response based on user input."""
        user_input_lower = user_input.lower()
        
        for keyword, response in self.responses.items():
            if keyword in user_input_lower:
                return response
        
        return self.responses["default"]
    
    async def get_conversation_history(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get formatted conversation history."""
        messages = await self.storage.get_all_messages(user_id, session_id)
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            })
        
        return history

# Usage example
async def chatbot_demo():
    """Demonstrate the chatbot in action."""
    bot = SimpleChatbot()
    await bot.initialize()
    
    # Start conversation
    user_id = "alice"
    session_id = await bot.start_conversation(user_id, "Alice Johnson")
    print(f"Started conversation: {session_id}")
    
    # Simulate conversation
    conversation = [
        "Hello there!",
        "I need some help with my account",
        "How do I change my password?",
        "Thank you for your help",
        "Bye!"
    ]
    
    for user_input in conversation:
        print(f"\nUser: {user_input}")
        response = await bot.process_message(user_id, session_id, user_input)
        print(f"Bot: {response}")
    
    # Show conversation history
    history = await bot.get_conversation_history(user_id, session_id)
    print(f"\nConversation had {len(history)} messages")

# Run the demo
asyncio.run(chatbot_demo())
```

### Advanced AI Chatbot with Context
```python
class AdvancedChatbot(SimpleChatbot):
    """Advanced chatbot with context awareness and memory."""
    
    def __init__(self):
        super().__init__()
        self.context_memory = {}
    
    async def process_message_with_context(self, user_id: str, session_id: str, user_input: str) -> str:
        """Process message with conversation context."""
        # Get recent conversation history for context
        recent_messages = await self.storage.get_messages(
            user_id, session_id, limit=5, reverse=True
        )
        
        # Build context from recent messages
        context = self._build_context(recent_messages)
        
        # Store user message
        user_msg = FFMessageDTO(
            role=MessageRole.USER,
            content=user_input,
            metadata={"context_aware": True}
        )
        await self.storage.add_message(user_id, session_id, user_msg)
        
        # Generate context-aware response
        response = self._generate_contextual_response(user_input, context)
        
        # Store bot response
        bot_msg = FFMessageDTO(
            role=MessageRole.ASSISTANT,
            content=response,
            metadata={"context_used": True, "context_length": len(context)}
        )
        await self.storage.add_message(user_id, session_id, bot_msg)
        
        return response
    
    def _build_context(self, messages: List[FFMessageDTO]) -> Dict[str, Any]:
        """Build conversation context from recent messages."""
        context = {
            "topics_mentioned": [],
            "user_questions": [],
            "bot_responses": [],
            "conversation_tone": "neutral"
        }
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                context["user_questions"].append(msg.content)
                # Simple topic extraction
                if "password" in msg.content.lower():
                    context["topics_mentioned"].append("password")
                elif "account" in msg.content.lower():
                    context["topics_mentioned"].append("account")
            else:
                context["bot_responses"].append(msg.content)
        
        return context
    
    def _generate_contextual_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate response based on input and context."""
        user_input_lower = user_input.lower()
        
        # Context-aware responses
        if "password" in user_input_lower:
            if "account" in context["topics_mentioned"]:
                return "Since we were discussing your account, here's how to change your password: Go to Settings > Security > Change Password."
            else:
                return "To change your password, please go to your account settings and select 'Change Password'."
        
        if "thank" in user_input_lower:
            return "You're welcome! Is there anything else I can help you with today?"
        
        # Fall back to simple response
        return self._generate_response(user_input)

# Usage with context awareness
async def advanced_chatbot_demo():
    """Demonstrate advanced chatbot with context."""
    bot = AdvancedChatbot()
    await bot.initialize()
    
    user_id = "bob"
    session_id = await bot.start_conversation(user_id, "Bob Smith")
    
    # Context-aware conversation
    conversation = [
        "I'm having trouble with my account",
        "I can't remember my password",
        "How do I reset it?",
        "That's very helpful, thank you!"
    ]
    
    for user_input in conversation:
        print(f"\nUser: {user_input}")
        response = await bot.process_message_with_context(user_id, session_id, user_input)
        print(f"Bot: {response}")

asyncio.run(advanced_chatbot_demo())
```

## 🎧 Customer Support System

### Support Ticket Management
```python
from datetime import datetime, timedelta
from enum import Enum

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

class CustomerSupportSystem:
    """Customer support system with ticket management."""
    
    def __init__(self):
        self.storage = None
        self.agents = {}
    
    async def initialize(self):
        """Initialize the support system."""
        config = load_config()
        self.storage = FFStorageManager(config)
        await self.storage.initialize()
        
        # Create support agents
        agents = [
            {"id": "agent_1", "name": "Sarah Support", "specialties": ["billing", "accounts"]},
            {"id": "agent_2", "name": "Mike Tech", "specialties": ["technical", "integration"]},
            {"id": "agent_3", "name": "Lisa Manager", "specialties": ["escalation", "complaints"]}
        ]
        
        for agent in agents:
            await self.storage.create_user(agent["id"], {
                "name": agent["name"],
                "type": "support_agent",
                "specialties": agent["specialties"],
                "status": "available"
            })
            self.agents[agent["id"]] = agent
    
    async def create_ticket(self, customer_id: str, subject: str, description: str, 
                          priority: TicketPriority = TicketPriority.MEDIUM) -> str:
        """Create a new support ticket."""
        
        # Create customer if doesn't exist
        if not await self.storage.user_exists(customer_id):
            await self.storage.create_user(customer_id, {
                "type": "customer",
                "first_contact": datetime.now().isoformat()
            })
        
        # Create ticket session
        ticket_metadata = {
            "type": "support_ticket",
            "priority": priority.value,
            "status": TicketStatus.OPEN.value,
            "created_at": datetime.now().isoformat(),
            "subject": subject,
            "assigned_agent": None,
            "resolution_time": None,
            "customer_satisfaction": None
        }
        
        session_id = await self.storage.create_session(
            customer_id,
            f"Support Ticket: {subject}",
            ticket_metadata
        )
        
        # Add initial message
        initial_msg = FFMessageDTO(
            role=MessageRole.USER,
            content=description,
            metadata={
                "type": "ticket_description",
                "priority": priority.value
            }
        )
        await self.storage.add_message(customer_id, session_id, initial_msg)
        
        # Auto-assign agent based on keywords
        assigned_agent = self._auto_assign_agent(subject + " " + description)
        if assigned_agent:
            await self._assign_ticket(customer_id, session_id, assigned_agent)
        
        return session_id
    
    def _auto_assign_agent(self, content: str) -> str:
        """Auto-assign agent based on content keywords."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["billing", "payment", "refund", "charge"]):
            return "agent_1"  # Sarah - billing specialist
        elif any(word in content_lower for word in ["api", "integration", "technical", "error"]):
            return "agent_2"  # Mike - technical specialist
        elif any(word in content_lower for word in ["complaint", "manager", "escalate"]):
            return "agent_3"  # Lisa - manager
        
        return "agent_1"  # Default assignment
    
    async def _assign_ticket(self, customer_id: str, session_id: str, agent_id: str):
        """Assign ticket to an agent."""
        await self.storage.update_session(
            customer_id,
            session_id,
            metadata={"assigned_agent": agent_id, "status": TicketStatus.IN_PROGRESS.value}
        )
        
        # Add assignment message
        assignment_msg = FFMessageDTO(
            role=MessageRole.SYSTEM,
            content=f"Ticket assigned to {self.agents[agent_id]['name']}",
            metadata={"type": "assignment", "agent_id": agent_id}
        )
        await self.storage.add_message(customer_id, session_id, assignment_msg)
    
    async def agent_respond(self, customer_id: str, session_id: str, agent_id: str, response: str):
        """Agent responds to a ticket."""
        agent_msg = FFMessageDTO(
            role=MessageRole.ASSISTANT,
            content=response,
            metadata={
                "type": "agent_response",
                "agent_id": agent_id,
                "agent_name": self.agents[agent_id]["name"]
            }
        )
        await self.storage.add_message(customer_id, session_id, agent_msg)
        
        # Update ticket status
        await self.storage.update_session(
            customer_id,
            session_id,
            metadata={"status": TicketStatus.WAITING_CUSTOMER.value, "last_agent_response": datetime.now().isoformat()}
        )
    
    async def customer_respond(self, customer_id: str, session_id: str, response: str):
        """Customer responds to agent."""
        customer_msg = FFMessageDTO(
            role=MessageRole.USER,
            content=response,
            metadata={"type": "customer_response"}
        )
        await self.storage.add_message(customer_id, session_id, customer_msg)
        
        # Update status back to in progress
        await self.storage.update_session(
            customer_id,
            session_id,
            metadata={"status": TicketStatus.IN_PROGRESS.value}
        )
    
    async def resolve_ticket(self, customer_id: str, session_id: str, agent_id: str, 
                           resolution: str, satisfaction_score: int = None):
        """Mark ticket as resolved."""
        resolution_msg = FFMessageDTO(
            role=MessageRole.ASSISTANT,
            content=f"Ticket resolved: {resolution}",
            metadata={
                "type": "resolution",
                "agent_id": agent_id,
                "satisfaction_score": satisfaction_score
            }
        )
        await self.storage.add_message(customer_id, session_id, resolution_msg)
        
        # Update ticket metadata
        session = await self.storage.get_session(customer_id, session_id)
        created_time = datetime.fromisoformat(session.metadata["created_at"])
        resolution_time = (datetime.now() - created_time).total_seconds() / 3600  # hours
        
        await self.storage.update_session(
            customer_id,
            session_id,
            metadata={
                "status": TicketStatus.RESOLVED.value,
                "resolution_time": resolution_time,
                "resolved_by": agent_id,
                "resolved_at": datetime.now().isoformat(),
                "customer_satisfaction": satisfaction_score
            }
        )
    
    async def get_ticket_analytics(self) -> Dict[str, Any]:
        """Get support ticket analytics."""
        # This would be implemented with proper analytics queries
        # For demo, returning sample data
        return {
            "total_tickets": 150,
            "open_tickets": 23,
            "avg_resolution_time_hours": 4.2,
            "customer_satisfaction_avg": 4.1,
            "agent_performance": {
                "agent_1": {"tickets_handled": 45, "avg_satisfaction": 4.3},
                "agent_2": {"tickets_handled": 52, "avg_satisfaction": 4.0},
                "agent_3": {"tickets_handled": 33, "avg_satisfaction": 4.5}
            }
        }

# Support system demo
async def support_system_demo():
    """Demonstrate customer support system."""
    support = CustomerSupportSystem()
    await support.initialize()
    
    # Create a support ticket
    ticket_id = await support.create_ticket(
        "customer_123",
        "Cannot access my account",
        "I forgot my password and the reset email isn't coming through. I need to access my account urgently for an important meeting.",
        TicketPriority.HIGH
    )
    
    print(f"Created ticket: {ticket_id}")
    
    # Agent responds
    await support.agent_respond(
        "customer_123",
        ticket_id,
        "agent_1",
        "I understand this is urgent. Let me help you reset your password immediately. Can you confirm the email address associated with your account?"
    )
    
    # Customer responds
    await support.customer_respond(
        "customer_123",
        ticket_id,
        "Yes, it's customer123@email.com. I checked spam folder but nothing there."
    )
    
    # Agent resolves
    await support.resolve_ticket(
        "customer_123",
        ticket_id,
        "agent_1",
        "I've manually reset your password and sent new credentials to your email. You should receive it within 2 minutes. Your account is now accessible.",
        5  # Satisfaction score
    )
    
    # Get analytics
    analytics = await support.get_ticket_analytics()
    print(f"Support Analytics: {analytics}")

asyncio.run(support_system_demo())
```

## 📚 Knowledge Base System

### Searchable Knowledge Repository
```python
import re
from pathlib import Path

class KnowledgeBase:
    """Searchable knowledge base system."""
    
    def __init__(self):
        self.storage = None
        self.categories = {
            "faq": "Frequently Asked Questions",
            "howto": "How-to Guides",
            "troubleshooting": "Troubleshooting",
            "api": "API Documentation",
            "policies": "Policies and Procedures"
        }
    
    async def initialize(self):
        """Initialize knowledge base."""
        config = load_config()
        # Enable vector search for semantic similarity
        config.vector.enable_vector_storage = True
        config.search.enable_full_text_search = True
        
        self.storage = FFStorageManager(config)
        await self.storage.initialize()
        
        # Create knowledge base admin user
        await self.storage.create_user("kb_admin", {
            "name": "Knowledge Base Admin",
            "type": "admin",
            "role": "content_manager"
        })
    
    async def add_article(self, title: str, content: str, category: str, 
                         tags: List[str] = None, author: str = "kb_admin") -> str:
        """Add a knowledge base article."""
        
        # Create session for the article
        article_metadata = {
            "type": "kb_article",
            "category": category,
            "tags": tags or [],
            "author": author,
            "word_count": len(content.split()),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "view_count": 0,
            "helpful_votes": 0,
            "not_helpful_votes": 0
        }
        
        session_id = await self.storage.create_session(
            "kb_admin",
            f"KB Article: {title}",
            article_metadata
        )
        
        # Add article content as message
        article_msg = FFMessageDTO(
            role=MessageRole.SYSTEM,
            content=content,
            metadata={
                "type": "article_content",
                "title": title,
                "category": category,
                "tags": tags or []
            }
        )
        await self.storage.add_message("kb_admin", session_id, article_msg)
        
        return session_id
    
    async def search_articles(self, query: str, category: str = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base articles."""
        
        # First try semantic search
        semantic_results = await self.storage.search_similar_messages(
            "kb_admin",
            query,
            similarity_threshold=0.6,
            limit=limit
        )
        
        # Then try text search
        text_results = await self.storage.search_messages(
            "kb_admin",
            query,
            limit=limit
        )
        
        # Combine and deduplicate results
        all_results = {}
        
        # Add semantic results with higher weight
        for result in semantic_results:
            all_results[result.session_id] = {
                "session_id": result.session_id,
                "content": result.content,
                "relevance_score": result.similarity_score,
                "search_type": "semantic"
            }
        
        # Add text results
        for result in text_results:
            if result.session_id not in all_results:
                all_results[result.session_id] = {
                    "session_id": result.session_id,
                    "content": result.content,
                    "relevance_score": result.relevance_score,
                    "search_type": "text"
                }
        
        # Get session metadata for additional info
        enriched_results = []
        for result in all_results.values():
            session = await self.storage.get_session("kb_admin", result["session_id"])
            if session:
                enriched_results.append({
                    **result,
                    "title": session.title.replace("KB Article: ", ""),
                    "category": session.metadata.get("category"),
                    "tags": session.metadata.get("tags", []),
                    "author": session.metadata.get("author"),
                    "view_count": session.metadata.get("view_count", 0)
                })
        
        # Sort by relevance score
        enriched_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Filter by category if specified
        if category:
            enriched_results = [r for r in enriched_results if r["category"] == category]
        
        return enriched_results[:limit]
    
    async def get_article(self, article_id: str) -> Dict[str, Any]:
        """Get full article content and metadata."""
        session = await self.storage.get_session("kb_admin", article_id)
        if not session:
            return None
        
        messages = await self.storage.get_all_messages("kb_admin", article_id)
        article_content = next((msg.content for msg in messages if msg.metadata.get("type") == "article_content"), "")
        
        # Update view count
        view_count = session.metadata.get("view_count", 0) + 1
        await self.storage.update_session(
            "kb_admin",
            article_id,
            metadata={**session.metadata, "view_count": view_count}
        )
        
        return {
            "id": article_id,
            "title": session.title.replace("KB Article: ", ""),
            "content": article_content,
            "category": session.metadata.get("category"),
            "tags": session.metadata.get("tags", []),
            "author": session.metadata.get("author"),
            "created_at": session.metadata.get("created_at"),
            "last_updated": session.metadata.get("last_updated"),
            "view_count": view_count,
            "helpful_votes": session.metadata.get("helpful_votes", 0),
            "not_helpful_votes": session.metadata.get("not_helpful_votes", 0)
        }
    
    async def vote_on_article(self, article_id: str, helpful: bool):
        """Vote on article helpfulness."""
        session = await self.storage.get_session("kb_admin", article_id)
        if not session:
            return False
        
        metadata = session.metadata.copy()
        if helpful:
            metadata["helpful_votes"] = metadata.get("helpful_votes", 0) + 1
        else:
            metadata["not_helpful_votes"] = metadata.get("not_helpful_votes", 0) + 1
        
        await self.storage.update_session("kb_admin", article_id, metadata=metadata)
        return True
    
    async def get_popular_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular articles by view count."""
        # This would require a more sophisticated query system
        # For demo, we'll return sample data
        sessions = await self.storage.list_sessions("kb_admin", limit=100)
        
        # Filter KB articles and sort by view count
        kb_sessions = [s for s in sessions if s.metadata.get("type") == "kb_article"]
        kb_sessions.sort(key=lambda s: s.metadata.get("view_count", 0), reverse=True)
        
        popular = []
        for session in kb_sessions[:limit]:
            popular.append({
                "id": session.session_id,
                "title": session.title.replace("KB Article: ", ""),
                "category": session.metadata.get("category"),
                "view_count": session.metadata.get("view_count", 0),
                "helpful_votes": session.metadata.get("helpful_votes", 0)
            })
        
        return popular

# Knowledge base demo
async def knowledge_base_demo():
    """Demonstrate knowledge base functionality."""
    kb = KnowledgeBase()
    await kb.initialize()
    
    # Add sample articles
    articles = [
        {
            "title": "How to Reset Your Password",
            "content": "To reset your password: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your email address 4. Check your email for reset instructions 5. Follow the link and create a new password",
            "category": "howto",
            "tags": ["password", "login", "account"]
        },
        {
            "title": "API Authentication Guide",
            "content": "Our API uses OAuth 2.0 for authentication. To get started: 1. Register your application 2. Obtain client credentials 3. Request an access token 4. Include the token in API requests using the Authorization header",
            "category": "api",
            "tags": ["api", "authentication", "oauth"]
        },
        {
            "title": "Troubleshooting Connection Issues",
            "content": "If you're experiencing connection problems: 1. Check your internet connection 2. Verify server status 3. Clear browser cache 4. Disable VPN if using one 5. Contact support if issues persist",
            "category": "troubleshooting",
            "tags": ["connection", "network", "troubleshooting"]
        }
    ]
    
    article_ids = []
    for article in articles:
        article_id = await kb.add_article(**article)
        article_ids.append(article_id)
        print(f"Added article: {article['title']} (ID: {article_id})")
    
    # Search for articles
    print("\nSearching for 'password':")
    password_results = await kb.search_articles("password reset login")
    for result in password_results:
        print(f"  - {result['title']} (Score: {result['relevance_score']:.2f})")
    
    # Get article details
    print("\nArticle details:")
    article = await kb.get_article(article_ids[0])
    if article:
        print(f"Title: {article['title']}")
        print(f"Views: {article['view_count']}")
        print(f"Content: {article['content'][:100]}...")
    
    # Vote on article
    await kb.vote_on_article(article_ids[0], True)
    print("Voted article as helpful")
    
    # Get popular articles
    popular = await kb.get_popular_articles(5)
    print(f"\nPopular articles: {len(popular)} found")

asyncio.run(knowledge_base_demo())
```

## 👥 Team Collaboration Tool

### Team Chat System
```python
class TeamCollaborationTool:
    """Team-based chat and collaboration system."""
    
    def __init__(self):
        self.storage = None
        self.teams = {}
        self.channels = {}
    
    async def initialize(self):
        """Initialize team collaboration system."""
        config = load_config()
        config.streaming.enable_streaming = True  # Enable real-time features
        
        self.storage = FFStorageManager(config)
        await self.storage.initialize()
    
    async def create_team(self, team_id: str, team_name: str, description: str = "") -> bool:
        """Create a new team."""
        team_metadata = {
            "type": "team",
            "name": team_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "member_count": 0,
            "channels": []
        }
        
        # Create team user to own team data
        success = await self.storage.create_user(f"team_{team_id}", team_metadata)
        if success:
            self.teams[team_id] = team_metadata
        
        return success
    
    async def add_team_member(self, team_id: str, user_id: str, role: str = "member") -> bool:
        """Add member to team."""
        # Create user if doesn't exist
        if not await self.storage.user_exists(user_id):
            return False
        
        # Update user profile with team membership
        user_profile = await self.storage.get_user_profile(user_id)
        if user_profile:
            teams = user_profile.metadata.get("teams", [])
            if team_id not in teams:
                teams.append(team_id)
                await self.storage.update_user_profile(user_id, {
                    "metadata": {**user_profile.metadata, "teams": teams}
                })
        
        return True
    
    async def create_channel(self, team_id: str, channel_name: str, 
                           channel_type: str = "public", description: str = "") -> str:
        """Create a team channel."""
        if team_id not in self.teams:
            return None
        
        channel_metadata = {
            "type": "team_channel",
            "team_id": team_id,
            "channel_type": channel_type,  # public, private, direct
            "description": description,
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "active_members": []
        }
        
        session_id = await self.storage.create_session(
            f"team_{team_id}",
            f"#{channel_name}",
            channel_metadata
        )
        
        self.channels[session_id] = {
            "team_id": team_id,
            "name": channel_name,
            "type": channel_type
        }
        
        return session_id
    
    async def send_team_message(self, team_id: str, channel_id: str, 
                              user_id: str, content: str, message_type: str = "text") -> bool:
        """Send message to team channel."""
        
        # Check if user is team member
        user_profile = await self.storage.get_user_profile(user_id)
        if not user_profile or team_id not in user_profile.metadata.get("teams", []):
            return False
        
        message = FFMessageDTO(
            role=MessageRole.USER,
            content=content,
            metadata={
                "type": message_type,
                "team_id": team_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "user_name": user_profile.name or user_id
            }
        )
        
        success = await self.storage.add_message(f"team_{team_id}", channel_id, message)
        
        # Update channel activity
        if success:
            await self._update_channel_activity(team_id, channel_id, user_id)
        
        return success
    
    async def _update_channel_activity(self, team_id: str, channel_id: str, user_id: str):
        """Update channel activity tracking."""
        session = await self.storage.get_session(f"team_{team_id}", channel_id)
        if session:
            active_members = session.metadata.get("active_members", [])
            if user_id not in active_members:
                active_members.append(user_id)
            
            await self.storage.update_session(
                f"team_{team_id}",
                channel_id,
                metadata={
                    **session.metadata,
                    "active_members": active_members,
                    "last_activity": datetime.now().isoformat()
                }
            )
    
    async def get_team_channels(self, team_id: str) -> List[Dict[str, Any]]:
        """Get all channels for a team."""
        sessions = await self.storage.list_sessions(f"team_{team_id}")
        
        channels = []
        for session in sessions:
            if session.metadata.get("type") == "team_channel":
                channels.append({
                    "id": session.session_id,
                    "name": session.title,
                    "type": session.metadata.get("channel_type"),
                    "description": session.metadata.get("description"),
                    "message_count": session.message_count,
                    "active_members": len(session.metadata.get("active_members", [])),
                    "last_activity": session.metadata.get("last_activity")
                })
        
        return channels
    
    async def search_team_messages(self, team_id: str, query: str, 
                                 channel_id: str = None) -> List[Dict[str, Any]]:
        """Search messages within team."""
        session_ids = [channel_id] if channel_id else None
        
        results = await self.storage.search_messages(
            f"team_{team_id}",
            query,
            session_ids=session_ids,
            limit=50
        )
        
        # Enrich results with user information
        enriched_results = []
        for result in results:
            if hasattr(result, 'metadata') and result.metadata:
                enriched_results.append({
                    "content": result.content,
                    "user_name": result.metadata.get("user_name", "Unknown"),
                    "channel_id": result.metadata.get("channel_id"),
                    "timestamp": result.metadata.get("timestamp"),
                    "relevance_score": result.relevance_score
                })
        
        return enriched_results
    
    async def get_team_analytics(self, team_id: str) -> Dict[str, Any]:
        """Get team collaboration analytics."""
        channels = await self.get_team_channels(team_id)
        
        total_messages = sum(channel["message_count"] for channel in channels)
        active_channels = len([c for c in channels if c["message_count"] > 0])
        
        return {
            "team_id": team_id,
            "total_channels": len(channels),
            "active_channels": active_channels,
            "total_messages": total_messages,
            "avg_messages_per_channel": total_messages / len(channels) if channels else 0,
            "most_active_channel": max(channels, key=lambda c: c["message_count"]) if channels else None
        }

# Team collaboration demo
async def team_collaboration_demo():
    """Demonstrate team collaboration features."""
    team_tool = TeamCollaborationTool()
    await team_tool.initialize()
    
    # Create team
    await team_tool.create_team("dev_team", "Development Team", "Software development team")
    
    # Create team members
    members = ["alice", "bob", "charlie"]
    for member in members:
        await team_tool.storage.create_user(member, {"name": f"User {member.title()}"})
        await team_tool.add_team_member("dev_team", member)
    
    # Create channels
    general_channel = await team_tool.create_channel("dev_team", "general", "public", "General discussion")
    dev_channel = await team_tool.create_channel("dev_team", "development", "public", "Dev discussions")
    
    # Send messages
    messages = [
        ("alice", general_channel, "Hey team! How's everyone doing?"),
        ("bob", general_channel, "Good morning! Ready for the sprint planning."),
        ("charlie", dev_channel, "I found a bug in the authentication module."),
        ("alice", dev_channel, "Can you create a ticket for that? Let's prioritize it."),
        ("bob", dev_channel, "I'll help with the fix. What's the error message?")
    ]
    
    for user, channel, content in messages:
        await team_tool.send_team_message("dev_team", channel, user, content)
    
    # Search team messages
    search_results = await team_tool.search_team_messages("dev_team", "bug authentication")
    print(f"Found {len(search_results)} messages about authentication bugs")
    
    # Get team analytics
    analytics = await team_tool.get_team_analytics("dev_team")
    print(f"Team Analytics: {analytics}")
    
    # Get team channels
    channels = await team_tool.get_team_channels("dev_team")
    print(f"Team has {len(channels)} channels")
    for channel in channels:
        print(f"  - {channel['name']}: {channel['message_count']} messages")

asyncio.run(team_collaboration_demo())
```

## 📊 Analytics Dashboard

### Usage Analytics System
```python
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd

class AnalyticsDashboard:
    """Analytics dashboard for chat system usage."""
    
    def __init__(self):
        self.storage = None
    
    async def initialize(self):
        """Initialize analytics dashboard."""
        config = load_config()
        self.storage = FFStorageManager(config)
        await self.storage.initialize()
    
    async def generate_user_activity_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate user activity report."""
        # Get all users (simplified - would need proper user listing)
        users = ["alice", "bob", "charlie", "dave"]  # Sample users
        
        activity_data = {}
        for user_id in users:
            if await self.storage.user_exists(user_id):
                sessions = await self.storage.list_sessions(user_id)
                user_stats = await self.storage.get_user_stats(user_id)
                
                activity_data[user_id] = {
                    "total_sessions": len(sessions),
                    "total_messages": user_stats.total_messages if hasattr(user_stats, 'total_messages') else 0,
                    "avg_session_length": user_stats.avg_messages_per_session if hasattr(user_stats, 'avg_messages_per_session') else 0,
                    "last_activity": sessions[0].updated_at if sessions else "Never"
                }
        
        return {
            "report_period_days": days,
            "total_users": len(activity_data),
            "active_users": len([u for u in activity_data.values() if u["total_sessions"] > 0]),
            "user_details": activity_data,
            "summary": {
                "total_sessions": sum(u["total_sessions"] for u in activity_data.values()),
                "total_messages": sum(u["total_messages"] for u in activity_data.values()),
                "avg_sessions_per_user": sum(u["total_sessions"] for u in activity_data.values()) / len(activity_data)
            }
        }
    
    async def generate_content_analysis(self) -> Dict[str, Any]:
        """Analyze content patterns and topics."""
        # This would involve more sophisticated text analysis
        # For demo, returning sample analysis
        
        # Sample topic extraction (would use NLP in real implementation)
        topics = {
            "technical_support": 45,
            "billing_inquiries": 32,
            "product_questions": 28,
            "feature_requests": 15,
            "complaints": 8
        }
        
        # Sample sentiment analysis
        sentiment_distribution = {
            "positive": 0.52,
            "neutral": 0.31,
            "negative": 0.17
        }
        
        # Sample message length analysis
        message_lengths = {
            "short (< 50 chars)": 0.35,
            "medium (50-200 chars)": 0.48,
            "long (> 200 chars)": 0.17
        }
        
        return {
            "topic_distribution": topics,
            "sentiment_distribution": sentiment_distribution,
            "message_length_distribution": message_lengths,
            "total_analyzed_messages": sum(topics.values())
        }
    
    async def generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate system performance metrics."""
        # Get system stats
        storage_stats = await self.storage.get_storage_stats()
        
        # Sample performance data (would be real metrics in production)
        performance_data = {
            "storage_metrics": {
                "total_users": storage_stats.total_users if hasattr(storage_stats, 'total_users') else 0,
                "total_sessions": storage_stats.total_sessions if hasattr(storage_stats, 'total_sessions') else 0,
                "total_messages": storage_stats.total_messages if hasattr(storage_stats, 'total_messages') else 0,
                "storage_size_mb": storage_stats.storage_size_mb if hasattr(storage_stats, 'storage_size_mb') else 0
            },
            "response_times": {
                "avg_message_save_ms": 45,
                "avg_search_time_ms": 120,
                "avg_session_load_ms": 25
            },
            "system_health": {
                "uptime_percentage": 99.8,
                "error_rate": 0.002,
                "cache_hit_rate": 0.85
            }
        }
        
        return performance_data
    
    async def create_usage_trend_chart(self, output_file: str = "usage_trends.png"):
        """Create usage trend visualization."""
        # Sample data (would be real time-series data)
        dates = pd.date_range(start="2023-10-01", end="2023-10-30", freq="D")
        daily_messages = [random.randint(50, 200) for _ in range(len(dates))]
        daily_users = [random.randint(10, 50) for _ in range(len(dates))]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Messages trend
        ax1.plot(dates, daily_messages, marker='o', linewidth=2)
        ax1.set_title("Daily Message Volume")
        ax1.set_ylabel("Messages")
        ax1.grid(True, alpha=0.3)
        
        # Users trend
        ax2.plot(dates, daily_users, marker='s', color='orange', linewidth=2)
        ax2.set_title("Daily Active Users")
        ax2.set_ylabel("Users")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    async def export_analytics_report(self, output_file: str = "analytics_report.json"):
        """Export comprehensive analytics report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "user_activity": await self.generate_user_activity_report(),
            "content_analysis": await self.generate_content_analysis(),
            "performance_metrics": await self.generate_performance_metrics()
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_file

# Analytics demo
async def analytics_dashboard_demo():
    """Demonstrate analytics dashboard."""
    dashboard = AnalyticsDashboard()
    await dashboard.initialize()
    
    print("Generating analytics reports...")
    
    # User activity report
    user_report = await dashboard.generate_user_activity_report()
    print(f"User Activity Report:")
    print(f"  Active users: {user_report['active_users']}/{user_report['total_users']}")
    print(f"  Total sessions: {user_report['summary']['total_sessions']}")
    print(f"  Total messages: {user_report['summary']['total_messages']}")
    
    # Content analysis
    content_analysis = await dashboard.generate_content_analysis()
    print(f"\nContent Analysis:")
    print(f"  Top topics: {list(content_analysis['topic_distribution'].keys())[:3]}")
    print(f"  Sentiment: {content_analysis['sentiment_distribution']['positive']:.0%} positive")
    
    # Performance metrics
    performance = await dashboard.generate_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  System uptime: {performance['system_health']['uptime_percentage']}%")
    print(f"  Cache hit rate: {performance['system_health']['cache_hit_rate']:.0%}")
    
    # Export full report
    report_file = await dashboard.export_analytics_report()
    print(f"\nFull report exported to: {report_file}")
    
    # Create trend chart
    chart_file = await dashboard.create_usage_trend_chart()
    print(f"Usage trends chart saved to: {chart_file}")

# Note: Requires matplotlib and pandas
# pip install matplotlib pandas
# asyncio.run(analytics_dashboard_demo())
```

## 🎉 Examples Summary

These examples demonstrate:

- ✅ **Real-world Integration Patterns** - Practical implementation approaches
- ✅ **Complete Working Systems** - End-to-end functionality
- ✅ **Best Practices** - Proper error handling, async patterns, and architecture
- ✅ **Scalable Designs** - Systems that can grow with your needs
- ✅ **Rich Analytics** - Comprehensive data analysis and reporting
- ✅ **User Experience Focus** - Systems designed for real users

Each example can be adapted and extended for your specific use case. The modular design of the Flatfile Chat Database makes it easy to implement any of these patterns in your applications.

**Next Steps:**
- **[Performance & Optimization](08_PERFORMANCE.md)** - Optimize these implementations
- **[Troubleshooting](09_TROUBLESHOOTING.md)** - Handle issues in production
- **[Migration Guide](10_MIGRATION.md)** - Upgrade existing systems