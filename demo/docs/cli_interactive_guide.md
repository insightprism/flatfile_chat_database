# CLI Interactive Demo - User Guide

## 💻 Menu-Driven Exploration

The CLI Interactive Demo provides a hands-on, menu-driven way to explore the Flatfile Chat Database. This guide will help you navigate all features and make the most of the interactive experience.

## 🎯 Who This Demo Is For

- **Hands-on learners**: Prefer interactive exploration over reading
- **CLI enthusiasts**: Comfortable with command-line interfaces
- **Feature testers**: Want to try each feature individually
- **System administrators**: Need to understand operational aspects

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **Terminal Access**: Command prompt, PowerShell, or terminal
3. **Dependencies**:
   ```bash
   cd flatfile_chat_database_v2/demo
   pip install -r requirements_demo.txt
   pip install -r ../requirements.txt
   ```

### Launching the Demo

```bash
cd flatfile_chat_database_v2/demo
python cli_interactive_demo.py
```

**Expected output**:
```
🚀 Initializing Flatfile Chat Database Demo...
✅ Demo initialized! Data will be stored in: ./demo_data_cli

============================================================
🏠 FLATFILE CHAT DATABASE - INTERACTIVE DEMO
============================================================
📊 Status: User: None | Session: None

📋 MENU OPTIONS:
1.  👤 User Management
2.  💬 Chat Sessions
3.  📄 Document Management
4.  🔍 Search & Retrieval
5.  📊 Statistics & Info
6.  ⚙️  Configuration
7.  📁 File Explorer
8.  🧪 Run Quick Test
9.  🧹 Cleanup Demo Data
0.  🚪 Exit
```

## 📚 Menu System Overview

The demo uses a hierarchical menu system. Each main option leads to a submenu with specific actions.

### Navigation Tips
- **Enter numbers** to select menu options
- **Use 0** to go back to previous menu
- **Follow prompts** for input when required
- **Press Ctrl+C** to exit at any time

## 🔍 Detailed Menu Guide

### 1. 👤 User Management

**Purpose**: Create and manage users in the system

#### Submenu Options:
```
1. Create new user
2. List all users  
3. Select current user
4. View user profile
0. Back to main menu
```

#### Step-by-Step Walkthrough:

**Creating Your First User:**
1. Select `1. Create new user`
2. **Enter user ID**: Choose a unique identifier (e.g., "john_doe")
3. **Enter display name**: Full name or display name (e.g., "John Doe")
4. **Enter department** (optional): Your department or team
5. **Enter role** (optional): Your job role or function

**What happens**: 
- User profile created and stored
- Profile file saved at `demo_data_cli/users/{user_id}/profile.json`
- User becomes available for selection

**Pro tip**: Create multiple users to test multi-user scenarios

### 2. 💬 Chat Sessions

**Purpose**: Create chat sessions and manage messages
**Prerequisite**: Must have selected a current user first

#### Submenu Options:
```
1. Create new session
2. List sessions
3. Select session  
4. Add message to session
5. View session messages
0. Back to main menu
```

#### Step-by-Step Walkthrough:

**Creating Your First Session:**
1. **Ensure you have a user selected** (see User Management)
2. Select `1. Create new session`
3. **Enter session title**: Descriptive name (e.g., "AI Discussion")
4. **Session created**: Note the generated session ID

**Adding Messages:**
1. Select `4. Add message to session`
2. **Choose message role**:
   - `1`: User message (your input)
   - `2`: Assistant message (AI response)
   - `3`: System message (system notifications)
3. **Enter message content**: Type your message
   - Press Enter twice when finished
   - Can include multiple lines
4. **Message stored**: Added to session with timestamp

**Viewing Conversations:**
1. Select `5. View session messages`
2. **See formatted output**: Messages with roles, timestamps, and content
3. **Navigate long conversations**: Output shows recent messages first

### 3. 📄 Document Management

**Purpose**: Add and manage documents for search and analysis
**Prerequisite**: Must have user and session selected

#### Submenu Options:
```
1. Add document from text
2. Add document from file
3. List documents
4. View document
0. Back to main menu
```

#### Step-by-Step Walkthrough:

**Adding Text Documents:**
1. Select `1. Add document from text`
2. **Enter filename**: Include extension (e.g., "notes.md")
3. **Enter content**: Type or paste document content
   - Press Enter twice when finished
   - Supports multi-line content
4. **Document stored**: Saved with metadata and processing info

**Adding File Documents:**
1. Select `2. Add document from file`
2. **Enter file path**: Absolute or relative path to existing file
3. **File validation**: System checks if file exists and is readable
4. **Document imported**: Content read and stored in database

**Managing Documents:**
- **List documents**: See all documents in current session
- **View document**: Display document content (truncated if long)
- **Metadata tracking**: Creation time, size, and custom metadata

### 4. 🔍 Search & Retrieval

**Purpose**: Test search capabilities across messages and documents
**Prerequisite**: Must have content (messages/documents) to search

#### Submenu Options:
```
1. Search messages
2. Search documents  
3. Search all content
4. Vector similarity search
0. Back to main menu
```

#### Step-by-Step Walkthrough:

**Text-Based Search:**
1. **Add some content first** (messages or documents)
2. Select search type (`1`, `2`, or `3`)
3. **Enter search query**: Keywords or phrases to find
4. **Review results**:
   - Relevance scores (0.0 to 1.0)
   - Content type (message/document)
   - Matching text snippets
   - Associated metadata

**Vector Similarity Search:**
1. Select `4. Vector similarity search`
2. **Enter search text**: Natural language query
3. **System processes**: Creates embeddings (mock for demo)
4. **Similarity results**: 
   - Semantic similarity scores
   - Related content chunks
   - Source document information

**Search Tips:**
- Try different query styles (keywords vs. questions)
- Search across different content types
- Note how relevance scores vary

### 5. 📊 Statistics & Info

**Purpose**: Monitor system performance and data usage

#### Information Displayed:
- **Current session stats**: Message count, document count, total size
- **Storage statistics**: Total size, file count, average file size
- **Configuration info**: Current settings and paths
- **Performance metrics**: Storage usage and organization

#### What to Look For:
- **Data growth**: How much data you've created
- **Storage efficiency**: File sizes and organization
- **System health**: All components working correctly

### 6. ⚙️ Configuration

**Purpose**: Understand system configuration options

#### Information Displayed:
- **Legacy configuration**: Current settings and paths
- **New modular configuration**: Domain-specific settings (if available)
- **Validation status**: Whether configurations are valid
- **Configuration summary**: Settings count by domain

#### Key Configuration Areas:
- **Storage paths**: Where data is stored
- **Feature flags**: What functionality is enabled
- **Performance settings**: Limits and optimization options
- **Integration settings**: External service configurations

### 7. 📁 File Explorer

**Purpose**: Understand the file system structure created

#### What You'll See:
```
demo_data_cli/
├── users/
│   ├── john_doe/
│   │   ├── profile.json
│   │   └── chat_session_123/
│   │       ├── session.json
│   │       ├── messages.jsonl
│   │       └── documents/
│   │           ├── metadata.json
│   │           └── doc_456.txt
```

#### Understanding the Structure:
- **Hierarchical organization**: Users → Sessions → Content
- **Human-readable files**: All JSON/JSONL format
- **Predictable naming**: Consistent file and folder names
- **Metadata tracking**: Separate metadata files for efficiency

### 8. 🧪 Run Quick Test

**Purpose**: Automated verification that all features work

#### What It Does:
1. **Creates test user**: Temporary user for testing
2. **Creates test session**: Sample session with metadata
3. **Adds test message**: Verifies message storage
4. **Adds test document**: Verifies document processing
5. **Runs search test**: Verifies search functionality
6. **Checks file structure**: Ensures proper file creation

#### When to Use:
- **First time setup**: Verify everything works
- **After changes**: Ensure system still functions
- **Troubleshooting**: Isolate issues with clean test data

### 9. 🧹 Cleanup Demo Data

**Purpose**: Remove all generated demo data

#### What It Does:
- **Confirms deletion**: Asks for confirmation before proceeding
- **Removes all files**: Deletes entire demo_data_cli directory
- **Resets state**: Clears current user and session selections
- **Recreates directory**: Sets up fresh environment

#### When to Use:
- **Starting fresh**: Clean slate for new demo session
- **Disk space**: Remove large amounts of test data
- **Reset environment**: Clear any problematic state

## 🎯 Recommended Workflow

### First-Time Users

1. **Start with Quick Test** (`8`): Verify everything works
2. **Create Your User** (`1` → `1`): Set up your profile
3. **Explore File Structure** (`7`): See what was created
4. **Create a Session** (`2` → `1`): Start a conversation
5. **Add Some Messages** (`2` → `4`): Create content to search
6. **Try Searching** (`4`): Test search capabilities
7. **View Statistics** (`5`): See system metrics

### Advanced Users

1. **Create Multiple Users**: Test multi-user scenarios
2. **Create Multiple Sessions**: Organize different topics
3. **Add Various Content Types**: Messages, documents, different formats
4. **Test All Search Types**: Compare different search strategies
5. **Monitor Performance**: Watch how data grows and performs
6. **Experiment with Configuration**: Understand system flexibility

## 💡 Tips and Tricks

### Efficient Navigation
- **Use numeric shortcuts**: Just type the number and press Enter
- **Remember menu hierarchy**: 0 always goes back
- **Exit gracefully**: Use menu option 0 rather than Ctrl+C when possible

### Content Creation
- **Plan your test data**: Create realistic scenarios for better testing
- **Use meaningful names**: Makes exploration easier
- **Mix content types**: Messages and documents together
- **Test edge cases**: Empty content, very long content, special characters

### Search Testing
- **Start simple**: Single keywords first
- **Get creative**: Try questions, phrases, technical terms
- **Compare search types**: See how different approaches work
- **Note performance**: Observe search speed with different data sizes

### Understanding Output
- **Read status line**: Shows current user and session
- **Watch for confirmations**: System confirms successful operations
- **Note file paths**: Shows where data is stored
- **Pay attention to counts**: Messages, documents, file sizes

## 🚫 Common Issues and Solutions

### "No user selected" Errors
**Problem**: Trying to use features without selecting a user
**Solution**: Go to User Management → Select current user

### "No session selected" Errors  
**Problem**: Trying to add messages/documents without a session
**Solution**: Go to Chat Sessions → Select session (create one if needed)

### "File not found" Errors
**Problem**: Trying to add document from invalid file path
**Solution**: 
- Use absolute paths (`/full/path/to/file.txt`)
- Check file exists and is readable
- Try relative paths from demo directory

### Search Returns No Results
**Problem**: Search queries don't find expected content
**Solution**:
- Ensure content exists (add messages/documents first)
- Try simpler search terms
- Check you're searching in the right scope (messages vs documents)

### Menu Not Responding
**Problem**: Interface seems frozen or unresponsive
**Solution**:
- Press Enter to continue
- Check for input prompts you missed
- Restart if necessary (Ctrl+C then rerun)

## 🔧 Customization Options

### Modify Demo Data Path
Edit the script to change where data is stored:
```python
self.demo_data_path = Path("./my_custom_path")
```

### Add Custom Menu Options
Extend the menu system with your own features:
```python
def my_custom_feature(self):
    print("My custom functionality")
    # Your code here
```

### Adjust Display Settings
Modify output formatting, colors, or verbosity levels

## 📊 Understanding the Experience

### What Makes This Demo Unique
- **Interactive discovery**: Learn by doing
- **Immediate feedback**: See results instantly
- **Flexible exploration**: Go at your own pace
- **Real system**: Not a simulation, actual database operations

### Skills You'll Develop
- **System navigation**: Understanding menu-driven interfaces
- **Data organization**: How to structure users, sessions, and content
- **Search strategies**: Different approaches to finding information
- **System monitoring**: How to track performance and usage
- **File system understanding**: Where and how data is stored

## 🔗 Next Steps

After mastering the CLI demo:

1. **Try the Jupyter Notebook**: Deep dive into implementation details
2. **Run the Automated Demo**: See comprehensive feature showcase  
3. **Integrate into your project**: Apply what you've learned
4. **Explore source code**: Understand how the CLI is implemented

## 🆘 Getting Help

If you encounter problems:

1. **Read error messages**: Usually contain helpful information
2. **Check prerequisites**: Ensure Python and dependencies are installed
3. **Try Quick Test**: Verify basic functionality works
4. **Start fresh**: Use Cleanup Demo Data to reset
5. **Check file permissions**: Ensure write access to demo directory

## 🎯 Success Metrics

You'll know you've mastered this demo when you can:

- ✅ Navigate all menus confidently
- ✅ Create users, sessions, and content systematically
- ✅ Use all search types effectively
- ✅ Interpret system statistics and file structures
- ✅ Troubleshoot common issues independently
- ✅ Understand the relationship between UI actions and file system changes

---

**Happy exploring! 💻 The CLI Interactive Demo gives you hands-on control over every aspect of the Flatfile Chat Database.**