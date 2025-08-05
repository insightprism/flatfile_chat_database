# Flatfile Chat Database - Demo Suite

Welcome to the comprehensive demo suite for the Flatfile Chat Database! This collection of demos showcases all the key features and capabilities of this file-based storage solution for AI chat applications.

## 🎯 What This Demo Shows

The Flatfile Chat Database is a powerful, file-based storage system designed specifically for AI chat applications. These demos demonstrate:

- **💾 File-Based Storage**: No database server required - everything stored as JSON/JSONL files
- **👥 User Management**: Store user profiles and preferences
- **💬 Chat Sessions**: Organize conversations into sessions with metadata
- **📄 Document Processing**: Add documents and create searchable content
- **🔍 Advanced Search**: Both text-based and semantic vector search
- **⚙️ Flexible Configuration**: Support for both legacy and modern config systems
- **🧠 AI Integration**: Built-in support for embeddings and RAG pipelines
- **📊 Performance**: Efficient storage and retrieval for chat applications

## 🚀 Demo Options

### 🎯 Choose Your Demo Experience
**New to the system?** Check our **[Demo Master Guide](docs/demo_master_guide.md)** to choose the perfect demo for your needs!

### 1. 📓 Jupyter Notebook Demo (Interactive)
**Best for**: Learning, exploration, and understanding the system step-by-step

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter and open the demo notebook
jupyter notebook jupyter_notebook_demo.ipynb
```

**Features**:
- Interactive code cells with explanations
- Visual outputs and data inspection
- Step-by-step walkthrough of all features
- Modifiable examples you can experiment with

**📖 [Complete Jupyter Notebook User Guide →](docs/jupyter_notebook_guide.md)**

### 2. 💻 CLI Interactive Demo (Menu-Driven)
**Best for**: Hands-on exploration through a command-line interface

```bash
# Run the interactive CLI demo
python cli_interactive_demo.py
```

**Features**:
- Menu-driven interface for easy navigation
- Create users, sessions, and documents interactively
- Perform searches and view results
- Explore system statistics and file structure
- Quick test functionality to verify everything works

**📖 [Complete CLI Interactive User Guide →](docs/cli_interactive_guide.md)**

### 3. 🤖 Automated Demo Script (Comprehensive)
**Best for**: Quick overview, presentations, or automated testing

```bash
# Run the full automated demo
python automated_demo_script.py

# Run in quiet mode (less output)
python automated_demo_script.py --quiet

# Clean up demo data after completion
python automated_demo_script.py --cleanup
```

**Features**:
- Comprehensive demonstration of all features
- Creates realistic sample data automatically
- Performance benchmarks
- Detailed system statistics
- No user interaction required

**📖 [Complete Automated Demo User Guide →](docs/automated_demo_guide.md)**

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 100MB free disk space for demo data
- Terminal or command prompt access

### Dependencies

Install the required dependencies:

```bash
# Core dependencies (from parent directory)
pip install -r ../requirements.txt

# Demo-specific dependencies
pip install -r requirements_demo.txt
```

Or install individually:
```bash
pip install numpy pandas jupyter matplotlib asyncio pathlib dataclasses
```

### 🧠 Optional: PrismMind Integration

For enhanced document processing capabilities, you can install PrismMind:

#### If PrismMind is in `/home/markly2/prismmind`:
The demos will automatically detect and use PrismMind from this location.

#### For a different location:
Edit the demo files to update the `prismmind_path` variable:

```python
# In any demo script, change this line:
prismmind_path = '/home/markly2/prismmind'

# To your actual path:
prismmind_path = '/your/path/to/prismmind'
```

#### Files to update (if needed):
- `jupyter_notebook_demo.ipynb` (cell 2)
- `cli_interactive_demo.py` (line ~21)
- `automated_demo_script.py` (line ~24)
- `test_basic_functionality.py` (line ~18)

**Note**: The demos work perfectly without PrismMind using legacy document processing. PrismMind just provides additional features like PDF support and enhanced text processing.

## 🎬 Quick Start

1. **Clone/navigate to the demo directory**:
   ```bash
   cd flatfile_chat_database_v2/demo
   ```

2. **Choose your preferred demo**:
   
   **For beginners**: Start with the Jupyter notebook
   ```bash
   jupyter notebook jupyter_notebook_demo.ipynb
   ```
   
   **For exploration**: Try the CLI interactive demo
   ```bash
   python cli_interactive_demo.py
   ```
   
   **For a quick overview**: Run the automated demo
   ```bash
   python automated_demo_script.py
   ```

3. **Follow the prompts** and explore the features!

## 📁 Demo Files Structure

```
demo/
├── README.md                          # This file
├── requirements_demo.txt              # Demo dependencies
├── jupyter_notebook_demo.ipynb        # Interactive Jupyter demo
├── cli_interactive_demo.py            # Menu-driven CLI demo  
├── automated_demo_script.py           # Comprehensive automated demo
└── sample_data/                       # Sample files for demos
    ├── documents/                     # Technical documents
    │   ├── python_best_practices.md   # Python coding guide
    │   └── api_design_guide.md        # REST API design guide
    └── chat_examples/                 # Example conversations
        ├── sample_conversation.json   # AI/ML discussion
        └── coding_session.json        # Code review session
```

## 🔍 What Each Demo Covers

### Core Features Demonstrated

| Feature | Jupyter | CLI | Automated |
|---------|---------|-----|-----------|
| User management | ✅ | ✅ | ✅ |
| Chat sessions | ✅ | ✅ | ✅ |
| Message storage | ✅ | ✅ | ✅ |
| Document processing | ✅ | ✅ | ✅ |
| Vector embeddings | ✅ | ⚠️* | ✅ |
| Text search | ✅ | ✅ | ✅ |
| Vector search | ✅ | ✅ | ✅ |
| Configuration systems | ✅ | ✅ | ✅ |
| File structure inspection | ✅ | ✅ | ✅ |
| Performance benchmarks | ⚠️* | ❌ | ✅ |

*⚠️ = Limited or simplified implementation*

### Sample Data

The demos include realistic sample data:

- **3 Demo Users**: Researcher, Developer, Analyst with detailed profiles
- **4 Chat Sessions**: Technical discussions on AI, databases, and statistics  
- **Technical Documents**: Best practices guides for Python and API design
- **Conversation Examples**: Real-world chat scenarios with code and explanations

## 🛠 Advanced Usage

### Customizing Demo Data

You can modify the sample data to test with your own content:

1. **Edit sample documents** in `sample_data/documents/`
2. **Modify conversation examples** in `sample_data/chat_examples/`
3. **Update user profiles** in the demo scripts

### Performance Testing

The automated demo includes performance benchmarks. To run more extensive tests:

```python
# In automated_demo_script.py, modify:
benchmark_session = await self.storage_manager.create_session(
    user_id="alice_researcher",
    title="Performance Benchmark Session"
)

# Increase message count for stress testing
message_count = 10000  # Increase from 50
```

### Integration with Your Code

To integrate the flatfile database into your own project:

```python
# Basic integration example
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config

# Initialize
config = load_config()
config.storage.base_path = "./my_chat_data"
storage = FFStorageManager(config)

# Use the API demonstrated in the demos
user_profile = FFUserProfileDTO(user_id="my_user", username="My User")
await storage.store_user_profile(user_profile)
```

## 🧹 Cleanup

Demo data is stored in temporary directories:

- **Jupyter Demo**: `./demo_data/`
- **CLI Demo**: `./demo_data_cli/`  
- **Automated Demo**: `./demo_data_automated/`

To clean up:

```bash
# Remove all demo data directories
rm -rf demo_data* 

# Or run automated demo with cleanup
python automated_demo_script.py --cleanup
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Make sure you're in the right directory
cd flatfile_chat_database_v2/demo

# Install missing dependencies
pip install -r requirements_demo.txt
```

**Permission Errors**:
```bash
# On Unix systems, ensure write permissions
chmod 755 .
```

**Memory Issues** (with large datasets):
- Use the CLI demo's chunked processing options
- Reduce sample data size in automated demo
- Monitor memory usage with the built-in tools

**Jupyter Notebook Issues**:
```bash
# Install/upgrade jupyter
pip install --upgrade jupyter

# Install required kernel
python -m ipykernel install --user
```

### Getting Help

If you encounter issues:

1. **Check Prerequisites**: Ensure Python 3.8+ and required packages
2. **Review Error Messages**: Most errors include helpful information
3. **Try Different Demos**: If one doesn't work, try another approach
4. **Check File Permissions**: Ensure write access to demo directories

## 🎯 Next Steps

After exploring the demos:

1. **Read the Documentation**: Check the main project documentation
2. **Explore the Source Code**: Look at the actual implementation
3. **Try Real Data**: Test with your own chat data
4. **Integrate**: Use the database in your own AI chat application
5. **Contribute**: Consider contributing improvements or examples

## 📚 Additional Resources

- **Main Project**: See the parent directory for full documentation
- **Configuration Guide**: Check `ARCHITECTURE_UPGRADE_SUMMARY.md`
- **Sample Configs**: Look in `../ff_preset_configs/` for configuration examples
- **Tests**: Review `../test_new_architecture.py` for more usage examples

## 🙏 Feedback

Found a bug or have suggestions? The demos are designed to showcase the system's capabilities. If something doesn't work as expected or you have ideas for improvements, please let us know!

---

**Happy exploring! 🚀**

*The Flatfile Chat Database makes chat data storage simple, efficient, and transparent. No hidden databases, no complex setups - just clean, organized files that you can inspect, backup, and version control like any other data.*