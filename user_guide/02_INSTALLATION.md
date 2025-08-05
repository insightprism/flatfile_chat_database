# Installation & Setup

Complete installation guide for the Flatfile Chat Database system, from development to production deployment.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows 10+
- **RAM**: 512MB available memory
- **Disk Space**: 100MB for installation + storage for your data
- **Network**: Internet access for initial package installation

### Recommended Requirements
- **Python**: 3.9+ (for optimal performance)
- **RAM**: 2GB+ for production workloads
- **Disk Space**: 1GB+ with SSD storage for better performance
- **CPU**: Multi-core processor for concurrent operations

### Supported Platforms
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- ✅ **macOS**: 10.15+ (Catalina or newer)
- ✅ **Windows**: Windows 10, Windows Server 2019+
- ✅ **Docker**: Any platform supporting Docker

## 🐍 Python Environment Setup

### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv ff_chat_env

# Activate (Linux/macOS)
source ff_chat_env/bin/activate

# Activate (Windows)
ff_chat_env\Scripts\activate

# Verify activation
which python  # Should point to virtual environment
```

### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n ff_chat python=3.9

# Activate environment
conda activate ff_chat

# Verify
conda info --envs
```

### Option 3: System Python (Not Recommended for Production)
```bash
# Only if you must use system Python
python --version  # Ensure 3.8+
```

## 📦 Installation Methods

### Method 1: Direct Installation (Development)

```bash
# Clone the repository
git clone <repository-url> flatfile_chat_database
cd flatfile_chat_database

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ff_storage_manager import FFStorageManager; print('Installation successful!')"
```

### Method 2: Package Installation (Future)
```bash
# When available as a package
pip install flatfile-chat-database
```

### Method 3: Docker Installation
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "your_app.py"]
```

```bash
# Build and run
docker build -t ff-chat-db .
docker run -v $(pwd)/data:/app/data ff-chat-db
```

## 🔧 Dependency Management

### Core Dependencies
The system requires these essential packages:

```txt
# requirements.txt (essential)
asyncio-tools>=0.3.0
aiofiles>=0.8.0
pydantic>=1.10.0
numpy>=1.21.0 (for vector operations)
```

### Optional Dependencies
```txt
# requirements-optional.txt
sentence-transformers>=2.2.0  # For embeddings
scikit-learn>=1.1.0          # For advanced search
psutil>=5.9.0                # For performance monitoring
```

### Development Dependencies
```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-asyncio>=0.20.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

## 🏗️ Installation Verification

### Basic Verification
```python
# test_installation.py
import asyncio
import sys
from pathlib import Path

async def verify_installation():
    """Comprehensive installation verification."""
    
    print("🔍 Verifying Flatfile Chat Database Installation\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Test core imports
    try:
        from ff_storage_manager import FFStorageManager
        from ff_class_configs.ff_configuration_manager_config import load_config
        from ff_class_configs.ff_chat_entities_config import FFMessageDTO, MessageRole
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test configuration loading
    try:
        config = load_config()
        print("✅ Configuration loading successful")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    # Test storage manager creation
    try:
        storage = FFStorageManager(config)
        await storage.initialize()
        print("✅ Storage manager initialization successful")
    except Exception as e:
        print(f"❌ Storage manager initialization failed: {e}")
        return False
    
    # Test basic operations
    try:
        # Create user
        await storage.create_user("test_install_user")
        
        # Create session
        session_id = await storage.create_session("test_install_user", "Install Test")
        
        # Add message
        message = FFMessageDTO(
            role=MessageRole.USER,
            content="Installation test message"
        )
        await storage.add_message("test_install_user", session_id, message)
        
        # Retrieve messages
        messages = await storage.get_all_messages("test_install_user", session_id)
        
        if len(messages) == 1:
            print("✅ Basic operations successful")
        else:
            print("❌ Basic operations failed - incorrect message count")
            return False
            
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return False
    
    print("\n🎉 Installation verification completed successfully!")
    print("The Flatfile Chat Database is ready for use.")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_installation())
    if not success:
        sys.exit(1)
```

Run the verification:
```bash
python test_installation.py
```

## 🔧 Configuration Setup

### Default Configuration
The system works with zero configuration:
```python
from ff_class_configs.ff_configuration_manager_config import load_config

# Uses built-in defaults
config = load_config()
```

### Environment-Specific Configuration

#### 1. Development Environment
```python
# Uses relaxed settings, verbose logging
config = load_config("development")
```

#### 2. Testing Environment
```python
# Uses fast, in-memory options where possible
config = load_config("test")
```

#### 3. Production Environment
```python
# Uses optimized, secure settings
config = load_config("production")
```

### Custom Configuration Directory
```python
# Use custom config directory
import os
os.environ['FF_CONFIG_PATH'] = '/path/to/custom/configs'
config = load_config()
```

### Configuration Files Location

The system looks for configuration files in this order:
1. `FF_CONFIG_PATH` environment variable
2. `./ff_preset_configs/` in project directory
3. Built-in defaults

```
ff_preset_configs/
├── ff_development_config.json
├── ff_test_config.json
├── ff_production_config.json
└── ff_flatfile_prismmind_config.json
```

## 📁 Directory Structure Setup

### Automatic Directory Creation
The system automatically creates necessary directories:
```
data/                          # Default base path
├── users/                     # User data
│   └── {user_id}/
│       ├── profile.json       # User profile
│       └── chat_session_*/    # Session directories
├── system/                    # System data
├── personas_global/           # Global personas
└── panel_sessions/           # Panel sessions
```

### Custom Directory Structure
```python
from ff_class_configs.ff_configuration_manager_config import load_config

config = load_config()
config.storage.base_path = "/custom/storage/path"
config.storage.user_data_path = "users"
config.storage.system_data_path = "system"
```

### Permissions Setup (Linux/macOS)
```bash
# Set up proper permissions
mkdir -p /opt/ff_chat_data
chown -R your_user:your_group /opt/ff_chat_data
chmod -R 755 /opt/ff_chat_data

# For production with dedicated user
sudo useradd -r -s /bin/false ff_chat
sudo mkdir -p /var/lib/ff_chat
sudo chown ff_chat:ff_chat /var/lib/ff_chat
sudo chmod 750 /var/lib/ff_chat
```

## 🐳 Docker Setup

### Basic Docker Setup
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV FF_CONFIG_PATH=/app/ff_preset_configs

# Expose port if needed
EXPOSE 8000

# Run verification on build
RUN python -c "from ff_storage_manager import FFStorageManager; print('Build verification passed')"

# Default command
CMD ["python", "-c", "print('Flatfile Chat Database ready')"]
```

### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  ff-chat-db:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FF_ENV=production
      - FF_CONFIG_PATH=/app/ff_preset_configs
    restart: unless-stopped
    
  # Optional: Add monitoring
  monitoring:
    image: prom/prometheus
    volumes:
      - ./monitoring:/etc/prometheus
    ports:
      - "9090:9090"
```

### Docker Commands
```bash
# Build image
docker build -t ff-chat-db .

# Run with volume mapping
docker run -v $(pwd)/data:/app/data ff-chat-db

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f ff-chat-db
```

## 🔐 Security Setup

### File Permissions (Production)
```bash
# Secure installation
sudo mkdir -p /opt/ff_chat
sudo chown root:ff_chat /opt/ff_chat
sudo chmod 750 /opt/ff_chat

# Data directory
sudo mkdir -p /var/lib/ff_chat
sudo chown ff_chat:ff_chat /var/lib/ff_chat
sudo chmod 700 /var/lib/ff_chat

# Log directory
sudo mkdir -p /var/log/ff_chat
sudo chown ff_chat:ff_chat /var/log/ff_chat
sudo chmod 755 /var/log/ff_chat
```

### Environment Variables
```bash
# .env file (don't commit to version control)
FF_CONFIG_PATH=/secure/path/to/configs
FF_DATA_PATH=/secure/path/to/data
FF_LOG_LEVEL=INFO
FF_ENVIRONMENT=production
```

## 🚀 Production Deployment

### Systemd Service (Linux)
```ini
# /etc/systemd/system/ff-chat-db.service
[Unit]
Description=Flatfile Chat Database Service
After=network.target

[Service]
Type=simple
User=ff_chat
Group=ff_chat
WorkingDirectory=/opt/ff_chat
Environment=FF_CONFIG_PATH=/etc/ff_chat
Environment=FF_DATA_PATH=/var/lib/ff_chat
ExecStart=/opt/ff_chat/venv/bin/python your_app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable ff-chat-db
sudo systemctl start ff-chat-db
sudo systemctl status ff-chat-db
```

### Process Manager (Alternative)
```bash
# Using supervisord
pip install supervisor

# /etc/supervisor/conf.d/ff-chat-db.conf
[program:ff-chat-db]
command=/path/to/venv/bin/python your_app.py
directory=/path/to/app
user=ff_chat
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/ff_chat/app.log
```

## 📊 Performance Optimization

### Memory Optimization
```python
config = load_config("production")

# Optimize memory usage
config.storage.message_cache_size = 1000
config.search.cache_size = 500
config.vector.cache_embeddings = True
config.compression.enable_compression = True
```

### Disk I/O Optimization
```python
# Enable compression to reduce disk usage
config.compression.compression_level = 6
config.storage.batch_write_size = 100

# Use faster serialization
config.storage.use_binary_format = True
```

## 🧪 Testing Your Installation

### Unit Tests
```bash
# Run basic tests
python -m pytest tests/test_core_functionality.py -v

# Run with coverage
python -m pytest --cov=. tests/
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/test_storage_integration.py -v

# Run performance tests
python -m pytest tests/test_performance.py -v --slow
```

### Load Testing
```python
# Simple load test
import asyncio
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config

async def load_test():
    config = load_config()
    storage = FFStorageManager(config)
    await storage.initialize()
    
    # Create many users concurrently
    tasks = []
    for i in range(100):
        task = storage.create_user(f"load_test_user_{i}")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"Created {sum(results)} users successfully")

asyncio.run(load_test())
```

## 🔍 Troubleshooting Installation

### Common Issues

#### ImportError: No module named 'ff_storage_manager'
```bash
# Solution 1: Check PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/flatfile_chat_database

# Solution 2: Install in development mode
pip install -e .
```

#### Permission Denied Errors
```bash
# Fix permissions
chmod -R 755 /path/to/installation
chown -R $USER:$USER /path/to/installation
```

#### Configuration Not Found
```bash
# Check config path
echo $FF_CONFIG_PATH

# Set config path
export FF_CONFIG_PATH=/path/to/configs
```

## ✅ Installation Checklist

- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created and activated
- [ ] Dependencies installed via pip
- [ ] Core imports working
- [ ] Configuration loading successfully
- [ ] Storage manager initializing
- [ ] Basic operations (create user, session, message) working
- [ ] File permissions set correctly
- [ ] Environment variables configured
- [ ] Testing suite running
- [ ] Production deployment (if applicable) configured

## 🎯 Next Steps

After successful installation:

1. **Configure for Your Environment**: See [Configuration Guide](03_CONFIGURATION.md)
2. **Learn Basic Operations**: Continue to [Basic Usage](04_BASIC_USAGE.md)
3. **Set Up Monitoring**: Configure logging and performance monitoring
4. **Plan Your Integration**: Review [API Reference](06_API_REFERENCE.md)
5. **Optimize Performance**: Check [Performance Guide](08_PERFORMANCE.md)

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in console output
2. **Run verification script**: Use the verification script above
3. **Check permissions**: Ensure proper file and directory permissions
4. **Review configuration**: Verify configuration files are valid JSON
5. **Test with minimal example**: Start with the simplest possible setup

The system is designed to be robust and provide helpful error messages to guide you through any issues.