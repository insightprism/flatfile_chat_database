"""
Quick fix for Python path issues in the demo notebook.

Add this to the top of your notebook BEFORE the imports:
"""

import sys
import os
from pathlib import Path

# Get the correct project directory (flatfile_chat_database_v2)
current_dir = Path.cwd()
if current_dir.name == 'demo':
    project_dir = current_dir.parent
else:
    project_dir = current_dir

# Ensure we're using the v2 project directory
project_v2_path = str(project_dir)

# Remove any old flatfile_chat_database paths to avoid conflicts
sys.path = [p for p in sys.path if 'flatfile_chat_database' not in p or 'flatfile_chat_database_v2' in p]

# Add the correct path first (so it takes priority)
if project_v2_path not in sys.path:
    sys.path.insert(0, project_v2_path)

print(f"✅ Using project directory: {project_v2_path}")
print(f"📁 Current working directory: {current_dir}")
print(f"🐍 Python path prioritizes v2 project")

# Verify correct imports
try:
    from backends import FlatfileBackend
    print(f"✅ Backend loaded from: {FlatfileBackend.__module__}")
    
    # Check if it's the correct backend
    import inspect
    backend_file = inspect.getfile(FlatfileBackend)
    if 'flatfile_chat_database_v2' in backend_file:
        print("✅ Using correct v2 backend")
    else:
        print(f"⚠️ WARNING: Using wrong backend from {backend_file}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")