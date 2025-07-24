#!/usr/bin/env python3
"""
Update imports in test files to work from the new location.
"""

import sys
import os

# Add parent directory to path so we can import from flatfile_chat_database
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print("Import paths updated!")
print(f"Added to sys.path: {parent_dir}")
print("\nYou can now run the test files from this directory.")