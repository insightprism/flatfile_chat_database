#!/usr/bin/env python3
"""
Fix all imports in test files to work from the new test directory.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix the sys.path import in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match the old import
    old_pattern = r'sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\)\)'
    new_pattern = 'sys.path.insert(0, str(Path(__file__).parent.parent))'
    
    # Check if already fixed
    if 'parent.parent' in content:
        print(f"✓ {filepath.name} - already fixed")
        return False
    
    # Replace the pattern
    updated_content = re.sub(old_pattern, new_pattern, content)
    
    if updated_content != content:
        with open(filepath, 'w') as f:
            f.write(updated_content)
        print(f"✓ {filepath.name} - fixed import path")
        return True
    else:
        print(f"⚠ {filepath.name} - no matching pattern found")
        return False

def main():
    test_dir = Path(__file__).parent
    python_files = list(test_dir.glob("*.py"))
    
    print(f"Fixing imports in {len(python_files)} Python files...\n")
    
    fixed_count = 0
    for file in python_files:
        if file.name != "fix_all_imports.py":
            if fix_imports_in_file(file):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()