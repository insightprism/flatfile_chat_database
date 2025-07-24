# PrismMind Package Analysis and Setup Requirements

## Current Structure Analysis

### ✅ **What's Already There:**
- **Complete codebase** with organized module structure
- **Dependencies defined** in `requirements.txt` (295 packages)
- **Proper directory structure** with logical separation:
  - `pm_engines/` - Core processing engines
  - `pm_config/` - Configuration management
  - `pm_utils/` - Utility functions and helpers
  - `pm_video_audio/` - Media processing capabilities
  - `pm_database/` - Database integration
  - `pm_user_guide/` - Documentation and examples

### ❌ **What's Missing for Package Installation:**

1. **Empty `__init__.py` files** - No exports defined
2. **No `setup.py` or `pyproject.toml`** - No installation configuration
3. **Inconsistent import patterns** - Mix of relative and absolute imports
4. **Missing package metadata** - No version, author, description

## Import Pattern Issues

### Current Import Problems:
```python
# Some files use:
from pm_engines.pm_base_engine import PmBaseEngine  # Relative
from PrismMind_v2.pm_config.pm_llm_engine_config import pm_get_llm_config  # Absolute
from pm_utils.pm_trace_handler_log_dec import pm_trace_handler_log_dec  # Relative
```

### Dependencies Between Modules:
- `pm_engines` depends on: `pm_config`, `pm_utils`
- `pm_config` depends on: None (base layer)
- `pm_utils` depends on: `pm_config`
- All modules have external dependencies from `requirements.txt`

## Package Setup Requirements

### 1. **Package Structure** ✅ (Already Good)
```
PrismMind_v2/
├── __init__.py              # Needs content
├── setup.py                 # Missing - need to create
├── requirements.txt         # ✅ Exists
├── pm_engines/
│   ├── __init__.py         # Empty - needs exports
│   └── *.py                # ✅ Engine files exist
├── pm_config/
│   ├── __init__.py         # Empty - needs exports  
│   └── *.py                # ✅ Config files exist
├── pm_utils/
│   ├── __init__.py         # Empty - needs exports
│   └── *.py                # ✅ Utility files exist
└── other modules...
```

### 2. **Critical Files to Create:**

#### A. `setup.py` - Package Installation
```python
from setuptools import setup, find_packages

setup(
    name="prismmind",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[...],  # From requirements.txt
    python_requires=">=3.8",
)
```

#### B. Update `__init__.py` files with proper exports
```python
# PrismMind_v2/__init__.py
from .pm_engines import *
from .pm_config import *
from .pm_utils import *

# pm_engines/__init__.py  
from .pm_run_engine_chain import pm_run_engine_chain
from .pm_base_engine import PmBaseEngine
# ... other key exports
```

### 3. **Import Fixing Strategy:**

#### Option A: **Make all imports relative** (Recommended)
- Change all imports to use relative paths within package
- Example: `from pm_engines.pm_base_engine import PmBaseEngine`

#### Option B: **Use absolute package imports**
- Change all imports to use full package name
- Example: `from prismmind.pm_engines.pm_base_engine import PmBaseEngine`

## Implementation Plan

### Phase 1: **Basic Package Setup** (Essential)
1. Create `setup.py` with dependencies from `requirements.txt`
2. Update main `__init__.py` with key exports
3. Fix critical import issues in main engines

### Phase 2: **Complete Package Integration** (Recommended)
1. Update all module `__init__.py` files with proper exports
2. Standardize all import patterns
3. Add package metadata and documentation

### Phase 3: **Installation Testing** (Validation)
1. Install package in development mode: `pip install -e .`
2. Test imports from external code
3. Verify our flatfile integration works

## Dependencies Analysis

### **Heavy Dependencies** (from requirements.txt):
- **AI/ML**: torch, transformers, spacy, sentence-transformers, ollama
- **PDF Processing**: pdfplumber, pypdfium2, python-docx
- **OCR**: pytesseract, easyocr, python-doctr
- **Web**: playwright, requests, fastapi
- **Data**: pandas, numpy, scipy

### **Core Dependencies for Basic Functionality:**
```
# Essential for engine chain
torch>=2.0.0
spacy>=3.8.0
transformers>=4.50.0

# File processing
pdfplumber>=0.11.0
pytesseract>=0.3.13
python-docx>=1.1.0

# Async and utilities
aiofiles>=24.0.0
asyncio
```

## Quick Fix Implementation

The **minimal viable package** needs:

1. **`setup.py`** - Basic installation capability
2. **Main `__init__.py`** - Export key functions like `pm_run_engine_chain`
3. **Engine `__init__.py`** - Export main engine classes
4. **Fix top 5 import issues** - In most critical files

This would make PrismMind installable and importable, enabling our flatfile integration to work.

## Estimated Implementation Time

- **Quick fix** (minimal viable): ~30 minutes
- **Complete package setup**: ~2 hours  
- **Full testing and validation**: ~1 hour

The quick fix is sufficient to enable the flatfile PrismMind integration to work properly.