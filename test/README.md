# Flatfile Chat Database - Test Directory

This directory contains all test files and demo applications for the Flatfile Chat Database system.

## Directory Structure

```
flatfile_chat_database_test/
├── Core Test Files
│   ├── test_flatfile.py              # Basic functionality tests
│   ├── test_advanced_search.py       # Advanced search tests
│   ├── test_phase4_features.py       # Production features tests
│   ├── test_complete_system.py       # Comprehensive system tests
│   ├── test_storage_integration.py   # Storage integration tests
│   └── test_utils.py                 # Test utilities
│
├── Demo Applications
│   ├── interactive_chat_demo.py      # Full interactive CLI chat demo
│   ├── quick_test_demo.py           # Quick demo of all features
│   ├── demo_interactive_test.py     # Automated demo test
│   └── demo_runner.py               # Demo runner with simulated input
│
├── CLI Demo Components
│   ├── chat_config.py               # Configuration for CLI demo
│   ├── chat_ui.py                   # UI utilities for CLI
│   └── chat_commands.py             # Command handlers for CLI
│
├── Test Runners
│   ├── test_runner.py               # Interactive test menu
│   └── run_tests.py                 # Run all tests script
│
├── Documentation
│   ├── README.md                    # This file
│   ├── TESTING_GUIDE.md            # Comprehensive testing guide
│   ├── QUICK_TEST_CHECKLIST.md     # Quick testing checklist
│   ├── TEST_GUIDE.md               # Original test guide
│   └── IMPLEMENTATION_SUMMARY.md    # Implementation summary
│
└── Test Data Directories (created during tests)
    ├── demo_chat_data/             # Demo data
    ├── test_chat_data/             # Test data
    └── chat_data/                  # Interactive demo data
```

## Quick Start

### 1. Run Quick Demo
```bash
python3 quick_test_demo.py
```

### 2. Run Interactive Test Menu
```bash
python3 test_runner.py
# Then select from the menu options
```

### 3. Run Interactive Chat Demo
```bash
python3 interactive_chat_demo.py
```

### 4. Run All Tests
```bash
python3 run_tests.py
```

## Important Notes

- All test files have been updated to import from the parent directory
- The main `flatfile_chat_database` package remains in the parent directory
- Test data is created in subdirectories within this test directory
- All imports use: `sys.path.insert(0, str(Path(__file__).parent.parent))`

## Common Commands for Interactive Demo

When running `interactive_chat_demo.py`:
- `/help` - Show all commands
- `/search <query>` - Search messages
- `/stats` - Show statistics
- `/sessions` - List sessions
- `/export` - Export current session
- `/exit` - Exit the demo