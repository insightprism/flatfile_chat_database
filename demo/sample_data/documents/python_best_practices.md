# Python Best Practices Guide

## Code Style and Formatting

### PEP 8 Compliance
Follow Python Enhancement Proposal 8 for consistent code style:

```python
# Good: Clear variable names and proper spacing
def calculate_monthly_payment(principal, annual_rate, years):
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    return principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

# Bad: Poor naming and spacing
def calc(p,r,y):
    mr=r/12
    np=y*12
    return p*(mr*(1+mr)**np)/((1+mr)**np-1)
```

### Type Hints
Use type hints for better code documentation and IDE support:

```python
from typing import List, Dict, Optional, Union

def process_user_data(
    users: List[Dict[str, Union[str, int]]], 
    filter_active: bool = True
) -> Optional[List[str]]:
    """Process user data and return list of usernames."""
    if not users:
        return None
    
    result = []
    for user in users:
        if filter_active and user.get('active', False):
            result.append(user['username'])
    
    return result
```

## Error Handling

### Use Specific Exceptions
```python
# Good: Specific exception handling
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Configuration file not found")
    config = {}
except json.JSONDecodeError:
    print("Invalid JSON in configuration file")
    config = {}

# Bad: Catching all exceptions
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except Exception:
    config = {}
```

### Custom Exceptions
```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass

def validate_email(email: str) -> str:
    if '@' not in email:
        raise ValidationError(f"Invalid email format: {email}")
    return email.lower()
```

## Performance Optimization

### Use List Comprehensions
```python
# Good: List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Bad: Traditional loop
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)
```

### Generator Expressions for Large Data
```python
# Memory efficient for large datasets
def process_large_file(filename):
    return (line.strip().upper() for line in open(filename) if line.strip())

# Use it
for processed_line in process_large_file('huge_file.txt'):
    # Process one line at a time without loading entire file
    print(processed_line)
```

## Testing Best Practices

### Unit Testing with pytest
```python
import pytest
from mymodule import calculate_monthly_payment

def test_monthly_payment_calculation():
    # Test normal case
    payment = calculate_monthly_payment(100000, 0.05, 30)
    assert abs(payment - 536.82) < 0.01
    
    # Test edge cases
    with pytest.raises(ValueError):
        calculate_monthly_payment(-100000, 0.05, 30)
    
    with pytest.raises(ValueError):
        calculate_monthly_payment(100000, -0.05, 30)

# Fixtures for reusable test data
@pytest.fixture
def sample_users():
    return [
        {'username': 'alice', 'active': True},
        {'username': 'bob', 'active': False},
        {'username': 'carol', 'active': True}
    ]

def test_process_user_data(sample_users):
    active_users = process_user_data(sample_users, filter_active=True)
    assert active_users == ['alice', 'carol']
```

## Code Organization

### Project Structure
```
project/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   └── test_services/
├── requirements.txt
├── setup.py
└── README.md
```

### Use `__init__.py` for Clean Imports
```python
# src/models/__init__.py
from .user import User, UserRole
from .product import Product

__all__ = ['User', 'UserRole', 'Product']

# Now you can import like this:
from models import User, Product
```

## Security Best Practices

### Never Hardcode Secrets
```python
import os
from pathlib import Path

# Good: Use environment variables or config files
DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')

# Better: Use a config management system
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
database_url = config.get('database', 'url')
```

### Validate Input Data
```python
import re
from typing import Optional

def validate_user_input(username: str, email: str) -> Dict[str, str]:
    errors = {}
    
    # Validate username
    if not username or len(username) < 3:
        errors['username'] = 'Username must be at least 3 characters'
    elif not re.match(r'^[a-zA-Z0-9_]+$', username):
        errors['username'] = 'Username can only contain letters, numbers, and underscores'
    
    # Validate email
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        errors['email'] = 'Invalid email format'
    
    return errors
```

## Memory Management

### Use Context Managers
```python
# Good: Automatic resource cleanup
def process_file(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

# Good: Custom context manager
from contextlib import contextmanager

@contextmanager
def database_transaction():
    conn = get_database_connection()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()

# Usage
with database_transaction() as conn:
    conn.execute("INSERT INTO users ...")
```

### Avoid Memory Leaks
```python
# Good: Clear references when done
def process_large_dataset():
    data = load_huge_dataset()
    processed = expensive_processing(data)
    
    # Clear reference to original data
    del data
    
    return processed

# Good: Use weak references when appropriate
import weakref

class Observer:
    def __init__(self):
        self._observers = weakref.WeakSet()
    
    def add_observer(self, observer):
        self._observers.add(observer)
```

## Concurrency and Async Programming

### Use asyncio for I/O-bound Tasks
```python
import asyncio
import aiohttp
from typing import List

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls(urls: List[str]) -> List[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = ['http://example.com', 'http://google.com']
results = asyncio.run(fetch_multiple_urls(urls))
```

### Threading for CPU-bound Tasks
```python
import concurrent.futures
from typing import List

def cpu_bound_task(n: int) -> int:
    # Simulate CPU-intensive work
    return sum(i * i for i in range(n))

def process_parallel(numbers: List[int]) -> List[int]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(cpu_bound_task, numbers))
    return results
```

## Documentation

### Use Docstrings
```python
def calculate_compound_interest(
    principal: float, 
    rate: float, 
    time: float, 
    compound_frequency: int = 1
) -> float:
    """
    Calculate compound interest.
    
    Args:
        principal: The initial amount of money
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time period in years
        compound_frequency: Number of times interest is compounded per year
        
    Returns:
        The final amount after compound interest
        
    Raises:
        ValueError: If any parameter is negative
        
    Examples:
        >>> calculate_compound_interest(1000, 0.05, 2)
        1102.5
        >>> calculate_compound_interest(1000, 0.05, 2, 4)
        1104.49
    """
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("All parameters must be non-negative")
    
    return principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
```

## Dependency Management

### requirements.txt vs setup.py
```python
# setup.py for installable packages
from setuptools import setup, find_packages

setup(
    name="myproject",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0"
        ]
    }
)
```

### Use Virtual Environments
```bash
# Create virtual environment
python -m venv myproject_env

# Activate (Linux/Mac)
source myproject_env/bin/activate

# Activate (Windows)
myproject_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Freeze current dependencies
pip freeze > requirements.txt
```

## Configuration Management

### Use Configuration Files
```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    database_url: str
    api_key: str
    debug: bool = False
    log_level: str = "INFO"
    max_connections: int = 100
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            database_url=os.getenv('DATABASE_URL', 'sqlite:///app.db'),
            api_key=os.getenv('API_KEY', ''),
            debug=os.getenv('DEBUG', 'False').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            max_connections=int(os.getenv('MAX_CONNECTIONS', '100'))
        )

# Usage
config = Config.from_env()
```

## Logging Best Practices

### Structured Logging
```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Structured logging
def log_user_action(user_id: str, action: str, **kwargs):
    log_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'action': action,
        **kwargs
    }
    logger.info(json.dumps(log_data))

# Usage
log_user_action('user123', 'login', ip_address='192.168.1.1', success=True)
```

## Code Quality Tools

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 21.9b0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  
  - repo: https://github.com/pycqa/isort
    rev: 5.9.3
    hooks:
      - id: isort
```

### Makefile for Common Tasks
```makefile
.PHONY: test lint format install clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --coverage

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
```

This guide covers essential Python best practices that will help you write cleaner, more maintainable, and more efficient code. Remember that these are guidelines - adapt them to your specific project needs and team preferences.