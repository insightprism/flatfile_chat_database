# RESTful API Design Guide

## Introduction

This guide covers best practices for designing RESTful APIs that are intuitive, maintainable, and scalable. Following these principles will help you create APIs that developers love to use.

## Core REST Principles

### 1. Resource-Based URLs
Structure URLs around resources (nouns), not actions (verbs):

```
✅ Good:
GET    /users              # Get list of users
GET    /users/123          # Get specific user
POST   /users              # Create new user
PUT    /users/123          # Update user
DELETE /users/123          # Delete user

❌ Bad:
GET    /getUsers
POST   /createUser
PUT    /updateUser/123
DELETE /removeUser/123
```

### 2. Use HTTP Methods Appropriately

| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| GET | Retrieve data | Yes | Yes |
| POST | Create resource | No | No |
| PUT | Update/Replace resource | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Remove resource | Yes | No |

### 3. HTTP Status Codes

Use appropriate status codes to communicate the result:

```python
# Success responses
200 OK              # Successful GET, PUT, PATCH
201 Created         # Successful POST
204 No Content      # Successful DELETE

# Client error responses  
400 Bad Request     # Invalid request syntax
401 Unauthorized    # Authentication required
403 Forbidden       # Access denied
404 Not Found       # Resource doesn't exist
409 Conflict        # Resource conflict
422 Unprocessable Entity  # Validation errors

# Server error responses
500 Internal Server Error  # Generic server error
503 Service Unavailable   # Temporary server overload
```

## URL Design Patterns

### Resource Hierarchies
```
/users/123/orders           # Orders for specific user
/users/123/orders/456       # Specific order for user
/categories/tech/products   # Products in tech category
```

### Query Parameters for Filtering
```
GET /users?active=true&role=admin
GET /products?category=electronics&price_min=100&price_max=500
GET /orders?created_after=2023-01-01&status=shipped
```

### Pagination Parameters
```
GET /users?page=2&limit=50
GET /users?offset=100&limit=50
```

### Sorting Parameters
```
GET /users?sort=created_at&order=desc
GET /products?sort=price,name&order=asc,desc
```

## Request/Response Design

### Request Body Structure
```json
{
  "data": {
    "type": "user",
    "attributes": {
      "email": "user@example.com",
      "first_name": "John",
      "last_name": "Doe"
    },
    "relationships": {
      "organization": {
        "data": { "type": "organization", "id": "456" }
      }
    }
  }
}
```

### Response Body Structure
```json
{
  "data": {
    "type": "user",
    "id": "123",
    "attributes": {
      "email": "user@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "created_at": "2023-01-15T10:30:00Z"
    },
    "relationships": {
      "organization": {
        "data": { "type": "organization", "id": "456" }
      }
    }
  },
  "meta": {
    "created_at": "2023-01-15T10:30:00Z",
    "version": "1.0"
  }
}
```

### Error Response Structure
```json
{
  "errors": [
    {
      "id": "validation_error",
      "status": "422",
      "code": "INVALID_EMAIL",
      "title": "Invalid email address",
      "detail": "The email address format is not valid",
      "source": {
        "pointer": "/data/attributes/email"
      }
    }
  ]
}
```

## Authentication and Authorization

### API Key Authentication
```http
GET /api/users HTTP/1.1
Host: api.example.com
Authorization: Bearer your-api-key-here
```

### JWT Token Authentication
```http
GET /api/users HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### OAuth 2.0 Flow
```python
# Authorization endpoint
GET /oauth/authorize?
    response_type=code&
    client_id=your_client_id&
    redirect_uri=https://yourapp.com/callback&
    scope=read write&
    state=random_string

# Token endpoint
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=authorization_code_here&
redirect_uri=https://yourapp.com/callback&
client_id=your_client_id&
client_secret=your_client_secret
```

## Versioning Strategies

### URL Path Versioning
```
https://api.example.com/v1/users
https://api.example.com/v2/users
```

### Header Versioning
```http
GET /users HTTP/1.1
Host: api.example.com
Accept: application/vnd.api+json;version=2
```

### Query Parameter Versioning
```
https://api.example.com/users?version=2
```

## Rate Limiting

### Response Headers
```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
Retry-After: 3600
```

### Rate Limit Exceeded Response
```json
{
  "errors": [
    {
      "status": "429",
      "code": "RATE_LIMIT_EXCEEDED",
      "title": "Rate limit exceeded",
      "detail": "You have exceeded the rate limit of 1000 requests per hour"
    }
  ]
}
```

## Content Negotiation

### Accept Headers
```http
# JSON response
Accept: application/json

# XML response  
Accept: application/xml

# Specific API version
Accept: application/vnd.api+json;version=2
```

### Content-Type Headers
```http
# Sending JSON
Content-Type: application/json

# Sending form data
Content-Type: application/x-www-form-urlencoded

# Sending multipart data
Content-Type: multipart/form-data
```

## Caching

### Cache-Control Headers
```http
# Public cache for 1 hour
Cache-Control: public, max-age=3600

# Private cache for 30 minutes
Cache-Control: private, max-age=1800

# No caching
Cache-Control: no-cache, no-store, must-revalidate
```

### ETag for Conditional Requests
```http
# Server response
HTTP/1.1 200 OK
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"

# Client conditional request
GET /users/123 HTTP/1.1
If-None-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"

# Server response if unchanged
HTTP/1.1 304 Not Modified
```

## Security Best Practices

### HTTPS Everywhere
- Always use HTTPS in production
- Redirect HTTP to HTTPS
- Use HSTS headers

### Input Validation
```python
from marshmallow import Schema, fields, validate

class UserSchema(Schema):
    email = fields.Email(required=True)
    age = fields.Integer(validate=validate.Range(min=0, max=120))
    role = fields.String(validate=validate.OneOf(['user', 'admin', 'moderator']))

# Usage
schema = UserSchema()
try:
    result = schema.load(request_data)
except ValidationError as err:
    return jsonify({'errors': err.messages}), 400
```

### SQL Injection Prevention
```python
# ❌ Vulnerable to SQL injection
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ Safe parameterized query
query = "SELECT * FROM users WHERE email = %s"
cursor.execute(query, (email,))
```

### Cross-Origin Resource Sharing (CORS)
```http
# Response headers
Access-Control-Allow-Origin: https://yourdomain.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
```

## Documentation

### OpenAPI/Swagger Specification
```yaml
openapi: 3.0.0
info:
  title: User Management API
  version: 1.0.0
  description: API for managing users

paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUser'
      responses:
        '201':
          description: User created successfully

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        email:
          type: string
          format: email
        first_name:
          type: string
        last_name:
          type: string
        created_at:
          type: string
          format: date-time
```

## Testing APIs

### Unit Testing
```python
import pytest
from unittest.mock import Mock, patch
from myapi import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

def test_get_user(client):
    response = client.get('/api/users/123')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['data']['id'] == '123'
    assert 'email' in data['data']['attributes']

def test_create_user(client):
    user_data = {
        'data': {
            'type': 'user',
            'attributes': {
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        }
    }
    
    response = client.post('/api/users', json=user_data)
    assert response.status_code == 201
    
    data = response.get_json()
    assert data['data']['attributes']['email'] == 'test@example.com'
```

### Integration Testing
```python
import requests
import pytest

@pytest.fixture
def api_base_url():
    return "http://localhost:5000/api"

def test_user_workflow(api_base_url):
    # Create user
    user_data = {
        'data': {
            'type': 'user',
            'attributes': {
                'email': 'integration@example.com',
                'first_name': 'Integration',
                'last_name': 'Test'
            }
        }
    }
    
    response = requests.post(f"{api_base_url}/users", json=user_data)
    assert response.status_code == 201
    
    user_id = response.json()['data']['id']
    
    # Get user
    response = requests.get(f"{api_base_url}/users/{user_id}")
    assert response.status_code == 200
    
    # Update user
    update_data = {
        'data': {
            'type': 'user',
            'attributes': {
                'first_name': 'Updated'
            }
        }
    }
    
    response = requests.patch(f"{api_base_url}/users/{user_id}", json=update_data)
    assert response.status_code == 200
    
    # Delete user
    response = requests.delete(f"{api_base_url}/users/{user_id}")
    assert response.status_code == 204
```

## Performance Optimization

### Database Query Optimization
```python
# ❌ N+1 query problem
users = User.query.all()
for user in users:
    print(user.organization.name)  # Triggers additional query

# ✅ Eager loading
users = User.query.options(joinedload(User.organization)).all()
for user in users:
    print(user.organization.name)  # No additional queries
```

### Response Compression
```python
from flask import Flask
from flask_compress import Compress

app = Flask(__name__)
Compress(app)  # Automatically compresses responses
```

### Async/Await for I/O Operations
```python
import asyncio
import aiohttp
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}/external-data")
async def get_user_external_data(user_id: str):
    async with aiohttp.ClientSession() as session:
        # Multiple async calls
        tasks = [
            fetch_user_profile(session, user_id),
            fetch_user_orders(session, user_id),
            fetch_user_preferences(session, user_id)
        ]
        
        profile, orders, preferences = await asyncio.gather(*tasks)
        
        return {
            "profile": profile,
            "orders": orders,
            "preferences": preferences
        }
```

## API Monitoring and Logging

### Request/Response Logging
```python
import logging
import time
from flask import Flask, request, g

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    
    logger.info({
        'method': request.method,
        'url': request.url,
        'status_code': response.status_code,
        'duration_ms': round(duration * 1000, 2),
        'user_agent': request.headers.get('User-Agent'),
        'ip_address': request.remote_addr
    })
    
    return response
```

### Health Check Endpoints
```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }

@app.route('/health/detailed')
def detailed_health_check():
    checks = {
        'database': check_database_connection(),
        'redis': check_redis_connection(),
        'external_api': check_external_api()
    }
    
    all_healthy = all(checks.values())
    
    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }, 200 if all_healthy else 503
```

## Deployment Considerations

### Environment Configuration
```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    database_url: str = os.getenv('DATABASE_URL')
    redis_url: str = os.getenv('REDIS_URL') 
    secret_key: str = os.getenv('SECRET_KEY')
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    def __post_init__(self):
        if not self.secret_key:
            raise ValueError("SECRET_KEY environment variable is required")
```

### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

This comprehensive guide provides the foundation for building well-designed RESTful APIs. Remember to always consider your specific use case and requirements when applying these principles.