# Configuration Guide

## Overview

The modular chat platform uses a hierarchical, JSON-based configuration system that supports environment variables, module-specific settings, and runtime overrides. This guide explains how to configure the system for different deployment scenarios.

## Configuration Hierarchy

Configuration values are loaded in the following priority order (highest to lowest):

1. **Environment Variables** - Override any configuration
2. **Runtime Overrides** - Programmatic configuration changes
3. **Environment-Specific Files** - production.json, development.json
4. **Module-Specific Files** - text_chat.json, memory.json
5. **Default Configuration** - default.json

## Directory Structure

```
config/
├── default.json              # Base configuration for all environments
├── environments/             # Environment-specific overrides
│   ├── development.json
│   ├── production.json
│   └── test.json
├── modules/                  # Module-specific configurations
│   ├── text_chat.json
│   ├── memory.json
│   ├── rag.json
│   ├── multi_agent.json
│   ├── tool_use.json
│   ├── persona.json
│   ├── multimodal.json
│   ├── topic_router.json
│   └── trace_logger.json
└── schemas/                  # JSON schemas for validation
    ├── module.schema.json
    └── config.schema.json
```

## Base Configuration

### default.json
```json
{
  "platform": {
    "name": "Modular Chat Platform",
    "version": "1.0.0",
    "environment": "${CHAT_ENV:-development}"
  },
  "core": {
    "message_bus": {
      "max_handlers_per_event": 100,
      "event_timeout_seconds": 30,
      "dead_letter_queue": true
    },
    "module_loader": {
      "lazy_loading": true,
      "health_check_interval": 60,
      "module_timeout_seconds": 10
    }
  },
  "storage": {
    "backend": "${STORAGE_BACKEND:-flatfile}",
    "base_path": "${STORAGE_PATH:-./data}",
    "backends": {
      "flatfile": {
        "compression": true,
        "atomic_writes": true,
        "file_permissions": "0600"
      },
      "s3": {
        "bucket": "${S3_BUCKET}",
        "region": "${AWS_REGION:-us-east-1}",
        "encryption": "AES256"
      }
    }
  },
  "security": {
    "encryption": {
      "algorithm": "AES-256-GCM",
      "key_rotation_days": 90
    },
    "authentication": {
      "provider": "${AUTH_PROVIDER:-local}",
      "session_timeout_minutes": 60
    }
  },
  "monitoring": {
    "metrics": {
      "enabled": true,
      "export_interval_seconds": 60,
      "exporters": ["prometheus", "cloudwatch"]
    },
    "logging": {
      "level": "${LOG_LEVEL:-INFO}",
      "format": "json",
      "outputs": ["console", "file"]
    }
  }
}
```

## Module Configurations

### Text Chat Module (modules/text_chat.json)
```json
{
  "text_chat": {
    "enabled": true,
    "max_message_length": 4096,
    "max_messages_per_session": 10000,
    "message_validation": {
      "strict_mode": true,
      "allowed_roles": ["user", "assistant", "system"],
      "require_session_id": true
    },
    "streaming": {
      "enabled": true,
      "chunk_size": 100,
      "timeout_seconds": 30,
      "backpressure_buffer": 1000
    },
    "rate_limiting": {
      "enabled": true,
      "strategy": "sliding_window",
      "limits": {
        "messages_per_minute": 60,
        "tokens_per_minute": 100000,
        "sessions_per_hour": 10
      }
    },
    "session": {
      "idle_timeout_minutes": 30,
      "max_sessions_per_user": 100,
      "auto_save_interval_seconds": 30
    },
    "ai_providers": {
      "default": "${AI_PROVIDER:-openai}",
      "openai": {
        "api_key": "${OPENAI_API_KEY}",
        "model": "${OPENAI_MODEL:-gpt-4}",
        "temperature": 0.7,
        "max_tokens": 2000
      },
      "anthropic": {
        "api_key": "${ANTHROPIC_API_KEY}",
        "model": "${ANTHROPIC_MODEL:-claude-3-opus}",
        "temperature": 0.7
      }
    }
  }
}
```

### Memory Module (modules/memory.json)
```json
{
  "memory": {
    "enabled": true,
    "short_term": {
      "provider": "redis",
      "max_items": 1000,
      "ttl_seconds": 3600,
      "compression": {
        "enabled": true,
        "algorithm": "lz4",
        "threshold_bytes": 1024
      }
    },
    "long_term": {
      "provider": "flatfile",
      "max_items_per_user": 10000,
      "consolidation": {
        "enabled": true,
        "schedule": "0 */6 * * *",
        "strategies": ["importance", "frequency", "recency"],
        "min_importance_score": 0.5
      }
    },
    "working_memory": {
      "max_size_mb": 100,
      "eviction_policy": "LRU",
      "persistence": false
    },
    "search": {
      "algorithm": "hybrid",
      "embedding_cache_size": 10000,
      "max_results": 50,
      "min_relevance_score": 0.7,
      "rerank": {
        "enabled": true,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
      }
    }
  }
}
```

### RAG Module (modules/rag.json)
```json
{
  "rag": {
    "enabled": true,
    "document_processing": {
      "supported_formats": [
        "pdf", "docx", "txt", "md", "html", "csv", "json"
      ],
      "max_file_size_mb": 50,
      "ocr": {
        "enabled": true,
        "languages": ["en", "es", "fr", "de"]
      }
    },
    "chunking": {
      "strategy": "${CHUNKING_STRATEGY:-semantic}",
      "chunk_size": 512,
      "chunk_overlap": 50,
      "min_chunk_size": 100,
      "strategies": {
        "fixed": {
          "size": 512
        },
        "semantic": {
          "model": "sentence-transformers/all-MiniLM-L6-v2",
          "similarity_threshold": 0.8
        },
        "recursive": {
          "separators": ["\n\n", "\n", ". ", " "],
          "chunk_size": 1000
        }
      }
    },
    "embedding": {
      "provider": "${EMBEDDING_PROVIDER:-openai}",
      "providers": {
        "openai": {
          "model": "text-embedding-3-small",
          "dimension": 1536,
          "batch_size": 100
        },
        "sentence_transformers": {
          "model": "all-MiniLM-L6-v2",
          "dimension": 384,
          "device": "cuda"
        }
      }
    },
    "vector_store": {
      "provider": "${VECTOR_STORE:-faiss}",
      "providers": {
        "faiss": {
          "index_type": "IVF1024,Flat",
          "metric": "cosine",
          "gpu": false
        },
        "pinecone": {
          "api_key": "${PINECONE_API_KEY}",
          "environment": "${PINECONE_ENV}",
          "index_name": "chat-platform"
        }
      }
    },
    "retrieval": {
      "top_k": 5,
      "rerank": true,
      "max_tokens_context": 2000,
      "include_metadata": true
    }
  }
}
```

### Multi-Agent Module (modules/multi_agent.json)
```json
{
  "multi_agent": {
    "enabled": true,
    "max_agents": 50,
    "max_agents_per_conversation": 10,
    "agent_types": {
      "specialist": {
        "timeout_seconds": 30,
        "max_retries": 3
      },
      "coordinator": {
        "can_create_agents": true,
        "can_terminate_agents": true
      },
      "observer": {
        "read_only": true
      }
    },
    "coordination": {
      "strategies": ["round_robin", "consensus", "leader_based", "market_based"],
      "default_strategy": "consensus",
      "timeout_seconds": 300,
      "max_turns": 50
    },
    "communication": {
      "protocol": "structured",
      "message_format": "json",
      "max_message_size_kb": 64,
      "broadcast_enabled": true,
      "channels": ["global", "team", "private"]
    },
    "consensus": {
      "required_agreement": 0.7,
      "voting_enabled": true,
      "max_iterations": 5,
      "tie_breaker": "coordinator"
    }
  }
}
```

### Tool Use Module (modules/tool_use.json)
```json
{
  "tool_use": {
    "enabled": true,
    "max_tools": 100,
    "max_concurrent_executions": 10,
    "execution": {
      "timeout_seconds": 30,
      "retry_policy": {
        "max_attempts": 3,
        "backoff_multiplier": 2,
        "max_backoff_seconds": 60
      },
      "circuit_breaker": {
        "enabled": true,
        "failure_threshold": 5,
        "recovery_timeout": 60
      }
    },
    "security": {
      "sandboxed": true,
      "network_policy": "restricted",
      "allowed_domains": [
        "api.openai.com",
        "api.anthropic.com",
        "*.amazonaws.com"
      ],
      "blocked_domains": [
        "localhost",
        "127.0.0.1",
        "*.internal"
      ],
      "authentication": {
        "required": true,
        "methods": ["api_key", "oauth2", "jwt"]
      }
    },
    "rate_limiting": {
      "per_tool": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000
      },
      "global": {
        "requests_per_minute": 1000,
        "cost_limit_per_hour": 10.0
      }
    },
    "discovery": {
      "enabled": true,
      "registry_url": "${TOOL_REGISTRY_URL}",
      "cache_ttl_seconds": 3600,
      "auto_register": false
    }
  }
}
```

## Environment-Specific Configurations

### Development Environment (environments/development.json)
```json
{
  "platform": {
    "debug": true,
    "hot_reload": true
  },
  "storage": {
    "base_path": "./dev_data"
  },
  "monitoring": {
    "logging": {
      "level": "DEBUG",
      "outputs": ["console"]
    }
  },
  "security": {
    "authentication": {
      "provider": "mock",
      "bypass_auth": true
    }
  },
  "modules": {
    "trace_logger": {
      "capture_all_events": true,
      "include_payload": true
    }
  }
}
```

### Production Environment (environments/production.json)
```json
{
  "platform": {
    "debug": false,
    "hot_reload": false
  },
  "storage": {
    "backend": "s3",
    "base_path": "s3://prod-chat-platform"
  },
  "monitoring": {
    "logging": {
      "level": "WARNING",
      "outputs": ["cloudwatch", "s3"]
    },
    "alerts": {
      "enabled": true,
      "channels": ["pagerduty", "slack"]
    }
  },
  "security": {
    "authentication": {
      "provider": "cognito",
      "mfa_required": true
    },
    "encryption": {
      "kms_key_id": "${KMS_KEY_ID}"
    }
  },
  "performance": {
    "caching": {
      "provider": "elasticache",
      "ttl_seconds": 300
    },
    "cdn": {
      "enabled": true,
      "provider": "cloudfront"
    }
  }
}
```

## Environment Variables

### Core Environment Variables
```bash
# Platform
CHAT_ENV=production                    # Environment name
CHAT_CONFIG_PATH=/etc/chat/config      # Configuration directory path

# Storage
STORAGE_BACKEND=s3                     # Storage backend type
STORAGE_PATH=s3://my-bucket/data       # Storage location
S3_BUCKET=my-chat-bucket               # S3 bucket name
AWS_REGION=us-east-1                   # AWS region

# Security
AUTH_PROVIDER=cognito                  # Authentication provider
JWT_SECRET=your-secret-key             # JWT signing key
ENCRYPTION_KEY=base64-encoded-key      # Data encryption key

# AI Providers
OPENAI_API_KEY=sk-...                  # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...           # Anthropic API key
AI_PROVIDER=openai                     # Default AI provider

# Monitoring
LOG_LEVEL=INFO                         # Logging level
METRICS_ENABLED=true                   # Enable metrics collection
TRACE_SAMPLING_RATE=0.1                # Trace sampling rate

# Performance
CACHE_PROVIDER=redis                   # Cache provider
REDIS_URL=redis://localhost:6379       # Redis connection URL
MAX_WORKERS=10                         # Maximum worker threads
```

### Module-Specific Environment Variables
```bash
# Text Chat
TEXT_CHAT_MAX_TOKENS=4096              # Maximum tokens per message
TEXT_CHAT_STREAMING=true               # Enable streaming responses

# Memory
MEMORY_PROVIDER=redis                  # Memory storage provider
MEMORY_TTL_SECONDS=3600                # Short-term memory TTL

# RAG
EMBEDDING_PROVIDER=openai              # Embedding provider
VECTOR_STORE=pinecone                  # Vector store provider
CHUNKING_STRATEGY=semantic             # Document chunking strategy

# Multi-Agent
MAX_AGENTS_PER_CONVERSATION=5          # Maximum agents in conversation
COORDINATION_STRATEGY=consensus        # Default coordination strategy

# Tool Use
TOOL_REGISTRY_URL=https://tools.api    # Tool registry endpoint
TOOL_SANDBOX_ENABLED=true              # Enable tool sandboxing
```

## Configuration Loading

### Programmatic Configuration
```python
from pathlib import Path
import json
import os

class ConfigLoader:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config = {}
        
    def load(self, environment: str = None):
        """Load configuration for environment."""
        # Determine environment
        env = environment or os.getenv("CHAT_ENV", "development")
        
        # Load in order
        self._load_file("default.json")
        self._load_file(f"environments/{env}.json")
        self._load_modules()
        self._apply_env_vars()
        
        return self.config
    
    def _load_file(self, filename: str):
        """Load and merge configuration file."""
        file_path = self.config_dir / filename
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
                self.config = deep_merge(self.config, data)
    
    def _load_modules(self):
        """Load all module configurations."""
        modules_dir = self.config_dir / "modules"
        for module_file in modules_dir.glob("*.json"):
            self._load_file(f"modules/{module_file.name}")
    
    def _apply_env_vars(self):
        """Apply environment variable overrides."""
        self.config = replace_env_vars(self.config)
```

### Environment Variable Substitution
```python
def replace_env_vars(config: dict) -> dict:
    """Replace ${VAR:-default} patterns with environment values."""
    import re
    
    def replace_value(value):
        if isinstance(value, str):
            # Match ${VAR:-default} pattern
            pattern = r'\$\{([^:}]+)(?::-([^}]+))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.getenv(var_name, default)
            
            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: replace_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_value(v) for v in value]
        return value
    
    return replace_value(config)
```

## Module Configuration Access

### Accessing Configuration in Modules
```python
# In module initialization
async def initialize(config: dict):
    """Initialize module with configuration."""
    module_config = config.get("text_chat", {})
    
    # Access nested configuration
    max_messages = module_config.get("max_messages_per_session", 10000)
    streaming_enabled = module_config.get("streaming", {}).get("enabled", True)
    
    # Access with defaults
    rate_limits = module_config.get("rate_limiting", {})
    msg_per_min = rate_limits.get("limits", {}).get("messages_per_minute", 60)
```

### Runtime Configuration Updates
```python
class ConfigurableModule:
    def __init__(self, initial_config: dict):
        self.config = initial_config
        self._config_watchers = []
    
    def update_config(self, updates: dict):
        """Update configuration at runtime."""
        self.config = deep_merge(self.config, updates)
        self._notify_watchers()
    
    def watch_config(self, callback):
        """Register configuration change callback."""
        self._config_watchers.append(callback)
    
    def _notify_watchers(self):
        """Notify all configuration watchers."""
        for watcher in self._config_watchers:
            watcher(self.config)
```

## Configuration Validation

### JSON Schema Validation
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "text_chat": {
      "type": "object",
      "properties": {
        "enabled": {"type": "boolean"},
        "max_message_length": {
          "type": "integer",
          "minimum": 1,
          "maximum": 1000000
        },
        "rate_limiting": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "limits": {
              "type": "object",
              "properties": {
                "messages_per_minute": {
                  "type": "integer",
                  "minimum": 1
                }
              }
            }
          }
        }
      },
      "required": ["enabled"]
    }
  }
}
```

### Validation Function
```python
import jsonschema

def validate_config(config: dict, schema_path: str):
    """Validate configuration against schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(config, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
```

## Performance Tuning

### Cache Configuration
```json
{
  "performance": {
    "caching": {
      "provider": "redis",
      "levels": {
        "l1": {
          "type": "memory",
          "max_size_mb": 512,
          "ttl_seconds": 60
        },
        "l2": {
          "type": "redis",
          "max_size_mb": 4096,
          "ttl_seconds": 300
        }
      }
    }
  }
}
```

### Connection Pooling
```json
{
  "connections": {
    "database": {
      "pool_size": 20,
      "max_overflow": 10,
      "timeout_seconds": 30
    },
    "redis": {
      "pool_size": 50,
      "max_connections": 100
    }
  }
}
```

## Security Configuration

### Encryption Settings
```json
{
  "security": {
    "encryption": {
      "at_rest": {
        "enabled": true,
        "algorithm": "AES-256-GCM",
        "key_provider": "kms",
        "key_rotation_days": 90
      },
      "in_transit": {
        "tls_version": "1.3",
        "cipher_suites": [
          "TLS_AES_256_GCM_SHA384",
          "TLS_CHACHA20_POLY1305_SHA256"
        ]
      }
    }
  }
}
```

### Access Control
```json
{
  "security": {
    "access_control": {
      "provider": "rbac",
      "policies": {
        "default_deny": true,
        "audit_all_access": true
      },
      "roles": {
        "admin": ["*"],
        "user": ["chat:*", "memory:read"],
        "guest": ["chat:read"]
      }
    }
  }
}
```

## Monitoring Configuration

### Metrics Collection
```json
{
  "monitoring": {
    "metrics": {
      "enabled": true,
      "interval_seconds": 60,
      "exporters": {
        "prometheus": {
          "endpoint": "/metrics",
          "port": 9090
        },
        "cloudwatch": {
          "namespace": "ChatPlatform",
          "region": "${AWS_REGION}"
        }
      },
      "custom_metrics": {
        "message_latency": {
          "type": "histogram",
          "buckets": [0.1, 0.5, 1.0, 2.0, 5.0]
        }
      }
    }
  }
}
```

### Distributed Tracing
```json
{
  "monitoring": {
    "tracing": {
      "enabled": true,
      "provider": "jaeger",
      "sampling_rate": 0.1,
      "exporters": {
        "jaeger": {
          "endpoint": "http://jaeger:14268/api/traces",
          "service_name": "chat-platform"
        }
      }
    }
  }
}
```

## Best Practices

### 1. Environment-Specific Values
- Never hardcode environment-specific values
- Use environment variables for secrets
- Keep production configs in secure storage

### 2. Configuration Organization
- Group related settings together
- Use descriptive key names
- Document all configuration options

### 3. Default Values
- Provide sensible defaults
- Make defaults safe for development
- Require explicit production values

### 4. Validation
- Validate all configuration on startup
- Fail fast on invalid configuration
- Provide helpful error messages

### 5. Security
- Never commit secrets to version control
- Rotate keys regularly
- Use least-privilege principles

## Troubleshooting

### Common Issues

#### Configuration Not Loading
```bash
# Check configuration path
echo $CHAT_CONFIG_PATH

# Verify files exist
ls -la /path/to/config/

# Check permissions
chmod 644 /path/to/config/*.json
```

#### Environment Variables Not Substituting
```bash
# Debug environment variables
env | grep CHAT_

# Test substitution
python -c "import os; print(os.getenv('CHAT_ENV', 'not set'))"
```

#### Module Not Enabled
```json
// Ensure module is enabled
{
  "text_chat": {
    "enabled": true  // Must be true
  }
}
```

### Debug Configuration
```python
# Enable configuration debugging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("config")

class DebugConfigLoader(ConfigLoader):
    def _load_file(self, filename: str):
        logger.debug(f"Loading config file: {filename}")
        super()._load_file(filename)
        logger.debug(f"Config after {filename}: {self.config}")
```

## Migration from v2

### Configuration Mapping
```python
# Old v2 configuration
old_config = {
    "storage": {
        "base_path": "./data",
        "max_message_size_bytes": 65536
    }
}

# New modular configuration
new_config = {
    "storage": {
        "base_path": "./data"
    },
    "modules": {
        "text_chat": {
            "max_message_length": 65536
        }
    }
}
```

This configuration system provides flexibility, security, and ease of management for the modular chat platform across all deployment environments.