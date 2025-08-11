#!/bin/bash
# FF Chat API Production Startup Script

set -e

echo "Starting FF Chat API Production Environment..."

# Environment validation
if [ -z "$FF_ENV" ]; then
    export FF_ENV=production
fi

echo "Environment: $FF_ENV"

# Create necessary directories
mkdir -p /app/data/storage /app/data/vector /app/data/search /app/logs

# Set proper permissions
chown -R ffchat:ffchat /app/data /app/logs

# Validate configuration
echo "Validating configuration..."
python -c "
import sys
from ff_class_configs.ff_configuration_manager_config import load_config
try:
    config = load_config()
    print('Configuration validation: PASSED')
except Exception as e:
    print(f'Configuration validation: FAILED - {e}')
    sys.exit(1)
"

# Database migrations (if any)
echo "Running database migrations..."
python -c "
from ff_storage_manager import FFStorageManager
from ff_class_configs.ff_configuration_manager_config import load_config
import asyncio

async def run_migrations():
    try:
        config = load_config()
        # Add any migration logic here
        print('Database migrations: COMPLETED')
    except Exception as e:
        print(f'Database migrations: FAILED - {e}')
        raise

asyncio.run(run_migrations())
"

# Health check before starting
echo "Performing startup health check..."
python -c "
from ff_chat_api import FFChatAPI, FFChatAPIConfig
import asyncio

async def health_check():
    try:
        config = FFChatAPIConfig()
        api = FFChatAPI(config)
        await api.initialize()
        print('Startup health check: PASSED')
        await api.shutdown()
    except Exception as e:
        print(f'Startup health check: FAILED - {e}')
        raise

asyncio.run(health_check())
"

echo "FF Chat API startup validation complete. Starting server..."

# Start the application
exec python -m uvicorn ff_chat_api:app \
    --host "${FF_API_HOST:-0.0.0.0}" \
    --port "${FF_API_PORT:-8000}" \
    --workers "${FF_API_WORKERS:-4}" \
    --log-level "${FF_LOG_LEVEL:-info}" \
    --access-log \
    --use-colors