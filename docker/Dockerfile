# FF Chat API Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_ENV=production

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
COPY requirements.txt /tmp/
COPY requirements-dev.txt /tmp/ 

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production runtime stage
FROM python:3.11-slim as production

# Set production environment
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FF_ENV=production \
    FF_LOG_LEVEL=INFO

# Create app user for security
RUN groupadd -r ffchat && useradd -r -g ffchat -s /bin/false ffchat

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
RUN mkdir -p /app /app/data /app/logs /app/config && \
    chown -R ffchat:ffchat /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=ffchat:ffchat . /app/

# Create data directories with proper permissions
RUN mkdir -p /app/data/storage /app/data/vector /app/data/search /app/logs && \
    chown -R ffchat:ffchat /app/data /app/logs

# Switch to non-root user
USER ffchat

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Production startup command
CMD ["python", "-m", "uvicorn", "ff_chat_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]