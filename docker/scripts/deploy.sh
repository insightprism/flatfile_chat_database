#!/bin/bash
# FF Chat API Production Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
COMPOSE_FILE="${PROJECT_DIR}/docker/docker-compose.yml"
ENV_FILE="${PROJECT_DIR}/.env.production"
BACKUP_DIR="${PROJECT_DIR}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check requirements
check_requirements() {
    log "Checking deployment requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if running as root or in docker group
    if ! docker ps &> /dev/null; then
        error "Cannot access Docker. Please run as root or add user to docker group"
        exit 1
    fi
    
    success "Requirements check passed"
}

# Create environment file
create_env_file() {
    log "Creating production environment file..."
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# FF Chat API Production Environment Variables

# API Configuration
FF_ENV=production
FF_LOG_LEVEL=INFO
FF_API_HOST=0.0.0.0
FF_API_PORT=8000
FF_API_WORKERS=4

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -hex 16)

# External Services (CONFIGURE THESE)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database
FF_STORAGE_BASE_PATH=/app/data/storage
FF_VECTOR_BASE_PATH=/app/data/vector
FF_SEARCH_BASE_PATH=/app/data/search

# Features
FF_ENABLE_CORS=true
FF_ENABLE_AUTH=true
FF_RATE_LIMIT_PER_MINUTE=100
EOF
        success "Environment file created at $ENV_FILE"
        warning "Please edit $ENV_FILE to configure external API keys"
    else
        log "Environment file already exists"
    fi
}

# Create SSL certificates (self-signed for development)
create_ssl_certs() {
    log "Creating SSL certificates..."
    
    SSL_DIR="${PROJECT_DIR}/docker/nginx/ssl"
    mkdir -p "$SSL_DIR"
    
    if [ ! -f "$SSL_DIR/server.crt" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/server.key" \
            -out "$SSL_DIR/server.crt" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        success "Self-signed SSL certificates created"
        warning "For production, replace with proper SSL certificates"
    else
        log "SSL certificates already exist"
    fi
}

# Create backup
create_backup() {
    log "Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_NAME="ff-chat-backup-$(date +%Y%m%d-%H%M%S)"
    
    # Backup data volumes
    if docker volume inspect ff_chat_data &> /dev/null; then
        docker run --rm -v ff_chat_data:/source -v "$BACKUP_DIR":/backup \
            alpine tar czf "/backup/${BACKUP_NAME}-data.tar.gz" -C /source .
        success "Data backup created: ${BACKUP_NAME}-data.tar.gz"
    fi
    
    # Backup configuration
    tar czf "$BACKUP_DIR/${BACKUP_NAME}-config.tar.gz" -C "$PROJECT_DIR" docker/ config/ || true
    success "Configuration backup created: ${BACKUP_NAME}-config.tar.gz"
}

# Build images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" build --no-cache ff-chat-api
    
    success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying FF Chat API services..."
    
    cd "$PROJECT_DIR"
    
    # Load environment variables
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    success "Services deployed successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            success "Health check passed"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, waiting..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Show deployment info
show_deployment_info() {
    log "Deployment Information:"
    echo ""
    echo -e "${GREEN}FF Chat API Endpoints:${NC}"
    echo "  API:          https://localhost/api/v1/"
    echo "  WebSocket:    wss://localhost/ws/"
    echo "  Health:       https://localhost/health"
    echo "  Docs:         https://localhost/docs"
    echo ""
    echo -e "${GREEN}Monitoring:${NC}"
    echo "  Prometheus:   http://localhost:9090"
    echo "  Grafana:      http://localhost:3000 (admin/admin123)"
    echo ""
    echo -e "${GREEN}Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Main deployment function
main() {
    log "Starting FF Chat API Production Deployment"
    
    check_requirements
    create_env_file
    create_ssl_certs
    
    # Create backup if services are running
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        create_backup
    fi
    
    build_images
    deploy_services
    
    if health_check; then
        show_deployment_info
        success "FF Chat API deployed successfully!"
    else
        error "Deployment failed health check"
        exit 1
    fi
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --build-only        Only build images, don't deploy"
    echo "  --no-backup         Skip backup creation"
    echo "  --check-health      Only perform health check"
    echo ""
}

# Parse command line arguments
BUILD_ONLY=false
NO_BACKUP=false
CHECK_HEALTH_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            usage
            exit 0
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        --check-health)
            CHECK_HEALTH_ONLY=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute based on options
if [ "$CHECK_HEALTH_ONLY" = true ]; then
    health_check
    exit $?
elif [ "$BUILD_ONLY" = true ]; then
    check_requirements
    build_images
    success "Build completed successfully"
else
    main
fi