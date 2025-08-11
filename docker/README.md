# FF Chat API Docker Deployment

This directory contains Docker deployment configurations for the FF Chat API production system.

## Quick Start

1. **Deploy Production Environment:**
   ```bash
   ./scripts/deploy.sh
   ```

2. **Deploy Development Environment:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

## Files Overview

- `Dockerfile` - Production Docker image with multi-stage build
- `Dockerfile.dev` - Development Docker image with hot reloading
- `docker-compose.yml` - Production deployment with monitoring stack
- `docker-compose.dev.yml` - Development environment
- `scripts/deploy.sh` - Automated production deployment script
- `scripts/start.sh` - Container startup script with validation

## Production Deployment

The production deployment includes:

- **FF Chat API** - Main application server with 4 workers
- **Redis** - Caching and session management
- **Nginx** - Reverse proxy with SSL termination
- **Prometheus** - Metrics collection
- **Grafana** - Monitoring dashboards
- **Fluentd** - Log aggregation

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 10GB+ disk space

### Deployment Steps

1. **Clone and prepare:**
   ```bash
   git clone <repository>
   cd flatfile_chat_database_v2/docker
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with your API keys and settings
   ```

3. **Deploy:**
   ```bash
   ./scripts/deploy.sh
   ```

4. **Verify deployment:**
   ```bash
   curl https://localhost/health
   ```

## Development Environment

For development with hot reloading:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

This provides:
- Hot reloading for code changes
- Debug logging
- Development tools
- Local Redis instance

## Configuration

### Environment Variables

Key environment variables for production:

```bash
# API Configuration
FF_ENV=production
FF_API_HOST=0.0.0.0
FF_API_PORT=8000
FF_API_WORKERS=4

# Security
JWT_SECRET_KEY=your-secret-key
REDIS_PASSWORD=your-redis-password

# External Services
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Storage
FF_STORAGE_BASE_PATH=/app/data/storage
FF_VECTOR_BASE_PATH=/app/data/vector
FF_SEARCH_BASE_PATH=/app/data/search
```

### Volume Mounts

- `ff_chat_data:/app/data` - Application data (persistent)
- `ff_chat_logs:/app/logs` - Application logs
- `redis_data:/data` - Redis data (persistent)
- `prometheus_data:/prometheus` - Metrics data
- `grafana_data:/var/lib/grafana` - Dashboard data

## Monitoring

### Prometheus Metrics

Access Prometheus at: http://localhost:9090

Available metrics:
- API request rates and latencies
- Error rates by endpoint
- WebSocket connection counts
- System resource usage

### Grafana Dashboards

Access Grafana at: http://localhost:3000
- Username: admin
- Password: admin123

Pre-configured dashboards:
- FF Chat API Overview
- System Performance
- Error Analysis
- User Activity

### Log Aggregation

Fluentd collects and processes logs:
- API access logs
- Error logs
- System logs
- Alert generation for critical errors

## SSL/TLS Configuration

### Development (Self-Signed)

The deployment script creates self-signed certificates for development:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/server.crt
```

### Production (Let's Encrypt)

For production, replace with proper certificates:

1. **Using Certbot:**
   ```bash
   certbot certonly --webroot -w /var/www/html -d yourdomain.com
   cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/server.crt
   cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/server.key
   ```

2. **Update nginx configuration with your domain**

## Scaling

### Horizontal Scaling

Scale API workers:
```bash
docker-compose up -d --scale ff-chat-api=3
```

### Load Balancing

Nginx automatically load balances across API instances using least connections.

## Health Checks

All services include health checks:

- **API**: `GET /health` endpoint
- **Redis**: `redis-cli ping`
- **Nginx**: Status endpoint on port 8080
- **Prometheus**: Self-monitoring

## Backup and Recovery

### Automated Backups

The deployment script creates backups before updates:

```bash
./scripts/deploy.sh  # Creates backup automatically
```

### Manual Backup

```bash
# Backup data volumes
docker run --rm -v ff_chat_data:/source -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data-backup.tar.gz -C /source .

# Backup Redis
docker exec ff-chat-redis redis-cli save
docker cp ff-chat-redis:/data/dump.rdb ./backups/
```

### Restore

```bash
# Restore data volume
docker run --rm -v ff_chat_data:/target -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/data-backup.tar.gz -C /target
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using ports
   sudo lsof -i:8000
   sudo lsof -i:443
   ```

2. **Permission issues:**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data
   ```

3. **Memory issues:**
   ```bash
   # Check resource usage
   docker stats
   ```

### Log Analysis

```bash
# API logs
docker logs ff-chat-api

# All service logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f redis
```

### Service Management

```bash
# Restart specific service
docker-compose restart ff-chat-api

# Update and restart
docker-compose pull
docker-compose up -d

# Clean shutdown
docker-compose down
```

## Security Considerations

1. **Change default passwords** in production
2. **Use proper SSL certificates** (not self-signed)
3. **Configure firewall** to restrict access
4. **Regular security updates** for base images
5. **Monitor logs** for security events
6. **Backup encryption** for sensitive data

## Performance Tuning

### API Workers

Adjust based on CPU cores:
```yaml
environment:
  - FF_API_WORKERS=8  # 2x CPU cores
```

### Redis Memory

Configure Redis memory limits:
```yaml
command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
```

### Nginx Caching

Enable caching for static content:
```nginx
location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8000/health`
3. Check metrics: http://localhost:9090
4. Review configuration files

## License

This deployment configuration is part of the FF Chat API system.