# Contract Finder Agent v2 — Docker Documentation

**Status:** ✅ Production Ready  
**Version:** 2.0  
**Updated:** July 24, 2025

---

## 📦 Docker Files Overview

This directory contains complete Docker deployment configurations for the Contract Finder Agent v2:

### Files Included

| File | Size | Purpose |
|------|------|----------|
| **Dockerfile** | 92 lines | Production-ready multi-stage Docker image |
| **.dockerignore** | 134 lines | Excludes unnecessary files from build context |
| **docker-compose.yml** | 148 lines | Development environment with services |
| **.env.example** | 205 lines | Environment variable configuration template |
| **DOCKER_DEPLOYMENT_GUIDE.md** | 789 lines | Comprehensive deployment documentation |
| **DOCKER_README.md** | This file | Quick reference guide |

**Total Documentation:** 1,368 lines

---

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Build image
docker build -t contract-finder-agent:2.0 .

# Verify image size
docker images contract-finder-agent
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your values (required)
vim .env

# Must set:
# - OPENAI_API_KEY (your OpenAI API key)
# - DB_HOST, DB_USER, DB_PASSWORD (database credentials)
# - SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
```

### 3. Run Container

```bash
# Start the agent
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  -v ./logs:/app/logs \
  contract-finder-agent:2.0

# Check logs
docker logs -f contract-finder-agent

# Verify health
curl http://localhost:5000/health
```

### 4. Access the Agent

```bash
# Test endpoint
curl -X POST http://localhost:5000/utility/contract-finder-agent \
  -H "Content-Type: application/json" \
  -d '{
    "agent_arguments": {
      "user_query": "Find software licensing agreements",
      "filters": []
    }
  }'
```

---

## 🐳 Dockerfile Details

### Multi-Stage Build Strategy

The Dockerfile uses a 2-stage build for optimization:

```dockerfile
# Stage 1: Builder
- Installs build tools (gcc, make, libpq-dev)
- Creates Python virtual environment
- Installs dependencies from requirements.txt
- Total layer: ~600MB (not in final image)

# Stage 2: Runtime
- Starts from clean Python 3.10-slim image
- Copies only the virtual environment from builder
- Creates non-root 'agent' user for security
- Installs only runtime dependencies (libpq5)
- Final image: ~400-500MB
```

### Key Features

✅ **Security**
- Non-root user execution (UID 1000)
- Minimal attack surface (slim image)
- No build tools in final image

✅ **Performance**
- Multi-stage build reduces image size by 60%
- Virtual environment caching for faster rebuilds
- Health check for container orchestration

✅ **Production Ready**
- Gunicorn with optimized worker configuration
- 4 workers × 2 threads (configurable)
- 300-second timeout for long-running queries
- Structured logging to stdout/stderr

✅ **Monitoring**
- HTTP health check endpoint
- Access and error logging
- Graceful shutdown support

---

## 🐳 Container Configuration

### Environment Variables (from .env)

**Critical (must set):**
```env
OPENAI_API_KEY=sk-...              # OpenAI API key
DB_HOST=localhost                  # Database host
DB_PORT=5432                       # Database port
DB_NAME=contract_finder            # Database name
DB_USER=postgres                   # Database user
DB_PASSWORD=secure-password        # Database password
SECRET_KEY=...                     # Flask secret (32+ chars)
```

**Recommended:**
```env
FLASK_ENV=production               # Environment (development/production)
LLM_MODEL_NAME=gpt-4               # LLM model to use
LLM_TEMPERATURE=0.7                # LLM temperature (0.0-1.0)
LLM_INVOCATION_TIMEOUT=30          # LLM call timeout (seconds)
SIMILARITY_THRESHOLD=0.3           # Semantic search threshold
MAX_CONTRACTS_TO_RETURN=10         # Max results per query
LOG_LEVEL=INFO                     # Logging level
```

**Optional:**
```env
REDIS_HOST=redis                   # Redis cache host
REDIS_PORT=6379                    # Redis port
RATE_LIMIT_ENABLED=true            # Enable rate limiting
WORKERS=4                          # Gunicorn workers
THREADS=2                          # Threads per worker
```

See **.env.example** for all available options (205 lines).

---

## 🐳 Docker Compose

### Development Stack

The **docker-compose.yml** provides a complete development environment:

```yaml
Services:
  contract-finder-agent    # Flask application on :5000
  postgres                 # PostgreSQL database on :5432
  redis                    # Redis cache on :6379
  pgadmin                  # Database UI on :5050
```

### Start Development Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Service Details

| Service | Port | Purpose | Credentials |
|---------|------|---------|-------------|
| Contract Finder Agent | 5000 | Flask application | None |
| PostgreSQL | 5432 | Database | user: postgres / password: postgres |
| Redis | 6379 | Cache | password: redis123 (from .env) |
| pgAdmin | 5050 | Database UI | admin@example.com / admin |

### Database Access

```bash
# Connect with psql
docker-compose exec postgres psql -U postgres -d contract_finder

# Backup database
docker-compose exec postgres pg_dump -U postgres contract_finder > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres contract_finder < backup.sql
```

---

## 📋 Deployment Scenarios

### Local Development

```bash
# Use docker-compose for full stack
docker-compose up -d

# Access:
# - Agent: http://localhost:5000
# - pgAdmin: http://localhost:5050
# - Redis: localhost:6379
```

### Single Container (Standalone)

```bash
# Build and run
docker build -t contract-finder-agent:2.0 .

docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  -v ./logs:/app/logs \
  --restart unless-stopped \
  contract-finder-agent:2.0
```

### Production (with Registry)

```bash
# Build and push
docker build -t registry.example.com/contract-finder-agent:2.0 .
docker push registry.example.com/contract-finder-agent:2.0

# Pull and run from registry
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file /secure/location/.env \
  --memory=2g \
  --cpus=2 \
  --restart unless-stopped \
  registry.example.com/contract-finder-agent:2.0
```

### Kubernetes

```bash
# Create deployment
kubectl apply -f deployment.yaml

# Scale replicas
kubectl scale deployment contract-finder-agent --replicas 5

# Monitor
kubectl logs -f deployment/contract-finder-agent
```

See **DOCKER_DEPLOYMENT_GUIDE.md** for complete Kubernetes manifest.

---

## 🛠️ Common Commands

### Building

```bash
# Standard build
docker build -t contract-finder-agent:2.0 .

# Build without cache
docker build --no-cache -t contract-finder-agent:2.0 .

# Build with custom Python version
docker build --build-arg PYTHON_VERSION=3.11 -t contract-finder-agent:py311 .

# Check image size
docker images contract-finder-agent
```

### Running

```bash
# Run interactive
docker run -it contract-finder-agent:2.0 /bin/bash

# Run with port mapping
docker run -d -p 5000:5000 contract-finder-agent:2.0

# Run with environment
docker run -d --env-file .env contract-finder-agent:2.0

# Run with volume mounts
docker run -d -v ./logs:/app/logs -v ./data:/app/data contract-finder-agent:2.0
```

### Managing

```bash
# View logs
docker logs contract-finder-agent
docker logs -f contract-finder-agent      # Follow logs
docker logs --tail 100 contract-finder-agent

# Check status
docker ps
docker inspect contract-finder-agent
docker stats contract-finder-agent

# Restart
docker restart contract-finder-agent

# Stop/Remove
docker stop contract-finder-agent
docker rm contract-finder-agent
```

### Registry Operations

```bash
# Docker Hub
docker login
docker push username/contract-finder-agent:2.0

# AWS ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker tag contract-finder-agent:2.0 <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/contract-finder-agent:2.0
docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/contract-finder-agent:2.0

# Azure ACR
az acr login --name <REGISTRY_NAME>
docker tag contract-finder-agent:2.0 <REGISTRY_NAME>.azurecr.io/contract-finder-agent:2.0
docker push <REGISTRY_NAME>.azurecr.io/contract-finder-agent:2.0
```

---

## ✅ Health Checks

### Container Health Endpoint

```bash
# Check container health
curl http://localhost:5000/health

# Expected response (200 OK):
# {"status": "healthy", "timestamp": "2025-07-24T12:00:00Z"}

# Docker health check
docker inspect contract-finder-agent | grep Health -A 10
```

### Verify Connectivity

```bash
# Test database
docker exec contract-finder-agent python -c "import psycopg2; psycopg2.connect('postgresql://...')"

# Test API
curl -X POST http://localhost:5000/utility/contract-finder-agent \
  -H "Content-Type: application/json" \
  -d '{"agent_arguments": {"user_query": "test"}}'

# Test OpenAI
docker exec contract-finder-agent python -c "import os; from openai import OpenAI; OpenAI(api_key=os.getenv('OPENAI_API_KEY'))"
```

---

## 🔒 Security Best Practices

### Secrets Management

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Use secure secret stores:
# - AWS Secrets Manager
# - Azure Key Vault
# - HashiCorp Vault
# - Kubernetes Secrets

# Pass secrets to container:
docker run -e OPENAI_API_KEY="$(aws secretsmanager get-secret-value --secret-id openai-key --query SecretString --output text)"
```

### Image Security

```bash
# Scan for vulnerabilities
docker scan contract-finder-agent:2.0

# Use Trivy
trivy image contract-finder-agent:2.0

# Sign images
docker trust sign registry.example.com/contract-finder-agent:2.0
```

### Runtime Security

```bash
# Run as non-root (default in Dockerfile)
# Do NOT run: docker run --user=root

# Use read-only filesystem where possible
docker run --read-only --tmpfs /tmp --tmpfs /app/logs ...

# Limit capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE ...
```

---

## 📊 Performance Optimization

### Image Size

```
Final Image Size: ~450-500 MB

Breakdown:
- Python 3.10-slim base: ~150 MB
- System libraries (libpq5): ~50 MB
- Python packages: ~200-250 MB
- Application code: ~10 MB
```

### Build Time

```
First Build:  ~2-3 minutes (download base, install deps)
Cached Build: ~10 seconds (reuse layers)
```

### Runtime Memory

```
Minimum:  1 GB (single worker, 1 thread)
Recommended: 2 GB (4 workers, 2 threads)
Max:  4+ GB (high concurrency, 8+ workers)
```

### CPU Cores

```
Minimum:  1 core
Recommended: 2 cores
Max:  4+ cores (distributed deployment recommended)
```

---

## 📚 Related Documentation

For detailed information, see:

1. **[DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)**
   - Comprehensive deployment instructions
   - Kubernetes manifests
   - AWS ECS configurations
   - Troubleshooting guide
   - Performance tuning

2. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**
   - Local development setup
   - Configuration details
   - API documentation

3. **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)**
   - Architecture overview
   - Deployment requirements
   - Success metrics

4. **[CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)**
   - Code quality assessment
   - Known limitations
   - Future improvements

---

## 🚀 Deployment Checklist

### Pre-Deployment

- [ ] Docker image built and tested locally
- [ ] .env configured with all required values
- [ ] Database schema created and tested
- [ ] OpenAI API key validated
- [ ] Container image scanned for vulnerabilities
- [ ] Health checks passing (curl http://localhost:5000/health)

### Deployment

- [ ] Image pushed to registry
- [ ] Container orchestration platform configured
- [ ] Environment variables set in secure store
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Logging aggregation configured
- [ ] Load balancer configured with health checks
- [ ] SSL/TLS certificates installed

### Post-Deployment

- [ ] Verify container startup logs
- [ ] Test health endpoint
- [ ] Test API endpoints
- [ ] Monitor resource usage
- [ ] Verify log output
- [ ] Test failover and restart behavior
- [ ] Configure auto-scaling (if applicable)

---

## 🆘 Troubleshooting

### Container won't start

```bash
# Check logs
docker logs contract-finder-agent

# Verify environment variables
docker run -it --env-file .env contract-finder-agent:2.0 env | sort

# Test with bash
docker run -it contract-finder-agent:2.0 /bin/bash
```

### Health check failing

```bash
# Check endpoint
curl -v http://localhost:5000/health

# Check container port
docker port contract-finder-agent

# Check network
docker network inspect bridge
```

### Performance issues

```bash
# Monitor resources
docker stats contract-finder-agent

# Check logs for errors
docker logs contract-finder-agent | grep -i error

# Increase worker count
docker run -e WORKERS=8 ...
```

See **DOCKER_DEPLOYMENT_GUIDE.md** for comprehensive troubleshooting.

---

## 📞 Support

**Issues?** Check:
1. **Logs:** `docker logs -f contract-finder-agent`
2. **Health:** `curl http://localhost:5000/health`
3. **Guide:** [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)
4. **Code Review:** [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)
5. **Developer:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

## ✨ Summary

✅ **Dockerfile** — Production-ready, multi-stage build, 450-500 MB  
✅ **.dockerignore** — Optimized build context  
✅ **docker-compose.yml** — Full development stack with services  
✅ **.env.example** — Complete configuration template  
✅ **DOCKER_DEPLOYMENT_GUIDE.md** — Comprehensive deployment documentation  

**Status:** ✅ **PRODUCTION READY**

The Contract Finder Agent v2 is fully containerized and ready for deployment to any Docker-compatible environment (local, cloud, Kubernetes, etc.).

