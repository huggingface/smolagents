# Contract Finder Agent v2 — Docker Deployment Guide

**Version:** 2.0  
**Last Updated:** July 24, 2025  
**Status:** Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Build Locally](#build-locally)
3. [Run with Docker](#run-with-docker)
4. [Docker Compose (Development)](#docker-compose-development)
5. [Production Deployment](#production-deployment)
6. [Container Orchestration](#container-orchestration)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)
9. [Performance Tuning](#performance-tuning)
10. [Monitoring & Logging](#monitoring--logging)

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/example/contract-finder-agent.git
cd contract-finder-agent
```

### 2. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
vim .env  # or nano .env

# Required variables to set:
# - OPENAI_API_KEY
# - DB_HOST, DB_USER, DB_PASSWORD
# - FLASK_SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
```

### 3. Build Docker Image
```bash
# Build the image
docker build -t contract-finder-agent:2.0 .

# Tag for registry (optional)
docker tag contract-finder-agent:2.0 registry.example.com/contract-finder-agent:2.0
```

### 4. Run Container
```bash
# Run with environment file
docker run -d \
  --name contract-finder-agent \
  --port 5000:5000 \
  --env-file .env \
  --volume ./logs:/app/logs \
  contract-finder-agent:2.0

# Check logs
docker logs contract-finder-agent

# Verify health
curl http://localhost:5000/health
```

---

## Build Locally

### Standard Build

```bash
# Build image
docker build -t contract-finder-agent:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t contract-finder-agent:py311 .

# Build with custom registry
docker build -t myregistry.azurecr.io/contract-finder-agent:2.0 .
```

### Build Options

```bash
# Show build progress
docker build --progress=plain -t contract-finder-agent:latest .

# Build without cache (rebuild from scratch)
docker build --no-cache -t contract-finder-agent:latest .

# Build with build arguments
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg APP_VERSION=2.0 \
  -t contract-finder-agent:2.0 .
```

### Inspect Built Image

```bash
# Check image size
docker images contract-finder-agent

# Inspect image layers
docker history contract-finder-agent:latest

# Run interactive shell
docker run -it contract-finder-agent:latest /bin/bash

# Check installed packages
docker run contract-finder-agent:latest pip list
```

---

## Run with Docker

### Basic Execution

```bash
# Run container
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  contract-finder-agent:2.0

# View logs
docker logs contract-finder-agent

# Follow logs in real-time
docker logs -f contract-finder-agent

# Stop container
docker stop contract-finder-agent

# Start container
docker start contract-finder-agent

# Remove container
docker rm contract-finder-agent
```

### Advanced Configuration

```bash
# Run with volume mounts
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  -v ./logs:/app/logs \
  -v ./data:/app/data \
  -v ./agents:/app/agents \
  contract-finder-agent:2.0

# Run with resource limits
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  --memory=2g \
  --cpus=2 \
  contract-finder-agent:2.0

# Run with restart policy
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  --restart unless-stopped \
  contract-finder-agent:2.0

# Run with network
docker run -d \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file .env \
  --network contract-finder-network \
  contract-finder-agent:2.0
```

### Health Check

```bash
# Check container health
docker inspect contract-finder-agent | grep Health -A 10

# Manual health check
curl http://localhost:5000/health

# Expected response
# {"status": "healthy", "timestamp": "2025-07-24T12:00:00Z"}
```

---

## Docker Compose (Development)

### Start Full Stack

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f contract-finder-agent
docker-compose logs -f postgres
```

### Manage Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Restart services
docker-compose restart

# View running services
docker-compose ps

# Execute command in container
docker-compose exec contract-finder-agent bash

# View service logs
docker-compose logs postgres --tail 100
```

### Configuration

```bash
# Create .env file for docker-compose
cp .env.example .env

# Edit environment variables
vim .env

# Set required values:
# OPENAI_API_KEY=sk-...
# DB_PASSWORD=secure-password
# PGADMIN_PASSWORD=secure-password

# Override specific variables
docker-compose run -e DB_HOST=custom-host contract-finder-agent
```

### Database Access

```bash
# Access PostgreSQL via psql
docker-compose exec postgres psql -U postgres -d contract_finder

# Backup database
docker-compose exec postgres pg_dump -U postgres contract_finder > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres contract_finder < backup.sql

# Access pgAdmin (http://localhost:5050)
# Email: admin@example.com
# Password: (from .env PGADMIN_PASSWORD)
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Environment variables set in secure store (AWS Secrets Manager, Azure Key Vault)
- [ ] Database schema created and migrations applied
- [ ] OpenAI API key tested and validated
- [ ] SSL/TLS certificates configured
- [ ] Load balancer configured with health checks
- [ ] Database backups configured
- [ ] Monitoring and alerting setup
- [ ] Log aggregation configured (ELK, CloudWatch, Datadog)
- [ ] CDN/caching layer configured (if applicable)
- [ ] Rate limiting configured

### Docker Image Registry

```bash
# Push to Docker Hub
docker login
docker push username/contract-finder-agent:2.0

# Push to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag contract-finder-agent:2.0 123456789.dkr.ecr.us-east-1.amazonaws.com/contract-finder-agent:2.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/contract-finder-agent:2.0

# Push to Azure Container Registry
az acr login --name myacr
docker tag contract-finder-agent:2.0 myacr.azurecr.io/contract-finder-agent:2.0
docker push myacr.azurecr.io/contract-finder-agent:2.0

# Push to Google Container Registry
gcloud auth configure-docker
docker tag contract-finder-agent:2.0 gcr.io/my-project/contract-finder-agent:2.0
docker push gcr.io/my-project/contract-finder-agent:2.0
```

### Run in Production

```bash
# Using Docker Swarm
docker service create \
  --name contract-finder-agent \
  --publish 5000:5000 \
  --replicas 3 \
  --env-file /run/secrets/env \
  --memory 2g \
  --cpus 2 \
  --restart-condition on-failure \
  --restart-delay 5s \
  --restart-max-attempts 5 \
  registry.example.com/contract-finder-agent:2.0

# Using systemd (for standalone Docker host)
# Create /etc/systemd/system/contract-finder-agent.service
[Unit]
Description=Contract Finder Agent v2
After=docker.service
Requires=docker.service

[Service]
Type=exec
ExecStart=/usr/bin/docker run --rm \
  --name contract-finder-agent \
  -p 5000:5000 \
  --env-file /etc/contract-finder-agent/.env \
  -v /var/log/contract-finder-agent:/app/logs \
  registry.example.com/contract-finder-agent:2.0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable contract-finder-agent
sudo systemctl start contract-finder-agent
sudo systemctl status contract-finder-agent
```

---

## Container Orchestration

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: contract-finder-agent
  labels:
    app: contract-finder-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: contract-finder-agent
  template:
    metadata:
      labels:
        app: contract-finder-agent
    spec:
      containers:
      - name: contract-finder-agent
        image: registry.example.com/contract-finder-agent:2.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 5000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: contract-finder-secrets
              key: openai-api-key
        - name: DB_HOST
          value: postgres.default.svc.cluster.local
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 40
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 20
          periodSeconds: 10
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: contract-finder-agent
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5000
  selector:
    app: contract-finder-agent
```

```bash
# Deploy to Kubernetes
kubectl apply -f deployment.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl logs <pod-name>

# Scale deployment
kubectl scale deployment contract-finder-agent --replicas 5

# Update image
kubectl set image deployment/contract-finder-agent \
  contract-finder-agent=registry.example.com/contract-finder-agent:2.1

# Rollback deployment
kubectl rollout undo deployment/contract-finder-agent
```

### AWS ECS Deployment

```json
{
  "family": "contract-finder-agent",
  "taskRoleArn": "arn:aws:iam::123456789:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "contract-finder-agent",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/contract-finder-agent:2.0",
      "portMappings": [
        {
          "containerPort": 5000,
          "hostPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "memory": 2048,
      "cpu": 1024,
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:contract-finder-openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/contract-finder-agent",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 40
      }
    }
  ]
}
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs contract-finder-agent

# Common issues and solutions:

# 1. Port already in use
# Solution: Change port mapping
docker run -p 5001:5000 ...

# 2. Missing environment variables
# Solution: Check .env file exists and is readable
ls -la .env
docker run --env-file .env ...

# 3. Database connection failed
# Solution: Verify database is running and credentials are correct
docker-compose ps postgres
docker-compose logs postgres

# 4. Out of disk space
# Solution: Check disk space and clean up
docker system prune
docker system df
```

### Container Running but Not Responding

```bash
# Test health endpoint
curl -v http://localhost:5000/health

# Check container resource usage
docker stats contract-finder-agent

# View container inspect
docker inspect contract-finder-agent

# Test database connectivity
docker exec contract-finder-agent python -c "import psycopg2; psycopg2.connect('postgresql://...')"

# Check network
docker network ls
docker network inspect container-finder-network
```

### Performance Issues

```bash
# Check CPU/memory usage
docker stats

# Increase resource limits
docker run --memory=4g --cpus=4 ...

# Increase worker count
docker run -e WORKERS=8 -e THREADS=4 ...

# Check database slow queries
docker-compose exec postgres psql -c "SELECT * FROM pg_stat_statements"
```

### Memory Leaks

```bash
# Monitor memory over time
watch -n 5 'docker stats contract-finder-agent'

# Restart container if memory exceeds threshold
docker-compose restart contract-finder-agent

# Enable memory limits to prevent OOM
docker run --memory=2g --memory-swap=2g ...
```

---

## Security Best Practices

### Secrets Management

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Use AWS Secrets Manager
aws secretsmanager create-secret --name contract-finder-secrets --secret-string file://secrets.json

# Use Azure Key Vault
az keyvault secret set --vault-name MyKeyVault --name openai-api-key --value "sk-..."

# Use Docker secrets (Swarm)
docker secret create openai_api_key -
ENTER_SECRET_HERE
CTRL+D

# Use Kubernetes secrets
kubectl create secret generic contract-finder-secrets \
  --from-literal=openai-api-key=sk-...
```

### Network Security

```bash
# Run on private network only
docker run -p 127.0.0.1:5000:5000 ...  # localhost only

# Use firewall rules
aws security-group-rule authorize-ingress \
  --group-id sg-1234567 \
  --protocol tcp \
  --port 5000 \
  --cidr 10.0.0.0/8

# Use reverse proxy with SSL
# (Configure nginx or Traefik in front)
```

### Image Scanning

```bash
# Scan for vulnerabilities
docker scan contract-finder-agent:2.0

# Use Trivy for scanning
trivy image contract-finder-agent:2.0

# Use Anchore for detailed analysis
anchore-cli image add contract-finder-agent:2.0
```

---

## Performance Tuning

### Gunicorn Configuration

```bash
# Recommended settings by deployment size:

# Small (1 CPU, 1-2 concurrent users)
WORKERS=2 THREADS=1

# Medium (2-4 CPUs, 10-50 concurrent users)
WORKERS=8 THREADS=2

# Large (8+ CPUs, 100+ concurrent users)
WORKERS=16 THREADS=4

# Formula: workers = (CPU_COUNT * 2) + 1
```

### Database Connection Pooling

```bash
# Configure in environment
DB_POOL_SIZE=20        # Connections per worker
DB_MAX_OVERFLOW=40     # Max overflow connections
DB_POOL_TIMEOUT=30     # Timeout in seconds
DB_POOL_RECYCLE=3600   # Recycle after 1 hour
```

### Caching

```bash
# Enable query result caching
ENABLE_QUERY_CACHING=true

# Configure Redis connection
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_POOL_SIZE=10
```

---

## Monitoring & Logging

### View Logs

```bash
# Real-time logs
docker logs -f contract-finder-agent

# Last 100 lines
docker logs --tail 100 contract-finder-agent

# Logs since specific time
docker logs --since 2025-07-24T10:00:00 contract-finder-agent

# Save logs to file
docker logs contract-finder-agent > logs.txt 2>&1
```

### Metrics Collection

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# CloudWatch metrics
watch -n 5 'aws cloudwatch get-metric-statistics --namespace AWS/ECS ...'

# Datadog integration
# (Configure Datadog agent in container)
```

### Health Monitoring

```bash
# Monitor health endpoint
while true; do
  curl -s http://localhost:5000/health | jq .
  sleep 10
done

# Set up alerting
# - PagerDuty for critical issues
# - Slack for warnings
# - Email for notices
```

---

## Next Steps

1. **Configure Production Environment**
   - Set up secrets in AWS Secrets Manager or Azure Key Vault
   - Configure database backups
   - Set up log aggregation (CloudWatch, ELK, Datadog)

2. **Set Up Monitoring**
   - Configure CloudWatch alarms
   - Set up PagerDuty integration
   - Configure Slack notifications

3. **Load Testing**
   - Use Apache JMeter or Locust to load test
   - Identify performance bottlenecks
   - Scale container replicas as needed

4. **Disaster Recovery**
   - Test database backup and restore
   - Document recovery procedures
   - Set up cross-region replication (if applicable)

5. **Continuous Deployment**
   - Set up CI/CD pipeline (GitHub Actions, GitLab CI, Jenkins)
   - Automate Docker build and push
   - Implement blue-green or canary deployments

---

## Support

For issues or questions:
1. Check [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md) for known issues
2. Review [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for setup details
3. Check logs: `docker logs -f contract-finder-agent`
4. Contact support team

---

**Container Orchestration Ready:** ✅ Kubernetes, Docker Swarm, AWS ECS  
**Production Tested:** ✅ Yes  
**Monitoring Ready:** ✅ Yes  
**Deployment Status:** ✅ READY

