# Contract Finder Agent v2 — Developer Guide

**Version:** 2.0
**Last Updated:** 2024
**Status:** Production Ready

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Local Development Setup](#local-development-setup)
5. [Environment Configuration](#environment-configuration)
6. [Database Setup](#database-setup)
7. [Running the Agent](#running-the-agent)
8. [Testing the Agent](#testing-the-agent)
9. [Docker Deployment](#docker-deployment)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)

---

## Project Overview

The Contract Finder Agent v2 is a Flask-based RAG (Retrieval-Augmented Generation) service that:
- Accepts natural language queries about contracts
- Applies semantic and metadata filters to retrieve relevant contracts
- Uses LLM to generate answers based on retrieved contract content
- Provides exact-match citations for retrieved content
- Supports pagination for discovering additional results

### Key Features
- ✅ Multi-filter semantic search with AND (intersection) logic
- ✅ Multi-component scoring (similarity + keyword + document)
- ✅ Session-based pagination for show-more functionality
- ✅ LLM-powered response generation with context injection
- ✅ Citation generation with exact text matching
- ✅ Comprehensive error handling and graceful degradation

---

## Project Structure

```
contract-finder-agent/
├── agents/
│   └── contract_finder_agent/
│       ├── contract_finder_agent.py              # Main orchestration logic
│       ├── contract_finder_agent_setup.py        # Configuration schema
│       ├── contract_finder_agent_trace.py        # Tracing schema
│       ├── contract_finder_agent_methods.py      # ACT 6-8 implementations
│       ├── contract_helper.py                    # Filter mappings
│       ├── contract_finder_utilities.py          # Helper functions
│       └── queries/
│           ├── retrieve_final_chunks.sql
│           ├── filter_query_associated_cols.sql
│           ├── filter_query_non_associated_cols.sql
│           ├── create_chat_data_table.sql
│           ├── fetch_show_more_docs.sql
│           ├── insert_retrieved_documents.sql
│           └── mark_contracts_as_displayed.sql
├── run.py                                        # Flask app entry point
├── Config.py                                     # Environment configuration
├── requirements.txt                              # Python dependencies
├── Dockerfile                                    # Container image definition
├── docker-compose.yml                            # Local Docker setup
├── Makefile                                      # Build automation
├── IMPLEMENTATION_VALIDATION.md                  # Validation report
└── DEVELOPER_GUIDE.md                           # This file
```

---

## Prerequisites

### Required Software
- **Python:** 3.9+
- **pip:** Latest version
- **Git:** For version control
- **Docker:** (Optional, for containerized deployment)
- **PostgreSQL/AlloyDB:** For contract database

### Required Credentials & Services
- **LLM Service:** OpenAI API key OR Azure OpenAI endpoint
- **Embedding Service:** Google Vertex AI OR Local embedding model
- **Database:** AlloyDB or PostgreSQL with contract schema
- **GCP Project:** For Pub/Sub, Cloud Logging, etc. (production)

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- `flask==3.0.3` — Web framework
- `llm-studio-agents==1.0.1` — Agent framework
- `llm-studio-tools==1.8.0` — LLM tools
- `pydantic==2.10.6` — Data validation
- `psycopg2-binary==2.9.9` — PostgreSQL driver
- `google-cloud-bigquery==3.36.0` — GCP integration

---

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd contract-finder-agent
```

### Step 2: Create Python Virtual Environment

```bash
# Create venv
python3.9 -m venv venv

# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import flask; import llm_studio_agents; print('Installation successful')"
```

---

## Environment Configuration

### Step 1: Create .env File

Create `local_testing/contracts.env` (referenced in Config.py line 11):

```bash
# Environment
ENV=development

# Service URLs
RLEF_URL=https://api.example.com
COPILOT_URL=https://copilot.example.com
SPT_URL=https://spt.example.com

# LLM Models
DOCUMENT_RETRIEVAL_MODEL_ID=gpt-4-1106-preview
EVALUATION_MODEL_ID=gpt-4
BEST_ANSWER_MODEL_ID=gpt-4

EVALUATION_COPILOT_ID=eval-copilot-id
BEST_ANSWER_COPILOT_ID=answer-copilot-id

TAG_PREFIX=contract-finder

# Embedding Model
GOOGLE_EMBEDDING_MODEL=textembedding-gecko@002

# GCP ENVIRONMENT
FIREBASE_URL=https://your-project.firebaseio.com
PUBSUB_PROJECT_ID=your-gcp-project
PUBSUB_AI_BACKEND_SUBSCRIPTION_ID=ai-backend-sub
PUBSUB_TOPIC_ID=ai-topic
VOICE_REQUEST_TOPIC_ID=voice-topic
VOICE_TRANSCRIPTION_SUBSCRIPTION_ID=transcription-sub
VOICE_PUBSUB_TIMEOUT=300

# COPILOT CONFIGURATION
COPILOT_TIMEOUT=60
COPILOT_THRESHOLD=70
RLEF_COPILOT_THRESHOLD=true

# INDIVIDUAL MODEL THRESHOLD
DEFAULT_INDIVIDUAL_GOOGLE_THRESHOLD=60
DEFAULT_INDIVIDUAL_AZURE_THRESHOLD=65

PUBSUB_TIMEOUT=30

# Default fallback message
DEFAULT_MESSAGE=I couldn't find relevant contracts for your query.

# Azure OpenAI
AZURE_API_TYPE=azure
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# DALLE Azure
DALLE_AZURE_API_VERSION=2024-02-01
DALLE_AZURE_OPENAI_API_KEY=your-dalle-key
DALLE_AZURE_OPENAI_ENDPOINT=https://your-dalle-resource.openai.azure.com/

# Data Lake
DATA_LAKE_ACCOUNT_NAME=yourlakeaccount
DATA_LAKE_ACCOUNT_KEY=your-key
DATA_LAKE_FILE_SYSTEM_NAME=contracts

# OpenSource Model
OPENSOURCE_URL=http://localhost:8080

# AlloyDB Configuration (CRITICAL)
DB_NAME=contracts_db
DB_USER=postgres
DB_PASS=your-secure-password
DB_HOST=localhost
DB_PORT=5432

# PROPOSALS
PROPOSALS_BUCKET_NAME=proposals-bucket
PROPOSALS_EMAIL_SENDER=noreply@company.com
PROPOSALS_EMAIL_TOKEN=email-token

# VISION
IMAGES_BUCKET_NAME=images-bucket

# SHAREPOINT
SHAREPOINT_TENANT_ID=tenant-id
SHAREPOINT_CLIENT_ID=client-id
SHAREPOINT_CLIENT_SECRET=client-secret

# MONGO
MONGO_URL=mongodb://localhost:27017/

# Base URLs
BACKEND_BASE_URL=http://localhost:5000
COPILOT_UI_URL=http://localhost:3000

# OpenAI Key
OPENAI_KEY=sk-your-openai-key

# Tavily API
TAVILY_API_KEY=your-tavily-api-key

# GitHub
PUBSUB_GITHUB_BATCH_INTERVAL=300
PUBSUB_GITHUB_BATCH_SIZE=100

# Web Crawler
WEB_CRAWLER_CLOUD_FUNCTION_URL=https://region-project.cloudfunctions.net/crawler
FIREBASE_LOG_URL=https://firebase-log-url
BIGQUERY_DATASET=analytics

# AGENTS
MAX_ITERATIONS_WITH_AGENTS=12
AGENTS_PUBLISH_QUERY_TOPIC_ID=agents-query
AGENTS_RECEIVE_RESPONSE_SUBSCRIPTION_ID=agents-response

# Node.js Chat History
EGPT_NODE_BASE_URL=http://localhost:3001

# Embedding Service
EMBEDDING_SERVICE=google-vertex-ai
CRYPTO_SECRET=your-secret-key

# MongoDB
MONGODB_NAME=contracts_db
MONGODB_CHAT_HISTORY_GRAPH_COLLECTION_NAME=chat_graphs
MONGODB_CHAT_HISTORY_GRAPH_NODES_COLLECTION_NAME=graph_nodes
MONGODB_CHAT_HISTORY_GRAPH_LINKS_COLLECTION_NAME=graph_links

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Server Configuration
MAX_WORKER_THREADS=10
MAX_MESSAGES=10
MAX_PUBSUB_BYTES=104857600
MAX_SUBSCRIBER_WORKER_THREADS=10

# Database Interface
USE_DATABASE=mongo
MONGO_MAX_POOL_SIZE=10
MONGO_MIN_POOL_SIZE=0
MONGO_MAX_IDLE_TIME_MS=10000
MONGO_SERVER_SELECTION_TIMEOUT_MS=3000
MONGO_SOCKET_TIMEOUT_MS=5000
MONGO_CONNECT_TIMEOUT_MS=5000

# Organization
ORGANIZATION_NAME=techolution
NODE_AUTH_TOKEN=auth-token
```

### Step 2: Update Config.py Constants

For local development, update `Config.py` lines 120-127:

```python
WORKER_COUNT = 2                    # Number of Gunicorn workers
WORKER_TIMEOUT = 120                # Worker timeout in seconds
WORKER_THREADS_COUNT = 4            # Threads per worker
MAX_REQUEST_TO_WORKER_RESTART = 100 # Requests before worker restart
WORKER_GRACEFUL_TIMEOUT = 30        # Graceful shutdown timeout
```

---

## Database Setup

### Step 1: Create Database

```bash
# Connect to PostgreSQL/AlloyDB
psql -h localhost -U postgres

# Create database
CREATE DATABASE contracts_db;
\c contracts_db
```

### Step 2: Create Required Tables

```sql
-- Contracts table
CREATE TABLE contract_test_v2 (
    contract_id UUID PRIMARY KEY,
    contract_name VARCHAR(255),
    contract_summary TEXT,
    contract_summary_embeddings vector(1536),
    product_types_embeddings vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content chunks table
CREATE TABLE contract_content_chunks (
    chunk_id UUID PRIMARY KEY,
    contract_id UUID REFERENCES contract_test_v2(contract_id),
    chunk_text TEXT,
    chunk_embeddings vector(1536),
    similarity_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session history table (created dynamically by agent, but define schema here)
CREATE TABLE contract_retrieved_document_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    question TEXT,
    contract_id UUID,
    doc_context JSONB,
    is_displayed BOOLEAN DEFAULT false,
    display_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session_question ON contract_retrieved_document_details(session_id, question);
CREATE INDEX idx_is_displayed ON contract_retrieved_document_details(is_displayed);
```

### Step 3: Create Vector Extension (if using PostgreSQL 13+)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 4: Load Sample Data (Optional)

See `local_testing/sample_data.sql` for sample contract data.

---

## Running the Agent

### Option 1: Local Development (Flask Debug Mode)

```bash
# Activate venv
source venv/bin/activate

# Set Flask debug mode
export FLASK_ENV=development
export FLASK_APP=run.py

# Run Flask development server
flask run --host=0.0.0.0 --port=5002

# Agent available at: http://localhost:5002/utility/contract-retrieval-rag-agent/
```

### Option 2: Production (Gunicorn)

```bash
# Activate venv
source venv/bin/activate

# Run Gunicorn (command from run.py lines 120-127)
gunicorn --workers 2 --worker-class gthread --bind 0.0.0.0:5002 \
  --timeout 120 --keep-alive 120 --max-requests 100 --max-requests-jitter 50 \
  --log-level info --threads 4 --access-logfile - --error-logfile - \
  --graceful-timeout 30 --limit-request-line 8190 run:app
```

### Option 3: Docker (Containerized)

```bash
# Build image
docker build -t contract-finder-agent:latest .

# Run container
docker run -p 5002:5002 \
  --env-file local_testing/contracts.env \
  contract-finder-agent:latest
```

### Option 4: Docker Compose (with Dependencies)

```bash
# Start all services (PostgreSQL, Redis, Agent)
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f contract-finder-agent

# Stop services
docker-compose down
```

---

## Testing the Agent

### Health Check

```bash
curl -X GET http://localhost:5002/utility/contract-retrieval-rag-agent/health

# Expected response:
{"status": "healthy"}
```

### Get Configuration

```bash
curl -X GET http://localhost:5002/utility/contract-retrieval-rag-agent/get_config
```

### Prediction Request (Contract Retrieval)

```bash
curl -X POST http://localhost:5002/utility/contract-retrieval-rag-agent/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "concierge_id": "test-concierge",
    "agent_id": "contract-finder-v2",
    "preview_id": "session-123",
    "question": "Find contracts related to software licensing",
    "agent_arguments": {
      "user_query": "software licensing agreements",
      "contract_details": "licensing and intellectual property",
      "clause_details": null,
      "party_details": null,
      "product_type": "software",
      "contract_effective_date": null,
      "in_scope_work_details": null,
      "licensing_right_details": null,
      "territory_country": null,
      "licensing_payment_details": null,
      "show_more_details": null
    }
  }'
```

### Show-More Pagination Request

```bash
curl -X POST http://localhost:5002/utility/contract-retrieval-rag-agent/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "concierge_id": "test-concierge",
    "agent_id": "contract-finder-v2",
    "preview_id": "session-123",
    "question": "Find contracts related to software licensing",
    "agent_arguments": {
      "user_query": "software licensing agreements",
      "contract_details": "licensing and intellectual property",
      "show_more_details": {
        "question": "software licensing agreements",
        "length": 10
      }
    }
  }'
```

### Response Structure

```json
{
  "response": "Based on the retrieved contracts, [LLM-generated answer]...",
  "documents": {
    "retrieved_documents": [
      {
        "contract_id": "contract-123",
        "chunk_text": "Relevant contract clause text...",
        "context_score": 0.92,
        "similarity_score": 0.88
      }
    ]
  },
  "citations": [
    {
      "title": "Contract 123",
      "description": "Highlighted text from source...",
      "customMetaData": {
        "contract_id": "contract-123",
        "relevance_score": 0.92
      }
    }
  ],
  "metadata": {
    "retrieved_documents_count": 5,
    "citations_count": 5,
    "execution_status": "success"
  }
}
```

### Using Postman

1. Import the request collection from `local_testing/contract-finder-postman.json`
2. Set variables:
   - `{{base_url}}` = http://localhost:5002
   - `{{session_id}}` = unique session identifier
3. Execute requests in order

---

## Docker Deployment

### Step 1: Build Docker Image

```bash
docker build -t contract-finder-agent:v2.0 .
```

### Step 2: Tag for Registry

```bash
# Docker Hub
docker tag contract-finder-agent:v2.0 username/contract-finder-agent:v2.0

# GCR (Google Container Registry)
docker tag contract-finder-agent:v2.0 gcr.io/project-id/contract-finder-agent:v2.0
```

### Step 3: Push to Registry

```bash
# Docker Hub
docker push username/contract-finder-agent:v2.0

# GCR
gcloud auth configure-docker
gcloud docker -- push gcr.io/project-id/contract-finder-agent:v2.0
```

### Step 4: Deploy to Kubernetes (Production)

Create `k8s-deployment.yaml`:

```yaml
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
        image: gcr.io/project-id/contract-finder-agent:v2.0
        ports:
        - containerPort: 5002
        env:
        - name: ENV
          valueFrom:
            configMapKeyRef:
              name: contract-config
              key: ENV
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: contract-secrets
              key: db-host
        - name: DB_PASS
          valueFrom:
            secretKeyRef:
              name: contract-secrets
              key: db-pass
        # ... additional environment variables
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /utility/contract-retrieval-rag-agent/health
            port: 5002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /utility/contract-retrieval-rag-agent/health
            port: 5002
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: contract-finder-agent-service
spec:
  selector:
    app: contract-finder-agent
  ports:
  - port: 5002
    targetPort: 5002
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl logs -f deployment/contract-finder-agent
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'llm_studio_agents'"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Database connection refused"

**Solution:**
```bash
# Check database is running
psql -h localhost -U postgres -c "SELECT version();"

# Check environment variables
echo $DB_HOST $DB_PORT $DB_USER

# Test connection
psql postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME -c "SELECT 1;"
```

### Issue: "LLM API key invalid"

**Solution:**
```bash
# Verify Azure OpenAI endpoint
curl -X GET https://your-resource.openai.azure.com/openai/deployments?api-version=2024-02-15-preview \
  -H "api-key: $AZURE_OPENAI_API_KEY"

# Or verify OpenAI key
curl -X GET https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_KEY"
```

### Issue: "Vector embeddings dimension mismatch"

**Solution:**
Ensure `embedding_dimensions` in Config matches vector dimension:
- Google Vertex AI: 1536 (default)
- OpenAI text-embedding-3-small: 1536
- OpenAI text-embedding-3-large: 3072

```python
# In contract_finder_agent_setup.py
embedding_dimensions: int = Field(
    1536,  # Match your embedding service
    description="Dimension of embeddings..."
)
```

### Issue: "Session table not found"

**Solution:**
The agent automatically creates the table on first run. If error persists:

```sql
CREATE TABLE IF NOT EXISTS contract_retrieved_document_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    question TEXT,
    contract_id UUID,
    doc_context JSONB,
    is_displayed BOOLEAN DEFAULT false,
    display_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Issue: "No contracts found" (but database has data)

**Solution:**
1. Check filter mappings in `contract_helper.py`
2. Verify contract_content_chunks embeddings are populated
3. Adjust `similarity_threshold` in ContractFinderAgentSetup (default: 0.3)

```python
similarity_threshold: float = Field(
    0.2,  # Lower threshold = more results
    description="Threshold of similarity..."
)
```

---

## Performance Optimization

### Database Optimization

1. **Create Indexes:**
```sql
CREATE INDEX idx_contract_embeddings ON contract_test_v2 USING ivfflat (contract_summary_embeddings vector_cosine_ops);
CREATE INDEX idx_chunk_embeddings ON contract_content_chunks USING ivfflat (chunk_embeddings vector_cosine_ops);
CREATE INDEX idx_chunk_contract ON contract_content_chunks(contract_id);
```

2. **Enable Partial Indexes:**
```sql
CREATE INDEX idx_undisplayed_docs ON contract_retrieved_document_details(session_id, question) WHERE is_displayed = false;
```

### Application Optimization

1. **Increase Worker Count:**
```python
# In Config.py
WORKER_COUNT = 4  # Increase for high traffic
WORKER_THREADS_COUNT = 8
```

2. **Connection Pooling:**
```python
# In Config.py
MONGO_MAX_POOL_SIZE = 20
MONGO_MIN_POOL_SIZE = 5
```

3. **Batch Configuration:**
```python
MAX_CHUNKS_TO_USE = 20  # Increase in ContractFinderAgentSetup
MAX_CONTRACTS_TO_RETURN = 50
CONTEXT_THRESHOLD = 0.7  # Lower to include more results
```

4. **Caching:**
Enable Redis caching for embeddings:
```python
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_DB = 0
```

### Monitoring

1. **Enable Cloud Logging (GCP):**
```bash
gcloud logging read "resource.type=gce_instance AND labels.pod_name=contract-finder-agent" \
  --limit 50 --format json
```

2. **Monitor Metrics:**
- Request latency: `avg_response_time`
- Throughput: `requests_per_second`
- Error rate: `error_percentage`
- Database query time: `avg_query_time_ms`

---

## Next Steps

1. ✅ Complete local development setup
2. ✅ Load sample contract data
3. ✅ Test prediction endpoint
4. ✅ Configure production environment variables
5. ✅ Build and push Docker image
6. ✅ Deploy to Kubernetes cluster
7. ✅ Set up monitoring and alerting
8. ✅ Configure CI/CD pipeline

---

## Support & Documentation

- **Implementation Validation:** See `IMPLEMENTATION_VALIDATION.md`
- **API Documentation:** See `run.py` docstrings
- **LLM Studio Docs:** https://github.com/techolution/llm-studio
- **Support Email:** contracts-team@company.com

---

**Last Updated:** 2024
**Version:** 2.0
**Status:** Production Ready ✅
