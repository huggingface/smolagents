# Contract Finder Agent v2 - Production Dockerfile
# Based on reference from Book Finder Agent
# Supports LLM Studio framework with custom Google Artifact Registry

FROM python:3.10-slim
LABEL authors=\\\"Engineering Team\\\" \\\\
      version=\\\"2.0\\\" \\\\
      description=\\\"Contract Finder Agent v2 - RAG service for contract retrieval\\\"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements file
COPY requirements.txt .

# Install dependencies with support for Google Artifact Registry (GAR)
# This allows pulling LLM Studio packages from custom enterprise registries
ARG GOOGLE_APPLICATION_CREDENTIALS
RUN --mount=type=secret,id=gcp_creds,target=/tmp/gcp_key.json,mode=0444 \\\\
    pip install keyring && \\\\
    pip install keyrings.google-artifactregistry-auth && \\\\
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_key.json && \\\\
    # Install from enterprise GAR with fallback to PyPI
    pip install --index-url https://us-central1-python.pkg.dev/alan-suite/llm-studio-pypi/simple \\\\
                --extra-index-url https://pypi.org/simple/ \\\\
                -r requirements.txt

# Copy necessary files
COPY . .
COPY agents/contract_finder_agent /app/agents/contract_finder_agent
COPY agents/run.py run.py

# Create non-root user for security
RUN useradd -m -u 1000 agent && \\\\
    mkdir -p /app/logs /app/data && \\\\
    chown -R agent:agent /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\\\
    curl \\\\
    libpq5 \\\\
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3\
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run application entry point
ENTRYPOINT ["python", "run.py"]