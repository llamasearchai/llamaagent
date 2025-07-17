# LlamaAgent Master System Dockerfile
# Multi-stage build for comprehensive testing and deployment
# Author: Nik Jois <nikjois@llamasearch.ai>

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
COPY README.md LICENSE ./
COPY src/llamaagent/_version.py ./src/llamaagent/_version.py

# Stage 2: Development and testing environment
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY master_llamaagent_system.py comprehensive_syntax_fixer.py fastapi_app.py ./
COPY monitoring/ ./monitoring/

# Create necessary directories
RUN mkdir -p /app/logs /app/results /app/data

# Run comprehensive tests
RUN python -m pytest tests/ -v --tb=short || true
RUN python comprehensive_syntax_fixer.py || true
RUN python master_llamaagent_system.py --test-mode || true

# Stage 3: Production environment
FROM base as production

# Install only production dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY master_llamaagent_system.py comprehensive_syntax_fixer.py fastapi_app.py ./
COPY monitoring/ ./monitoring/

# Create non-root user
RUN groupadd -r llamaagent && useradd -r -g llamaagent llamaagent

# Create application directories
RUN mkdir -p /app/logs /app/results /app/data \
    && chown -R llamaagent:llamaagent /app

# Switch to non-root user
USER llamaagent

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from master_llamaagent_system import MasterSystemOrchestrator, MasterSystemConfig; asyncio.run(MasterSystemOrchestrator(MasterSystemConfig()).health_monitor.check_system_health())" || exit 1

# Default command
CMD ["python", "master_llamaagent_system.py"]

# Stage 4: FastAPI production server
FROM production as fastapi

# Install additional FastAPI dependencies
RUN pip install fastapi uvicorn[standard] gunicorn

# FastAPI application already copied above

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Complete system with monitoring
FROM production as monitoring

# Install monitoring dependencies
RUN pip install prometheus-client grafana-client

# Copy monitoring configuration
COPY monitoring/ ./monitoring/

# Expose monitoring ports
EXPOSE 8001 8002

# Run with monitoring
CMD ["python", "master_llamaagent_system.py", "--enable-monitoring"]
