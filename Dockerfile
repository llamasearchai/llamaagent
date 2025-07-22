# LlamaAgent Docker Image
# Author: Nik Jois <nikjois@llamasearch.ai>

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create non-root user
RUN useradd --create-home --shell /bin/bash llamaagent
USER llamaagent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.llamaagent.api"]
