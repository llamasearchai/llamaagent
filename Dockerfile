FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .[api]

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash llamaagent
USER llamaagent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use *uvicorn* as the entrypoint so additional CLI flags (e.g. `--help`) work
ENTRYPOINT ["uvicorn"]

# Default command arguments
CMD ["llamaagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
