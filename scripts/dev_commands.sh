#!/bin/bash
# Development commands for LlamaAgent

# Activate virtual environment
source .venv/bin/activate

# Available commands:
echo "Available development commands:"
echo "  make test          - Run tests"
echo "  make lint          - Run linting"
echo "  make format        - Format code"
echo "  make check         - Run all checks"
echo "  make install-dev   - Install in development mode"
echo "  make clean         - Clean build artifacts"

# Hatch commands
echo "  hatch run test     - Run tests with hatch"
echo "  hatch run lint     - Run linting with hatch"
echo "  hatch run format   - Format code with hatch"
echo "  hatch run check    - Run all checks with hatch"

# Tox commands
echo "  tox                - Run tests in multiple environments"
echo "  tox -e lint        - Run linting"
echo "  tox -e docs        - Build documentation"
echo "  tox -e security    - Run security checks"
