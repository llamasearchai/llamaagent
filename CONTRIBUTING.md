# Contributing to LlamaAgent

Thank you for your interest in contributing to LlamaAgent! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai).

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)
- PostgreSQL (optional, for database features)
- Redis (optional, for caching)

### Development Setup

1. **Fork and Clone**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/llamaagent.git
cd llamaagent

# Add upstream remote
git remote add upstream https://github.com/originalusername/llamaagent.git
```

2. **Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install Dependencies**

```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

4. **Verify Installation**

```bash
# Run tests to verify setup
pytest

# Run linting
ruff check src/ tests/
black --check src/ tests/
mypy src/
```

5. **Environment Configuration**

Create a `.env` file in the project root:

```bash
# LLM Provider settings (optional)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Database settings (optional)
DATABASE_URL=postgresql://user:pass@localhost/llamaagent_dev
REDIS_URL=redis://localhost:6379

# Development settings
LLAMAAGENT_DEBUG=true
LLAMAAGENT_LOG_LEVEL=DEBUG
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check `help wanted` for areas needing contribution
- Create a new issue for bugs or feature requests

### 2. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 3. Make Changes

- Follow the [code standards](#code-standards)
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new agent capability

- Implement new feature X
- Add comprehensive tests
- Update documentation
- Fixes #123"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Code Standards

### Python Style

- **Formatting**: Use [Black](https://black.readthedocs.io/) for code formatting
- **Linting**: Use [Ruff](https://docs.astral.sh/ruff/) for linting
- **Type Hints**: All functions must include type hints
- **Docstrings**: Use Google-style docstrings

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Naming Conventions

- **Classes**: PascalCase (`ReactAgent`, `ToolRegistry`)
- **Functions/Methods**: snake_case (`execute_task`, `get_provider`)
- **Variables**: snake_case (`agent_config`, `task_result`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private**: Leading underscore (`_internal_method`)

### Import Organization

```python
# Standard library imports
import asyncio
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
import pytest
from pydantic import BaseModel

# Local imports
from llamaagent.agents.base import BaseAgent
from llamaagent.tools import CalculatorTool
```

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_providers.py
├── integration/       # Integration tests
│   ├── test_api.py
│   └── test_workflow.py
├── e2e/              # End-to-end tests
│   └── test_complete_system.py
└── performance/      # Performance tests
    └── test_benchmarks.py
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from llamaagent import ReactAgent, AgentConfig
from llamaagent.tools import CalculatorTool

class TestReactAgent:
    """Test suite for ReactAgent."""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="TestAgent",
            tools=["calculator"],
            temperature=0.7
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Create test agent."""
        return ReactAgent(
            config=agent_config,
            tools=[CalculatorTool()]
        )
    
    async def test_execute_simple_task(self, agent):
        """Test simple task execution."""
        response = await agent.execute("What is 2 + 2?")
        
        assert response.success
        assert "4" in response.content
        assert response.execution_time > 0
    
    async def test_execute_with_context(self, agent):
        """Test task execution with context."""
        context = {"previous_result": 10}
        response = await agent.execute("Add 5 to the previous result", context)
        
        assert response.success
        assert response.metadata.get("context") == context
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_agents.py

# Run with coverage
pytest --cov=llamaagent --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only

# Run tests in parallel
pytest -n auto
```

## Documentation

### Docstring Format

```python
def execute_task(
    self,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0
) -> AgentResponse:
    """Execute a task and return response.
    
    Args:
        task: The task to execute
        context: Optional context dictionary
        timeout: Execution timeout in seconds
        
    Returns:
        AgentResponse with execution results
        
    Raises:
        AgentExecutionError: If task execution fails
        TimeoutError: If execution exceeds timeout
        
    Example:
        >>> agent = ReactAgent(config=config)
        >>> response = await agent.execute("Calculate 2 + 2")
        >>> print(response.content)
        "4"
    """
```

### Documentation Updates

- Update API documentation for new features
- Add examples for new functionality
- Update README for significant changes
- Add migration guides for breaking changes

## Issue Guidelines

### Bug Reports

Use the bug report template and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs or error messages

### Feature Requests

Use the feature request template and include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Potential breaking changes

### Questions

Use the question template for:

- Usage questions
- Best practices
- Architecture discussions

## Pull Request Guidelines

### PR Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)
- [ ] Related issues referenced
- [ ] Reviewer assigned

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Changes are tested in staging environment
4. **Approval**: Maintainer approves and merges the PR

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (X.Y.0): New features, backward compatible
- **Patch** (X.Y.Z): Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Tag created
- [ ] PyPI package published
- [ ] GitHub release created
- [ ] Docker image published

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and community support
- **Email**: Direct contact with maintainers

### Maintainers

- **Nik Jois** ([@nikjois](https://github.com/nikjois)) - Lead Maintainer
  - Email: [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)
  - Focus: Architecture, core features, releases

### Recognition

Contributors are recognized in:

- Repository README
- Release notes
- Annual contributor reports
- Conference talks and presentations

### Becoming a Maintainer

Regular contributors may be invited to become maintainers based on:

- Consistent, high-quality contributions
- Understanding of project architecture
- Community involvement and support
- Alignment with project values

## Getting Help

### Resources

- **Documentation**: [https://llamaagent.readthedocs.io](https://llamaagent.readthedocs.io)
- **API Reference**: [https://llamaagent.readthedocs.io/api](https://llamaagent.readthedocs.io/api)
- **Examples**: [https://github.com/yourusername/llamaagent/tree/main/examples](https://github.com/yourusername/llamaagent/tree/main/examples)

### Support

- **Bug Reports**: Create an issue with the bug report template
- **Feature Requests**: Create an issue with the feature request template
- **Questions**: Use GitHub Discussions or Discord
- **Security Issues**: Email [security@llamasearch.ai](mailto:security@llamasearch.ai)

### Development Tips

1. **Start Small**: Begin with small, focused changes
2. **Ask Questions**: Don't hesitate to ask for clarification
3. **Follow Patterns**: Look at existing code for patterns
4. **Test Thoroughly**: Write comprehensive tests
5. **Document Changes**: Update documentation as needed

## License

By contributing to LlamaAgent, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to LlamaAgent! Your contributions help make this project better for everyone.