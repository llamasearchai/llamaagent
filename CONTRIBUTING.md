# Contributing to LlamaAgent

Thank you for your interest in contributing to LlamaAgent! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our Code of Conduct.

## How to Contribute

There are many ways to contribute to LlamaAgent:

1. **Report bugs**: Submit issues for any bugs you encounter.
2. **Suggest features**: Submit issues with feature requests or enhancement ideas.
3. **Improve documentation**: Help improve the documentation by fixing typos, clarifying explanations, or adding examples.
4. **Contribute code**: Submit pull requests with bug fixes, features, or improvements.

## Development Setup

### Prerequisites


- Python 3.8 or higher
- Git


### Setting Up the Development Environment

1. **Fork the repository** by clicking the "Fork" button on the top-right of the repository page.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llamaagent.git
   cd llamaagent
   ```

3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/llamasearchai/llamaagent.git
   ```


4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

6. **Run tests**:
   ```bash
   pytest
   ```


## Pull Request Guidelines

- Pull requests should focus on a single feature or fix.
- Include tests for any new functionality or bug fixes.
- Update documentation to reflect any changes in public API or behavior.
- Follow the existing code style and conventions.
- Keep your pull request up-to-date with the main branch.

## Code Style Guidelines


We follow standard Python conventions:

- Use [Black](https://black.readthedocs.io/) for code formatting.
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
- Use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Add type hints for function parameters and return values.


## License

By contributing to LlamaAgent, you agree that your contributions will be licensed under the project's MIT License.
