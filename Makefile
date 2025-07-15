.PHONY: help install test lint format check clean docs security
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	uv pip install -e .[dev,all]

test: ## Run tests
	pytest tests/

test-cov: ## Run tests with coverage
	pytest tests/ --cov=llamaagent --cov-report=html --cov-report=term

lint: ## Run linting
	ruff check src tests
	black --check src tests
	mypy src

format: ## Format code
	black src tests
	ruff check --fix src tests
	ruff format src tests
	isort src tests

check: format lint test ## Run all checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	sphinx-build -b html docs docs/_build/html

security: ## Run security checks
	bandit -r src/llamaagent
	safety check

tox-all: ## Run all tox environments
	tox

build: ## Build package
	python -m build

install-hooks: ## Install pre-commit hooks
	pre-commit install

validate: ## Validate package installation
	python -c "import sys; sys.path.insert(0, 'src'); import llamaagent; print('✅ Package imports successfully')"

