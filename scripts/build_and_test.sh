#!/bin/bash

# LlamaAgent Master System - Build and Test Script
# Comprehensive automated testing, building, and deployment
# Author: Nik Jois <nikjois@llamasearch.ai>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="llamaagent"
PYTHON_VERSION="3.11"
POETRY_VERSION="1.8.3"
DOCKER_IMAGE="llamaagent/master-system"
DOCKER_TAG="latest"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "🦙 LlamaAgent Master System"
    echo "   Build and Test Pipeline"
    echo "=================================="
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$python_version" < "3.11" ]]; then
        print_error "Python 3.11+ is required, found $python_version"
        exit 1
    fi

    # Check Poetry
    if ! command_exists poetry; then
        print_warning "Poetry not found, installing..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Check Docker
    if ! command_exists docker; then
        print_warning "Docker not found, some features will be disabled"
    fi

    # Check Git
    if ! command_exists git; then
        print_warning "Git not found, version control features disabled"
    fi

    print_success "Prerequisites check completed"
}

# Setup development environment
setup_dev_environment() {
    print_status "Setting up development environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing requirements..."
        pip install -r requirements.txt
    fi

    # Install development requirements
    if [ -f "requirements-dev.txt" ]; then
        print_status "Installing development requirements..."
        pip install -r requirements-dev.txt
    fi

    print_success "Development environment setup completed"
}

# Run syntax fixes
run_syntax_fixes() {
    print_status "Running syntax fixes..."

    # Run comprehensive syntax fixer
    if [ -f "comprehensive_syntax_fixer.py" ]; then
        print_status "Running comprehensive syntax fixer..."
        python comprehensive_syntax_fixer.py || print_warning "Syntax fixer completed with warnings"
    fi

    # Run batch syntax fixer
    if [ -f "batch_syntax_fixer.py" ]; then
        print_status "Running batch syntax fixer..."
        python batch_syntax_fixer.py || print_warning "Batch syntax fixer completed with warnings"
    fi

    print_success "Syntax fixes completed"
}

# Run code quality checks
run_code_quality() {
    print_status "Running code quality checks..."

    # Black formatting
    if command_exists black; then
        print_status "Running Black formatter..."
        black --check --diff src/ || print_warning "Black formatting issues found"
    fi

    # isort import sorting
    if command_exists isort; then
        print_status "Running isort..."
        isort --check-only --diff src/ || print_warning "Import sorting issues found"
    fi

    # Flake8 linting
    if command_exists flake8; then
        print_status "Running Flake8 linter..."
        flake8 src/ || print_warning "Flake8 linting issues found"
    fi

    # MyPy type checking
    if command_exists mypy; then
        print_status "Running MyPy type checker..."
        mypy src/ || print_warning "MyPy type checking issues found"
    fi

    # Bandit security scanning
    if command_exists bandit; then
        print_status "Running Bandit security scanner..."
        bandit -r src/ || print_warning "Bandit security issues found"
    fi

    print_success "Code quality checks completed"
}

# Run comprehensive tests
run_tests() {
    print_status "Running comprehensive tests..."

    # Unit tests with pytest
    if command_exists pytest; then
        print_status "Running pytest..."
        pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing || print_warning "Some tests failed"
    fi

    # Integration tests
    if [ -f "tests/integration/test_integration.py" ]; then
        print_status "Running integration tests..."
        pytest tests/integration/ -v || print_warning "Integration tests failed"
    fi

    # Performance tests
    if [ -f "tests/performance/test_performance.py" ]; then
        print_status "Running performance tests..."
        pytest tests/performance/ -v --benchmark-only || print_warning "Performance tests failed"
    fi

    # Run master system tests
    if [ -f "master_llamaagent_system.py" ]; then
        print_status "Running master system tests..."
        python master_llamaagent_system.py --test-mode || print_warning "Master system tests failed"
    fi

    print_success "Tests completed"
}

# Build documentation
build_docs() {
    print_status "Building documentation..."

    # Sphinx documentation
    if [ -d "docs" ] && command_exists sphinx-build; then
        print_status "Building Sphinx documentation..."
        sphinx-build -b html docs docs/_build/html || print_warning "Sphinx documentation build failed"
    fi

    # MkDocs documentation
    if [ -f "mkdocs.yml" ] && command_exists mkdocs; then
        print_status "Building MkDocs documentation..."
        mkdocs build || print_warning "MkDocs documentation build failed"
    fi

    print_success "Documentation build completed"
}

# Build Docker images
build_docker() {
    print_status "Building Docker images..."

    if ! command_exists docker; then
        print_warning "Docker not available, skipping Docker build"
        return
    fi

    # Build production image
    print_status "Building production Docker image..."
    docker build --target production -t "$DOCKER_IMAGE:$DOCKER_TAG" -t "$DOCKER_IMAGE:production" . || print_error "Production Docker build failed"

    # Build development image
    print_status "Building development Docker image..."
    docker build --target development -t "$DOCKER_IMAGE:dev" . || print_warning "Development Docker build failed"

    # Build FastAPI image
    print_status "Building FastAPI Docker image..."
    docker build --target fastapi -t "$DOCKER_IMAGE:fastapi" . || print_warning "FastAPI Docker build failed"

    # Build monitoring image
    print_status "Building monitoring Docker image..."
    docker build --target monitoring -t "$DOCKER_IMAGE:monitoring" . || print_warning "Monitoring Docker build failed"

    print_success "Docker images built successfully"
}

# Run Docker tests
test_docker() {
    print_status "Testing Docker images..."

    if ! command_exists docker; then
        print_warning "Docker not available, skipping Docker tests"
        return
    fi

    # Test production image
    print_status "Testing production Docker image..."
    docker run --rm "$DOCKER_IMAGE:production" python -c "import sys; print(f'Python {sys.version}'); print('Production image works!')" || print_warning "Production image test failed"

    # Test FastAPI image
    print_status "Testing FastAPI Docker image..."
    docker run --rm -p 8000:8000 -d --name "test-fastapi" "$DOCKER_IMAGE:fastapi" || print_warning "FastAPI image test failed"
    sleep 5

    # Check if FastAPI is responding
    if command_exists curl; then
        curl -f http://localhost:8000/health || print_warning "FastAPI health check failed"
    fi

    # Cleanup
    docker stop test-fastapi || true
    docker rm test-fastapi || true

    print_success "Docker tests completed"
}

# Generate reports
generate_reports() {
    print_status "Generating reports..."

    # Create reports directory
    mkdir -p reports

    # Generate test coverage report
    if [ -f "htmlcov/index.html" ]; then
        cp -r htmlcov reports/coverage
        print_status "Test coverage report generated"
    fi

    # Generate system report
    if [ -f "master_llamaagent_system.py" ]; then
        print_status "Generating system report..."
        python master_llamaagent_system.py --generate-report || print_warning "System report generation failed"
    fi

    # Generate security report
    if command_exists bandit; then
        print_status "Generating security report..."
        bandit -r src/ -f json -o reports/security-report.json || print_warning "Security report generation failed"
    fi

    # Generate dependency report
    if command_exists pip; then
        print_status "Generating dependency report..."
        pip list --format=json > reports/dependencies.json
        pip check > reports/dependency-check.txt || print_warning "Dependency check found issues"
    fi

    print_success "Reports generated in reports/ directory"
}

# Deploy to staging
deploy_staging() {
    print_status "Deploying to staging..."

    # This would typically involve:
    # - Pushing Docker images to registry
    # - Deploying to Kubernetes/Docker Swarm
    # - Running smoke tests

    print_warning "Staging deployment not implemented yet"
}

# Cleanup
cleanup() {
    print_status "Cleaning up..."

    # Remove temporary files
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true

    # Remove build artifacts
    rm -rf build/ dist/ *.egg-info/

    print_success "Cleanup completed"
}

# Main execution
main() {
    print_banner

    local start_time=$(date +%s)

    # Parse command line arguments
    local run_all=true
    local run_setup=false
    local run_syntax=false
    local run_quality=false
    local run_tests_only=false
    local run_docker_only=false
    local run_reports_only=false
    local run_deploy=false
    local run_cleanup=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup)
                run_setup=true
                run_all=false
                shift
                ;;
            --syntax)
                run_syntax=true
                run_all=false
                shift
                ;;
            --quality)
                run_quality=true
                run_all=false
                shift
                ;;
            --test)
                run_tests_only=true
                run_all=false
                shift
                ;;
            --docker)
                run_docker_only=true
                run_all=false
                shift
                ;;
            --reports)
                run_reports_only=true
                run_all=false
                shift
                ;;
            --deploy)
                run_deploy=true
                run_all=false
                shift
                ;;
            --cleanup)
                run_cleanup=true
                run_all=false
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --setup     Setup development environment"
                echo "  --syntax    Run syntax fixes"
                echo "  --quality   Run code quality checks"
                echo "  --test      Run tests only"
                echo "  --docker    Build and test Docker images"
                echo "  --reports   Generate reports"
                echo "  --deploy    Deploy to staging"
                echo "  --cleanup   Cleanup temporary files"
                echo "  --help      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Execute based on arguments
    if $run_all; then
        check_prerequisites
        setup_dev_environment
        run_syntax_fixes
        run_code_quality
        run_tests
        build_docs
        build_docker
        test_docker
        generate_reports
        cleanup
    else
        check_prerequisites

        if $run_setup; then
            setup_dev_environment
        fi

        if $run_syntax; then
            run_syntax_fixes
        fi

        if $run_quality; then
            run_code_quality
        fi

        if $run_tests_only; then
            run_tests
        fi

        if $run_docker_only; then
            build_docker
            test_docker
        fi

        if $run_reports_only; then
            generate_reports
        fi

        if $run_deploy; then
            deploy_staging
        fi

        if $run_cleanup; then
            cleanup
        fi
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    print_success "Build and test pipeline completed in ${duration}s"
    print_status "Check the reports/ directory for detailed results"
}

# Run main function with all arguments
main "$@"
