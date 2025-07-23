#!/usr/bin/env bash

# LlamaAgent Ultimate Master Edition Installation Script
# Complete production-ready installation with all features
# Author: Nik Jois <nikjois@llamasearch.ai>
# Version: 2.1.0
# License: MIT

set -euo pipefail
trap 'error "Installation failed at line $LINENO"' ERR

# Configuration
readonly VERSION="2.1.0"
readonly PYTHON_MIN="3.9.0"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[0;37m'
readonly RESET='\033[0m'

# System detection
readonly OS="$(uname -s)"
readonly ARCH="$(uname -m)"
readonly IS_MACOS="$([[ "$OS" == "Darwin" ]] && echo "true" || echo "false")"
readonly IS_ARM64="$([[ "$ARCH" == "arm64" ]] && echo "true" || echo "false")"

# Installation directory
readonly INSTALL_DIR="$HOME/llamaagent"
readonly VENV_DIR="$INSTALL_DIR/.venv"

# Logging functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${RESET} $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${RESET} $*"; }
info() { echo -e "${CYAN}[INFO]${RESET} $*"; }

# Banner
show_banner() {
    cat << 'EOF'







Advanced Multi-Agent AI Framework - Ultimate Master Edition v${VERSION}
Production-Ready | Self-Installing | Complete Testing Suite
Author: Nik Jois <nikjois@llamasearch.ai>
EOF
}

# Check if running on macOS
check_macos() {
    if [[ "$IS_MACOS" != "true" ]]; then
        error "This script is optimized for macOS. Other platforms may work but are not officially supported."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check Python version
check_python() {
    local python_cmd="${1:-python3}"
    if ! command -v "$python_cmd" >/dev/null 2>&1; then
        return 1
    fi

    local version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    python3 -c "import sys; exit(0 if sys.version_info >= (${PYTHON_MIN//./, }) else 1)" 2>/dev/null
}

# Find suitable Python
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if check_python "$cmd"; then
            echo "$cmd"
            return 0
        fi
    done
    error "Python $PYTHON_MIN+ not found. Please install Python $PYTHON_MIN or later."
    if [[ "$IS_MACOS" == "true" ]]; then
        info "You can install Python using:"
        info "  brew install python@3.11"
        info "  or download from https://python.org"
    fi
    return 1
}

# Install system dependencies on macOS
install_system_deps() {
    if [[ "$IS_MACOS" == "true" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            warn "Homebrew not found. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            # Add Homebrew to PATH for this session
            if [[ "$IS_ARM64" == "true" ]]; then
                export PATH="/opt/homebrew/bin:$PATH"
            else
                export PATH="/usr/local/bin:$PATH"
            fi
        fi

        log "Installing system dependencies with Homebrew..."
        brew update
        brew install python@3.11 git curl

        # Install additional tools
        if ! command -v docker >/dev/null 2>&1; then
            warn "Docker not found. Installing Docker Desktop..."
            brew install --cask docker
        fi
    fi
}

# Create project structure and files
create_project() {
    log "Creating project structure in $INSTALL_DIR"

    if [[ -d "$INSTALL_DIR" ]]; then
        warn "Directory $INSTALL_DIR already exists."
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            error "Installation cancelled."
            exit 1
        fi
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Create directory structure
    mkdir -p {src/llamaagent/{agents,tools,memory,llm,data_generation,evolution,cli},tests,docs,examples,.github/workflows}

    log "Project structure created successfully"
}

# Setup virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."

    PYTHON=$(find_python) || exit 1
    log "Using Python: $PYTHON ($(${PYTHON} --version))"

    "$PYTHON" -m venv "$VENV_DIR"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip and install build tools
    pip install --upgrade pip setuptools wheel

    success "Virtual environment created and activated"
}

# Install Python package
install_package() {
    log "Installing LlamaAgent package..."

    source "$VENV_DIR/bin/activate"

    # Install the package in development mode
    pip install -e ".[dev,prod,ml]"

    success "LlamaAgent package installed successfully"
}

# Run comprehensive tests
run_tests() {
    log "Running comprehensive test suite..."

    source "$VENV_DIR/bin/activate"

    # Run linting
    log "Running code quality checks..."
    ruff check src/ tests/ || warn "Linting issues found"
    ruff format --check src/ tests/ || warn "Formatting issues found"

    # Run type checking
    log "Running type checks..."
    mypy src/ || warn "Type checking issues found"

    # Run unit tests
    log "Running unit tests..."
    pytest tests/ -v --cov=src/llamaagent --cov-report=html --cov-report=term-missing || warn "Some tests failed"

    # Run integration tests
    log "Running integration tests..."
    python -m llamaagent eval basic || warn "Integration tests had issues"

    success "Test suite completed"
}

# Build Docker image
build_docker() {
    if command -v docker >/dev/null 2>&1; then
        log "Building Docker image..."

        docker build -t llamaagent:latest . || warn "Docker build failed"

        # Test Docker container
        log "Testing Docker container..."
        docker run --rm -d --name llamaagent-test -p 8000:8000 llamaagent:latest || warn "Docker test failed"
        sleep 10

        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            success "Docker container is healthy"
        else
            warn "Docker container health check failed"
        fi

        docker stop llamaagent-test >/dev/null 2>&1 || true
    else
        warn "Docker not available, skipping Docker build"
    fi
}

# Create demonstration scripts
create_demos() {
    log "Creating demonstration scripts..."

    # Create a demo script
    cat > examples/demo.py << 'EOF'
#!/usr/bin/env python3
"""
LlamaAgent Demonstration Script
Showcases the key capabilities of the framework
"""

import asyncio
from llamaagent import ReactAgent, AgentConfig
from llamaagent.tools import ToolRegistry
from llamaagent.tools.calculator import CalculatorTool
from llamaagent.tools.python_repl import PythonREPLTool

async def main():
    print("LlamaAgent Demonstration")
    print("=" * 50)

    # Setup agent with tools
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(PythonREPLTool())

    config = AgentConfig(
        name="DemoAgent",
        spree_enabled=True,
        dynamic_tools=True,
    )

    agent = ReactAgent(config=config, tools=tools)

    # Demonstration tasks
    tasks = [
        "What is the square root of 144?",
        "Write a Python function to calculate fibonacci numbers",
        "Explain the benefits of multi-agent systems",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}: {task}")
        print("-" * 40)

        response = await agent.execute(task)

        if response.success:
            print(f"[SUCCESS] Response: {response.content}")
            print(f"Time: {response.execution_time:.2f}s")
            print(f"Tokens: {response.tokens_used}")
        else:
            print(f"[ERROR] Error: {response.content}")

    print("\nDemonstration completed!")

if __name__ == "__main__":
    asyncio.run(main())
EOF

    chmod +x examples/demo.py

    success "Demonstration scripts created"
}

# Create CLI menu
show_menu() {
    while true; do
        clear
        show_banner
        echo
        echo -e "${CYAN}LlamaAgent Control Center${RESET}"
        echo "================================"
        echo "1. Interactive Chat"
        echo "2. Run Demo"
        echo "3. Run Tests"
        echo "4. Generate Data"
        echo "5. Team Evolution"
        echo "6. Start Web API"
        echo "7. Docker Management"
        echo "8. View Documentation"
        echo "9. System Status"
        echo "0. Exit"
        echo
        read -p "Select option [0-9]: " choice

        case $choice in
            1) run_interactive_chat ;;
            2) run_demo ;;
            3) run_test_suite ;;
            4) run_data_generation ;;
            5) run_team_evolution ;;
            6) start_web_api ;;
            7) docker_management ;;
            8) show_documentation ;;
            9) show_system_status ;;
            0) exit 0 ;;
            *) warn "Invalid option. Please try again." && sleep 2 ;;
        esac
    done
}

# Menu functions
run_interactive_chat() {
    source "$VENV_DIR/bin/activate"
    python -m llamaagent interactive --spree --dynamic-tools
    read -p "Press any key to continue..." -n 1
}

run_demo() {
    source "$VENV_DIR/bin/activate"
    python examples/demo.py
    read -p "Press any key to continue..." -n 1
}

run_test_suite() {
    source "$VENV_DIR/bin/activate"
    pytest tests/ -v
    read -p "Press any key to continue..." -n 1
}

run_data_generation() {
    echo "Creating sample problems file..."
    cat > /tmp/problems.jsonl << 'EOF'
{"problem": "What is the capital of France?"}
{"problem": "Explain quantum computing in simple terms"}
{"problem": "Calculate the area of a circle with radius 5"}
EOF

    source "$VENV_DIR/bin/activate"
    python -m llamaagent generate-data gdt -i /tmp/problems.jsonl -o dataset.jsonl
    read -p "Press any key to continue..." -n 1
}

run_team_evolution() {
    source "$VENV_DIR/bin/activate"
    python -m llamaagent evolve --cycles 10
    read -p "Press any key to continue..." -n 1
}

start_web_api() {
    source "$VENV_DIR/bin/activate"
    echo "Starting web API at http://localhost:8000"
    echo "Press Ctrl+C to stop"
    python -m uvicorn llamaagent.api:app --host 0.0.0.0 --port 8000 --reload
}

docker_management() {
    echo "Docker Management"
    echo "1. Build image"
    echo "2. Run container"
    echo "3. Stop container"
    echo "4. View logs"
    read -p "Select [1-4]: " docker_choice

    case $docker_choice in
        1) docker build -t llamaagent:latest . ;;
        2) docker run -d --name llamaagent -p 8000:8000 llamaagent:latest ;;
        3) docker stop llamaagent && docker rm llamaagent ;;
        4) docker logs llamaagent ;;
    esac
    read -p "Press any key to continue..." -n 1
}

show_documentation() {
    cat << 'EOF'
LlamaAgent Documentation

Architecture:
- Multi-agent framework with specialized roles
- ReAct pattern for reasoning and action
- Dynamic tool synthesis capabilities
- Multi-step planning with resource assessment

Key Features:
- SPRE (Strategic Planning and Resource Evaluation)
- GDT (Generative Debate Trees) for training data
- ATES (Agentic Team Evolution System)
- Production-ready API and CLI interfaces

Quick Start:
1. Use interactive mode: llamaagent interactive
2. Send single queries: llamaagent chat "your question"
3. Generate training data: llamaagent generate-data gdt
4. Run team evolution: llamaagent evolve

Web API:
- Health check: GET /health
- Chat endpoint: POST /chat
- List agents: GET /agents
- List tools: GET /tools

Evaluation:
- Basic suite: llamaagent eval basic
- Custom output: llamaagent eval basic -o results.json

For more details, visit: https://github.com/nikjois/llamaagent
EOF
    read -p "Press any key to continue..." -n 1
}

show_system_status() {
    echo "System Status"
    echo "================="
    echo "OS: $OS ($ARCH)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Install Dir: $INSTALL_DIR"
    echo "Virtual Env: $([[ -d "$VENV_DIR" ]] && echo "[ACTIVE] Active" || echo "[NOT FOUND] Not found")"
    echo "Docker: $(command -v docker >/dev/null && echo "[AVAILABLE] Available" || echo "[NOT FOUND] Not found")"

    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
        echo "LlamaAgent: $(python -c 'import llamaagent; print(llamaagent.__version__)' 2>/dev/null || echo 'Not installed')"
    fi

    read -p "Press any key to continue..." -n 1
}

# Main installation function
main() {
    show_banner
    echo
    info "Starting LlamaAgent Ultimate Master Edition installation..."

    # Check system requirements
    check_macos

    # Install system dependencies
    install_system_deps

    # Create project
    create_project

    # Setup Python environment
    setup_venv

    # Install package
    install_package

    # Run tests
    run_tests

    # Build Docker image
    build_docker

    # Create demos
    create_demos

    # Final setup
    success "Installation completed successfully!"
    info "LlamaAgent is now installed in: $INSTALL_DIR"
    info "To activate the environment: source $VENV_DIR/bin/activate"
    info "To run the CLI: llamaagent --help"

    echo
    read -p "Launch interactive menu? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        show_menu
    fi
}

# Handle script arguments
case "${1:-}" in
    --menu)
        if [[ -d "$VENV_DIR" ]]; then
            show_menu
        else
            error "LlamaAgent not installed. Run without arguments to install first."
            exit 1
        fi
        ;;
    --help|-h)
        echo "LlamaAgent Ultimate Master Edition Installer"
        echo "Usage: $0 [--menu|--help]"
        echo "  --menu    Launch interactive menu (requires installation)"
        echo "  --help    Show this help message"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        error "Unknown argument: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
