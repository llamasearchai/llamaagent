#!/bin/bash
#
# LlamaAgent Master Program Startup Script
# Complete setup and launch for the AI agent system
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo ""
echo "           LlamaAgent Master Program Launcher              "
echo "                                                           "
echo "  Complete AI Agent System with Dynamic Task Planning      "
echo "                                                           "
echo "  Version: 2.0.0 | Author: Nik Jois                       "
echo ""
echo -e "${NC}"

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}FAIL Python 3 is not installed${NC}"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        echo -e "${RED}FAIL Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi

    echo -e "${GREEN}PASS Python $PYTHON_VERSION${NC}"

    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}PASS Docker is installed${NC}"
        DOCKER_AVAILABLE=true
    else
        echo -e "${YELLOW}WARNING:  Docker not found (optional)${NC}"
        DOCKER_AVAILABLE=false
    fi
}

# Function to setup environment
setup_environment() {
    echo -e "\n${YELLOW}Setting up environment...${NC}"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --quiet --upgrade pip

    # Install requirements
    echo "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install --quiet -r requirements.txt
    fi

    # Install package in development mode
    pip install --quiet -e .

    # Set Python path
    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

    echo -e "${GREEN}PASS Environment ready${NC}"
}

# Function to check API keys
check_api_keys() {
    echo -e "\n${YELLOW}Checking API keys...${NC}"

    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}WARNING:  OPENAI_API_KEY not set${NC}"
        echo "OpenAI integration will be disabled."
        echo "To enable, run: export OPENAI_API_KEY='your-key'"
        OPENAI_ENABLED=false
    else
        echo -e "${GREEN}PASS OpenAI API key configured${NC}"
        OPENAI_ENABLED=true
    fi

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${YELLOW}WARNING:  ANTHROPIC_API_KEY not set${NC}"
    else
        echo -e "${GREEN}PASS Anthropic API key configured${NC}"
    fi
}

# Function to create necessary directories
create_directories() {
    echo -e "\n${YELLOW}Creating directories...${NC}"

    mkdir -p logs data cache config/ssl nginx grafana/dashboards grafana/datasources prometheus

    echo -e "${GREEN}PASS Directories created${NC}"
}

# Function to run tests
run_tests() {
    echo -e "\n${YELLOW}Running system tests...${NC}"

    if python3 test_master_program.py > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}PASS All tests passed${NC}"
    else
        echo -e "${RED}FAIL Tests failed${NC}"
        echo "Check /tmp/test_output.log for details"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to start with Docker
start_docker() {
    echo -e "\n${BLUE}Starting with Docker...${NC}"

    # Build and start containers
    docker-compose -f docker-compose.master.yml up -d --build

    echo -e "${GREEN}PASS Docker containers started${NC}"
    echo -e "Services:"
    echo -e "  - API: http://localhost:8000"
    echo -e "  - Prometheus: http://localhost:9091"
    echo -e "  - Grafana: http://localhost:3000 (admin/admin)"
    echo -e "  - PostgreSQL: localhost:5432"
    echo -e "  - Redis: localhost:6379"
}

# Function to start standalone
start_standalone() {
    echo -e "\n${BLUE}Starting standalone server...${NC}"

    # Check if port is already in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}FAIL Port 8000 is already in use${NC}"
        echo "Please stop the existing service or choose a different port"
        exit 1
    fi

    # Start the server
    echo -e "${GREEN}Starting LlamaAgent Master Program...${NC}"
    echo -e "API will be available at: ${BLUE}http://localhost:8000${NC}"
    echo -e "WebSocket endpoint: ${BLUE}ws://localhost:8000/ws${NC}"
    echo -e "Documentation: ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "\nPress Ctrl+C to stop\n"

    python3 llamaagent_master_program.py server --host 0.0.0.0 --port 8000
}

# Function to show menu
show_menu() {
    echo -e "\n${YELLOW}How would you like to start the system?${NC}"
    echo "1) Standalone mode (quick start)"
    echo "2) Docker mode (full stack)"
    echo "3) Development mode (with reload)"
    echo "4) Run demo"
    echo "5) Run CLI monitor"
    echo "6) Execute a task"
    echo "7) Exit"
    echo
    read -p "Enter choice (1-7): " choice

    case $choice in
        1)
            start_standalone
            ;;
        2)
            if [ "$DOCKER_AVAILABLE" = true ]; then
                start_docker
            else
                echo -e "${RED}Docker is not installed${NC}"
                exit 1
            fi
            ;;
        3)
            echo -e "${GREEN}Starting in development mode...${NC}"
            python3 llamaagent_master_program.py server --reload
            ;;
        4)
            echo -e "${GREEN}Running demo...${NC}"
            python3 llamaagent_master_program.py demo
            ;;
        5)
            echo -e "${GREEN}Starting monitor...${NC}"
            python3 llamaagent_master_program.py monitor
            ;;
        6)
            read -p "Enter task description: " task
            echo -e "${GREEN}Executing task...${NC}"
            python3 llamaagent_master_program.py execute "$task"
            ;;
        7)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    # Check dependencies
    check_dependencies

    # Setup environment
    setup_environment

    # Check API keys
    check_api_keys

    # Create directories
    create_directories

    # Run tests (optional)
    read -p "Run system tests? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi

    # Show menu
    show_menu
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; exit 0' INT

# Run main function
main
