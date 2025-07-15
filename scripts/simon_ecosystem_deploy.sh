#!/bin/bash

# Simon Willison's LLM Ecosystem - Complete Deployment Script
# Automated setup and deployment with all dependencies
# Author: Nik Jois <nikjois@llamasearch.ai>

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_MODE="${1:-development}"  # development or production
ENVIRONMENT_FILE="${PROJECT_ROOT}/.env"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    log "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/exports"
    mkdir -p "${PROJECT_ROOT}/backups"
    mkdir -p "${PROJECT_ROOT}/ssl"
    mkdir -p "${PROJECT_ROOT}/grafana/dashboards"
    mkdir -p "${PROJECT_ROOT}/grafana/datasources"
    
    # Set permissions
    chmod 755 "${PROJECT_ROOT}/data"
    chmod 755 "${PROJECT_ROOT}/logs"
    chmod 755 "${PROJECT_ROOT}/exports"
    
    log "Directories created successfully"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
        log "Creating .env file from template..."
        cat > "$ENVIRONMENT_FILE" << EOF
# Simon Willison's LLM Ecosystem - Environment Configuration
# Copy this file to .env and fill in your API keys

# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
POSTGRES_PASSWORD=simon123
REDIS_PASSWORD=redis123

# Monitoring
GRAFANA_PASSWORD=admin123

# Security
JWT_SECRET_KEY=$(openssl rand -base64 32)

# Application Settings
LOG_LEVEL=INFO
ENABLE_COMMAND_TOOL=false
MAX_WORKERS=4

# Deployment Mode
DEPLOYMENT_MODE=${DEPLOYMENT_MODE}
EOF
        warn "Please edit ${ENVIRONMENT_FILE} and add your API keys before running the deployment"
    else
        log "Environment file already exists"
    fi
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certs() {
    if [[ "$DEPLOYMENT_MODE" == "development" ]]; then
        log "Generating self-signed SSL certificates for development..."
        
        if [[ ! -f "${PROJECT_ROOT}/ssl/cert.pem" ]]; then
            openssl req -x509 -newkey rsa:4096 -keyout "${PROJECT_ROOT}/ssl/key.pem" \
                -out "${PROJECT_ROOT}/ssl/cert.pem" -days 365 -nodes \
                -subj "/C=US/ST=CA/L=San Francisco/O=LlamaAgent/CN=localhost"
            
            log "SSL certificates generated"
        else
            log "SSL certificates already exist"
        fi
    fi
}

# Create configuration files
create_config_files() {
    log "Creating configuration files..."
    
    # Nginx configuration
    cat > "${PROJECT_ROOT}/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream simon_api {
        server simon-ecosystem:8000;
    }
    
    upstream datasette {
        server simon-datasette:8001;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://simon_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /datasette/ {
            proxy_pass http://datasette/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

    # Prometheus configuration
    cat > "${PROJECT_ROOT}/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'simon-ecosystem'
    static_configs:
      - targets: ['simon-ecosystem:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
      
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # Grafana datasource configuration
    mkdir -p "${PROJECT_ROOT}/grafana/datasources"
    cat > "${PROJECT_ROOT}/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log "Configuration files created"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-openai.txt
    
    # Install Simon's ecosystem
    pip install \
        llm>=0.17.0 \
        llm-anthropic>=0.3.0 \
        llm-openai-plugin>=0.2.0 \
        llm-gemini>=0.2.0 \
        llm-mistral>=0.1.0 \
        sqlite-utils>=3.37.0 \
        datasette>=1.0.0
    
    # Install LlamaAgent in development mode
    pip install -e .
    
    log "Dependencies installed successfully"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build the main Simon ecosystem image
    docker build -f Dockerfile.simon -t simon-llm-ecosystem:latest \
        --target "$DEPLOYMENT_MODE" .
    
    log "Docker images built successfully"
}

# Start services
start_services() {
    log "Starting Simon's LLM Ecosystem services..."
    
    cd "$PROJECT_ROOT"
    
    # Start with Docker Compose
    if [[ "$DEPLOYMENT_MODE" == "production" ]]; then
        docker-compose -f docker-compose.simon.yml up -d
    else
        docker-compose -f docker-compose.simon.yml up -d simon-ecosystem datasette redis jupyter
    fi
    
    log "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log "API service is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "API service failed to start within timeout"
        fi
        sleep 2
    done
    
    # Wait for Datasette to be ready
    for i in {1..15}; do
        if curl -f http://localhost:8002 &> /dev/null; then
            log "Datasette service is ready"
            break
        fi
        if [[ $i -eq 15 ]]; then
            warn "Datasette service may not be ready"
        fi
        sleep 2
    done
}

# Run initial setup
initial_setup() {
    log "Running initial setup..."
    
    # Create initial database tables if needed
    python3 -c "
import asyncio
from src.llamaagent.llm.simon_ecosystem import SimonLLMEcosystem, SimonEcosystemConfig

async def setup():
    config = SimonEcosystemConfig()
    ecosystem = SimonLLMEcosystem(config)
    health = await ecosystem.health_check()
    print(f'Ecosystem health: {health}')

asyncio.run(setup())
" || warn "Initial setup script failed"
    
    log "Initial setup completed"
}

# Run tests
run_tests() {
    if [[ "$DEPLOYMENT_MODE" == "development" ]]; then
        log "Running tests..."
        
        cd "$PROJECT_ROOT"
        source .venv/bin/activate
        
        python -m pytest tests/test_simon_ecosystem_integration.py -v || warn "Some tests failed"
        
        log "Tests completed"
    fi
}

# Display deployment information
show_deployment_info() {
    log "Deployment completed successfully!"
    
    echo ""
    echo -e "${BLUE}=== Simon's LLM Ecosystem - Deployment Information ===${NC}"
    echo ""
    echo -e "${GREEN}Services:${NC}"
    echo "  • API Server:     http://localhost:8000"
    echo "  • API Docs:       http://localhost:8000/docs"
    echo "  • Datasette:      http://localhost:8002"
    echo "  • Health Check:   http://localhost:8000/health"
    
    if [[ "$DEPLOYMENT_MODE" == "development" ]]; then
        echo "  • Jupyter Lab:    http://localhost:8888"
    fi
    
    if docker-compose -f docker-compose.simon.yml ps | grep -q prometheus; then
        echo "  • Prometheus:     http://localhost:9090"
        echo "  • Grafana:        http://localhost:3000 (admin/admin123)"
    fi
    
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo "  • View logs:      docker-compose -f docker-compose.simon.yml logs -f"
    echo "  • Stop services:  docker-compose -f docker-compose.simon.yml down"
    echo "  • Restart:        docker-compose -f docker-compose.simon.yml restart"
    echo "  • Shell access:   docker-compose -f docker-compose.simon.yml exec simon-ecosystem bash"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Edit .env file with your API keys"
    echo "2. Test the API endpoints"
    echo "3. Explore the Jupyter notebooks"
    echo "4. Check the comprehensive documentation"
    echo ""
    
    if [[ ! -f "$ENVIRONMENT_FILE" ]] || grep -q "your_.*_api_key_here" "$ENVIRONMENT_FILE"; then
        warn "Remember to add your API keys to ${ENVIRONMENT_FILE}"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up on exit..."
    # Add any cleanup tasks here
}

# Main deployment function
main() {
    log "Starting Simon's LLM Ecosystem deployment in $DEPLOYMENT_MODE mode..."
    
    trap cleanup EXIT
    
    check_prerequisites
    create_directories
    setup_environment
    generate_ssl_certs
    create_config_files
    install_dependencies
    build_images
    start_services
    wait_for_services
    initial_setup
    run_tests
    show_deployment_info
    
    log "Deployment process completed!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        "development"|"dev")
            DEPLOYMENT_MODE="development"
            main
            ;;
        "production"|"prod")
            DEPLOYMENT_MODE="production"
            main
            ;;
        "stop")
            log "Stopping Simon's LLM Ecosystem..."
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.simon.yml down
            log "Services stopped"
            ;;
        "restart")
            log "Restarting Simon's LLM Ecosystem..."
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.simon.yml restart
            log "Services restarted"
            ;;
        "logs")
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.simon.yml logs -f
            ;;
        "status")
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.simon.yml ps
            ;;
        "help"|"--help"|"-h")
            echo "Simon's LLM Ecosystem Deployment Script"
            echo ""
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  development, dev  Deploy in development mode (default)"
            echo "  production, prod  Deploy in production mode"
            echo "  stop             Stop all services"
            echo "  restart          Restart all services"
            echo "  logs             Show service logs"
            echo "  status           Show service status"
            echo "  help             Show this help message"
            echo ""
            ;;
        *)
            warn "Unknown command: ${1:-}. Using development mode."
            DEPLOYMENT_MODE="development"
            main
            ;;
    esac
fi 