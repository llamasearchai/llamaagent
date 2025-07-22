#!/bin/bash
# LlamaAgent Advanced Deployment Script
# Automated deployment with cutting-edge features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/cutting_edge_config.yaml"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.advanced.yml"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"
ENABLE_GPU="${ENABLE_GPU:-false}"
SCALE_REPLICAS="${SCALE_REPLICAS:-3}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        log_warning "Running as root. Consider using a non-root user for better security."
    fi
    
    log_success "Prerequisites check completed"
}

setup_environment() {
    log_info "Setting up environment for ${ENVIRONMENT}..."
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/cache"
    
    # Copy configuration templates
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log_info "Creating .env file from template..."
        cat > "${PROJECT_ROOT}/.env" << EOF
# LlamaAgent Advanced Environment Configuration
ENVIRONMENT=${ENVIRONMENT}

# API Keys (set these with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=llamaagent
DB_USER=llamaagent
DB_PASSWORD=$(openssl rand -base64 32)

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Advanced Features
ENABLE_MULTIMODAL=true
ENABLE_REASONING=true
ENABLE_VISION=true
ENABLE_ORCHESTRATION=true

# Performance
WORKERS=4
MAX_PARALLEL_AGENTS=5
BUDGET_LIMIT=100.0
EOF
        log_success ".env file created. Please update with your API keys."
    fi
    
    log_success "Environment setup completed"
}

pull_cutting_edge_models() {
    log_info "Pulling cutting-edge models..."
    
    # Check if Ollama is available
    if command -v ollama &> /dev/null; then
        log_info "Pulling local models with Ollama..."
        
        # Pull cutting-edge models
        models=(
            "llama3.2-vision:11b"
            "qwen2-vl:7b"
            "deepseek-r1"
            "mistral-small:3.2"
            "codellama:13b"
        )
        
        for model in "${models[@]}"; do
            log_info "Pulling ${model}..."
            if ollama pull "$model"; then
                log_success "Successfully pulled ${model}"
            else
                log_warning "Failed to pull ${model}, continuing..."
            fi
        done
    else
        log_warning "Ollama not found. Skipping local model setup."
        log_info "Install Ollama from https://ollama.ai for local model support"
    fi
}

build_docker_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build base image
    docker build \
        -f docker/Dockerfile.advanced \
        --target production \
        -t llamaagent:advanced \
        .
    
    # Build development image if in development environment
    if [ "$ENVIRONMENT" = "development" ]; then
        docker build \
            -f docker/Dockerfile.advanced \
            --target development \
            -t llamaagent:advanced-dev \
            .
    fi
    
    log_success "Docker images built successfully"
}

create_docker_compose() {
    log_info "Creating Docker Compose configuration..."
    
    cat > "$DOCKER_COMPOSE_FILE" << EOF
version: '3.8'

services:
  llamaagent-advanced:
    image: llamaagent:advanced
    container_name: llamaagent-advanced
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: ${SCALE_REPLICAS}
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  postgres:
    image: postgres:15-alpine
    container_name: llamaagent-postgres
    environment:
      POSTGRES_DB: llamaagent
      POSTGRES_USER: llamaagent
      POSTGRES_PASSWORD: \${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: llamaagent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: llamaagent-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'

  prometheus:
    image: prom/prometheus:latest
    container_name: llamaagent-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: llamaagent-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  ollama_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: llamaagent-network
EOF
    
    log_success "Docker Compose configuration created"
}

deploy_with_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "LlamaAgent Advanced is running successfully!"
        log_info "API available at: http://localhost:8000"
        log_info "Documentation at: http://localhost:8000/docs"
        log_info "Monitoring at: http://localhost:3000 (Grafana)"
    else
        log_error "Health check failed. Check logs with: docker-compose logs"
        exit 1
    fi
}

run_health_checks() {
    log_info "Running comprehensive health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success " API health check passed"
    else
        log_error " API health check failed"
        return 1
    fi
    
    # Check database connection
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success " Database health check passed"
    else
        log_error " Database health check failed"
        return 1
    fi
    
    # Check cutting-edge endpoints
    endpoints=(
        "/models/advanced"
        "/providers"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f "http://localhost:8000${endpoint}" > /dev/null 2>&1; then
            log_success " Endpoint ${endpoint} is accessible"
        else
            log_warning " Endpoint ${endpoint} may not be ready yet"
        fi
    done
    
    log_success "Health checks completed"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Set environment (development|staging|production)"
    echo "  -t, --type          Deployment type (docker|kubernetes|local)"
    echo "  -g, --gpu           Enable GPU support (true|false)"
    echo "  -r, --replicas      Number of replicas to deploy"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --environment production --replicas 5"
    echo "  $0 --environment development --type docker"
    echo "  $0 --gpu true --replicas 3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -g|--gpu)
            ENABLE_GPU="$2"
            shift 2
            ;;
        -r|--replicas)
            SCALE_REPLICAS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main deployment flow
main() {
    log_info "Starting LlamaAgent Advanced deployment..."
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Deployment Type: ${DEPLOYMENT_TYPE}"
    log_info "GPU Support: ${ENABLE_GPU}"
    log_info "Replicas: ${SCALE_REPLICAS}"
    
    check_prerequisites
    setup_environment
    
    case $DEPLOYMENT_TYPE in
        docker)
            build_docker_images
            create_docker_compose
            deploy_with_docker
            ;;
        local)
            log_info "Starting local development server..."
            cd "$PROJECT_ROOT"
            python -m uvicorn src.llamaagent.api.main:app --host 0.0.0.0 --port 8000 --reload
            ;;
        *)
            log_error "Unsupported deployment type: ${DEPLOYMENT_TYPE}"
            exit 1
            ;;
    esac
    
    # Run health checks
    sleep 10
    run_health_checks
    
    log_success "LAUNCH: LlamaAgent Advanced deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Update API keys in .env file"
    echo "2. Visit http://localhost:8000/docs for API documentation"
    echo "3. Try the cutting-edge endpoints:"
    echo "   - POST /multimodal/analyze"
    echo "   - POST /reasoning/advanced"
    echo "   - POST /litellm/universal"
    echo "   - POST /vision/analyze"
    echo "4. Monitor with Grafana at http://localhost:3000"
}

# Run main function
main "$@" 