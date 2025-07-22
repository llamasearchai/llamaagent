#!/bin/bash
# Production deployment script for LlamaAgent
# Author: Nik Jois <nikjois@llamasearch.ai>
#
# This script handles the complete deployment process for LlamaAgent
# including Docker build, database migrations, and service orchestration.

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly ENV_FILE="${PROJECT_ROOT}/.env"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.production.yml"
readonly K8S_DIR="${PROJECT_ROOT}/k8s"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
SKIP_BUILD="${SKIP_BUILD:-false}"
SKIP_MIGRATIONS="${SKIP_MIGRATIONS:-false}"
SKIP_HEALTH_CHECK="${SKIP_HEALTH_CHECK:-false}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-docker}" # docker or k8s

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" >&2
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        # Add any cleanup logic here
    fi
    exit $exit_code
}

trap cleanup EXIT

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy LlamaAgent to production environment.

OPTIONS:
    -e, --environment ENV       Deployment environment (default: production)
    -m, --mode MODE            Deployment mode: docker or k8s (default: docker)
    --skip-build               Skip Docker image building
    --skip-migrations          Skip database migrations
    --skip-health-check        Skip post-deployment health checks
    --no-backup                Skip pre-deployment backup
    --debug                    Enable debug output
    -h, --help                 Show this help message

EXAMPLES:
    # Standard production deployment
    $0

    # Deploy to staging with debug output
    $0 --environment staging --debug

    # Quick deployment without backup
    $0 --no-backup --skip-health-check

    # Kubernetes deployment
    $0 --mode k8s

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --skip-migrations)
                SKIP_MIGRATIONS="true"
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK="true"
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY="false"
                shift
                ;;
            --debug)
                DEBUG="1"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment..."

    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root is not recommended for security reasons"
    fi

    # Check required commands
    local required_commands=("docker" "docker-compose")
    if [[ "$DEPLOYMENT_MODE" == "k8s" ]]; then
        required_commands+=("kubectl" "helm")
    fi

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command '$cmd' not found"
        fi
    done

    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warn "Environment file not found at $ENV_FILE"
        log_info "Creating from template..."
        cp "${PROJECT_ROOT}/.env.production" "$ENV_FILE"
        log_warn "Please edit $ENV_FILE with your configuration before continuing"
        exit 1
    fi

    # Source environment variables
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a

    # Validate critical environment variables
    local required_vars=("SECRET_KEY" "DATABASE_URL" "REDIS_URL")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error_exit "Required environment variable '$var' is not set"
        fi
    done

    # Check if using default passwords (security warning)
    if [[ "${POSTGRES_PASSWORD:-}" == "secure_password_change_this" ]]; then
        error_exit "Please change the default PostgreSQL password in $ENV_FILE"
    fi

    log_info "Environment validation completed"
}

# Pre-deployment backup
create_backup() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "false" ]]; then
        log_info "Skipping backup (--no-backup specified)"
        return 0
    fi

    log_info "Creating pre-deployment backup..."

    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Database backup
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log_info "Backing up PostgreSQL database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres \
            pg_dump -U "${POSTGRES_USER}" "${POSTGRES_DB}" > "$backup_dir/database.sql"
    fi

    # Application data backup
    if [[ -d "${PROJECT_ROOT}/data" ]]; then
        log_info "Backing up application data..."
        tar -czf "$backup_dir/app_data.tar.gz" -C "${PROJECT_ROOT}" data/
    fi

    # Configuration backup
    log_info "Backing up configuration..."
    cp "$ENV_FILE" "$backup_dir/environment.env"

    log_info "Backup created at: $backup_dir"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping image build (--skip-build specified)"
        return 0
    fi

    log_info "Building Docker images..."

    # Build production image
    docker build \
        -f Dockerfile.production \
        --target runtime \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$(git describe --tags --always)" \
        -t llamaagent:latest \
        -t llamaagent:"$(git describe --tags --always)" \
        "$PROJECT_ROOT"

    log_info "Docker images built successfully"
}

# Database migrations
run_migrations() {
    if [[ "$SKIP_MIGRATIONS" == "true" ]]; then
        log_info "Skipping database migrations (--skip-migrations specified)"
        return 0
    fi

    log_info "Running database migrations..."

    # Start database if not running
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis

    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres \
           pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" &>/dev/null; then
            break
        fi

        log_debug "Database not ready, attempt $attempt/$max_attempts"
        sleep 2
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        error_exit "Database failed to become ready within timeout"
    fi

    # Run migrations
    docker-compose -f "$COMPOSE_FILE" run --rm \
        -e SKIP_MIGRATIONS=false \
        api python -m llamaagent.database.migrate

    log_info "Database migrations completed"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."

    # Pull external images
    docker-compose -f "$COMPOSE_FILE" pull

    # Deploy services
    docker-compose -f "$COMPOSE_FILE" up -d

    log_info "Docker deployment completed"
}

# Deploy with Kubernetes
deploy_k8s() {
    log_info "Deploying with Kubernetes..."

    # Check if kubectl is configured
    if ! kubectl cluster-info &>/dev/null; then
        error_exit "kubectl is not configured or cluster is not accessible"
    fi

    # Apply base resources
    kubectl apply -f "$K8S_DIR/base/"

    # Apply environment-specific overlays
    if [[ -d "$K8S_DIR/overlays/$ENVIRONMENT" ]]; then
        kubectl apply -k "$K8S_DIR/overlays/$ENVIRONMENT"
    fi

    # Wait for deployment
    kubectl rollout status deployment/llamaagent-api -n llamaagent --timeout=300s

    log_info "Kubernetes deployment completed"
}

# Health checks
run_health_checks() {
    if [[ "$SKIP_HEALTH_CHECK" == "true" ]]; then
        log_info "Skipping health checks (--skip-health-check specified)"
        return 0
    fi

    log_info "Running post-deployment health checks..."

    local api_url
    if [[ "$DEPLOYMENT_MODE" == "k8s" ]]; then
        api_url="http://$(kubectl get service llamaagent-api -n llamaagent -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000"
    else
        api_url="http://localhost:8000"
    fi

    # Wait for API to be ready
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$api_url/health" >/dev/null 2>&1; then
            break
        fi

        log_debug "API not ready, attempt $attempt/$max_attempts"
        sleep 5
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        error_exit "API failed health check within timeout"
    fi

    # Run comprehensive health checks
    local health_response
    health_response=$(curl -s "$api_url/health")

    if [[ "$health_response" == *"healthy"* ]]; then
        log_info "PASS API health check passed"
    else
        error_exit "FAIL API health check failed: $health_response"
    fi

    # Check database connectivity
    if curl -f -s "$api_url/health/database" >/dev/null 2>&1; then
        log_info "PASS Database connectivity check passed"
    else
        log_warn "WARNING:  Database connectivity check failed"
    fi

    # Check cache connectivity
    if curl -f -s "$api_url/health/cache" >/dev/null 2>&1; then
        log_info "PASS Cache connectivity check passed"
    else
        log_warn "WARNING:  Cache connectivity check failed"
    fi

    log_info "Health checks completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."

    local report_file="${PROJECT_ROOT}/deployment_report_$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "environment": "$ENVIRONMENT",
    "mode": "$DEPLOYMENT_MODE",
    "version": "$(git describe --tags --always)",
    "commit": "$(git rev-parse HEAD)",
    "branch": "$(git rev-parse --abbrev-ref HEAD)"
  },
  "configuration": {
    "skip_build": $SKIP_BUILD,
    "skip_migrations": $SKIP_MIGRATIONS,
    "skip_health_check": $SKIP_HEALTH_CHECK,
    "backup_created": $BACKUP_BEFORE_DEPLOY
  },
  "services": {
    "api": "deployed",
    "worker": "deployed",
    "database": "deployed",
    "cache": "deployed",
    "monitoring": "deployed"
  }
}
EOF

    log_info "Deployment report saved to: $report_file"
}

# Main deployment function
main() {
    log_info "Starting LlamaAgent deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Mode: $DEPLOYMENT_MODE"

    # Validate environment
    validate_environment

    # Create backup
    create_backup

    # Build images
    build_images

    # Run migrations
    run_migrations

    # Deploy based on mode
    case "$DEPLOYMENT_MODE" in
        docker)
            deploy_docker
            ;;
        k8s)
            deploy_k8s
            ;;
        *)
            error_exit "Unknown deployment mode: $DEPLOYMENT_MODE"
            ;;
    esac

    # Health checks
    run_health_checks

    # Generate report
    generate_report

    log_info "SUCCESS: LlamaAgent deployment completed successfully!"
    log_info "API should be available at: http://localhost:8000"
    log_info "Monitoring: http://localhost:3000 (Grafana)"
    log_info "Metrics: http://localhost:9090 (Prometheus)"
}

# Parse arguments and run main function
parse_args "$@"
main