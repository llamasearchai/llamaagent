#!/bin/bash
# =============================================================================
# LlamaAgent Docker Entrypoint Script
# 
# Handles:
# - Environment setup
# - Database migrations
# - Service initialization
# - Health checks
# - Graceful shutdown
#
# Author: Nik Jois <nikjois@llamasearch.ai>
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STARTUP_TIMEOUT=300
HEALTH_CHECK_INTERVAL=10
MAX_STARTUP_RETRIES=30

# Logging functions
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

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    # Kill background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Remove temporary files
    rm -f /tmp/llamaagent_*.tmp 2>/dev/null || true
    
    # Flush logs
    sync
    
    log_success "Cleanup completed"
}

# Trap signals for graceful shutdown
trap cleanup EXIT INT TERM

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "LLAMAAGENT_ENV"
        "LLAMAAGENT_PORT"
        "LLAMAAGENT_HOST"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate numeric values
    if ! [[ "$LLAMAAGENT_PORT" =~ ^[0-9]+$ ]] || [ "$LLAMAAGENT_PORT" -lt 1 ] || [ "$LLAMAAGENT_PORT" -gt 65535 ]; then
        log_error "Invalid port number: $LLAMAAGENT_PORT"
        exit 1
    fi
    
    # Check for API keys
    local api_keys_present=false
    local api_key_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "TOGETHER_API_KEY" "COHERE_API_KEY" "HUGGINGFACE_API_KEY")
    
    for var in "${api_key_vars[@]}"; do
        if [ -n "${!var}" ] && [ "${!var}" != "sk-placeholder" ] && [ "${!var}" != "your-api-key" ]; then
            api_keys_present=true
            break
        fi
    done
    
    if [ "$api_keys_present" = false ]; then
        log_warning "No valid LLM provider API keys found. The system will use mock responses."
    fi
    
    log_success "Environment validation completed"
}

# Wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log_info "Waiting for $service_name to be ready at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        if [ $i -eq $timeout ]; then
            log_error "$service_name failed to start within $timeout seconds"
            return 1
        fi
        
        log_info "Waiting for $service_name... ($i/$timeout)"
        sleep 1
    done
}

# Check database connection
check_database() {
    log_info "Checking database connection..."
    
    if [ -n "$DATABASE_URL" ]; then
        python -c "
import psycopg2
import os
import sys
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"
        log_success "Database connection verified"
    else
        log_warning "No DATABASE_URL provided, skipping database check"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    if [ -f "alembic.ini" ]; then
        alembic upgrade head
        log_success "Database migrations completed"
    else
        log_warning "No alembic.ini found, skipping migrations"
    fi
}

# Initialize application data
initialize_app() {
    log_info "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/{data,logs,cache,checkpoints,uploads}
    
    # Set proper permissions
    chmod 755 /app/{data,logs,cache,checkpoints,uploads}
    
    # Initialize configuration if not exists
    if [ ! -f "/app/config/app.yaml" ]; then
        log_info "Creating default configuration..."
        python -c "
import yaml
import os

config = {
    'app': {
        'name': 'LlamaAgent',
        'version': '1.0.0',
        'environment': os.environ.get('LLAMAAGENT_ENV', 'development'),
        'debug': os.environ.get('LLAMAAGENT_ENV', 'development') == 'development'
    },
    'database': {
        'url': os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    },
    'redis': {
        'url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    },
    'logging': {
        'level': os.environ.get('LOG_LEVEL', 'INFO'),
        'file': '/app/logs/app.log'
    }
}

os.makedirs('/app/config', exist_ok=True)
with open('/app/config/app.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        log_success "Default configuration created"
    fi
    
    log_success "Application initialization completed"
}

# Health check function
health_check() {
    log_info "Performing health check..."
    
    # Check if main application is responding
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Application health check passed"
        return 0
    else
        log_error "Application health check failed"
        return 1
    fi
}

# Graceful shutdown handler
graceful_shutdown() {
    log_info "Received shutdown signal, performing graceful shutdown..."
    
    # Stop background processes
    if [ -n "$FASTAPI_PID" ]; then
        log_info "Stopping FastAPI server..."
        kill -TERM $FASTAPI_PID
        wait $FASTAPI_PID
    fi
    
    # Cleanup
    log_info "Cleaning up..."
    
    log_success "Graceful shutdown completed"
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting LlamaAgent container..."
    log_info "Environment: ${LLAMAAGENT_ENV:-development}"
    
    # Wait for dependencies
    if [ -n "$DATABASE_URL" ] && [[ "$DATABASE_URL" == postgresql* ]]; then
        DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
        wait_for_service "$DB_HOST" 5432 "PostgreSQL"
    fi
    
    if [ -n "$REDIS_URL" ]; then
        REDIS_HOST=$(echo $REDIS_URL | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
        wait_for_service "$REDIS_HOST" 6379 "Redis"
    fi
    
    # Initialize application
    initialize_app
    
    # Check database connection
    check_database
    
    # Run migrations
    run_migrations
    
    # Start the application based on the command
    case "$1" in
        "fastapi"|"api"|"server")
            log_info "Starting FastAPI server..."
            exec uvicorn src.llamaagent.api.main:app \
                --host 0.0.0.0 \
                --port 8000 \
                --workers ${WORKERS:-4} \
                --log-level ${LOG_LEVEL:-info} \
                --access-log \
                --use-colors
            ;;
        "worker")
            log_info "Starting worker process..."
            exec python -m src.llamaagent.worker
            ;;
        "scheduler")
            log_info "Starting scheduler..."
            exec python -m src.llamaagent.scheduler
            ;;
        "cli")
            log_info "Starting CLI mode..."
            exec python -m src.llamaagent.cli "${@:2}"
            ;;
        "test")
            log_info "Running tests..."
            exec pytest tests/ -v "${@:2}"
            ;;
        "shell")
            log_info "Starting interactive shell..."
            exec python -i -c "
import sys
sys.path.insert(0, '/app/src')
from llamaagent import *
print('LlamaAgent shell ready. All modules imported.')
"
            ;;
        "supervisord")
            log_info "Starting supervisord..."
            exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
            ;;
        *)
            if [ $# -eq 0 ]; then
                log_info "No command specified, starting FastAPI server..."
                exec uvicorn src.llamaagent.api.main:app \
                    --host 0.0.0.0 \
                    --port 8000 \
                    --workers ${WORKERS:-4} \
                    --log-level ${LOG_LEVEL:-info}
            else
                log_info "Executing custom command: $*"
                exec "$@"
            fi
            ;;
    esac
}

# Execute main function
main "$@" 