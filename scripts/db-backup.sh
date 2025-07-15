#!/bin/bash
# Database Backup Script for LlamaAgent
# Author: Nik Jois <nikjois@llamasearch.ai>

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
POSTGRES_DB="${POSTGRES_DB:-llamaagent}"
POSTGRES_USER="${POSTGRES_USER:-llamaagent}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-llamaagent}"
PGHOST="${PGHOST:-postgres}"
PGPORT="${PGPORT:-5432}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION="${COMPRESSION:-gzip}"
NOTIFICATION_URL="${NOTIFICATION_URL:-}"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/llamaagent_backup_$TIMESTAMP.sql"
BACKUP_FILE_COMPRESSED="$BACKUP_FILE.gz"
LOG_FILE="$BACKUP_DIR/backup.log"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Cleanup function
cleanup() {
    if [[ -f "$BACKUP_FILE" ]]; then
        rm -f "$BACKUP_FILE"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Send notification
send_notification() {
    local message="$1"
    local status="$2"
    
    if [[ -n "$NOTIFICATION_URL" ]]; then
        curl -X POST "$NOTIFICATION_URL" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"$message\", \"status\": \"$status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
            --max-time 10 --silent || true
    fi
}

# Create backup directory
create_backup_dir() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        mkdir -p "$BACKUP_DIR"
        log "Created backup directory: $BACKUP_DIR"
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    if ! pg_isready -h "$PGHOST" -p "$PGPORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t 30; then
        log_error "Database is not accessible"
        send_notification "Database backup failed: Database not accessible" "error"
        exit 1
    fi
    
    log "Database connectivity confirmed"
}

# Perform backup
perform_backup() {
    log "Starting database backup..."
    log "Database: $POSTGRES_DB"
    log "Host: $PGHOST:$PGPORT"
    log "User: $POSTGRES_USER"
    log "Backup file: $BACKUP_FILE_COMPRESSED"
    
    # Set password for pg_dump
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Perform backup with pg_dump
    if pg_dump -h "$PGHOST" -p "$PGPORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        --verbose \
        --no-password \
        --format=custom \
        --compress=9 \
        --no-owner \
        --no-privileges \
        --create \
        --clean \
        --if-exists > "$BACKUP_FILE"; then
        
        log "Database dump completed successfully"
    else
        log_error "Database dump failed"
        send_notification "Database backup failed: pg_dump error" "error"
        exit 1
    fi
    
    # Compress backup if not already compressed
    if [[ "$COMPRESSION" == "gzip" ]]; then
        log "Compressing backup file..."
        if gzip "$BACKUP_FILE"; then
            log "Backup compressed successfully"
        else
            log_error "Backup compression failed"
            send_notification "Database backup failed: Compression error" "error"
            exit 1
        fi
    fi
    
    # Verify backup file
    if [[ ! -f "$BACKUP_FILE_COMPRESSED" ]]; then
        log_error "Backup file not found after compression"
        send_notification "Database backup failed: Backup file missing" "error"
        exit 1
    fi
    
    local backup_size=$(du -h "$BACKUP_FILE_COMPRESSED" | cut -f1)
    log "Backup completed successfully"
    log "Backup size: $backup_size"
    log "Backup location: $BACKUP_FILE_COMPRESSED"
}

# Test backup integrity
test_backup() {
    log "Testing backup integrity..."
    
    if [[ "$COMPRESSION" == "gzip" ]]; then
        if gzip -t "$BACKUP_FILE_COMPRESSED"; then
            log "Backup integrity test passed"
        else
            log_error "Backup integrity test failed"
            send_notification "Database backup warning: Integrity test failed" "warning"
        fi
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
    
    local deleted_count=0
    
    # Find and delete old backup files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "llamaagent_backup_*.sql*" -type f -mtime +$RETENTION_DAYS -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "Cleaned up $deleted_count old backup files"
    else
        log "No old backup files to clean up"
    fi
}

# Generate backup report
generate_report() {
    local backup_size=$(du -h "$BACKUP_FILE_COMPRESSED" | cut -f1)
    local disk_usage=$(df -h "$BACKUP_DIR" | tail -1 | awk '{print $5}')
    local backup_count=$(find "$BACKUP_DIR" -name "llamaagent_backup_*.sql*" -type f | wc -l)
    
    local report="Database Backup Report
========================
Timestamp: $(date)
Database: $POSTGRES_DB
Host: $PGHOST:$PGPORT
Status: SUCCESS
Backup File: $BACKUP_FILE_COMPRESSED
Backup Size: $backup_size
Disk Usage: $disk_usage
Total Backups: $backup_count
Retention Days: $RETENTION_DAYS
"
    
    echo "$report" | tee -a "$LOG_FILE"
    
    # Send success notification
    send_notification "Database backup completed successfully. Size: $backup_size" "success"
}

# Handle errors
handle_error() {
    local exit_code=$?
    log_error "Backup process failed with exit code: $exit_code"
    send_notification "Database backup failed with exit code: $exit_code" "error"
    exit $exit_code
}

# Set error trap
trap handle_error ERR

# Main backup process
main() {
    log "Starting LlamaAgent database backup process"
    
    # Pre-backup checks
    create_backup_dir
    check_database
    
    # Perform backup
    perform_backup
    test_backup
    
    # Post-backup tasks
    cleanup_old_backups
    generate_report
    
    log "Database backup process completed successfully"
}

# Execute main function
main "$@" 