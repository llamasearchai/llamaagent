-- LlamaAgent Database Initialization Script
-- Author: Nik Jois <nikjois@llamasearch.ai>
-- PostgreSQL Database Setup for Production

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE llamaagent'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'llamaagent')\gexec

-- Connect to the database
\c llamaagent;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS llamaagent;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS cache;
CREATE SCHEMA IF NOT EXISTS tasks;
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS conversations;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set default schema
SET search_path TO llamaagent, public;

-- Create custom types
CREATE TYPE user_role AS ENUM ('admin', 'user', 'agent', 'system');
CREATE TYPE agent_status AS ENUM ('active', 'inactive', 'maintenance', 'error');
CREATE TYPE task_status AS ENUM ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled');
CREATE TYPE conversation_status AS ENUM ('active', 'completed', 'archived');
CREATE TYPE provider_type AS ENUM ('openai', 'anthropic', 'together', 'cohere', 'huggingface', 'ollama', 'mock');

-- Users table
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- API Keys table
CREATE TABLE IF NOT EXISTS auth.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    permissions JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sessions table
CREATE TABLE IF NOT EXISTS auth.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL,
    status agent_status DEFAULT 'active',
    configuration JSONB DEFAULT '{}'::jsonb,
    capabilities JSONB DEFAULT '[]'::jsonb,
    provider_configs JSONB DEFAULT '{}'::jsonb,
    owner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations.conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents.agents(id) ON DELETE SET NULL,
    status conversation_status DEFAULT 'active',
    context JSONB DEFAULT '{}'::jsonb,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Messages table
CREATE TABLE IF NOT EXISTS conversations.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations.conversations(id) ON DELETE CASCADE,
    sender_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES agents.agents(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(50) DEFAULT 'text',
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    tokens_used INTEGER DEFAULT 0,
    processing_time INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks.tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL,
    status task_status DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents.agents(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES conversations.conversations(id) ON DELETE SET NULL,
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Knowledge Base table
CREATE TABLE IF NOT EXISTS knowledge.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(100) DEFAULT 'text',
    source_url TEXT,
    source_type VARCHAR(100),
    owner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    tags TEXT[],
    checksum VARCHAR(64),
    file_size INTEGER,
    mime_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Cache table
CREATE TABLE IF NOT EXISTS cache.cache_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- LLM Providers table
CREATE TABLE IF NOT EXISTS agents.llm_providers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    provider_type provider_type NOT NULL,
    configuration JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    health_status VARCHAR(20) DEFAULT 'unknown',
    last_health_check TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Agent Executions table
CREATE TABLE IF NOT EXISTS agents.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks.tasks(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations.conversations(id) ON DELETE SET NULL,
    provider_id UUID REFERENCES agents.llm_providers(id) ON DELETE SET NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_estimate DECIMAL(10, 4) DEFAULT 0,
    execution_time INTERVAL,
    success BOOLEAN DEFAULT false,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Analytics Events table
CREATE TABLE IF NOT EXISTS analytics.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES agents.agents(id) ON DELETE SET NULL,
    session_id UUID,
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- System Metrics table
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15, 4) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    tags JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit Logs table
CREATE TABLE IF NOT EXISTS monitoring.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON auth.users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username ON auth.users(username);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role ON auth.users(role);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON auth.users(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id ON auth.api_keys(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_hash ON auth.api_keys(key_hash);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_active ON auth.api_keys(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_id ON auth.sessions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_token ON auth.sessions(session_token);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_expires_at ON auth.sessions(expires_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_owner_id ON agents.agents(owner_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status ON agents.agents(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_type ON agents.agents(type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_created_at ON agents.agents(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_id ON conversations.conversations(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_agent_id ON conversations.conversations(agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_status ON conversations.conversations(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_created_at ON conversations.conversations(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_id ON conversations.messages(conversation_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_sender_id ON conversations.messages(sender_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_created_at ON conversations.messages(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_role ON conversations.messages(role);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_user_id ON tasks.tasks(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_agent_id ON tasks.tasks(agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status ON tasks.tasks(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_created_at ON tasks.tasks(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_scheduled_at ON tasks.tasks(scheduled_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_owner_id ON knowledge.documents(owner_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_type ON knowledge.documents(document_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created_at ON knowledge.documents(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tags ON knowledge.documents USING GIN(tags);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cache_key ON cache.cache_entries(cache_key);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cache_expires_at ON cache.cache_entries(expires_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_executions_agent_id ON agents.executions(agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_executions_task_id ON agents.executions(task_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_executions_created_at ON agents.executions(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type ON analytics.events(event_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_user_id ON analytics.events(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_created_at ON analytics.events(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_name ON monitoring.system_metrics(metric_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_created_at ON monitoring.system_metrics(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_user_id ON monitoring.audit_logs(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_action ON monitoring.audit_logs(action);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_created_at ON monitoring.audit_logs(created_at);

-- Full text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_fts ON knowledge.documents USING GIN(to_tsvector('english', content));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_title_fts ON conversations.conversations USING GIN(to_tsvector('english', title));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_content_fts ON conversations.messages USING GIN(to_tsvector('english', content));

-- Create functions for automatic updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON auth.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON auth.api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON auth.sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents.agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations.conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks.tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON knowledge.documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_llm_providers_updated_at BEFORE UPDATE ON agents.llm_providers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create default admin user
INSERT INTO auth.users (id, username, email, password_hash, full_name, role, is_active, is_verified)
VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@llamaagent.local',
    crypt('admin123', gen_salt('bf')),
    'System Administrator',
    'admin',
    true,
    true
) ON CONFLICT (username) DO NOTHING;

-- Create default agent
INSERT INTO agents.agents (id, name, description, type, status, configuration, capabilities)
VALUES (
    uuid_generate_v4(),
    'Default Assistant',
    'Default LlamaAgent assistant with general capabilities',
    'react',
    'active',
    '{"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 2048}',
    '["text_generation", "code_generation", "question_answering", "summarization"]'
) ON CONFLICT DO NOTHING;

-- Create default LLM providers
INSERT INTO agents.llm_providers (name, provider_type, configuration, is_active)
VALUES 
    ('OpenAI GPT-4', 'openai', '{"model": "gpt-4", "api_base": "https://api.openai.com/v1"}', true),
    ('OpenAI GPT-3.5', 'openai', '{"model": "gpt-3.5-turbo", "api_base": "https://api.openai.com/v1"}', true),
    ('Anthropic Claude', 'anthropic', '{"model": "claude-3-sonnet-20240229", "api_base": "https://api.anthropic.com"}', true),
    ('Together AI', 'together', '{"model": "meta-llama/Llama-2-70b-chat-hf", "api_base": "https://api.together.xyz"}', true),
    ('Mock Provider', 'mock', '{"responses": ["I am a mock response for testing purposes."]}', true)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA llamaagent TO llamaagent;
GRANT USAGE ON SCHEMA auth TO llamaagent;
GRANT USAGE ON SCHEMA monitoring TO llamaagent;
GRANT USAGE ON SCHEMA cache TO llamaagent;
GRANT USAGE ON SCHEMA tasks TO llamaagent;
GRANT USAGE ON SCHEMA agents TO llamaagent;
GRANT USAGE ON SCHEMA conversations TO llamaagent;
GRANT USAGE ON SCHEMA knowledge TO llamaagent;
GRANT USAGE ON SCHEMA analytics TO llamaagent;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA llamaagent TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cache TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tasks TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agents TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA conversations TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA knowledge TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO llamaagent;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA llamaagent TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA auth TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cache TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tasks TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agents TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA conversations TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA knowledge TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO llamaagent;

-- Create views for easier querying
CREATE OR REPLACE VIEW llamaagent.active_conversations AS
SELECT 
    c.id,
    c.title,
    c.created_at,
    c.updated_at,
    u.username,
    u.full_name,
    a.name as agent_name,
    COUNT(m.id) as message_count
FROM conversations.conversations c
LEFT JOIN auth.users u ON c.user_id = u.id
LEFT JOIN agents.agents a ON c.agent_id = a.id
LEFT JOIN conversations.messages m ON c.id = m.conversation_id
WHERE c.status = 'active'
GROUP BY c.id, c.title, c.created_at, c.updated_at, u.username, u.full_name, a.name;

CREATE OR REPLACE VIEW llamaagent.recent_tasks AS
SELECT 
    t.id,
    t.title,
    t.type,
    t.status,
    t.created_at,
    t.updated_at,
    u.username,
    a.name as agent_name
FROM tasks.tasks t
LEFT JOIN auth.users u ON t.user_id = u.id
LEFT JOIN agents.agents a ON t.agent_id = a.id
ORDER BY t.created_at DESC;

-- Final database statistics
SELECT 
    'Database initialization completed successfully' as status,
    NOW() as completed_at,
    COUNT(*) as total_tables
FROM information_schema.tables 
WHERE table_schema IN ('llamaagent', 'auth', 'monitoring', 'cache', 'tasks', 'agents', 'conversations', 'knowledge', 'analytics'); 