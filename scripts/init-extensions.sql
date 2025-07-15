-- PostgreSQL Extensions Initialization Script
-- Author: Nik Jois <nikjois@llamasearch.ai>
-- Setup vector database extensions for LlamaAgent

-- Connect to the database
\c llamaagent;

-- Create vector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create additional useful extensions
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_buffercache";
CREATE EXTENSION IF NOT EXISTS "pgstattuple";
CREATE EXTENSION IF NOT EXISTS "pg_visibility";

-- Create vector-specific tables and functions
CREATE SCHEMA IF NOT EXISTS vectors;
SET search_path TO vectors, public;

-- Document embeddings table
CREATE TABLE IF NOT EXISTS vectors.document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI embeddings dimension
    model_name VARCHAR(100) NOT NULL DEFAULT 'text-embedding-ada-002',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Conversation embeddings table
CREATE TABLE IF NOT EXISTS vectors.conversation_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations.conversations(id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES conversations.messages(id) ON DELETE CASCADE,
    embedding vector(1536),
    model_name VARCHAR(100) NOT NULL DEFAULT 'text-embedding-ada-002',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Agent memory embeddings table
CREATE TABLE IF NOT EXISTS vectors.agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents.agents(id) ON DELETE CASCADE,
    memory_text TEXT NOT NULL,
    embedding vector(1536),
    importance_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Code embeddings table for code search
CREATE TABLE IF NOT EXISTS vectors.code_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path TEXT NOT NULL,
    function_name TEXT,
    code_block TEXT NOT NULL,
    embedding vector(1536),
    language VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for vector operations
CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector ON vectors.document_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_conversation_embeddings_vector ON vectors.conversation_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_agent_memory_vector ON vectors.agent_memory 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_code_embeddings_vector ON vectors.code_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create additional indexes for performance
CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id ON vectors.document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_conversation_embeddings_conversation_id ON vectors.conversation_embeddings(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id ON vectors.agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_language ON vectors.code_embeddings(language);

-- Create functions for vector operations
CREATE OR REPLACE FUNCTION vectors.similarity_search_documents(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.78,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_text TEXT,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        de.id,
        de.document_id,
        de.chunk_text,
        1 - (de.embedding <=> query_embedding) as similarity,
        de.metadata
    FROM vectors.document_embeddings de
    WHERE 1 - (de.embedding <=> query_embedding) > match_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION vectors.similarity_search_conversations(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.78,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    conversation_id UUID,
    message_id UUID,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.id,
        ce.conversation_id,
        ce.message_id,
        1 - (ce.embedding <=> query_embedding) as similarity,
        ce.metadata
    FROM vectors.conversation_embeddings ce
    WHERE 1 - (ce.embedding <=> query_embedding) > match_threshold
    ORDER BY ce.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION vectors.similarity_search_agent_memory(
    agent_id UUID,
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.78,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    memory_text TEXT,
    similarity FLOAT,
    importance_score FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        am.id,
        am.memory_text,
        1 - (am.embedding <=> query_embedding) as similarity,
        am.importance_score,
        am.metadata
    FROM vectors.agent_memory am
    WHERE am.agent_id = similarity_search_agent_memory.agent_id
    AND 1 - (am.embedding <=> query_embedding) > match_threshold
    ORDER BY am.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION vectors.similarity_search_code(
    query_embedding vector(1536),
    language_filter VARCHAR(50) DEFAULT NULL,
    match_threshold float DEFAULT 0.78,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    file_path TEXT,
    function_name TEXT,
    code_block TEXT,
    similarity FLOAT,
    language VARCHAR(50),
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.id,
        ce.file_path,
        ce.function_name,
        ce.code_block,
        1 - (ce.embedding <=> query_embedding) as similarity,
        ce.language,
        ce.metadata
    FROM vectors.code_embeddings ce
    WHERE (language_filter IS NULL OR ce.language = language_filter)
    AND 1 - (ce.embedding <=> query_embedding) > match_threshold
    ORDER BY ce.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create trigger function for updating embeddings
CREATE OR REPLACE FUNCTION vectors.update_embedding_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_document_embeddings_timestamp
    BEFORE UPDATE ON vectors.document_embeddings
    FOR EACH ROW EXECUTE FUNCTION vectors.update_embedding_timestamp();

-- Create function to clean up old embeddings
CREATE OR REPLACE FUNCTION vectors.cleanup_old_embeddings()
RETURNS void AS $$
BEGIN
    -- Delete document embeddings for deleted documents
    DELETE FROM vectors.document_embeddings 
    WHERE document_id NOT IN (SELECT id FROM knowledge.documents);
    
    -- Delete conversation embeddings for deleted conversations/messages
    DELETE FROM vectors.conversation_embeddings 
    WHERE conversation_id NOT IN (SELECT id FROM conversations.conversations)
    OR message_id NOT IN (SELECT id FROM conversations.messages);
    
    -- Delete agent memory for deleted agents
    DELETE FROM vectors.agent_memory 
    WHERE agent_id NOT IN (SELECT id FROM agents.agents);
    
    -- Update access counts for agent memory
    UPDATE vectors.agent_memory 
    SET access_count = access_count + 1, 
        last_accessed = NOW()
    WHERE last_accessed < NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate embedding statistics
CREATE OR REPLACE FUNCTION vectors.get_embedding_statistics()
RETURNS TABLE (
    table_name TEXT,
    embedding_count BIGINT,
    avg_dimension INTEGER,
    last_updated TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'document_embeddings'::TEXT,
        COUNT(*)::BIGINT,
        AVG(array_length(embedding::real[], 1))::INTEGER,
        MAX(created_at)
    FROM vectors.document_embeddings
    
    UNION ALL
    
    SELECT 
        'conversation_embeddings'::TEXT,
        COUNT(*)::BIGINT,
        AVG(array_length(embedding::real[], 1))::INTEGER,
        MAX(created_at)
    FROM vectors.conversation_embeddings
    
    UNION ALL
    
    SELECT 
        'agent_memory'::TEXT,
        COUNT(*)::BIGINT,
        AVG(array_length(embedding::real[], 1))::INTEGER,
        MAX(created_at)
    FROM vectors.agent_memory
    
    UNION ALL
    
    SELECT 
        'code_embeddings'::TEXT,
        COUNT(*)::BIGINT,
        AVG(array_length(embedding::real[], 1))::INTEGER,
        MAX(created_at)
    FROM vectors.code_embeddings;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for embedding analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS vectors.embedding_analytics AS
SELECT 
    'document_embeddings' as table_name,
    COUNT(*) as total_embeddings,
    AVG(array_length(embedding::real[], 1)) as avg_dimension,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created,
    COUNT(DISTINCT document_id) as unique_documents
FROM vectors.document_embeddings

UNION ALL

SELECT 
    'conversation_embeddings' as table_name,
    COUNT(*) as total_embeddings,
    AVG(array_length(embedding::real[], 1)) as avg_dimension,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created,
    COUNT(DISTINCT conversation_id) as unique_conversations
FROM vectors.conversation_embeddings

UNION ALL

SELECT 
    'agent_memory' as table_name,
    COUNT(*) as total_embeddings,
    AVG(array_length(embedding::real[], 1)) as avg_dimension,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created,
    COUNT(DISTINCT agent_id) as unique_agents
FROM vectors.agent_memory

UNION ALL

SELECT 
    'code_embeddings' as table_name,
    COUNT(*) as total_embeddings,
    AVG(array_length(embedding::real[], 1)) as avg_dimension,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created,
    COUNT(DISTINCT language) as unique_languages
FROM vectors.code_embeddings;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_analytics_table_name 
    ON vectors.embedding_analytics(table_name);

-- Grant permissions
GRANT USAGE ON SCHEMA vectors TO llamaagent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA vectors TO llamaagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA vectors TO llamaagent;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA vectors TO llamaagent;

-- Create scheduled job for cleanup (requires pg_cron extension)
-- This will be enabled if pg_cron is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Schedule cleanup job to run daily at 2 AM
        PERFORM cron.schedule('embedding-cleanup', '0 2 * * *', 'SELECT vectors.cleanup_old_embeddings();');
        -- Schedule analytics refresh every 6 hours
        PERFORM cron.schedule('embedding-analytics-refresh', '0 */6 * * *', 'REFRESH MATERIALIZED VIEW vectors.embedding_analytics;');
    END IF;
EXCEPTION
    WHEN others THEN
        NULL; -- Ignore errors if pg_cron is not available
END;
$$;

-- Insert sample data for testing
INSERT INTO vectors.document_embeddings (document_id, chunk_text, embedding, model_name, metadata)
SELECT 
    d.id,
    'Sample text chunk for testing vector operations',
    '[0.1, 0.2, 0.3]'::vector(3), -- Sample 3D vector for testing
    'test-model',
    '{"test": true}'::jsonb
FROM knowledge.documents d
LIMIT 1
ON CONFLICT DO NOTHING;

-- Log completion
SELECT 
    'Vector extensions initialized successfully' as status,
    NOW() as completed_at,
    COUNT(*) as total_vector_functions
FROM information_schema.routines 
WHERE routine_schema = 'vectors'; 