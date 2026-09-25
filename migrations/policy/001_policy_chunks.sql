CREATE EXTENSION IF NOT EXISTS vector;

-- Width is BAAI/bge-base-en-v1.5. The expense-policy table stays vector(384).
CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_slug TEXT NOT NULL,
    document TEXT NOT NULL,
    version TEXT NOT NULL,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    parent_section TEXT,
    parent_title TEXT,
    text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    source_file TEXT NOT NULL,
    search_text TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(trim(section_title)) > 0),
    CHECK (length(trim(text)) > 0)
);

CREATE INDEX IF NOT EXISTS policy_chunks_embedding_hnsw
ON policy_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS policy_chunks_search_gin
ON policy_chunks USING gin (to_tsvector('english', search_text));

ALTER TABLE policy_chunks DROP COLUMN IF EXISTS embedding_model;
ALTER TABLE policy_chunks DROP COLUMN IF EXISTS content_hash;

CREATE TABLE IF NOT EXISTS policy_facts (
    document_slug TEXT NOT NULL,
    version TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    value TEXT NOT NULL,
    section TEXT NOT NULL,
    PRIMARY KEY (document_slug, version, fact_key),
    CHECK (length(trim(fact_key)) > 0),
    CHECK (length(trim(value)) > 0)
);
