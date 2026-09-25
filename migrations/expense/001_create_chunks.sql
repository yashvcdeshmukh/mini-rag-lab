CREATE EXTENSION IF NOT EXISTS vector;

-- Width matches mini_rag.embeddings.EMBEDDING_DIM for all-MiniLM-L6-v2.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    version TEXT NOT NULL,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(trim(section_title)) > 0)
);
