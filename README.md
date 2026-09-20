# Mini RAG Lab

A small lab for building a grounded expense-policy assistant with retrieval-augmented generation (RAG). The assistant will answer questions using the employee expense policy document rather than guessing.

This project is implemented on stacked feature branches. The current tip for this work is `feature/pgvector-storage`. `main` still has only the initial policy files.

The source policy lives in `policy.md`. Embeddings use `all-MiniLM-L6-v2` (384 dimensions, L2-normalized) so later retrieval can rank with pgvector cosine distance (`<=>`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
```

## Postgres (pgvector)

Start the database, then apply the migration. The SQL is idempotent. Docker init scripts only run on an empty data volume, so `psql` is the primary apply path:

```bash
docker compose up -d
psql "$DATABASE_URL" -f migrations/001_create_chunks.sql
```
