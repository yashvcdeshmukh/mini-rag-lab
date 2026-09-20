# Mini RAG Lab

A small lab for building a grounded expense-policy assistant with retrieval-augmented generation (RAG). The assistant will answer questions using the employee expense policy document rather than guessing.

This project is implemented on stacked feature branches. The current tip is `feature/ask-cli`. `main` still has only the initial policy files.

The source policy lives in `policy.md`. Embeddings use `all-MiniLM-L6-v2` (384 dimensions, L2-normalized). Retrieval ranks with pgvector cosine distance (`<=>`). Six rows are scanned sequentially; an ANN index would not help at this size.

## Setup

```bash
git checkout feature/ask-cli
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
```

Default `pytest` stays offline. Integration tests that download MiniLM or call Ollama are opted in with `pytest -m integration`.

## Ingest the policy

```bash
docker compose up -d
psql "$DATABASE_URL" -f migrations/001_create_chunks.sql
python -m mini_rag.ingest --policy policy.md
```

`psql` is the primary migration path. Compose init scripts only run on an empty data volume; the SQL is idempotent, so re-running `psql` is safe.

Re-running ingest upserts the same chunk IDs and removes leftover rows for this document, so a version bump still leaves exactly six stored chunks.

## Ask a question

Ollama must be running with `qwen3:8b`. Default pytest does not need Ollama.

```bash
python -m mini_rag.ask --question "How much can I spend on food each day?"
```

The CLI prints assignment JSON on stdout: `answer`, `citation`, and up to three `retrieved_chunks` with numeric distances. Saved live output for the six required questions is in `examples/required-questions.json`.
