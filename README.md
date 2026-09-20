# Mini RAG Lab

A small lab for building a grounded expense-policy assistant with retrieval-augmented generation (RAG). The assistant will answer questions using the employee expense policy document rather than guessing.

This project is implemented on stacked feature branches. The current tip for this work is `feature/structural-chunking`. `main` still has only the initial policy files.

The source policy lives in `policy.md`.

## Current step

Policy parsing and structural chunking are in place. The document is split at numbered section headings into exactly six chunks, each with title, version, section metadata, and a stable chunk ID.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
