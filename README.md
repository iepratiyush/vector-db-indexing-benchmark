# pgvector Sandbox

Minimal local setup to experiment with vector embeddings, indexing, and cosine similarity using PostgreSQL + pgvector.

## Stack

- **pgvector** — PostgreSQL extension for vector storage and ANN search
- **sentence-transformers** — local embedding model, no API key required
- **psycopg2 + pgvector** — Python DB driver with vector type support

## Prerequisites

- Docker + Docker Compose
- Python 3.10+

## Setup

```bash
# 1. Start the database
docker compose up -d

# 2. Install Python dependencies
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `example-st.py` | Insert documents, generate embeddings via sentence-transformers, cosine similarity search |
| `indexing_demo.py` | Compare IVFFlat vs HNSW — timing and recall@5 benchmark |
| `docker-compose.yml` | pgvector container |
| `requirements.txt` | Python dependencies |

## Run

```bash
# Basic embedding + search demo
python example-st.py

# Index comparison (IVFFlat vs HNSW vs sequential scan)
python indexing_demo.py
```

## Embedding model

| Setting | Value |
|---------|-------|
| Model | `all-MiniLM-L6-v2` (downloaded once to `~/.cache/huggingface`) |
| Dimensions | 384 |
| Requires API key | No |

## Index quick reference

| Index | Build speed | Query speed | Recall | Best for |
|-------|------------|-------------|--------|---------|
| IVFFlat | Fast | Good | Tunable via `probes` | Large datasets, batch inserts |
| HNSW | Slow | Better | High by default | Low-latency queries |
| Sequential scan | — | Slow | Exact (1.00) | Small datasets, ground truth |

See `indexing_demo.py` for how to create, switch, and benchmark each index type.
