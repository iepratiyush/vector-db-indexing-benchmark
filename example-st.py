"""
Minimal pgvector example: embed text → store → cosine similarity search.
Uses a local sentence-transformers model — no API key required.

Model: all-MiniLM-L6-v2 (~80MB, downloaded once to ~/.cache/huggingface)
Dimensions: 384

Setup:
  docker compose up -d
  pip install -r requirements.txt

Run:
  python example-st.py
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DB_URL    = "postgresql://postgres:postgres@localhost:5432/vectors"
EMBED_DIM = 384

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()


def setup(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS items_st (
                id   SERIAL PRIMARY KEY,
                text TEXT,
                vec  vector({EMBED_DIM})
            )
        """)
        cur.execute("TRUNCATE TABLE items_st")
    conn.commit()



def search(conn, query: str, top_k: int = 3):
    vec = get_embedding(query)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT text, 1 - (vec <=> %s::vector) AS similarity
            FROM items_st
            ORDER BY vec <=> %s::vector
            LIMIT %s
        """, (vec, vec, top_k))
        rows = cur.fetchall()
    print(f"\nTop {top_k} results for {query!r}:")
    for text, score in rows:
        print(f"  [{score:.4f}] {text}")


if __name__ == "__main__":
    conn = psycopg2.connect(DB_URL)
    setup(conn)
    register_vector(conn)

    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "PostgreSQL is a powerful open-source relational database",
        "Vector databases enable semantic similarity search",
        "Python is widely used in data science and AI",
    ]

    # Batch encode all docs in one call (faster than one-by-one)
    vecs = model.encode(docs)
    with conn.cursor() as cur:
        for text, vec in zip(docs, vecs):
            cur.execute("INSERT INTO items_st (text, vec) VALUES (%s, %s)", (text, vec.tolist()))
            print(f"Inserted: {text!r}")
    conn.commit()

    search(conn, "AI and deep learning")
    search(conn, "database storage systems")

    conn.close()
