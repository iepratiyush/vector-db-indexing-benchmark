"""
Demonstrates IVFFlat vs HNSW indexing in pgvector.

Uses the AG News dataset (120k real news snippets across World, Sports,
Business, and Sci/Tech) embedded with sentence-transformers so that nearest-
neighbour structure reflects genuine semantic similarity — no synthetic data,
no artificial clustering.

First run downloads AG News (~10 MB) and all-MiniLM-L6-v2 (~80 MB) to
~/.cache/huggingface. Subsequent runs use the local cache.

Setup: docker compose up -d && pip install -r requirements.txt
Run:   python indexing_demo.py
"""

import time
import psycopg2
import psycopg2.extras
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

DB_URL    = "postgresql://postgres:postgres@localhost:5432/vectors"
MODEL     = "all-MiniLM-L6-v2"
EMBED_DIM = 384
N_ROWS    = 10_000
BATCH     = 256


def load_corpus(n: int = N_ROWS, seed: int = 42) -> list[str]:
    """Load n real news snippets from AG News, shuffled so all four categories
    (World, Sports, Business, Sci/Tech) are represented evenly."""
    dataset = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=seed)
    return dataset.select(range(min(n, len(dataset))))["text"]


def seed_real(conn, model: SentenceTransformer, n: int = N_ROWS):
    corpus = load_corpus(n)

    print(f"Encoding {n:,} news snippets...", flush=True)
    t0 = time.perf_counter()
    vecs = model.encode(corpus, batch_size=BATCH, show_progress_bar=True).tolist()
    print(f"Encoded in {time.perf_counter() - t0:.1f}s.", flush=True)

    print("Inserting...", flush=True)
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for i in range(0, n, BATCH):
            psycopg2.extras.execute_values(
                cur, "INSERT INTO index_test (text, vec) VALUES %s",
                list(zip(corpus[i:i+BATCH], vecs[i:i+BATCH]))
            )
    conn.commit()
    print(f"Inserted {n:,} rows in {time.perf_counter() - t0:.1f}s.")


def reset_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS index_test")
        cur.execute(f"""
            CREATE TABLE index_test (
                id   SERIAL PRIMARY KEY,
                text TEXT,
                vec  vector({EMBED_DIM})
            )
        """)
    conn.commit()
    print("Table reset.")


def drop_index(conn):
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_test_vec")
    conn.commit()


# --- IVFFlat ---
# Divides vectors into `lists` clusters (Voronoi cells).
# At query time, scans `probes` nearest clusters.
# Rule of thumb: lists ≈ sqrt(row_count), probes = lists / 10 (min 1).
# Must have data inserted BEFORE building (it clusters existing rows).
def create_ivfflat(conn, lists: int = 10, probes: int = 3):
    drop_index(conn)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE INDEX idx_test_vec
            ON index_test
            USING ivfflat (vec vector_cosine_ops)
            WITH (lists = {lists})
        """)
        # probes is a session-level setting, not part of the index definition
        cur.execute(f"SET ivfflat.probes = {probes}")
    conn.commit()
    print(f"IVFFlat index created (lists={lists}, probes={probes}).")


# --- HNSW ---
# Builds a multi-layer graph. No data required at index creation time.
# m               = max connections per node (default 16)
# ef_construction = search width during build (default 64)
# ef_search       = search width at query time (session setting, default 40)
def create_hnsw(conn, m: int = 16, ef_construction: int = 64, ef_search: int = 64):
    drop_index(conn)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE INDEX idx_test_vec
            ON index_test
            USING hnsw (vec vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction})
        """)
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
    conn.commit()
    print(f"HNSW index created (m={m}, ef_construction={ef_construction}, ef_search={ef_search}).")


def timed_search(conn, vec: list[float], top_k: int = 5) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT text, 1 - (vec <=> %s::vector) AS similarity
            FROM index_test
            ORDER BY vec <=> %s::vector
            LIMIT %s
        """, (vec, vec, top_k))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000  # ms


def get_ground_truth(conn, vec: list[float], top_k: int = 5) -> set[int]:
    """Exact top-k via sequential scan. Call BEFORE creating any index."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM index_test ORDER BY vec <=> %s::vector LIMIT %s", (vec, top_k))
        return {row[0] for row in cur.fetchall()}


def explain_query(conn, vec: list[float]):
    with conn.cursor() as cur:
        cur.execute("""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
            SELECT text, 1 - (vec <=> %s::vector) AS similarity
            FROM index_test
            ORDER BY vec <=> %s::vector
            LIMIT 5
        """, (vec, vec))
        plan = "\n".join(row[0] for row in cur.fetchall())
    print(plan)


def benchmark(conn, label: str, vec: list[float], ground_truth: set[int], runs: int = 5, top_k: int = 5):
    times = [timed_search(conn, vec, top_k) for _ in range(runs)]
    avg = sum(times) / len(times)
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM index_test ORDER BY vec <=> %s::vector LIMIT %s", (vec, top_k))
        approx = {row[0] for row in cur.fetchall()}
    recall = len(ground_truth & approx) / top_k
    print(f"{label}: avg {avg:.2f}ms over {runs} runs  |  recall@{top_k} = {recall:.2f}")


if __name__ == "__main__":
    model = SentenceTransformer(MODEL)

    conn = psycopg2.connect(DB_URL)
    reset_table(conn)
    register_vector(conn)
    seed_real(conn, model, N_ROWS)

    query = "artificial intelligence breakthroughs in machine learning research"
    vec = model.encode(query).tolist()
    ground_truth = get_ground_truth(conn, vec)

    # --- IVFFlat ---
    # lists ≈ sqrt(N) is the standard starting point; probes = lists / 10
    # scans 10% of clusters — reasonable recall without full table scan.
    lists  = max(1, int(N_ROWS ** 0.5))   # 100 for 10k rows
    probes = max(1, lists // 10)           # 10
    print(f"\n{'='*50}\nIVFFlat Index (lists={lists}, probes={probes})\n{'='*50}")
    create_ivfflat(conn, lists=lists, probes=probes)
    benchmark(conn, "IVFFlat", vec, ground_truth)
    explain_query(conn, vec)

    # --- HNSW ---
    # ef_construction=128 builds a denser graph than the default 64 — worth it
    # at 10k rows where build time is negligible.
    # ef_search=100 explores more candidates at query time for higher recall.
    m                = 16
    ef_construction  = 128
    ef_search        = 100
    print(f"\n{'='*50}\nHNSW Index (m={m}, ef_construction={ef_construction}, ef_search={ef_search})\n{'='*50}")
    create_hnsw(conn, m=m, ef_construction=ef_construction, ef_search=ef_search)
    benchmark(conn, "HNSW   ", vec, ground_truth)
    explain_query(conn, vec)

    # --- Sequential scan (no index) ---
    print(f"\n{'='*50}\nNo Index (sequential scan)\n{'='*50}")
    drop_index(conn)
    benchmark(conn, "SeqScan", vec, ground_truth, runs=3)
    explain_query(conn, vec)

    conn.close()
