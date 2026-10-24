"""
Demonstrates IVFFlat vs HNSW indexing in pgvector.

Seeds the table with random vectors (no embedding API needed) so you can
test index behaviour at realistic scale without hitting rate limits.

No env vars required. Just: docker compose up -d && python indexing_demo.py
"""

import time
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

DB_URL    = "postgresql://postgres:postgres@localhost:5432/vectors"
EMBED_DIM = 768
N_ROWS    = 50_000   # increase to 200k+ for more pronounced timing differences
BATCH     = 5_000    # rows per INSERT batch


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


def seed_random(conn, n: int = N_ROWS):
    """Insert n unit-normalized random vectors (no API calls)."""
    rng = np.random.default_rng(42)
    print(f"Seeding {n:,} random vectors...", flush=True)
    t0 = time.perf_counter()
    inserted = 0
    with conn.cursor() as cur:
        while inserted < n:
            batch_size = min(BATCH, n - inserted)
            vecs = rng.standard_normal((batch_size, EMBED_DIM)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # unit normalize
            rows = list(zip([f"doc_{inserted + i}" for i in range(batch_size)], vecs.tolist()))
            psycopg2.extras.execute_values(
                cur, "INSERT INTO index_test (text, vec) VALUES %s", rows
            )
            inserted += batch_size
    conn.commit()
    print(f"Done in {time.perf_counter() - t0:.1f}s.")


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
# m           = max connections per node (default 16, higher = better recall, more memory)
# ef_construction = search width during build (default 64, higher = better recall, slower build)
# ef_search   = search width at query time (session setting, default 40)
def create_hnsw(conn, m: int = 16, ef_construction: int = 64, ef_search: int = 40):
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


def random_query_vec() -> list[float]:
    """Generate a single unit-normalized random vector to use as a query."""
    rng = np.random.default_rng()
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


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
    conn = psycopg2.connect(DB_URL)
    reset_table(conn)
    register_vector(conn)
    seed_random(conn, N_ROWS)

    vec = random_query_vec()
    ground_truth = get_ground_truth(conn, vec)  # exact results before any index exists

    # --- IVFFlat ---
    lists = max(1, int(N_ROWS ** 0.5))
    probes = max(1, lists // 10)
    print(f"\n{'='*50}\nIVFFlat Index (lists={lists}, probes={probes})\n{'='*50}")
    create_ivfflat(conn, lists=lists, probes=probes)
    benchmark(conn, "IVFFlat", vec, ground_truth)
    explain_query(conn, vec)

    # --- HNSW ---
    print(f"\n{'='*50}\nHNSW Index (m=16, ef_construction=64, ef_search=64)\n{'='*50}")
    create_hnsw(conn, m=16, ef_construction=64, ef_search=64)
    benchmark(conn, "HNSW   ", vec, ground_truth)
    explain_query(conn, vec)

    # --- Sequential scan (no index) ---
    print(f"\n{'='*50}\nNo Index (sequential scan)\n{'='*50}")
    drop_index(conn)
    benchmark(conn, "SeqScan", vec, ground_truth, runs=3)
    explain_query(conn, vec)

    conn.close()
