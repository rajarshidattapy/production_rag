"""Benchmark HNSW settings and BM25 build/query cost at corpus scale.

Measures Chroma's HNSW directly on clustered synthetic vectors (so the nearest-neighbour
problem is non-trivial) against brute-force ground truth, then times BM25.

Usage:
    python scripts/bench_vector_store.py --n 50000 --queries 200
    python scripts/bench_vector_store.py --skip-hnsw     # BM25 only
"""

from __future__ import annotations

import argparse
import tempfile
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

DIM = 384


def make_vectors(n: int, n_queries: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(n // 1000, 10), DIM)).astype(np.float32)
    data = centers[rng.integers(0, len(centers), n)] + 1.2 * rng.normal(size=(n, DIM)).astype(
        np.float32
    )
    base = data[rng.choice(n, n_queries, replace=False)]
    queries = base + 0.8 * rng.normal(size=base.shape).astype(np.float32)
    return data, queries


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def bench_hnsw(n: int, n_queries: int, k: int = 10) -> None:
    data, queries = make_vectors(n, n_queries)
    truth = np.argsort(-(unit(queries) @ unit(data).T), axis=1)[:, :k]
    ids = [str(i) for i in range(n)]
    print(f"\n== HNSW  n={n} dim={DIM} queries={n_queries} k={k} ==")
    print(
        f"{'M':>4} {'ef_c':>5} {'ef_s':>5} {'build_s':>8} {'recall@10':>10} {'p50_ms':>7} {'p95_ms':>7}"
    )

    # ef_search only takes effect at collection creation (col.modify() does not re-tune a built
    # index), so each setting gets its own collection.
    for m, ef_c, ef_s in [
        (16, 100, 10),
        (16, 100, 100),
        (16, 100, 200),
        (32, 200, 100),
        (32, 200, 200),
    ]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            client = chromadb.PersistentClient(
                path=tmp, settings=ChromaSettings(anonymized_telemetry=False)
            )
            col = client.create_collection(
                f"bench_{uuid.uuid4().hex[:8]}",
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "max_neighbors": m,
                        "ef_construction": ef_c,
                        "ef_search": ef_s,
                    }
                },
            )
            t0 = time.perf_counter()
            batch = 5000
            for s in range(0, n, batch):
                col.add(ids=ids[s : s + batch], embeddings=data[s : s + batch].tolist())
            build = time.perf_counter() - t0
            lat, hit = [], 0
            for q, t in zip(queries, truth, strict=True):
                s = time.perf_counter()
                res = col.query(query_embeddings=[q.tolist()], n_results=k)["ids"][0]
                lat.append((time.perf_counter() - s) * 1000)
                hit += len({int(r) for r in res} & set(t.tolist()))
            recall = hit / (k * n_queries)
            print(
                f"{m:>4} {ef_c:>5} {ef_s:>5} {build:>8.1f} {recall:>10.3f} "
                f"{np.percentile(lat, 50):>7.1f} {np.percentile(lat, 95):>7.1f}"
            )


def bench_bm25(n: int) -> None:
    from src.retrieval.hybrid import HybridRetriever

    class FakeStore:
        collection_name = "bench"
        persist_path = Path(tempfile.mkdtemp()) / "chroma"

        def __init__(self) -> None:
            rng = np.random.default_rng(1)
            vocab = [f"w{i}" for i in range(5000)]
            self.rows = [
                {
                    "id": f"c{i}",
                    "document": " ".join(rng.choice(vocab, 120)),
                    "metadata": {"source": f"d{i % 50}.txt"},
                    "score": 1.0,
                }
                for i in range(n)
            ]

        @property
        def _by_id(self) -> dict[str, dict]:
            if getattr(self, "_cache_n", -1) != len(self.rows):
                self._cache = {r["id"]: r for r in self.rows}
                self._cache_n = len(self.rows)
            return self._cache

        def iter_chunks(self, page_size: int = 5000, include_metadata: bool = True):
            for s in range(0, len(self.rows), page_size):
                yield self.rows[s : s + page_size]

        def get_chunks(self, ids: list[str]) -> list[dict]:
            by_id = self._by_id
            return [by_id[i] for i in ids if i in by_id]

        def count(self) -> int:
            return len(self.rows)

        def similarity_search(self, query: str, k: int = 5, where=None) -> list[dict]:
            return []

    print(f"\n== BM25  n={n} (~120 tokens/chunk) ==")
    store = FakeStore()
    r = HybridRetriever(store, persist_path=store.persist_path.parent / "bm25.json")  # type: ignore[arg-type]
    t = time.perf_counter()
    r.build_index()
    print(f"build_index (tokenize+index+save): {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    r2 = HybridRetriever(store, persist_path=r.persist_path)  # type: ignore[arg-type]
    assert r2._load_index()
    print(f"cold start _load_index:            {time.perf_counter() - t:.2f}s")
    lat = []
    for q in ["w1 w2 w3", "w100 w4000", "w7 w8 w9 w10 w11"] * 10:
        s = time.perf_counter()
        r2._bm25_search(q, 20)
        lat.append((time.perf_counter() - s) * 1000)
    print(
        f"_bm25_search latency:              p50 {np.percentile(lat, 50):.0f} ms  p95 {np.percentile(lat, 95):.0f} ms"
    )
    # Ingesting a small batch: full rebuild (old behaviour) vs incremental append.
    from src.ingestion.chunker import Chunk

    extra = [
        Chunk(content="w1 w2 fresh document", metadata={"source": "n.txt"}, chunk_id=f"new{i}")
        for i in range(50)
    ]
    store.rows.extend(
        {"id": c.chunk_id, "document": c.content, "metadata": c.metadata, "score": 1.0}
        for c in extra
    )
    t = time.perf_counter()
    assert r2.add_chunks(extra)
    r2._bm25_search("w1 w2", 20)
    print(f"+50 chunks, incremental append:    {time.perf_counter() - t:.2f}s")
    r2.invalidate_index()
    t = time.perf_counter()
    r2._bm25_search("w1 w2", 20)
    print(f"+50 chunks, full rebuild (old):    {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    a = ap.parse_args()
    if not a.skip_bm25:
        bench_bm25(a.n)
    if not a.skip_hnsw:
        bench_hnsw(a.n, a.queries)
