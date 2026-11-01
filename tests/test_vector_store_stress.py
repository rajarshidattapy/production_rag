"""Stress test for the vector store (ChromaDB) backend.

One test, run as ordered phases against a single store, so a failure message
names the phase that broke. Embeddings are a deterministic per-text random
vector (the conftest mock is near-constant, which makes HNSW ranking meaningless),
so recall assertions are exact and the test needs no model.

Scale with environment variables:
  RAG_STRESS_CHUNKS       corpus size (default 5000)
  RAG_STRESS_THREADS      concurrent workers (default 16)
  RAG_STRESS_CHROMA_HOST  run against a Chroma server instead of an embedded store
  RAG_STRESS_CHROMA_PORT  port for the server above (default 8000)
"""

from __future__ import annotations

import hashlib
import os
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import VectorStore, _ChromaEmbeddingFunction

N_CHUNKS = int(os.environ.get("RAG_STRESS_CHUNKS", "5000"))
N_THREADS = int(os.environ.get("RAG_STRESS_THREADS", "16"))
CHROMA_HOST = os.environ.get("RAG_STRESS_CHROMA_HOST") or None
CHROMA_PORT = int(os.environ.get("RAG_STRESS_CHROMA_PORT", "8000"))

# Generous ceiling: this guards against pathological slowdowns, not benchmarks.
MAX_P95_QUERY_SECONDS = 5.0


@pytest.fixture(autouse=True)
def spread_out_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the conftest mock with well-separated, deterministic unit-ish vectors."""

    def encode(self, input_texts: list[str]) -> list[list[float]]:
        out = []
        for text in input_texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            r = random.Random(seed)
            out.append([r.gauss(0.0, 1.0) for _ in range(384)])
        return out

    monkeypatch.setattr(_ChromaEmbeddingFunction, "_encode", encode)


def _make_chunks(n: int, prefix: str = "c") -> list[Chunk]:
    rng = random.Random(1234)
    topics = ["alpha", "beta", "gamma", "delta"]
    chunks: list[Chunk] = []
    for i in range(n):
        body = f"{prefix} chunk {i} " + " ".join(
            rng.choice(topics) for _ in range(rng.randint(5, 60))
        )
        chunks.append(
            Chunk(
                content=body,
                metadata={
                    "source": f"doc{i % 50}.txt",
                    "filename": f"doc{i % 50}.txt",
                    "topic": topics[i % 4],
                },
                chunk_id=f"{prefix}-{i}",
                doc_id=f"doc{i % 50}",
            )
        )
    return chunks


def _open_store(path: Path, name: str) -> VectorStore:
    return VectorStore(
        collection_name=name,
        persist_path=path,
        embedding_dim=384,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT if CHROMA_HOST else None,
        embedding_query_cache_size=0,
    )


def test_vector_store_stress(temp_chroma_dir: Path) -> None:
    name = f"stress_{uuid.uuid4().hex[:8]}"
    store = _open_store(temp_chroma_dir, name)
    chunks = _make_chunks(N_CHUNKS)
    by_id = {c.chunk_id: c for c in chunks}

    # ---- Phase 1: bulk ingest (exceeds Chroma's max batch size) ----------
    assert store.count() == 0
    t0 = time.perf_counter()
    assert store.add_chunks(chunks) == N_CHUNKS
    ingest_s = time.perf_counter() - t0
    assert store.count() == N_CHUNKS, "bulk ingest lost or duplicated chunks"
    assert store.add_chunks([]) == 0

    # ---- Phase 2: recall — a chunk's own text should be found ------------
    # HNSW is approximate, so assert a rate rather than 100%.
    rng = random.Random(7)
    sample = rng.sample(chunks, min(200, N_CHUNKS))
    top1 = top5 = 0
    for c in sample:
        hits = store.similarity_search(c.content, k=5)
        assert hits, "no results for an indexed chunk"
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True), "results not ordered by score"
        ids = [h["id"] for h in hits]
        top1 += ids[0] == c.chunk_id
        if c.chunk_id in ids:
            top5 += 1
            hit = hits[ids.index(c.chunk_id)]
            assert hit["document"] == c.content
            assert hit["metadata"]["filename"] == c.metadata["filename"]
            assert hit["score"] > 0.999
    assert top1 / len(sample) >= 0.95, f"recall@1 {top1 / len(sample):.2%}"
    assert top5 / len(sample) >= 0.98, f"recall@5 {top5 / len(sample):.2%}"

    # ---- Phase 3: metadata filtering --------------------------------------
    hits = store.similarity_search("alpha beta", k=50, where={"topic": "gamma"})
    assert hits and all(h["metadata"]["topic"] == "gamma" for h in hits)
    assert store.similarity_search("alpha", k=5, where={"topic": "does-not-exist"}) == []

    # ---- Phase 4: awkward inputs ------------------------------------------
    assert len(store.similarity_search("alpha", k=1)) == 1
    # k beyond the corpus size is fine; very large k (~100k) trips SQLite's
    # "too many SQL variables" inside embedded Chroma, so API-level top_k must stay bounded.
    big_k = min(N_CHUNKS * 2, 5000)
    assert len(store.similarity_search("alpha", k=big_k)) <= min(N_CHUNKS, big_k)
    for q in ["", " ", "日本語のクエリ 🚀", "a" * 20_000, "'; DROP TABLE x; --", "\x00\n\t"]:
        res = store.similarity_search(q.replace("\x00", ""), k=3)
        assert isinstance(res, list) and len(res) <= 3

    # Chroma rejects empty metadata; the failure must be loud and leave the store untouched.
    with pytest.raises(ValueError):
        store.add_chunks([Chunk(content="no metadata", metadata={}, chunk_id="odd-nometa")])
    assert store.count() == N_CHUNKS

    odd = [
        Chunk(content="x", metadata={"source": "s.txt"}, chunk_id="odd-short"),
        Chunk(
            content="日本語 🚀 émoji ünïcode", metadata={"source": "u.txt"}, chunk_id="odd-unicode"
        ),
        Chunk(content="long " * 20_000, metadata={"n": 1, "f": 1.5}, chunk_id="odd-long"),
    ]
    assert store.add_chunks(odd) == len(odd)
    assert store.count() == N_CHUNKS + len(odd)
    got = store.similarity_search("日本語 🚀 émoji ünïcode", k=1)
    assert got[0]["id"] == "odd-unicode"
    assert (
        got[0]["metadata"]["source"] == "u.txt"
    )  # non-string metadata is stringified, not rejected
    store.delete_chunks([c.chunk_id for c in odd])
    assert store.count() == N_CHUNKS

    # ---- Phase 5: duplicate IDs must not corrupt the collection -----------
    try:
        store.add_chunks(chunks[:100])
    except Exception:
        pass  # rejecting duplicates is acceptable; silently growing is not
    assert store.count() == N_CHUNKS, "re-adding existing ids changed the collection size"

    # ---- Phase 6: concurrent readers + writers ----------------------------
    new_chunks = _make_chunks(max(N_CHUNKS // 5, 100), prefix="w")
    writer_batches = [new_chunks[i : i + 50] for i in range(0, len(new_chunks), 50)]
    query_pool = [c.content for c in rng.sample(chunks, min(100, N_CHUNKS))]
    latencies: list[float] = []

    def read(q: str) -> str:
        s = time.perf_counter()
        res = store.similarity_search(q, k=5)
        latencies.append(time.perf_counter() - s)
        return res[0]["document"]

    def write(batch: list[Chunk]) -> int:
        return store.add_chunks(batch)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        read_futs = [
            pool.submit(read, query_pool[i % len(query_pool)]) for i in range(N_THREADS * 25)
        ]
        write_futs = [pool.submit(write, b) for b in writer_batches]
        read_docs = [f.result(timeout=300) for f in read_futs]  # re-raises worker exceptions
        written = sum(f.result(timeout=300) for f in write_futs)

    assert written == len(new_chunks)
    assert store.count() == N_CHUNKS + len(new_chunks), "concurrent writes lost chunks"
    # Readers raced with writers but old chunks are never removed, so reads should still hit.
    exact = sum(d in set(query_pool) for d in read_docs) / len(read_docs)
    assert exact >= 0.95, f"only {exact:.2%} of concurrent reads returned the expected top-1"
    p95 = statistics.quantiles(latencies, n=20)[-1]
    assert p95 < MAX_P95_QUERY_SECONDS, f"p95 query latency {p95:.2f}s under concurrency"

    # ---- Phase 7: full scan (used by BM25 index build) --------------------
    everything = store.get_all_chunks()
    assert len(everything) == store.count()
    assert {c["id"] for c in everything} >= set(list(by_id)[:50])

    # ---- Phase 8: deletes --------------------------------------------------
    victims = [c.chunk_id for c in chunks[::2]]
    store.delete_chunks(victims)
    store.delete_chunks(["never-existed-1", "never-existed-2"])  # must not raise
    assert store.count() == N_CHUNKS + len(new_chunks) - len(victims)
    probe = store.similarity_search(chunks[0].content, k=10)
    assert chunks[0].chunk_id not in {h["id"] for h in probe}, "deleted chunk still retrievable"
    survivor = chunks[1]
    assert survivor.chunk_id in {h["id"] for h in store.similarity_search(survivor.content, k=5)}

    # ---- Phase 9: persistence across a fresh client -----------------------
    expected_count = store.count()
    reopened = _open_store(temp_chroma_dir, name)
    assert reopened.count() == expected_count, "data did not survive reopening the store"
    assert survivor.chunk_id in {h["id"] for h in reopened.similarity_search(survivor.content, k=5)}

    # ---- Phase 10: two collections stay isolated --------------------------
    other = _open_store(temp_chroma_dir, f"{name}_other")
    other.add_chunks(_make_chunks(10, prefix="o"))
    assert other.count() == 10 and reopened.count() == expected_count
    assert all(h["id"].startswith("o-") for h in other.similarity_search("alpha", k=5))
    other.delete_collection()

    # ---- Phase 11: wipe and reuse ------------------------------------------
    reopened.delete_collection()
    assert reopened.count() == 0
    assert reopened.get_all_chunks() == []
    assert reopened.similarity_search("alpha", k=5) == []
    assert reopened.add_chunks(_make_chunks(25, prefix="r")) == 25
    assert reopened.count() == 25

    print(f"\nstress: {N_CHUNKS} chunks ingested in {ingest_s:.1f}s, query p95 {p95 * 1000:.0f} ms")
