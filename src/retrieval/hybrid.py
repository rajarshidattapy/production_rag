"""Hybrid retrieval combining BM25 keyword search with vector similarity."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from array import array
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Maximal runs of ASCII alphanumerics and Latin-extended letters (de/es/fr text).
_TOKEN_RE = re.compile(r"[a-zA-Z0-9À-ɏ]+")

# Persisted-index format and tokenizer revisions; a mismatch forces a rebuild from the store.
_FORMAT_VERSION = 3
_TOKENIZER_VERSION = 1

# Smallest delta journal we bother compacting back into the base file.
_MIN_COMPACT_BYTES = 1024 * 1024
assert array("Q").itemsize == 8 and array("I").itemsize == 4, (
    "postings are persisted as packed uint64"
)


class _InvertedBM25:
    """Okapi BM25 over an inverted index, with append support.

    Scores are identical to ``rank_bm25.BM25Okapi`` (same IDF with the epsilon floor,
    same k1/b length normalisation) but each query only touches the postings of its own
    terms instead of scanning every document, and ``add_documents`` appends without
    rebuilding. Each posting is one packed ``uint64`` (doc index << 32 | term frequency)
    in a per-term ``array('Q')``: 8 bytes per posting and a single append in the hot loop.
    """

    def __init__(
        self,
        corpus: Iterable[list[str]] = (),
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self._doc_len = array("I")
        self._total_tokens = 0
        self._postings: dict[str, array] = {}
        self._average_idf: float | None = None  # lazily recomputed after appends
        self.add_documents(corpus)

    # -- building --------------------------------------------------------------------------

    def add_documents(self, tokenized_docs: Iterable[list[str]]) -> None:
        postings = self._postings
        for tokens in tokenized_docs:
            doc_bits = self.corpus_size << 32
            for term, tf in Counter(tokens).items():
                posting = postings.get(term)
                if posting is None:
                    posting = postings[term] = array("Q")
                posting.append(doc_bits | tf)
            self._doc_len.append(len(tokens))
            self._total_tokens += len(tokens)
            self.corpus_size += 1
        self._average_idf = None

    # -- scoring ---------------------------------------------------------------------------

    def _raw_idf(self, df: int) -> float:
        return math.log(self.corpus_size - df + 0.5) - math.log(df + 0.5)

    def _get_average_idf(self) -> float:
        if self._average_idf is None:
            if not self._postings:
                self._average_idf = 0.0
            else:
                df = np.fromiter(
                    (len(p) for p in self._postings.values()),
                    dtype=np.float64,
                    count=len(self._postings),
                )
                raw = np.log(self.corpus_size - df + 0.5) - np.log(df + 0.5)
                self._average_idf = float(raw.sum() / len(raw))
        return self._average_idf

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        if not self.corpus_size:
            return scores
        avgdl = self._total_tokens / self.corpus_size
        floor = self.epsilon * self._get_average_idf()
        doc_len = np.array(self._doc_len, dtype=np.float64)
        for term in query:
            posting = self._postings.get(term)
            if posting is None:
                continue
            idf = self._raw_idf(len(posting))
            if idf < 0:
                idf = floor
            packed = np.frombuffer(posting, dtype=np.uint64)
            idx = (packed >> np.uint64(32)).astype(np.intp)
            tf = (packed & np.uint64(0xFFFFFFFF)).astype(np.float64)
            del packed  # release the buffer export so the array can grow again
            scores[idx] += idf * (
                tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len[idx] / avgdl))
            )
        return scores

    # -- (de)serialisation -----------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flatten to plain numpy arrays (CSR layout) for ``np.savez``."""
        terms = list(self._postings)
        lengths = np.fromiter(
            (len(self._postings[t]) for t in terms), dtype=np.int64, count=len(terms)
        )
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        postings = (
            np.concatenate([np.frombuffer(self._postings[t], dtype=np.uint64) for t in terms])
            if terms
            else np.zeros(0, dtype=np.uint64)
        )
        return {
            "terms": np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8),
            "offsets": offsets,
            "postings": postings,
            "doc_len": np.array(self._doc_len, dtype=np.uint32),
        }

    @classmethod
    def from_arrays(
        cls,
        terms: np.ndarray,
        offsets: np.ndarray,
        postings: np.ndarray,
        doc_len: np.ndarray,
    ) -> _InvertedBM25:
        obj = cls()
        names = terms.tobytes().decode("utf-8").split("\n") if terms.size else []
        if len(offsets) != len(names) + 1 or (len(offsets) and int(offsets[-1]) != len(postings)):
            raise ValueError("corrupt BM25 index: term/posting mismatch")
        postings = np.ascontiguousarray(postings, dtype=np.uint64)
        for i, term in enumerate(names):
            posting = array("Q")
            posting.frombytes(postings[int(offsets[i]) : int(offsets[i + 1])].tobytes())
            obj._postings[term] = posting
        obj._doc_len.frombytes(np.ascontiguousarray(doc_len, dtype=np.uint32).tobytes())
        obj.corpus_size = len(obj._doc_len)
        obj._total_tokens = int(np.asarray(doc_len, dtype=np.int64).sum())
        return obj


class HybridRetriever:
    """Fuses BM25 (keyword) and vector (semantic) search results using RRF.

    Reciprocal Rank Fusion (RRF) combines rankings from multiple methods
    into a single relevance score, smoothing out individual method weaknesses.

    The BM25 side keeps only chunk IDs and term postings in memory (about 1 KB per
    chunk). Text and metadata for the few hits of each query are fetched from the
    vector store, which already holds them, instead of being duplicated here.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        alpha: float = 0.6,
        rrf_k: int = 60,
        persist_path: str | Path | None = None,
        max_index_mb: int = 500,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if max_index_mb <= 0:
            raise ValueError(f"max_index_mb must be positive, got {max_index_mb}")

        self.vector_store = vector_store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self._max_index_bytes = max_index_mb * 1024 * 1024
        self.persist_path = (
            Path(persist_path)
            if persist_path
            else vector_store.persist_path.parent / f"bm25_index_{vector_store.collection_name}.idx"
        )
        # Append-only journal of chunks added since the base file was written, so an
        # ingest costs O(new chunks) on disk instead of rewriting the whole index.
        self._delta_path = self.persist_path.with_name(self.persist_path.name + ".delta")
        self._base_bytes = 0
        self._delta_bytes = 0
        self._bm25: _InvertedBM25 | None = None
        self._id_set: set[str] = set()
        # Guards the index: appends mutate it while searches read it.
        self._index_lock = threading.RLock()
        self._corpus_ids: list[str] = []
        self._stale = False  # Start False to allow loading from disk on query.
        # If disk load fails or misses, search will trigger build_index.

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Build the BM25 index by streaming every chunk out of the vector store.

        Pages are tokenized and dropped as they arrive, so peak memory is the index
        itself, not the corpus text.
        """
        bm25 = _InvertedBM25()
        ids: list[str] = []
        for page in self.vector_store.iter_chunks(include_metadata=False):
            ids.extend(c["id"] for c in page)
            bm25.add_documents(self._tokenize(c["document"]) for c in page)

        with self._index_lock:
            if not ids:
                logger.warning("No chunks found to build BM25 index")
            self._bm25 = bm25
            self._corpus_ids = ids
            self._id_set = set(ids)
            self._stale = False
            logger.info("BM25 index built with %d documents", len(ids))
            self._save_index()

    def add_chunks(self, chunks: list[Any]) -> bool:
        """Append newly ingested chunks to the live index instead of rebuilding it.

        Cost is proportional to the new chunks, in memory and on disk (appended to a
        journal), not the whole corpus. Returns False -- after invalidating, so the next
        search does a full rebuild -- whenever the incremental path can't be trusted:
        no live index, an empty corpus, or chunks without a stable ID.
        """
        with self._index_lock:
            if (
                self._bm25 is None
                or self._stale
                or not self._corpus_ids
                or any(not c.chunk_id for c in chunks)
            ):
                self.invalidate_index()
                return False

            # Chroma ignores re-adds of an existing ID, so mirror that to stay in sync.
            fresh = [c for c in chunks if c.chunk_id not in self._id_set]
            if fresh:
                new_ids = [c.chunk_id for c in fresh]
                if len(set(new_ids)) != len(new_ids):
                    self.invalidate_index()
                    return False
                new_texts = [c.content for c in fresh]
                self._bm25.add_documents([self._tokenize(t) for t in new_texts])
                self._corpus_ids.extend(new_ids)
                self._id_set.update(new_ids)
                self._append_delta(new_ids, new_texts)
            return True

    def invalidate_index(self) -> None:
        """Mark the BM25 index as stale, forcing a rebuild on next search.

        Call this after any ingestion/deletion against the underlying vector
        store — a count-only staleness check would miss same-count
        replacements (e.g. re-ingesting updated documents with the same
        chunk count).
        """
        self._stale = True
        for path in (self.persist_path, self._delta_path):
            if path and path.exists():
                try:
                    path.unlink()
                    logger.info("Removed stale BM25 index file: %s", path)
                except Exception:
                    logger.debug("Failed to remove stale BM25 index file", exc_info=True)
        self._base_bytes = self._delta_bytes = 0

    def _drop_persisted(self) -> None:
        for path in (self.persist_path, self._delta_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed to remove %s", path, exc_info=True)
        self._base_bytes = self._delta_bytes = 0

    def _save_index(self) -> None:
        """Write the index to disk as an uncompressed ``.npz`` (compacting any journal).

        Postings are stored as plain integer arrays and loaded with ``allow_pickle=False``,
        so there is no pickle risk, and restart skips tokenizing the whole corpus. The file
        is written to a temp name and renamed, so a crash never leaves a half-written index.
        """
        if self.persist_path is None or self._bm25 is None:
            return

        tmp = self.persist_path.with_name(self.persist_path.name + ".tmp")
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            ids_json = json.dumps(self._corpus_ids, ensure_ascii=False).encode("utf-8")
            # Use os.open with 0o600 so the index (which holds document ids) is readable only
            # by the owning process — not world-readable.
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, "wb") as f:
                arrays: dict[str, Any] = {
                    "version": np.array([_FORMAT_VERSION]),
                    "tokenizer": np.array([_TOKENIZER_VERSION]),
                    "ids": np.frombuffer(ids_json, dtype=np.uint8),
                    **self._bm25.to_arrays(),
                }
                np.savez(f, **arrays)
            size = tmp.stat().st_size
            if size > self._max_index_bytes:
                # A file we would refuse to load is just wasted disk and a rebuild per restart.
                tmp.unlink()
                self._drop_persisted()
                logger.warning(
                    "BM25 index is %.0f MB, over the %d MB limit — not persisted. "
                    "Raise RAG_BM25_MAX_INDEX_MB or use use_hybrid=false.",
                    size / (1024 * 1024),
                    self._max_index_bytes // (1024 * 1024),
                )
                return
            os.replace(tmp, self.persist_path)
            # The base now contains everything the journal held.
            self._delta_path.unlink(missing_ok=True)
            self._base_bytes, self._delta_bytes = size, 0
            logger.info("Saved BM25 index to %s", self.persist_path)
        except Exception:
            logger.exception("Failed to save BM25 index to %s", self.persist_path)
            tmp.unlink(missing_ok=True)

    def _append_delta(self, ids: list[str], texts: list[str]) -> None:
        """Journal newly added chunks (one JSON line each) instead of rewriting the base file."""
        if not self.persist_path.exists():
            return  # nothing persisted to extend; the next full build will write the base
        try:
            lines = "".join(
                json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n"
                for i, t in zip(ids, texts, strict=True)
            )
            fd = os.open(self._delta_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(lines)
            self._delta_bytes += len(lines.encode("utf-8"))
            if self._base_bytes + self._delta_bytes > self._max_index_bytes:
                self._drop_persisted()
                logger.warning(
                    "BM25 index outgrew the %d MB limit — no longer persisted. "
                    "Raise RAG_BM25_MAX_INDEX_MB or use use_hybrid=false.",
                    self._max_index_bytes // (1024 * 1024),
                )
            elif self._delta_bytes > max(self._base_bytes // 4, _MIN_COMPACT_BYTES):
                self._save_index()  # compact: amortised O(1) per chunk
        except Exception:
            logger.exception("Failed to journal BM25 chunks to %s", self._delta_path)
            self._drop_persisted()  # never leave a base that disagrees with memory

    def _load_index(self) -> bool:
        """Load the persisted index (plus journal) from disk. Returns True on success."""
        if self.persist_path is None or not self.persist_path.exists():
            return False

        # Guard against unbounded memory usage from a corrupted/oversized index file.
        try:
            base_size = self.persist_path.stat().st_size
            delta_size = self._delta_path.stat().st_size if self._delta_path.exists() else 0
        except OSError:
            return False
        file_size = base_size + delta_size
        if file_size > self._max_index_bytes:
            logger.warning(
                "BM25 index file %s is %.0f MB, exceeding the %d MB limit (RAG_BM25_MAX_INDEX_MB) — skipping disk load",
                self.persist_path,
                file_size / (1024 * 1024),
                self._max_index_bytes // (1024 * 1024),
            )
            return False

        try:
            with np.load(self.persist_path, allow_pickle=False) as z:
                if (
                    int(z["version"][0]) != _FORMAT_VERSION
                    or int(z["tokenizer"][0]) != _TOKENIZER_VERSION
                ):
                    logger.info(
                        "BM25 index %s is from another format version; rebuilding",
                        self.persist_path,
                    )
                    return False
                ids: list[str] = json.loads(z["ids"].tobytes().decode("utf-8"))
                bm25 = _InvertedBM25.from_arrays(
                    z["terms"], z["offsets"], z["postings"], z["doc_len"]
                )
            if bm25.corpus_size != len(ids):
                raise ValueError("corrupt BM25 index: id/document count mismatch")

            seen = set(ids)
            if delta_size:
                new_texts: list[str] = []
                with open(self._delta_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            break  # torn final write after a crash; the count check rebuilds
                        if row["id"] not in seen:
                            seen.add(row["id"])
                            ids.append(row["id"])
                            new_texts.append(row["text"])
                bm25.add_documents(self._tokenize(t) for t in new_texts)

            with self._index_lock:
                self._bm25, self._corpus_ids, self._id_set = bm25, ids, seen
                self._stale = False
            self._base_bytes, self._delta_bytes = base_size, delta_size
            logger.info("Loaded BM25 index from %s (%d docs)", self.persist_path, len(ids))
            return True
        except Exception:
            logger.warning("Failed to load BM25 index from %s, will rebuild", self.persist_path)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        where: dict[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search: BM25 + vector, fused with RRF.

        Args:
            query: The search query string.
            k: Number of final results.
            where: Optional metadata filter, applied to both BM25 and vector search.
            alpha: Per-call override of the vector/BM25 weight. The retriever is
                shared across requests, so callers that need a different balance
                (e.g. self-healing retries) pass it here instead of mutating
                ``self.alpha``.

        Returns:
            Ranked list of result dicts (id, document, metadata, score).
        """
        if alpha is None:
            alpha = self.alpha
        elif not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # --- BM25 scores ---
        bm25_results = self._bm25_search(query, k, where)
        bm25_rank = {r["id"]: i for i, r in enumerate(bm25_results)}

        # --- Vector scores ---
        vector_results = self.vector_store.similarity_search(query, k=k, where=where)
        vector_rank = {r["id"]: i for i, r in enumerate(vector_results)}

        # --- RRF fusion ---
        all_ids = set(bm25_rank.keys()) | set(vector_rank.keys())

        rrf_scores: dict[str, float] = {}
        for doc_id in all_ids:
            bm25_r = bm25_rank.get(doc_id, k)  # default to worst rank
            vec_r = vector_rank.get(doc_id, k)
            # Weighted RRF
            score = alpha * (1.0 / (self.rrf_k + vec_r + 1)) + (1.0 - alpha) * (
                1.0 / (self.rrf_k + bm25_r + 1)
            )
            rrf_scores[doc_id] = score

        # --- Build result list ---
        seen_docs: dict[str, dict[str, Any]] = {}
        for r in bm25_results:
            seen_docs[r["id"]] = r
        for r in vector_results:
            if r["id"] not in seen_docs:
                seen_docs[r["id"]] = r

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

        results = []
        for doc_id in sorted_ids:
            if doc_id in seen_docs:
                result = dict(seen_docs[doc_id])
                result["score"] = round(rrf_scores[doc_id], 4)
                results.append(result)

        return results

    def _bm25_search(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run BM25 keyword search, optionally over only the chunks matching ``where``."""
        current_count = self.vector_store.count()
        allowed = self.vector_store.ids_matching(where) if where else None

        with self._index_lock:
            # Load from disk if current state is memory-cold but not stale
            if self._bm25 is None and not self._stale:
                self._load_index()

            # Rebuild if still missing, stale, or count differs from DB
            if self._bm25 is None or self._stale or len(self._corpus_ids) != current_count:
                # Try loading from disk if memory is clean but disk index matches DB count
                if (
                    self._bm25 is None
                    and self._load_index()
                    and len(self._corpus_ids) == current_count
                ):
                    pass
                else:
                    self.build_index()

            if self._bm25 is None:
                raise RuntimeError("BM25 index unavailable after build attempt")

            if not self._corpus_ids:
                return []

            scores = self._bm25.get_scores(self._tokenize(query))
            n = len(scores)
            if allowed is not None:
                # The index holds no metadata, so filtered-out chunks are simply scored as misses.
                keep = np.fromiter((i in allowed for i in self._corpus_ids), dtype=bool, count=n)
                scores[~keep] = 0.0
            if n > 8 * k:
                # Avoid sorting the whole corpus: keep everything scoring at least the k-th best
                # (or every positive score if fewer than k match), then order those.
                kth = np.partition(scores, n - k)[n - k]
                candidates = np.flatnonzero(scores >= kth if kth > 0 else scores > 0)
            else:
                candidates = np.arange(n)
            # Stable descending sort keeps the original tie order (lower index first).
            top = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
            hits = [(self._corpus_ids[i], float(scores[i])) for i in top if scores[i] > 0]

        if not hits:
            return []
        # The index holds only ids and term postings; text and metadata live in the vector store.
        rows = {r["id"]: r for r in self.vector_store.get_chunks([i for i, _ in hits])}
        return [
            {
                "id": doc_id,
                "document": rows[doc_id]["document"],
                "metadata": rows[doc_id]["metadata"],
                "score": round(score, 4),
            }
            for doc_id, score in hits
            if doc_id in rows  # chunks deleted since indexing are skipped
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric boundaries, preserving Unicode letters.

        Handles hyphenated terms (e.g. "GPT-4" → ["gpt", "4"]), punctuation-adjacent
        words, and multilingual Latin-extended characters used in de/es/fr text.
        Empty tokens are filtered out.
        """
        return _TOKEN_RE.findall(text.lower())
