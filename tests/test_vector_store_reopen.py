"""Recovery from embedded Chroma's "hnsw segment reader" failure on a fresh collection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chromadb
import pytest
from chromadb.errors import InternalError

from src.ingestion.chunker import Chunk
from src.retrieval import vector_store as vector_store_module
from src.retrieval.vector_store import VectorStore

READER_ERROR = "Error creating hnsw segment reader: Nothing found on disk"


class BrokenReads:
    """A collection whose reads fail like the real bug, while everything else still works."""

    def __init__(self, collection: Any, message: str = READER_ERROR) -> None:
        self._collection = collection
        self._message = message
        self.queries = 0

    def query(self, **_kwargs: Any) -> Any:
        self.queries += 1
        raise InternalError(self._message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._collection, name)


def _chunk(i: int) -> Chunk:
    return Chunk(content=f"Document {i}", metadata={"source": "a"}, chunk_id=f"c{i}", doc_id="a")


def _store(root: Path, name: str = "reopen_en") -> VectorStore:
    return VectorStore(collection_name=name, persist_path=root)


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vector_store_module,
        "retry_with_backoff",
        lambda fn, retries=3, backoff_in_seconds=1.0: fn(),
    )


def test_read_failure_recovers_by_reopening_the_client(
    temp_chroma_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    old_system = store._client._system
    broken = BrokenReads(store._collection)
    store._collection = broken

    with caplog.at_level(logging.WARNING):
        hits = store.similarity_search("document", k=5)

    assert [h["id"] for h in hits] == ["c0"]
    assert broken.queries == 1
    assert store._client._system is not old_system  # a really new system, not the cached one
    assert "Reopened the Chroma client" in caplog.text


def test_other_stores_on_the_path_move_to_the_new_client(temp_chroma_dir: Path) -> None:
    english, german = _store(temp_chroma_dir), _store(temp_chroma_dir, "reopen_de")
    english.add_chunks([_chunk(0)])
    german.add_chunks([_chunk(1)])
    english._collection = BrokenReads(english._collection)
    english.similarity_search("document", k=5)

    assert [h["id"] for h in german.similarity_search("document", k=5)] == ["c1"]
    assert german._client._system is english._client._system

    german.add_chunks([_chunk(2)])  # both can still write and read through it
    assert german.count() == 2
    assert english.count() == 1


def test_stores_on_other_paths_are_left_alone(temp_chroma_dir: Path, tmp_path: Path) -> None:
    other = _store(tmp_path / "elsewhere")
    other.add_chunks([_chunk(5)])
    other_system = other._client._system
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    store._collection = BrokenReads(store._collection)
    store.similarity_search("document", k=5)

    assert [h["id"] for h in other.similarity_search("document", k=5)] == ["c5"]
    assert other._client._system is other_system


def test_new_stores_after_a_reopen_use_the_new_client(temp_chroma_dir: Path) -> None:
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    store._collection = BrokenReads(store._collection)
    store.similarity_search("document", k=5)

    newcomer = _store(temp_chroma_dir)
    assert newcomer._client._system is store._client._system
    assert newcomer.count() == 1


def test_a_store_that_stays_broken_does_not_reopen_on_every_query(temp_chroma_dir: Path) -> None:
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    store._collection = BrokenReads(store._collection)
    store.similarity_search("document", k=5)  # the one allowed reopen
    generation = store._client_generation
    system = store._client._system

    store._collection = BrokenReads(store._collection)
    for _ in range(5):
        with pytest.raises(InternalError, match="hnsw segment reader"):
            store.similarity_search("document", k=5)

    assert store._client_generation == generation
    assert store._client._system is system


def test_it_can_reopen_again_after_the_interval(
    temp_chroma_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    store._collection = BrokenReads(store._collection)
    store.similarity_search("document", k=5)

    monkeypatch.setattr(vector_store_module, "_REOPEN_MIN_INTERVAL_SECONDS", 0.0)
    store._collection = BrokenReads(store._collection)
    assert [h["id"] for h in store.similarity_search("document", k=5)] == ["c0"]


def test_other_internal_errors_are_not_treated_as_this_one(temp_chroma_dir: Path) -> None:
    store = _store(temp_chroma_dir)
    store.add_chunks([_chunk(0)])
    system = store._client._system
    store._collection = BrokenReads(store._collection, message="Something else went wrong")

    with pytest.raises(InternalError, match="Something else"):
        store.similarity_search("document", k=5)
    assert store._client._system is system


def test_server_mode_never_reopens(monkeypatch: pytest.MonkeyPatch) -> None:
    http_client = MagicMock()
    monkeypatch.setattr(chromadb, "HttpClient", http_client)
    store = VectorStore(collection_name="reopen_server", chroma_host="chroma", chroma_port=8000)
    broken = MagicMock()
    broken.query.side_effect = InternalError(READER_ERROR)
    store._collection = broken

    with pytest.raises(InternalError):
        store.similarity_search("document", k=5)
    assert store._client is http_client.return_value
    assert http_client.call_count == 1
