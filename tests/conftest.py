"""Shared test fixtures for RAG pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.chunker import Chunker
from src.ingestion.loader import Document


def _raise_windows_open_file_limit() -> None:
    """Embedded Chroma keeps four C runtime file streams open per index, and the suite creates
    hundreds of temporary stores in one process. Windows allows 512 by default, after which
    anything else that opens a file fails (Chroma fixes this in an unreleased version, #7869).
    Production runs on Linux and keeps only a few collections open, so this is a test-only limit.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.cdll.ucrtbase._setmaxstdio(8192)
    except (OSError, AttributeError):
        pass


_raise_windows_open_file_limit()


@pytest.fixture(autouse=True)
def isolate_chroma_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force all tests to use an isolated ChromaDB directory, avoiding locks.

    Also sets data_dir to tmp_path so the API path-confinement check passes
    for test fixtures that create documents under tmp_path.
    """
    from src.config import settings

    monkeypatch.setattr(settings, "chroma_path", tmp_path / "test_chroma_rag")
    # A developer's .env may point at a real Chroma server. Tests must never touch it.
    monkeypatch.setattr(settings, "chroma_host", None)
    monkeypatch.setattr(settings, "chroma_port", None)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "allow_store_reset", True)
    # Loading real models at every app start would make the suite slow; warm-up has its tests.
    monkeypatch.setattr(settings, "warm_up_on_start", False)


@pytest.fixture(autouse=True)
def fresh_login_limits() -> None:
    """Every test client shares one address, so attempts would otherwise add up across tests."""
    api = sys.modules.get("src.api")
    if api is not None:
        api._auth_limiter.clear()


@pytest.fixture(autouse=True)
def local_ingest_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a real Kafka broker from a developer's .env, and undo an earlier shutdown."""
    from src.config import settings

    monkeypatch.setattr(settings, "ingest_queue", "local")
    api = sys.modules.get("src.api")
    if api is not None:
        from src.ingest_queue import LocalIngestQueue

        if not isinstance(api._ingest_queue, LocalIngestQueue):
            monkeypatch.setattr(api, "_ingest_queue", LocalIngestQueue(api._run_queued_ingest_job))
        # A test client used as a context manager runs the shutdown hook when it exits.
        api._ingest_shutting_down.clear()


@pytest.fixture(autouse=True)
def no_real_follow_up_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a real LLM from a developer's .env. Tests of rewriting patch this themselves."""
    from src.generation.generator import Generator

    async def unchanged(self: Generator, question: str, history: list[dict]) -> str:
        return question

    monkeypatch.setattr(Generator, "condense_question", unchanged)


@pytest.fixture(autouse=True)
def disable_monitoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the API on the plain pipeline, whatever MONITOR_ENABLED is in a developer's .env."""
    from src.monitoring.config import settings as monitoring_settings

    monkeypatch.setattr(monitoring_settings, "enabled", False)


@pytest.fixture(autouse=True)
def mock_embedding_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the embedding function to return dummy vectors, avoiding loading SentenceTransformer."""
    import hashlib

    from src.retrieval.vector_store import _ChromaEmbeddingFunction

    def dummy_encode(self, input_texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input_texts:
            # Deterministic vector based on text hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vector = []
            for i in range(384):
                byte_val = h[i % len(h)]
                val = ((byte_val * (i + 1)) % 256) / 256.0
                vector.append(val)
            embeddings.append(vector)
        return embeddings

    monkeypatch.setattr(_ChromaEmbeddingFunction, "_encode", dummy_encode)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return a test configuration with small values for fast tests."""
    return Settings(
        chunk_size=200,
        chunk_overlap=20,
        chroma_path=tmp_path / "test_chroma_rag",
        chroma_host=None,
        chroma_port=None,
        top_k_final=3,
        top_k_retrieval=5,
    )


@pytest.fixture
def sample_text() -> str:
    """Return a multi-paragraph sample text for chunking tests."""
    return (
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
        "information retrieval with text generation.\n\n"
        "It retrieves relevant document chunks from a knowledge base and provides "
        "them as context to a large language model.\n\n"
        "This enables the model to generate answers grounded in factual information "
        "rather than relying solely on its training data.\n\n"
        "RAG is widely used in enterprise applications such as customer support "
        "chatbots and internal knowledge base assistants."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    """Return a Document fixture."""
    return Document(
        content=sample_text,
        metadata={"source": "test.txt", "filename": "test.txt", "filetype": ".txt"},
        doc_id="test_doc_1",
    )


@pytest.fixture
def chunker() -> Chunker:
    """Return a chunker with small chunk size for testing."""
    return Chunker(chunk_size=200, chunk_overlap=20, strategy="recursive")


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for ChromaDB storage."""
    return tmp_path / "chroma_test"


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample text files for ingestion tests."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "doc1.txt").write_text(
        "RAG combines retrieval with generation for accurate answers."
    )
    (docs_dir / "doc2.txt").write_text(
        "Hybrid search merges BM25 keyword search with vector similarity using RRF."
    )
    return docs_dir
