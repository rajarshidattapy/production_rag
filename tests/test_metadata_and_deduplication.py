"""Metadata-contract and ingestion-deduplication regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.ingestion.chunker import Chunk, Chunker
from src.ingestion.loader import Document
from src.ingestion.metadata import (
    MetadataValidationError,
    annotate_content_fingerprints,
    prepare_document_metadata,
    validate_chunks,
)
from src.pipeline import RAGPipeline


def _document() -> Document:
    return Document(
        content="A short, valid document.",
        metadata={"source": "handbook.txt", "owner": "shared"},
        doc_id="handbook-v1",
    )


def test_chunk_metadata_has_a_stable_contract() -> None:
    document = _document()
    prepare_document_metadata(document)
    chunks = Chunker().chunk(document)

    validate_chunks(chunks)
    metadata = chunks[0].metadata
    assert metadata["metadata_schema_version"] == 1
    assert metadata["document_id"] == "handbook-v1"
    assert metadata["source"] == "handbook.txt"
    assert metadata["owner"] == "shared"


def test_invalid_chunk_metadata_is_rejected() -> None:
    with pytest.raises(MetadataValidationError, match="source"):
        validate_chunks(
            [
                Chunk(
                    content="text",
                    chunk_id="chunk-1",
                    doc_id="document-1",
                    metadata={
                        "metadata_schema_version": 1,
                        "document_id": "document-1",
                        "source": "",
                        "owner": "shared",
                    },
                )
            ]
        )


def test_ingest_skips_exact_content_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "first.txt").write_text(
        "The company handbook says employees receive 25 days leave."
    )
    (tmp_path / "copy.txt").write_text("The company handbook says employees receive 25 days leave.")
    pipeline = RAGPipeline()
    monkeypatch.setattr(pipeline.config, "semantic_deduplication", False)

    assert pipeline.ingest(tmp_path) == 1
    assert pipeline.stats()["chunks_in_store"] == 1


def test_near_duplicate_from_another_document_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = RAGPipeline()
    document = _document()
    prepare_document_metadata(document)
    chunks = Chunker().chunk(document)
    chunk = chunks[0]
    chunk.metadata["document_key"] = "shared:handbook.txt"
    annotate_content_fingerprints(chunks)

    class Store:
        def ids_matching(self, where: dict[str, Any]) -> set[str]:
            return set()

        def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
            return []

        def near_duplicate_matches(
            self, candidates: list[Chunk], threshold: float, where: dict[str, Any] | None = None
        ) -> dict[str, dict[str, Any]]:
            return {
                candidates[0].chunk_id: {
                    "id": "other-chunk",
                    "metadata": {"document_key": "shared:duplicate.txt", "owner": "shared"},
                    "score": threshold,
                }
            }

    monkeypatch.setattr(pipeline, "_get_vector_store", lambda language: Store())
    assert pipeline._deduplicate_chunks(chunks, set()) == []
