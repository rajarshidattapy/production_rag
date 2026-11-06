"""The persisted metadata contract for searchable chunks and parent sections."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from src.ingestion.loader import MetadataValue

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk
    from src.ingestion.loader import Document


METADATA_SCHEMA_VERSION = 1
_WHITESPACE_RE = re.compile(r"\s+")


class MetadataValidationError(ValueError):
    """Raised when a document or chunk cannot safely be stored."""


def prepare_document_metadata(document: Document) -> None:
    """Add invariant fields and reject malformed document metadata.

    Optional location fields (``page`` and ``section``) are added later by the
    chunker.  Required fields are deliberately small and Chroma-compatible so
    every searchable record has a stable identity, owner, and source reference.
    """
    metadata = document.metadata
    source = metadata.get("source")
    owner = metadata.get("owner")
    if not isinstance(source, str) or not source.strip():
        raise MetadataValidationError("document metadata requires a non-empty 'source'")
    if not isinstance(owner, str) or not owner.strip():
        raise MetadataValidationError("document metadata requires a non-empty 'owner'")
    if not document.doc_id:
        raise MetadataValidationError("document requires a non-empty doc_id")

    metadata["metadata_schema_version"] = METADATA_SCHEMA_VERSION
    metadata["document_id"] = document.doc_id
    metadata["source"] = source.strip()
    metadata["owner"] = owner.strip()


def validate_chunk_metadata(chunk: Chunk) -> None:
    """Validate the metadata copied onto one chunk or parent section."""
    metadata = chunk.metadata
    required: dict[str, type[MetadataValue] | tuple[type[MetadataValue], ...]] = {
        "metadata_schema_version": int,
        "document_id": str,
        "source": str,
        "owner": str,
    }
    for field, expected in required.items():
        value = metadata.get(field)
        if not isinstance(value, expected) or (isinstance(value, str) and not value.strip()):
            raise MetadataValidationError(
                f"chunk {chunk.chunk_id!r} has invalid '{field}' metadata"
            )
    if metadata["metadata_schema_version"] != METADATA_SCHEMA_VERSION:
        raise MetadataValidationError(
            f"chunk {chunk.chunk_id!r} uses unsupported metadata schema "
            f"{metadata['metadata_schema_version']!r}"
        )
    if metadata["document_id"] != chunk.doc_id:
        raise MetadataValidationError(f"chunk {chunk.chunk_id!r} document_id does not match doc_id")

    page = metadata.get("page")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        raise MetadataValidationError(f"chunk {chunk.chunk_id!r} has invalid 'page' metadata")
    section = metadata.get("section")
    if section is not None and (not isinstance(section, str) or not section.strip()):
        raise MetadataValidationError(f"chunk {chunk.chunk_id!r} has invalid 'section' metadata")


def validate_chunks(chunks: list[Chunk]) -> None:
    """Validate a batch immediately before it is persisted."""
    for chunk in chunks:
        validate_chunk_metadata(chunk)


def annotate_content_fingerprints(chunks: list[Chunk]) -> None:
    """Store a stable exact-content fingerprint used by deduplication."""
    for chunk in chunks:
        normalized = _WHITESPACE_RE.sub(" ", chunk.content).strip().casefold()
        chunk.metadata["content_fingerprint"] = hashlib.sha256(normalized.encode()).hexdigest()
