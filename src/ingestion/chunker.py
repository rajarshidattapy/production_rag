"""Intelligent document chunking with metadata propagation."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk derived from a document."""

    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""


# Parent sections are the larger context a chunk came from, for retrieval to hand to the LLM.
# A section bigger than this is split, so a parent stays a useful size.
PARENT_MAX_CHARS = 4000

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_MARKDOWN_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_PARAGRAPH_BREAK_RE = re.compile(r"(\n\s*\n)")


def _compute_parent_id(doc_id: str, index: int) -> str:
    return hashlib.md5(f"{doc_id}::parent-{index}".encode()).hexdigest()


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown at headings outside code blocks. Titles are heading paths, "A > B"."""
    sections: list[tuple[str | None, str]] = []
    headings: list[tuple[int, str]] = []
    title: str | None = None
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _MARKDOWN_FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _MARKDOWN_HEADING_RE.match(line.rstrip("\r\n"))
        if heading:
            if "".join(lines).strip():
                sections.append((title, "".join(lines)))
            level = len(heading[1])
            headings = [h for h in headings if h[0] < level] + [(level, heading[2].strip())]
            title = " > ".join(name for _, name in headings)
            lines = []
        lines.append(line)
    sections.append((title, "".join(lines)))
    return sections


def _cap_section(text: str) -> list[str]:
    """Split a section bigger than ``PARENT_MAX_CHARS`` at paragraph breaks, or hard if needed."""
    if len(text) <= PARENT_MAX_CHARS:
        return [text]
    pieces: list[str] = []
    current = ""
    for part in _PARAGRAPH_BREAK_RE.split(text):
        if len(current) + len(part) > PARENT_MAX_CHARS and current.strip():
            pieces.append(current)
            current = ""
        while len(part) > PARENT_MAX_CHARS:
            pieces.append(part[:PARENT_MAX_CHARS])
            part = part[PARENT_MAX_CHARS:]
        current += part
    if current.strip():
        pieces.append(current)
    return pieces


def _sections(document: Document) -> list[tuple[int | None, str | None, str]]:
    """``(page, title, text)`` for each part of a document, before size capping."""
    if document.pages:
        return [(n, None, text) for n, text in enumerate(document.pages, start=1)]
    if document.metadata.get("filetype") == ".md":
        return [(None, title, text) for title, text in _markdown_sections(document.content)]
    return [(None, None, document.content)]


def _compute_chunk_id(content: str, doc_id: str, index: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    raw = f"{doc_id}::chunk-{index}::{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


class Chunker:
    """Splits documents into chunks with configurable overlap.

    Supports multiple splitting strategies:
      - "recursive": split on paragraph/sentence boundaries (default, recommended)
      - "fixed": fixed-size character chunks
      - "sentence": sentence-boundary aware chunks
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        strategy: str = "recursive",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        self._strategies: dict[str, Callable[[str], list[str]]] = {
            "recursive": self._recursive_split,
            "fixed": self._fixed_size_split,
            "sentence": self._sentence_split,
        }

        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Choose from: {list(self._strategies.keys())}"
            )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks. See ``chunk_with_parents``."""
        return self.chunk_with_parents(document)[1]

    def chunk_with_parents(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:
        """Split a document into parent sections, and each section into chunks.

        A parent is a PDF page, a Markdown heading's section, or the whole text, split
        further if bigger than ``PARENT_MAX_CHARS``. Chunks never cross a parent, so their
        ``page`` and ``section`` metadata is exact, and each records its ``parent_id``. No
        overlap is carried across a parent boundary.

        Returns ``(parents, chunks)``.
        """
        split = self._strategies[self.strategy]
        parents: list[Chunk] = []
        raw_chunks: list[tuple[Chunk, str]] = []
        for page, title, section in _sections(document):
            for piece in _cap_section(section):
                if not piece.strip():
                    continue
                metadata: dict[str, str | int | float] = {
                    **document.metadata,
                    "parent_index": len(parents),
                }
                if page is not None:
                    metadata["page"] = page
                if title:
                    metadata["section"] = title
                parent = Chunk(
                    content=piece.strip(),
                    metadata=metadata,
                    chunk_id=_compute_parent_id(document.doc_id, len(parents)),
                    doc_id=document.doc_id,
                )
                parents.append(parent)
                raw_chunks.extend((parent, text) for text in split(piece))

        pages = [
            p.metadata["page"] for p, text in raw_chunks if "page" in p.metadata and text.strip()
        ]
        last_page = max(pages, default=None)
        chunks: list[Chunk] = []

        for i, (parent, text) in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue

            metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "parent_id": parent.chunk_id,
            }
            for key in ("page", "section"):
                if key in parent.metadata:
                    metadata[key] = parent.metadata[key]
            if "page" in metadata:
                # The last page that has text: a blank back cover can't answer anything.
                metadata["is_last_page"] = metadata["page"] == last_page

            chunk = Chunk(
                content=text,
                metadata=metadata,
                chunk_id=_compute_chunk_id(text, document.doc_id, i),
                doc_id=document.doc_id,
            )
            chunks.append(chunk)

        logger.debug("Split doc %s into %d chunks", document.doc_id, len(chunks))
        return parents, chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple documents into chunks."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """Recursively split on paragraph then sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]

        # Try paragraph split first
        paragraphs = re.split(r"\n\s*\n", text)
        return self._merge_chunks(paragraphs)

    def _fixed_size_split(self, text: str) -> list[str]:
        """Fixed-size character-level splitting."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        """Split on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return self._merge_chunks(sentences)

    def _merge_chunks(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks respecting chunk_size and overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            if current_len + seg_len > self.chunk_size and current:
                chunks.append("".join(current))
                # Keep overlap trailing content
                overlap_text = ""
                overlap_chars = 0
                for s in reversed(current):
                    if overlap_chars + len(s) > self.chunk_overlap:
                        needed = self.chunk_overlap - overlap_chars
                        if needed > 0:
                            overlap_text = s[-needed:] + overlap_text
                        break
                    overlap_text = s + overlap_text
                    overlap_chars += len(s)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(seg)
            current_len += seg_len

        if current:
            chunks.append("".join(current))
        return chunks
