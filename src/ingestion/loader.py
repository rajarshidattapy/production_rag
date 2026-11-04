"""Document loader supporting text, markdown, and PDF files."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single loaded document with content and metadata."""

    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    doc_id: str = ""
    # Text of each physical page, in order, for paginated formats. Empty otherwise.
    pages: list[str] = field(default_factory=list)


class DocumentLoader:
    """Load documents from various file types.

    Supported formats: .txt, .md, .pdf, .json (as raw text).
    """

    SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf", ".json"}
    # The service's own bookkeeping in the data directory is not a document.
    EXCLUDED_NAMES: set[str] = {"ingest_jobs.json"}

    def __init__(
        self,
        base_path: Path | str | None = None,
        recursive: bool = True,
        max_file_size_mb: float = 50.0,
        max_pdf_pages: int = 1000,
    ) -> None:
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.recursive = recursive
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_pdf_pages = max_pdf_pages

    def list_files(self, source: Path | str) -> list[Path]:
        """Return the files ``source`` covers: itself, or a directory's supported files."""
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source does not exist: {source_path}")
        if source_path.is_file():
            return [source_path]

        pattern = "**/*" if self.recursive else "*"
        return [
            file_path
            for file_path in sorted(source_path.glob(pattern))
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and file_path.name not in self.EXCLUDED_NAMES
            and file_path.is_file()
        ]

    def load(self, source: Path | str) -> list[Document]:
        """Load a single file or all supported files from a directory."""
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source does not exist: {source_path}")

        if source_path.is_file():
            return [self._load_file(source_path)]
        return self._load_directory(source_path)

    def _load_directory(self, directory: Path) -> list[Document]:
        """Load all supported files from a directory."""
        docs: list[Document] = []
        for file_path in self.list_files(directory):
            try:
                docs.append(self._load_file(file_path))
            except Exception:
                logger.exception("Failed to load %s, skipping", file_path)
        logger.info("Loaded %d documents from %s", len(docs), directory)
        return docs

    def _load_file(self, file_path: Path) -> Document:
        """Load a single file based on its extension."""
        size = file_path.stat().st_size
        if size > self.max_file_size_bytes:
            raise ValueError(
                f"File {file_path} is {size / (1024 * 1024):.1f} MB, "
                f"exceeds max_file_size_mb limit of {self.max_file_size_bytes / (1024 * 1024):.1f} MB"
            )

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        return self._load_text(file_path)

    def _load_text(self, file_path: Path) -> Document:
        """Load a plain text or markdown file."""
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": file_path.suffix.lower(),
            },
            doc_id=str(file_path),
        )

    def _load_pdf(self, file_path: Path) -> Document:
        """Load a PDF file using pypdf."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            ) from None

        reader = pypdf.PdfReader(str(file_path))
        if len(reader.pages) > self.max_pdf_pages:
            raise ValueError(
                f"PDF {file_path} has {len(reader.pages)} pages, "
                f"exceeds max_pdf_pages limit of {self.max_pdf_pages}"
            )

        pages = [page.extract_text() or "" for page in reader.pages]

        content = "\n\n".join(pages)
        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": ".pdf",
                "pages": len(reader.pages),
            },
            doc_id=str(file_path),
            pages=pages,
        )

    def iter_docs(self, source: Path | str) -> Iterator[Document]:
        """Lazily yield documents from a source without materialising the full list.

        Unlike ``load()``, this generator processes one file at a time and is
        suitable for large directories where loading everything into memory
        upfront would be expensive.
        """
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source does not exist: {source_path}")

        if source_path.is_file():
            yield self._load_file(source_path)
            return

        for file_path in self.list_files(source_path):
            try:
                yield self._load_file(file_path)
            except Exception:
                logger.exception("Failed to load %s, skipping", file_path)
