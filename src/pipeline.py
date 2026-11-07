"""Main RAG pipeline orchestrating ingestion → retrieval → reranking → generation."""

from __future__ import annotations

import contextvars
import gettext
import hashlib
import logging
import re
import time
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from src.healing.state_machine import HealingEngine, HealingObserver
    from src.healing.types import HealingResult
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

from src.config import settings
from src.generation.citations import Citation, CitationFormatter
from src.generation.generator import Generator
from src.ingestion import screening
from src.ingestion.chunker import Chunk, Chunker
from src.ingestion.loader import Document, DocumentLoader
from src.ingestion.metadata import (
    annotate_content_fingerprints,
    prepare_document_metadata,
    validate_chunks,
)
from src.retrieval.vector_store import VectorStore
from src.utils.page_query import page_filter

logger = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(["en", "de", "es"])

T = TypeVar("T")

# Owner of documents added without an account. Everyone can read them.
SHARED_OWNER = "shared"
# Whose documents retrieval may read during this request. The API always sets it. ``None``
# means no restriction, for trusted in-process callers such as the CLI and evaluation. A
# context variable reaches every retrieval path, including worker threads and healing retries.
retrieval_owners: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar(
    "retrieval_owners", default=None
)
# Leave out documents limited to roles. Set for callers that have no roles, such as MCP
# clients, until retrieval can check a caller's roles. Chunks stored before `restricted`
# was recorded are left out too.
retrieval_unrestricted_only: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "retrieval_unrestricted_only", default=False
)


# Browser uploads are saved as "<32 hex chars>_<original name>".
_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{32}_")


DEFAULT_COLLECTION = "default"


def _folder_rule(path: Path, data_dir: Path, rules: Mapping[str, T]) -> T | None:
    """The value for the longest configured folder ``path`` is in, if any."""
    try:
        parts = path.resolve().relative_to(data_dir.resolve()).parts
    except ValueError:
        return None
    best: T | None = None
    best_depth = -1
    for folder, value in rules.items():
        key = tuple(p for p in Path(folder).parts if p not in ("", "."))
        if parts[: len(key)] == key and len(key) > best_depth:
            best, best_depth = value, len(key)
    return best


def _source_authority(path: Path, data_dir: Path, rules: dict[str, int], default: int) -> int:
    """Authority of the longest configured folder ``path`` is in, or the default."""
    found = _folder_rule(path, data_dir, rules)
    return default if found is None else found


def _document_key(path: Path, data_dir: Path, owner: str) -> str:
    """What links versions of one document. Same owner and same name means same document.

    Uploads get a fresh path each time, so they are matched by their original filename.
    Files on the server are matched by their path inside the data directory.
    """
    try:
        relative = path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        relative = path.resolve()
    if relative.parts[:1] == ("uploads",):
        relative = Path("uploads") / _UPLOAD_PREFIX_RE.sub("", relative.name)
    return f"{owner}:{relative.as_posix()}"


def _within_visible_owners(where: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Narrow a metadata filter to the documents this request may read."""
    owners = retrieval_owners.get()
    scopes: list[dict[str, Any]] = []
    if owners is not None:
        scopes.append({"owner": {"$in": list(owners)}})
    if retrieval_unrestricted_only.get():
        scopes.append({"restricted": False})
    if where is not None:
        scopes.append(where)
    if not scopes:
        return None
    return scopes[0] if len(scopes) == 1 else {"$and": scopes}


def readable(metadata: Mapping[str, Any]) -> bool:
    """Whether this request may read a chunk or parent fetched by ID, by its metadata."""
    owners = retrieval_owners.get()
    if owners is not None and metadata.get("owner") not in owners:
        return False
    return not (retrieval_unrestricted_only.get() and metadata.get("restricted") is not False)


# Chunks embedded and stored per step of an ingest. It sets how soon new text is searchable,
# how often progress moves, and how long a request to stop can take to be noticed.
_INGEST_STEP_CHUNKS = 256
# A collection can have many thousands of files; a progress report only needs a sample.
_MAX_LISTED_FILES = 100


@dataclass
class IngestProgress:
    """Running totals for one ingest. ``files_done`` includes files that could not be read."""

    files_total: int = 0
    files_done: int = 0
    files_indexed: int = 0
    files_failed: int = 0
    chunks_ingested: int = 0
    stopped: bool = False
    indexed_names: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def add_indexed(self, name: str) -> None:
        self.files_indexed += 1
        if len(self.indexed_names) < _MAX_LISTED_FILES:
            self.indexed_names.append(name)

    def add_failure(self, name: str, error: str) -> None:
        self.files_failed += 1
        if len(self.failures) < _MAX_LISTED_FILES:
            self.failures.append({"file": name, "error": error})


def _detect_language(text: str) -> str:
    """Detect language of text, falling back to 'en' for unsupported languages."""
    from langdetect import DetectorFactory, LangDetectException, detect

    # Detection is randomized by default. Text has to route the same way every time, or
    # re-running an ingest would store a chunk again in a second language's collection.
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
    except LangDetectException:
        return "en"
    return lang if lang in _SUPPORTED_LANGUAGES else "en"


def _setup_translation(lang: str) -> contextvars.Token | None:
    """Install a gettext translation for ``lang`` if none is already active.

    Returns a context-var Token for cleanup (or None if the caller need not reset).
    """
    from src.utils.i18n import _current_translation, get_translation

    if isinstance(get_translation(), gettext.GNUTranslations):
        return None
    try:
        translation = gettext.translation(
            domain="messages",
            localedir=str(Path(__file__).parent / "translator"),
            languages=[lang],
            fallback=True,
        )
        return _current_translation.set(translation)
    except Exception:
        return None


class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents, retrieve context, generate answers.

    Supports automatic language routing to separate collections per language,
    and runtime localization of prompts and responses.
    """

    MAX_QUESTION_LENGTH = 2000

    def __init__(self, llm_provider: Literal["openai", "anthropic"] | None = None) -> None:
        self.config = settings

        # Ingestion
        self.loader = DocumentLoader(
            max_file_size_mb=self.config.max_file_size_mb,
            max_pdf_pages=self.config.max_pdf_pages,
        )
        self.chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy="recursive",
        )

        # Dictionary of vector stores and hybrid retrievers, mapped by language
        self._vector_stores: dict[str, VectorStore] = {}
        self._hybrid_retrievers: dict[str, HybridRetriever] = {}
        import threading

        self._lock = threading.RLock()

        # Generation
        self.generator = Generator(
            provider=llm_provider or self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self.citation_formatter = CitationFormatter()

        # Phase 2: Reranker (lazy-loaded)
        self._reranker: CrossEncoderReranker | None = None
        self._parent_store: VectorStore | None = None

        # Phase 4: Self-healing engine (lazy-loaded; only built when healing is used)
        self._healing_engine: HealingEngine | None = None
        self._healing_observer: HealingObserver | None = None

    @property
    def vector_store(self) -> VectorStore:
        """The Chroma-backed vector store (defaulting to English for backward compatibility)."""
        return self._get_vector_store("en")

    def _open_store(self, collection_name: str) -> VectorStore:
        return VectorStore(
            collection_name=collection_name,
            persist_path=self.config.chroma_path,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            chroma_host=self.config.chroma_host,
            chroma_port=self.config.chroma_port,
            embedding_query_cache_size=self.config.embedding_query_cache_size,
            redis_url=self.config.redis_url,
            redis_key_prefix=self.config.redis_key_prefix,
            redis_socket_timeout_seconds=self.config.redis_socket_timeout_seconds,
            redis_embedding_cache_ttl_seconds=self.config.redis_embedding_cache_ttl_seconds,
            hnsw_ef_search=self.config.hnsw_ef_search,
            hnsw_max_neighbors=self.config.hnsw_max_neighbors,
            hnsw_ef_construction=self.config.hnsw_ef_construction,
            ingest_batch_size=self.config.ingest_batch_size,
        )

    def _get_parent_store(self) -> VectorStore:
        """Parent sections, fetched by ID. One collection for every language."""
        with self._lock:
            if self._parent_store is None:
                self._parent_store = self._open_store("rag_parents")
            return self._parent_store

    def get_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch parent sections by ID, in the order asked. Callers must check ``owner``."""
        return self._get_parent_store().get_chunks(parent_ids)

    def _get_vector_store(self, lang: str) -> VectorStore:
        """Construct and retrieve the vector store for a specific language."""
        with self._lock:
            if lang not in self._vector_stores:
                collection_name = f"rag_docs_{lang}"
                self._vector_stores[lang] = self._open_store(collection_name)
            return self._vector_stores[lang]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: Path | str,
        on_progress: Callable[[IngestProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        owner: str | None = None,
    ) -> int:
        """Load, chunk, and index documents from a file or directory.

        Automatically detects document chunk language and routes to the matching collection.

        Works through a directory one file at a time, holding only a small buffer of chunks,
        so the collection's text is never all in memory and text is searchable as soon as its
        batch is stored. A file that can't be read is recorded and skipped, except when it
        is the only file requested, where the error is raised. Chunks already stored are
        skipped, so running an interrupted ingest again resumes it instead of redoing the work.

        Args:
            source: Path to a file or directory.
            on_progress: Called with running totals as the work advances.
            should_stop: Checked between batches. When true, chunks stored so far stay.
            owner: The user these documents belong to. Without one they are shared.

        Returns:
            Number of chunks newly added.
        """
        files = self.loader.list_files(source)
        root = Path(source).resolve()
        progress = IngestProgress(files_total=len(files))
        stop = should_stop or (lambda: False)
        report = (lambda: on_progress(progress)) if on_progress else (lambda: None)
        pending: dict[str, list[Chunk]] = {}
        pending_parents: list[Chunk] = []
        # Exact-content fingerprints of chunks retained in this ingest. This catches repeated
        # files before the first batch has reached Chroma.
        seen_fingerprints: set[tuple[str, str]] = set()
        screened: list[tuple[str, str, str]] = []
        ingested_at = time.time()
        report()

        for path in files:
            if stop():
                progress.stopped = True
                break
            name = path.name if path == root else str(path.relative_to(root))
            try:
                document = self.loader._load_file(path)
                self._record_provenance(document, path, owner, ingested_at)
                self._screen(document, name)
                screened.append(
                    (
                        str(document.metadata["document_key"]),
                        str(document.metadata["version_id"]),
                        str(document.metadata["screening"]),
                    )
                )
                parents, chunks = self.chunker.chunk_with_parents(document)
                parents, chunks, excluded = self._flag_and_exclude(parents, chunks, name)
                annotate_content_fingerprints([*parents, *chunks])
                chunk_languages = {id(chunk): _detect_language(chunk.content) for chunk in chunks}
                chunks = self._deduplicate_chunks(
                    chunks, seen_fingerprints, chunk_languages=chunk_languages
                )
                kept_parent_ids = {str(chunk.metadata["parent_id"]) for chunk in chunks}
                parents = [parent for parent in parents if parent.chunk_id in kept_parent_ids]
            except Exception as exc:
                if path == root:
                    raise
                logger.exception("Failed to load %s, skipping", path)
                progress.add_failure(name, str(exc))
            else:
                if not chunks:
                    progress.add_failure(
                        name,
                        "All of its text looked like instructions to the model, so none was indexed."
                        if excluded
                        else "No extractable text (it may be a scanned image).",
                    )
                pending_parents.extend(parents)
                for chunk in chunks:
                    pending.setdefault(chunk_languages[id(chunk)], []).append(chunk)
                self._store_pending(
                    pending, pending_parents, progress, report, _INGEST_STEP_CHUNKS, stop
                )
                if progress.stopped:
                    # Stopped partway through a long file: it will be finished by a re-run.
                    pending.clear()
                    break
                if chunks:
                    progress.add_indexed(name)
            progress.files_done += 1
            report()

        self._store_pending(pending, pending_parents, progress, report, 1)
        self._remove_other_screenings(screened)
        logger.info("Ingested %d chunks from %s", progress.chunks_ingested, source)
        return progress.chunks_ingested

    def _record_provenance(
        self, document: Document, path: Path, owner: str | None, ingested_at: float
    ) -> None:
        """Owner, version, timestamps, authority, and access, copied onto every chunk and parent.

        The version is part of each chunk's ID, so an edited file is stored as a new version
        beside the old one instead of being mistaken for chunks that are already stored.
        """
        meta = document.metadata
        meta["owner"] = owner or SHARED_OWNER
        meta["ingested_at"] = ingested_at
        meta["document_key"] = _document_key(path, self.config.data_dir, str(meta["owner"]))
        meta["authority"] = _source_authority(
            path,
            self.config.data_dir,
            self.config.source_authority,
            self.config.default_source_authority,
        )
        # Who may read it beyond its owner. `restricted` is explicit, because a filter can't
        # match a list that isn't there.
        meta["collection"] = (
            _folder_rule(path, self.config.data_dir, self.config.source_collections)
            or DEFAULT_COLLECTION
        )
        roles = sorted(
            set(_folder_rule(path, self.config.data_dir, self.config.source_roles) or [])
        )
        meta["restricted"] = bool(roles)
        if roles:
            meta["roles"] = roles
        version = str(meta.get("content_hash", ""))[:16]
        meta["version_id"] = version
        # Two users can add the same file and must each get their own copy.
        prefix = f"{owner}/" if owner else ""
        document.doc_id = f"{prefix}{document.doc_id}@{version}"

    def _screen(self, document: Document, name: str) -> None:
        """Strip hidden text and chat markup, and redact what the settings say to, in place."""
        cfg = self.config

        def clean(text: str) -> tuple[str, Counter[str]]:
            return screening.redact(
                screening.neutralize(text),
                credentials=cfg.redact_credentials,
                pii=cfg.pii_mode == "redact",
            )

        found: Counter[str] = Counter()
        if document.pages:
            cleaned = [clean(page) for page in document.pages]
            document.pages = [text for text, _ in cleaned]
            document.content = "\n\n".join(document.pages)
            for _, counts in cleaned:
                found.update(counts)
        else:
            document.content, found = clean(document.content)
        trusted = cfg.injection_trusted_authority
        document.metadata["screening"] = (
            f"rules={screening.RULES_VERSION};"
            f"credentials={'redact' if cfg.redact_credentials else 'keep'};"
            f"pii={cfg.pii_mode};injection={cfg.injection_mode};"
            f"trusted={'none' if trusted is None else trusted}"
        )
        # Part of every chunk ID, so changed settings produce new chunks when a file is
        # ingested again; the copies screened the old way are removed once those are stored.
        fingerprint = hashlib.md5(str(document.metadata["screening"]).encode()).hexdigest()[:8]
        document.doc_id = f"{document.doc_id}~{fingerprint}"
        prepare_document_metadata(document)
        if found:
            document.metadata["redacted"] = sorted(found)
            logger.info("Redacted from %s: %s", name, dict(found))

    def _remove_other_screenings(self, screened: list[tuple[str, str, str]]) -> int:
        """Delete these files' chunks and parents that were screened with other settings.

        Runs after the new copies are stored, so a file is never left with nothing. Chunks
        from before screening was recorded have no ``screening`` and are left alone.
        """
        stores: list[tuple[str | None, VectorStore]] = [
            (lang, self._get_vector_store(lang)) for lang in sorted(_SUPPORTED_LANGUAGES)
        ]
        stores.append((None, self._get_parent_store()))
        removed = 0
        for start in range(0, len(screened), 100):
            clauses: list[dict[str, Any]] = [
                {
                    "$and": [
                        {"document_key": key},
                        {"version_id": version},
                        {"screening": {"$ne": settings_used}},
                    ]
                }
                for key, version, settings_used in screened[start : start + 100]
            ]
            where = clauses[0] if len(clauses) == 1 else {"$or": clauses}
            for lang, store in stores:
                ids = sorted(store.ids_matching(where))
                for batch in range(0, len(ids), 500):
                    store.delete_chunks(ids[batch : batch + 500])
                if ids:
                    removed += len(ids)
                    if lang is not None and lang in self._hybrid_retrievers:
                        self._hybrid_retrievers[lang].invalidate_index()
        if removed:
            logger.info("Removed %d chunks and parents screened with other settings", removed)
        return removed

    def _flag_and_exclude(
        self, parents: list[Chunk], chunks: list[Chunk], name: str
    ) -> tuple[list[Chunk], list[Chunk], int]:
        """Record injection signals and personal data per chunk and parent; maybe exclude.

        Exclusion works on whole parent sections, because a phrase can span two chunks and
        the parent is what parent-child retrieval would hand to the model.
        """
        flag_pii = self.config.pii_mode == "flag"
        trusted = self.config.injection_trusted_authority
        for record in (*parents, *chunks):
            signals = screening.injection_signals(record.content)
            authority = record.metadata.get("authority")
            exempt = (
                bool(signals)
                and trusted is not None
                and isinstance(authority, int)
                and authority >= trusted
            )
            record.metadata["injection_suspected"] = bool(signals) and not exempt
            if exempt:
                record.metadata["injection_exempt"] = True
            if signals:
                record.metadata["injection_signals"] = signals
            if flag_pii:
                kinds = screening.pii_types(record.content)
                record.metadata["pii_found"] = bool(kinds)
                if kinds:
                    record.metadata["pii_types"] = kinds
        if self.config.injection_mode != "exclude":
            return parents, chunks, 0
        blocked = {p.chunk_id for p in parents if p.metadata["injection_suspected"]}
        kept = [c for c in chunks if c.metadata["parent_id"] not in blocked]
        excluded = len(chunks) - len(kept)
        if excluded:
            logger.warning(
                "Left %d of %d chunks of %s out of the index: they looked like instructions "
                "to the model",
                excluded,
                len(chunks),
                name,
            )
        return [p for p in parents if p.chunk_id not in blocked], kept, excluded

    def _store_pending(
        self,
        pending: dict[str, list[Chunk]],
        parents: list[Chunk],
        progress: IngestProgress,
        report: Callable[[], None],
        at_least: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Store buffered chunks one step at a time while a language has ``at_least`` waiting.

        Parents are stored first, so a stored chunk's parent can always be fetched.
        """
        if parents and any(len(chunks) >= at_least for chunks in pending.values()):
            store = self._get_parent_store()
            known = store.existing_ids([p.chunk_id for p in parents])
            fresh = [p for p in parents if p.chunk_id not in known]
            if fresh:
                validate_chunks(fresh)
                store.add_unsearchable(fresh)
            parents.clear()
        for lang, chunks in pending.items():
            while len(chunks) >= at_least:
                step = chunks[:_INGEST_STEP_CHUNKS]
                del chunks[:_INGEST_STEP_CHUNKS]
                progress.chunks_ingested += self._store_chunks(lang, step)
                report()
                if should_stop is not None and should_stop():
                    progress.stopped = True
                    return

    def _store_chunks(self, lang: str, chunks: list[Chunk]) -> int:
        vs = self._get_vector_store(lang)
        known = vs.existing_ids([c.chunk_id for c in chunks])
        fresh = [c for c in chunks if c.chunk_id not in known]
        if not fresh:
            return 0
        validate_chunks(fresh)
        count = vs.add_chunks(fresh)
        # Keep the corresponding hybrid index current (appends; falls back to a rebuild)
        if lang in self._hybrid_retrievers:
            self._hybrid_retrievers[lang].add_chunks(fresh)
        return count

    def _deduplicate_chunks(
        self,
        chunks: list[Chunk],
        seen_fingerprints: set[tuple[str, str]],
        *,
        chunk_languages: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Remove exact and near-duplicate chunks from other documents before indexing.

        Exact duplicates are cheap metadata lookups. Near duplicates query the existing vector
        collection in a batch, preserving chunks from the same logical document so chunk overlap
        and a new document version cannot make a document incomplete.
        """
        grouped: dict[tuple[str, str], list[Chunk]] = {}
        retained: list[Chunk] = []
        exact_removed = 0
        for chunk in chunks:
            fingerprint = str(chunk.metadata["content_fingerprint"])
            owner = str(chunk.metadata["owner"])
            identity = (owner, fingerprint)
            if identity in seen_fingerprints:
                exact_removed += 1
                continue
            seen_fingerprints.add(identity)
            language = (
                chunk_languages[id(chunk)]
                if chunk_languages is not None
                else _detect_language(chunk.content)
            )
            grouped.setdefault((language, owner), []).append(chunk)

        for (language, owner), candidates in grouped.items():
            store = self._get_vector_store(language)
            fingerprints = [str(chunk.metadata["content_fingerprint"]) for chunk in candidates]
            existing_ids = store.ids_matching(
                {"$and": [{"owner": owner}, {"content_fingerprint": {"$in": fingerprints}}]}
            )
            existing = store.get_chunks(sorted(existing_ids)) if existing_ids else []
            documents_by_fingerprint: dict[str, set[str]] = {}
            for row in existing:
                metadata = row["metadata"]
                fingerprint = str(metadata.get("content_fingerprint"))
                documents_by_fingerprint.setdefault(fingerprint, set()).add(
                    str(metadata.get("document_key"))
                )
            fresh = []
            for chunk in candidates:
                fingerprint = str(chunk.metadata["content_fingerprint"])
                matching_documents = documents_by_fingerprint.get(fingerprint, set())
                # Keep a same-document retry: deterministic chunk IDs make _store_chunks skip
                # its existing records, while ingestion progress still counts the file. Only a
                # copy from another logical document is a duplicate to suppress here.
                if (
                    matching_documents
                    and str(chunk.metadata.get("document_key")) not in matching_documents
                ):
                    continue
                fresh.append(chunk)
            exact_removed += len(candidates) - len(fresh)
            if not fresh or not self.config.semantic_deduplication:
                retained.extend(fresh)
                continue

            matches = store.near_duplicate_matches(
                fresh, self.config.semantic_deduplication_threshold, where={"owner": owner}
            )
            for chunk in fresh:
                match = matches.get(chunk.chunk_id)
                matched_metadata = (match or {}).get("metadata", {})
                matched_document = matched_metadata.get("document_key")
                if (
                    match is not None
                    and matched_metadata.get("owner") == owner
                    and matched_document != chunk.metadata.get("document_key")
                ):
                    logger.info(
                        "Skipped near-duplicate chunk %s (%.3f similar to %s)",
                        chunk.chunk_id,
                        match["score"],
                        match["id"],
                    )
                    continue
                retained.append(chunk)
        if exact_removed:
            logger.info("Skipped %d exact duplicate chunks", exact_removed)
        return retained

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        self_heal: bool | None = None,
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG pipeline.

        Args:
            question: The user's query string.
            top_k: Number of final context chunks (default: from config).
            use_hybrid: Enable BM25 + vector hybrid search (Phase 2).
            use_reranker: Enable cross-encoder re-ranking (Phase 2).
            self_heal: Run the verify/repair loop (Phase 4). ``None`` uses
                ``config.self_healing_enabled``. Use ``query_with_healing`` to
                also get the healing metadata.

        Returns:
            Tuple of (answer_text, list_of_citations).

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        if self._healing_requested(self_heal):
            result = self.query_with_healing(
                question, top_k=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker
            )
            return result.answer, result.citations

        from src.utils.i18n import _, _current_translation

        lang = _detect_language(question)
        token = _setup_translation(lang)

        try:
            question = question.strip()
            if not question:
                raise ValueError(_("Question must not be empty."))
            if len(question) > self.MAX_QUESTION_LENGTH:
                raise ValueError(
                    _(
                        "Question is too long ({len_question} chars). "
                        "Maximum allowed is {max_len} characters."
                    ).format(len_question=len(question), max_len=self.MAX_QUESTION_LENGTH)
                )

            k, where = self._page_scope(question, top_k)

            contexts = self._retrieve(
                question,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                k=k,
                lang=lang,
                where=where,
            )

            if not contexts:
                logger.warning("No relevant context found for query: %s", question)
                return (
                    _(
                        "I could not find any relevant information in the knowledge base to answer your question."
                    ),
                    [],
                )

            if use_reranker:
                contexts = self._apply_reranker(question, contexts, top_k=k)

            contexts_for_generation = self._apply_context_budget(contexts)
            answer = self.generator.generate(question, contexts_for_generation)
            citations = self.citation_formatter.build_citations(contexts)

            logger.info("Answered query in %d context chunks", len(contexts))
            return answer, citations
        finally:
            if token is not None:
                _current_translation.reset(token)

    async def query_async(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        self_heal: bool | None = None,
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG pipeline asynchronously.

        Args:
            question: The user's query string.
            top_k: Number of final context chunks (default: from config).
            use_hybrid: Enable BM25 + vector hybrid search (Phase 2).
            use_reranker: Enable cross-encoder re-ranking (Phase 2).
            self_heal: Run the verify/repair loop (Phase 4). ``None`` uses
                ``config.self_healing_enabled``.

        Returns:
            Tuple of (answer_text, list_of_citations).

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        if self._healing_requested(self_heal):
            result = await self.query_with_healing_async(
                question, top_k=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker
            )
            return result.answer, result.citations

        import asyncio

        from src.utils.i18n import _, _current_translation

        lang = _detect_language(question)
        token = _setup_translation(lang)

        try:
            question = question.strip()
            if not question:
                raise ValueError(_("Question must not be empty."))
            if len(question) > self.MAX_QUESTION_LENGTH:
                raise ValueError(
                    _(
                        "Question is too long ({len_question} chars). "
                        "Maximum allowed is {max_len} characters."
                    ).format(len_question=len(question), max_len=self.MAX_QUESTION_LENGTH)
                )

            k, where = await asyncio.to_thread(self._page_scope, question, top_k)

            contexts = await asyncio.to_thread(
                self._retrieve,
                question,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                k=k,
                lang=lang,
                where=where,
            )

            if not contexts:
                logger.warning("No relevant context found for query: %s", question)
                return (
                    _(
                        "I could not find any relevant information in the knowledge base to answer your question."
                    ),
                    [],
                )

            if use_reranker:
                contexts = await asyncio.to_thread(
                    self._apply_reranker, question, contexts, top_k=k
                )

            contexts_for_generation = self._apply_context_budget(contexts)
            answer = await self.generator.generate_async(question, contexts_for_generation)
            citations = self.citation_formatter.build_citations(contexts)

            logger.info("Answered query in %d context chunks (async)", len(contexts))
            return answer, citations
        finally:
            if token is not None:
                _current_translation.reset(token)

    # ------------------------------------------------------------------
    # Self-healing query (Phase 4)
    # ------------------------------------------------------------------

    def _healing_requested(self, self_heal: bool | None) -> bool:
        return self.config.self_healing_enabled if self_heal is None else self_heal

    def _validate_question(self, question: str) -> str:
        from src.utils.i18n import _

        question = question.strip()
        if not question:
            raise ValueError(_("Question must not be empty."))
        if len(question) > self.MAX_QUESTION_LENGTH:
            raise ValueError(
                _(
                    "Question is too long ({len_question} chars). "
                    "Maximum allowed is {max_len} characters."
                ).format(len_question=len(question), max_len=self.MAX_QUESTION_LENGTH)
            )
        return question

    def _get_healing_engine(self) -> HealingEngine:
        with self._lock:
            if self._healing_engine is None:
                from src.healing.integration import build_healing_engine

                self._healing_engine = build_healing_engine(self, observer=self._healing_observer)
            return self._healing_engine

    def set_healing_observer(self, observer: HealingObserver | None) -> None:
        """Attach an observer that receives one span per healing step (used by monitoring)."""
        with self._lock:
            self._healing_observer = observer
            if self._healing_engine is not None:
                from src.healing.state_machine import NullObserver

                self._healing_engine.observer = observer or NullObserver()

    def query_with_healing(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> HealingResult:
        """Answer with the verify → diagnose → repair → re-verify loop, abstaining if it can't.

        Returns:
            A ``HealingResult`` with the answer, citations, healing metadata
            (``.info``) and a per-attempt audit trail (``.history``).

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        from src.utils.i18n import _, _current_translation

        lang = _detect_language(question)
        token = _setup_translation(lang)
        try:
            question = self._validate_question(question)
            k, where = self._page_scope(question, top_k)
            result = self._get_healing_engine().run(
                question,
                lang=lang,
                top_k=k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                where=where,
            )
            if result.abstained:
                result.answer = _(result.answer)
            logger.info(
                "Healing query finished: status=%s attempts=%d",
                result.info.status,
                result.info.attempts,
            )
            return result
        finally:
            if token is not None:
                _current_translation.reset(token)

    async def query_with_healing_async(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> HealingResult:
        """Async wrapper around ``query_with_healing`` (runs the loop in a worker thread)."""
        import asyncio

        return await asyncio.to_thread(
            self.query_with_healing,
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

    async def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        run_blocking: Callable[..., Awaitable[Any]] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Answer without self-healing, yielding ``(kind, value)`` events as they happen.

        Events are ``("contexts", passages)``, then ``("token", text)`` for each generated
        chunk, then ``("citations", citations)``. ``run_blocking`` runs the blocking
        retrieval and reranking steps; it defaults to ``asyncio.to_thread``.
        """
        import asyncio

        run = run_blocking or asyncio.to_thread
        k, where = await run(self._page_scope, question, top_k)
        contexts = await run(
            self._retrieve,
            question,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            k=k,
            lang=_detect_language(question),
            where=where,
        )
        if not contexts:
            yield ("contexts", [])
            yield (
                "token",
                "I could not find any relevant information in the knowledge "
                "base to answer your question.",
            )
            yield ("citations", [])
            return

        if use_reranker:
            contexts = await run(self._apply_reranker, question, contexts, top_k=k)
        contexts = self._apply_context_budget(contexts)
        yield ("contexts", contexts)

        async for chunk in self.generator.generate_stream(question, contexts):
            yield ("token", chunk)

        yield ("citations", self.citation_formatter.build_citations(contexts))

    def search(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the passages ``query()`` would answer from, best first, without generating.

        ``filename`` keeps the search to one document.

        Raises:
            ValueError: If the question is empty or exceeds the maximum length.
        """
        question = self._validate_question(question)
        k, where = self._page_scope(question, top_k)
        if filename:
            document = {"filename": filename}
            where = document if where is None else {"$and": [where, document]}
        contexts = self._retrieve(
            question,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            k=k,
            lang=_detect_language(question),
            where=where,
        )
        if use_reranker and contexts:
            return self._apply_reranker(question, contexts, top_k=k)
        # A filtered search returns its hits in reading order.
        return sorted(contexts, key=lambda c: c.get("score", 0.0), reverse=True)[:k]

    def _page_scope(self, question: str, top_k: int | None) -> tuple[int, dict[str, Any] | None]:
        """Return the chunk count and page filter to retrieve with for ``question``.

        A page question wants the page, not its ``top_k`` best-matching fragments, so it
        gets at least ``top_k_retrieval`` chunks; the context budget still caps the prompt.
        Corpora with no page metadata (text files, PDFs ingested before pages were
        recorded) get plain retrieval, since the reference can't be resolved there and
        may not even mean a document page.
        """
        k = top_k or self.config.top_k_final
        where = page_filter(question)
        paged = _within_visible_owners({"page": {"$gte": 1}})
        if where is None or not any(
            self._get_vector_store(lang).has_paged_chunks(paged)
            for lang in sorted(_SUPPORTED_LANGUAGES)
        ):
            return k, None
        return max(k, self.config.top_k_retrieval), where

    def _retrieve_pages(
        self,
        query: str,
        where: dict[str, Any],
        use_hybrid: bool,
        fetch_k: int,
        alpha: float | None,
    ) -> list[dict[str, Any]]:
        """Retrieve from the requested pages, returning the best ``fetch_k`` in reading order.

        Searches every language collection: chunks are routed by language one at a time, so
        one page can span collections, and a page question says little about the language
        of the page it is asking about.
        """
        hits: list[dict[str, Any]] = []
        for lang in sorted(_SUPPORTED_LANGUAGES):
            if use_hybrid:
                hits.extend(
                    self._get_hybrid_retriever(lang).search(
                        query, k=fetch_k, where=where, alpha=alpha
                    )
                )
            else:
                hits.extend(
                    self._get_vector_store(lang).similarity_search(query, k=fetch_k, where=where)
                )
        best = sorted(hits, key=lambda hit: hit["score"], reverse=True)[:fetch_k]
        return sorted(
            best,
            key=lambda hit: (
                str(hit["metadata"].get("source", "")),
                hit["metadata"].get("page", 0),
                hit["metadata"].get("chunk_index", 0),
            ),
        )

    def _retrieve(
        self,
        query: str,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        k: int = 5,
        lang: str = "en",
        fetch_k: int | None = None,
        alpha: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context chunks.

        ``fetch_k`` and ``alpha`` are optional overrides used by self-healing
        retries (wider candidate pools, re-balanced hybrid fusion); the
        defaults reproduce the original behaviour exactly. ``where`` restricts
        retrieval to specific document pages.
        """
        if fetch_k is None:
            fetch_k = self.config.top_k_retrieval if use_reranker else k

        if where is not None:
            scoped = _within_visible_owners(where)
            assert scoped is not None
            return self._retrieve_pages(query, scoped, use_hybrid, fetch_k, alpha)

        visible = _within_visible_owners()
        options: dict[str, Any] = {} if visible is None else {"where": visible}
        if use_hybrid:
            if alpha is not None:
                options["alpha"] = alpha
            return self._get_hybrid_retriever(lang).search(query, k=fetch_k, **options)
        return self._get_vector_store(lang).similarity_search(query, k=fetch_k, **options)

    def _apply_context_budget(
        self, contexts: list[dict[str, Any]], budget: int | None = None
    ) -> list[dict[str, Any]]:
        """Trim retrieved contexts so their combined text stays within the character budget.

        ``budget`` defaults to ``config.max_context_chars``; self-healing passes
        a larger one when a retry needs more context.
        """
        if budget is None:
            budget = self.config.max_context_chars
        if budget <= 0:
            return contexts

        result: list[dict[str, Any]] = []
        used = 0
        for ctx in contexts:
            doc = ctx.get("document", "")
            remaining = budget - used
            if remaining <= 0:
                break
            if len(doc) > remaining:
                ctx = {**ctx, "document": doc[:remaining]}
                result.append(ctx)
                break
            result.append(ctx)
            used += len(doc)

        return result

    def _apply_reranker(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank retrieved contexts using cross-encoder."""
        return self._get_reranker().rerank(
            query,
            contexts,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Lazy-loaded components (Phase 2)
    # ------------------------------------------------------------------

    def _get_hybrid_retriever(self, lang: str = "en") -> HybridRetriever:
        with self._lock:
            if lang not in self._hybrid_retrievers:
                from src.retrieval.hybrid import HybridRetriever

                self._hybrid_retrievers[lang] = HybridRetriever(
                    vector_store=self._get_vector_store(lang),
                    alpha=self.config.hybrid_alpha,
                    rrf_k=self.config.rrf_k,
                    max_index_mb=self.config.bm25_max_index_mb,
                )
                retriever = self._hybrid_retrievers[lang]
                # Reuse the persisted BM25 corpus when present; rebuild only if missing/unloadable.
                if not retriever._load_index():
                    retriever.build_index()
            return self._hybrid_retrievers[lang]

    def _get_reranker(self) -> CrossEncoderReranker:
        with self._lock:
            if self._reranker is None:
                from src.retrieval.reranker import CrossEncoderReranker

                self._reranker = CrossEncoderReranker(
                    model_name=self.config.reranker_model,
                )
            return self._reranker

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Find a chunk by ID. An ID doesn't say which language collection holds it."""
        for lang in sorted(_SUPPORTED_LANGUAGES):
            found = self._get_vector_store(lang).get_chunks([chunk_id])
            if found:
                return found[0]
        return None

    def reset(self) -> None:
        """Clear the vector store and all cached components."""
        # Reset default vector store
        self.vector_store.delete_collection()

        # Reset all other initialized language vector stores
        for lang, vs in list(self._vector_stores.items()):
            if lang != "en":
                vs.delete_collection()

        self._get_parent_store().delete_collection()

        self._vector_stores.clear()
        self._hybrid_retrievers.clear()
        self._parent_store = None
        self._reranker = None
        logger.info("Pipeline reset: vector store cleared")

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics, totalled across every language collection."""
        stores = [self._get_vector_store(lang) for lang in sorted(_SUPPORTED_LANGUAGES)]
        cache_stats = [store.embedding_cache_stats() for store in stores]
        return {
            "chunks_in_store": sum(store.count() for store in stores),
            "embedding_model": self.config.embedding_model,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_cache_hits": sum(c["hits"] for c in cache_stats),
            "embedding_cache_misses": sum(c["misses"] for c in cache_stats),
        }
