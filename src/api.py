"""FastAPI application exposing the RAG pipeline as a long-lived HTTP service.

Running this app keeps the embedding model and Chroma client warm in memory,
avoiding the per-CLI-invocation cold start incurred by scripts/*.py.

Run locally with:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.chat.rate_limit import AttemptLimiter
from src.chat.store import AccountLockedError, ChatStore, User, UsernameTakenError
from src.config import settings as _settings
from src.generation.citations import CitationFormatter
from src.healing.metrics import HEALING_REGISTRY
from src.pipeline import SHARED_OWNER, IngestProgress, RAGPipeline, retrieval_owners
from src.retrieval.vector_store import VectorStoreUnavailableError
from src.utils.i18n import _

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------
_RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()


def _check_api_key(authorization: str | None = Header(None)) -> None:
    """Dependency that enforces Bearer token auth when RAG_API_KEY is set."""
    if not _RAG_API_KEY:
        return  # Auth disabled — no key configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization[len("Bearer ") :]
    # Constant-time comparison — a naive `!=` short-circuits on the first
    # mismatched byte, leaking a timing side-channel that lets a remote
    # attacker recover the key byte-by-byte across many requests.
    if not secrets.compare_digest(token, _RAG_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key.")


async def setup_locale(accept_language: str | None = Header(None)):
    import gettext

    from src.utils.i18n import _current_translation

    lang = "en"
    if accept_language:
        parts = [p.split(";")[0].split("-")[0].strip().lower() for p in accept_language.split(",")]
        for p in parts:
            if p in ["de", "es", "en"]:
                lang = p
                break

    try:
        translation = gettext.translation(
            domain="messages",
            localedir=str(Path(__file__).parent / "translator"),
            languages=[lang],
            fallback=True,
        )
    except Exception:
        translation = gettext.NullTranslations()

    token = _current_translation.set(translation)
    try:
        yield
    finally:
        _current_translation.reset(token)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    _interrupt_active_ingest_jobs()


app = FastAPI(
    lifespan=_lifespan,
    title="Production RAG API",
    version="1.1.0",
    description="HTTP service layer for the Production-Grade RAG pipeline.",
    dependencies=[Depends(setup_locale)],
)

# Applied per-route (not app-wide) so /healthz stays reachable without a
# key — required for Docker/Kubernetes liveness probes, which never send
# an Authorization header.
_auth = Depends(_check_api_key)
_UI_PATH = Path(__file__).resolve().parents[1] / "index.html"


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or generate a unique X-Request-ID for every request.

    Enables distributed trace correlation across Langfuse, OTel, and logs.
    The client may supply its own ID; we forward it unchanged, or generate
    a UUID4 if absent.
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


app.add_middleware(_RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Prometheus metrics — pull-based scrape endpoint, independent of the
# push-based OTel/Langfuse pipeline in src/monitoring/. A dedicated
# registry (rather than the global default) keeps this isolated and
# import-order-safe under repeated test-suite app construction.
# ---------------------------------------------------------------------------
_metrics_registry = CollectorRegistry()
_http_requests_total = Counter(
    "rag_http_requests_total",
    "Total HTTP requests handled, by path and status code",
    ["path", "method", "status_code"],
    registry=_metrics_registry,
)
_http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds, by path",
    ["path", "method"],
    registry=_metrics_registry,
)
# A stream has already answered 200 by the time it fails, so rag_http_requests_total can't
# show these. Subtract them from its 200s to get the streams that really succeeded.
_http_stream_errors_total = Counter(
    "rag_http_stream_errors_total",
    "Streaming responses that failed after they started, by path and reason",
    ["path", "reason"],
    registry=_metrics_registry,
)
# Requests that match no route share this label. Raw URLs would create one permanent
# time series per URL, which anyone probing random paths could grow without limit.
_UNMATCHED_PATH = "unmatched"


class _PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every request, keyed by route.

    Uses the matched route template (e.g. ``/query``) rather than the raw
    URL so the label cardinality stays bounded regardless of query params.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled, so the server is about to answer 500. Without this the error
            # rate would miss exactly the failures it exists to show.
            self._record(request, 500, start)
            raise
        self._record(request, response.status_code, start)
        return response

    @staticmethod
    def _record(request: Request, status_code: int, start: float) -> None:
        route = request.scope.get("route")
        path = getattr(route, "path", _UNMATCHED_PATH)
        _http_requests_total.labels(path=path, method=request.method, status_code=status_code).inc()
        _http_request_duration_seconds.labels(path=path, method=request.method).observe(
            time.perf_counter() - start
        )


app.add_middleware(_PrometheusMiddleware)

_cors_origins_raw = os.environ.get("RAG_CORS_ORIGINS", "*")
_cors_origins: list[str] = (
    ["*"]
    if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Accept-Language", "Content-Type"],
)


@app.exception_handler(VectorStoreUnavailableError)
async def vector_store_unavailable_handler(request: Request, exc: VectorStoreUnavailableError):
    logger.warning("Vector store unavailable: %s", exc)
    return Response(
        status_code=503,
        content='{"detail": "The vector database is unavailable."}',
        media_type="application/json",
    )


# --- OpenAI Exception Handlers ---
try:
    import openai

    @app.exception_handler(openai.RateLimitError)
    async def openai_rate_limit_handler(request: Request, exc: openai.RateLimitError):
        return Response(
            status_code=429,
            content='{"detail": "Rate limit exceeded on upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(openai.APIConnectionError)
    async def openai_connection_handler(request: Request, exc: openai.APIConnectionError):
        return Response(
            status_code=503,
            content='{"detail": "Failed to connect to upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(openai.APIStatusError)
    async def openai_status_handler(request: Request, exc: openai.APIStatusError):
        status = 502
        if exc.status_code == 429:
            status = 429
        return Response(
            status_code=status,
            content=f'{{"detail": "Upstream LLM provider returned error status: {exc.status_code}."}}',
            media_type="application/json",
        )
except ImportError:
    pass

# --- Anthropic Exception Handlers ---
try:
    import anthropic

    @app.exception_handler(anthropic.RateLimitError)
    async def anthropic_rate_limit_handler(request: Request, exc: anthropic.RateLimitError):
        return Response(
            status_code=429,
            content='{"detail": "Rate limit exceeded on upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(anthropic.APIConnectionError)
    async def anthropic_connection_handler(request: Request, exc: anthropic.APIConnectionError):
        return Response(
            status_code=503,
            content='{"detail": "Failed to connect to upstream LLM provider API."}',
            media_type="application/json",
        )

    @app.exception_handler(anthropic.APIStatusError)
    async def anthropic_status_handler(request: Request, exc: anthropic.APIStatusError):
        status = 502
        if exc.status_code == 429:
            status = 429
        return Response(
            status_code=status,
            content=f'{{"detail": "Upstream LLM provider returned error status: {exc.status_code}."}}',
            media_type="application/json",
        )
except ImportError:
    pass

# Module-level singleton, constructed lazily on first use so that /healthz
# does not force-load the embedding model or Chroma client.
_pipeline: Any = None


def _build_pipeline() -> Any:
    """Build the pipeline, instrumented with OTel and Langfuse when MONITOR_ENABLED=true."""
    pipeline = RAGPipeline()

    from src.monitoring.config import settings as monitoring_settings

    if not monitoring_settings.enabled:
        return pipeline

    # Imported only when enabled: src.monitoring.wrappers patches the LLM clients on import.
    from src.monitoring import MetricsCollector, MonitoredRAGPipeline, Tracer

    logger.info("Monitoring enabled: wrapping the pipeline with MonitoredRAGPipeline")
    return MonitoredRAGPipeline(
        pipeline, tracer=Tracer(enabled=True), metrics=MetricsCollector(enabled=True)
    )


def get_pipeline() -> Any:
    """Return the shared pipeline instance, constructing it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def _query_timeout() -> float | None:
    timeout = _settings.query_timeout_seconds
    return timeout if timeout > 0 else None


def reset_pipeline() -> None:
    """Drop the cached pipeline singleton so it is rebuilt on next access.

    Primarily useful for tests, where fixtures patch configuration (e.g.
    ``settings.chroma_path``) after this module has already been imported.
    """
    global _pipeline
    _pipeline = None


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    chunks_in_store: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    embedding_cache_hits: int
    embedding_cache_misses: int


class IngestRequest(BaseModel):
    source: str = Field(..., description="Path to a file or directory to ingest.")
    reset: bool = Field(False, description="If true, clear the vector store before ingesting.")


class IngestResponse(BaseModel):
    chunks_ingested: int
    total_chunks: int


class AsyncIngestRequest(IngestRequest):
    label: str | None = Field(None, max_length=200, description="Name to show for this job.")
    chat_id: str | None = Field(
        None, description="Your chat to note the result in when the job ends. Requires signing in."
    )


class UploadResponse(BaseModel):
    filename: str
    source: str
    size_bytes: int
    directory: str


class IngestJobResponse(BaseModel):
    job_id: str
    status: str


class IngestFailure(BaseModel):
    file: str
    error: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    chunks_ingested: int | None = None
    total_chunks: int | None = None
    error: str | None = None
    label: str | None = None
    chat_id: str | None = None
    created_at: float | None = None
    files_total: int | None = None
    files_done: int | None = None
    files_failed: int | None = None
    failures: list[IngestFailure] = []
    cancel_requested: bool = False


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    use_hybrid: bool = False
    use_reranker: bool = False
    self_heal: bool | None = Field(
        None,
        description=(
            "Run the self-healing verify/repair loop. Omit to use the server default "
            "(RAG_SELF_HEALING_ENABLED)."
        ),
    )


class CitationResponse(BaseModel):
    chunk_id: str
    source: str
    filename: str
    text_snippet: str
    score: float
    page: int | None = None


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str


class HealingResponse(BaseModel):
    status: str
    attempts: int
    recovered: bool
    reason: str | None = None
    failure_types: list[str] = []
    query_rewrites: list[str] = []
    final_faithfulness: float | None = None
    final_relevance: float | None = None
    citation_valid: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    # Present only when self-healing ran; omitted otherwise so existing
    # clients see byte-identical responses.
    healing: HealingResponse | None = None


def _healing_requested(request: QueryRequest) -> bool:
    return _settings.self_healing_enabled if request.self_heal is None else request.self_heal


def _healing_payload(info: Any) -> dict[str, Any]:
    return info.model_dump(mode="json")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Lightweight liveness check that does not load heavy models."""
    return HealthResponse(status="ok")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the lightweight local RAG user interface."""
    return FileResponse(_UI_PATH)


@app.get("/readyz", response_model=HealthResponse, dependencies=[_auth])
def readyz() -> HealthResponse:
    """Readiness probe checking database access and eager-loading models."""
    try:
        # 1. Eagerly import heavy dependencies
        import sentence_transformers  # noqa: F401

        # 2. Warm up components for all supported languages
        pipeline = get_pipeline()
        for lang in ["en", "de", "es"]:
            # Objects built by an earlier call are cached, so ask the database directly.
            pipeline._get_vector_store(lang).heartbeat()
            _ = pipeline._get_hybrid_retriever(lang)
        _ = pipeline._get_reranker()

        return HealthResponse(status="ok")
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Service not ready.",
        ) from exc


@app.get("/stats", response_model=StatsResponse, dependencies=[_auth])
def stats() -> dict[str, Any]:
    """Return pipeline statistics, constructing the pipeline if needed."""
    pipeline = get_pipeline()
    return pipeline.stats()


@app.get("/metrics", dependencies=[_auth])
def metrics() -> Response:
    """Prometheus scrape endpoint: HTTP request counts/latency plus rag_healing_* metrics."""
    return Response(
        content=generate_latest(_metrics_registry) + generate_latest(HEALING_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


def _refuse_reset_unless_allowed(reset: bool) -> None:
    """``reset`` deletes every user's documents, so it stays off unless an operator enables it."""
    if reset and not _settings.allow_store_reset:
        raise HTTPException(
            status_code=403,
            detail="Clearing the document store is turned off (RAG_ALLOW_STORE_RESET).",
        )


def _visible_owners(rag_session: str | None) -> tuple[str, ...]:
    """Documents a request may read: shared ones, plus the signed-in user's own."""
    user = _user_for_session(rag_session)
    return (SHARED_OWNER, user.id) if user else (SHARED_OWNER,)


def _resolve_ingest_source(source: str) -> Path:
    """Validate and resolve an ingest source path, confined to the data directory.

    Shared by both the synchronous and async-job ingest endpoints so the
    path-traversal guard can't drift between them. Raises HTTPException
    (400) on any validation failure.
    """
    try:
        source_path = Path(source).resolve(strict=False)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    # Confine ingest paths to the configured data directory to prevent path-traversal.
    allowed_root = Path(_settings.data_dir).resolve()
    try:
        source_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Path must be inside the configured data directory ({allowed_root}).",
        ) from None

    # The account database lives in the data directory too. A single file loads as text
    # whatever its extension, which would make password hashes and private chats searchable.
    chat_db = Path(_settings.chat_db_file).resolve()
    if source_path.parent == chat_db.parent and source_path.name.startswith(chat_db.name):
        raise HTTPException(status_code=400, detail="This path cannot be ingested.")

    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=_("Source path does not exist: {source_path}").format(source_path=source),
        )

    return source_path


@app.post("/upload", response_model=UploadResponse, dependencies=[_auth])
async def upload_pdf(
    request: Request,
    filename: str = Query(..., min_length=1),
    batch: str | None = Query(None, pattern=r"^[0-9a-f]{32}$"),
) -> UploadResponse:
    """Save a browser-uploaded PDF under the configured data directory.

    Files are intentionally stored inside ``data/uploads`` so the existing
    ingestion guard remains the single authority for allowed source paths.
    The request body is raw PDF bytes, avoiding an extra multipart dependency.
    Files sent with the same ``batch`` share a directory, which one ingest job can then cover.
    """
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be uploaded.")

    max_bytes = int(_settings.max_file_size_mb * 1024 * 1024)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {_settings.max_file_size_mb:g} MB upload limit.",
        )

    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {_settings.max_file_size_mb:g} MB upload limit.",
        )

    upload_dir = Path(_settings.data_dir).resolve() / "uploads"
    if batch:
        upload_dir = upload_dir / batch
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
    await asyncio.to_thread(destination.write_bytes, content)
    return UploadResponse(
        filename=safe_name,
        source=str(destination),
        size_bytes=len(content),
        directory=str(upload_dir),
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[_auth])
async def ingest(request: IngestRequest, rag_session: str | None = Cookie(None)) -> IngestResponse:
    """Ingest documents from a file or directory into the vector store.

    Blocks for the full duration of ingestion. For large corpora that risk
    client/proxy timeouts, use ``POST /ingest/async`` instead. Documents added while signed
    in belong to that user; otherwise they are shared with everyone.
    """
    _refuse_reset_unless_allowed(request.reset)
    source_path = _resolve_ingest_source(request.source)
    pipeline = get_pipeline()
    user = await asyncio.to_thread(_user_for_session, rag_session)

    if request.reset:
        await asyncio.to_thread(pipeline.reset)

    try:
        chunks_ingested = await asyncio.to_thread(
            pipeline.ingest, source_path, owner=user.id if user else None
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = await asyncio.to_thread(pipeline.stats)
    total_chunks = stats["chunks_in_store"]
    return IngestResponse(chunks_ingested=chunks_ingested, total_chunks=total_chunks)


# ---------------------------------------------------------------------------
# Async ingestion — job-ID + polling, for corpora large enough to risk a
# client/proxy timeout on the synchronous /ingest endpoint above.
# ---------------------------------------------------------------------------
_INGEST_JOBS_MAX = 500
_ingest_jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
# One at a time, in order: indexing is CPU-bound, so parallel jobs would only slow each other,
# and a job with reset=True must not run underneath another job.
_ingest_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-ingest")
# Stop signal for each job this process hasn't finished yet.
_active_ingest_jobs: dict[str, threading.Event] = {}
_ingest_jobs_shut_down: set[str] = set()
# Identifies this process, so a job left unfinished by a previous one can be recognised.
_INSTANCE_ID = uuid.uuid4().hex
_ingest_jobs_file_lock = threading.Lock()
_ingest_jobs_lock = threading.RLock()
# Rewriting the status file on every batch would add up on a long job.
_INGEST_PROGRESS_SAVE_SECONDS = 2.0
_RESUMABLE_JOB_STATUSES = {"cancelled", "interrupted", "failed"}
_UNFINISHED_JOB_STATUSES = {"pending", "running"}
# Past this many documents a chat records a count, not every filename.
_MAX_CHAT_FILE_NAMES = 20
_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{32}_")


def _ingest_jobs_path() -> Path:
    return Path(_settings.data_dir) / "ingest_jobs.json"


def _read_persisted_ingest_jobs() -> dict[str, dict[str, Any]]:
    try:
        with open(_ingest_jobs_path(), encoding="utf-8") as f:
            jobs = json.load(f)
    except (OSError, ValueError):
        return {}
    return jobs if isinstance(jobs, dict) else {}


def _persist_ingest_job(job_id: str, record: dict[str, Any]) -> None:
    """Save job status so it survives a restart. Best effort: never fails the job itself."""
    path = _ingest_jobs_path()
    tmp = path.with_name(f"{path.name}.{_INSTANCE_ID}.tmp")
    try:
        with _ingest_jobs_file_lock:
            jobs = _read_persisted_ingest_jobs()
            jobs.pop(job_id, None)
            jobs[job_id] = record
            for old_id in list(jobs)[: max(0, len(jobs) - _INGEST_JOBS_MAX)]:
                del jobs[old_id]
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(jobs, f)
            os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not save ingest job status: %s", exc)


def _record_ingest_job(job_id: str, *, persist: bool = True, **fields: Any) -> None:
    with _ingest_jobs_lock:
        _ingest_jobs[job_id] = {**_ingest_jobs.get(job_id, {}), **fields, "owner": _INSTANCE_ID}
        _ingest_jobs.move_to_end(job_id)
        while len(_ingest_jobs) > _INGEST_JOBS_MAX:
            _ingest_jobs.popitem(last=False)
        record = dict(_ingest_jobs[job_id])
    if persist:
        _persist_ingest_job(job_id, record)


def _find_ingest_job(job_id: str) -> dict[str, Any] | None:
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
    return job or _read_persisted_ingest_jobs().get(job_id)


def _ingest_job_view(job_id: str, job: dict[str, Any]) -> IngestJobStatusResponse:
    """A job as clients see it: internal fields dropped, and orphaned work called what it is."""
    if job.get("status") in _UNFINISHED_JOB_STATUSES and job.get("owner") != _INSTANCE_ID:
        job = {
            **job,
            "status": "interrupted",
            "error": "The server restarted before this job finished.",
        }
    return IngestJobStatusResponse(
        job_id=job_id, **{k: v for k, v in job.items() if k not in ("owner", "user_id", "source")}
    )


def _progress_fields(progress: IngestProgress) -> dict[str, Any]:
    return {
        "files_total": progress.files_total,
        "files_done": progress.files_done,
        "files_failed": progress.files_failed,
        "chunks_ingested": progress.chunks_ingested,
        "failures": progress.failures,
    }


def _ingest_chat_note(
    status: str, progress: IngestProgress, label: str | None
) -> tuple[list[str], str]:
    """Return the documents to list on the chat and the note to add to its history."""
    count = progress.files_indexed
    names = sorted(_UPLOAD_PREFIX_RE.sub("", name) for name in progress.indexed_names)
    if count > _MAX_CHAT_FILE_NAMES:
        files, added = [f"{count} documents"], f"{count} documents"
    else:
        files, added = names, ", ".join(names)

    parts = [f"Added {added} to this chat."] if count else []
    if progress.files_failed:
        parts.append(f"{progress.files_failed} of {progress.files_total} could not be indexed.")
    if status == "cancelled":
        parts.append(f"Indexing was cancelled after {progress.files_done} files.")
    elif status == "failed":
        parts.append(f"Indexing stopped with an error after {progress.files_done} files.")
    return files, " ".join(parts) or f"No documents were found in {label or 'that location'}."


def _note_ingest_result_in_chat(job_id: str, status: str, progress: IngestProgress) -> None:
    """Tell the chat how the job ended, so the result is there even if the tab was closed."""
    with _ingest_jobs_lock:
        job = dict(_ingest_jobs.get(job_id, {}))
    if not job.get("chat_id") or status == "interrupted":
        return  # an interrupted job is expected to be run again
    try:
        files, note = _ingest_chat_note(status, progress, job.get("label"))
        get_chat_store().add_files(
            job["user_id"],
            job["chat_id"],
            files,
            note=note,
            replacing=job.get("replaces_chat_files"),
        )
        _record_ingest_job(job_id, persist=False, chat_files=files)
    except Exception:
        logger.exception("Could not note ingest job %s in its chat", job_id)


def _run_ingest_job(job_id: str, source_path: Path, reset: bool, owner: str | None) -> None:
    stop = _active_ingest_jobs[job_id]
    latest = IngestProgress()
    last_saved = 0.0

    def on_progress(progress: IngestProgress) -> None:
        nonlocal latest, last_saved
        latest = progress
        now = time.monotonic()
        save = now - last_saved >= _INGEST_PROGRESS_SAVE_SECONDS
        if save:
            last_saved = now
        _record_ingest_job(job_id, persist=save, **_progress_fields(progress))

    def stopped_status() -> str:
        return "interrupted" if job_id in _ingest_jobs_shut_down else "cancelled"

    try:
        if stop.is_set():  # cancelled while waiting its turn
            status = stopped_status()
            _note_ingest_result_in_chat(job_id, status, latest)
            _record_ingest_job(job_id, status=status)
            return

        _record_ingest_job(job_id, status="running")
        pipeline = get_pipeline()
        if reset:
            pipeline.reset()
        chunks_ingested = pipeline.ingest(
            source_path, on_progress=on_progress, should_stop=stop.is_set, owner=owner
        )
        total_chunks = pipeline.stats()["chunks_in_store"]
        status = stopped_status() if latest.stopped else "completed"
        # The note goes first: whoever sees the job end expects to find it in the chat.
        _note_ingest_result_in_chat(job_id, status, latest)
        _record_ingest_job(
            job_id,
            status=status,
            total_chunks=total_chunks,
            **{**_progress_fields(latest), "chunks_ingested": chunks_ingested},
        )
    except Exception as exc:
        logger.warning("Ingest job %s failed: %s", job_id, exc)
        _note_ingest_result_in_chat(job_id, "failed", latest)
        _record_ingest_job(job_id, status="failed", error=str(exc), **_progress_fields(latest))
    finally:
        _active_ingest_jobs.pop(job_id, None)
        _ingest_jobs_shut_down.discard(job_id)


def _enqueue_ingest_job(source_path: Path, reset: bool = False, **fields: Any) -> str:
    job_id = str(uuid.uuid4())
    _active_ingest_jobs[job_id] = threading.Event()
    _record_ingest_job(
        job_id, status="pending", source=str(source_path), created_at=time.time(), **fields
    )
    _ingest_worker.submit(_run_ingest_job, job_id, source_path, reset, fields.get("user_id"))
    return job_id


def _interrupt_active_ingest_jobs() -> None:
    """Called at shutdown: stop promptly, so exit doesn't wait for a long job to finish."""
    for job_id, stop in list(_active_ingest_jobs.items()):
        _ingest_jobs_shut_down.add(job_id)
        stop.set()


def _user_for_session(rag_session: str | None) -> User | None:
    return get_chat_store().user_for_session(rag_session) if rag_session else None


def _job_for_requester(job_id: str, rag_session: str | None) -> dict[str, Any]:
    """Find a job this caller may act on. A signed-in user's job belongs to that user alone."""
    job = _find_ingest_job(job_id)
    if job is not None and job.get("user_id"):
        user = _user_for_session(rag_session)
        if user is None or user.id != job["user_id"]:
            job = None
    if job is None:
        raise HTTPException(status_code=404, detail="No ingest job found with that ID.")
    return job


@app.post("/ingest/async", response_model=IngestJobResponse, status_code=202, dependencies=[_auth])
async def ingest_async(
    request: AsyncIngestRequest, rag_session: str | None = Cookie(None)
) -> IngestJobResponse:
    """Enqueue ingestion as a background job and return immediately.

    Poll ``GET /ingest/jobs/{job_id}`` for status. Path validation happens
    synchronously before the job is created, so a bad path still fails fast
    with a 400 rather than surfacing as an async job failure. Jobs run one at a
    time in the order they arrive. Text is searchable as each batch is stored.
    """
    _refuse_reset_unless_allowed(request.reset)
    source_path = _resolve_ingest_source(request.source)

    user = await asyncio.to_thread(_user_for_session, rag_session)
    if request.chat_id:
        if user is None:
            raise HTTPException(status_code=401, detail="Please sign in.")
        store = get_chat_store()
        if not await asyncio.to_thread(store.chat_exists, user.id, request.chat_id):
            raise HTTPException(status_code=404, detail="No chat found with that ID.")

    job_id = await asyncio.to_thread(
        _enqueue_ingest_job,
        source_path,
        reset=request.reset,
        label=request.label or source_path.name,
        user_id=user.id if user else None,
        chat_id=request.chat_id,
    )
    return IngestJobResponse(job_id=job_id, status="pending")


@app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse, dependencies=[_auth])
def get_ingest_job(job_id: str, rag_session: str | None = Cookie(None)) -> IngestJobStatusResponse:
    """Return the current status of an async ingestion job."""
    return _ingest_job_view(job_id, _job_for_requester(job_id, rag_session))


@app.post(
    "/ingest/jobs/{job_id}/cancel",
    response_model=IngestJobStatusResponse,
    dependencies=[_auth],
)
def cancel_ingest_job(
    job_id: str, rag_session: str | None = Cookie(None)
) -> IngestJobStatusResponse:
    """Stop a job after its current batch. Text already indexed stays searchable."""
    job = _job_for_requester(job_id, rag_session)
    stop = _active_ingest_jobs.get(job_id)
    if stop is None or job.get("status") not in _UNFINISHED_JOB_STATUSES:
        raise HTTPException(status_code=409, detail="This job is not running on this server.")
    stop.set()
    _record_ingest_job(job_id, cancel_requested=True)
    return _ingest_job_view(job_id, {**job, "cancel_requested": True})


@app.post(
    "/ingest/jobs/{job_id}/retry",
    response_model=IngestJobResponse,
    status_code=202,
    dependencies=[_auth],
)
def retry_ingest_job(job_id: str, rag_session: str | None = Cookie(None)) -> IngestJobResponse:
    """Run a job that didn't finish again. Text already indexed is skipped, not embedded twice."""
    job = _job_for_requester(job_id, rag_session)
    if _ingest_job_view(job_id, job).status not in _RESUMABLE_JOB_STATUSES:
        raise HTTPException(status_code=409, detail="Only a job that didn't finish can be rerun.")
    new_id = _enqueue_ingest_job(
        _resolve_ingest_source(job["source"]),
        label=job.get("label"),
        user_id=job.get("user_id"),
        chat_id=job.get("chat_id"),
        # A re-run covers the whole source again, so its result replaces the earlier one. An
        # interrupted attempt recorded nothing, so what it was going to replace passes along.
        replaces_chat_files=job.get("chat_files") or job.get("replaces_chat_files"),
    )
    return IngestJobResponse(job_id=new_id, status="pending")


@app.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
    dependencies=[_auth],
)
async def query(
    request: QueryRequest, response: Response, rag_session: str | None = Cookie(None)
) -> QueryResponse:
    """Answer a question using the RAG pipeline.

    With self-healing enabled (per request or server default), the answer is
    verified and repaired before being returned, or the service abstains; the
    ``healing`` field then describes what happened.
    """
    from src.utils.usage import UsageTracker, request_usage

    tracker = UsageTracker()
    token = request_usage.set(tracker)
    owners_token = retrieval_owners.set(await asyncio.to_thread(_visible_owners, rag_session))

    try:
        pipeline = get_pipeline()
        healing: HealingResponse | None = None

        try:
            if _healing_requested(request):
                result = await asyncio.wait_for(
                    pipeline.query_with_healing_async(
                        request.question,
                        top_k=request.top_k,
                        use_hybrid=request.use_hybrid,
                        use_reranker=request.use_reranker,
                    ),
                    _query_timeout(),
                )
                answer, citations = result.answer, result.citations
                healing = HealingResponse(**_healing_payload(result.info))
                response.headers["X-RAG-Healing-Status"] = result.info.status
                response.headers["X-RAG-Healing-Attempts"] = str(result.info.attempts)
            else:
                answer, citations = await asyncio.wait_for(
                    pipeline.query_async(
                        request.question,
                        top_k=request.top_k,
                        use_hybrid=request.use_hybrid,
                        use_reranker=request.use_reranker,
                    ),
                    _query_timeout(),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="The query timed out.") from exc

        citation_responses = [
            CitationResponse(
                chunk_id=c["chunk_id"],
                source=c["source"],
                filename=c["filename"],
                text_snippet=c["text_snippet"],
                score=c["score"],
                page=c.get("page"),
            )
            for c in CitationFormatter.to_dict(citations)
        ]

        # Populate response headers with token usage metrics
        response.headers["X-RAG-Prompt-Tokens"] = str(tracker.prompt_tokens)
        response.headers["X-RAG-Completion-Tokens"] = str(tracker.completion_tokens)
        response.headers["X-RAG-Total-Tokens"] = str(tracker.total_tokens)
        response.headers["X-RAG-LLM-Latency-Sec"] = f"{tracker.total_latency:.4f}"

        return QueryResponse(answer=answer, citations=citation_responses, healing=healing)
    finally:
        retrieval_owners.reset(owners_token)
        request_usage.reset(token)


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse, dependencies=[_auth])
def get_chunk(chunk_id: str, rag_session: str | None = Cookie(None)) -> ChunkResponse:
    """Return the full text of a passage, for expanding a citation's short ``text_snippet``."""
    chunk = get_pipeline().get_chunk(chunk_id)
    if chunk is None or chunk["metadata"].get("owner") not in _visible_owners(rag_session):
        raise HTTPException(
            status_code=404, detail="This passage is no longer in the document store."
        )
    return ChunkResponse(chunk_id=chunk_id, text=chunk["document"])


# ---------------------------------------------------------------------------
# Accounts and chat history. Login uses a cookie because the Authorization header
# already carries RAG_API_KEY, and HttpOnly keeps the token away from page scripts.
# /query, /upload, and /ingest stay open to anyone who passes the API key check, but a
# caller who isn't signed in can only read and add shared documents.
# ---------------------------------------------------------------------------
_SESSION_COOKIE = "rag_session"
_chat_store: ChatStore | None = None
_chat_store_lock = threading.Lock()
_auth_limiter = AttemptLimiter()


def get_chat_store() -> ChatStore:
    """Return the shared store, reopening it if the configured path has changed."""
    global _chat_store
    with _chat_store_lock:
        path = Path(_settings.chat_db_file)
        if _chat_store is None or _chat_store.path != path:
            _chat_store = ChatStore(path)
        return _chat_store


def _current_user(rag_session: str | None = Cookie(None)) -> User:
    user = get_chat_store().user_for_session(rag_session) if rag_session else None
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in.")
    return user


_signed_in = Depends(_current_user)


def _client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _refuse_if_too_many_attempts(address: str) -> None:
    """Checked before any password hashing, which is deliberately expensive."""
    if _auth_limiter.blocked(
        address, _settings.auth_ip_max_attempts, _settings.auth_ip_window_seconds
    ):
        raise HTTPException(
            status_code=429,
            detail="Too many attempts from this address. Try again in a few minutes.",
            headers={"Retry-After": str(int(_settings.auth_ip_window_seconds))},
        )


def _start_session(user: User, request: Request, response: Response) -> None:
    ttl_seconds = _settings.session_ttl_hours * 3600
    response.set_cookie(
        _SESSION_COOKIE,
        get_chat_store().create_session(user.id, ttl_seconds),
        max_age=int(ttl_seconds),
        httponly=True,
        samesite="strict",
        secure=_settings.session_cookie_secure or request.url.scheme == "https",
    )


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._@+-]+$")
    password: str = Field(..., max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=256)


class AccountResponse(BaseModel):
    username: str


class ChatCreateRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)


class ChatSummary(BaseModel):
    id: str
    title: str
    files: list[str]
    created_at: float
    updated_at: float


class ChatMessage(BaseModel):
    role: str
    text: str
    citations: list[CitationResponse] = []
    healing: HealingResponse | None = None
    created_at: float


class ChatDetail(ChatSummary):
    messages: list[ChatMessage]


class ChatFilesRequest(BaseModel):
    filenames: list[str] = Field(..., min_length=1, max_length=100)


class ChatTurnResponse(QueryResponse):
    chat: ChatSummary
    # What was actually searched for, when a follow-up had to be rewritten to stand alone.
    searched_for: str | None = None


class ImportedMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., max_length=200_000)
    citations: list[CitationResponse] = Field(default=[], max_length=200)
    healing: HealingResponse | None = None


class ChatImportRequest(BaseModel):
    import_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=200)
    files: list[str] = Field(default=[], max_length=100)
    created_at: float
    messages: list[ImportedMessage] = Field(..., max_length=10_000)


class ChatImportResponse(BaseModel):
    imported: bool


@app.post("/auth/register", response_model=AccountResponse, status_code=201, dependencies=[_auth])
def register(body: RegisterRequest, request: Request, response: Response) -> AccountResponse:
    """Create an account and sign in to it."""
    if not _settings.allow_signup:
        raise HTTPException(status_code=403, detail="Sign-up is turned off on this server.")
    if len(body.password) < _settings.min_password_length:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {_settings.min_password_length} characters.",
        )
    address = _client_address(request)
    _refuse_if_too_many_attempts(address)
    _auth_limiter.record(address)
    try:
        user = get_chat_store().create_user(body.username, body.password)
    except UsernameTakenError:
        raise HTTPException(status_code=409, detail="That username is taken.") from None
    _start_session(user, request, response)
    return AccountResponse(username=user.username)


@app.post("/auth/login", response_model=AccountResponse, dependencies=[_auth])
def login(body: LoginRequest, request: Request, response: Response) -> AccountResponse:
    """Sign in. Repeated failures block the account and the address for a few minutes."""
    address = _client_address(request)
    _refuse_if_too_many_attempts(address)
    try:
        user = get_chat_store().authenticate(body.username, body.password)
    except AccountLockedError:
        _auth_limiter.record(address)
        raise HTTPException(
            status_code=429, detail="Too many failed attempts. Try again in a few minutes."
        ) from None
    if user is None:
        _auth_limiter.record(address)
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    _start_session(user, request, response)
    return AccountResponse(username=user.username)


@app.post("/auth/logout", status_code=204, dependencies=[_auth])
def logout(response: Response, rag_session: str | None = Cookie(None)) -> None:
    """End this session on the server, not just in this browser."""
    if rag_session:
        get_chat_store().delete_session(rag_session)
    response.delete_cookie(_SESSION_COOKIE)


@app.get("/auth/me", response_model=AccountResponse, dependencies=[_auth])
def me(user: User = _signed_in) -> AccountResponse:
    return AccountResponse(username=user.username)


@app.get("/ingest/jobs", response_model=list[IngestJobStatusResponse], dependencies=[_auth])
def list_ingest_jobs(user: User = _signed_in) -> list[IngestJobStatusResponse]:
    """List your most recent ingest jobs, newest first, including any cut off by a restart."""
    with _ingest_jobs_lock:
        jobs = {**_read_persisted_ingest_jobs(), **_ingest_jobs}
    mine = [(i, j) for i, j in jobs.items() if j.get("user_id") == user.id]
    mine.sort(key=lambda item: item[1].get("created_at") or 0, reverse=True)
    return [_ingest_job_view(i, j) for i, j in mine[:50]]


@app.get("/chats", response_model=list[ChatSummary], dependencies=[_auth])
def list_chats(user: User = _signed_in) -> list[dict[str, Any]]:
    """List your chats, most recently active first, without their messages."""
    return get_chat_store().list_chats(user.id)


@app.post("/chats", response_model=ChatSummary, status_code=201, dependencies=[_auth])
def create_chat(body: ChatCreateRequest, user: User = _signed_in) -> dict[str, Any]:
    store = get_chat_store()
    return store.create_chat(user.id, body.title) if body.title else store.create_chat(user.id)


@app.post("/chats/import", response_model=ChatImportResponse, dependencies=[_auth])
def import_chat(body: ChatImportRequest, user: User = _signed_in) -> ChatImportResponse:
    """Copy in a chat saved elsewhere. Sending the same ``import_id`` again changes nothing."""
    imported = get_chat_store().import_chat(
        user.id,
        body.import_id,
        body.title,
        body.files,
        body.created_at,
        [m.model_dump(mode="json", exclude_none=True) for m in body.messages],
    )
    return ChatImportResponse(imported=imported)


@app.get(
    "/chats/{chat_id}",
    response_model=ChatDetail,
    response_model_exclude_none=True,
    dependencies=[_auth],
)
def get_chat(chat_id: str, user: User = _signed_in) -> dict[str, Any]:
    chat = get_chat_store().get_chat(user.id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="No chat found with that ID.")
    return chat


@app.delete("/chats/{chat_id}", status_code=204, dependencies=[_auth])
def delete_chat(chat_id: str, user: User = _signed_in) -> None:
    """Delete a chat's history. Documents it added stay in the shared store."""
    if not get_chat_store().delete_chat(user.id, chat_id):
        raise HTTPException(status_code=404, detail="No chat found with that ID.")


@app.post(
    "/chats/{chat_id}/files",
    response_model=ChatDetail,
    response_model_exclude_none=True,
    dependencies=[_auth],
)
def add_chat_files(chat_id: str, body: ChatFilesRequest, user: User = _signed_in) -> dict[str, Any]:
    """Record documents that were ingested for this chat."""
    chat = get_chat_store().add_files(user.id, chat_id, body.filenames)
    if chat is None:
        raise HTTPException(status_code=404, detail="No chat found with that ID.")
    return chat


async def _standalone_question(question: str, messages: list[dict[str, Any]]) -> str:
    """Rewrite a follow-up using the chat so far. If that fails, ask the question as written."""
    if not question or not any(m["role"] == "user" for m in messages):
        return question
    try:
        rewritten = await asyncio.wait_for(
            get_pipeline().generator.condense_question(question, messages), _query_timeout()
        )
    except Exception as exc:
        logger.warning("Could not rewrite a follow-up question: %s", exc)
        return question
    rewritten = rewritten.strip()
    return rewritten if 0 < len(rewritten) <= RAGPipeline.MAX_QUESTION_LENGTH else question


@app.post(
    "/chats/{chat_id}/messages",
    response_model=ChatTurnResponse,
    response_model_exclude_none=True,
    dependencies=[_auth],
)
async def send_chat_message(
    chat_id: str,
    body: QueryRequest,
    response: Response,
    user: User = _signed_in,
    rag_session: str | None = Cookie(None),
) -> ChatTurnResponse:
    """Ask a question in a chat and save it with its answer.

    A follow-up is first rewritten to stand alone using the chat so far, so that retrieval
    and generation understand it. The question and answer are saved together once the answer
    exists, so a failed query leaves the history unchanged.
    """
    store = get_chat_store()
    existing = await asyncio.to_thread(store.get_chat, user.id, chat_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="No chat found with that ID.")

    asked = body.question.strip()
    standalone = await _standalone_question(asked, existing["messages"])
    result = await query(
        body.model_copy(update={"question": standalone}), response, rag_session=rag_session
    )

    chat = await asyncio.to_thread(
        store.add_exchange,
        user.id,
        chat_id,
        asked,
        result.answer,
        [c.model_dump(exclude_none=True) for c in result.citations],
        result.healing.model_dump(mode="json", exclude_none=True) if result.healing else None,
    )
    if chat is None:  # deleted while the answer was being generated
        raise HTTPException(status_code=404, detail="No chat found with that ID.")
    return ChatTurnResponse(
        **result.model_dump(),
        chat=ChatSummary(**chat),
        searched_for=standalone if standalone != asked else None,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint — Server-Sent Events
# ---------------------------------------------------------------------------


# Word-sized pieces (keeping trailing whitespace) for streaming an already
# verified answer, so clients render it the same way as live LLM tokens.
_SSE_CHUNK_RE = re.compile(r"\S+\s*|\s+")


async def _next_event(events: AsyncGenerator[tuple[str, Any], None]) -> tuple[str, Any] | None:
    try:
        return await anext(events)
    except StopAsyncIteration:
        return None


@app.post("/query/stream", dependencies=[_auth])
async def query_stream(
    request: QueryRequest, rag_session: str | None = Cookie(None)
) -> StreamingResponse:
    """Answer a question and stream tokens via Server-Sent Events (SSE).

    Clients should connect with ``Accept: text/event-stream``.

    Each SSE event is one of:
    - ``data: {"token": "<text>"}``   — a generated text chunk
    - ``data: {"healing": {...}}``     — healing metadata (only when self-healing ran)
    - ``data: {"citations": [...]}``   — final citation list (last event before DONE)
    - ``data: [DONE]``                 — stream complete

    With self-healing enabled, an answer cannot be un-sent once streamed, so
    the verify/repair loop runs to completion first and only the *verified*
    answer (or the abstention message) is streamed. Time-to-first-token is
    therefore the full healing latency in that mode.

    Example (curl)::

        curl -N -X POST http://localhost:8000/query/stream \\
             -H 'Content-Type: application/json' \\
             -d '{"question": "What is RAG?", "use_hybrid": true}'
    """
    pipeline = get_pipeline()
    timeout = _query_timeout()
    owners = await asyncio.to_thread(_visible_owners, rag_session)

    async def _event_stream() -> AsyncGenerator[str, None]:
        deadline = None if timeout is None else time.monotonic() + timeout
        # Set here because the body runs while the response is sent, after this route returns.
        retrieval_owners.set(owners)
        try:
            question = request.question.strip()
            if not question:
                yield f"data: {json.dumps({'error': 'Question must not be empty.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if len(question) > RAGPipeline.MAX_QUESTION_LENGTH:
                yield f"data: {json.dumps({'error': 'Question exceeds maximum length.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if _healing_requested(request):
                result = await asyncio.wait_for(
                    pipeline.query_with_healing_async(
                        question,
                        top_k=request.top_k,
                        use_hybrid=request.use_hybrid,
                        use_reranker=request.use_reranker,
                    ),
                    timeout,
                )
                for piece in _SSE_CHUNK_RE.findall(result.answer):
                    yield f"data: {json.dumps({'token': piece})}\n\n"
                yield f"data: {json.dumps({'healing': _healing_payload(result.info)})}\n\n"
                citation_dicts = CitationFormatter.to_dict(result.citations)
                yield f"data: {json.dumps({'citations': citation_dicts})}\n\n"
                return

            events = pipeline.stream_answer(
                question,
                top_k=request.top_k,
                use_hybrid=request.use_hybrid,
                use_reranker=request.use_reranker,
            )
            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                event = await asyncio.wait_for(_next_event(events), remaining)
                if event is None:
                    break
                kind, value = event
                if kind == "token":
                    yield f"data: {json.dumps({'token': value})}\n\n"
                elif kind == "citations":
                    # Emit citations as final structured event
                    citation_dicts = CitationFormatter.to_dict(value)
                    yield f"data: {json.dumps({'citations': citation_dicts})}\n\n"

        except ValueError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except TimeoutError:
            _http_stream_errors_total.labels(path="/query/stream", reason="timeout").inc()
            yield f"data: {json.dumps({'error': 'The query timed out.'})}\n\n"
        except VectorStoreUnavailableError:
            _http_stream_errors_total.labels(path="/query/stream", reason="unavailable").inc()
            yield f"data: {json.dumps({'error': 'The vector database is unavailable.'})}\n\n"
        except Exception as exc:
            _http_stream_errors_total.labels(path="/query/stream", reason="internal").inc()
            yield f"data: {json.dumps({'error': 'Internal server error during streaming.'})}\n\n"
            raise exc
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
