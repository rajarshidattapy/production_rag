"""Regression tests for deployment, reliability and observability defects found by the drill."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH = "master"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    import src.api as api_app

    api_app.reset_pipeline()
    yield api_app
    api_app.reset_pipeline()


@pytest.fixture
def client(api) -> TestClient:
    return TestClient(api.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", ["ci.yml", "evaluate.yml", "docker-publish.yml"])
def test_workflows_trigger_on_the_default_branch(workflow: str) -> None:
    text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    branches = re.findall(r"branches: \[(.*?)\]", text)
    assert branches and all(b == DEFAULT_BRANCH for b in branches)


def test_ci_install_file_exists() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for requirements in re.findall(r"pip install -r (\S+)", text):
        assert (ROOT / requirements).exists()


def test_compose_chroma_service_matches_the_image() -> None:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "chroma_data:/data" in text, "Chroma 1.x persists to /data; other paths lose data"
    assert "chromadb/chroma:latest" not in text
    chroma = text[text.index("chroma:") : text.index("  api:")]
    healthcheck = re.search(r"^\s*test: (.*)$", chroma, re.M)
    assert healthcheck, "the api service waits on this healthcheck"
    assert "curl" not in healthcheck[1], "the Chroma image does not contain curl"
    assert "/api/v1" not in healthcheck[1], "/api/v1 returns 410 on Chroma 1.x"


def test_docker_image_includes_the_ui() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = text[text.index("AS runtime") :]
    assert re.search(r"^COPY .*\bindex\.html\b", runtime, re.M)


def test_reranker_default_is_consistent() -> None:
    from src.config import Settings
    from src.retrieval.reranker import CrossEncoderReranker

    default = Settings.model_fields["reranker_model"].default
    assert CrossEncoderReranker().model_name == default
    assert f"RAG_RERANKER_MODEL={default}" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert default in (ROOT / "scripts" / "cache_models.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------


def _anthropic_reply() -> MagicMock:
    reply = MagicMock()
    reply.content = [MagicMock(text="ok")]
    reply.usage.input_tokens = 1
    reply.usage.output_tokens = 1
    return reply


def test_anthropic_call_matches_the_sdk_signature() -> None:
    """autospec rejects keyword arguments the installed SDK does not accept."""
    from src.generation.llm_client import LLMClient

    with patch(
        "anthropic.resources.messages.Messages.create",
        autospec=True,
        return_value=_anthropic_reply(),
    ) as create:
        answer = LLMClient(provider="anthropic", model="m").complete("hi", system="be brief")
    assert answer == "ok"
    create.assert_called_once()


def test_anthropic_async_call_matches_the_sdk_signature() -> None:
    from src.generation.llm_client import LLMClient

    async def fake(*args: Any, **kwargs: Any) -> MagicMock:
        return _anthropic_reply()

    with patch(
        "anthropic.resources.messages.AsyncMessages.create", autospec=True, side_effect=fake
    ):
        answer = asyncio.run(LLMClient(provider="anthropic", model="m").complete_async("hi"))
    assert answer == "ok"


def test_explicit_base_url_beats_openrouter_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG_OPENROUTER_MODE once sent a local drill's traffic to OpenRouter."""
    from src.generation.llm_client import OPENROUTER_BASE_URL, LLMClient

    monkeypatch.setenv("RAG_OPENROUTER_MODE", "true")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = LLMClient(provider="openai", model="gpt-4o-mini")
    assert client.openai_client_kwargs().get("base_url") == OPENROUTER_BASE_URL

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    assert "base_url" not in client.openai_client_kwargs()


def test_drill_refuses_to_run_unless_llm_calls_stay_local(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from operational_drill import assert_llm_stays_local
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("RAG_OPENROUTER_MODE", "true")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        assert_llm_stays_local()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    with pytest.raises(SystemExit):
        assert_llm_stays_local()

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    assert_llm_stays_local()


# ---------------------------------------------------------------------------
# Vector database outages
# ---------------------------------------------------------------------------


def test_unreachable_chroma_is_not_reported_as_bad_input(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chromadb

    from src.config import settings

    def refuse(**kwargs: Any) -> None:
        raise ValueError("Could not connect to a Chroma server. Are you sure it is running?")

    monkeypatch.setattr(chromadb, "HttpClient", refuse)
    monkeypatch.setattr(settings, "chroma_host", "127.0.0.1")
    monkeypatch.setattr(settings, "chroma_port", 9)
    monkeypatch.setattr("src.utils.retry.time.sleep", lambda _: None)

    response = client.post("/query", json={"question": "What is hybrid search?"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_connection_loss_mid_call_returns_503(
    api, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.utils.retry.time.sleep", lambda _: None)
    pipeline = api.get_pipeline()
    store = pipeline._get_vector_store("en")
    collection = MagicMock()
    collection.query.side_effect = httpx.ConnectError("refused")
    store._collection = collection

    response = client.post("/query", json={"question": "What is hybrid search in retrieval?"})
    assert response.status_code == 503
    assert collection.query.call_count == 4  # first try plus three retries


def test_readyz_detects_an_outage_after_startup(
    api, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", MagicMock())
    pipeline = api.get_pipeline()
    assert client.get("/readyz").status_code == 200

    for lang in ("en", "de", "es"):
        monkeypatch.setattr(
            pipeline._get_vector_store(lang)._client,
            "heartbeat",
            MagicMock(side_effect=httpx.ConnectError("refused")),
        )
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200


def test_store_recovers_after_another_replica_resets_it(temp_chroma_dir: Path) -> None:
    from src.ingestion.chunker import Chunk
    from src.retrieval.vector_store import VectorStore

    replica_a = VectorStore(collection_name="shared_store", persist_path=temp_chroma_dir)
    replica_b = VectorStore(collection_name="shared_store", persist_path=temp_chroma_dir)
    chunk = Chunk(content="hybrid search", metadata={"source": "a.txt"}, chunk_id="c1")
    replica_a.add_chunks([chunk])
    assert replica_b.count() == 1

    replica_a.delete_collection()

    assert replica_b.count() == 0
    assert replica_b.similarity_search("hybrid", k=1) == []
    assert replica_b.add_chunks([chunk]) == 1
    assert replica_a.count() == 1


# ---------------------------------------------------------------------------
# Query deadline
# ---------------------------------------------------------------------------


async def _hang(*args: Any, **kwargs: Any) -> None:
    await asyncio.sleep(30)


def test_query_deadline_returns_504(
    api, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "query_timeout_seconds", 0.05)
    monkeypatch.setattr(api.get_pipeline(), "query_async", _hang)

    response = client.post("/query", json={"question": "What is hybrid search?"})
    assert response.status_code == 504


def test_query_deadline_can_be_disabled(api, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "query_timeout_seconds", 0)
    assert api._query_timeout() is None


def test_stream_deadline_reports_an_error_and_finishes(
    api, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import settings

    async def hanging_stream(*args: Any, **kwargs: Any):
        yield ("contexts", [])
        await asyncio.sleep(30)

    monkeypatch.setattr(settings, "query_timeout_seconds", 0.05)
    monkeypatch.setattr(api.get_pipeline(), "stream_answer", hanging_stream)

    response = client.post("/query/stream", json={"question": "What is hybrid search?"})
    assert "timed out" in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# Async ingest jobs
# ---------------------------------------------------------------------------


def test_finished_job_status_survives_a_restart(
    api, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api._record_ingest_job("job-1", status="completed", chunks_ingested=3, total_chunks=3)

    api._ingest_jobs.clear()  # what a restart does to the in-memory record
    monkeypatch.setattr(api, "_INSTANCE_ID", "a-new-process")

    body = client.get("/ingest/jobs/job-1").json()
    assert body["status"] == "completed"
    assert body["chunks_ingested"] == 3
    assert "owner" not in body


def test_job_cut_off_by_a_restart_is_reported_as_interrupted(
    api, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    api._record_ingest_job("job-2", status="running")
    assert client.get("/ingest/jobs/job-2").json()["status"] == "running"

    api._ingest_jobs.clear()
    monkeypatch.setattr(api, "_INSTANCE_ID", "a-new-process")

    body = client.get("/ingest/jobs/job-2").json()
    assert body["status"] == "interrupted"
    assert "restarted" in body["error"]


def test_unknown_job_is_still_404(client: TestClient) -> None:
    assert client.get("/ingest/jobs/nope").status_code == 404


def test_job_file_is_bounded_and_tolerates_corruption(
    api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "_INGEST_JOBS_MAX", 2)
    path = api._ingest_jobs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    for job_id in ("a", "b", "c"):
        api._record_ingest_job(job_id, status="completed")

    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["b", "c"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_total_covers_every_language(api, client: TestClient) -> None:
    from src.ingestion.chunker import Chunk

    pipeline = api.get_pipeline()
    for lang in ("en", "de", "es"):
        pipeline._get_vector_store(lang).add_chunks(
            [Chunk(content=f"{lang} text", metadata={"source": "x"}, chunk_id=f"{lang}-1")]
        )
    assert client.get("/stats").json()["chunks_in_store"] == 3


# ---------------------------------------------------------------------------
# Monitoring wiring
# ---------------------------------------------------------------------------


def test_reading_monitoring_settings_does_not_patch_the_llm_clients() -> None:
    code = (
        "import sys; from src.monitoring.config import settings; "
        "sys.exit(1 if 'src.monitoring.wrappers' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_api_uses_plain_pipeline_when_monitoring_is_off(api) -> None:
    from src.pipeline import RAGPipeline

    assert type(api.get_pipeline()) is RAGPipeline


@pytest.fixture
def monitored(api, monkeypatch: pytest.MonkeyPatch):
    """The API running with MONITOR_ENABLED=true, recording every hook instead of exporting."""
    from src.monitoring.config import settings as monitoring_settings
    from src.monitoring.extensions import BaseExtension
    from src.monitoring.metrics import MetricsCollector

    calls: list[str] = []

    class Recorder(BaseExtension):
        def __getattribute__(self, name: str) -> Any:
            if name.startswith("on_"):

                def hook(*args: Any, **kwargs: Any) -> None:
                    step = args[0] if name.startswith("on_step_") and args else ""
                    calls.append(f"{name}:{step}".rstrip(":"))

                return hook
            return super().__getattribute__(name)

    monkeypatch.setattr(monitoring_settings, "enabled", True)
    monkeypatch.setattr(MetricsCollector, "_init_meter", lambda self: None)
    pipeline = api.get_pipeline()
    pipeline.extensions.append(Recorder())
    return pipeline, calls


def _fake_contexts(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return [{"id": "c1", "document": "Hybrid search.", "metadata": {"source": "a"}, "score": 0.9}]


def test_api_uses_the_monitored_pipeline_when_enabled(monitored) -> None:
    from src.monitoring import MonitoredRAGPipeline

    pipeline, _ = monitored
    assert isinstance(pipeline, MonitoredRAGPipeline)


def test_query_endpoint_is_traced_when_monitoring_is_enabled(
    client: TestClient, monitored, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, calls = monitored
    inner = pipeline._pipeline
    store = inner._get_vector_store("en")
    monkeypatch.setattr(store, "similarity_search", _fake_contexts)
    monkeypatch.setattr(
        "src.generation.llm_client.LLMClient.complete", lambda self, *a, **k: "Answer [1]."
    )

    response = client.post("/query", json={"question": "What is hybrid search in retrieval?"})

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Answer [1]."
    for expected in (
        "on_query_start",
        "on_step_start:retrieve",
        "on_step_start:generate",
        "on_query_end",
    ):
        assert expected in calls, calls


def test_stream_endpoint_is_traced_when_monitoring_is_enabled(
    client: TestClient, monitored, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, calls = monitored
    inner = pipeline._pipeline
    monkeypatch.setattr(inner._get_vector_store("en"), "similarity_search", _fake_contexts)

    async def tokens(*args: Any, **kwargs: Any):
        yield "Answer "
        yield "[1]."

    monkeypatch.setattr(inner.generator, "generate_stream", tokens)

    response = client.post(
        "/query/stream", json={"question": "What is hybrid search in retrieval?"}
    )

    assert '"token": "Answer "' in response.text
    assert '"citations"' in response.text
    for expected in (
        "on_query_start",
        "on_step_start:retrieve",
        "on_step_start:generate",
        "on_first_token",
        "on_generation_llm_call",
        "on_step_end:generate",
        "on_query_end",
    ):
        assert expected in calls, calls


def test_concurrent_queries_keep_separate_token_totals(monitored) -> None:
    import threading

    pipeline, _ = monitored
    seen: dict[str, dict[str, int]] = {}

    def worker(name: str, total: int) -> None:
        pipeline._last_query_tokens = {"prompt": 0, "completion": 0, "total": total}
        barrier.wait()
        seen[name] = pipeline._last_query_tokens

    barrier = threading.Barrier(2)
    threads = [threading.Thread(target=worker, args=(n, t)) for n, t in (("a", 1), ("b", 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["a"]["total"] == 1 and seen["b"]["total"] == 2
