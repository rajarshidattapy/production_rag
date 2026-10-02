"""Integration tests: self-healing wired into RAGPipeline, the API, and monitoring.

Uses the real Chroma store (with the conftest's deterministic dummy embeddings)
and mocks only the LLM calls, so nothing here needs network access or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import api as api_app
from src.healing.metrics import healing_metrics
from src.healing.state_machine import ABSTENTION_MESSAGE
from src.pipeline import RAGPipeline

QUESTION = "What does hybrid search merge?"


def _grounded_answer(
    query: str, contexts: list[dict[str, Any]], system_prompt: str | None = None
) -> str:
    """Mock generator that cites whichever passage actually contains the fact."""
    for i, ctx in enumerate(contexts, start=1):
        if "Hybrid search" in ctx.get("document", ""):
            return f"Hybrid search merges BM25 keyword search with vector similarity [{i}]."
    return "I cannot find sufficient information in the provided documents to answer this question."


def _configure(pipeline: RAGPipeline, **overrides: Any) -> RAGPipeline:
    # Dummy hash embeddings produce arbitrary cosine scores, so disable the
    # score floor; judge/rewriter LLM calls are disabled so no network is used.
    defaults = {
        "min_retrieval_score": 0.0,
        "verifier_enabled": False,
        "healing_llm_rewrite": False,
    }
    defaults.update(overrides)
    pipeline.config = pipeline.config.model_copy(update=defaults)
    pipeline._healing_engine = None
    return pipeline


@pytest.fixture
def pipeline(sample_docs_dir: Path) -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    p.ingest(sample_docs_dir)
    return _configure(p)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


class TestPipelineIntegration:
    def test_healing_off_by_default_keeps_original_path(self, pipeline: RAGPipeline) -> None:
        with patch.object(pipeline.generator, "generate", return_value="plain answer") as gen:
            with patch("src.healing.integration.build_healing_engine") as build:
                answer, _ = pipeline.query(QUESTION)
        assert answer == "plain answer"  # unverified answer returned as before
        gen.assert_called_once()
        build.assert_not_called()

    def test_query_with_healing_passes(self, pipeline: RAGPipeline) -> None:
        with patch.object(pipeline.generator, "generate", side_effect=_grounded_answer):
            result = pipeline.query_with_healing(QUESTION, use_hybrid=True, top_k=2)
        assert result.info.status == "passed"
        assert "[" in result.answer
        # Citations are aligned with the passages the generator saw.
        cited = int(result.answer.split("[")[1].split("]")[0])
        assert "Hybrid search" in result.citations[cited - 1].text_snippet

    def test_config_flag_routes_query_through_healing(self, pipeline: RAGPipeline) -> None:
        _configure(pipeline, self_healing_enabled=True)
        with patch.object(pipeline.generator, "generate", return_value="No citations at all."):
            answer, citations = pipeline.query(QUESTION, use_hybrid=True)
        assert answer == ABSTENTION_MESSAGE
        assert citations == []

    def test_query_self_heal_false_overrides_config(self, pipeline: RAGPipeline) -> None:
        _configure(pipeline, self_healing_enabled=True)
        with patch.object(pipeline.generator, "generate", return_value="raw"):
            answer, _ = pipeline.query(QUESTION, self_heal=False)
        assert answer == "raw"

    def test_retry_changes_generation_prompt(self, pipeline: RAGPipeline) -> None:
        calls: list[str | None] = []

        def gen(query, contexts, system_prompt=None):
            calls.append(system_prompt)
            if len(calls) == 1:
                return "Hybrid search merges things [9]."  # invalid citation
            return _grounded_answer(query, contexts)

        with patch.object(pipeline.generator, "generate", side_effect=gen):
            result = pipeline.query_with_healing(QUESTION, use_hybrid=True, top_k=2)

        assert result.info.status == "recovered"
        assert calls[0] is None and calls[1] is not None
        assert "[9]" in calls[1]

    def test_empty_store_abstains_without_llm_calls(self) -> None:
        p = _configure(RAGPipeline())
        p.reset()
        with patch.object(p.generator, "generate") as gen:
            result = p.query_with_healing(QUESTION)
        assert result.abstained and result.info.reason == "empty_knowledge_base"
        gen.assert_not_called()

    def test_validation_errors_still_raised(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValueError):
            pipeline.query_with_healing("   ")

    @pytest.mark.asyncio
    async def test_query_async_with_healing(self, pipeline: RAGPipeline) -> None:
        with patch.object(pipeline.generator, "generate", side_effect=_grounded_answer):
            answer, citations = await pipeline.query_async(
                QUESTION, use_hybrid=True, self_heal=True
            )
        assert "Hybrid search merges" in answer and citations

    def test_retrieve_overrides_reach_hybrid_search(self, pipeline: RAGPipeline) -> None:
        retriever = pipeline._get_hybrid_retriever("en")
        with patch.object(retriever, "search", wraps=retriever.search) as search:
            pipeline._retrieve(QUESTION, use_hybrid=True, k=2, fetch_k=7, alpha=0.3)
        search.assert_called_once_with(QUESTION, k=7, alpha=0.3)

    def test_retrieve_default_call_unchanged(self, pipeline: RAGPipeline) -> None:
        retriever = pipeline._get_hybrid_retriever("en")
        with patch.object(retriever, "search", wraps=retriever.search) as search:
            pipeline._retrieve(QUESTION, use_hybrid=True, k=2)
        search.assert_called_once_with(QUESTION, k=2)

    def test_context_budget_override(self, pipeline: RAGPipeline) -> None:
        ctx = [{"document": "a" * 10}, {"document": "b" * 10}]
        assert len(pipeline._apply_context_budget(ctx, budget=10)) == 1
        assert pipeline._apply_context_budget(ctx) == ctx


def test_hybrid_alpha_override_changes_ranking_without_mutating_retriever(
    sample_docs_dir: Path,
) -> None:
    p = RAGPipeline()
    p.reset()
    p.ingest(sample_docs_dir)
    retriever = p._get_hybrid_retriever("en")
    original_alpha = retriever.alpha
    retriever.search("BM25 keyword", k=2, alpha=0.0)
    assert retriever.alpha == original_alpha
    with pytest.raises(ValueError):
        retriever.search("BM25", k=2, alpha=1.5)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------


@pytest.fixture
def client(sample_docs_dir: Path) -> TestClient:
    api_app.reset_pipeline()
    c = TestClient(api_app.app)
    assert (
        c.post("/ingest", json={"source": str(sample_docs_dir), "reset": True}).status_code == 200
    )
    _configure(api_app.get_pipeline())
    yield c
    api_app.reset_pipeline()


def _sse_events(text: str) -> list[Any]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


class TestAPIIntegration:
    def test_query_without_healing_has_no_healing_field(self, client: TestClient) -> None:
        pipeline = api_app.get_pipeline()
        with patch.object(pipeline.generator, "generate_async", new=MagicMock()) as gen:

            async def _ans(*a, **k):
                return "plain"

            gen.side_effect = _ans
            body = client.post("/query", json={"question": QUESTION}).json()
        assert body["answer"] == "plain"
        assert "healing" not in body
        assert set(body) == {"answer", "citations"}

    def test_query_with_self_heal_returns_metadata(self, client: TestClient) -> None:
        pipeline = api_app.get_pipeline()
        with patch.object(pipeline.generator, "generate", side_effect=_grounded_answer):
            resp = client.post(
                "/query", json={"question": QUESTION, "self_heal": True, "use_hybrid": True}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["healing"]["status"] == "passed"
        assert body["healing"]["attempts"] == 1
        assert body["healing"]["recovered"] is False
        assert resp.headers["X-RAG-Healing-Status"] == "passed"
        assert body["citations"]

    def test_query_abstention_is_structured(self, client: TestClient) -> None:
        pipeline = api_app.get_pipeline()
        with patch.object(pipeline.generator, "generate", return_value="No citations here."):
            body = client.post("/query", json={"question": QUESTION, "self_heal": True}).json()
        assert body["answer"] == ABSTENTION_MESSAGE
        assert body["citations"] == []
        assert body["healing"]["status"] == "abstained"
        assert body["healing"]["attempts"] == pipeline.config.max_healing_attempts
        assert body["healing"]["reason"]

    def test_query_self_heal_validation_error_is_400(self, client: TestClient) -> None:
        assert client.post("/query", json={"question": "", "self_heal": True}).status_code == 400

    def test_stream_after_healing(self, client: TestClient) -> None:
        pipeline = api_app.get_pipeline()
        calls = {"n": 0}

        def gen(query, contexts, system_prompt=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Hybrid search merges things [9]."  # fails verification
            return _grounded_answer(query, contexts)

        with patch.object(pipeline.generator, "generate", side_effect=gen):
            resp = client.post(
                "/query/stream",
                json={"question": QUESTION, "self_heal": True, "use_hybrid": True},
            )
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        tokens = "".join(e["token"] for e in events if isinstance(e, dict) and "token" in e)
        assert tokens.startswith("Hybrid search merges BM25")  # only the verified answer streamed
        assert "[9]" not in tokens  # the failed draft never reached the client

        keys = [next(iter(e)) if isinstance(e, dict) else e for e in events]
        assert keys.index("healing") < keys.index("citations")
        healing = next(e["healing"] for e in events if isinstance(e, dict) and "healing" in e)
        assert healing["status"] == "recovered" and healing["attempts"] == 2
        assert events[-1] == "[DONE]"

    def test_stream_without_healing_unchanged(self, client: TestClient) -> None:
        pipeline = api_app.get_pipeline()

        async def fake_stream(q, ctx, system_prompt=None):
            for t in ["Hello", " world"]:
                yield t

        with patch.object(pipeline.generator, "generate_stream", side_effect=fake_stream):
            resp = client.post("/query/stream", json={"question": QUESTION})
        events = _sse_events(resp.text)
        assert [e["token"] for e in events if isinstance(e, dict) and "token" in e] == [
            "Hello",
            " world",
        ]
        assert not any(isinstance(e, dict) and "healing" in e for e in events)

    def test_metrics_endpoint_exposes_healing_metrics(self, client: TestClient) -> None:
        before = healing_metrics.value("rag_healing_queries_total", {"outcome": "abstained"})
        pipeline = api_app.get_pipeline()
        with patch.object(pipeline.generator, "generate", return_value="uncited"):
            client.post("/query", json={"question": QUESTION, "self_heal": True})
        text = client.get("/metrics").text
        assert "rag_healing_queries_total" in text
        assert "rag_healing_failures_total" in text
        assert "rag_http_requests_total" in text  # existing metrics still there
        after = healing_metrics.value("rag_healing_queries_total", {"outcome": "abstained"})
        assert after == before + 1


# ----------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------


class _RecordingExtension:
    def __init__(self) -> None:
        self.steps: list[str] = []
        self.queries = 0

    def on_query_start(self, question, metadata):
        self.queries += 1

    def on_step_start(self, step_name, input_data, metadata):
        self.steps.append(step_name)


def test_monitored_pipeline_traces_each_healing_attempt(
    pipeline: RAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.monitoring.config import settings as monitoring_settings
    from src.monitoring.extensions import LangfuseTracingExtension, OTelMetricsExtension
    from src.monitoring.wrappers import MonitoredRAGPipeline

    # Keep the prompt registry out of the repo's data/ directory.
    monkeypatch.setattr(monitoring_settings, "baseline_dir", str(tmp_path))

    calls = {"n": 0}

    def gen(query, contexts, system_prompt=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # Generator says the context lacks the answer → RETRIEVAL_FAILURE → rewrite + re-retrieve.
            return "I cannot find sufficient information in the provided documents to answer this question."
        return _grounded_answer(query, contexts)

    rec = _RecordingExtension()
    tracing = LangfuseTracingExtension(tracer=MagicMock())
    metrics = OTelMetricsExtension(metrics=MagicMock())
    # Patch before wrapping: MonitoredRAGPipeline captures generator.generate
    # at construction and wraps it with its tracing layer.
    with patch.object(pipeline.generator, "generate", side_effect=gen):
        monitored = MonitoredRAGPipeline(pipeline, extensions=[tracing, metrics, rec])  # type: ignore[list-item]
        result = monitored.query_with_healing(QUESTION, use_hybrid=True, top_k=2)

    assert result.info.status == "recovered"
    assert rec.queries == 1
    healing_steps = [s for s in rec.steps if s not in ("retrieve", "generate", "rerank")]
    assert healing_steps == [
        "analyze",
        "retrieve_attempt_1",
        "generate_attempt_1",
        "verify_attempt_1",
        "query_rewrite",
        "retrieve_attempt_2",
        "generate_attempt_2",
        "verify_attempt_2",
    ]
    # The pipeline-level wrappers still emit their own spans inside each attempt.
    assert "retrieve" in rec.steps and "generate" in rec.steps
