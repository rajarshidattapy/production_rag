"""Start-up warm-up, one load per model, and the load-test script."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import uvicorn
from fastapi.testclient import TestClient

from scripts import load_test
from src import api as api_app
from src.config import settings
from src.retrieval import vector_store as vector_store_module
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import _ChromaEmbeddingFunction


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    api_app.reset_pipeline()
    api_app._models_warm.clear()
    yield
    api_app.reset_pipeline()
    api_app._models_warm.clear()


def _fake_cross_encoder() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = [0.5]
    return MagicMock(return_value=model)


class TestWarmUp:
    def test_models_load_before_the_first_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "warm_up_on_start", True)
        cross_encoder = _fake_cross_encoder()
        order: list[str] = []
        real_warm = api_app._warm_up

        def tracked_warm() -> None:
            real_warm()
            order.append("warm-up done")

        with (
            patch("sentence_transformers.CrossEncoder", cross_encoder),
            patch.object(api_app, "_warm_up", tracked_warm),
            patch.object(
                _ChromaEmbeddingFunction,
                "embed_document",
                wraps=_ChromaEmbeddingFunction.embed_document,
                autospec=True,
            ) as embed,
        ):
            with TestClient(api_app.app) as client:
                order.append("first request")
                assert client.get("/healthz").status_code == 200
        assert order == ["warm-up done", "first request"]
        embed.assert_called()
        cross_encoder.assert_called_once()

    def test_reranker_can_be_left_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "warm_up_on_start", True)
        monkeypatch.setattr(settings, "warm_up_reranker", False)
        cross_encoder = _fake_cross_encoder()
        with patch("sentence_transformers.CrossEncoder", cross_encoder):
            with TestClient(api_app.app):
                pass
        cross_encoder.assert_not_called()

    def test_a_model_that_cannot_load_stops_start_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "warm_up_on_start", True)
        with patch.object(
            _ChromaEmbeddingFunction, "embed_document", side_effect=OSError("model not found")
        ):
            with pytest.raises(OSError, match="model not found"):
                with TestClient(api_app.app):
                    pass

    def test_search_index_trouble_does_not_stop_start_up(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(settings, "warm_up_on_start", True)
        monkeypatch.setattr(settings, "warm_up_reranker", False)
        pipeline = api_app.get_pipeline()
        with patch.object(
            pipeline, "_get_hybrid_retriever", side_effect=RuntimeError("chroma down")
        ):
            with TestClient(api_app.app) as client:
                assert client.get("/healthz").status_code == 200
        assert "Could not warm up the en search index" in caplog.text

    def test_search_indexes_are_built_during_warm_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "warm_up_on_start", True)
        monkeypatch.setattr(settings, "warm_up_reranker", False)
        with TestClient(api_app.app):
            assert set(api_app.get_pipeline()._hybrid_retrievers) == {"en", "de", "es"}

    def test_readyz_loads_models_when_warm_up_was_skipped(self) -> None:
        cross_encoder = _fake_cross_encoder()
        with patch("sentence_transformers.CrossEncoder", cross_encoder):
            client = TestClient(api_app.app)
            assert client.get("/readyz").status_code == 200
            assert client.get("/readyz").status_code == 200
        cross_encoder.assert_called_once()
        assert api_app._models_warm.is_set()

    def test_readyz_is_not_ready_when_a_model_cannot_load(self) -> None:
        with patch.object(_ChromaEmbeddingFunction, "embed_document", side_effect=OSError("nope")):
            assert TestClient(api_app.app).get("/readyz").status_code == 503


class TestOneLoadPerModel:
    def test_concurrent_first_requests_load_the_embedding_model_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads: list[int] = []

        def slow_model(name: str) -> object:
            loads.append(1)
            time.sleep(0.2)
            return object()

        monkeypatch.setattr(vector_store_module, "_EMBEDDING_MODEL_CACHE", {})
        with patch("sentence_transformers.SentenceTransformer", slow_model):
            functions = [_ChromaEmbeddingFunction("concurrency-test-model") for _ in range(8)]
            threads = [threading.Thread(target=f._lazy_load) for f in functions]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(loads) == 1
        assert len({id(f._model) for f in functions}) == 1

    def test_concurrent_first_reranks_load_the_reranker_once(self) -> None:
        loads: list[int] = []
        model = MagicMock()
        model.predict.return_value = [0.5]

        def slow_cross_encoder(name: str) -> MagicMock:
            loads.append(1)
            time.sleep(0.2)
            return model

        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder", slow_cross_encoder):
            threads = [
                threading.Thread(target=reranker.rerank, args=("q", [{"document": "d"}]))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(loads) == 1


class TestLoadTestHelpers:
    def test_percentiles(self) -> None:
        values = list(range(1, 101))
        assert load_test.percentile(values, 50) == 50
        assert load_test.percentile(values, 95) == 95
        assert load_test.percentile(values, 99) == 99
        assert load_test.percentile([], 50) is None
        assert load_test.percentile([7], 99) == 7

    def test_error_windows_merge_nearby_failures(self) -> None:
        timeline = [(1, True), (10, False), (12, False), (13, True), (30, False), (31, True)]
        assert load_test.error_windows(timeline) == [
            {"from_s": 10, "to_s": 12},
            {"from_s": 30, "to_s": 30},
        ]

    def test_memory_trend_ignores_start_up_growth(self) -> None:
        settling = [(t, 500 + t * 10) for t in range(0, 30, 5)]  # grows early
        flat = [(t, 800.0) for t in range(30, 65, 5)]
        assert load_test.memory_trend([*settling, *flat]) == 0.0
        leaking = [(t, 500 + t) for t in range(0, 65, 5)]
        assert load_test.memory_trend(leaking) == 60.0

    def test_questions_vary_but_are_reproducible(self) -> None:
        first = load_test.questions(300)
        assert first == load_test.questions(300)
        assert len(set(first)) == 300

    def test_memory_of_this_process_can_be_read(self) -> None:
        import os

        value = load_test.rss_mb(os.getpid())
        assert value is None or value > 10

    def _args(self, **overrides: Any) -> argparse.Namespace:
        base = {"max_error_rate": 0.01, "max_p95_ms": 0.0, "max_memory_growth": 10.0}
        return argparse.Namespace(**{**base, **overrides})

    def test_verdict_catches_each_problem(self) -> None:
        good = {"error_rate": 0.0, "p95_ms": 100.0, "p50_ms": 50.0}
        report: dict[str, Any] = {"baseline": good, "soak": good, "recovery": good}
        assert load_test.verdict(report, self._args()) == []
        bad = {
            "baseline": good,
            "soak": {"error_rate": 0.05, "p95_ms": 900.0, "p50_ms": 400.0},
            "recovery": {"error_rate": 0.1, "p95_ms": 300.0, "p50_ms": 200.0},
            "memory_mb": {"trend_mb_per_min": 25.0},
        }
        problems = load_test.verdict(bad, self._args(max_p95_ms=500))
        assert len(problems) == 5


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_load_test_runs_end_to_end_against_a_live_server(sample_docs_dir: Any) -> None:
    client = TestClient(api_app.app)
    assert client.post("/ingest", json={"source": str(sample_docs_dir)}).status_code == 200
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(api_app.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    generator = api_app.get_pipeline().generator
    with patch.object(generator, "generate_async", side_effect=_slow_answer):
        thread.start()
        while not server.started:
            time.sleep(0.05)
        try:
            code = load_test.main(
                [
                    "run",
                    "--url",
                    f"http://127.0.0.1:{port}",
                    "--concurrency",
                    "4",
                    "--duration",
                    "3",
                    "--baseline-requests",
                    "3",
                    "--memory-interval",
                    "1",
                ]
            )
        finally:
            server.should_exit = True
            thread.join(10)
    assert code == 0


async def _slow_answer(*_args: Any, **_kwargs: Any) -> str:
    import asyncio

    await asyncio.sleep(0.05)
    return "Answer [1]."


class TestRetries:
    def _error(self, status: int) -> Exception:
        error = RuntimeError(f"HTTP {status}")
        error.status_code = status  # type: ignore[attr-defined]
        return error

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_rejected_requests_are_not_retried(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.utils import retry

        sleeps: list[float] = []
        monkeypatch.setattr(retry.time, "sleep", sleeps.append)
        calls: list[int] = []

        def call() -> None:
            calls.append(1)
            raise self._error(status)

        with pytest.raises(RuntimeError):
            retry.retry_with_backoff(call)
        assert calls == [1] and sleeps == []

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
    def test_transient_failures_are_still_retried(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.utils import retry

        monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
        calls: list[int] = []

        def call() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise self._error(status)
            return "ok"

        assert retry.retry_with_backoff(call) == "ok"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_async_retries_follow_the_same_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.utils import retry

        async def no_sleep(_s: float) -> None:
            return None

        monkeypatch.setattr(retry.asyncio, "sleep", no_sleep)
        calls: list[int] = []

        async def call() -> None:
            calls.append(1)
            raise self._error(401)

        with pytest.raises(RuntimeError):
            await retry.async_retry_with_backoff(call)
        assert calls == [1]

    def test_real_sdk_errors_carry_a_status(self) -> None:
        import httpx
        import openai

        response = httpx.Response(422, request=httpx.Request("POST", "http://x"))
        error = openai.UnprocessableEntityError("bad", response=response, body=None)
        from src.utils.retry import _is_permanent

        assert _is_permanent(error) is True


def test_app_log_lines_are_shown_under_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    app_logger = logging.getLogger("src")
    monkeypatch.setattr(app_logger, "handlers", [])
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(settings, "log_level", "INFO")
    api_app._configure_logging()
    assert app_logger.level == logging.INFO
    assert app_logger.handlers
