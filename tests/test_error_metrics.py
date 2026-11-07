"""Unhandled server errors must show up in the request metrics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src import api as api_app
from src.retrieval.vector_store import VectorStoreUnavailableError


@pytest.fixture(autouse=True)
def _fresh_pipeline() -> Iterator[None]:
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()


@pytest.fixture
def client() -> TestClient:
    # Like a real server: an unhandled exception becomes a 500 response.
    return TestClient(api_app.app, raise_server_exceptions=False)


def _requests(client: TestClient, path: str, status: str) -> float:
    """Requests to ``path`` that got ``status``, across every method."""
    total = 0.0
    for family in text_string_to_metric_families(client.get("/metrics").text):
        if family.name == "rag_http_requests":
            for sample in family.samples:
                if (
                    sample.name == "rag_http_requests_total"
                    and sample.labels["path"] == path
                    and sample.labels["status_code"] == status
                ):
                    total += sample.value
    return total


def _latency_count(client: TestClient, path: str) -> float:
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if (
                sample.name == "rag_http_request_duration_seconds_count"
                and sample.labels["path"] == path
            ):
                return sample.value
    return 0.0


def test_unhandled_exception_counts_as_a_500(client: TestClient) -> None:
    before = _requests(client, "/query", "500")
    latency_before = _latency_count(client, "/query")
    pipeline = api_app.get_pipeline()
    with patch.object(pipeline, "query_async", side_effect=RuntimeError("database on fire")):
        response = client.post("/query", json={"question": "anything", "self_heal": False})

    assert response.status_code == 500
    assert _requests(client, "/query", "500") == before + 1
    assert _latency_count(client, "/query") == latency_before + 1


def test_chat_route_failure_counts_under_its_route(client: TestClient) -> None:
    client.post("/auth/register", json={"username": "alice", "password": "long enough pw"})
    chat_id = client.post("/chats", json={}).json()["id"]
    path = "/chats/{chat_id}/messages"
    before = _requests(client, path, "500")
    pipeline = api_app.get_pipeline()
    with patch.object(pipeline, "query_async", side_effect=RuntimeError("boom")):
        response = client.post(
            f"/chats/{chat_id}/messages", json={"question": "anything", "self_heal": False}
        )

    assert response.status_code == 500
    assert _requests(client, path, "500") == before + 1


def test_the_exception_still_reaches_the_server(client: TestClient) -> None:
    strict = TestClient(api_app.app)  # re-raises what the app didn't handle
    pipeline = api_app.get_pipeline()
    with patch.object(pipeline, "query_async", side_effect=RuntimeError("database on fire")):
        with pytest.raises(RuntimeError, match="database on fire"):
            strict.post("/query", json={"question": "anything", "self_heal": False})


def test_normal_requests_are_counted_once(client: TestClient) -> None:
    before = _requests(client, "/healthz", "200")
    client.get("/healthz")
    assert _requests(client, "/healthz", "200") == before + 1


def _stream_errors(client: TestClient, reason: str) -> float:
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if (
                sample.name == "rag_http_stream_errors_total"
                and sample.labels["reason"] == reason
                and sample.labels["path"] == "/query/stream"
            ):
                return sample.value
    return 0.0


def _failing_stream(error: Exception) -> Any:
    async def stream(*_args: Any, **_kwargs: Any) -> Any:
        yield ("token", "partial answer ")
        raise error

    return stream


@pytest.mark.parametrize(
    ("error", "reason", "message"),
    [
        (RuntimeError("llm down"), "internal", "Internal server error"),
        (TimeoutError(), "timeout", "timed out"),
        (VectorStoreUnavailableError("gone"), "unavailable", "unavailable"),
    ],
)
def test_failure_during_a_stream_is_counted(
    client: TestClient, error: Exception, reason: str, message: str
) -> None:
    before = _stream_errors(client, reason)
    pipeline = api_app.get_pipeline()
    with patch.object(pipeline, "stream_answer", _failing_stream(error)):
        response = client.post("/query/stream", json={"question": "hello", "self_heal": False})

    assert response.status_code == 200  # already sent when the failure happened
    assert "partial answer" in response.text
    assert message in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert _stream_errors(client, reason) == before + 1


def test_failure_during_a_self_healing_stream_is_counted(client: TestClient) -> None:
    before = _stream_errors(client, "internal")
    pipeline = api_app.get_pipeline()
    with patch.object(
        pipeline, "query_with_healing_async", AsyncMock(side_effect=RuntimeError("down"))
    ):
        client.post("/query/stream", json={"question": "hello", "self_heal": True})
    assert _stream_errors(client, "internal") == before + 1


def test_a_bad_question_on_a_stream_is_not_a_server_error(client: TestClient) -> None:
    pipeline = api_app.get_pipeline()
    before = {r: _stream_errors(client, r) for r in ("internal", "timeout", "unavailable")}
    client.post("/query/stream", json={"question": "   ", "self_heal": False})
    with patch.object(pipeline, "stream_answer", _failing_stream(ValueError("bad input"))):
        client.post("/query/stream", json={"question": "hello", "self_heal": False})
    assert {r: _stream_errors(client, r) for r in before} == before


def test_a_successful_stream_is_not_counted_as_failed(client: TestClient) -> None:
    async def stream(*_args: Any, **_kwargs: Any) -> Any:
        yield ("token", "fine")
        yield ("citations", [])

    before = _stream_errors(client, "internal")
    with patch.object(api_app.get_pipeline(), "stream_answer", stream):
        client.post("/query/stream", json={"question": "hello", "self_heal": False})
    assert _stream_errors(client, "internal") == before


def test_unknown_urls_share_one_label(client: TestClient) -> None:
    before = _requests(client, "unmatched", "404")
    for i in range(5):
        assert client.get(f"/scanner/probe{i}.php?x={i}").status_code == 404
    metrics = client.get("/metrics").text
    assert "scanner" not in metrics
    assert _requests(client, "unmatched", "404") == before + 5


def test_wrong_method_does_not_record_the_raw_url(client: TestClient) -> None:
    client.put("/chats/abc123secret", json={})
    assert "abc123secret" not in client.get("/metrics").text


def test_real_routes_keep_their_template_on_a_404(client: TestClient) -> None:
    client.post("/auth/register", json={"username": "alice", "password": "long enough pw"})
    before = _requests(client, "/chats/{chat_id}", "404")
    assert client.get("/chats/no-such-chat").status_code == 404
    assert _requests(client, "/chats/{chat_id}", "404") == before + 1
    assert "no-such-chat" not in client.get("/metrics").text
