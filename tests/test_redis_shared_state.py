"""Optional Redis cache and cross-replica rate-limit behavior."""

from __future__ import annotations

import fnmatch
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import redis
from fastapi import HTTPException

from src import api as api_app
from src.config import settings
from src.retrieval.vector_store import _ChromaEmbeddingFunction
from src.shared_state import RedisAttemptLimiter, RedisEmbeddingCache


class FakeRedis:
    """Small command-level fake; production code still uses the official redis-py client."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.attempts: dict[str, list[str]] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int) -> bool:
        assert ex >= 60
        self.values[key] = value
        return True

    def eval(self, script: str, key_count: int, key: str, *args: Any) -> int:
        assert key_count == 1
        if "ZADD" in script:
            self.attempts.setdefault(key, []).append(str(args[0]))
            return 1
        return int(len(self.attempts.get(key, [])) >= int(args[1]))

    def scan_iter(self, match: str, count: int):
        assert count == 500
        yield from (key for key in self.attempts if fnmatch.fnmatch(key, match))

    def unlink(self, *keys: str) -> None:
        for key in keys:
            self.attempts.pop(key, None)


class BrokenRedis:
    def __getattr__(self, name: str):
        def unavailable(*args: Any, **kwargs: Any) -> None:
            raise redis.ConnectionError("offline")

        return unavailable


def _embedding_function(cache: RedisEmbeddingCache, calls: list[list[str]]):
    fn = _ChromaEmbeddingFunction("shared-model", query_cache_size=2, shared_cache=cache)

    def encode(input_texts: list[str]) -> list[list[float]]:
        calls.append(input_texts)
        return [[float(len(text)), 1.0] for text in input_texts]

    fn._encode = encode  # type: ignore[method-assign]
    return fn


def test_embedding_is_reused_by_another_process_without_storing_raw_query() -> None:
    client = FakeRedis()
    first_calls: list[list[str]] = []
    second_calls: list[list[str]] = []
    first = _embedding_function(
        RedisEmbeddingCache(client, "rag", "shared-model", 300), first_calls
    )
    second = _embedding_function(
        RedisEmbeddingCache(client, "rag", "shared-model", 300), second_calls
    )

    expected = first.embed_query(["private question"])
    actual = second.embed_query(["private question"])

    np.testing.assert_allclose(actual, expected)
    assert first_calls == [["private question"]]
    assert second_calls == []
    assert all("private question" not in key for key in client.values)
    assert second.cache_stats() == {"hits": 1, "misses": 0, "size": 1}


def test_query_lru_is_thread_safe() -> None:
    fn = _ChromaEmbeddingFunction("thread-test", query_cache_size=2)
    fn._encode = lambda values: [[float(len(value))] for value in values]  # type: ignore[method-assign]
    failures: list[Exception] = []

    def query(worker: int) -> None:
        try:
            for index in range(500):
                fn.embed_query([f"q-{(worker + index) % 5}"])
        except Exception as exc:  # pragma: no cover - assertion records the race
            failures.append(exc)

    threads = [threading.Thread(target=query, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert fn.cache_stats()["size"] <= 2


def test_rate_limit_is_shared_and_keys_hide_the_client_address() -> None:
    client = FakeRedis()
    first = RedisAttemptLimiter(client, "rag")
    second = RedisAttemptLimiter(client, "rag")
    address = "203.0.113.42"

    assert first.blocked(address, 2, 60) is False
    first.record(address)
    second.record(address)
    assert second.blocked(address, 2, 60) is True
    assert all(address not in key for key in client.attempts)

    first.clear()
    assert second.blocked(address, 2, 60) is False


def test_redis_outage_falls_back_to_the_local_limiter() -> None:
    limiter = RedisAttemptLimiter(BrokenRedis(), "rag")
    assert limiter.blocked("client", 1, 60) is False
    limiter.record("client")
    assert limiter.blocked("client", 1, 60) is True


def test_compose_keeps_redis_private_and_configures_the_api() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    redis_block = compose.split("  redis:", 1)[1].split("\n  chroma:", 1)[0]
    assert "redis:7.2.17-alpine" in redis_block
    assert "ports:" not in redis_block
    assert "RAG_REDIS_URL=redis://redis:6379/0" in compose


def test_readiness_fails_when_configured_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "redis_url", "redis://redis:6379/0")
    monkeypatch.setattr(api_app._models_warm, "is_set", lambda: True)
    monkeypatch.setattr(api_app, "get_redis_client", lambda url, timeout: BrokenRedis())

    with pytest.raises(HTTPException) as exc:
        api_app.readyz()
    assert exc.value.status_code == 503
