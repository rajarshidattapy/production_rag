"""Optional Redis-backed state shared by API replicas.

Redis is an optimization for query embeddings and the authority for cross-replica
authentication throttling. Operations fall back locally after a transient Redis failure so an
outage does not take down ordinary queries, while readiness reports the configured dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from typing import Any

from src.chat.rate_limit import AttemptLimiter

logger = logging.getLogger(__name__)

_clients: dict[tuple[str, float], Any] = {}
_clients_lock = threading.Lock()


def get_redis_client(url: str, socket_timeout: float) -> Any:
    """Return one pooled redis-py client per URL and timeout pair."""
    key = (url, socket_timeout)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            import redis

            client = redis.Redis.from_url(
                url,
                socket_connect_timeout=socket_timeout,
                socket_timeout=socket_timeout,
                health_check_interval=30,
                decode_responses=False,
            )
            _clients[key] = client
        return client


def close_redis_clients() -> None:
    """Close shared connection pools during application shutdown."""
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            logger.debug("Could not close a Redis client", exc_info=True)


class RedisEmbeddingCache:
    """Redis L2 cache for model/query embeddings; keys never contain raw query text."""

    def __init__(self, client: Any, prefix: str, model_name: str, ttl_seconds: int) -> None:
        self._client = client
        self._prefix = prefix.strip(":")
        self._model = hashlib.sha256(model_name.encode()).hexdigest()[:16]
        self._ttl_seconds = ttl_seconds
        self._warned = False
        self._warning_lock = threading.Lock()

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode()).hexdigest()
        return f"{self._prefix}:embedding:{self._model}:{digest}"

    def get(self, query: str) -> list[float] | None:
        try:
            value = self._client.get(self._key(query))
            self._warned = False
        except Exception as exc:
            self._warn_once(exc)
            return None
        if value is None:
            return None
        try:
            decoded = json.loads(value)
            if not isinstance(decoded, list) or not all(
                isinstance(v, int | float) for v in decoded
            ):
                raise ValueError("embedding is not a numeric list")
            return [float(v) for v in decoded]
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignored malformed embedding in Redis key %s", self._key(query))
            return None

    def set(self, query: str, vector: list[float]) -> None:
        try:
            self._client.set(
                self._key(query),
                json.dumps(vector, separators=(",", ":")),
                ex=self._ttl_seconds,
            )
            self._warned = False
        except Exception as exc:
            self._warn_once(exc)

    def _warn_once(self, exc: Exception) -> None:
        with self._warning_lock:
            if not self._warned:
                logger.warning("Redis embedding cache unavailable; using local cache: %s", exc)
                self._warned = True


_BLOCKED_SCRIPT = """
local now = redis.call('TIME')
local timestamp = tonumber(now[1]) + tonumber(now[2]) / 1000000
local window = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', timestamp - window)
local count = redis.call('ZCARD', KEYS[1])
redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
if count >= tonumber(ARGV[2]) then return 1 else return 0 end
"""

_RECORD_SCRIPT = """
local now = redis.call('TIME')
local timestamp = tonumber(now[1]) + tonumber(now[2]) / 1000000
redis.call('ZADD', KEYS[1], timestamp, ARGV[1])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""


class RedisAttemptLimiter:
    """Sliding-window attempt limiter shared by every process using the same Redis."""

    def __init__(self, client: Any, prefix: str) -> None:
        self._client = client
        self._prefix = prefix.strip(":")
        self._fallback = AttemptLimiter()
        self._window_ms = 300_000
        self._warned = False

    def _key(self, value: str) -> str:
        digest = hashlib.sha256(value.encode()).hexdigest()
        return f"{self._prefix}:auth-attempts:{digest}"

    def blocked(self, key: str, max_attempts: int, window_seconds: float) -> bool:
        self._window_ms = max(1, math.ceil(window_seconds * 1000))
        try:
            blocked = self._client.eval(
                _BLOCKED_SCRIPT, 1, self._key(key), window_seconds, max_attempts
            )
            self._warned = False
            return bool(blocked)
        except Exception as exc:
            self._warn_once(exc)
            return self._fallback.blocked(key, max_attempts, window_seconds)

    def record(self, key: str) -> None:
        try:
            self._client.eval(_RECORD_SCRIPT, 1, self._key(key), uuid.uuid4().hex, self._window_ms)
            self._warned = False
        except Exception as exc:
            self._warn_once(exc)
            self._fallback.record(key)

    def clear(self) -> None:
        self._fallback.clear()
        try:
            keys = list(self._client.scan_iter(match=f"{self._prefix}:auth-attempts:*", count=500))
            if keys:
                self._client.unlink(*keys)
        except Exception as exc:
            self._warn_once(exc)

    def _warn_once(self, exc: Exception) -> None:
        if not self._warned:
            logger.warning("Redis rate limiter unavailable; using local limiter: %s", exc)
            self._warned = True


def build_attempt_limiter(
    redis_url: str | None, prefix: str, socket_timeout: float
) -> AttemptLimiter | RedisAttemptLimiter:
    """Use Redis when configured and preserve the existing local default otherwise."""
    if not redis_url:
        return AttemptLimiter()
    return RedisAttemptLimiter(get_redis_client(redis_url, socket_timeout), prefix)
