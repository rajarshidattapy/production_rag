"""In-memory limiter for repeated authentication attempts from one client address."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque


class AttemptLimiter:
    """Counts attempts per key over a sliding window.

    State lives in this process, so each replica counts separately. Tracked keys are capped,
    so a flood of different addresses can't grow memory without limit.
    """

    def __init__(self, max_tracked: int = 10_000) -> None:
        self._max_tracked = max_tracked
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def blocked(self, key: str, max_attempts: int, window_seconds: float) -> bool:
        with self._lock:
            attempts = self._attempts.get(key)
            if attempts is None:
                return False
            cutoff = time.monotonic() - window_seconds
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if not attempts:
                del self._attempts[key]
                return False
            return len(attempts) >= max_attempts

    def record(self, key: str) -> None:
        with self._lock:
            self._attempts.setdefault(key, deque()).append(time.monotonic())
            self._attempts.move_to_end(key)
            while len(self._attempts) > self._max_tracked:
                self._attempts.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._attempts.clear()
