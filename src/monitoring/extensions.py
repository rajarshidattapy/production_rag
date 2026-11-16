from __future__ import annotations

import logging
import threading
from typing import Any

from .config import settings
from .metrics import MetricsCollector
from .tracing import Tracer

logger = logging.getLogger(__name__)


class BaseExtension:
    """Base class for monitoring extensions.

    Implement these lifecycle hooks to intercept and monitor queries, steps,
    and model generations.
    """

    def on_query_start(self, question: str, metadata: dict[str, Any]) -> None:
        """Called when a query execution begins."""
        pass

    def on_query_end(
        self,
        answer: str,
        citations: list[Any],
        elapsed: float,
        tokens: dict[str, int],
        cost: float,
    ) -> None:
        """Called when a query execution completes successfully."""
        pass

    def on_query_error(self, exc: Exception) -> None:
        """Called when a query execution raises an exception."""
        pass

    def on_step_start(
        self,
        step_name: str,
        input_data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Called when a sub-step (e.g. retrieve, rerank, generate) starts."""
        pass

    def on_step_end(
        self,
        step_name: str,
        output_data: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when a sub-step completes successfully."""
        pass

    def on_step_error(self, step_name: str, exc: Exception) -> None:
        """Called when a sub-step raises an exception."""
        pass

    def on_generation_llm_call(
        self,
        model: str,
        provider: str,
        prompt: str,
        query: str,
        response: str,
        usage: dict[str, int],
        cost: float,
    ) -> None:
        """Called when generator.generate completes a model call."""
        pass

    def on_first_token(self, ttft: float) -> None:
        """Called when the first token is received in a streaming query."""
        pass


class OTelTracingExtension(BaseExtension):
    """Extension to handle OpenTelemetry tracing for pipeline operations."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()
        self._local = threading.local()

    @property
    def _ctx(self) -> Any:
        return getattr(self._local, "ctx", None)

    @_ctx.setter
    def _ctx(self, val: Any) -> None:
        self._local.ctx = val

    @property
    def _span(self) -> Any:
        return getattr(self._local, "span", None)

    @_span.setter
    def _span(self, val: Any) -> None:
        self._local.span = val

    @property
    def _step_contexts(self) -> dict[str, Any]:
        if not hasattr(self._local, "step_contexts"):
            self._local.step_contexts = {}
        return self._local.step_contexts

    @property
    def _step_spans(self) -> dict[str, Any]:
        if not hasattr(self._local, "step_spans"):
            self._local.step_spans = {}
        return self._local.step_spans

    @staticmethod
    def _execute(fn: Any, *args: Any, **kwargs: Any) -> None:
        # A telemetry failure must never break the query path.
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("OTel tracing backend error: %s", e)

    def on_query_start(self, question: str, metadata: dict[str, Any]) -> None:
        try:
            self._ctx = self.tracer.trace_step(
                "query", input={"question": question}, metadata=metadata
            )
            self._span = self._ctx.__enter__()
        except Exception as e:
            logger.warning("Failed to start query trace: %s", e)

    def on_query_end(
        self,
        answer: str,
        citations: list[Any],
        elapsed: float,
        tokens: dict[str, int],
        cost: float,
    ) -> None:
        if not hasattr(self, "_span") or not hasattr(self, "_ctx"):
            return

        def task():
            output_val = {"answer": answer, "citations_count": len(citations)}
            self._span["output"] = output_val
            self._ctx.__exit__(None, None, None)

        self._execute(task)

    def on_query_error(self, exc: Exception) -> None:
        if not hasattr(self, "_span") or not hasattr(self, "_ctx"):
            return

        def task():
            self._ctx.__exit__(type(exc), exc, exc.__traceback__)

        self._execute(task)

    def on_step_start(
        self,
        step_name: str,
        input_data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            ctx = self.tracer.trace_step(step_name, input=input_data, metadata=metadata)
            span = ctx.__enter__()
            self._step_contexts[step_name] = ctx
            self._step_spans[step_name] = span
        except Exception as e:
            logger.warning("Failed to start span for %s: %s", step_name, e)

    def on_step_end(
        self,
        step_name: str,
        output_data: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ctx = self._step_contexts.pop(step_name, None)
        span = self._step_spans.pop(step_name, None)
        if not ctx or not span:
            return

        def task():
            span["output"] = output_data
            if metadata:
                span["metadata"].update(metadata)
                # Reassign so the OTel adapter records the augmented metadata.
                span["metadata"] = span["metadata"]
            ctx.__exit__(None, None, None)

        self._execute(task)

    def on_step_error(self, step_name: str, exc: Exception) -> None:
        ctx = self._step_contexts.pop(step_name, None)
        self._step_spans.pop(step_name, None)
        if not ctx:
            return

        def task():
            ctx.__exit__(type(exc), exc, exc.__traceback__)

        self._execute(task)

    def on_generation_llm_call(
        self,
        model: str,
        provider: str,
        prompt: str,
        query: str,
        response: str,
        usage: dict[str, int],
        cost: float,
    ) -> None:
        active_span = None
        if self.tracer._active_spans:
            active_span = self.tracer._active_spans[-1]

        if active_span:

            def task():
                self.tracer.capture_generation(
                    span=active_span,
                    model=model,
                    provider=provider,
                    prompt=prompt,
                    query=query,
                    response=response,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

            self._execute(task)


class OTelMetricsExtension(BaseExtension):
    """Extension to publish metrics via OpenTelemetry collector."""

    def __init__(self, metrics: MetricsCollector | None = None) -> None:
        self.metrics = metrics or MetricsCollector(enabled=settings.enabled, prefix="rag")

    def _execute(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        # A telemetry failure must never break the query path. The underlying MetricsCollector
        # methods already keep their own local latencies/costs/counts regardless of whether
        # the OTel meter is configured, so nothing further is needed here on failure.
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("OTel Collector connection error: %s", e)

    def on_query_end(
        self,
        answer: str,
        citations: list[Any],
        elapsed: float,
        tokens: dict[str, int],
        cost: float,
    ) -> None:
        self._execute(self.metrics.record_latency, "query_total", elapsed)
        self._execute(self.metrics.record_query_count)
        self._execute(self.metrics.record_context_count, len(citations))
        self._execute(
            self.metrics.record_tokens,
            tokens.get("prompt", 0),
            tokens.get("completion", 0),
            tokens.get("total", 0),
        )
        self._execute(self.metrics.record_cost, cost)

    def on_query_error(self, exc: Exception) -> None:
        self._execute(self.metrics.record_error, "query", type(exc).__name__)

    def on_first_token(self, ttft: float) -> None:
        if not self.metrics.enabled:
            return

        def record_ttft():
            if not hasattr(self, "_ttft_hist") and self.metrics._meter is not None:
                self._ttft_hist = self.metrics._meter.create_histogram(
                    name=f"{self.metrics.prefix}_time_to_first_token_seconds",
                    description="Time to first token in streaming queries",
                    unit="s",
                )
            if hasattr(self, "_ttft_hist"):
                self._ttft_hist.record(ttft)

        self._execute(record_ttft)

    def on_step_end(
        self,
        step_name: str,
        output_data: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute(self.metrics.record_latency, step_name, elapsed)

    def on_step_error(self, step_name: str, exc: Exception) -> None:
        self._execute(self.metrics.record_error, step_name, type(exc).__name__)


class GuardrailExtension(BaseExtension):
    """Custom pluggable extension validating content for real-time safety.

    If output toxicity or length violations are detected, raises a ValueError
    to block the query response instantly.
    """

    def __init__(self, max_length: int = 5000, blocked_keywords: list[str] | None = None) -> None:
        self.max_length = max_length
        self.blocked_keywords = blocked_keywords or [
            "restricted_secret_api_key",
            "internal_confidential",
        ]

    def on_query_start(self, question: str, metadata: dict[str, Any]) -> None:
        for keyword in self.blocked_keywords:
            if keyword in question:
                raise ValueError(f"Query contains blocked keyword: {keyword}")

    def on_query_end(
        self,
        answer: str,
        citations: list[Any],
        elapsed: float,
        tokens: dict[str, int],
        cost: float,
    ) -> None:
        if len(answer) > self.max_length:
            raise ValueError("Response length exceeds guardrail configuration limit")
        for keyword in self.blocked_keywords:
            if keyword in answer:
                raise ValueError(
                    "Response blocked by real-time output policy (contains restricted keywords)"
                )
