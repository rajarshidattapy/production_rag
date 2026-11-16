from __future__ import annotations

import logging
import threading
import uuid
import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .config import PricingConfig, settings

logger = logging.getLogger(__name__)


def _json_attribute(name: str, value: dict[str, Any] | None) -> dict[str, str]:
    """Store structured pipeline data as a bounded OTLP attribute."""
    return {name: json.dumps(value or {}, default=str)[:8192]}


class OTelSpan:
    """Mapping-compatible adapter around an OpenTelemetry span."""

    def __init__(self, span: Any, name: str, input_data: dict[str, Any], metadata: dict[str, Any]):
        self.span = span
        self.name = name
        self.data: dict[str, Any] = {
            "name": name,
            "input": input_data,
            "output": {},
            "metadata": metadata,
        }
        self.span.set_attributes(
            {"rag.span.kind": "pipeline_step", **_json_attribute("rag.input", input_data), **_json_attribute("rag.metadata", metadata)}
        )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        if key == "output":
            self.span.set_attributes(_json_attribute("rag.output", value))
        elif key == "metadata":
            self.span.set_attributes(_json_attribute("rag.metadata", value))

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self[key] = value

    def record_exception(self, exc: BaseException) -> None:
        from opentelemetry.trace import Status, StatusCode

        self.span.record_exception(exc)
        self.span.set_attribute("error.type", type(exc).__name__)
        self.span.set_status(Status(StatusCode.ERROR, str(exc)[:1024]))


class Tracer:
    """OpenTelemetry-based tracer for RAG pipeline steps.

    Usage:
        tracer = Tracer(enabled=settings.enabled)
        with tracer.trace_step("retrieve", input={"query": q}) as span:
            results = vector_store.search(q)
            span["output"] = {"results_count": len(results)}
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._otel_tracer: Any = None
        self._otel_provider: Any = None
        self._pricing = PricingConfig(settings.pricing_file)
        self._local = threading.local()
        if enabled:
            self._init_otel()

    @property
    def _trace_id(self) -> str:
        if not hasattr(self._local, "trace_id"):
            self._local.trace_id = ""
        return self._local.trace_id

    @_trace_id.setter
    def _trace_id(self, val: str) -> None:
        self._local.trace_id = val

    @property
    def _active_spans(self) -> list[Any]:
        if not hasattr(self._local, "active_spans"):
            self._local.active_spans = []
        return self._local.active_spans

    @_active_spans.setter
    def _active_spans(self, val: list[Any]) -> None:
        self._local.active_spans = val

    def _init_otel(self) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import RESOURCE_ATTRIBUTES, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self._otel_provider = TracerProvider(
                resource=Resource.create({RESOURCE_ATTRIBUTES.SERVICE_NAME: settings.otel_service_name})
            )
            self._otel_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces",
                        timeout=settings.otel_export_timeout_ms / 1000,
                    )
                )
            )
            self._otel_tracer = self._otel_provider.get_tracer(settings.otel_service_name)
            logger.info("OTel tracing initialised: endpoint=%s", settings.otel_exporter_otlp_endpoint)
        except Exception as exc:
            logger.warning("OTel tracing unavailable; keeping local trace context: %s", exc)

    def get_trace_id(self) -> str:
        if self._trace_id:
            return self._trace_id
        self._trace_id = str(uuid.uuid4())
        return self._trace_id

    @contextmanager
    def trace_step(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        if not self.enabled or self._otel_tracer is None:
            span: dict[str, Any] = {
                "name": name,
                "input": input or {},
                "output": {},
                "metadata": metadata or {},
            }
            self._active_spans.append(span)
            try:
                yield span
            finally:
                self._active_spans.pop()
                if not self._active_spans:
                    self._trace_id = ""
            return

        # start_as_current_span supplies parent/child context automatically, so Tempo
        # receives one trace per query with nested pipeline-step spans.
        with self._otel_tracer.start_as_current_span(name) as raw_span:
            current = OTelSpan(raw_span, name, input or {}, metadata or {})
            self._active_spans.append(current)
            self.get_trace_id()
            try:
                yield current
            except Exception as exc:
                current.record_exception(exc)
                raise
            finally:
                self._active_spans.pop()
                if not self._active_spans:
                    self._trace_id = ""

    def capture_generation(
        self,
        span: Any,
        model: str,
        provider: str,
        prompt: str,
        query: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        if not self.enabled or self._otel_tracer is None:
            return
        cost = self._pricing.get_cost(provider, model, prompt_tokens, completion_tokens)
        with self._otel_tracer.start_as_current_span(f"{getattr(span, 'name', 'generate')}.llm_call") as generation:
            generation.set_attributes(
                {
                    "gen_ai.system": provider,
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": prompt_tokens,
                    "gen_ai.usage.output_tokens": completion_tokens,
                    "rag.tokens.total": total_tokens,
                    "rag.cost.usd": cost,
                    "rag.prompt": (prompt + "\n" + query)[:8192],
                    "rag.response": response[:8192],
                }
            )
