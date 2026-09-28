"""Healing metrics, exported to Prometheus and (when configured) OpenTelemetry.

Metric names (OTel name → Prometheus name):

    rag.healing.attempts           → rag_healing_attempts (histogram, per query)
    rag.healing.success            → rag_healing_success_total
    rag.healing.failures           → rag_healing_failures_total{failure_type}
    rag.healing.failure_type       → (same counter, the failure_type label)
    rag.healing.abstentions        → rag_healing_abstentions_total
    rag.healing.query_rewrites     → rag_healing_query_rewrites_total
    rag.healing.retrieval_retries  → rag_healing_retrieval_retries_total
    rag.healing.regenerations      → rag_healing_regenerations_total
    rag.healing.latency            → rag_healing_latency_seconds (histogram)

plus ``rag_healing_queries_total{outcome=passed|recovered|abstained}``.

The OTel meter is obtained lazily from the global meter provider; if
``src.monitoring`` never installed one (or OTel isn't installed), those calls
are no-ops and only Prometheus records.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram

logger = logging.getLogger(__name__)

HEALING_REGISTRY = CollectorRegistry()


class HealingMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.queries = Counter(
            "rag_healing_queries_total",
            "Healing-enabled queries by final outcome",
            ["outcome"],
            registry=r,
        )
        self.attempts = Histogram(
            "rag_healing_attempts",
            "Generate/verify attempts per healing-enabled query",
            buckets=(1, 2, 3, 4, 5, 6, 8),
            registry=r,
        )
        self.success = Counter(
            "rag_healing_success", "Queries that failed verification and were recovered", registry=r
        )
        self.failures = Counter(
            "rag_healing_failures",
            "Failed verifications by diagnosed failure type",
            ["failure_type"],
            registry=r,
        )
        self.abstentions = Counter(
            "rag_healing_abstentions", "Queries that abstained after exhausting healing", registry=r
        )
        self.query_rewrites = Counter(
            "rag_healing_query_rewrites", "Query rewrites performed", registry=r
        )
        self.retrieval_retries = Counter(
            "rag_healing_retrieval_retries", "Retrieval retries performed", registry=r
        )
        self.regenerations = Counter(
            "rag_healing_regenerations", "Answer regenerations performed", registry=r
        )
        self.latency = Histogram(
            "rag_healing_latency_seconds",
            "End-to-end latency of healing-enabled queries",
            buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
            registry=r,
        )
        self._otel: dict[str, Any] | None = None
        self._otel_lock = threading.Lock()

    # ------------------------------------------------------------------
    # OTel (lazy, best-effort)
    # ------------------------------------------------------------------

    def _otel_instruments(self) -> dict[str, Any]:
        if self._otel is not None:
            return self._otel
        with self._otel_lock:
            if self._otel is not None:
                return self._otel
            instruments: dict[str, Any] = {}
            try:
                from opentelemetry import metrics as otel_metrics

                meter = otel_metrics.get_meter("rag.healing")
                instruments = {
                    "attempts": meter.create_histogram("rag.healing.attempts", unit="1"),
                    "success": meter.create_counter("rag.healing.success", unit="1"),
                    "failures": meter.create_counter("rag.healing.failures", unit="1"),
                    "abstentions": meter.create_counter("rag.healing.abstentions", unit="1"),
                    "query_rewrites": meter.create_counter("rag.healing.query_rewrites", unit="1"),
                    "retrieval_retries": meter.create_counter(
                        "rag.healing.retrieval_retries", unit="1"
                    ),
                    "regenerations": meter.create_counter("rag.healing.regenerations", unit="1"),
                    "latency": meter.create_histogram("rag.healing.latency", unit="s"),
                }
            except Exception as exc:  # ImportError or SDK misconfiguration
                logger.debug("OTel healing instruments unavailable: %s", exc)
            self._otel = instruments
            return instruments

    def _otel_add(self, name: str, value: float = 1, attrs: dict[str, str] | None = None) -> None:
        inst = self._otel_instruments().get(name)
        if inst is None:
            return
        try:
            if hasattr(inst, "add"):
                inst.add(value, attrs or {})
            else:
                inst.record(value, attrs or {})
        except Exception as exc:
            logger.debug("OTel healing metric %s failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Recording API used by the healing engine
    # ------------------------------------------------------------------

    def record_failure(self, failure_type: str) -> None:
        self.failures.labels(failure_type=failure_type).inc()
        self._otel_add("failures", 1, {"failure_type": failure_type})

    def record_query_rewrite(self) -> None:
        self.query_rewrites.inc()
        self._otel_add("query_rewrites")

    def record_retrieval_retry(self) -> None:
        self.retrieval_retries.inc()
        self._otel_add("retrieval_retries")

    def record_regeneration(self) -> None:
        self.regenerations.inc()
        self._otel_add("regenerations")

    def record_outcome(self, outcome: str, attempts: int, latency: float) -> None:
        self.queries.labels(outcome=outcome).inc()
        self.attempts.observe(attempts)
        self.latency.observe(latency)
        self._otel_add("attempts", attempts)
        self._otel_add("latency", latency)
        if outcome == "recovered":
            self.success.inc()
            self._otel_add("success")
        elif outcome == "abstained":
            self.abstentions.inc()
            self._otel_add("abstentions")

    def value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Current value of a Prometheus sample (0.0 if never recorded). Handy for tests."""
        v = self.registry.get_sample_value(name, labels or {})
        return float(v) if v is not None else 0.0


healing_metrics = HealingMetrics(HEALING_REGISTRY)
