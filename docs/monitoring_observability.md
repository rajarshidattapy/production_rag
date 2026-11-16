# Monitoring and observability

## Metrics

`GET /metrics` exposes Prometheus-format HTTP and self-healing metrics. Scrape every replica with Prometheus and build Grafana dashboards and alerts from the aggregated data. `/healthz` is liveness; `/readyz` is readiness; `/stats` is a lightweight operational snapshot.

## OpenTelemetry

Set `MONITOR_ENABLED=true` to instrument the monitored pipeline. Configure `MONITOR_OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`) and `MONITOR_OTEL_SERVICE_NAME`. Metrics export to `/v1/metrics`; traces export to `/v1/traces`. Send traces to an OTLP collector connected to Grafana Tempo or another trace backend.

Pipeline traces cover query, retrieval, reranking, generation, and healing steps. They include structured input/output metadata plus model, usage, and cost attributes where available. Telemetry export is best effort and must never fail a user request.

## Practical alerts

Alert on readiness failures, elevated HTTP 5xx rate, p95 request latency, queue backlog, and sustained self-healing failures or abstentions. Do not use raw questions or documents as Prometheus labels; that would create unbounded cardinality and expose user data.
