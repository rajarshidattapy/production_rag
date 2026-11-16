# Production readiness

Use this as the release gate, not as a claim that every deployment is automatically safe.

## Required before launch

- HTTPS, authentication, and secret management are configured.
- Data, database, and broker volumes are durable and backup/restore has been tested.
- `/healthz`, `/readyz`, Prometheus scraping, and OTLP export are monitored.
- The full test suite, retrieval evaluation, and operational drill pass in the target-like environment.
- Golden-dataset thresholds and self-healing behavior are reviewed for the intended workload.
- Capacity tests cover expected document count, concurrent requests, queue backlog, and LLM-provider failures.

## Explicit decisions

Choose and document: retention policy, access model, per-user cost limits, incident owner, rollback procedure, and data-deletion process. Review [feature.md](feature.md) for known gaps before setting a production launch date.
