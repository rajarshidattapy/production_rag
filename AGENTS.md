# Repository Documentation Contract

For every task that changes repository code, configuration, dependencies, public behavior, or project structure:

1. Read `docs/README.md`, `docs/files.md`, and every guide relevant to the change before making changes.
2. Update `docs/files.md` to keep the file/folder map and component descriptions accurate.
3. Update `docs/README.md` whenever the change affects setup, dependencies, configuration, architecture, operations, or user-visible behavior.
4. Update the relevant focused guide: `api.md`, `mcp.md`, `monitoring_observability.md`, `evaluation.md`, `self_healing.md`, `security.md`, `deploy.md`, `testing.md`, `production_ready.md`, or `novelties.md`.
5. Do not make documentation-only churn: if none of these documents is affected, state that explicitly in the final handoff.
6. Verify links, commands, and file paths added or changed in documentation.

Documentation is part of the production deliverable. Complete these steps before reporting implementation work as finished.  

## Documentation map

- `docs/README.md` — project overview, architecture, Quickstart, configuration, and guide index.
- `docs/api.md` — HTTP and MCP API contract index.
- `docs/mcp.md` — MCP enablement, tools, and exposure guidance.
- `docs/monitoring_observability.md` — Prometheus, OpenTelemetry, dashboards, and alerts.
- `docs/evaluation.md` — golden datasets, quality metrics, and evaluation commands.
- `docs/self_healing.md` — verification, repair, retry, and abstention behavior.
- `docs/security.md` — authentication, secrets, content screening, and operational hardening.
- `docs/deploy.md` — deployment, production readiness, and security deployment checklist.
- `docs/testing.md` — test commands and testing expectations.
- `docs/novelties.md` — implemented capabilities that distinguish the current system.
- `docs/production_ready.md` — legacy path; deployment readiness now lives in `docs/deploy.md`.
- `docs/files.md` — file and folder map for the codebase.
- `docs/feature.md` — future features and known gaps.
- `docs/read.md` — topics to research before implementation.
