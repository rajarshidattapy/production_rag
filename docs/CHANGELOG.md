# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Self-healing RAG (Phase 4, opt-in)** — `src/healing/`. A post-generation verifier
  (deterministic citation/retrieval/refusal checks, then an LLM judge with Pydantic-validated
  output) feeds a deterministic failure classifier (`RETRIEVAL_FAILURE`, `QUERY_FAILURE`,
  `GENERATION_FAILURE`, `CITATION_FAILURE`, `CONTEXT_FAILURE`, `UNKNOWN_FAILURE`). A small state
  machine repairs the diagnosed failure — query rewrite with adapted hybrid retrieval (wider
  candidate pool, re-balanced α, multi-query RRF fusion), context expansion, or regeneration with
  a repair prompt — re-verifies, and returns a structured abstention after
  `RAG_MAX_HEALING_ATTEMPTS`. See `docs/SELF_HEALING.md`.
- `/query` and `/query/stream` accept optional `self_heal`; responses carry an optional `healing`
  object (omitted when healing didn't run). SSE emits an extra `healing` event in healing mode.
- `RAGPipeline.query_with_healing[_async]()`; `query()`/`query_async()` accept `self_heal`.
- `rag.healing.*` metrics (Prometheus on `/metrics` + OTel) and per-attempt Langfuse spans
  (`retrieve_attempt_N`, `verify_attempt_N`, `query_rewrite`, …) via `MonitoredRAGPipeline`.
- Baseline-vs-healed evaluation (`src/evaluation/healing_eval.py`,
  `scripts/evaluate.py --healing-report`, `--self-heal`) and a hard dataset of paraphrased,
  terse, vague, multi-hop, and unanswerable questions
  (`data/golden_dataset/healing_dataset.jsonl`).
- `scripts/query.py --self-heal`.

### Changed (backward compatible)
- `RAGPipeline._retrieve` accepts optional `fetch_k` / `alpha` overrides and
  `_apply_context_budget` an optional `budget`; defaults reproduce previous behaviour.
- `HybridRetriever.search` accepts an optional per-call `alpha` (no shared-state mutation).
- `EvalExample` gains optional `expected_behavior` and `category` fields.

## [1.1.0] - 2026-07-02

### Security
- Closed an unauthenticated pre-auth RCE (chromadb CVE PYSEC-2026-311) exposed via `docker-compose.yml` publishing ChromaDB's raw REST API to the host network, bypassing this project's own API auth entirely.
- Fixed a timing side-channel in API key comparison (`secrets.compare_digest` instead of `!=`).
- Stopped `/readyz` from leaking internal exception details to unauthenticated clients.
- Fixed `/healthz` incorrectly requiring auth, which would have caused Docker/Kubernetes health checks to fail and restart the container in a loop whenever `RAG_API_KEY` was enabled.
- Fixed a mypy/numpy stub incompatibility that was silently breaking the CI type-check job on a fresh dependency install.

### Added
- **Prometheus `/metrics` endpoint** — request count and latency histograms by route, independent of the existing OTel/Langfuse pipeline.
- **Query embedding cache** — in-process LRU cache for repeated/paraphrased query embeddings (`RAG_EMBEDDING_QUERY_CACHE_SIZE`), ~10x faster on a cache hit; hit/miss stats surfaced via `/stats`.
- **Async ingestion** — `POST /ingest/async` + `GET /ingest/jobs/{job_id}` for background ingestion with job-ID polling, avoiding client/proxy timeouts on large corpora.
- **Full RAGAS-style evaluation suite** — added `ContextPrecisionScorer` (Average Precision @ k) and `ContextRecallScorer` (reference-answer statement attribution), completing all four standard RAG quality dimensions (faithfulness, answer relevance, context precision, context recall).

### Repository / CI
- Branch protection on `main` with required status checks.
- Dependabot enabled (pip, GitHub Actions, Docker base image) plus GitHub vulnerability alerts and automated security fixes.
- Added `SECURITY.md`.
- Docker publish workflow now cancels superseded in-progress builds on new pushes.

