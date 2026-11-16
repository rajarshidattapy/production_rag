# Novelties in this project

This page records implemented capabilities that distinguish the current RAG service. Planned work belongs in [feature.md](feature.md).

## Retrieval and answers

- Dense vector retrieval with optional BM25 hybrid retrieval and cross-encoder reranking.
- Page-aware PDF retrieval, with citations carrying source, passage preview, score, and page when available.
- Context screening for hidden text, prompt-injection signals, credentials, and configurable PII handling.
- Optional self-healing: verify an answer, classify failures, retry retrieval or generation, and abstain when evidence remains insufficient.
- English, German, and Spanish language-aware collections.

## Product and operations

- Synchronous and durable asynchronous ingestion, with RabbitMQ-backed jobs, progress, cancellation, and retry.
- Browser chat UI, persisted account/chat history, and a read-only MCP interface.
- Golden-dataset evaluation for answer quality, retrieval metrics, and baseline-versus-healed comparisons.
- Prometheus metrics plus OpenTelemetry traces and metrics exported through OTLP.
- OpenAI-, Anthropic-, and OpenRouter-compatible generation paths, with request usage tracking and retry handling.

## Design stance

The project favors grounded answers with explicit citations over unsupported responses. Optional features remain opt-in where they increase cost or latency: hybrid retrieval, reranking, self-healing, and monitoring instrumentation.
