# API reference

The FastAPI schema is the source of truth while the service is running: `GET /openapi.json` and the interactive `GET /docs` UI. This page is the compact operational index.

## Authentication

When `RAG_API_KEY` is set, send `Authorization: Bearer <key>` to protected endpoints. Browser users authenticate with the session cookie created by `/auth/register` or `/auth/login`. Health checks and the UI do not require either.

## Service and observability

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness; does not initialize heavy models. |
| GET | `/readyz` | Readiness: models, database, and search indexes are available. |
| GET | `/stats` | Pipeline and store statistics. |
| GET | `/metrics` | Prometheus-format application metrics. |
| GET | `/` | Browser UI. |
| GET | `/docs.html` | Documentation browser. |
| GET | `/documentation/{name}` | Read one public Markdown file from `docs/`. |

## Knowledge base

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/upload` | Upload a PDF for later asynchronous ingestion. |
| POST | `/ingest` | Ingest a server-side file or directory. |
| POST | `/ingest/async` | Queue ingestion; returns `202` and a job ID. |
| GET | `/ingest/jobs` | List the signed-in user's recent jobs. |
| GET | `/ingest/jobs/{job_id}` | Read a job's status and progress. |
| POST | `/ingest/jobs/{job_id}/cancel` | Request cancellation. |
| POST | `/ingest/jobs/{job_id}/retry` | Retry a failed or cancelled job. |
| POST | `/query` | Return an answer and citation previews. |
| POST | `/query/stream` | Stream answer events as server-sent events. |
| GET | `/chunks/{chunk_id}` | Fetch the full passage behind a citation preview. |

`/query` and `/query/stream` accept a question plus optional retrieval controls: `top_k`, `use_hybrid`, `use_reranker`, and `self_heal`. Responses contain `answer` and `citations`; healed requests may also contain `healing` metadata. The API returns `X-Request-ID` on responses and usage headers where available.

## Accounts and chats

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/auth/register` | Create an account and session. |
| POST | `/auth/login` | Create a session. |
| POST | `/auth/logout` | End the current session. |
| GET | `/auth/me` | Read the current account. |
| GET, POST | `/chats` | List or create chats. |
| POST | `/chats/import` | Import a chat idempotently by `import_id`. |
| GET, DELETE | `/chats/{chat_id}` | Read or delete a chat. |
| POST | `/chats/{chat_id}/files` | Attach uploaded files to a chat. |
| POST | `/chats/{chat_id}/messages` | Send a chat turn and receive an answer. |

## MCP

`/mcp` exposes the read-only MCP server when enabled. It provides knowledge-base search, question answering, passage/section lookup, document listing, and store information. See the generated MCP tool schema for exact inputs.

## Errors

The API uses standard HTTP status codes: `400` for invalid requests, `401`/`403` for authentication or authorization failures, `404` for missing resources, `429` for rate limits, `503` for unavailable dependencies, and `5xx` for server or upstream failures.
