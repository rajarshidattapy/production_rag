This is a production-ready RAG (Retrieval-Augmented Generation) application: it ingests documents, indexes them in ChromaDB, retrieves relevant passages, asks an LLM to generate a cited answer, and exposes everything through an API, browser UI, CLI scripts, and MCP tools.

## Project root


| File / folder                                               | What it contains                                                                                                        |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `src` (src)                                                 | Main Python application code.                                                                                           |
| `tests` (tests)                                             | Automated tests for every major feature.                                                                                |
| `data` (data)                                               | Runtime data: uploaded documents, vector index, SQLite databases, evaluation datasets.                                  |
| `scripts` (scripts)                                         | Command-line utilities for ingesting, querying, evaluating, benchmarking, and administration.                           |
| `docs` (docs)                                               | Project documentation: [`README.md`](README.md) overview, [`api.md`](api.md) API index, [`novelties.md`](novelties.md) implemented capabilities, and [`files.md`](files.md) file map. |
| `index.html` (index.html)                                   | Single-page browser chat/upload UI served by the API.                                                                   |
| `pyproject.toml` (pyproject.toml)                           | Python project metadata, tooling configuration, lint/test settings.                                                     |
| `requirements.txt` (requirements.txt)                       | Python dependencies.                                                                                                    |
| [`setup.py`](http://setup.py) ([setup.py](http://setup.py)) | Package/install definition and console entry points.                                                                    |
| `src/deploy/Dockerfile`                                     | Container image build instructions for the API/worker.                                                                  |
| `src/deploy/docker-compose.yml`                             | Local multi-service deployment: app, worker, RabbitMQ, Redis, monitoring services.                                      |
| `.env` (R:/production-grade-rag.env)                        | Local runtime settings and secrets; do not commit/share it.                                                             |
| `.env.example` (R:/production-grade-rag.env.example)        | Safe template showing all available environment variables.                                                              |
| `.github` (R:/production-grade-rag.github)                  | GitHub automation: CI, Docker publishing, dependency updates, supply-chain checks.                                      |
| `graphify-out` (graphify-out)                               | Generated codebase graph/report artifacts; not application source.                                                      |
| `.venv` (R:/production-grade-rag.venv)                      | Local Python virtual environment.                                                                                       |
| `.pytest_cache`, `.mypy_cache`, `.ruff_cache`               | Generated caches from tests, type checking, and linting.                                                                |
| `.git`                                                      | Git history and repository metadata.                                                                                    |


## `src/` — application code


| Area               | Files                                                                                              | Responsibility                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| API layer          | [`api.py`](http://api.py) (src/[api.py](http://api.py))                                            | FastAPI app and HTTP endpoints: health, upload, ingest, query, streaming, users, chats, job status, metrics.                    |
| Core orchestration | [`pipeline.py`](http://pipeline.py) (src/[pipeline.py](http://pipeline.py))                        | The central RAG flow: load → screen → chunk → index → retrieve → generate → cite; also controls access and document visibility. |
| Settings           | [`config.py`](http://config.py) (src/[config.py](http://config.py))                                | Reads and validates `RAG_*` environment configuration.                                                                          |
| Background worker  | [`worker.py`](http://worker.py) (src/[worker.py](http://worker.py))                                | RabbitMQ consumer that processes queued ingestion jobs.                                                                         |
| MCP                | `mcp_[server.py](http://server.py)` (src/mcp\_[server.py](http://server.py))                       | Read-only MCP interface for external AI clients: search, ask, inspect passages/sections/documents.                              |
| Auditing           | [`audit.py`](http://audit.py) (src/[audit.py](http://audit.py))                                    | Audit logging for security-sensitive operations.                                                                                |
| Secrets            | `secret_[files.py](http://files.py)` (src/secret\_[files.py](http://files.py))                     | Supports reading secrets from mounted files, useful in containers/Kubernetes.                                                   |
| Shared state       | `shared_[state.py](http://state.py)` (src/shared\_[state.py](http://state.py))                     | Redis-backed embedding cache and distributed rate/attempt limits.                                                               |
| Prompts            | `prompt_[catalog.py](http://catalog.py)` (src/prompt\_[catalog.py](http://catalog.py))             | Defines built-in prompt templates.                                                                                              |
| Prompt versions    | `prompt_[management.py](http://management.py)` (src/prompt\_[management.py](http://management.py)) | Stores, versions, activates, validates, and rolls back prompts.                                                                 |


### `src/ingestion/` — document preparation


| File                                                                                      | Purpose                                                                                    |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [`loader.py`](http://loader.py) (src/ingestion/[loader.py](http://loader.py))             | Loads PDF, TXT, Markdown, and JSON files; extracts file/PDF metadata.                      |
| [`screening.py`](http://screening.py) (src/ingestion/[screening.py](http://screening.py)) | Detects/removes unsafe hidden text, prompt-injection signals, secrets, and PII.            |
| [`chunker.py`](http://chunker.py) (src/ingestion/[chunker.py](http://chunker.py))         | Splits documents into parent sections and retrieval chunks.                                |
| [`metadata.py`](http://metadata.py) (src/ingestion/[metadata.py](http://metadata.py))     | Builds and validates metadata such as versions, hashes, source details, and access fields. |
| [`jobs.py`](http://jobs.py) (src/ingestion/[jobs.py](http://jobs.py))                     | SQLite-backed ingestion-job state, progress, cancellation, retries.                        |
| [`queue.py`](http://queue.py) (src/ingestion/[queue.py](http://queue.py))                 | RabbitMQ publishing/consuming for background ingestion.                                    |


### `src/retrieval/` — finding relevant knowledge


| File                                                                                     | Purpose                                                                                          |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `vector_[store.py](http://store.py)` (src/retrieval/vector\_[store.py](http://store.py)) | ChromaDB storage, embeddings, language collections, filtering, deduplication, and vector search. |
| [`hybrid.py`](http://hybrid.py) (src/retrieval/[hybrid.py](http://hybrid.py))            | BM25 keyword retrieval plus vector search fused through reciprocal-rank fusion.                  |
| [`reranker.py`](http://reranker.py) (src/retrieval/[reranker.py](http://reranker.py))    | Cross-encoder reranking to reorder retrieved passages by relevance.                              |


### `src/generation/` — answer generation


| File                                                                                       | Purpose                                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [`generator.py`](http://generator.py) (src/generation/[generator.py](http://generator.py)) | Builds grounded prompts and generates RAG answers.                            |
| `llm_[client.py](http://client.py)` (src/generation/llm\_[client.py](http://client.py))    | Provider abstraction for OpenAI, Anthropic, and OpenRouter-compatible models. |
| [`citations.py`](http://citations.py) (src/generation/[citations.py](http://citations.py)) | Produces citation objects and concise source snippets.                        |


### `src/healing/` — self-correcting RAG

Optional self-healing logic that verifies the generated answer and retries intelligently.


| File                                                                                                         | Purpose                                                                          |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `state_[machine.py](http://machine.py)` (src/healing/state\_[machine.py](http://machine.py))                 | Controls verify → repair/retrieve again → abstain decision flow.                 |
| [`verifier.py`](http://verifier.py) (src/healing/[verifier.py](http://verifier.py))                          | Checks citation validity, retrieval quality, refusals, and answer grounding.     |
| `failure_[classifier.py](http://classifier.py)` (src/healing/failure\_[classifier.py](http://classifier.py)) | Identifies whether a failure is caused by retrieval, generation, citations, etc. |
| `query_[rewriter.py](http://rewriter.py)` (src/healing/query\_[rewriter.py](http://rewriter.py))             | Rewrites weak queries for a better second retrieval attempt.                     |
| `adaptive_[retrieval.py](http://retrieval.py)` (src/healing/adaptive\_[retrieval.py](http://retrieval.py))   | Widens or changes retrieval strategy after failures.                             |
| `query_[analysis.py](http://analysis.py)` (src/healing/query\_[analysis.py](http://analysis.py))             | Extracts query terms and measures context coverage.                              |
| [`integration.py`](http://integration.py) (src/healing/[integration.py](http://integration.py))              | Connects the healing engine to the normal RAG pipeline.                          |
| [`metrics.py`](http://metrics.py) (src/healing/[metrics.py](http://metrics.py))                              | Collects repair/self-healing metrics.                                            |
| [`types.py`](http://types.py) (src/healing/[types.py](http://types.py))                                      | Shared Pydantic models/enums for healing state.                                  |
| [`prompts.py`](http://prompts.py) (src/healing/[prompts.py](http://prompts.py))                              | Prompt templates used during verification and repair.                            |


### Other `src/` packages


| Folder                        | Purpose                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `chat` (src/chat)             | SQLite chat history, users, authentication, sessions, and rate limits.                           |
| `evaluation` (src/evaluation) | Golden-dataset evaluation for answer quality, retrieval quality, relevance, and faithfulness.    |
| `monitoring` (src/monitoring) | Prometheus metrics, OpenTelemetry tracing, Langfuse hooks, pricing/usage tracking wrappers.      |
| `translator` (src/translator) | English, German, and Spanish translation catalogs (`.po` source / `.mo` compiled).               |
| `utils` (src/utils)           | Shared retry logic, language translation helpers, page-number query parsing, and usage counters. |


## `scripts/` — commands for operating the system


| Script                                                                                                 | What it does                                                                |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| [`ingest.py`](http://ingest.py) (scripts/[ingest.py](http://ingest.py))                                | Index a document or directory from the command line.                        |
| [`query.py`](http://query.py) (scripts/[query.py](http://query.py))                                    | Ask the knowledge base from the command line.                               |
| [`evaluate.py`](http://evaluate.py) (scripts/[evaluate.py](http://evaluate.py))                        | Run answer-quality evaluation against a golden dataset.                     |
| `evaluate_[retrieval.py](http://retrieval.py)` (scripts/evaluate\_[retrieval.py](http://retrieval.py)) | Measure Recall, Precision, MRR, and NDCG for vector/hybrid/reranked search. |
| `load_[test.py](http://test.py)` (scripts/load\_[test.py](http://test.py))                             | Concurrent-load and soak testing for `/query`.                              |
| `operational_[drill.py](http://drill.py)` (scripts/operational\_[drill.py](http://drill.py))           | Exercises operational failure/recovery scenarios.                           |
| `bench_vector_[store.py](http://store.py)` (scripts/bench\_vector\_[store.py](http://store.py))        | Benchmarks vector-store performance.                                        |
| `cache_[models.py](http://models.py)` (scripts/cache\_[models.py](http://models.py))                   | Pre-downloads/caches ML models for deployment.                              |
| `compile_[locales.py](http://locales.py)` (scripts/compile\_[locales.py](http://locales.py))           | Compiles translation `.po` files into `.mo` files.                          |
| `manage_[prompts.py](http://prompts.py)` (scripts/manage\_[prompts.py](http://prompts.py))             | Create/list/activate/roll back prompts.                                     |
| `manage_[users.py](http://users.py)` (scripts/manage\_[users.py](http://users.py))                     | User/account administration.                                                |
| `bump_[version.py](http://version.py)` (scripts/bump\_[version.py](http://version.py))                 | Updates the project version consistently.                                   |


## `data/` — runtime and sample data


| Folder/file                             | Contents                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `sample_docs` (data/sample\_docs)       | Small example documents used for demos and retrieval evaluation.          |
| `golden_dataset` (data/golden\_dataset) | Question/answer datasets and expected sources for regression evaluation.  |
| `uploads/`                              | PDFs uploaded from the browser UI.                                        |
| `chroma_db/`, `chroma_server/`          | Persisted ChromaDB vector-store files.                                    |
| `bm25_index_rag_docs_en.json`           | Persisted English BM25 keyword index.                                     |
| `chat.db`                               | SQLite database for users, sessions, chats, messages, and ingestion jobs. |
| `prompts.db`                            | SQLite database holding managed prompt versions.                          |
| `eval_results/`                         | Generated evaluation results.                                             |
| `monitoring/`                           | Locally persisted monitoring/tracing-related runtime data.                |


## `tests/`

Each `test_*.py` mirrors a feature area: API, ingestion, retrieval, hybrid search, self-healing, document versions, access control, MCP, chat history, prompt rollback, monitoring, Redis state, and operational/load behavior. [`conftest.py`](http://conftest.py) (tests/[conftest.py](http://conftest.py)) provides common fixtures and test setup.

No files were changed while inspecting the project.
