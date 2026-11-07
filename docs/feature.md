## Not implemented yet, but identified as next production upgrades:

**Reading documents**

- **Injection detection only knows its patterns.** Paraphrased or novel instructions aren't caught, and the prompt's caution is the only defence then. A trained classifier (e.g. a fine-tuned prompt-injection model) would catch more, at extra CPU time per chunk. *From the code.*
- **OCR Fallback:** Automatically extract text from scanned PDFs and image-based documents when ordinary PDF text extraction fails.
- **Ingestion Quality Validation:** Detect corrupted files, empty extraction, malformed pages, unsupported formats, and extraction failures before indexing.
- **Document Structure Awareness:** Preserve headings, sections, paragraphs, and hierarchical relationships during parsing and chunking. Markdown headings are now recorded as each chunk's `section`; PDF and plain-text structure isn't.
- **Table-Aware Extraction:** Preserve table headers, rows, columns, and cell relationships instead of flattening tables into ambiguous text.

**Updating and removing documents**

- **Documents can't be deleted or updated.** There's no route for it (`delete_chunks` is unused), and an edited file adds new chunks beside the old ones, so answers can cite stale text. Needs a document table that tracks each file's chunks. The MCP `list_documents` tool would read it too: today it reads the metadata of every chunk again after each change to the store, on each replica. *From the code.*
- **Document Versioning:** Maintain document versions and ensure updated files replace or supersede stale indexed content. Each version is now recorded (`version_id`, `document_key`, `document_date`), and an edited file is stored beside the old one; nothing yet hides or removes the old version.
- **Uploads are never cleaned up.** `data/uploads` only grows, even after a chat is deleted. *From the code.*
- `/upload` **holds each file in memory**, up to 50 MB per concurrent upload. Stream to disk. *From the code.*

**Embedding and indexing jobs**

- **Embedding is the bottleneck.** PyTorch is pinned to one thread, so a million chunks would take about 70 hours. Use a GPU or more threads for the indexing worker. *Measured at about 4 chunks per second; the 70 hours is extrapolated.*
- **Batch size needs revisiting with faster embedding.** Small batches add overhead that the slow CPU model hides. *Measured: 551 s vs 360 s for 40,000 chunks with stub embeddings.*
- **The shared ingest queue has no per-user fairness.** One large tenant can make every later tenant wait behind its jobs; schedule across users before adding more workers. *From the code.*
- **Run again re-reads every file.** Track finished files by content hash so a resume skips them. *From the code.*
- **Idempotent Ingestion Jobs:** Prevent duplicate job submissions from creating repeated or inconsistent document indexing operations.
- **Dead-Letter Job Handling:** Preserve failed ingestion jobs for inspection, diagnosis, and controlled reprocessing.
- **The audit log doesn't record what the indexing worker does.** The API records each queued, cancelled, and retried job, but the worker's own outcome, including a `reset` that wipes every document, isn't written. Call `audit.record` in `src/worker.py` when a job finishes. *From the code.*

### Stage 2: Retrieval

*Mostly `src/retrieval/`, plus retrieval in `src/pipeline.py`.*

**Vector search**

- **Chroma is only tested to 50k chunks** on a single node, with its index in memory. Load test it at your target size, and move to Qdrant only if that shows Chroma is the bottleneck. *Measured at 50k; nothing larger has been run.*

**Hybrid search (BM25)**

- **The owner filter costs more as the corpus grows.** Each hybrid query fetches every chunk ID the user may read, then checks the whole index against them in Python, because the index holds no metadata. Store owners in the index, or give large tenants their own collections. *From the code.*
- **A cold BM25 build blocks every query**, including vector-only ones in other languages, because it holds the pipeline lock while it builds. Build outside that lock. *Reproduced: another language's store waited out the whole build.*
- **Scoring runs inside the index lock and scans an array per query**, so hybrid queries run one at a time per language in each process. *Measured: 4.8 ms at 100k chunks, 62 ms at 1M, which caps it near 16 queries per second.*
- **Replicas rebuild the whole index when another replica ingests**, because staleness is detected by chunk count. This gets worse as the corpus grows. Needs a shared search index, or syncing only the new chunks. *Reproduced: 4 full rebuilds during an 18 s ingest.*
- **The persisted index has a 500 MB cap** (`RAG_BM25_MAX_INDEX_MB`, roughly 500k chunks). Past that it isn't saved, so every restart rebuilds it. *From the code.*

**Understanding the question**

- **Conversation-Aware Retrieval:** Resolve references such as "that document" and "the previous method" using relevant conversation context.
- **Query Intent Classification:** Classify questions into factual lookup, summarization, comparison, analytical, or multi-hop queries before choosing a retrieval strategy.
- **Query Routing:** Dynamically select vector, hybrid, metadata-filtered, or multi-hop retrieval based on query intent.
- **Multi-Hop Retrieval:** Retrieve and combine evidence from multiple documents when a question requires connecting several facts.

**Choosing and shaping context**

- **Parent-Child Retrieval:** Retrieve precise child chunks for relevance while supplying their larger parent sections for sufficient context. Ingestion already records each chunk's `parent_id` and stores the parent's text (`RAGPipeline.get_parents`), so this is retrieval-only work.
- **Temporal Retrieval:** Prefer current document versions and distinguish historical information from the latest available facts. The metadata it needs is already recorded on every chunk: `document_key`, `version_id`, `document_date`, and `ingested_at`.
- **Context Compression:** Remove redundant passages and compress retrieved evidence while preserving information needed to answer accurately.
- **Fine-Grained Authorization:** Enforce document-, collection-, and role-level permissions during retrieval, not merely at the API endpoint. Chunks and parents now carry `collection`, `restricted`, and `roles`, set per folder with `RAG_SOURCE_COLLECTIONS` and `RAG_SOURCE_ROLES`; users don't have roles yet, and retrieval doesn't filter on them. MCP clients have no roles either, so `/mcp` leaves out every restricted document.

### Stage 3: Generation

*Mostly `src/generation/`.*

- **LLM clients are created per call**, so connections aren't reused. Share one client per provider. *From the code.*
- **No response cache.** Repeated questions pay for the LLM every time. *From the code.*
- **LLM Retry with Exponential Backoff:** Retry transient provider failures using exponential delays and jitter to avoid synchronized retry storms.
- **LLM Provider Fallback:** Automatically switch to a configured alternative provider when the primary LLM becomes unavailable.
- **LLM Circuit Breaker:** Temporarily stop requests to an unhealthy provider and prevent repeated failures from exhausting resources.

### Stage 4: Verification and self-healing

*Mostly `src/healing/` and `src/evaluation/`.*

**Checking answers**

- **Claim-Level Verification:** Break generated answers into atomic claims and independently verify each claim against its supporting retrieved evidence.
- **Semantic Entailment Verification:** Use an entailment model or equivalent semantic check to determine whether evidence actually supports a claim beyond lexical overlap.
- **Verifier Ensemble:** Combine deterministic validation, an LLM judge, and semantic verification to reduce dependence on a single evaluator.
- **Source Conflict Resolution:** Detect contradictory documents and resolve conflicts using source authority, document versions, and timestamps. Chunks already carry `authority`, set per folder with `RAG_SOURCE_AUTHORITY`, plus versions and dates.
- **Confidence Calibration:** Calibrate verifier confidence and acceptance thresholds using labelled examples rather than relying entirely on manually selected thresholds.

**Repairing answers**

- **Expected-Utility Repair Selection:** Choose the cheapest repair action likely to fix a diagnosed failure instead of blindly executing retries.
- **Semantic Retry Termination:** Stop healing attempts when new retrieval results provide no meaningful evidence improvement.
- **Healing Cost Budget:** Enforce per-query limits on additional LLM calls, tokens, retrieval attempts, and total healing latency.

### Stage 5: API and users

*Mostly `src/api.py` and `src/chat/`.*

**Handling load**

- **Concurrent queries are CPU-bound.** The service's own time per query grew from about 60 ms alone to about 340 ms at 16 concurrent, capping throughput at about 12 queries per second with a 0.8 s LLM where 20 would be possible. Profile retrieval under load (query embedding, BM25, language detection) before adding replicas. *Measured with `scripts/load_test.py`: 2,219 queries in 3 minutes, p95 1.46 s.*
- **All blocking work shares one thread pool** of CPU count + 4, capped at 32. Retrieval and reranking use it, and self-healing holds a thread through every LLM call in its loop, so slow LLM responses starve all other queries. A 504 doesn't free the thread. Needs separate pools or an async healing loop. *From the code.*
- **Bulkhead Isolation:** Isolate ingestion, retrieval, reranking, and generation resources so overload in one subsystem does not bring down the others.
- **Request Cancellation Propagation:** Cancel downstream retrieval, reranking, and LLM operations when the client disconnects or the request times out.
- **No per-user rate limits or spending caps.** Nothing limits LLM cost on `/query` or MCP's `ask_knowledge_base`, and self-healing can make several LLM calls per question. MCP clients all use the one `RAG_API_KEY`, so one client can't be limited or revoked alone; give each its own key. *From the code.*
- **Chat and message lists aren't paginated**, so a heavy user gets everything on every load. *From the code.*

### Stage 6: Infrastructure

*Docker, CI, and running on several servers. Do this just before deploying.*

**Several servers**

- **SQLite and** `ingest_jobs.json` **tie the app to one host.** API replicas on separate machines need a database server for accounts, chats, and jobs. *From the code.*
- **The dashboard shows only the replica that answers it.** Use Prometheus and Grafana across replicas. *From the code.*
- **Database Migration Framework:** Version database schema changes and apply them safely across deployments.





## Final boss: Deploy

The cheapest setup that works is **one VM running Docker Compose**: the API, indexing worker, RabbitMQ, Chroma, Redis, and Caddy for HTTPS, all on the same machine. Expect about $26 to$**30 a month plus your LLM bill**. This setup hasn't been run yet, so check everything on the first deploy.

### What to use


| Choice                                                                  | Use                                            | Why                                                                                                        |
| ----------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Cloud                                                                   | **Vultr** Regular Cloud Compute, 2 vCPU / 4 GB | Cheapest of the three, and storage and transfer are included                                               |
| Terraform                                                               | **Skip for now**                               | Creating one VM by hand takes minutes. Add it when you need to rebuild servers or run a second environment |
| Ansible                                                                 | **Skip for now**                               | A short setup script covers one VM. Add it when you manage several servers                                 |
| AWS or GCP                                                              | Only if a client requires it                   | Same stack, but they cost more for the same machine (see below)                                            |
| SageMaker, Kubernetes, load balancer, managed databases, Grafana        | **Skip**                                       | The LLM is an outside API and the local models run on CPU, so one VM needs none of these                   |


4 GB is only a starting point: the API previously peaked at about 1.2 to 1.4 GB while indexing (measured on Windows), and the separate worker now loads its own embedding model while Redis, RabbitMQ, the reranker, and Chroma also need memory. Check real use with `docker stats`; expect to move to 8 GB if it gets tight or if you use the reranker heavily.

### Monthly cost

Prices checked in November 2026, mostly from third-party listings, so confirm them before buying. Storage and transfer are included only where noted.


| Option            | Machine                                 | Price     | Catch                                                  |
| ----------------- | --------------------------------------- | --------- | ------------------------------------------------------ |
| **Vultr Regular** | 2 vCPU / 4 GB, 80 GB SSD, 3 TB transfer | about $20 | Newer High Performance tier is about $24               |
| AWS Lightsail     | 2 vCPU / 4 GB, 80 GB SSD, 4 TB transfer | about $24 | Burstable CPU, so long indexing jobs can get throttled |
| GCP e2-medium     | 2 shared vCPU / 4 GB                    | about $24 | Shared cores; disk and outbound traffic cost extra     |
| AWS EC2 t3.medium | 2 vCPU / 4 GB                           | about $30 | Burstable CPU; disk and outbound traffic cost extra    |


Vultr's automatic backups add 20%, about $4 to$5. A domain is roughly $1 a month (an estimate). The LLM bill is separate and will probably be the part that changes most, so set a spending limit with your provider.

### Change in the repo first

1. **Install CPU-only PyTorch in the Dockerfile.** The image currently pulls the full GPU build, which adds several GB and does nothing on a CPU VM.
2. **Add Caddy for HTTPS** and stop publishing port 8000, which sends passwords and session cookies unencrypted. In `docker-compose.yml`, remove `ports` from `api`, then add this service and a `caddy_data` entry under `volumes`:
   ```yaml
        caddy:
          image: caddy:2
          restart: unless-stopped
          ports: ["80:80", "443:443"]
          command: caddy reverse-proxy --from rag.example.com --to api:8000
          volumes: [caddy_data:/data]
   ```

      Also set `FORWARDED_ALLOW_IPS=*` in `.env`, but only once port 8000 isn't published. Without it, every user shares Caddy's address and one login limit. With `RAG_MCP_ENABLED=true`, also set `RAG_MCP_ALLOWED_HOSTS=rag.example.com`, or `/mcp` refuses requests for that host.
3. **Pull the image instead of building on the VM.** Replace `build:` with the image the publish workflow pushes to GHCR. Building on a 4 GB VM is slow, because it downloads PyTorch and the models.
4. **Give the indexing worker shutdown time** with `stop_grace_period: 90s` on `worker` (already set in Compose). The Docker default of 10 s can cut off a batch; RabbitMQ requeues the unacknowledged job, but a graceful stop avoids repeated work.
5. **Restore a backup once**, so you know it works before you need it.



&nbsp;
