## Not implemented yet, but identified as next production upgrades:

### Stage 1: Ingestion

*Mostly `src/ingestion/`, plus ingestion in `src/pipeline.py` and `src/api.py`. Changing what gets extracted or stored means re-ingesting, so do this stage first.*

- **Temporal Retrieval** and **Source Conflict Resolution** need document versions, timestamps, and source authority.
- **Fine-Grained Authorization** needs roles or collections on each chunk, beyond today's `owner`.
- **Prompt Injection Defense** and **PII Detection** are cheapest here, before text reaches the index.

**Reading documents**

- **OCR Fallback:** Automatically extract text from scanned PDFs and image-based documents when ordinary PDF text extraction fails.
- **Ingestion Quality Validation:** Detect corrupted files, empty extraction, malformed pages, unsupported formats, and extraction failures before indexing.
- **Document Structure Awareness:** Preserve headings, sections, paragraphs, and hierarchical relationships during parsing and chunking. Markdown headings are now recorded as each chunk's `section`; PDF and plain-text structure isn't.
- **Table-Aware Extraction:** Preserve table headers, rows, columns, and cell relationships instead of flattening tables into ambiguous text.
- **Prompt Injection Defense:** Detect and neutralize malicious instructions embedded in uploaded documents before they influence LLM behaviour.
- **PII Detection and Redaction:** Identify and redact sensitive information such as emails, phone numbers, credentials, and personal identifiers before indexing or generation.

**Chunks and metadata**

- **Chunk Metadata Validation:** Enforce a consistent metadata schema for document IDs, page numbers, sections, ownership, and source references.
- **Semantic Deduplication:** Identify duplicate and near-duplicate documents or chunks to reduce redundant storage and retrieval noise.
- **Document Provenance Tracking:** Track each document's origin, uploader, upload timestamp, content hash, version, and trust level.

**Updating and removing documents**

- **Documents can't be deleted or updated.** There's no route for it (`delete_chunks` is unused), and an edited file adds new chunks beside the old ones, so answers can cite stale text. Needs a document table that tracks each file's chunks. *From the code.*
- **Document Versioning:** Maintain document versions and ensure updated files replace or supersede stale indexed content.
- **Uploads are never cleaned up.** `data/uploads` only grows, even after a chat is deleted. *From the code.*
- `/upload` **holds each file in memory**, up to 50 MB per concurrent upload. Stream to disk. *From the code.*

**Embedding and indexing jobs**

- **Embedding is the bottleneck.** PyTorch is pinned to one thread, so a million chunks would take about 70 hours. Use a GPU or more threads for the indexing worker. *Measured at about 4 chunks per second; the 70 hours is extrapolated.*
- **Batch size needs revisiting with faster embedding.** Small batches add overhead that the slow CPU model hides. *Measured: 551 s vs 360 s for 40,000 chunks with stub embeddings.*
- **Indexing runs inside the API process.** It competes with queries for CPU, a deploy interrupts it, and each replica has its own queue, where one big job makes everyone else's wait. Needs a separate worker service with a shared queue and per-user fairness. *From the code.*
- **Run again re-reads every file.** Track finished files by content hash so a resume skips them. *From the code.*
- **Idempotent Ingestion Jobs:** Prevent duplicate job submissions from creating repeated or inconsistent document indexing operations.
- **Dead-Letter Job Handling:** Preserve failed ingestion jobs for inspection, diagnosis, and controlled reprocessing.
- **Job status is a whole file, read on every poll** and rewritten on every save, holding up to 500 jobs. Move it to the database. *From the code.*

### Stage 2: Retrieval

*Mostly `src/retrieval/`, plus retrieval in `src/pipeline.py`.*

**Vector search**

- **The query embedding cache isn't thread-safe.** Concurrent queries can raise `KeyError` once the cache is full, which fails the request. Needs a lock. *Reproduced: 2,086 errors with 8 threads and a small cache, with thread switching forced to be frequent.*
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
- **Temporal Retrieval:** Prefer current document versions and distinguish historical information from the latest available facts.
- **Context Compression:** Remove redundant passages and compress retrieved evidence while preserving information needed to answer accurately.
- **Fine-Grained Authorization:** Enforce document-, collection-, and role-level permissions during retrieval, not merely at the API endpoint.

**Measuring quality**

- **Adaptive Search Weight Calibration:** Tune BM25 and vector search weights against a representative evaluation dataset instead of relying on fixed defaults.
- **Retrieval Regression Testing:** Track Recall@K, Precision@K, MRR, and NDCG to detect retrieval-quality regressions after code or model changes.

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
- **Source Conflict Resolution:** Detect contradictory documents and resolve conflicts using source authority, document versions, and timestamps.
- **Confidence Calibration:** Calibrate verifier confidence and acceptance thresholds using labelled examples rather than relying entirely on manually selected thresholds.

**Repairing answers**

- **Expected-Utility Repair Selection:** Choose the cheapest repair action likely to fix a diagnosed failure instead of blindly executing retries.
- **Semantic Retry Termination:** Stop healing attempts when new retrieval results provide no meaningful evidence improvement.
- **Healing Cost Budget:** Enforce per-query limits on additional LLM calls, tokens, retrieval attempts, and total healing latency.

**Testing the verifier**

- **Adversarial Healing Evaluation:** Test the healing pipeline against prompt injection, fabricated citations, misleading evidence, contradictory sources, and verifier failures.
- **False-Acceptance Evaluation:** Measure how often the verifier incorrectly approves unsupported answers and how often it rejects correct ones.

### Stage 5: API and users

*Mostly `src/api.py` and `src/chat/`.*

**Handling load**

- **All blocking work shares one thread pool** of CPU count + 4, capped at 32. Retrieval and reranking use it, and self-healing holds a thread through every LLM call in its loop, so slow LLM responses starve all other queries. A 504 doesn't free the thread. Needs separate pools or an async healing loop. *From the code.*
- **Bulkhead Isolation:** Isolate ingestion, retrieval, reranking, and generation resources so overload in one subsystem does not bring down the others.
- **Request Cancellation Propagation:** Cancel downstream retrieval, reranking, and LLM operations when the client disconnects or the request times out.
- **No per-user rate limits or spending caps.** Nothing limits LLM cost on `/query`, and self-healing can make several LLM calls per question. *From the code.*
- **Chat and message lists aren't paginated**, so a heavy user gets everything on every load. *From the code.*

**Security**

- **Audit Logging:** Record security-sensitive actions such as uploads, queries, document modifications, deletions, and administrative operations without exposing sensitive content.
- **Secrets Management:** Store API keys, database credentials, and encryption keys using a dedicated secrets manager instead of relying solely on environment variables.
- **Login limits are counted per process**, so each replica allows its own 20 attempts. A shared store such as Redis, or limits at the proxy, would fix it. *From the code.*

**Access for other tools**

- **MCP server:** when deploying, expose the RAG service as an MCP server with `search_knowledge_base` (required) and `ask_knowledge_base` (optional, grounded answer with citations); keep ingestion and admin operations private.

### Stage 6: Infrastructure

*Docker, CI, and running on several servers. Do this just before deploying.*

**Several servers**

- **Use a Chroma server in production, not embedded Chroma.** Embedded mode can fail to read a freshly written collection and does not suit replicas; keep Chroma behind the internal Docker network.
- **SQLite and** `ingest_jobs.json` **tie the app to one host.** API replicas on separate machines need a database server for accounts, chats, and jobs. *From the code.*
- **The dashboard shows only the replica that answers it.** Use Prometheus and Grafana across replicas. *From the code.*
- **Database Migration Framework:** Version database schema changes and apply them safely across deployments.

**Startup and load**

- **Graceful Model Warm-Up:** Load and initialize embedding, reranking, and other model components before accepting production traffic.
- **Load and Soak Testing:** Validate latency, throughput, memory consumption, and recovery behaviour under sustained concurrent traffic.

**Supply chain**

- **Dependency Vulnerability Scanning:** Scan Python dependencies and container images for known security vulnerabilities.
- **Software Bill of Materials (SBOM):** Generate a machine-readable inventory of dependencies to improve supply-chain transparency.
- **Container Image Signing:** Sign release images and verify their authenticity before deployment.

&nbsp;

## Final boss: Deploy

The cheapest setup that works is **one VM running Docker Compose**: the API, Chroma, and Caddy for HTTPS, all on the same machine. Expect about $26 to$**30 a month plus your LLM bill**. This setup hasn't been run yet, so check everything on the first deploy.

### What to use


| Choice                                                                  | Use                                            | Why                                                                                                        |
| ----------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Cloud                                                                   | **Vultr** Regular Cloud Compute, 2 vCPU / 4 GB | Cheapest of the three, and storage and transfer are included                                               |
| Terraform                                                               | **Skip for now**                               | Creating one VM by hand takes minutes. Add it when you need to rebuild servers or run a second environment |
| Ansible                                                                 | **Skip for now**                               | A short setup script covers one VM. Add it when you manage several servers                                 |
| AWS or GCP                                                              | Only if a client requires it                   | Same stack, but they cost more for the same machine (see below)                                            |
| SageMaker, Kubernetes, load balancer, managed databases, Grafana, Redis | **Skip**                                       | The LLM is an outside API and the local models run on CPU, so one VM needs none of these                   |


4 GB is the starting point: the API alone peaked at about 1.2 to 1.4 GB while indexing (measured on Windows), and the reranker and Chroma are on top of that and not yet measured. Check real use with `docker stats`, and move to 8 GB if it gets tight, or if you use the reranker heavily.

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

     Also set `FORWARDED_ALLOW_IPS=*` in `.env`, but only once port 8000 isn't published. Without it, every user shares Caddy's address and one login limit.
3. **Pull the image instead of building on the VM.** Replace `build:` with the image the publish workflow pushes to GHCR. Building on a 4 GB VM is slow, because it downloads PyTorch and the models.
4. **Give shutdown time to finish** with `stop_grace_period: 90s` on `api`. The Docker default of 10 s can cut off an indexing batch. That's recoverable with Run again, but it's avoidable.
5. **Restore a backup once**, so you know it works before you need it.



&nbsp;
