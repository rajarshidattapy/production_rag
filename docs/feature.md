## Not implemented yet, but identified as next production upgrades:

### Fix when scaling

Roughly in the order they will hurt. Each says how it was confirmed: *reproduced*, *measured*, or *from the code*.

**Concurrent traffic**

- **The query embedding cache isn't thread-safe.** Concurrent queries can raise `KeyError` once the cache is full, which fails the request. Needs a lock. *Reproduced: 2,086 errors with 8 threads and a small cache, with thread switching forced to be frequent.*
- **All blocking work shares one thread pool** of CPU count + 4, capped at 32. Retrieval and reranking use it, and self-healing holds a thread through every LLM call in its loop, so slow LLM responses starve all other queries. A 504 doesn't free the thread. Needs separate pools or an async healing loop. *From the code.*
- **No per-user rate limits or spending caps.** Nothing limits LLM cost on `/query`, and self-healing can make several LLM calls per question. *From the code.*
- **LLM clients are created per call**, so connections aren't reused. Share one client per provider. *From the code.*
- **No response cache.** Repeated questions pay for the LLM every time. *From the code.*

**Hybrid search (BM25)**

- **The owner filter costs more as the corpus grows.** Each hybrid query fetches every chunk ID the user may read, then checks the whole index against them in Python, because the index holds no metadata. Store owners in the index, or give large tenants their own collections. *From the code.*
- **A cold BM25 build blocks every query**, including vector-only ones in other languages, because it holds the pipeline lock while it builds. Build outside that lock. *Reproduced: another language's store waited out the whole build.*
- **Scoring runs inside the index lock and scans an array per query**, so hybrid queries run one at a time per language in each process. *Measured: 4.8 ms at 100k chunks, 62 ms at 1M, which caps it near 16 queries per second.*
- **Replicas rebuild the whole index when another replica ingests**, because staleness is detected by chunk count. This gets worse as the corpus grows. Needs a shared search index, or syncing only the new chunks. *Reproduced: 4 full rebuilds during an 18 s ingest.*
- **The persisted index has a 500 MB cap** (`RAG_BM25_MAX_INDEX_MB`, roughly 500k chunks). Past that it isn't saved, so every restart rebuilds it. *From the code.*

**Indexing**

- **Embedding is the bottleneck.** PyTorch is pinned to one thread, so a million chunks would take about 70 hours. Use a GPU or more threads for the indexing worker. *Measured at about 4 chunks per second; the 70 hours is extrapolated.*
- **Batch size needs revisiting with faster embedding.** Small batches add overhead that the slow CPU model hides. *Measured: 551 s vs 360 s for 40,000 chunks with stub embeddings.*
- **Indexing runs inside the API process.** It competes with queries for CPU, a deploy interrupts it, and each replica has its own queue, where one big job makes everyone else's wait. Needs a separate worker service with a shared queue and per-user fairness. *From the code.*
- **Run again re-reads every file.** Track finished files by content hash so a resume skips them. *From the code.*

**Data that only grows**

- **Documents can't be deleted or updated.** There's no route for it (`delete_chunks` is unused), and an edited file adds new chunks beside the old ones, so answers can cite stale text. Needs a document table that tracks each file's chunks. *From the code.*
- **Uploads are never cleaned up.** `data/uploads` only grows, even after a chat is deleted. *From the code.*
- **Chat and message lists aren't paginated**, so a heavy user gets everything on every load. *From the code.*

**Moving to several servers**

- **Login limits are counted per process**, so each replica allows its own 20 attempts. A shared store such as Redis, or limits at the proxy, would fix it. *From the code.*
- **Use a Chroma server in production, not embedded Chroma.** Embedded 1.5.9 sometimes can't read a freshly written collection, and only a new client clears it ([upstream issue](https://github.com/chroma-core/chroma/issues/7946)). The app now reopens the client automatically, at most every 30 s, but each reopen leaves the old client in memory until it can be collected. The `docker-compose.yml` setup already uses a server. *Reproduced: 6 of 204 attempts embedded, 0 of 84 on a server.*
- **SQLite and** `ingest_jobs.json` **tie the app to one host.** API replicas on separate machines need a database server for accounts, chats, and jobs. *From the code.*
- **Job status is a whole file, read on every poll** and rewritten on every save, holding up to 500 jobs. Move it to the database. *From the code.*
- `/upload` **holds each file in memory**, up to 50 MB per concurrent upload. Stream to disk. *From the code.*
- **The dashboard shows only the replica that answers it.** Use Prometheus and Grafana across replicas. *From the code.*
- **Chroma is only tested to 50k chunks** on a single node, with its index in memory. Load test it at your target size, and move to Qdrant only if that shows Chroma is the bottleneck. *Measured at 50k; nothing larger has been run.*



## Additional features to add to Production RAG

### - MCP

When deploying, expose the RAG service as an MCP server with `search_knowledge_base` (required) and `ask_knowledge_base` (optional, grounded answer with citations); keep ingestion and admin operations private.

### Security and data protection

1. Missing improvement: Prompt Injection Defense — Detect and neutralize malicious instructions embedded in uploaded documents before they influence LLM behaviour.
2. Missing improvement: PII Detection and Redaction — Identify and redact sensitive information such as emails, phone numbers, credentials, and personal identifiers before indexing or generation.
3. Missing improvement: Document Provenance Tracking — Track each document's origin, uploader, upload timestamp, content hash, version, and trust level.
4. Missing improvement: Fine-Grained Authorization — Enforce document-, collection-, and role-level permissions during retrieval, not merely at the API endpoint.
5. Missing improvement: Audit Logging — Record security-sensitive actions such as uploads, queries, document modifications, deletions, and administrative operations without exposing sensitive content.
6. Missing improvement: Secrets Management — Store API keys, database credentials, and encryption keys using a dedicated secrets manager instead of relying solely on environment variables.

### Ingestion and document processing

7. Missing improvement: OCR Fallback — Automatically extract text from scanned PDFs and image-based documents when ordinary PDF text extraction fails.
8. Missing improvement: Ingestion Quality Validation — Detect corrupted files, empty extraction, malformed pages, unsupported formats, and extraction failures before indexing.
9. Missing improvement: Semantic Deduplication — Identify duplicate and near-duplicate documents or chunks to reduce redundant storage and retrieval noise.
10. Missing improvement: Document Versioning — Maintain document versions and ensure updated files replace or supersede stale indexed content.
11. Missing improvement: Document Structure Awareness — Preserve headings, sections, paragraphs, and hierarchical relationships during parsing and chunking.
12. Missing improvement: Table-Aware Extraction — Preserve table headers, rows, columns, and cell relationships instead of flattening tables into ambiguous text.
13. Missing improvement: Chunk Metadata Validation — Enforce a consistent metadata schema for document IDs, page numbers, sections, ownership, and source references.

### Retrieval and context engineering

14. Missing improvement: Parent-Child Retrieval — Retrieve precise child chunks for relevance while supplying their larger parent sections for sufficient context.
15. Missing improvement: Multi-Hop Retrieval — Retrieve and combine evidence from multiple documents when a question requires connecting several facts.
16. Missing improvement: Query Intent Classification — Classify questions into factual lookup, summarization, comparison, analytical, or multi-hop queries before choosing a retrieval strategy.
17. Missing improvement: Query Routing — Dynamically select vector, hybrid, metadata-filtered, or multi-hop retrieval based on query intent.
18. Missing improvement: Context Compression — Remove redundant passages and compress retrieved evidence while preserving information needed to answer accurately.
19. Missing improvement: Temporal Retrieval — Prefer current document versions and distinguish historical information from the latest available facts.
20. Missing improvement: Adaptive Search Weight Calibration — Tune BM25 and vector search weights against a representative evaluation dataset instead of relying on fixed defaults.
21. Missing improvement: Retrieval Regression Testing — Track Recall@K, Precision@K, MRR, and NDCG to detect retrieval-quality regressions after code or model changes.

### Self-healing and answer verification

23. Missing improvement: Claim-Level Verification — Break generated answers into atomic claims and independently verify each claim against its supporting retrieved evidence.
24. Missing improvement: Semantic Entailment Verification — Use an entailment model or equivalent semantic check to determine whether evidence actually supports a claim beyond lexical overlap.
25. Missing improvement: Verifier Ensemble — Combine deterministic validation, an LLM judge, and semantic verification to reduce dependence on a single evaluator.
26. Missing improvement: Source Conflict Resolution — Detect contradictory documents and resolve conflicts using source authority, document versions, and timestamps.
27. Missing improvement: Confidence Calibration — Calibrate verifier confidence and acceptance thresholds using labelled examples rather than relying entirely on manually selected thresholds.
28. Missing improvement: Expected-Utility Repair Selection — Choose the cheapest repair action likely to fix a diagnosed failure instead of blindly executing retries.
29. Missing improvement: Semantic Retry Termination — Stop healing attempts when new retrieval results provide no meaningful evidence improvement.
30. Missing improvement: Healing Cost Budget — Enforce per-query limits on additional LLM calls, tokens, retrieval attempts, and total healing latency.
31. Missing improvement: Adversarial Healing Evaluation — Test the healing pipeline against prompt injection, fabricated citations, misleading evidence, contradictory sources, and verifier failures.
32. Missing improvement: False-Acceptance Evaluation — Measure how often the verifier incorrectly approves unsupported answers and how often it rejects correct ones.

### Production reliability and infrastructure

33. Missing improvement: LLM Retry with Exponential Backoff — Retry transient provider failures using exponential delays and jitter to avoid synchronized retry storms.
34. Missing improvement: LLM Provider Fallback — Automatically switch to a configured alternative provider when the primary LLM becomes unavailable.
35. Missing improvement: LLM Circuit Breaker — Temporarily stop requests to an unhealthy provider and prevent repeated failures from exhausting resources.
36. Missing improvement: Request Cancellation Propagation — Cancel downstream retrieval, reranking, and LLM operations when the client disconnects or the request times out.
37. Missing improvement: Bulkhead Isolation — Isolate ingestion, retrieval, reranking, and generation resources so overload in one subsystem does not bring down the others.
38. Missing improvement: Idempotent Ingestion Jobs — Prevent duplicate job submissions from creating repeated or inconsistent document indexing operations.
39. Missing improvement: Dead-Letter Job Handling — Preserve failed ingestion jobs for inspection, diagnosis, and controlled reprocessing.
40. Missing improvement: Database Migration Framework — Version database schema changes and apply them safely across deployments.
41. Missing improvement: Dependency Vulnerability Scanning — Scan Python dependencies and container images for known security vulnerabilities.
42. Missing improvement: Software Bill of Materials (SBOM) — Generate a machine-readable inventory of dependencies to improve supply-chain transparency.
43. Missing improvement: Container Image Signing — Sign release images and verify their authenticity before deployment.
44. Missing improvement: Graceful Model Warm-Up — Load and initialize embedding, reranking, and other model components before accepting production traffic.
45. Missing improvement: Load and Soak Testing — Validate latency, throughput, memory consumption, and recovery behaviour under sustained concurrent traffic.



&nbsp;

## Deploy

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
5. 
6. **Restore a backup once**, so you know it works before you need it.



&nbsp;
