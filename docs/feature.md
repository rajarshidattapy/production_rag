## Not implemented yet, but identified as next production upgrades:

### Before going to production

- **Run Docker and CI once, with a real LLM.** The stack has never run on Linux or in Docker, and CI has never run on GitHub.
- **Count unhandled 500s on the dashboard.** They currently skip the metrics, so the error rate can look healthy during an outage.
- **Stop anyone from wiping the store.** `/ingest` with `reset: true` deletes every user's documents.
- **Rate-limit logins by IP** if the server is public. The lockout is per account, and the reverse proxy can do this.
- **Per-user documents**, if users shouldn't see each other's PDFs. Any account can retrieve any uploaded document today.
- **Follow-up questions**, soon after launch. Answers don't use earlier messages in the chat.

### Fix when scaling

Roughly in the order they will hurt. Each says how it was confirmed: *reproduced*, *measured*, or *from the code*.

**Concurrent traffic**

- **The query embedding cache isn't thread-safe.** Concurrent queries can raise `KeyError` once the cache is full, which fails the request. Needs a lock. *Reproduced: 2,086 errors with 8 threads and a small cache, with thread switching forced to be frequent.*
- **All blocking work shares one thread pool** of CPU count + 4, capped at 32. Retrieval and reranking use it, and self-healing holds a thread through every LLM call in its loop, so slow LLM responses starve all other queries. A 504 doesn't free the thread. Needs separate pools or an async healing loop. *From the code.*
- **No per-user rate limits or spending caps.** Nothing limits LLM cost on `/query`, and self-healing can make several LLM calls per question. *From the code.*
- **LLM clients are created per call**, so connections aren't reused. Share one client per provider. *From the code.*
- **No response cache.** Repeated questions pay for the LLM every time. *From the code.*

**Hybrid search (BM25)**

- **A cold BM25 build blocks every query**, including vector-only ones in other languages, because it holds the pipeline lock while it builds. Build outside that lock. *Reproduced: another language's store waited out the whole build.*
- **Scoring runs inside the index lock and scans an array per query**, so hybrid queries run one at a time per language in each process. *Measured: 4.8 ms at 100k chunks, 62 ms at 1M, which caps it near 16 queries per second.*
- **Replicas rebuild the whole index when another replica ingests**, because staleness is detected by chunk count. This gets worse as the corpus grows, and the page filter adds a per-query Python scan too. Needs a shared search index, or syncing only the new chunks. *Reproduced: 4 full rebuilds during an 18 s ingest.*
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

- **SQLite and** `ingest_jobs.json` **tie the app to one host.** API replicas on separate machines need a database server for accounts, chats, and jobs. *From the code.*
- **Job status is a whole file, read on every poll** and rewritten on every save, holding up to 500 jobs. Move it to the database. *From the code.*
- `/upload` **holds each file in memory**, up to 50 MB per concurrent upload. Stream to disk. *From the code.*
- **The dashboard shows only the replica that answers it.** Use Prometheus and Grafana across replicas. *From the code.*
- **Chroma is only tested to 50k chunks** on a single node, with its index in memory. Load test it at your target size, and move to Qdrant only if that shows Chroma is the bottleneck. *Measured at 50k; nothing larger has been run.*

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
3. **Pull the image instead of building on the VM.** Replace `build:` with the image the publish workflow pushes to GHCR. Building on a 4 GB VM is slow, because it downloads PyTorch and the models.
4. **Give shutdown time to finish** with `stop_grace_period: 90s` on `api`. The Docker default of 10 s can cut off an indexing batch. That's recoverable with Run again, but it's avoidable.

1. 
2. **Restore a backup once**, so you know it works before you need it.



