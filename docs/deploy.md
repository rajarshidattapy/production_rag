# Deployment/ Production ready and Security

## Local Compose

```bash
docker compose -f src/deploy/docker-compose.yml up --build
```

The deployment definition runs the API, ingestion worker, RabbitMQ, Redis, and supporting services. Provide secrets through environment variables or mounted secret files; never bake `.env` into an image.

## Current deployment decision: cheapest AWS path

Run the whole stack on **one AWS Lightsail Linux instance** with Docker Compose. Lightsail is the VM, CPU, RAM, SSD storage, public IP, and transfer allowance in one fixed-price bundle; it runs the API, worker, Chroma, RabbitMQ, Redis, and persistent Docker/data volumes. Cloudflare is not a replacement for that server or storage.

Start with the **8 GB RAM / 2 vCPU / 160 GB SSD** bundle. Its current public-IPv4 price is about **$44/month**. The 4 GB / 80 GB plan is about $24/month, but is appropriate only for light internal use: API and indexing worker can both load models while Chroma and the broker run. Confirm current regional pricing before purchase in the [Lightsail pricing page](https://aws.amazon.com/lightsail/pricing/).

| Component | Use now | Defer until there is a demonstrated need |
| --- | --- | --- |
| Compute and storage | One Lightsail instance with persistent disk/volumes | EC2 fleet, ECS, EKS/Kubernetes |
| Application runtime | Docker Compose | Kubernetes and a load balancer |
| Deployment automation | Manual console + Compose commands | Terraform and Ansible |
| Reverse proxy | None while using a raw IP for internal testing | Caddy once a domain exists |
| DNS/CDN/tunnel | None without a domain | Cloudflare DNS, Tunnel, and Access |
| Backups | Daily Lightsail snapshots plus an off-instance copy of `data/` | Managed database/storage services |

### No-domain operation

Without a domain, access the service at `http://<static-ip>:8000`. Attach a Lightsail static IP so it does not change after a stop/start. This is suitable only for private testing because normal trusted HTTPS certificates require a hostname; keep `RAG_SESSION_COOKIE_SECURE=false` only in this temporary HTTP mode and do not send sensitive documents or real credentials over the public internet.

Cloudflare Quick Tunnels can produce a temporary `trycloudflare.com` URL without a domain, but they are development-only: URLs change on restart and they do not support this application's SSE streaming endpoint. A stable Cloudflare Tunnel requires a domain managed in Cloudflare. Do not add Caddy until a domain exists, because it cannot obtain a normal public certificate for a raw IP address.

### First deployment runbook

1. Create the Lightsail instance and attach a static IP.
2. Allow SSH only from your IP; allow TCP 8000 only while testing the raw-IP setup.
3. Install Docker Engine and the Compose plugin.
4. Clone the repository, create `secrets/` with owner-readable secret files, and set production variables in a non-committed `.env`.
5. Start the stack with `docker compose -f src/deploy/docker-compose.yml up -d --build`.
6. Verify `http://<static-ip>:8000/healthz`, then `/readyz`, and inspect `docker compose ... ps` and logs.
7. Enable daily snapshots and test restoring both the instance and application data before trusting the deployment.

Do not expose Chroma, Redis, or RabbitMQ ports publicly. Keep `RAG_ALLOW_STORE_RESET=false`, set a strong `RAG_API_KEY`, and restrict `RAG_CORS_ORIGINS` even for a small deployment.

## Production checklist

- Terminate TLS at a reverse proxy and do not expose an unencrypted API with session cookies.
- Set `RAG_SESSION_COOKIE_SECURE=true` under HTTPS.
- Use durable volumes for data, Chroma, SQLite, and message-broker state.
- Configure `RAG_API_KEY`, host validation for MCP, and a restrictive CORS policy.
- Run the API and worker separately; monitor queue depth and worker health.
- Scrape Prometheus and export OTLP telemetry before relying on the service operationally.

### Stage 6: Infrastructure

*Docker, CI, and running on several servers. Do this just before deploying.*

**Several servers**

- **SQLite and** `ingest_jobs.json` **tie the app to one host.** API replicas on separate machines need a database server for accounts, chats, and jobs. *From the code.*
- **The dashboard shows only the replica that answers it.** Use Prometheus and Grafana across replicas. *From the code.*
- **Database Migration Framework:** Version database schema changes and apply them safely across deployments.



## Final boss: Deploy

The cheapest setup that works is **one VM running Docker Compose**: the API, indexing worker, RabbitMQ, Chroma, Redis, and Caddy for HTTPS, all on the same machine. Expect about $26 to$**30 a month plus your LLM bill**. This setup hasn't been run yet, so check everything on the first deploy.

### What to use


| Choice                                                           | Use                                            | Why                                                                                                        |
| ---------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Cloud                                                            | **Vultr** Regular Cloud Compute, 2 vCPU / 4 GB | Cheapest of the three, and storage and transfer are included                                               |
| Terraform                                                        | **Skip for now**                               | Creating one VM by hand takes minutes. Add it when you need to rebuild servers or run a second environment |
| Ansible                                                          | **Skip for now**                               | A short setup script covers one VM. Add it when you manage several servers                                 |
| AWS or GCP                                                       | Only if a client requires it                   | Same stack, but they cost more for the same machine (see below)                                            |
| SageMaker, Kubernetes, load balancer, managed databases, Grafana | **Skip**                                       | The LLM is an outside API and the local models run on CPU, so one VM needs none of these                   |


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

1. **Install CPU-only PyTorch in** `src/deploy/Dockerfile`**.** The image currently pulls the full GPU build, which adds several GB and does nothing on a CPU VM.
2. **Add Caddy for HTTPS** and stop publishing port 8000, which sends passwords and session cookies unencrypted. In `src/deploy/docker-compose.yml`, remove `ports` from `api`, then add this service and a `caddy_data` entry under `volumes`:
   ```yaml
         caddy:
           image: caddy:2
           restart: unless-stopped
           ports: ["80:80", "443:443"]
           command: caddy reverse-proxy --from rag.example.com --to api:8000
           volumes: [caddy_data:/data]
   ```
   ```
   Also set `FORWARDED_ALLOW_IPS=*` in `.env`, but only once port 8000 isn't published. Without it, every user shares Caddy's address and one login limit. With `RAG_MCP_ENABLED=true`, also set `RAG_MCP_ALLOWED_HOSTS=rag.example.com`, or `/mcp` refuses requests for that host.
   ```
3. **Pull the image instead of building on the VM.** Replace `build:` with the image the publish workflow pushes to GHCR. Building on a 4 GB VM is slow, because it downloads PyTorch and the models.
4. **Give the indexing worker shutdown time** with `stop_grace_period: 90s` on `worker` (already set in Compose). The Docker default of 10 s can cut off a batch; RabbitMQ requeues the unacknowledged job, but a graceful stop avoids repeated work.
5. **Restore a backup once**, so you know it works before you need it.



&nbsp;

&nbsp;

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



&nbsp;

# Security

## Baseline controls

- Protect the service with `RAG_API_KEY` and HTTPS in production.
- Store provider keys in environment variables or secret files; never commit `.env`.
- Use secure session cookies under TLS and configure restrictive CORS origins.
- Keep `RAG_ALLOW_STORE_RESET=false` outside controlled administrative workflows.
- Restrict MCP hosts with `RAG_MCP_ALLOWED_HOSTS`.

## Content safety

Before indexing, the service screens hidden control text, prompt-injection signals, credentials, and configurable PII. Retrieved sources are presented as quoted data, never executable instructions. These controls reduce risk; they are not a guarantee against novel attacks.

## Operational safety

Avoid sensitive values in logs, Prometheus labels, and trace attributes. Restrict access to `/metrics`, backups, data volumes, and the vector store. Rotate credentials after exposure and test restoration from backups.
