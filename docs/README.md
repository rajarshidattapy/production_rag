<div align="center">

# Production-Grade RAG Pipeline

**Retrieval-augmented generation as a service: vector-search baseline, optional hybrid retrieval and cross-encoder reranking, multi-provider LLMs, MCP tool exposure, and end-to-end observability.**

[![CI](https://img.shields.io/github/actions/workflow/status/rajarshidattapy/production_rag/ci.yml?branch=master&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/rajarshidattapy/production_rag/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-proprietary-555555?style=flat-square)](../LICENSE)

[Quickstart](#quickstart) Â· [Architecture](#architecture) Â· [Configuration](#configuration) Â· [Deployment](#deployment) Â· [Observability](#observability)

</div>

---

| Layer | Stack |
| :-- | :-- |
| **Core** | ![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-2.14%2B-E92063?style=flat-square&logo=pydantic&logoColor=white) |
| **Retrieval & models** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5%2B-FFDE2D?style=flat-square) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white) ![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=flat-square&logo=anthropic&logoColor=white) ![OpenRouter](https://img.shields.io/badge/OpenRouter-6566F1?style=flat-square&logo=openrouter&logoColor=white) |
| **Observability** | ![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.30%2B-F5A800?style=flat-square&logo=opentelemetry&logoColor=white) ![Prometheus client](https://img.shields.io/badge/Prometheus_client-0.26%2B-E6522C?style=flat-square&logo=prometheus&logoColor=white) ![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=grafana&logoColor=white) |
| **Async & messaging** | ![Celery](https://img.shields.io/badge/Celery-37814A?style=flat-square&logo=celery&logoColor=white) ![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=flat-square&logo=rabbitmq&logoColor=white) ![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white) ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white) |
| **Infra & deploy** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white) ![Terraform](https://img.shields.io/badge/Terraform-844FBA?style=flat-square&logo=terraform&logoColor=white) ![Ansible](https://img.shields.io/badge/Ansible-EE0000?style=flat-square&logo=ansible&logoColor=white) ![Caddy](https://img.shields.io/badge/Caddy-1F88C0?style=flat-square&logo=caddy&logoColor=white) |
| **Cloud** | ![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonwebservices&logoColor=white) ![SageMaker](https://img.shields.io/badge/Amazon_SageMaker-FF9900?style=flat-square) ![Vultr](https://img.shields.io/badge/Vultr-007BFC?style=flat-square&logo=vultr&logoColor=white) |


---

## High-level overview

```
Documents -> Loader -> Chunker -> ChromaDB (dense vectors, per language)
                                     |
                                     +-> BM25 index (built on first hybrid query)

POST /ingest/async -> RabbitMQ -> indexing worker -> Loader (one job at a time)

Query -> Detect language -> Vector search            (default)
                            or Hybrid search         (use_hybrid=true)
                         -> Cross-encoder reranker   (only if use_reranker=true)
                         -> Context budget -> LLM -> Answer + citation list
                         -> Verify and repair        (only if self_heal=true)

MCP clients -> /mcp (read-only) -> search_knowledge_base, ask_knowledge_base, get_passage,
                               get_section, list_documents, knowledge_base_info
```

## Architecture

**Ingestion**

```mermaid
flowchart TD
    UI[Browser UI] -->|PDF only| UPLOAD["POST /upload<br/>checks .pdf extension and size"]
    UPLOAD --> SAVE[Save under data/uploads]
    SAVE --> ASYNC["POST /ingest/async"]
    ASYNC --> QUEUE[(RabbitMQ)]
    QUEUE --> WORKER["Indexing worker<br/>one job at a time"]
    WORKER --> INGEST[Ingest]
    SERVER["POST /ingest<br/>or scripts/ingest.py"] -->|path inside the data directory| INGEST
    INGEST --> LOAD["Load PDF, TXT, MD, or JSON<br/>(JSON is stored as raw text)"]
    LOAD --> CHUNK[Split into overlapping chunks]
    CHUNK --> LANG[Detect language per chunk]
    LANG --> EMBED[Embed and write in batches]
    EMBED --> CHROMA[("ChromaDB<br/>one collection per language")]
    CHROMA -.->|"built on first hybrid query,<br/>then appended on ingest"| BM25[("BM25 index<br/>one per language")]
```

**Query**

```mermaid
flowchart TD
    Q["POST /query, POST /query/stream,<br/>or scripts/query.py"] --> QLANG["Detect query language<br/>(selects one collection)"]
    QLANG --> MODE{use_hybrid?}
    MODE -->|no, default| VEC[Vector search]
    MODE -->|yes| HYB["Hybrid search: BM25 + vector,<br/>fused with alpha-weighted RRF"]
    VEC --> RR{use_reranker?}
    HYB --> RR
    RR -->|no, default| BUDGET[Apply context budget]
    RR -->|yes| RERANK["Cross-encoder rerank<br/>(wider candidate pool down to top_k)"]
    RERANK --> BUDGET
    BUDGET --> GEN[LLM generation]
    GEN --> HEAL{self_heal?}
    HEAL -->|no, default| LIST1["Build citation list<br/>(no validation)"]
    LIST1 --> RESP[Return answer and citations]
    HEAL -->|yes| VERIFY["Verify the answer,<br/>including citation validation"]
    VERIFY -->|pass| LIST2[Build citation list]
    LIST2 --> RESP
    VERIFY -->|attempts exhausted| ABSTAIN["Structured abstention<br/>(no citations)"]
    ABSTAIN --> RESP
    VERIFY -->|"fail: generation<br/>or citation"| REGEN[Repair prompt, same passages]
    REGEN --> GEN
    VERIFY -->|"fail: retrieval, query,<br/>or context"| RETRY["Rewrite query or widen retrieval<br/>(2nd+ retry: multi-query RRF)"]
    RETRY --> MODE
```

Hybrid search, reranking, and self-healing are three independent switches, all off by default. Instrumentation is shown separately in [Monitoring & Observability](#monitoring--observability), because it differs by entry point.


For detailed operations, configuration, deployment, evaluation, and security guidance, see [api.md](api.md), [novelties.md](novelties.md), [feature.md](feature.md), [files.md](files.md), and the source-level documentation.
