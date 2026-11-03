## Not implemented yet, but identified as next production upgrades:

- Page-aware PDF ingestion and page-specific retrieval
- Shortened citation previews
- Persistent server-side chat history/users instead of browser `localStorage`
- CLI-provider mode using Codex, Claude Code, or OpenCode locally
- Background/asynchronous ingestion UI for very large document collections
- **Next scaling steps:** API replicas, a response cache, a GPU or separate reranker, and Qdrant only if Chroma proves to be the bottleneck.

- **Formatting and type errors:** `ruff format` still flags [`generator.py`](http://generator.py), `[test_healing_](#orca-native-chat-file=test_healing_evaluation.py)[evaluation.py](http://evaluation.py)` and `[test_](#orca-native-chat-file=test_pipeline.py)[pipeline.py](http://pipeline.py)`, and `mypy` still reports 59 errors in untouched files, which I left alone.
- Hybrid retrieval flow doesn't clearly distinguish dense vector search, BM25 retrieval and RRF fusion as separate operations.
- Cross-encoder reranking is optional, but the diagram can make it appear mandatory.
- Self-healing architecture oversimplifies the actual state machine: query analysis, failure classification, adaptive retrieval, query rewriting and bounded retries deserve explicit representation.
- Citation generation, citation metadata construction and citation verification are different operations and aren't sufficiently distinguished.
- Observability is represented too narrowly instead of showing instrumentation across the entire pipeline.



## Optimizing for 20 million docs:

but can't sent to LLM: (cost, latency) - knowledge graph

1. structure pipeline -&gt; entities, relationship

make ontology - classes/ person looking for (classes)

relationship - regex

small llms (distil models)

- entity resolution
- tf-idf nlp



# Pipeline:

1. distributed, parallel computing (PC, BDA)

- Learn DBMS



## Deployment:

1. AWS sagemaker

(Docker → CI/CD → AWS → Kubernetes)



Learn about:

1. Add a reranker and measure if it helped
2. Reduce LLM cost by 70% without hurting quality
3. Cut first-token latency for a chat app
4. Design a semantic cache with hit-rate tracking
5. Route requests between a cheap and a smart model
6. Build an agent that uses tools reliably
7. Your agent gets stuck in a loop. Fix it
8. Design memory for a long-running assistant
9. Make LLM output always valid JSON
10. Detect and reduce hallucinations
11. Build an eval suite for a customer support bot
12. LLM-as-a-judge: how to make it trustworthy
13. Block prompt injection in a tool-using agent
14. Redact PII before it hits the model
15. Fine-tune vs RAG vs prompting: pick one and defend it
16. Fine-tune a small model with LoRA on 5K examples
17. Serve an open model at scale (batching, KV cache, quantization)
18. Stream responses over SSE and handle disconnects
19. Trace every step of a multi-agent workflow
20. Roll back a bad prompt change in production



## Changed:

- Made ingestion more production-friendly.
  - Added `RAG_INGEST_BATCH_SIZE`, default `2000`
  - Chroma inserts are batched and retried
  - The app respects a lower batch limit reported by Chroma
  - Verified with retrieval/pipeline tests.
- Confirmed BM25 persistence is configurable for large corpora.
  - `RAG_BM25_MAX_INDEX_MB` controls the persisted BM25 corpus safety limit.
  - Existing protection prevents large/corrupt indexes from exhausting memory on restart.
- Indexed the codebase locally with Graphify.
  - Generated a graph with 1,299 nodes, 2,858 edges, and 68 communities.
  - Created local architecture artifacts: `graphify-out/graph.html`, `GRAPH_REPORT.md`, and `graph.json`.
- Diagnosed key RAG quality limitations.
  - PDFs currently lose per-page metadata during ingestion.
  - “What is on the first page?” cannot reliably retrieve page 1 yet.
  - Citation previews currently render full chunks rather than short snippets.
  - Cross-encoder reranking is bounded by `RAG_TOP_K_RETRIEVAL` (default 20), then only top results go to the LLM.



- **Vector database (ChromaDB)**

  - **Paginated full scan:** `get_all_chunks()` used one unbounded `get()`, which failed with `too many SQL variables` above about 25–30k chunks. It now pages through `iter_chunks()`.
  - **Explicit HNSW settings:** `RAG_HNSW_EF_SEARCH` (200), `RAG_HNSW_MAX_NEIGHBORS` (16), and `RAG_HNSW_EF_CONSTRUCTION` (100). I measured 50k vectors: recall@10 was 0.74 at `ef_search=10`, 0.996 at 100, and 1.000 at 200, all with the same latency of about 1.4 ms. Old collections are raised to the new `ef_search` at startup.
  - **`delete_collection()` bug:** It recreated the collection without the cosine setting, so scores silently used L2 distance after a reset.
  - **Server-mode bug:** Every query through `RAG_CHROMA_HOST` (the Docker Compose setup) failed because the HTTP client needs NumPy arrays and the embedder returned lists. It now returns `float32` arrays.

- **Hybrid search (BM25)**

  - **Size limit:** The hardcoded 500 MB cap is now `RAG_BM25_MAX_INDEX_MB`. An index that would never be loaded is no longer written to disk.
  - **Startup:** It loads the persisted index first and rebuilds only if that fails.
  - **Own BM25 implementation:** I replaced `rank_bm25` scoring with an inverted index (NumPy only). It scores identically, which a test checks. Query p50 at 50k chunks went from 103 ms to 2–3 ms.
  - **Incremental ingest:** New chunks are appended to the live index, so an ingest costs 0.02 s instead of a 5–8 s rebuild. A lock protects the index from concurrent reads and writes.
  - **Journaled persistence:** Ingests append to a journal file and compact later. Index writes are atomic, and a torn write after a crash is tolerated.
  - **Index-only memory:** The retriever keeps only chunk IDs and term postings, and fetches hit text from Chroma by ID. Measured at 100k chunks, memory fell from about 291 MB to 120 MB. It is extrapolated to about 1.2 GB at 1M chunks, not run there. Cold start dropped from 7.3 s to 0.2 s.

- **Measurement and tests**

  - **`tests/test_vector_store_stress.py`:** One test with 11 phases. It covers bulk ingest, recall, filters, awkward input, duplicates, concurrent readers and writers, deletes, persistence, isolation, and reuse. It scales with `RAG_STRESS_CHUNKS` and can target a Chroma server with `RAG_STRESS_CHROMA_HOST`. It passes at 50k embedded and at 20k against a live server.
  - **`scripts/bench_vector_store.py`:** Reruns the HNSW and BM25 measurements.
  - **New tests:** Incremental add matches a full rebuild, the inverted index matches `rank_bm25`, the journal survives a restart, and the embedder returns NumPy arrays.

- **Dependencies, CI, and docs**

  - **`rank_bm25`:** It is now dev-only, and `/readyz` no longer imports it.
  - **`requirements-dev.txt`:** I created it because CI referenced a file that did not exist.
  - **Docs and `.env.example`:** Both are updated for the new settings and the BM25 format.
  - **Checks:** `ruff check` passes. `ruff format` passes for the files I touched. `mypy` has no new errors.

- **UI**

  - **Source preview** (committed): Collapsible **Sources (N)** cards with a number badge, filename without the hash prefix, score pill, 180-character preview, and click-to-expand.

- **Advice given but not implemented**

  - **Next scaling steps:** API replicas, a response cache, a GPU or separate reranker, and Qdrant only if Chroma proves to be the bottleneck.
- **Page-aware PDF ingestion:** It would fix queries like “what's on page 1”.
  - **Formatting and type errors:** `ruff format` still flags `generator.py`, `test_healing_evaluation.py`, and `test_pipeline.py`; `mypy` still reports 59 errors in untouched files, which I left alone.



&nbsp;
