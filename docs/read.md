## Notes:

- in future chroma might be bottleneck, since it has single instance, need a dedicated vector db like pinecone or qdrant, or ofc: chromadb pro



## Not implemented yet, but identified as next production upgrades:

- Page-aware PDF ingestion and page-specific retrieval
- Shortened citation previews
- Persistent server-side chat history/users instead of browser `localStorage`
- CLI-provider mode using Codex, Claude Code, or OpenCode locally
- Background/asynchronous ingestion UI for very large document collections



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



&nbsp;

### Your RAG gives wrong answers. Debug it  

Learn about:



1. BM25 vs dense vs hybrid search: when to use each
2. Add a reranker and measure if it helped
3. Reduce LLM cost by 70% without hurting quality
4. Cut first-token latency for a chat app
5. Design a semantic cache with hit-rate tracking
6. Route requests between a cheap and a smart model
7. Build an agent that uses tools reliably
8. Your agent gets stuck in a loop. Fix it
9. Design memory for a long-running assistant
10. Make LLM output always valid JSON
11. Detect and reduce hallucinations
12. Build an eval suite for a customer support bot
13. LLM-as-a-judge: how to make it trustworthy
14. Block prompt injection in a tool-using agent
15. Redact PII before it hits the model
16. Fine-tune vs RAG vs prompting: pick one and defend it
17. Fine-tune a small model with LoRA on 5K examples
18. Serve an open model at scale (batching, KV cache, quantization)
19. Stream responses over SSE and handle disconnects
20. Trace every step of a multi-agent workflow
21. Roll back a bad prompt change in production



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
  - Created local architecture artifacts: `graphify-out/graph.html`, `GRAPH_[REPORT.md](http://REPORT.md)`, and `graph.json`.
- Diagnosed key RAG quality limitations.
  - PDFs currently lose per-page metadata during ingestion.
  - “What is on the first page?” cannot reliably retrieve page 1 yet.
  - Citation previews currently render full chunks rather than short snippets.
  - Cross-encoder reranking is bounded by `RAG_TOP_K_RETRIEVAL` (default 20), then only top results go to the LLM.

