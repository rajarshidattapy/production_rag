# Read:

[https://x.com/\_jaydeepkarale/status/2107744969146458595](https://x.com/_jaydeepkarale/status/2107744969146458595)

## Not implemented yet, but identified as next production upgrades:

- **Next scaling steps:** API replicas, a response cache, a GPU or separate reranker, and Qdrant only if Chroma proves to be the bottleneck.
- **Formatting and type errors:** `ruff format` still flags [`generator.py`](http://generator.py), `[test_healing_](#orca-native-chat-file=test_healing_evaluation.py)[evaluation.py](http://evaluation.py)` and `[test_](#orca-native-chat-file=test_pipeline.py)[pipeline.py](http://pipeline.py)`, and `mypy` still reports 59 errors in untouched files, which I left alone.
- Hybrid retrieval flow doesn't clearly distinguish dense vector search, BM25 retrieval and RRF fusion as separate operations.
- Cross-encoder reranking is optional, but the diagram can make it appear mandatory.
- Self-healing architecture oversimplifies the actual state machine: query analysis, failure classification, adaptive retrieval, query rewriting and bounded retries deserve explicit representation.
- Citation generation, citation metadata construction and citation verification are different operations and aren't sufficiently distinguished.
- Observability is represented too narrowly instead of showing instrumentation across the entire pipeline.
- **Status:** production-oriented, not production-validated. It has not been run in a production deployment, and its Docker deployment and CI workflows have not been exercised. [++Production Readiness++](https://github.com/rajarshidattapy/production_rag#production-readiness) lists what has been verified, what has not, and the known gaps.
- **Shared documents:** besides the shared retrieval above, [`/ingest`](#orca-native-chat-file=%2Fingest) with `reset: true` still clears the store for everyone.
- **No follow-up context:** answers don't use earlier messages, so "what about page 3?" won't know what "that" refers to. The saved history makes this possible to add.
- **No self-service password reset:** an operator has to do it with the script.
- **Login lockout is per account, not per IP:** someone can keep a known account locked, or try one password across many accounts.
- **Dashboard undercount (already there):** unhandled 500s don't appear in the dashboard's error rate, because the metrics code never sees them. The stubbed LLM failure didn't show up there.
- **Previews show the start of a passage:** they don't show the part that matched the question, so a passage opening with boilerplate gets a boilerplate preview. Centring the preview on the question's terms is a small follow-up if you want it.
- **Removed documents:** new chats save only the preview, so once a document is removed, its full text can't be opened from old chats.
- **Odd score on page questions:** the expanded card for "What is on page 1?" showed a score of `-0.06`. The page filter chooses those passages, so similarity to the question means little there. It looks strange, and hiding the score on those results would be a small fix.
- **Cancel can take about a minute on a CPU:** it waits for the current batch. Graceful shutdown waits too (14.7 s when I tested it), so a container stop timeout may end it first. That counts as a crash, and Run again recovers it.
- **Uploads need the tab open:** files go up one at a time until the job starts, so for very large collections use **Server folder**. Its path must be inside `RAG_DATA_DIR`.
- **Run again re-reads every file:** it only skips the embedding.
- **Each replica has its own queue:** jobs on different replicas run at the same time, and cancel only works on the replica running the job.
- **Scale tested so far:** text files only, up to 40,000 chunks, and that largest run used stub embeddings. Scanned PDFs get no OCR.

  




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

