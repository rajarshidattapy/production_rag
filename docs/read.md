# Read:

[https://x.com/\_jaydeepkarale/status/2107744969146458595](https://x.com/_jaydeepkarale/status/2107744969146458595)



# Haven't scaled yet:

1. **Grafana:** the built-in Observability dashboard covers a single server, so it only becomes necessary with several replicas.
2. **Redis:** nothing needs a shared cache or shared state yet, because the app runs as one process with an in-memory query cache and no response cache.
3. **Kafka:** there's no high-volume event stream or multiple consumers, only document jobs that a single in-process queue handles.
4. **RabbitMQ:** background jobs run in a one-worker queue inside the API process, so it's only worth adding once indexing moves to separate worker services.
5. Celery: Indexing is the only long-running job, and a one-worker thread queue handles it for now, so Celery's broker and separate workers aren't worth adding until we run replicas.



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

