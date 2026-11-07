# Read:

[https://x.com/\_jaydeepkarale/status/2107744969146458595](https://x.com/_jaydeepkarale/status/2107744969146458595)



# Haven't scaled yet:

1. **RabbitMQ:** background jobs run in a one-worker queue inside the API process, so it's only worth adding once indexing moves to separate worker services.
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
4. Route requests between a cheap and a smart model
5. Build an agent that uses tools reliably
6. Your agent gets stuck in a loop. Fix it
7. Design memory for a long-running assistant
8. Make LLM output always valid JSON
9. Detect and reduce hallucinations
10. Build an eval suite for a customer support bot
11. LLM-as-a-judge: how to make it trustworthy
12. Block prompt injection in a tool-using agent
13. Redact PII before it hits the model
14. Fine-tune vs RAG vs prompting: pick one and defend it
15. Fine-tune a small model with LoRA on 5K examples
16. Serve an open model at scale (batching, KV cache, quantization)
17. Stream responses over SSE and handle disconnects
18. Trace every step of a multi-agent workflow
19. Roll back a bad prompt change in production

