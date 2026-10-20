[https://x.com/\_jaydeepkarale/status/2107744969146458595](https://x.com/_jaydeepkarale/status/2107744969146458595)



&nbsp;

Refer to:

1. fastgraph rag by ms: [https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)

Notes:



20 million docs, but can't sent to LLM: (cost, latency) - knowledge graph

1. structure pipeline -&gt; entities, relationship

make ontology - classes/ person looking for (classes)

relationship - regex

small llms (distil models)

- entity resolution
- tf-idf nlp

Pipeline:

1. distributed, parallel computing (PC, BDA)

- Learn DBMS



&nbsp;

Deployment:

1. AWS sagemaker

(Docker → CI/CD → AWS → Kubernetes)



&nbsp;

Your RAG gives wrong answers. Debug it  

Learn about:

1. Chunking strategies
2. BM25 vs dense vs hybrid search: when to use each
3. Add a reranker and measure if it helped
4. Reduce LLM cost by 70% without hurting quality
5. Cut first-token latency for a chat app
6. Design a semantic cache with hit-rate tracking
7. Route requests between a cheap and a smart model
8. Build an agent that uses tools reliably
9. Your agent gets stuck in a loop. Fix it
10. Design memory for a long-running assistant
11. Make LLM output always valid JSON
12. Detect and reduce hallucinations
13. Build an eval suite for a customer support bot
14. LLM-as-a-judge: how to make it trustworthy
15. Block prompt injection in a tool-using agent
16. Redact PII before it hits the model
17. Fine-tune vs RAG vs prompting: pick one and defend it
18. Fine-tune a small model with LoRA on 5K examples
19. Serve an open model at scale (batching, KV cache, quantization)
20. Stream responses over SSE and handle disconnects
21. Trace every step of a multi-agent workflow
22. Roll back a bad prompt change in production

