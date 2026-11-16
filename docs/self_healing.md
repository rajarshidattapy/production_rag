# Self-healing

Self-healing is opt-in (`RAG_SELF_HEALING_ENABLED=true` or per-request `self_heal=true`). It verifies a generated answer against retrieved evidence and either returns a grounded answer, retries a targeted repair, or abstains.

## Flow

1. Retrieve evidence and generate an answer.
2. Check citations, retrieval quality, answer relevance, and faithfulness.
3. Classify failure as retrieval, generation, citation, query, or context related.
4. Regenerate with repair guidance or broaden/rewrite retrieval.
5. Stop after `RAG_MAX_HEALING_ATTEMPTS`; return a structured abstention when evidence remains insufficient.

Useful controls include `RAG_VERIFIER_ENABLED`, `RAG_RELEVANCE_THRESHOLD`, `RAG_FAITHFULNESS_THRESHOLD`, `RAG_MIN_RETRIEVAL_SCORE`, and the `RAG_HEALING_*` retry settings. It improves reliability for hard questions but increases latency and LLM usage; measure it with [evaluation](evaluation.md) before enabling it broadly.
