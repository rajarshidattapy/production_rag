# Testing

## Fast feedback

```bash
pytest -q
pytest -q tests/test_evaluation.py
python scripts/evaluate_retrieval.py --modes vector,hybrid,hybrid_reranked
python scripts/operational_drill.py
```

The test suite covers API behavior, ingestion, retrieval, generation, self-healing, access control, citations, observability, and operations. Retrieval evaluation and the operational drill are complementary checks: the first measures relevance; the second exercises failure and recovery behavior.

Run tests with representative provider credentials only when a test explicitly needs a live provider. Unit and integration tests should stub external LLM calls and use isolated runtime data.
