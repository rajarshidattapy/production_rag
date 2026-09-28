## Goal

Convert the existing production RAG pipeline into a genuinely **self-healing RAG system**.

Do NOT rewrite the project from scratch. Preserve the existing architecture, APIs, tests, observability, evaluation, Docker setup, hybrid retrieval, reranking, and dual LLM support wherever possible.

The final system must have a real feedback loop:

```text
Query
  ↓
Query Analysis
  ↓
Hybrid Retrieval
(BM25 + Vector + RRF)
  ↓
Cross-Encoder Reranking
  ↓
Generation
  ↓
Verifier
  ↓
┌───────────────┴───────────────┐
│                               │
PASS                            FAIL
│                               │
Final Answer             Failure Diagnosis
                                ↓
                    ┌───────────┼───────────┐
                    ↓           ↓           ↓
               Retrieval     Query       Generation
                Failure      Failure       Failure
                    ↓           ↓           ↓
                 Retrieve    Rewrite     Regenerate
                    └───────────┬───────────┘
                                ↓
                              Verify
                                ↓
                         PASS / RETRY
                                ↓
                         Max retries?
                                ↓
                             Abstain
```

## 1. Add a verifier

Create a dedicated verification layer after generation.

The verifier should evaluate:

- Faithfulness / groundedness
- Answer relevance
- Whether claims are supported by retrieved context
- Citation validity
- Retrieval sufficiency
- Contradictions between retrieved documents
- Whether the answer should be regenerated

Return structured output such as:

```json
{
  "passed": false,
  "faithfulness": 0.61,
  "relevance": 0.88,
  "citation_valid": false,
  "retrieval_sufficient": false,
  "failure_type": "retrieval",
  "reason": "The answer contains claims not supported by the retrieved context."
}
```

Do not rely on free-form LLM text to control the workflow. Use structured/Pydantic output.

## 2. Add failure diagnosis

Create:

`src/healing/failure_classifier.py`

Classify failures into:

```text
RETRIEVAL_FAILURE
QUERY_FAILURE
GENERATION_FAILURE
CITATION_FAILURE
CONTEXT_FAILURE
UNKNOWN_FAILURE
```

The classifier should use verifier signals and deterministic checks where possible.

Avoid using an LLM for things that can be checked deterministically.

## 3. Add query rewriting

Create:

`src/healing/query_rewriter.py`

When retrieval is insufficient:

```text
original query
      ↓
query analysis
      ↓
rewritten query
      ↓
hybrid retrieval
      ↓
reranking
```

Support multiple retrieval attempts.

Keep the original query available for final answer generation.

## 4. Add adaptive retrieval

The healing system should be able to change retrieval behavior after failure.

For example:

### Attempt 1
Normal:

```text
top_k = 20
hybrid retrieval
reranking
```

### Attempt 2

If context is insufficient:

```text
rewrite query
increase retrieval candidates
adjust dense/BM25 balance
rerank again
```

### Attempt 3

If still insufficient:

```text
broader retrieval
different query formulation
larger context budget
```

Do not endlessly retry.

Make:

```text
MAX_HEALING_ATTEMPTS
```

configurable through environment variables.

Default to 2–3 retries.

## 5. Add generation repair

If retrieval is good but the answer is not:

```text
Verifier
   ↓
generation failure
   ↓
regenerate using verified context
```

The regeneration prompt must explicitly instruct the model to only make claims supported by retrieved context.

If citation validation fails, regenerate rather than blindly returning the answer.

## 6. Add abstention

Self-healing must NOT mean endlessly trying to manufacture an answer.

After maximum healing attempts:

```text
Unable to answer reliably from the available context.
```

Return a structured response indicating:

```json
{
  "status": "abstained",
  "reason": "insufficient_verified_context",
  "attempts": 3
}
```

This is a critical part of the system.

## 7. Use LangGraph only where it adds value

If LangGraph is introduced, use it for the healing state machine rather than wrapping every existing function unnecessarily.

Create something conceptually like:

```text
START
 ↓
analyze
 ↓
retrieve
 ↓
rerank
 ↓
generate
 ↓
verify
 ↓
route
 ├── PASS → END
 ├── RETRIEVAL_FAILURE → rewrite → retrieve
 ├── QUERY_FAILURE → rewrite → retrieve
 ├── GENERATION_FAILURE → regenerate
 ├── CITATION_FAILURE → regenerate
 └── MAX_RETRIES → abstain
```

Keep retrieval, reranking, generation, and verification as independently testable components.

## 8. Preserve existing production features

Do NOT remove:

- BM25
- Vector search
- RRF
- Cross-encoder reranking
- SSE streaming
- Async ingestion
- FastAPI
- OpenAI backend
- Anthropic backend
- OpenTelemetry
- Prometheus metrics
- Langfuse
- Existing evaluation framework
- Docker
- GitHub Actions
- Existing tests

Integrate healing into the current pipeline.

## 9. Extend observability

Every healing attempt should be observable.

Track:

```text
rag.healing.attempts
rag.healing.success
rag.healing.failures
rag.healing.abstentions
rag.healing.failure_type
rag.healing.query_rewrites
rag.healing.retrieval_retries
rag.healing.regenerations
rag.healing.latency
```

Langfuse traces should clearly show:

```text
query
 ├── retrieve_attempt_1
 ├── rerank_attempt_1
 ├── generate_attempt_1
 ├── verify_attempt_1
 ├── query_rewrite
 ├── retrieve_attempt_2
 ├── rerank_attempt_2
 ├── generate_attempt_2
 └── verify_attempt_2
```

This should make the healing behavior debuggable.

## 10. Extend evaluation

Do not only measure final answer quality.

Add evaluation for:

- Initial retrieval failure rate
- Healing success rate
- Average healing attempts
- Final faithfulness
- Final answer relevance
- Citation validity
- Abstention rate
- False-positive healing
- Latency overhead caused by healing

Create a golden dataset containing deliberately difficult queries where the first retrieval attempt is expected to fail.

The system should demonstrate that healing improves final answer quality.

## 11. Add tests

Maintain all existing tests.

Add comprehensive tests for:

- Verifier
- Failure classifier
- Query rewriting
- Retrieval retry
- Generation retry
- Citation failure
- Successful healing
- Failed healing
- Maximum retry handling
- Abstention
- LangGraph routing
- Healing metrics
- SSE responses after healing

Use mocks for LLM calls.

Do not make the test suite dependent on external API availability.

Target:

```text
existing tests + substantial healing test coverage
```

Do not artificially inflate test count just to claim a number.

## 12. API compatibility

Keep the existing `/query` and `/query/stream` APIs compatible.

Add optional metadata such as:

```json
{
  "answer": "...",
  "citations": [],
  "healing": {
    "attempts": 2,
    "recovered": true
  }
}
```

Do not break existing clients.

## 13. Configuration

Add settings such as:

```text
RAG_SELF_HEALING_ENABLED=true
RAG_MAX_HEALING_ATTEMPTS=3
RAG_VERIFIER_ENABLED=true
RAG_FAITHFULNESS_THRESHOLD=0.7
RAG_RELEVANCE_THRESHOLD=0.7
RAG_MIN_RETRIEVAL_SCORE=...
```

Use the existing Pydantic settings architecture.

## 14. Important engineering constraint

Do NOT implement fake self-healing.

This is NOT sufficient:

```text
generate → evaluate → log failure
```

Self-healing means:

```text
generate
   ↓
detect failure
   ↓
diagnose failure
   ↓
change system behavior
   ↓
retry
   ↓
verify again
```

The second attempt must actually differ from the first attempt based on the diagnosed failure.

## 15. Final deliverables

After implementation:

1. Update the architecture documentation.
2. Add a self-healing architecture diagram to the README.
3. Document the healing state machine.
4. Document configuration variables.
5. Document failure types and repair strategies.
6. Add tests.
7. Run the complete test suite.
8. Run the evaluation suite.
9. Verify Docker build.
10. Verify `/query` and `/query/stream`.
11. Report exactly what files were changed and why.
12. Report any limitations honestly.

Before making changes, inspect the existing repository thoroughly and understand the current pipeline. Do not duplicate functionality that already exists.

The final implementation should be something you can credibly describe as:

**"A production-grade self-healing RAG pipeline that detects retrieval, grounding, and citation failures, automatically repairs the query/retrieval/generation path, re-verifies the result, and abstains when reliable recovery is impossible."**