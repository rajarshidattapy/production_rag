"""End-to-end tests of the healing state machine with a scripted fake backend."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy
from src.healing.metrics import HealingMetrics
from src.healing.query_rewriter import QueryRewriter
from src.healing.state_machine import ABSTENTION_MESSAGE, HealingEngine
from src.healing.types import FailureType
from src.healing.verifier import Verifier

QUESTION = "How does BM25 rank documents?"

GOOD_CTX = [
    {
        "id": "bm25",
        "document": "BM25 ranks documents using term frequency and inverse document frequency.",
        "metadata": {"source": "hybrid_search.txt", "filename": "hybrid_search.txt"},
        "score": 0.8,
    }
]
BAD_CTX = [
    {
        "id": "rerank",
        "document": "Cross-encoders jointly encode a query and passage pair.",
        "metadata": {"source": "reranker.txt", "filename": "reranker.txt"},
        "score": 0.6,
    }
]
GOOD_ANSWER = "BM25 ranks documents using term frequency and inverse document frequency [1]."


class FakeBackend:
    """Scripted stand-in for the RAG pipeline components."""

    def __init__(
        self,
        retrieval: dict[str, list[dict[str, Any]]] | None = None,
        default_ctx: list[dict[str, Any]] | None = None,
        answers: list[str] | None = None,
        empty_corpus: bool = False,
    ) -> None:
        self.retrieval = retrieval or {}
        self.default_ctx = default_ctx if default_ctx is not None else GOOD_CTX
        self.answers = list(answers or [GOOD_ANSWER])
        self.empty_corpus = empty_corpus
        self.retrieve_calls: list[dict[str, Any]] = []
        self.rerank_calls: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []

    def retrieve(self, query, *, use_hybrid, k, fetch_k, alpha, lang):
        self.retrieve_calls.append(
            {"query": query, "use_hybrid": use_hybrid, "k": k, "fetch_k": fetch_k, "alpha": alpha}
        )
        return [dict(c) for c in self.retrieval.get(query, self.default_ctx)]

    def rerank(self, query, contexts, top_k):
        self.rerank_calls.append(query)
        return [{**c, "rerank_score": 0.9} for c in contexts[:top_k]]

    def apply_context_budget(self, contexts, budget):
        return contexts

    def generate(self, query, contexts, system_prompt):
        self.generate_calls.append(
            {"query": query, "contexts": contexts, "system_prompt": system_prompt}
        )
        return self.answers[min(len(self.generate_calls) - 1, len(self.answers) - 1)]

    def build_citations(self, contexts):
        return [c["id"] for c in contexts]

    def corpus_is_empty(self, lang):
        return self.empty_corpus


class RecordingObserver:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []

    def on_step_start(self, name, input_data, metadata):
        self.started.append(name)

    def on_step_end(self, name, output, elapsed, metadata=None):
        self.ended.append(name)

    def on_step_error(self, name, exc):
        pass


def _engine(
    backend: FakeBackend,
    *,
    verifier: Verifier | None = None,
    max_attempts: int = 3,
    rewriter: QueryRewriter | None = None,
    observer=None,
) -> HealingEngine:
    return HealingEngine(
        backend=backend,
        verifier=verifier or Verifier(llm_client=None),
        rewriter=rewriter or QueryRewriter(None),
        policy=AdaptiveRetrievalPolicy(),
        max_attempts=max_attempts,
        metrics=HealingMetrics(),
        observer=observer,
    )


def _scripted_judge(*verdicts: dict[str, Any]) -> MagicMock:
    base = {
        "faithfulness": 0.95,
        "relevance": 0.9,
        "context_sufficient": True,
        "contradiction_detected": False,
        "unsupported_claims": [],
    }
    client = MagicMock()
    client.complete.side_effect = [json.dumps({**base, **v}) for v in verdicts]
    return client


# ----------------------------------------------------------------------


def test_passes_first_attempt_without_repairs() -> None:
    backend = FakeBackend()
    engine = _engine(backend)
    result = engine.run(QUESTION)

    assert result.info.status == "passed"
    assert result.info.attempts == 1 and not result.info.recovered
    assert result.answer == GOOD_ANSWER
    assert result.citations == ["bm25"]
    assert len(backend.generate_calls) == 1
    assert backend.generate_calls[0]["system_prompt"] is None  # default prompt on attempt 1
    assert engine.metrics.value("rag_healing_queries_total", {"outcome": "passed"}) == 1


def test_retrieval_failure_rewrites_and_recovers_with_different_retrieval() -> None:
    rewriter = MagicMock(spec=QueryRewriter)
    rewriter.rewrite.return_value = ("BM25 term frequency inverse document frequency", "expand")
    backend = FakeBackend(
        retrieval={QUESTION: BAD_CTX, "BM25 term frequency inverse document frequency": GOOD_CTX},
        answers=["Cross-encoders encode pairs [1].", GOOD_ANSWER],
    )
    engine = _engine(backend, rewriter=rewriter)
    result = engine.run(QUESTION, use_hybrid=False, use_reranker=True)

    assert result.info.status == "recovered" and result.info.recovered
    assert result.info.attempts == 2
    assert result.info.failure_types[0] in (
        FailureType.RETRIEVAL_FAILURE,
        FailureType.QUERY_FAILURE,
    )
    assert result.info.query_rewrites == ["BM25 term frequency inverse document frequency"]

    first, second = backend.retrieve_calls
    assert first["query"] == QUESTION
    assert second["query"] != first["query"]  # behaviour actually changed
    assert second["fetch_k"] > first["fetch_k"]
    assert second["use_hybrid"] and second["alpha"] is not None
    # Reranking and generation stay anchored to the ORIGINAL question.
    assert backend.rerank_calls == [QUESTION, QUESTION]
    assert all(c["query"] == QUESTION for c in backend.generate_calls)
    assert engine.metrics.value("rag_healing_query_rewrites_total") == 1
    assert engine.metrics.value("rag_healing_retrieval_retries_total") == 1
    assert engine.metrics.value("rag_healing_success_total") == 1


def test_generation_failure_regenerates_with_repair_prompt_on_same_context() -> None:
    verifier = Verifier(
        llm_client=_scripted_judge(
            {"faithfulness": 0.3, "unsupported_claims": ["BM25 uses neural embeddings"]},
            {},
        )
    )
    backend = FakeBackend(answers=[GOOD_ANSWER, GOOD_ANSWER])
    engine = _engine(backend, verifier=verifier)
    result = engine.run(QUESTION)

    assert result.info.status == "recovered"
    assert result.info.failure_types == [FailureType.GENERATION_FAILURE]
    assert len(backend.retrieve_calls) == 1  # no re-retrieval for a generation failure
    repair = backend.generate_calls[1]["system_prompt"]
    assert repair is not None
    assert "BM25 uses neural embeddings" in repair
    assert "ONLY claims that are directly supported" in repair
    assert backend.generate_calls[1]["contexts"] == backend.generate_calls[0]["contexts"]
    assert engine.metrics.value("rag_healing_regenerations_total") == 1


def test_citation_failure_triggers_citation_repair() -> None:
    backend = FakeBackend(answers=["BM25 ranks documents by term frequency [4].", GOOD_ANSWER])
    engine = _engine(backend)
    result = engine.run(QUESTION)

    assert result.info.failure_types == [FailureType.CITATION_FAILURE]
    assert result.info.status == "recovered"
    repair = backend.generate_calls[1]["system_prompt"]
    assert "[4]" in repair  # feedback names the invalid citation
    assert "Valid source numbers are 1 to 1" in repair


def test_abstains_after_max_attempts() -> None:
    backend = FakeBackend(answers=["No citations here at all."])
    engine = _engine(backend, max_attempts=3)
    result = engine.run(QUESTION)

    assert result.abstained
    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []
    assert result.info.attempts == 3
    assert result.info.reason is not None
    assert len(backend.generate_calls) == 3
    assert engine.metrics.value("rag_healing_abstentions_total") == 1


@pytest.mark.parametrize("max_attempts", [1, 2, 4])
def test_max_attempts_is_respected(max_attempts: int) -> None:
    backend = FakeBackend(retrieval={}, default_ctx=BAD_CTX, answers=["unrelated [1]"])
    result = _engine(backend, max_attempts=max_attempts).run(QUESTION)
    assert result.abstained
    assert result.info.attempts == max_attempts
    assert len(result.history) == max_attempts


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _engine(FakeBackend(), max_attempts=0)


def test_empty_corpus_abstains_without_rewriting() -> None:
    rewriter = MagicMock(spec=QueryRewriter)
    backend = FakeBackend(default_ctx=[], empty_corpus=True)
    result = _engine(backend, rewriter=rewriter).run(QUESTION)

    assert result.abstained and result.info.reason == "empty_knowledge_base"
    assert backend.generate_calls == []  # never call the LLM with no context
    rewriter.rewrite.assert_not_called()


def test_no_context_skips_generation_but_retries_retrieval() -> None:
    backend = FakeBackend(retrieval={QUESTION: []}, default_ctx=GOOD_CTX)
    result = _engine(backend).run(QUESTION)
    assert result.info.status == "recovered"
    assert len(backend.generate_calls) == 1  # only the attempt that had context


def test_repeated_generation_failure_escalates_to_retrieval() -> None:
    verifier = Verifier(
        llm_client=_scripted_judge({"faithfulness": 0.2}, {"faithfulness": 0.2}, {})
    )
    backend = FakeBackend(answers=[GOOD_ANSWER])
    result = _engine(backend, verifier=verifier, max_attempts=3).run(QUESTION)

    assert result.info.status == "recovered"
    assert [h.action for h in result.history] == ["initial", "regenerate", "rewrite_and_retrieve"]
    assert len(backend.retrieve_calls) == 2


def test_observer_sees_attempt_scoped_spans() -> None:
    obs = RecordingObserver()
    backend = FakeBackend(
        retrieval={QUESTION: BAD_CTX},
        answers=["Cross-encoders encode pairs [1].", GOOD_ANSWER],
    )
    _engine(backend, observer=obs).run(QUESTION, use_reranker=True)

    assert obs.started == [
        "analyze",
        "retrieve_attempt_1",
        "rerank_attempt_1",
        "generate_attempt_1",
        "verify_attempt_1",
        "query_rewrite",
        "retrieve_attempt_2",
        "rerank_attempt_2",
        "generate_attempt_2",
        "verify_attempt_2",
    ]
    assert obs.ended == obs.started


def test_broken_observer_does_not_break_query() -> None:
    obs = MagicMock()
    obs.on_step_start.side_effect = RuntimeError("langfuse down")
    result = _engine(FakeBackend(), observer=obs).run(QUESTION)
    assert result.info.status == "passed"


def test_generator_exception_propagates() -> None:
    backend = FakeBackend()
    backend.generate = MagicMock(side_effect=RuntimeError("rate limited"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rate limited"):
        _engine(backend).run(QUESTION)


@pytest.mark.asyncio
async def test_run_async() -> None:
    result = await _engine(FakeBackend()).run_async(QUESTION)
    assert result.info.status == "passed"
