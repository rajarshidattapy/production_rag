"""Tests for the verifier and failure classifier (src/healing/verifier.py, failure_classifier.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.healing.failure_classifier import FailureClassifier
from src.healing.query_analysis import analyze_query, term_coverage
from src.healing.types import FailureType, VerificationResult
from src.healing.verifier import (
    Verifier,
    check_citations,
    extract_citation_markers,
    is_refusal,
)

CTX = [
    {
        "id": "c1",
        "document": "BM25 ranks documents using term frequency and inverse document frequency.",
        "score": 0.8,
    },
    {
        "id": "c2",
        "document": "Reciprocal Rank Fusion combines rankings; the constant k is typically 60.",
        "score": 0.7,
    },
]
QUESTION = "How does BM25 rank documents?"
GOOD_ANSWER = "BM25 ranks documents using term frequency and inverse document frequency [1]."


def _judge(**overrides) -> MagicMock:
    verdict = {
        "faithfulness": 0.95,
        "relevance": 0.9,
        "context_sufficient": True,
        "contradiction_detected": False,
        "unsupported_claims": [],
        "explanation": "ok",
    }
    verdict.update(overrides)
    client = MagicMock()
    client.complete.return_value = "Here is my verdict:\n" + json.dumps(verdict)
    return client


# ----------------------------------------------------------------------
# Deterministic helpers
# ----------------------------------------------------------------------


class TestDeterministicChecks:
    def test_extract_citation_markers_handles_groups_and_adjacent(self) -> None:
        assert extract_citation_markers("A [1]. B [2, 3]. C [4][5].") == [1, 2, 3, 4, 5]

    def test_citations_valid_when_in_range_and_supported(self) -> None:
        c = check_citations(GOOD_ANSWER, CTX)
        assert c.valid and c.out_of_range == [] and c.support_ratio == 1.0

    def test_citations_out_of_range_invalid(self) -> None:
        c = check_citations("BM25 ranks documents by term frequency [7].", CTX)
        assert not c.valid
        assert c.out_of_range == [7]

    def test_citations_missing_invalid(self) -> None:
        c = check_citations("BM25 ranks documents by term frequency.", CTX)
        assert not c.valid and not c.has_citations

    def test_misattributed_citation_lowers_support(self) -> None:
        # The sentence is about BM25 term frequency but cites the RRF passage.
        c = check_citations("BM25 ranks documents using term frequency statistics [2].", CTX)
        assert c.support_ratio < 0.5
        assert not c.valid

    @pytest.mark.parametrize(
        "text",
        [
            "I cannot find sufficient information in the provided documents to answer this question.",
            "Ich kann in den bereitgestellten Dokumenten keine ausreichenden Informationen finden.",
            "No puedo encontrar información suficiente en los documentos proporcionados.",
        ],
    )
    def test_refusal_detection_multilingual(self, text: str) -> None:
        assert is_refusal(text)

    def test_term_coverage_prefix_tolerant(self) -> None:
        cov, missing = term_coverage(
            ["rerank", "latency"], [{"document": "Reranking adds latency."}]
        )
        assert cov == 1.0 and missing == []

    def test_vague_query_detected(self) -> None:
        assert analyze_query("How does it work?").is_vague
        assert not analyze_query("How does BM25 keyword search work?").is_vague


# ----------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------


class TestVerifier:
    def test_passes_with_good_answer_and_judge(self) -> None:
        v = Verifier(llm_client=_judge())
        r = v.verify(QUESTION, GOOD_ANSWER, CTX)
        assert r.passed
        assert r.judge_available
        assert r.faithfulness == pytest.approx(0.95)
        assert r.failure_type is None

    def test_structured_output_shape(self) -> None:
        r = Verifier(llm_client=_judge()).verify(QUESTION, GOOD_ANSWER, CTX)
        dumped = r.model_dump()
        for key in (
            "passed",
            "faithfulness",
            "relevance",
            "citation_valid",
            "retrieval_sufficient",
            "failure_type",
            "reason",
        ):
            assert key in dumped

    def test_low_faithfulness_is_generation_failure(self) -> None:
        v = Verifier(
            llm_client=_judge(faithfulness=0.4, unsupported_claims=["BM25 was invented in 2020"])
        )
        r = v.verify(QUESTION, GOOD_ANSWER, CTX)
        assert not r.passed
        assert r.failure_type == FailureType.GENERATION_FAILURE
        assert "BM25 was invented in 2020" in r.reason

    def test_low_relevance_is_generation_failure(self) -> None:
        r = Verifier(llm_client=_judge(relevance=0.2)).verify(QUESTION, GOOD_ANSWER, CTX)
        assert r.failure_type == FailureType.GENERATION_FAILURE

    def test_judge_context_insufficient_is_retrieval_failure(self) -> None:
        r = Verifier(llm_client=_judge(context_sufficient=False)).verify(QUESTION, GOOD_ANSWER, CTX)
        assert not r.retrieval_sufficient
        assert r.failure_type == FailureType.RETRIEVAL_FAILURE

    def test_unresolved_contradiction_is_context_failure(self) -> None:
        r = Verifier(llm_client=_judge(contradiction_detected=True)).verify(
            QUESTION, GOOD_ANSWER, CTX
        )
        assert r.failure_type == FailureType.CONTEXT_FAILURE

    def test_citation_failure_skips_judge(self) -> None:
        judge = _judge()
        r = Verifier(llm_client=judge).verify(QUESTION, "BM25 ranks by term frequency [9].", CTX)
        assert r.failure_type == FailureType.CITATION_FAILURE
        judge.complete.assert_not_called()  # deterministic failure — no LLM spend

    def test_no_contexts_is_retrieval_failure(self) -> None:
        r = Verifier(llm_client=None).verify(QUESTION, "", [])
        assert r.failure_type == FailureType.RETRIEVAL_FAILURE

    def test_vocabulary_mismatch_is_query_failure(self) -> None:
        ctx = [{"id": "x", "document": "Cross-encoders jointly encode pairs.", "score": 0.5}]
        r = Verifier(llm_client=None).verify(
            "What is photosynthesis chlorophyll?", "whatever [1]", ctx
        )
        assert r.retrieval.query_term_coverage == 0.0
        assert r.failure_type == FailureType.QUERY_FAILURE

    def test_paraphrased_query_with_low_coverage_passes_when_judge_agrees(self) -> None:
        # Dense retrieval found the right passage despite little word overlap;
        # lexical coverage alone must not force a pointless rewrite.
        question = "Which exact-word ranking approach can't tell two words mean the same thing?"
        r = Verifier(llm_client=_judge(), min_query_coverage=0.9).verify(question, GOOD_ANSWER, CTX)
        assert r.retrieval.query_term_coverage < 0.9
        assert r.passed

    def test_low_coverage_gates_retrieval_without_judge(self) -> None:
        question = "Which exact-word ranking approach can't tell two words mean the same thing?"
        r = Verifier(llm_client=None, min_query_coverage=0.9).verify(question, GOOD_ANSWER, CTX)
        assert not r.passed and not r.retrieval_sufficient

    def test_refusal_is_retrieval_failure(self) -> None:
        ans = "I cannot find sufficient information in the provided documents to answer this question."
        r = Verifier(llm_client=_judge()).verify(QUESTION, ans, CTX)
        assert r.is_refusal and r.failure_type == FailureType.RETRIEVAL_FAILURE

    def test_low_vector_score_fails_retrieval(self) -> None:
        ctx = [{**CTX[0], "score": 0.05}]
        r = Verifier(llm_client=None, min_retrieval_score=0.2).verify(QUESTION, GOOD_ANSWER, ctx)
        assert not r.retrieval_sufficient

    def test_rrf_scores_not_held_to_absolute_floor(self) -> None:
        ctx = [{**CTX[0], "score": 0.016}]  # typical RRF magnitude
        r = Verifier(llm_client=None, min_retrieval_score=0.2).verify(
            QUESTION, GOOD_ANSWER, ctx, score_kind="rrf"
        )
        assert r.retrieval_sufficient and r.passed

    def test_judge_exception_degrades_to_deterministic(self) -> None:
        client = MagicMock()
        client.complete.side_effect = RuntimeError("provider down")
        r = Verifier(llm_client=client).verify(QUESTION, GOOD_ANSWER, CTX)
        assert r.passed and not r.judge_available and r.faithfulness is None

    def test_malformed_judge_output_ignored(self) -> None:
        client = MagicMock()
        client.complete.return_value = '{"faithfulness": 7, "relevance": "high"}'
        r = Verifier(llm_client=client).verify(QUESTION, GOOD_ANSWER, CTX)
        assert not r.judge_available

    def test_judge_disabled_never_calls_llm(self) -> None:
        judge = _judge()
        Verifier(llm_client=judge, judge_enabled=False).verify(QUESTION, GOOD_ANSWER, CTX)
        judge.complete.assert_not_called()


# ----------------------------------------------------------------------
# Classifier priority
# ----------------------------------------------------------------------


class TestFailureClassifier:
    def _result(self, **kw) -> VerificationResult:
        base = VerificationResult(passed=False, retrieval_sufficient=True, citation_valid=True)
        base.retrieval.num_contexts = 3
        base.retrieval.query_term_coverage = 0.8
        return base.model_copy(update=kw)

    def test_empty_answer_is_generation_failure(self) -> None:
        ft, _ = FailureClassifier().classify(self._result(empty_answer=True))
        assert ft == FailureType.GENERATION_FAILURE

    def test_vague_query_is_query_failure(self) -> None:
        ft, _ = FailureClassifier().classify(
            self._result(retrieval_sufficient=False), analyze_query("tell me about it")
        )
        assert ft == FailureType.QUERY_FAILURE

    def test_retrieval_beats_citation(self) -> None:
        ft, _ = FailureClassifier().classify(
            self._result(retrieval_sufficient=False, citation_valid=False)
        )
        assert ft == FailureType.RETRIEVAL_FAILURE

    def test_unknown_when_no_rule_matches(self) -> None:
        ft, _ = FailureClassifier().classify(self._result())
        assert ft == FailureType.UNKNOWN_FAILURE
