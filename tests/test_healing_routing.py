"""Tests for routing, adaptive retrieval policy, query rewriting, and healing metrics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy, fuse_ranked_lists
from src.healing.metrics import HealingMetrics
from src.healing.prompts import GENERATION_REPAIR_PROMPT, fill
from src.healing.query_analysis import analyze_query
from src.healing.query_rewriter import QueryRewriter
from src.healing.state_machine import abstention_reason, route
from src.healing.types import FailureType, RepairAction, VerificationResult


def _failed(ft: FailureType) -> VerificationResult:
    return VerificationResult(passed=False, failure_type=ft)


# ----------------------------------------------------------------------
# route()
# ----------------------------------------------------------------------


class TestRoute:
    def test_pass_accepts(self) -> None:
        assert route(VerificationResult(passed=True), 1, 3) == RepairAction.ACCEPT

    @pytest.mark.parametrize("ft", [FailureType.RETRIEVAL_FAILURE, FailureType.QUERY_FAILURE])
    def test_retrieval_side_failures_rewrite(self, ft: FailureType) -> None:
        assert route(_failed(ft), 1, 3) == RepairAction.REWRITE_AND_RETRIEVE

    @pytest.mark.parametrize(
        "ft",
        [FailureType.GENERATION_FAILURE, FailureType.CITATION_FAILURE, FailureType.UNKNOWN_FAILURE],
    )
    def test_generation_side_failures_regenerate(self, ft: FailureType) -> None:
        assert route(_failed(ft), 1, 3) == RepairAction.REGENERATE

    def test_context_failure_expands_context(self) -> None:
        assert route(_failed(FailureType.CONTEXT_FAILURE), 1, 3) == RepairAction.EXPAND_CONTEXT

    def test_max_attempts_abstains_even_if_repairable(self) -> None:
        assert route(_failed(FailureType.GENERATION_FAILURE), 3, 3) == RepairAction.ABSTAIN

    def test_pass_on_last_attempt_still_accepts(self) -> None:
        assert route(VerificationResult(passed=True), 3, 3) == RepairAction.ACCEPT

    def test_repeated_generation_failure_escalates_to_retrieval(self) -> None:
        action = route(
            _failed(FailureType.GENERATION_FAILURE),
            2,
            4,
            last_action=RepairAction.REGENERATE,
            last_failure=FailureType.GENERATION_FAILURE,
        )
        assert action == RepairAction.REWRITE_AND_RETRIEVE

    def test_abstention_reasons(self) -> None:
        assert abstention_reason(FailureType.RETRIEVAL_FAILURE) == "insufficient_verified_context"
        assert abstention_reason(FailureType.CITATION_FAILURE) == "unverifiable_citations"


# ----------------------------------------------------------------------
# AdaptiveRetrievalPolicy — each retry must differ from the last
# ----------------------------------------------------------------------


class TestAdaptiveRetrievalPolicy:
    def _initial(self, policy: AdaptiveRetrievalPolicy, **kw):
        args = {
            "final_k": 5,
            "top_k_retrieval": 20,
            "use_hybrid": False,
            "use_reranker": False,
            "context_budget": 12000,
        }
        args.update(kw)
        return policy.initial_plan("q", **args)

    def test_initial_plan_matches_non_healing_behaviour(self) -> None:
        p = AdaptiveRetrievalPolicy()
        assert self._initial(p).fetch_k == 5
        assert self._initial(p, use_reranker=True).fetch_k == 20

    def test_first_retry_rewrites_widens_and_rebalances(self) -> None:
        p = AdaptiveRetrievalPolicy(base_alpha=0.6)
        init = self._initial(p, use_reranker=True)
        retry = p.retrieval_retry_plan(init, FailureType.RETRIEVAL_FAILURE, "q", ["q2"], 1)
        assert retry.queries == ["q2"]
        assert retry.fetch_k > init.fetch_k and retry.final_k > init.final_k
        assert retry.use_hybrid  # escalated
        assert retry.alpha == pytest.approx(0.35)  # keyword-leaning

    def test_query_failure_biases_dense(self) -> None:
        p = AdaptiveRetrievalPolicy(base_alpha=0.6)
        retry = p.retrieval_retry_plan(self._initial(p), FailureType.QUERY_FAILURE, "q", ["q2"], 1)
        assert retry.alpha == pytest.approx(0.85)

    def test_second_retry_goes_multi_query_with_bigger_budget(self) -> None:
        p = AdaptiveRetrievalPolicy()
        r1 = p.retrieval_retry_plan(self._initial(p), FailureType.RETRIEVAL_FAILURE, "q", ["q2"], 1)
        r2 = p.retrieval_retry_plan(r1, FailureType.RETRIEVAL_FAILURE, "q", ["q2", "q3"], 2)
        assert r2.queries == ["q", "q2", "q3"]
        assert r2.context_budget > r1.context_budget
        assert r2.fetch_k > r1.fetch_k

    def test_limits_are_respected(self) -> None:
        p = AdaptiveRetrievalPolicy(max_fetch_k=30, max_final_k=6)
        plan = self._initial(p, use_reranker=True)
        for i in range(1, 5):
            plan = p.retrieval_retry_plan(plan, FailureType.RETRIEVAL_FAILURE, "q", ["a", "b"], i)
        assert plan.fetch_k <= 30 and plan.final_k <= 6

    def test_context_expansion(self) -> None:
        p = AdaptiveRetrievalPolicy()
        init = self._initial(p)
        exp = p.context_expansion_plan(init)
        assert exp.queries == init.queries
        assert exp.final_k > init.final_k and exp.context_budget > init.context_budget

    def test_fuse_ranked_lists_rewards_agreement(self) -> None:
        a = [{"id": "x", "document": "x"}, {"id": "y", "document": "y"}]
        b = [{"id": "y", "document": "y"}, {"id": "z", "document": "z"}]
        fused = fuse_ranked_lists([a, b], k=3)
        assert fused[0]["id"] == "y"
        assert {d["id"] for d in fused} == {"x", "y", "z"}


# ----------------------------------------------------------------------
# QueryRewriter
# ----------------------------------------------------------------------


class TestQueryRewriter:
    def _llm(self, query: str) -> MagicMock:
        m = MagicMock()
        m.complete.return_value = json.dumps({"rewritten_query": query, "rationale": "r"})
        return m

    def test_uses_llm_rewrite(self) -> None:
        rw = QueryRewriter(self._llm("BM25 term frequency ranking"))
        q, strategy = rw.rewrite(
            analyze_query("how does bm25 rank"), ["how does bm25 rank"], [], "d"
        )
        assert q == "BM25 term frequency ranking" and strategy == "expand"

    def test_repeated_llm_output_falls_back_to_deterministic(self) -> None:
        rw = QueryRewriter(self._llm("How does BM25 rank?"))
        q, strategy = rw.rewrite(
            analyze_query("How does BM25 rank?"), ["How does BM25 rank?"], [], "d"
        )
        assert q.lower() != "how does bm25 rank?"
        assert strategy.endswith(":deterministic")

    def test_llm_error_falls_back(self) -> None:
        m = MagicMock()
        m.complete.side_effect = RuntimeError("boom")
        q, _ = QueryRewriter(m).rewrite(analyze_query("What is RRF?"), ["What is RRF?"], [], "d")
        assert "reciprocal" in q  # deterministic acronym expansion

    def test_strategies_rotate(self) -> None:
        assert [QueryRewriter.strategy_for(i) for i in range(4)] == [
            "expand",
            "keywords",
            "decompose",
            "expand",
        ]

    def test_deterministic_rewrites_never_repeat(self) -> None:
        rw = QueryRewriter(None)
        analysis = analyze_query("What is RAG?")
        seen = ["What is RAG?"]
        for i in range(6):
            q, _ = rw.rewrite(analysis, list(seen), [], "d", rewrite_index=i)
            assert q.lower() not in {s.lower() for s in seen}
            seen.append(q)


# ----------------------------------------------------------------------
# Prompts & metrics
# ----------------------------------------------------------------------


def test_repair_prompt_keeps_context_placeholder_and_escapes_braces() -> None:
    prompt = fill(GENERATION_REPAIR_PROMPT, feedback="claim {x} unsupported", num_sources=3)
    rendered = prompt.format(context="CTX")  # what Generator.generate does
    assert "CTX" in rendered
    assert "claim {x} unsupported" in rendered
    assert "ONLY claims that are directly supported" in rendered
    assert "1 to 3" in rendered


def test_healing_metrics_record_to_prometheus() -> None:
    m = HealingMetrics()
    m.record_failure("RETRIEVAL_FAILURE")
    m.record_query_rewrite()
    m.record_retrieval_retry()
    m.record_regeneration()
    m.record_outcome("recovered", attempts=2, latency=1.5)
    m.record_outcome("abstained", attempts=3, latency=3.0)

    assert m.value("rag_healing_failures_total", {"failure_type": "RETRIEVAL_FAILURE"}) == 1
    assert m.value("rag_healing_query_rewrites_total") == 1
    assert m.value("rag_healing_retrieval_retries_total") == 1
    assert m.value("rag_healing_regenerations_total") == 1
    assert m.value("rag_healing_success_total") == 1
    assert m.value("rag_healing_abstentions_total") == 1
    assert m.value("rag_healing_queries_total", {"outcome": "recovered"}) == 1
    assert m.value("rag_healing_attempts_count") == 2
    assert m.value("rag_healing_attempts_sum") == 5
    assert m.value("rag_healing_latency_seconds_count") == 2
