"""Adaptive retrieval policy: how retrieval changes after each diagnosed failure.

Attempt 1 is exactly what the caller asked for (same hybrid/reranker flags and
top-k the non-healing path would use). Later attempts change behaviour based on
the diagnosis:

- QUERY_FAILURE (vocabulary mismatch / vague query):
    rewritten query, more candidates, dense-leaning fusion (alpha ↑) because
    embeddings tolerate wording mismatch better than BM25.
- RETRIEVAL_FAILURE (context doesn't cover the question):
    rewritten/expanded query, more candidates, keyword-leaning fusion (alpha ↓)
    so the explicit terms added by the rewrite actually drive ranking.
- Second and later retrieval retries broaden further: multi-query retrieval
  (original + every rewrite, fused with RRF), more candidates again, and a
  larger context budget.
- CONTEXT_FAILURE: same query, more passages and a larger context budget so
  conflicting evidence is fully visible to the generator.
- GENERATION/CITATION/UNKNOWN: retrieval is kept as-is; only generation changes.
"""

from __future__ import annotations

from typing import Any

from src.healing.types import FailureType, RetrievalPlan


class AdaptiveRetrievalPolicy:
    def __init__(
        self,
        base_alpha: float = 0.6,
        fetch_k_multiplier: float = 2.0,
        max_fetch_k: int = 100,
        max_final_k: int = 12,
        final_k_step: int = 2,
        context_budget_multiplier: float = 1.5,
        max_context_budget: int = 36000,
        escalate_hybrid: bool = True,
        rerank_on_retry: bool = False,
    ) -> None:
        self.base_alpha = base_alpha
        self.fetch_k_multiplier = fetch_k_multiplier
        self.max_fetch_k = max_fetch_k
        self.max_final_k = max_final_k
        self.final_k_step = final_k_step
        self.context_budget_multiplier = context_budget_multiplier
        self.max_context_budget = max_context_budget
        self.escalate_hybrid = escalate_hybrid
        self.rerank_on_retry = rerank_on_retry

    def initial_plan(
        self,
        question: str,
        final_k: int,
        top_k_retrieval: int,
        use_hybrid: bool,
        use_reranker: bool,
        context_budget: int,
    ) -> RetrievalPlan:
        return RetrievalPlan(
            queries=[question],
            fetch_k=top_k_retrieval if use_reranker else final_k,
            final_k=final_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            alpha=None,
            context_budget=context_budget,
            strategy="initial",
        )

    def _widen(self, prev: RetrievalPlan) -> tuple[int, int]:
        final_k = min(self.max_final_k, prev.final_k + self.final_k_step)
        fetch_k = min(
            self.max_fetch_k,
            max(int(prev.fetch_k * self.fetch_k_multiplier), final_k),
        )
        return fetch_k, final_k

    def _grow_budget(self, budget: int) -> int:
        if budget <= 0:  # budget disabled — keep disabled
            return budget
        return min(self.max_context_budget, int(budget * self.context_budget_multiplier))

    def retrieval_retry_plan(
        self,
        prev: RetrievalPlan,
        failure: FailureType,
        original_question: str,
        rewrites: list[str],
        retry_index: int,
    ) -> RetrievalPlan:
        """Plan for the ``retry_index``-th retrieval retry (1-based). ``rewrites`` must be non-empty."""
        fetch_k, final_k = self._widen(prev)
        use_hybrid = prev.use_hybrid or self.escalate_hybrid
        use_reranker = prev.use_reranker or self.rerank_on_retry

        if retry_index <= 1:
            if failure == FailureType.QUERY_FAILURE:
                alpha = min(0.9, self.base_alpha + 0.25)
                strategy = "rewrite+dense_bias"
            else:
                alpha = max(0.2, self.base_alpha - 0.25)
                strategy = "rewrite+lexical_bias"
            return RetrievalPlan(
                queries=[rewrites[-1]],
                fetch_k=fetch_k,
                final_k=final_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                alpha=alpha if use_hybrid else None,
                context_budget=prev.context_budget,
                strategy=strategy,
            )

        # Broader: fuse original + all rewrites, bigger budget, balanced alpha.
        queries = [original_question, *dict.fromkeys(rewrites)]
        return RetrievalPlan(
            queries=queries,
            fetch_k=fetch_k,
            final_k=final_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            alpha=self.base_alpha if use_hybrid else None,
            context_budget=self._grow_budget(prev.context_budget),
            strategy="multi_query+broad",
        )

    def context_expansion_plan(self, prev: RetrievalPlan) -> RetrievalPlan:
        fetch_k, final_k = self._widen(prev)
        return prev.model_copy(
            update={
                "fetch_k": fetch_k,
                "final_k": final_k,
                "context_budget": self._grow_budget(prev.context_budget),
                "strategy": "expand_context",
            }
        )


def fuse_ranked_lists(
    result_lists: list[list[dict[str, Any]]], k: int, rrf_k: int = 60
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across the result lists of several queries."""
    if len(result_lists) == 1:
        return result_lists[0][:k]
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            doc_id = r.get("id") or r.get("document", "")[:64]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs.setdefault(doc_id, r)
    ordered = sorted(scores, key=lambda d: scores[d], reverse=True)[:k]
    return [{**docs[d], "score": round(scores[d], 6)} for d in ordered]
