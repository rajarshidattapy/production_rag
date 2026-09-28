"""Query rewriting for retrieval repair.

The rewriter produces a *new* search query for the next retrieval attempt. The
original question is never replaced — it is still what the generator answers
and what the reranker scores against. Rewritten queries only drive recall.

Guarantees:
- The returned query differs (case/whitespace-insensitively) from every query
  already tried. If the LLM repeats itself, returns junk, or fails, a
  deterministic rewrite is used instead.
- Each attempt uses a different strategy (expand → keywords → decompose), so
  successive attempts genuinely explore different formulations.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from src.generation.llm_client import LLMClient
from src.healing.prompts import QUERY_REWRITE_PROMPT, REWRITE_STRATEGIES
from src.healing.query_analysis import content_terms
from src.healing.types import QueryAnalysis

logger = logging.getLogger(__name__)

STRATEGY_ORDER: tuple[str, ...] = ("expand", "keywords", "decompose")

# Small domain-agnostic expansion table for the deterministic fallback. It only
# needs to push BM25/embeddings toward different vocabulary, not be complete.
_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "rag": ("retrieval-augmented", "generation"),
    "llm": ("large", "language", "model"),
    "llms": ("large", "language", "models"),
    "bm25": ("keyword", "search"),
    "rrf": ("reciprocal", "rank", "fusion"),
    "ci": ("continuous", "integration"),
    "rerank": ("cross-encoder", "re-ranking"),
    "reranker": ("cross-encoder", "re-ranking"),
    "embedding": ("vector", "semantic"),
    "embeddings": ("vector", "semantic"),
    "hallucination": ("unsupported", "claims", "faithfulness"),
    "hallucinations": ("unsupported", "claims", "faithfulness"),
    "accuracy": ("faithfulness", "precision"),
    "search": ("retrieval",),
    "fast": ("latency", "speed"),
    "slow": ("latency",),
}


class _RewriteOutput(BaseModel):
    rewritten_query: str
    rationale: str = ""


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


class QueryRewriter:
    """Rewrites a failing query into a different retrieval query."""

    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None

    @staticmethod
    def strategy_for(rewrite_index: int) -> str:
        """Strategy for the Nth rewrite (0-based), cycling through STRATEGY_ORDER."""
        return STRATEGY_ORDER[rewrite_index % len(STRATEGY_ORDER)]

    def rewrite(
        self,
        analysis: QueryAnalysis,
        previous_queries: list[str],
        missing_terms: list[str],
        diagnosis: str,
        rewrite_index: int = 0,
    ) -> tuple[str, str]:
        """Return ``(new_query, strategy_used)``; new_query is never in previous_queries."""
        strategy = self.strategy_for(rewrite_index)
        tried = {_norm(q) for q in previous_queries} | {_norm(analysis.original)}

        if self.use_llm:
            candidate = self._llm_rewrite(
                analysis, previous_queries, missing_terms, diagnosis, strategy
            )
            if candidate and _norm(candidate) not in tried:
                return candidate, strategy
            logger.info("LLM rewrite unusable (empty or repeated); using deterministic rewrite")

        return self.deterministic_rewrite(analysis, tried, strategy), f"{strategy}:deterministic"

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_rewrite(
        self,
        analysis: QueryAnalysis,
        previous_queries: list[str],
        missing_terms: list[str],
        diagnosis: str,
        strategy: str,
    ) -> str | None:
        from src.evaluation.metrics import _extract_json_object

        assert self.llm_client is not None
        prompt = QUERY_REWRITE_PROMPT.format(
            question=analysis.original,
            previous="; ".join(f'"{q}"' for q in previous_queries) or "none",
            question_type=analysis.question_type,
            missing_terms=", ".join(missing_terms) or "none",
            diagnosis=diagnosis,
            strategy=REWRITE_STRATEGIES[strategy],
        )
        try:
            raw = self.llm_client.complete(prompt=prompt, temperature=0.3, max_tokens=200)
        except Exception as exc:
            logger.warning("Query rewrite LLM call failed: %s", exc)
            return None
        data = _extract_json_object(raw or "")
        if data is None:
            return None
        try:
            out = _RewriteOutput.model_validate(data)
        except ValidationError:
            return None
        query = out.rewritten_query.strip()
        # Guard against runaway outputs being used as a search query.
        if not query or len(query) > 500:
            return None
        return query

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def deterministic_rewrite(analysis: QueryAnalysis, tried: set[str], strategy: str) -> str:
        """Rule-based rewrite that is guaranteed to differ from every query in ``tried``."""
        terms = analysis.content_terms or content_terms(analysis.original)

        expanded: list[str] = []
        for t in terms:
            expanded.append(t)
            expanded.extend(e for e in _EXPANSIONS.get(t, ()) if e not in expanded)

        if strategy == "expand":
            candidates = [" ".join(expanded), " ".join(terms)]
        elif strategy == "keywords":
            candidates = [" ".join(terms), " ".join(sorted(set(expanded)))]
        else:  # decompose: focus on the most specific (longest) terms
            focus = sorted(terms, key=len, reverse=True)[:3]
            candidates = [" ".join(focus), " ".join(expanded[: max(1, len(expanded) // 2)])]

        for c in candidates:
            if c and _norm(c) not in tried:
                return c

        # Last resort: append a distinguishing suffix so the query is still new.
        base = " ".join(expanded) or analysis.original
        suffix = 1
        while _norm(f"{base} overview {suffix}") in tried:
            suffix += 1
        return f"{base} overview {suffix}"
