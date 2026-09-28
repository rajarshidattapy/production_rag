"""Glue between ``RAGPipeline`` and the healing engine.

``PipelineBackend`` routes every healing step through the pipeline's existing
methods (``_retrieve``, ``_apply_reranker``, ``_apply_context_budget``,
``generator.generate``), so the monitoring wrappers installed by
``MonitoredRAGPipeline`` still see and trace each call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.generation.llm_client import LLMClient
from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy
from src.healing.failure_classifier import FailureClassifier
from src.healing.query_rewriter import QueryRewriter
from src.healing.state_machine import HealingEngine, HealingObserver
from src.healing.verifier import Verifier

if TYPE_CHECKING:
    from src.pipeline import RAGPipeline


class PipelineBackend:
    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def retrieve(
        self,
        query: str,
        *,
        use_hybrid: bool,
        k: int,
        fetch_k: int,
        alpha: float | None,
        lang: str,
    ) -> list[dict[str, Any]]:
        return self.pipeline._retrieve(
            query, use_hybrid=use_hybrid, k=k, lang=lang, fetch_k=fetch_k, alpha=alpha
        )

    def rerank(
        self, query: str, contexts: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        return self.pipeline._apply_reranker(query, contexts, top_k=top_k)

    def apply_context_budget(
        self, contexts: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        return self.pipeline._apply_context_budget(contexts, budget=budget)

    def generate(
        self, query: str, contexts: list[dict[str, Any]], system_prompt: str | None
    ) -> str:
        return self.pipeline.generator.generate(query, contexts, system_prompt=system_prompt)

    def build_citations(self, contexts: list[dict[str, Any]]) -> list[Any]:
        return self.pipeline.citation_formatter.build_citations(contexts)

    def corpus_is_empty(self, lang: str) -> bool:
        try:
            return self.pipeline._get_vector_store(lang).count() == 0
        except Exception:
            return False


def build_healing_engine(
    pipeline: RAGPipeline, observer: HealingObserver | None = None
) -> HealingEngine:
    """Construct a ``HealingEngine`` for ``pipeline`` from its settings."""
    cfg = pipeline.config
    generator = pipeline.generator

    judge_client = LLMClient(
        provider=generator.provider,
        model=cfg.verifier_llm_model or generator.model,
    )
    classifier = FailureClassifier(
        faithfulness_threshold=cfg.faithfulness_threshold,
        relevance_threshold=cfg.relevance_threshold,
    )
    verifier = Verifier(
        llm_client=judge_client,
        judge_enabled=cfg.verifier_enabled,
        faithfulness_threshold=cfg.faithfulness_threshold,
        relevance_threshold=cfg.relevance_threshold,
        min_retrieval_score=cfg.min_retrieval_score,
        min_query_coverage=cfg.min_query_coverage,
        min_citation_support=cfg.min_citation_support,
        classifier=classifier,
    )
    rewriter = QueryRewriter(
        llm_client=LLMClient(provider=generator.provider, model=generator.model),
        use_llm=cfg.healing_llm_rewrite,
    )
    policy = AdaptiveRetrievalPolicy(
        base_alpha=cfg.hybrid_alpha,
        fetch_k_multiplier=cfg.healing_fetch_k_multiplier,
        max_fetch_k=cfg.healing_max_fetch_k,
        context_budget_multiplier=cfg.healing_context_budget_multiplier,
        escalate_hybrid=cfg.healing_escalate_hybrid,
        rerank_on_retry=cfg.healing_rerank_on_retry,
    )
    return HealingEngine(
        backend=PipelineBackend(pipeline),
        verifier=verifier,
        rewriter=rewriter,
        policy=policy,
        max_attempts=cfg.max_healing_attempts,
        top_k_retrieval=cfg.top_k_retrieval,
        context_budget=cfg.max_context_chars,
        observer=observer,
    )
