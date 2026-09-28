"""The self-healing state machine.

    START → analyze → retrieve → rerank → generate → verify → route
      route ├─ PASS                        → END (answer)
            ├─ RETRIEVAL/QUERY_FAILURE     → rewrite → retrieve (adapted plan) → …
            ├─ CONTEXT_FAILURE             → retrieve (wider, bigger budget) → …
            ├─ GENERATION/CITATION/UNKNOWN → regenerate (repair prompt) → …
            │    (same failure twice after regenerating → escalate to rewrite → retrieve)
            └─ attempts exhausted          → abstain

This is a deliberately small hand-written graph rather than a LangGraph
dependency: there are five nodes and one routing function. ``route`` is a pure
function over verification results so every transition is unit-testable, and
retrieval, reranking, generation, and verification remain separate components
reached through the ``RAGBackend`` protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, TypeVar

from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy, fuse_ranked_lists
from src.healing.metrics import HealingMetrics, healing_metrics
from src.healing.prompts import (
    CITATION_REPAIR_PROMPT,
    CONTEXT_REPAIR_PROMPT,
    GENERATION_REPAIR_PROMPT,
    fill,
)
from src.healing.query_analysis import analyze_query
from src.healing.query_rewriter import QueryRewriter
from src.healing.types import (
    AttemptRecord,
    FailureType,
    HealingInfo,
    HealingResult,
    QueryAnalysis,
    RepairAction,
    RetrievalPlan,
    VerificationResult,
)
from src.healing.verifier import ScoreKind, Verifier

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = "Unable to answer reliably from the available context."

_RETRIEVAL_FAILURES = {FailureType.RETRIEVAL_FAILURE, FailureType.QUERY_FAILURE}
_GENERATION_FAILURES = {
    FailureType.GENERATION_FAILURE,
    FailureType.CITATION_FAILURE,
    FailureType.UNKNOWN_FAILURE,
}

_VERIFY_SPAN_FIELDS = {
    "passed",
    "failure_type",
    "reason",
    "faithfulness",
    "relevance",
    "citation_valid",
    "retrieval_sufficient",
}

T = TypeVar("T")


# ----------------------------------------------------------------------
# Collaborator protocols
# ----------------------------------------------------------------------


class RAGBackend(Protocol):
    """The existing pipeline components the healing loop drives."""

    def retrieve(
        self,
        query: str,
        *,
        use_hybrid: bool,
        k: int,
        fetch_k: int,
        alpha: float | None,
        lang: str,
    ) -> list[dict[str, Any]]: ...

    def rerank(
        self, query: str, contexts: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]: ...

    def apply_context_budget(
        self, contexts: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]: ...

    def generate(
        self, query: str, contexts: list[dict[str, Any]], system_prompt: str | None
    ) -> str: ...

    def build_citations(self, contexts: list[dict[str, Any]]) -> list[Any]: ...

    def corpus_is_empty(self, lang: str) -> bool: ...


class HealingObserver(Protocol):
    """Receives one span per healing step (e.g. ``retrieve_attempt_2``, ``query_rewrite``)."""

    def on_step_start(
        self, name: str, input_data: dict[str, Any], metadata: dict[str, Any]
    ) -> None: ...

    def on_step_end(
        self,
        name: str,
        output: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def on_step_error(self, name: str, exc: Exception) -> None: ...


class NullObserver:
    def on_step_start(
        self, name: str, input_data: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        pass

    def on_step_end(
        self,
        name: str,
        output: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_step_error(self, name: str, exc: Exception) -> None:
        pass


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def route(
    verification: VerificationResult,
    attempt: int,
    max_attempts: int,
    last_action: RepairAction | None = None,
    last_failure: FailureType | None = None,
) -> RepairAction:
    """Decide the next transition after verifying attempt number ``attempt`` (1-based)."""
    if verification.passed:
        return RepairAction.ACCEPT
    if attempt >= max_attempts:
        return RepairAction.ABSTAIN

    failure = verification.failure_type or FailureType.UNKNOWN_FAILURE
    if failure in _RETRIEVAL_FAILURES:
        return RepairAction.REWRITE_AND_RETRIEVE
    if failure == FailureType.CONTEXT_FAILURE:
        return RepairAction.EXPAND_CONTEXT
    # Generation-side failure. If regenerating already failed the same way, the
    # context is the more likely culprit — escalate to retrieval repair.
    if last_action == RepairAction.REGENERATE and last_failure == failure:
        return RepairAction.REWRITE_AND_RETRIEVE
    return RepairAction.REGENERATE


def abstention_reason(failure: FailureType | None) -> str:
    if failure in _RETRIEVAL_FAILURES or failure == FailureType.CONTEXT_FAILURE:
        return "insufficient_verified_context"
    if failure == FailureType.CITATION_FAILURE:
        return "unverifiable_citations"
    if failure == FailureType.GENERATION_FAILURE:
        return "unverifiable_answer"
    return "verification_failed"


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


class HealingEngine:
    """Runs the generate → verify → diagnose → repair → re-verify loop."""

    def __init__(
        self,
        backend: RAGBackend,
        verifier: Verifier,
        rewriter: QueryRewriter,
        policy: AdaptiveRetrievalPolicy,
        max_attempts: int = 3,
        top_k_retrieval: int = 20,
        context_budget: int = 12000,
        metrics: HealingMetrics | None = None,
        observer: HealingObserver | None = None,
        abstention_message: str = ABSTENTION_MESSAGE,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.backend = backend
        self.verifier = verifier
        self.rewriter = rewriter
        self.policy = policy
        self.max_attempts = max_attempts
        self.top_k_retrieval = top_k_retrieval
        self.context_budget = context_budget
        self.metrics = metrics or healing_metrics
        self.observer: HealingObserver = observer or NullObserver()
        self.abstention_message = abstention_message

    # -- instrumentation helper ------------------------------------------------

    def _step(
        self,
        name: str,
        input_data: dict[str, Any],
        fn: Callable[[], T],
        summarize: Callable[[T], dict[str, Any]],
    ) -> T:
        self._safe(self.observer.on_step_start, name, input_data, {})
        start = time.monotonic()
        try:
            out = fn()
        except Exception as exc:
            self._safe(self.observer.on_step_error, name, exc)
            raise
        self._safe(self.observer.on_step_end, name, summarize(out), time.monotonic() - start)
        return out

    @staticmethod
    def _safe(hook: Callable[..., Any], *args: Any) -> None:
        # Observability must never break the query path.
        try:
            hook(*args)
        except Exception as exc:
            logger.warning("Healing observer hook failed: %s", exc)

    # -- nodes ---------------------------------------------------------------

    def _retrieve(
        self, question: str, plan: RetrievalPlan, lang: str, attempt: int
    ) -> list[dict[str, Any]]:
        def do_retrieve() -> list[dict[str, Any]]:
            lists = [
                self.backend.retrieve(
                    q,
                    use_hybrid=plan.use_hybrid,
                    k=plan.final_k,
                    fetch_k=plan.fetch_k,
                    alpha=plan.alpha,
                    lang=lang,
                )
                for q in plan.queries
            ]
            return fuse_ranked_lists(lists, k=plan.fetch_k)

        contexts = self._step(
            f"retrieve_attempt_{attempt}",
            {
                "queries": plan.queries,
                "strategy": plan.strategy,
                **plan.model_dump(exclude={"queries"}),
            },
            do_retrieve,
            lambda r: {"num_contexts": len(r), "chunk_ids": [c.get("id") for c in r]},
        )
        if not contexts:
            return contexts
        if plan.use_reranker:
            # Rerank against the ORIGINAL question: rewrites exist to improve
            # recall, but relevance is judged against what the user asked.
            return self._step(
                f"rerank_attempt_{attempt}",
                {"query": question, "candidates": len(contexts), "top_k": plan.final_k},
                lambda: self.backend.rerank(question, contexts, plan.final_k),
                lambda r: {"selected_ids": [c.get("id") for c in r]},
            )
        return contexts[: plan.final_k]

    @staticmethod
    def _score_kind(plan: RetrievalPlan) -> ScoreKind:
        if plan.use_reranker:
            return "rerank"
        if plan.use_hybrid or len(plan.queries) > 1:
            return "rrf"
        return "vector"

    # -- main loop -----------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        lang: str = "en",
        top_k: int = 5,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> HealingResult:
        started = time.monotonic()
        analysis: QueryAnalysis = self._step(
            "analyze",
            {"question": question},
            lambda: analyze_query(question),
            lambda a: a.model_dump(),
        )
        plan = self.policy.initial_plan(
            question,
            final_k=top_k,
            top_k_retrieval=self.top_k_retrieval,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            context_budget=self.context_budget,
        )

        history: list[AttemptRecord] = []
        rewrites: list[str] = []
        failures: list[FailureType] = []
        retrieval_retries = 0
        system_prompt: str | None = None
        last_action: RepairAction | None = None
        last_failure: FailureType | None = None
        attempt = 1
        action = "initial"

        contexts = self._retrieve(question, plan, lang, attempt)

        while True:
            attempt_start = time.monotonic()
            gen_contexts = self.backend.apply_context_budget(contexts, plan.context_budget)
            truncated = len(gen_contexts) < len(contexts) or any(
                g.get("document") != c.get("document")
                for g, c in zip(gen_contexts, contexts, strict=False)
            )

            if gen_contexts:
                answer = self._step(
                    f"generate_attempt_{attempt}",
                    {
                        "query": question,
                        "contexts": len(gen_contexts),
                        "repair_prompt": system_prompt is not None,
                    },
                    partial(self.backend.generate, question, gen_contexts, system_prompt),
                    lambda a: {"answer": a},
                )
            else:
                answer = ""  # nothing to ground on — don't pay for an LLM call

            verification = self._step(
                f"verify_attempt_{attempt}",
                {"attempt": attempt},
                partial(
                    self.verifier.verify,
                    question,
                    answer,
                    gen_contexts,
                    analysis=analysis,
                    score_kind=self._score_kind(plan),
                    truncated=truncated,
                ),
                lambda v: v.model_dump(include=_VERIFY_SPAN_FIELDS),
            )
            history.append(
                AttemptRecord(
                    attempt=attempt,
                    action=action,
                    plan=plan,
                    verification=verification,
                    latency_seconds=round(time.monotonic() - attempt_start, 4),
                )
            )

            next_action = route(verification, attempt, self.max_attempts, last_action, last_failure)

            if next_action == RepairAction.ACCEPT:
                outcome = "passed" if attempt == 1 else "recovered"
                return self._finish(
                    outcome,
                    answer,
                    gen_contexts,
                    history,
                    rewrites,
                    failures,
                    verification,
                    started,
                )

            failure = verification.failure_type or FailureType.UNKNOWN_FAILURE
            failures.append(failure)
            self.metrics.record_failure(failure.value)

            if next_action == RepairAction.ABSTAIN:
                return self._abstain(
                    abstention_reason(failure), history, rewrites, failures, started
                )

            if (
                failure == FailureType.RETRIEVAL_FAILURE
                and verification.retrieval.num_contexts == 0
                and self.backend.corpus_is_empty(lang)
            ):
                # No rewrite can find documents in an empty index.
                return self._abstain("empty_knowledge_base", history, rewrites, failures, started)

            attempt += 1
            feedback = self._feedback(verification)

            if next_action == RepairAction.REWRITE_AND_RETRIEVE:
                new_query, strategy = self._step(
                    "query_rewrite",
                    {
                        "attempt": attempt,
                        "previous": [question, *rewrites],
                        "failure": failure.value,
                    },
                    partial(
                        self.rewriter.rewrite,
                        analysis,
                        previous_queries=[question, *rewrites],
                        missing_terms=verification.retrieval.missing_terms,
                        diagnosis=verification.reason,
                        rewrite_index=len(rewrites),
                    ),
                    lambda r: {"rewritten_query": r[0], "strategy": r[1]},
                )
                rewrites.append(new_query)
                retrieval_retries += 1
                self.metrics.record_query_rewrite()
                self.metrics.record_retrieval_retry()
                plan = self.policy.retrieval_retry_plan(
                    plan, failure, question, rewrites, retrieval_retries
                )
                plan = plan.model_copy(update={"strategy": f"{plan.strategy}[{strategy}]"})
                contexts = self._retrieve(question, plan, lang, attempt)
                template = GENERATION_REPAIR_PROMPT

            elif next_action == RepairAction.EXPAND_CONTEXT:
                retrieval_retries += 1
                self.metrics.record_retrieval_retry()
                plan = self.policy.context_expansion_plan(plan)
                contexts = self._retrieve(question, plan, lang, attempt)
                template = CONTEXT_REPAIR_PROMPT

            else:  # REGENERATE — same verified contexts, stricter prompt
                self.metrics.record_regeneration()
                template = (
                    CITATION_REPAIR_PROMPT
                    if failure == FailureType.CITATION_FAILURE
                    else GENERATION_REPAIR_PROMPT
                )

            # The repair prompt's valid-source range must match what the
            # generator will actually see after budgeting.
            n_visible = len(self.backend.apply_context_budget(contexts, plan.context_budget))
            system_prompt = fill(template, feedback=feedback, num_sources=max(n_visible, 1))

            last_action = next_action
            last_failure = failure
            action = next_action.value

    async def run_async(self, question: str, **kwargs: Any) -> HealingResult:
        # The loop is inherently sequential (each step depends on the last
        # verification), so one worker thread is the right shape; it also
        # keeps a single implementation for sync and async callers.
        return await asyncio.to_thread(self.run, question, **kwargs)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _feedback(v: VerificationResult) -> str:
        lines = [f"- {v.failure_type.value if v.failure_type else 'FAILURE'}: {v.reason}"]
        if v.unsupported_claims:
            lines.append("- Unsupported claims to remove: " + "; ".join(v.unsupported_claims[:5]))
        if v.citations.out_of_range:
            lines.append(f"- Invalid citation numbers used: {v.citations.out_of_range}")
        return "\n".join(lines)

    def _finish(
        self,
        outcome: str,
        answer: str,
        contexts: list[dict[str, Any]],
        history: list[AttemptRecord],
        rewrites: list[str],
        failures: list[FailureType],
        verification: VerificationResult,
        started: float,
    ) -> HealingResult:
        latency = time.monotonic() - started
        self.metrics.record_outcome(outcome, len(history), latency)
        info = HealingInfo(
            status="passed" if outcome == "passed" else "recovered",
            attempts=len(history),
            recovered=outcome == "recovered",
            failure_types=failures,
            query_rewrites=rewrites,
            final_faithfulness=verification.faithfulness,
            final_relevance=verification.relevance,
            citation_valid=verification.citation_valid,
        )
        return HealingResult(
            answer=answer,
            # Citations are built from the passages the generator actually saw,
            # so [n] in the answer maps to citations[n-1].
            citations=self.backend.build_citations(contexts),
            contexts=contexts,
            info=info,
            history=history,
            latency_seconds=round(latency, 4),
        )

    def _abstain(
        self,
        reason: str,
        history: list[AttemptRecord],
        rewrites: list[str],
        failures: list[FailureType],
        started: float,
    ) -> HealingResult:
        latency = time.monotonic() - started
        self.metrics.record_outcome("abstained", len(history), latency)
        last = history[-1].verification if history else None
        logger.info("Healing abstained after %d attempt(s): %s", len(history), reason)
        return HealingResult(
            answer=self.abstention_message,
            citations=[],
            contexts=[],
            info=HealingInfo(
                status="abstained",
                attempts=len(history),
                recovered=False,
                reason=reason,
                failure_types=failures,
                query_rewrites=rewrites,
                final_faithfulness=last.faithfulness if last else None,
                final_relevance=last.relevance if last else None,
                citation_valid=last.citation_valid if last else None,
            ),
            history=history,
            latency_seconds=round(latency, 4),
        )
