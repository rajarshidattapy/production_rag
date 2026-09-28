"""Shared data types for the self-healing loop.

Everything that crosses a component boundary (verifier → classifier → router →
repair strategy) is a Pydantic model or enum, so workflow control never depends
on parsing free-form LLM text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FailureType(StrEnum):
    """Diagnosed cause of a failed verification."""

    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    QUERY_FAILURE = "QUERY_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    CITATION_FAILURE = "CITATION_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class RepairAction(StrEnum):
    """What the router decides to do after a verification."""

    ACCEPT = "accept"
    REWRITE_AND_RETRIEVE = "rewrite_and_retrieve"
    EXPAND_CONTEXT = "expand_context"
    REGENERATE = "regenerate"
    ABSTAIN = "abstain"


class QueryAnalysis(BaseModel):
    """Deterministic analysis of a user query."""

    original: str
    content_terms: list[str] = Field(default_factory=list)
    question_type: str = "other"
    is_vague: bool = False


class RetrievalSignals(BaseModel):
    """Deterministic retrieval-quality signals computed from retrieved context."""

    num_contexts: int = 0
    top_score: float | None = None
    score_kind: Literal["rerank", "vector", "rrf", "none"] = "none"
    query_term_coverage: float = 0.0
    missing_terms: list[str] = Field(default_factory=list)
    truncated_by_budget: bool = False


class CitationSignals(BaseModel):
    """Deterministic citation checks on a generated answer."""

    markers: list[int] = Field(default_factory=list)
    out_of_range: list[int] = Field(default_factory=list)
    has_citations: bool = False
    support_ratio: float = 1.0
    valid: bool = False


class JudgeVerdict(BaseModel):
    """Structured output of the LLM judge. Parsed and validated, never trusted as raw text."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    context_sufficient: bool
    contradiction_detected: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = ""


class VerificationResult(BaseModel):
    """Output of the verifier for one generation attempt."""

    passed: bool
    faithfulness: float | None = None
    relevance: float | None = None
    citation_valid: bool = False
    retrieval_sufficient: bool = False
    contradiction_detected: bool = False
    is_refusal: bool = False
    empty_answer: bool = False
    judge_available: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    failure_type: FailureType | None = None
    reason: str = ""
    retrieval: RetrievalSignals = Field(default_factory=RetrievalSignals)
    citations: CitationSignals = Field(default_factory=CitationSignals)


class RetrievalPlan(BaseModel):
    """How to retrieve for one attempt. Each healing retry produces a *different* plan."""

    queries: list[str]
    fetch_k: int
    final_k: int
    use_hybrid: bool
    use_reranker: bool
    alpha: float | None = None
    context_budget: int
    strategy: str = "initial"


class AttemptRecord(BaseModel):
    """Audit record for one generate→verify cycle."""

    attempt: int
    action: str
    plan: RetrievalPlan
    verification: VerificationResult
    latency_seconds: float = 0.0


class HealingInfo(BaseModel):
    """Healing metadata surfaced to API clients (optional, additive field)."""

    status: Literal["passed", "recovered", "abstained"]
    attempts: int
    recovered: bool
    reason: str | None = None
    failure_types: list[FailureType] = Field(default_factory=list)
    query_rewrites: list[str] = Field(default_factory=list)
    final_faithfulness: float | None = None
    final_relevance: float | None = None
    citation_valid: bool | None = None


class HealingResult(BaseModel):
    """Full result of a healed query."""

    model_config = {"arbitrary_types_allowed": True}

    answer: str
    citations: list[Any] = Field(default_factory=list)
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    info: HealingInfo
    history: list[AttemptRecord] = Field(default_factory=list)
    latency_seconds: float = 0.0

    @property
    def abstained(self) -> bool:
        return self.info.status == "abstained"
