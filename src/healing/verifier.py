"""Post-generation verifier.

Two layers:

1. **Deterministic checks** (always on, zero cost): empty answer, refusal
   detection, retrieval score floor, citation marker range, and lexical
   support of each cited sentence by the passage it cites. Query-term
   coverage is also computed; it gates retrieval only when no judge verdict
   is available (see ``verify``).
2. **LLM judge** (``RAG_VERIFIER_ENABLED``): faithfulness, relevance, context
   sufficiency, and unresolved contradictions, returned as JSON and validated
   into a ``JudgeVerdict`` Pydantic model. It only runs when the deterministic
   layer passes — there's no point paying for a judge call on an answer that
   already cites a non-existent source.

If the judge call fails or returns malformed output the verifier degrades to
deterministic-only (``judge_available=False``) rather than failing the query.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import ValidationError

from src.generation.llm_client import LLMClient
from src.healing.failure_classifier import FailureClassifier
from src.healing.prompts import VERIFIER_JUDGE_PROMPT
from src.healing.query_analysis import analyze_query, lexical_overlap, term_coverage
from src.healing.types import (
    CitationSignals,
    JudgeVerdict,
    QueryAnalysis,
    RetrievalSignals,
    VerificationResult,
)

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Phrases the generator is instructed to use when context is insufficient
# (see DEFAULT_SYSTEM_PROMPT and its de/es translations), plus common variants.
REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot find sufficient information",
    "could not find any relevant information",
    "can't find sufficient information",
    "does not contain sufficient information",
    "doesn't contain sufficient information",
    "do not contain sufficient information",
    "not enough information",
    "insufficient information",
    "keine ausreichenden informationen",
    "keine relevanten informationen",
    "información suficiente",
    "ninguna información relevante",
)

ScoreKind = Literal["rerank", "vector", "rrf", "none"]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def extract_citation_markers(answer: str) -> list[int]:
    """Return every source number cited in ``answer`` (``[1]``, ``[2, 3]``, ``[1][4]``)."""
    markers: list[int] = []
    for group in _CITATION_RE.findall(answer):
        markers.extend(int(n) for n in re.split(r"\s*[,;]\s*", group) if n)
    return markers


def check_citations(
    answer: str,
    contexts: list[dict[str, Any]],
    min_support: float = 0.5,
    min_sentence_overlap: float = 0.3,
) -> CitationSignals:
    """Validate citation markers against the passages actually shown to the generator."""
    n = len(contexts)
    markers = extract_citation_markers(answer)
    out_of_range = sorted({m for m in markers if m < 1 or m > n})

    cited_sentences = 0
    supported = 0
    for sentence in _SENTENCE_RE.split(answer):
        sent_markers = [m for m in extract_citation_markers(sentence) if 1 <= m <= n]
        if not sent_markers:
            continue
        cited_sentences += 1
        text = _CITATION_RE.sub(" ", sentence)
        reference = " ".join(contexts[m - 1].get("document", "") for m in set(sent_markers))
        if lexical_overlap(text, reference) >= min_sentence_overlap:
            supported += 1

    support_ratio = supported / cited_sentences if cited_sentences else 0.0
    has_citations = bool(markers)
    return CitationSignals(
        markers=markers,
        out_of_range=out_of_range,
        has_citations=has_citations,
        support_ratio=round(support_ratio, 4),
        valid=has_citations and not out_of_range and support_ratio >= min_support,
    )


def retrieval_signals(
    analysis: QueryAnalysis,
    contexts: list[dict[str, Any]],
    score_kind: ScoreKind,
    truncated: bool = False,
) -> RetrievalSignals:
    coverage, missing = term_coverage(analysis.content_terms, contexts)
    top_score: float | None = None
    if contexts and score_kind == "rerank":
        scores = [c["rerank_score"] for c in contexts if c.get("rerank_score") is not None]
        top_score = max(scores) if scores else None
    elif contexts and score_kind in ("vector", "rrf"):
        scores = [c["score"] for c in contexts if c.get("score") is not None]
        top_score = max(scores) if scores else None
    return RetrievalSignals(
        num_contexts=len(contexts),
        top_score=top_score,
        score_kind=score_kind if top_score is not None else "none",
        query_term_coverage=round(coverage, 4),
        missing_terms=missing,
        truncated_by_budget=truncated,
    )


def _format_numbered(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{i}] {c.get('document', '')}" for i, c in enumerate(contexts, start=1))


class Verifier:
    """Verifies a generated answer against its question and retrieved context."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        judge_enabled: bool = True,
        faithfulness_threshold: float = 0.7,
        relevance_threshold: float = 0.7,
        min_retrieval_score: float = 0.2,
        min_query_coverage: float = 0.4,
        min_citation_support: float = 0.5,
        classifier: FailureClassifier | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.judge_enabled = judge_enabled and llm_client is not None
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold
        self.min_retrieval_score = min_retrieval_score
        self.min_query_coverage = min_query_coverage
        self.min_citation_support = min_citation_support
        self.classifier = classifier or FailureClassifier(
            faithfulness_threshold=faithfulness_threshold,
            relevance_threshold=relevance_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        question: str,
        answer: str,
        contexts: list[dict[str, Any]],
        analysis: QueryAnalysis | None = None,
        score_kind: ScoreKind = "vector",
        truncated: bool = False,
    ) -> VerificationResult:
        analysis = analysis or analyze_query(question)
        retrieval = retrieval_signals(analysis, contexts, score_kind, truncated)
        citations = check_citations(answer, contexts, min_support=self.min_citation_support)
        refusal = is_refusal(answer)
        empty = not answer.strip()

        # RRF scores are rank-derived (~1/(k+rank)), not an absolute relevance
        # measure, so the score floor only applies to vector/rerank scores.
        score_ok = (
            retrieval.top_score is None
            or retrieval.score_kind == "rrf"
            or retrieval.top_score >= self.min_retrieval_score
        )
        base_retrieval_ok = retrieval.num_contexts > 0 and score_ok
        # Lexical coverage is a heuristic, not a fact: a paraphrased query can
        # retrieve exactly the right passage with little word overlap (that's
        # what dense retrieval is for). So it only gates retrieval when there
        # is no judge to ask; with a judge it's advisory (and still drives the
        # QUERY vs RETRIEVAL diagnosis).
        lexical_ok = retrieval.query_term_coverage >= self.min_query_coverage

        result = VerificationResult(
            passed=False,
            citation_valid=citations.valid,
            retrieval_sufficient=base_retrieval_ok and lexical_ok,
            is_refusal=refusal,
            empty_answer=empty,
            retrieval=retrieval,
            citations=citations,
        )

        hard_checks_ok = base_retrieval_ok and not refusal and not empty and citations.valid
        if hard_checks_ok and self.judge_enabled:
            verdict = self._judge(question, answer, contexts)
            if verdict is not None:
                result.judge_available = True
                result.faithfulness = verdict.faithfulness
                result.relevance = verdict.relevance
                result.contradiction_detected = verdict.contradiction_detected
                result.unsupported_claims = verdict.unsupported_claims
                result.retrieval_sufficient = base_retrieval_ok and verdict.context_sufficient

        result.passed = (
            hard_checks_ok
            and result.retrieval_sufficient
            and not result.contradiction_detected
            and (result.faithfulness is None or result.faithfulness >= self.faithfulness_threshold)
            and (result.relevance is None or result.relevance >= self.relevance_threshold)
        )

        if result.passed:
            result.reason = "All verification checks passed."
        else:
            failure_type, reason = self.classifier.classify(result, analysis)
            result.failure_type = failure_type
            result.reason = reason
        return result

    # ------------------------------------------------------------------
    # LLM judge
    # ------------------------------------------------------------------

    def _judge(
        self, question: str, answer: str, contexts: list[dict[str, Any]]
    ) -> JudgeVerdict | None:
        from src.evaluation.metrics import _extract_json_object

        assert self.llm_client is not None
        prompt = VERIFIER_JUDGE_PROMPT.format(
            question=question, context=_format_numbered(contexts), answer=answer
        )
        try:
            raw = self.llm_client.complete(prompt=prompt, temperature=0.0, max_tokens=512)
        except Exception as exc:
            logger.warning("Verifier judge call failed, using deterministic checks only: %s", exc)
            return None

        data = _extract_json_object(raw or "")
        if data is None:
            logger.warning("Verifier judge returned no JSON object; ignoring judge output")
            return None
        try:
            return JudgeVerdict.model_validate(data)
        except ValidationError as exc:
            logger.warning("Verifier judge output failed schema validation: %s", exc)
            return None
