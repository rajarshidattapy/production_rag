"""Failure diagnosis: map verifier signals to a single, actionable failure type.

Pure and deterministic — no model calls. The LLM judge (if enabled) contributes
*signals* to the verification result; this module decides what they mean.

Rule order matters and reflects which repair is most likely to help:

1. Nothing retrieved                                    → RETRIEVAL_FAILURE
2. Retrieval insufficient and no query term found in
   any passage, or the query itself is too vague         → QUERY_FAILURE
3. Retrieval insufficient (low coverage/score, judge
   says context lacks the answer, or model refused)      → RETRIEVAL_FAILURE
4. Passages contradict each other                        → CONTEXT_FAILURE
5. Empty answer                                          → GENERATION_FAILURE
6. Citations missing / out of range / misattributed      → CITATION_FAILURE
7. Faithfulness or relevance below threshold             → GENERATION_FAILURE
8. Anything else that failed                             → UNKNOWN_FAILURE
"""

from __future__ import annotations

from src.healing.types import FailureType, QueryAnalysis, VerificationResult


class FailureClassifier:
    """Classifies a failed verification into a ``FailureType``."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevance_threshold: float = 0.7,
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold

    def classify(
        self,
        result: VerificationResult,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[FailureType, str]:
        """Return ``(failure_type, human-readable reason)`` for a failed verification."""
        r = result.retrieval

        if r.num_contexts == 0:
            return (
                FailureType.RETRIEVAL_FAILURE,
                "Retrieval returned no passages for the query.",
            )

        if not result.retrieval_sufficient:
            if analysis is not None and analysis.is_vague:
                return (
                    FailureType.QUERY_FAILURE,
                    "The query has too few specific terms for retrieval to match on.",
                )
            if r.query_term_coverage == 0.0 and r.missing_terms:
                return (
                    FailureType.QUERY_FAILURE,
                    "None of the query's key terms appear in the retrieved passages "
                    f"(missing: {', '.join(r.missing_terms[:5])}); likely a vocabulary mismatch.",
                )
            if result.is_refusal:
                return (
                    FailureType.RETRIEVAL_FAILURE,
                    "The generator reported that the retrieved context does not contain the answer.",
                )
            return (
                FailureType.RETRIEVAL_FAILURE,
                "Retrieved context is insufficient to answer the question "
                f"(term coverage {r.query_term_coverage:.0%}).",
            )

        if result.is_refusal:
            # Deterministic checks thought retrieval looked fine, but the model
            # read the passages and found no answer — trust the model here.
            return (
                FailureType.RETRIEVAL_FAILURE,
                "The generator reported that the retrieved context does not contain the answer.",
            )

        if result.contradiction_detected:
            return (
                FailureType.CONTEXT_FAILURE,
                "Retrieved passages contradict each other on a point relevant to the question.",
            )

        if result.empty_answer:
            return FailureType.GENERATION_FAILURE, "The generator returned an empty answer."

        if not result.citation_valid:
            c = result.citations
            if c.out_of_range:
                why = f"cites non-existent sources {c.out_of_range}"
            elif not c.has_citations:
                why = "contains no source citations"
            else:
                why = f"cites sources that do not support the cited sentences (support {c.support_ratio:.0%})"
            return FailureType.CITATION_FAILURE, f"The answer {why}."

        if result.faithfulness is not None and result.faithfulness < self.faithfulness_threshold:
            claims = "; ".join(result.unsupported_claims[:3])
            detail = f" Unsupported: {claims}" if claims else ""
            return (
                FailureType.GENERATION_FAILURE,
                f"Faithfulness {result.faithfulness:.2f} is below threshold "
                f"{self.faithfulness_threshold:.2f}.{detail}",
            )

        if result.relevance is not None and result.relevance < self.relevance_threshold:
            return (
                FailureType.GENERATION_FAILURE,
                f"Answer relevance {result.relevance:.2f} is below threshold "
                f"{self.relevance_threshold:.2f}.",
            )

        return (
            FailureType.UNKNOWN_FAILURE,
            result.reason or "Verification failed for an unknown reason.",
        )
