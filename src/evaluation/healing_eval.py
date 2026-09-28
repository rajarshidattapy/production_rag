"""Evaluation of the self-healing loop against a baseline (healing off).

Every example is run twice through the same pipeline:

1. **Baseline** — ``pipeline.query(..., self_heal=False)`` (today's behaviour).
2. **Healed**   — ``pipeline.query_with_healing(...)``.

Final answers are scored with the *existing* evaluation judges
(``FaithfulnessScorer`` / ``AnswerRelevanceScorer``), not with the healing
verifier, so healing is not grading its own homework.

Reported metrics
----------------
initial_failure_rate            first attempt failed verification (any reason)
initial_retrieval_failure_rate  first attempt failed with RETRIEVAL/QUERY failure
healing_success_rate            recovered / first-attempt failures
avg_healing_attempts            mean generate→verify attempts per query
final_faithfulness              judge faithfulness of healed, non-abstained answers
final_relevance                 judge relevance of healed, non-abstained answers
baseline_faithfulness/relevance same judges on the baseline answers (for the delta)
citation_validity_rate          deterministic citation check on healed answers
abstention_rate                 healed queries that abstained
correct_abstention_rate         abstained on examples marked expected_behavior=abstain
false_abstention_rate           abstained on examples marked expected_behavior=answer
false_positive_healing_rate     healing triggered but the baseline answer was already
                                judged faithful AND relevant (unnecessary repair)
avg_latency_overhead_seconds    mean(healed latency − baseline latency)
latency_overhead_ratio          mean healed latency / mean baseline latency
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.metrics import AnswerRelevanceScorer, FaithfulnessScorer
from src.healing.types import FailureType
from src.healing.verifier import check_citations, is_refusal

logger = logging.getLogger(__name__)

DEFAULT_HEALING_DATASET = Path("data/golden_dataset/healing_dataset.jsonl")
_RETRIEVAL_SIDE = {FailureType.RETRIEVAL_FAILURE.value, FailureType.QUERY_FAILURE.value}


@dataclass
class HealingEvalRecord:
    example_id: str
    question: str
    category: str
    expected_behavior: str
    # baseline
    baseline_answer: str = ""
    baseline_latency: float = 0.0
    baseline_refused: bool = False
    baseline_faithfulness: float | None = None
    baseline_relevance: float | None = None
    # healed
    healed_answer: str = ""
    healed_status: str = ""
    healed_latency: float = 0.0
    attempts: int = 0
    failure_types: list[str] = field(default_factory=list)
    query_rewrites: list[str] = field(default_factory=list)
    first_attempt_failed: bool = False
    first_failure_type: str | None = None
    recovered: bool = False
    abstained: bool = False
    citation_valid: bool | None = None
    final_faithfulness: float | None = None
    final_relevance: float | None = None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


class HealingEvaluator:
    """Runs baseline vs. healed queries and reports healing-specific metrics."""

    def __init__(
        self,
        pipeline: Any,
        dataset: GoldenDataset | None = None,
        eval_model: str = "gpt-4o-mini",
        eval_provider: Literal["openai", "anthropic"] = "openai",
        faithfulness_threshold: float = 0.7,
        relevance_threshold: float = 0.7,
        results_dir: Path | str | None = None,
        faithfulness_scorer: Any | None = None,
        relevance_scorer: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.dataset = dataset or GoldenDataset(DEFAULT_HEALING_DATASET)
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold
        self.results_dir = Path(results_dir) if results_dir else Path("data/eval_results")
        self.faithfulness_scorer = faithfulness_scorer or FaithfulnessScorer(
            model=eval_model, provider=eval_provider
        )
        self.relevance_scorer = relevance_scorer or AnswerRelevanceScorer(
            model=eval_model, provider=eval_provider
        )

    # ------------------------------------------------------------------

    def run(self, use_hybrid: bool = True, use_reranker: bool = False) -> list[HealingEvalRecord]:
        records = [
            self._evaluate(ex, use_hybrid=use_hybrid, use_reranker=use_reranker)
            for ex in self.dataset.examples
        ]
        self._save(records)
        return records

    def _score(
        self, question: str, answer: str, contexts: list[dict[str, Any]]
    ) -> tuple[float, float]:
        faith = self.faithfulness_scorer.score(answer, contexts).get("faithfulness_score", 0.0)
        rel = self.relevance_scorer.score(question, answer).get("relevance_score", 0.0)
        return float(faith), float(rel)

    def _evaluate(self, ex: EvalExample, use_hybrid: bool, use_reranker: bool) -> HealingEvalRecord:
        rec = HealingEvalRecord(
            example_id=ex.id,
            question=ex.question,
            category=ex.category,
            expected_behavior=ex.expected_behavior,
        )

        # --- Baseline -------------------------------------------------
        t0 = time.perf_counter()
        base_answer, base_citations = self.pipeline.query(
            ex.question, use_hybrid=use_hybrid, use_reranker=use_reranker, self_heal=False
        )
        rec.baseline_latency = round(time.perf_counter() - t0, 4)
        rec.baseline_answer = base_answer
        rec.baseline_refused = is_refusal(base_answer)
        base_contexts = [
            {"document": c.text_snippet, "metadata": {"source": c.source}} for c in base_citations
        ]
        if base_contexts:
            rec.baseline_faithfulness, rec.baseline_relevance = self._score(
                ex.question, base_answer, base_contexts
            )

        # --- Healed ---------------------------------------------------
        t0 = time.perf_counter()
        result = self.pipeline.query_with_healing(
            ex.question, use_hybrid=use_hybrid, use_reranker=use_reranker
        )
        rec.healed_latency = round(time.perf_counter() - t0, 4)
        rec.healed_answer = result.answer
        rec.healed_status = result.info.status
        rec.attempts = result.info.attempts
        rec.failure_types = [f.value for f in result.info.failure_types]
        rec.query_rewrites = list(result.info.query_rewrites)
        rec.recovered = result.info.recovered
        rec.abstained = result.abstained
        rec.first_attempt_failed = (
            bool(result.history) and not result.history[0].verification.passed
        )
        if rec.first_attempt_failed:
            ft = result.history[0].verification.failure_type
            rec.first_failure_type = ft.value if ft else None

        if not result.abstained and result.contexts:
            rec.citation_valid = check_citations(result.answer, result.contexts).valid
            rec.final_faithfulness, rec.final_relevance = self._score(
                ex.question, result.answer, result.contexts
            )
        return rec

    # ------------------------------------------------------------------

    def summarize(self, records: list[HealingEvalRecord]) -> dict[str, Any]:
        n = len(records)
        if not n:
            return {"count": 0}
        failed_first = [r for r in records if r.first_attempt_failed]
        answered = [r for r in records if not r.abstained]
        expect_abstain = [r for r in records if r.expected_behavior == "abstain"]
        expect_answer = [r for r in records if r.expected_behavior == "answer"]

        def baseline_ok(r: HealingEvalRecord) -> bool:
            return (
                r.baseline_faithfulness is not None
                and r.baseline_relevance is not None
                and r.baseline_faithfulness >= self.faithfulness_threshold
                and r.baseline_relevance >= self.relevance_threshold
                and not r.baseline_refused
            )

        cited = [r for r in answered if r.citation_valid is not None]
        base_lat = _mean([r.baseline_latency for r in records]) or 0.0
        heal_lat = _mean([r.healed_latency for r in records]) or 0.0

        return {
            "count": n,
            "initial_failure_rate": _rate(len(failed_first), n),
            "initial_retrieval_failure_rate": _rate(
                sum(1 for r in failed_first if r.first_failure_type in _RETRIEVAL_SIDE), n
            ),
            "healing_success_rate": _rate(
                sum(1 for r in failed_first if r.recovered), len(failed_first)
            ),
            "avg_healing_attempts": _mean([float(r.attempts) for r in records]),
            "final_faithfulness": _mean(
                [r.final_faithfulness for r in answered if r.final_faithfulness is not None]
            ),
            "final_relevance": _mean(
                [r.final_relevance for r in answered if r.final_relevance is not None]
            ),
            "baseline_faithfulness": _mean(
                [r.baseline_faithfulness for r in records if r.baseline_faithfulness is not None]
            ),
            "baseline_relevance": _mean(
                [r.baseline_relevance for r in records if r.baseline_relevance is not None]
            ),
            "citation_validity_rate": _rate(sum(1 for r in cited if r.citation_valid), len(cited)),
            "abstention_rate": _rate(sum(1 for r in records if r.abstained), n),
            "correct_abstention_rate": _rate(
                sum(1 for r in expect_abstain if r.abstained), len(expect_abstain)
            ),
            "false_abstention_rate": _rate(
                sum(1 for r in expect_answer if r.abstained), len(expect_answer)
            ),
            "false_positive_healing_rate": _rate(
                sum(1 for r in failed_first if baseline_ok(r)), len(failed_first)
            ),
            "avg_latency_overhead_seconds": round(heal_lat - base_lat, 4),
            "latency_overhead_ratio": round(heal_lat / base_lat, 3) if base_lat else None,
        }

    def _save(self, records: list[HealingEvalRecord]) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"healing_eval_{stamp}.json"
        payload = {
            "timestamp": stamp,
            "summary": self.summarize(records),
            "records": [asdict(r) for r in records],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Healing evaluation saved to %s", path)
        return path

    def print_report(self, records: list[HealingEvalRecord]) -> None:
        s = self.summarize(records)

        def fmt(v: Any) -> str:
            if v is None:
                return "n/a"
            return f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else str(v)

        print("=" * 64)
        print("SELF-HEALING EVALUATION REPORT")
        print("=" * 64)
        rows = [
            ("Examples", s["count"]),
            ("Initial failure rate", fmt(s["initial_failure_rate"])),
            ("Initial retrieval failure rate", fmt(s["initial_retrieval_failure_rate"])),
            ("Healing success rate", fmt(s["healing_success_rate"])),
            ("Avg attempts", s["avg_healing_attempts"]),
            (
                "Faithfulness baseline → healed",
                f"{s['baseline_faithfulness']} → {s['final_faithfulness']}",
            ),
            ("Relevance baseline → healed", f"{s['baseline_relevance']} → {s['final_relevance']}"),
            ("Citation validity", fmt(s["citation_validity_rate"])),
            ("Abstention rate", fmt(s["abstention_rate"])),
            ("Correct abstentions (unanswerable)", fmt(s["correct_abstention_rate"])),
            ("False abstentions (answerable)", fmt(s["false_abstention_rate"])),
            ("False-positive healing", fmt(s["false_positive_healing_rate"])),
            (
                "Latency overhead (s, ×)",
                f"{s['avg_latency_overhead_seconds']}, {s['latency_overhead_ratio']}",
            ),
        ]
        for label, value in rows:
            print(f"  {label:<36} {value}")
        print()
        for r in records:
            print(
                f"  [{r.healed_status:<9}] attempts={r.attempts} {r.category:<20} {r.question[:52]}"
            )
