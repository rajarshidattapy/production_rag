"""Tests for the baseline-vs-healing evaluator and runner self_heal passthrough."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.healing_eval import DEFAULT_HEALING_DATASET, HealingEvaluator
from src.evaluation.runner import EvaluationRunner
from src.generation.citations import Citation
from src.healing.types import (
    AttemptRecord,
    FailureType,
    HealingInfo,
    HealingResult,
    RetrievalPlan,
    VerificationResult,
)

PLAN = RetrievalPlan(
    queries=["q"], fetch_k=5, final_k=5, use_hybrid=True, use_reranker=False, context_budget=1000
)
CTX = [{"id": "c1", "document": "RRF uses a constant k of 60.", "metadata": {"source": "h.txt"}}]
CIT = [Citation(chunk_id="c1", source="h.txt", filename="h.txt", text_snippet=CTX[0]["document"])]


def _attempt(n: int, passed: bool, ft: FailureType | None = None) -> AttemptRecord:
    return AttemptRecord(
        attempt=n, action="initial", plan=PLAN,
        verification=VerificationResult(passed=passed, failure_type=ft),
    )


def _recovered() -> HealingResult:
    return HealingResult(
        answer="RRF uses a constant k of 60 [1].",
        citations=CIT,
        contexts=CTX,
        info=HealingInfo(status="recovered", attempts=2, recovered=True,
                         failure_types=[FailureType.QUERY_FAILURE]),
        history=[_attempt(1, False, FailureType.QUERY_FAILURE), _attempt(2, True)],
    )


def _abstained() -> HealingResult:
    return HealingResult(
        answer="Unable to answer reliably from the available context.",
        info=HealingInfo(status="abstained", attempts=3, recovered=False,
                         reason="insufficient_verified_context"),
        history=[_attempt(i, False, FailureType.RETRIEVAL_FAILURE) for i in (1, 2, 3)],
    )


def _dataset(tmp_path: Path) -> GoldenDataset:
    ds = GoldenDataset(tmp_path / "h.jsonl")
    ds.save([
        EvalExample(question="RRF k?", reference_answer="60", id="a", category="terse"),
        EvalExample(question="Who invented BM25?", reference_answer="n/a", id="b",
                    expected_behavior="abstain", category="unanswerable"),
    ])
    return GoldenDataset(tmp_path / "h.jsonl")


def _scorer(key: str, value: float) -> MagicMock:
    m = MagicMock()
    m.score.return_value = {key: value}
    return m


def test_healing_dataset_loads_with_new_fields() -> None:
    examples = GoldenDataset(DEFAULT_HEALING_DATASET).load()
    assert len(examples) >= 15
    assert {e.expected_behavior for e in examples} == {"answer", "abstain"}
    assert all(e.category for e in examples)


def test_evaluator_runs_baseline_and_healed_and_summarizes(tmp_path: Path) -> None:
    pipeline = MagicMock()
    pipeline.query.return_value = ("unhelpful answer", CIT)
    pipeline.query_with_healing.side_effect = [_recovered(), _abstained()]

    ev = HealingEvaluator(
        pipeline,
        dataset=_dataset(tmp_path),
        results_dir=tmp_path / "out",
        faithfulness_scorer=_scorer("faithfulness_score", 0.9),
        relevance_scorer=_scorer("relevance_score", 0.4),  # baseline "not ok" → no false positive
    )
    records = ev.run(use_hybrid=True)

    # Baseline is always run with healing explicitly off.
    for call in pipeline.query.call_args_list:
        assert call.kwargs["self_heal"] is False

    s = ev.summarize(records)
    assert s["count"] == 2
    assert s["initial_failure_rate"] == 1.0
    assert s["initial_retrieval_failure_rate"] == 1.0
    assert s["healing_success_rate"] == 0.5
    assert s["avg_healing_attempts"] == 2.5
    assert s["abstention_rate"] == 0.5
    assert s["correct_abstention_rate"] == 1.0
    assert s["false_abstention_rate"] == 0.0
    assert s["citation_validity_rate"] == 1.0
    assert s["false_positive_healing_rate"] == 0.0
    assert s["final_faithfulness"] == 0.9

    saved = list((tmp_path / "out").glob("healing_eval_*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["summary"]["count"] == 2


def test_false_positive_healing_detected(tmp_path: Path) -> None:
    pipeline = MagicMock()
    pipeline.query.return_value = ("RRF uses a constant k of 60 [1].", CIT)
    pipeline.query_with_healing.side_effect = [_recovered(), _abstained()]
    ev = HealingEvaluator(
        pipeline,
        dataset=_dataset(tmp_path),
        results_dir=tmp_path / "out",
        faithfulness_scorer=_scorer("faithfulness_score", 0.95),
        relevance_scorer=_scorer("relevance_score", 0.95),  # baseline already good
    )
    s = ev.summarize(ev.run())
    assert s["false_positive_healing_rate"] == 1.0


def test_runner_passes_self_heal_only_when_set(tmp_path: Path) -> None:
    pipeline = MagicMock()
    pipeline.query.return_value = ("a", [])
    runner = EvaluationRunner(pipeline=pipeline, dataset=_dataset(tmp_path), results_dir=tmp_path)
    ex = runner.dataset.examples[0]
    for scorer in ("scorer", "relevance_scorer", "context_precision_scorer", "context_recall_scorer"):
        setattr(runner, scorer, MagicMock(score=MagicMock(return_value={})))

    runner._evaluate_single(ex, False, False)
    assert "self_heal" not in pipeline.query.call_args.kwargs

    runner._evaluate_single(ex, False, False, True)
    assert pipeline.query.call_args.kwargs["self_heal"] is True
