"""Retrieval metrics, regression checks against a baseline, and search-weight calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import evaluate_retrieval
from src.evaluation.dataset import EvalExample, GoldenDataset
from src.evaluation.retrieval_metrics import mean_scores, score_query


class TestMetrics:
    def test_perfect_ranking(self) -> None:
        s = score_query(["a.txt", "a.txt", "b.txt"], ["a.txt"], relevant_chunks_in_corpus=2, k=3)
        assert s.recall == 1.0
        assert s.precision == pytest.approx(2 / 3)
        assert s.mrr == 1.0
        assert s.ndcg == pytest.approx(1.0)
        assert s.hit_rate == 1.0

    def test_relevant_chunk_ranked_third(self) -> None:
        s = score_query(["x", "y", "a.txt"], ["a.txt"], relevant_chunks_in_corpus=1, k=3)
        assert s.mrr == pytest.approx(1 / 3)
        assert s.ndcg == pytest.approx((1 / math.log2(4)) / 1.0)
        assert s.precision == pytest.approx(1 / 3)

    def test_nothing_relevant_retrieved(self) -> None:
        s = score_query(["x", "y"], ["a.txt"], relevant_chunks_in_corpus=3, k=2)
        assert s.as_dict() == {
            "recall": 0.0,
            "precision": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0,
        }

    def test_recall_counts_documents_not_chunks(self) -> None:
        s = score_query(
            ["a.txt", "a.txt", "a.txt"], ["a.txt", "b.txt"], relevant_chunks_in_corpus=4, k=3
        )
        assert s.recall == 0.5

    def test_only_the_top_k_count(self) -> None:
        s = score_query(["x", "y", "a.txt"], ["a.txt"], relevant_chunks_in_corpus=1, k=2)
        assert s.hit_rate == 0.0 and s.mrr == 0.0

    def test_ndcg_ideal_is_capped_by_what_the_corpus_holds(self) -> None:
        # Only one relevant chunk exists, so finding it first is a perfect score.
        s = score_query(["a.txt", "x", "y"], ["a.txt"], relevant_chunks_in_corpus=1, k=3)
        assert s.ndcg == pytest.approx(1.0)

    def test_mean(self) -> None:
        a = score_query(["a.txt"], ["a.txt"], 1, 1)
        b = score_query(["x"], ["a.txt"], 1, 1)
        assert mean_scores([a, b])["mrr"] == 0.5
        assert mean_scores([])["ndcg"] == 0.0


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "apples.txt").write_text(
        "Apples are red fruit that grow on apple trees in orchards.", encoding="utf-8"
    )
    (docs / "rockets.txt").write_text(
        "Rockets burn fuel to launch satellites into orbit around Earth.", encoding="utf-8"
    )
    (docs / "violins.txt").write_text(
        "Violins are string instruments played with a bow in orchestras.", encoding="utf-8"
    )
    dataset = tmp_path / "dataset.jsonl"
    GoldenDataset(dataset).save(
        [
            EvalExample(
                question="Which fruit grows on trees in orchards?",
                reference_answer="Apples",
                expected_sources=["apples.txt"],
            ),
            EvalExample(
                question="How do rockets launch satellites into orbit?",
                reference_answer="Fuel",
                expected_sources=["rockets.txt"],
            ),
            EvalExample(
                question="Which string instrument is played with a bow?",
                reference_answer="Violin",
                expected_sources=["violins.txt"],
            ),
            EvalExample(question="Unlabelled question", reference_answer="-", expected_sources=[]),
        ]
    )
    return docs, dataset


def _run(corpus: tuple[Path, Path], tmp_path: Path, *extra: str) -> int:
    docs, dataset = corpus
    return evaluate_retrieval.main(
        [
            "--corpus",
            str(docs),
            "--dataset",
            str(dataset),
            "--k",
            "2",
            "--baseline",
            str(tmp_path / "baseline.json"),
            *extra,
        ]
    )


class TestRegressionCheck:
    def test_baseline_round_trip(self, corpus: tuple[Path, Path], tmp_path: Path) -> None:
        assert _run(corpus, tmp_path, "--update-baseline") == 0
        saved = json.loads((tmp_path / "baseline.json").read_text())
        assert set(saved["results"]) == {"vector", "hybrid"}
        assert set(saved["results"]["hybrid"]) == {"recall", "precision", "mrr", "ndcg", "hit_rate"}
        assert saved["identity"]["k"] == 2
        assert _run(corpus, tmp_path) == 0

    def test_a_drop_beyond_tolerance_fails(self, corpus: tuple[Path, Path], tmp_path: Path) -> None:
        _run(corpus, tmp_path, "--update-baseline")
        path = tmp_path / "baseline.json"
        saved = json.loads(path.read_text())
        saved["results"]["hybrid"]["mrr"] = min(1.0, saved["results"]["hybrid"]["mrr"] + 0.5)
        if (
            saved["results"]["hybrid"]["mrr"]
            <= json.loads(path.read_text())["results"]["hybrid"]["mrr"]
        ):
            saved["results"]["hybrid"]["mrr"] = 2.0  # already perfect: pretend it used to be higher
        path.write_text(json.dumps(saved))
        assert _run(corpus, tmp_path) == 1

    def test_small_noise_within_tolerance_passes(
        self, corpus: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _run(corpus, tmp_path, "--update-baseline")
        path = tmp_path / "baseline.json"
        saved = json.loads(path.read_text())
        saved["results"]["vector"]["ndcg"] += 0.01
        path.write_text(json.dumps(saved))
        assert _run(corpus, tmp_path, "--tolerance", "0.02") == 0

    def test_changed_inputs_are_not_compared(
        self, corpus: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _run(corpus, tmp_path, "--update-baseline")
        (corpus[0] / "apples.txt").write_text("Apples changed.", encoding="utf-8")
        assert _run(corpus, tmp_path) == 2

    def test_missing_baseline_says_how_to_make_one(
        self, corpus: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(corpus, tmp_path) == 2
        assert "--update-baseline" in capsys.readouterr().err

    def test_store_is_isolated_from_the_app(
        self, corpus: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from src.config import settings
        from src.pipeline import RAGPipeline

        app = RAGPipeline()
        app.reset()
        _run(corpus, tmp_path, "--update-baseline")
        assert app.stats()["chunks_in_store"] == 0
        assert settings.chroma_path == tmp_path / "test_chroma_rag"


class TestCalibration:
    def test_sweep_runs_and_keeps_defaults_without_a_clear_gain(
        self, corpus: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(corpus, tmp_path, "--calibrate", "--min-gain", "10") == 0
        out = capsys.readouterr().out
        assert "alpha" in out and "rrf_k" in out
        assert "Keep the current settings" in out

    def test_suggests_settings_when_one_wins(
        self,
        corpus: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        real = evaluate_retrieval.Evaluation.score

        def favour_keywords(self, mode, alpha=None, rrf_k=None):  # type: ignore[no-untyped-def]
            means, per_query = real(self, mode, alpha=alpha, rrf_k=rrf_k)
            if alpha == 0.2 and rrf_k == 30:
                means = {**means, "ndcg": means["ndcg"] + 0.5}
            return means, per_query

        monkeypatch.setattr(evaluate_retrieval.Evaluation, "score", favour_keywords)
        assert _run(corpus, tmp_path, "--calibrate") == 0
        out = capsys.readouterr().out
        assert "RAG_HYBRID_ALPHA=0.2" in out and "RAG_RRF_K=30" in out

    def test_rrf_k_is_restored_after_each_run(
        self, corpus: tuple[Path, Path], tmp_path: Path
    ) -> None:
        docs, dataset = corpus
        evaluation = evaluate_retrieval.Evaluation(docs, dataset, 2, tmp_path / "w")
        evaluation.score(evaluate_retrieval.MODES["hybrid"], alpha=0.3, rrf_k=10)
        assert (
            evaluation.pipeline._get_hybrid_retriever("en").rrf_k
            == evaluation.pipeline.config.rrf_k
        )

    def test_unlabelled_questions_are_skipped(
        self, corpus: tuple[Path, Path], tmp_path: Path
    ) -> None:
        docs, dataset = corpus
        evaluation = evaluate_retrieval.Evaluation(docs, dataset, 2, tmp_path / "w")
        assert len(evaluation.examples) == 3
