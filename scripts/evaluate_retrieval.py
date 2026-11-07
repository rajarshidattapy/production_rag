#!/usr/bin/env python3
"""Retrieval regression testing and search-weight calibration. No LLM is called.

Indexes a corpus into a throwaway store, runs every golden-dataset question, and scores the
ranked chunks with Recall@K, Precision@K, MRR, and NDCG@K (see ``src/evaluation/retrieval_metrics``).

    # Check for regressions against the saved baseline (exits 1 on a regression)
    python -m scripts.evaluate_retrieval

    # Save the current scores as the new baseline, after a deliberate change
    python -m scripts.evaluate_retrieval --update-baseline

    # Tune RAG_HYBRID_ALPHA and RAG_RRF_K on the dataset
    python -m scripts.evaluate_retrieval --calibrate

Point ``--dataset`` and ``--corpus`` at a representative sample of your own documents: the
bundled sample is too small for calibration to find a meaningful difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.config import settings
from src.evaluation.dataset import GoldenDataset
from src.evaluation.retrieval_metrics import METRICS, QueryScore, mean_scores, score_query

DEFAULT_BASELINE = Path("data/golden_dataset/retrieval_baseline.json")
MODES = {
    "vector": {"use_hybrid": False, "use_reranker": False},
    "hybrid": {"use_hybrid": True, "use_reranker": False},
    "hybrid_reranked": {"use_hybrid": True, "use_reranker": True},
}
ALPHAS = [round(a / 10, 1) for a in range(11)]
RRF_KS = [10, 30, 60, 100]


def _fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


class Evaluation:
    """One indexed corpus, scored under different retrieval settings."""

    def __init__(self, corpus: Path, dataset_path: Path, k: int, workdir: Path) -> None:
        from src.pipeline import RAGPipeline, _detect_language

        self.k = k
        self.examples = [e for e in GoldenDataset(dataset_path).load() if e.expected_sources]
        if not self.examples:
            raise SystemExit(f"No questions with expected_sources in {dataset_path}")
        self._detect = _detect_language
        self.pipeline = RAGPipeline()
        # Its own store, always embedded and empty, so results never depend on what's indexed.
        self.pipeline.config = settings.model_copy(
            update={
                "chroma_path": workdir / "chroma",
                "chroma_host": None,
                "chroma_port": None,
                "data_dir": workdir,
            }
        )
        self.pipeline.reset()
        self.pipeline.ingest(corpus)
        self.chunks_per_source: Counter[str] = Counter()
        for lang in ("en", "de", "es"):
            for chunk in self.pipeline._get_vector_store(lang).get_all_chunks():
                self.chunks_per_source[chunk["metadata"]["filename"]] += 1
        files = [p for p in corpus.rglob("*") if p.is_file()] if corpus.is_dir() else [corpus]
        self.identity = {
            "dataset": _fingerprint([dataset_path]),
            "corpus": _fingerprint(files),
            "embedding_model": self.pipeline.config.embedding_model,
            "chunk_size": self.pipeline.config.chunk_size,
            "k": k,
        }

    def ranked_sources(
        self, question: str, mode: dict[str, bool], alpha: float | None
    ) -> list[str]:
        lang = self._detect(question)
        contexts = self.pipeline._retrieve(question, k=self.k, lang=lang, alpha=alpha, **mode)
        if mode["use_reranker"]:
            contexts = self.pipeline._apply_reranker(question, contexts, top_k=self.k)
        return [str(c["metadata"].get("filename", "")) for c in contexts]

    def score(
        self, mode: dict[str, bool], alpha: float | None = None, rrf_k: int | None = None
    ) -> tuple[dict[str, float], list[QueryScore]]:
        if rrf_k is not None:
            for lang in ("en", "de", "es"):
                self.pipeline._get_hybrid_retriever(lang).rrf_k = rrf_k
        scores = []
        for example in self.examples:
            sources = self.ranked_sources(example.question, mode, alpha)
            relevant = sum(self.chunks_per_source[s] for s in example.expected_sources)
            scores.append(score_query(sources, example.expected_sources, relevant, self.k))
        if rrf_k is not None:
            for lang in ("en", "de", "es"):
                self.pipeline._get_hybrid_retriever(lang).rrf_k = self.pipeline.config.rrf_k
        return mean_scores(scores), scores


def regression(args: argparse.Namespace, evaluation: Evaluation) -> int:
    modes = [m for m in args.modes.split(",") if m]
    results = {name: evaluation.score(MODES[name])[0] for name in modes}
    report: dict[str, Any] = {"identity": evaluation.identity, "results": results}
    print(json.dumps(report, indent=2))

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved the baseline to {baseline_path}")
        return 0
    if not baseline_path.exists():
        print(
            f"No baseline at {baseline_path}. Create one with --update-baseline.", file=sys.stderr
        )
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("identity") != evaluation.identity:
        changed = sorted(
            key
            for key in {*baseline.get("identity", {}), *evaluation.identity}
            if baseline.get("identity", {}).get(key) != evaluation.identity.get(key)
        )
        print(
            f"The baseline was measured on different inputs ({', '.join(changed)}), so the scores "
            "can't be compared. If that change is intended, run --update-baseline.",
            file=sys.stderr,
        )
        return 2

    problems = []
    for mode, scores in results.items():
        before = baseline["results"].get(mode)
        if before is None:
            continue
        for metric in METRICS:
            drop = before[metric] - scores[metric]
            if drop > args.tolerance:
                problems.append(
                    f"{mode} {metric}@{args.k}: {before[metric]:.3f} -> {scores[metric]:.3f} "
                    f"(-{drop:.3f}, tolerance {args.tolerance})"
                )
    if problems:
        print("Retrieval quality regressed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"No retrieval regression beyond {args.tolerance} on any metric.")
    return 0


def calibrate(args: argparse.Namespace, evaluation: Evaluation) -> int:
    mode = MODES["hybrid_reranked" if args.reranker else "hybrid"]
    metric = args.metric
    config = evaluation.pipeline.config
    default = (config.hybrid_alpha, config.rrf_k)
    grid: dict[tuple[float, int], tuple[dict[str, float], list[QueryScore]]] = {}
    for rrf_k in RRF_KS:
        for alpha in ALPHAS:
            grid[(alpha, rrf_k)] = evaluation.score(mode, alpha=alpha, rrf_k=rrf_k)
    if default not in grid:
        grid[default] = evaluation.score(mode, alpha=default[0], rrf_k=default[1])

    print(f"{metric}@{args.k} by alpha (rows) and rrf_k (columns); 1.0 alpha = pure vector")
    print("alpha  " + "".join(f"{k:>8}" for k in RRF_KS))
    for alpha in ALPHAS:
        print(f"{alpha:5.1f}  " + "".join(f"{grid[(alpha, k)][0][metric]:8.3f}" for k in RRF_KS))

    default_score = grid[default][0][metric]
    best_value = max(scores[metric] for scores, _ in grid.values())
    # Among equally good settings, the one nearest the current default changes least.
    best = min(
        (s for s, (scores, _) in grid.items() if scores[metric] == best_value),
        key=lambda s: (abs(s[0] - default[0]), abs(s[1] - default[1])),
    )
    gain = best_value - default_score
    wins = losses = 0
    for new, old in zip(grid[best][1], grid[default][1], strict=True):
        delta = getattr(new, metric) - getattr(old, metric)
        wins += delta > 1e-9
        losses += delta < -1e-9
    print(
        f"\ndefault alpha={default[0]} rrf_k={default[1]}: {metric}={default_score:.4f}\n"
        f"best    alpha={best[0]} rrf_k={best[1]}: {metric}={best_value:.4f} "
        f"(+{gain:.4f}; better on {wins} questions, worse on {losses}, of {len(evaluation.examples)})"
    )
    if gain < args.min_gain or best == default:
        print(
            f"Keep the current settings: no setting beats them by at least {args.min_gain}. "
            "A larger, representative dataset may show a difference."
        )
        return 0
    print(f"\nSuggested settings for .env:\n  RAG_HYBRID_ALPHA={best[0]}\n  RAG_RRF_K={best[1]}")
    if losses:
        print(f"Check the {losses} questions that got worse before switching.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=str(settings.golden_dataset_path))
    parser.add_argument("--corpus", default="data/sample_docs")
    parser.add_argument("--k", type=int, default=settings.top_k_final)
    parser.add_argument(
        "--modes", default="vector,hybrid", help=f"Comma-separated: {', '.join(MODES)}"
    )
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--tolerance", type=float, default=0.02, help="Largest allowed drop per metric"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Sweep RAG_HYBRID_ALPHA and RAG_RRF_K"
    )
    parser.add_argument(
        "--metric", choices=METRICS, default="ndcg", help="What calibration maximises"
    )
    parser.add_argument("--min-gain", type=float, default=0.01, help="Smallest gain worth a change")
    parser.add_argument("--reranker", action="store_true", help="Calibrate with the reranker on")
    args = parser.parse_args(argv)

    # Chroma may still hold its files open at exit, which Windows won't let be deleted.
    with tempfile.TemporaryDirectory(
        prefix="rag_retrieval_eval_", ignore_cleanup_errors=True
    ) as workdir:
        evaluation = Evaluation(Path(args.corpus), Path(args.dataset), args.k, Path(workdir))
        try:
            return calibrate(args, evaluation) if args.calibrate else regression(args, evaluation)
        finally:
            evaluation.pipeline.reset()


if __name__ == "__main__":
    sys.exit(main())
