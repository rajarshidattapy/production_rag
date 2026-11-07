"""Retrieval quality metrics: Recall@K, Precision@K, MRR, and NDCG@K.

Relevance comes from the golden dataset's ``expected_sources``: a retrieved chunk is relevant
when it came from one of those files. That works at document level, so labels survive
re-chunking and re-ingesting, which would change chunk IDs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

METRICS = ("recall", "precision", "mrr", "ndcg", "hit_rate")


@dataclass(frozen=True)
class QueryScore:
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in METRICS}


def score_query(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    relevant_chunks_in_corpus: int,
    k: int,
) -> QueryScore:
    """Score one ranked list of retrieved chunks, given by their source file.

    - **Recall@K:** share of the expected documents that appear in the top K.
    - **Precision@K:** share of the top K chunks that come from an expected document.
    - **MRR:** 1 / rank of the first relevant chunk, or 0 if none is in the top K.
    - **NDCG@K:** binary gains, normalised by the best ranking the corpus allows, which has
      ``min(K, relevant_chunks_in_corpus)`` relevant chunks at the top.
    """
    expected = set(expected_sources)
    top = list(retrieved_sources[:k])
    relevant = [source in expected for source in top]

    recall = len(expected & set(top)) / len(expected) if expected else 0.0
    precision = sum(relevant) / k if k else 0.0
    first = next((rank for rank, hit in enumerate(relevant, start=1) if hit), None)
    mrr = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(relevant, start=1) if hit)
    ideal_hits = min(k, relevant_chunks_in_corpus)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return QueryScore(recall, precision, mrr, ndcg, 1.0 if first else 0.0)


def mean_scores(scores: Sequence[QueryScore]) -> dict[str, float]:
    if not scores:
        return dict.fromkeys(METRICS, 0.0)
    return {name: round(sum(getattr(s, name) for s in scores) / len(scores), 4) for name in METRICS}
