# Evaluation

Evaluation turns “it feels better” into a repeatable quality gate.

## What is measured

- Answer faithfulness and relevance against a golden dataset.
- Retrieval Recall@K, Precision@K, MRR, NDCG, and hit rate.
- Baseline versus self-healed outcomes, including abstentions and repair attempts.

## Run it

```bash
python scripts/evaluate.py --create-sample-dataset
python scripts/evaluate.py --fail-on-threshold
python scripts/evaluate.py --healing-report --hybrid
python scripts/evaluate_retrieval.py --modes vector,hybrid,hybrid_reranked
```

## Why this is compelling

1. A golden dataset makes quality regressions observable before release.
2. Retrieval metrics separate “the right evidence was not found” from “the model answered poorly.”
3. Faithfulness scoring checks whether answers stay grounded in the cited passages.
4. Healing reports quantify whether retries improve outcomes rather than merely adding latency and cost.
5. Threshold-based commands can fail CI, making quality a release criterion instead of a dashboard afterthought.

Keep representative questions, expected answers, and expected sources under `data/golden_dataset/`. Update a baseline only after reviewing an intentional quality change.
