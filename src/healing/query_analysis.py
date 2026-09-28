"""Deterministic query analysis and lexical coverage checks.

These run on every attempt and cost nothing: no model calls. They feed the
failure classifier (e.g. "none of the query's key terms appear in the retrieved
context" is a query/vocabulary problem, not a generation problem).
"""

from __future__ import annotations

import re
from typing import Any

from src.healing.types import QueryAnalysis

# Same token boundary rule as HybridRetriever._tokenize so coverage numbers
# line up with what BM25 actually matches on.
_TOKEN_RE = re.compile(r"[^a-zA-Z0-9À-ɏ]+")

_STOPWORDS: frozenset[str] = frozenset(
    # English
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my of on or our should so than that the their them then there these they "
    "this those to was we were what when where which who whom why will with would you your "
    "about also any each just more most much not only other some such very "
    "explain describe tell give list name show please define "
    # German
    "der die das und ist sind ein eine einer eines wie was wer warum wo welche welcher mit von zu "
    "im in den dem des nicht auch für auf "
    # Spanish
    "el la los las un una unos unas y es son como qué que quien por para con de del en al se "
    "cual cuales cómo".split()
)

_QUESTION_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("comparative", ("differ", "difference", "compare", "versus", " vs ", "better than")),
    ("procedural", ("how do", "how does", "how can", "how to", "steps", "process")),
    ("causal", ("why",)),
    ("definition", ("what is", "what are", "define", "meaning of")),
    ("enumeration", ("list", "name ", "which ", "examples")),
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric boundaries (Unicode Latin aware)."""
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


def content_terms(text: str) -> list[str]:
    """Return de-duplicated, order-preserving non-stopword tokens of length > 1."""
    seen: set[str] = set()
    terms: list[str] = []
    for tok in tokenize(text):
        if tok in _STOPWORDS or len(tok) <= 1 or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
    return terms


def analyze_query(question: str) -> QueryAnalysis:
    """Cheap structural analysis of the query used to steer rewriting and diagnosis."""
    terms = content_terms(question)
    lowered = f" {question.lower()} "
    qtype = "other"
    for name, markers in _QUESTION_TYPES:
        if any(m in lowered for m in markers):
            qtype = name
            break
    return QueryAnalysis(
        original=question,
        content_terms=terms,
        question_type=qtype,
        # A query with fewer than two content terms gives retrieval almost
        # nothing to match on ("tell me about it", "how does that work?").
        is_vague=len(terms) < 2,
    )


def term_coverage(terms: list[str], contexts: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Fraction of ``terms`` present anywhere in the contexts, plus the missing terms.

    Matching is prefix-tolerant in one direction (``rerank`` matches ``reranking``)
    so simple inflections don't count as misses.
    """
    if not terms:
        return 1.0, []
    vocab: set[str] = set()
    for ctx in contexts:
        vocab.update(tokenize(ctx.get("document", "")))
    missing: list[str] = []
    for term in terms:
        if term in vocab:
            continue
        if len(term) >= 4 and any(v.startswith(term) for v in vocab):
            continue
        missing.append(term)
    return (len(terms) - len(missing)) / len(terms), missing


def lexical_overlap(sentence: str, reference: str) -> float:
    """Fraction of the sentence's content terms that also appear in ``reference``."""
    terms = content_terms(sentence)
    if not terms:
        return 1.0
    ref_vocab = set(tokenize(reference))
    hits = sum(
        1
        for t in terms
        if t in ref_vocab or (len(t) >= 4 and any(v.startswith(t[:4]) for v in ref_vocab))
    )
    return hits / len(terms)
