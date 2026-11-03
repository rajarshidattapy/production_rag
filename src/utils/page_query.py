"""Recognise questions that name specific document pages ("what's on page 3?").

Dense and keyword search can't answer these: a page number is a location, not
something the page's text talks about. This turns such a reference into a Chroma
metadata filter on the ``page`` / ``is_last_page`` fields set by the chunker.

Covers the languages the pipeline routes (en, de, es). Page numbers are physical
positions in the file, not the labels printed on the pages.
"""

from __future__ import annotations

import re
from typing import Any


def _numbered(words: str) -> dict[str, int]:
    return {word: i for i, word in enumerate(words.split(), start=1)}


_NUMBER_WORDS: dict[str, int] = {
    **_numbered("one two three four five six seven eight nine ten"),
    **_numbered("eins zwei drei vier fünf sechs sieben acht neun zehn"),
    **_numbered("uno dos tres cuatro cinco seis siete ocho nueve diez"),
}
_ORDINALS: dict[str, int] = {
    **_numbered("first second third fourth fifth sixth seventh eighth ninth tenth"),
    **_numbered("erste zweite dritte vierte fünfte sechste siebte achte neunte zehnte"),
    **_numbered("primera segunda tercera cuarta quinta sexta séptima octava novena décima"),
    "septima": 7,
    "decima": 10,
}

_NUM = r"\d{1,6}"
_START = r"(?<![\w.])"
_PAGE = r"(?:page|seite|p[aá]gina)"
_PAGES = r"(?:pages|seiten|p[aá]ginas)"
_RANGE_SEP = r"-|–|—|to|through|thru|bis|hasta|a"
_LIST_SEP = r",|&|and|und|y"
_WORDS = "|".join(_NUMBER_WORDS)
_ORDINAL_WORDS = "|".join(sorted(_ORDINALS, key=len, reverse=True))

# "first page load" and friends are web/OS terms, not a location in a document.
_NOT_COMPOUND = (
    r"(?!\s+(?:loads?|loading|views?|visits?|hits?|ranks?|rankings?|results?|faults?|tables?|size))"
)

# page 5, p. 5, pp. 3-5, pages 2, 4 and 6, Seite 5, S. 5, páginas 3 a 5, pág. 5
_NUMBERED_RE = re.compile(
    rf"{_START}(?:pages?|pgs?\.?|pp?\.|seiten?|s\.|p[aá]ginas?|p[aá]gs?\.?)\s*"
    r"(?:number\s+|no\.?\s*|nr\.?\s*|n[uú]mero\s+|#\s*)?"
    rf"(?P<first>{_NUM})\b(?P<tail>(?:\s*(?:{_RANGE_SEP}|{_LIST_SEP})\s*{_NUM}\b)*)",
    re.IGNORECASE,
)
_TAIL_RE = re.compile(rf"\s*({_RANGE_SEP}|{_LIST_SEP})\s*({_NUM})\b", re.IGNORECASE)
_RANGE_SEP_RE = re.compile(rf"^(?:{_RANGE_SEP})$", re.IGNORECASE)

# first 3 pages, ersten drei Seiten, primeras 3 páginas
_FIRST_N_RE = re.compile(
    rf"{_START}(?:first|ersten|primeras)\s+(?P<n>{_NUM}|{_WORDS})\s+{_PAGES}\b", re.IGNORECASE
)
# first page, 3rd page, zweiten Seite, primera página
_ORDINAL_RE = re.compile(
    rf"{_START}(?:(?P<word>{_ORDINAL_WORDS})[nrs]?|(?P<num>{_NUM})(?:st|nd|rd|th))\s+"
    rf"{_PAGE}\b{_NOT_COMPOUND}",
    re.IGNORECASE,
)
# page one, Seite zwei, página tres
_WORD_RE = re.compile(rf"{_START}{_PAGE}\s+(?P<word>{_WORDS})\b{_NOT_COMPOUND}", re.IGNORECASE)
# last page, letzten Seite, última página, página final
_LAST_RE = re.compile(
    rf"{_START}(?<!to\s)(?:(?:last|final|letzte[nrs]?|[uú]ltima)\s+{_PAGE}|p[aá]gina\s+final)\b"
    rf"{_NOT_COMPOUND}",
    re.IGNORECASE,
)

_ALL_PATTERNS = (_NUMBERED_RE, _FIRST_N_RE, _ORDINAL_RE, _WORD_RE, _LAST_RE)


def _span_clause(first: int, last: int) -> dict[str, Any]:
    low, high = min(first, last), max(first, last)
    if low == high:
        return {"page": low}
    return {"$and": [{"page": {"$gte": low}}, {"page": {"$lte": high}}]}


def _numbered_clauses(match: re.Match[str]) -> list[dict[str, Any]]:
    spans = [[int(match["first"])] * 2]
    for separator, number in _TAIL_RE.findall(match["tail"]):
        if _RANGE_SEP_RE.match(separator):
            spans[-1][1] = int(number)
        else:
            spans.append([int(number)] * 2)
    return [_span_clause(*span) for span in spans]


def page_filter(question: str) -> dict[str, Any] | None:
    """Return a Chroma ``where`` filter for the pages ``question`` names, or ``None``."""
    clauses: list[dict[str, Any]] = []
    for match in _NUMBERED_RE.finditer(question):
        clauses.extend(_numbered_clauses(match))
    for match in _FIRST_N_RE.finditer(question):
        n = match["n"].lower()
        clauses.append(_span_clause(1, _NUMBER_WORDS[n] if n in _NUMBER_WORDS else int(n)))
    for match in _ORDINAL_RE.finditer(question):
        page = _ORDINALS[match["word"].lower()] if match["word"] else int(match["num"])
        clauses.append({"page": page})
    for match in _WORD_RE.finditer(question):
        clauses.append({"page": _NUMBER_WORDS[match["word"].lower()]})
    if _LAST_RE.search(question):
        clauses.append({"is_last_page": True})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def strip_page_reference(question: str) -> str:
    """Remove page references, leaving the words that describe what the user wants to know."""
    for pattern in _ALL_PATTERNS:
        question = pattern.sub(" ", question)
    return " ".join(question.split())
