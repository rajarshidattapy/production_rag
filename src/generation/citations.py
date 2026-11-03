"""Source citation formatting and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SNIPPET_CHARS = 180


def shorten_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and cut at a word boundary, marking any cut with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    last_space = cut.rfind(" ")
    # Text with few spaces (CJK, long URLs) has no useful word boundary to cut at.
    if last_space >= limit // 2:
        cut = cut[:last_space]
    return cut.rstrip(" ,;:") + "…"


@dataclass
class Citation:
    """A single source citation with locator information.

    ``text_snippet`` is a short preview for display. ``text`` is the whole passage, for
    in-process consumers such as evaluation; it is not serialized into API responses, where
    ``GET /chunks/{chunk_id}`` serves it on demand.
    """

    chunk_id: str
    source: str
    filename: str
    text_snippet: str
    score: float = 0.0
    page: int | None = None
    text: str = ""


class CitationFormatter:
    """Formats source citations alongside LLM-generated answers."""

    @staticmethod
    def build_citations(results: list[dict[str, Any]]) -> list[Citation]:
        """Build Citation objects from retrieval results."""
        citations: list[Citation] = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            score = r.get("rerank_score")
            if score is None:
                score = r.get("score")
            if score is None:
                score = 0.0
            document = r.get("document", "")
            citations.append(
                Citation(
                    chunk_id=r.get("id", f"chunk-{i}"),
                    source=meta.get("source", str(meta)),
                    filename=meta.get("filename", "unknown"),
                    text_snippet=shorten_snippet(document),
                    score=score,
                    page=meta.get("page"),
                    text=document,
                )
            )
        return citations

    @staticmethod
    def format_answer_with_citations(
        answer: str,
        citations: list[Citation],
        format: str = "inline",
    ) -> str:
        """Format an answer with source citations appended.

        Args:
            answer: The LLM-generated answer text.
            citations: List of Citation objects.
            format: "inline" (bracketed numbers) or "endnote" (numbered list at bottom).

        Returns:
            Formatted answer string with citations.
        """
        result = answer.strip()

        if format == "endnote":
            if citations:
                result += "\n\n---\n**Sources:**\n"
                for i, c in enumerate(citations, start=1):
                    result += f"{i}. {CitationFormatter._locator(c)} — *{c.text_snippet}*\n"
        elif format == "inline":
            if citations:
                result += "\n\n**Citations:**\n"
                for i, c in enumerate(citations, start=1):
                    result += f"[{i}] {CitationFormatter._locator(c)}\n"

        return result

    @staticmethod
    def _locator(citation: Citation) -> str:
        if citation.page is None:
            return citation.filename
        return f"{citation.filename}, p. {citation.page}"

    @staticmethod
    def to_dict(citations: list[Citation]) -> list[dict[str, Any]]:
        """Serialize citations to a JSON-compatible dict list.

        ``page`` appears only for sources that have pages.
        """
        return [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "filename": c.filename,
                "text_snippet": c.text_snippet,
                "score": c.score,
                **({} if c.page is None else {"page": c.page}),
            }
            for c in citations
        ]
