"""Tests for short citation previews and on-demand full passages."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import api as api_app
from src.evaluation.dataset import EvalExample
from src.evaluation.healing_eval import HealingEvaluator
from src.evaluation.runner import EvaluationRunner
from src.generation.citations import (
    SNIPPET_CHARS,
    Citation,
    CitationFormatter,
    shorten_snippet,
)
from tests.test_healing_evaluation import _abstained, _dataset, _recovered

TAIL = "uniquetailphrase"
LONG_TEXT = "Hybrid search merges keyword and vector retrieval. " * 12 + TAIL


def _contexts(text: str = LONG_TEXT) -> list[dict[str, Any]]:
    return [{"id": "c1", "document": text, "metadata": {"source": "a.txt", "filename": "a.txt"}}]


class TestShortenSnippet:
    def test_short_text_is_unchanged(self) -> None:
        assert shorten_snippet("A short passage.") == "A short passage."
        assert shorten_snippet("") == ""

    def test_text_exactly_at_the_limit_is_not_cut(self) -> None:
        text = "x" * SNIPPET_CHARS
        assert shorten_snippet(text) == text

    def test_whitespace_is_collapsed(self) -> None:
        assert shorten_snippet("  line one\n\n  line\ttwo ") == "line one line two"

    def test_cuts_at_a_word_boundary_within_the_limit(self) -> None:
        out = shorten_snippet(LONG_TEXT)
        assert len(out) <= SNIPPET_CHARS
        assert out.endswith("…")
        assert LONG_TEXT.startswith(out[:-1])
        assert LONG_TEXT[len(out) - 1] == " "  # the next character was a word break

    def test_does_not_leave_punctuation_before_the_ellipsis(self) -> None:
        assert shorten_snippet("alpha beta, gamma delta", limit=13) == "alpha beta…"

    @pytest.mark.parametrize(
        "text",
        ["日本語のテキスト" * 50, "x" * 500, "short " + "y" * 500],
    )
    def test_text_without_useful_word_breaks_is_still_cut(self, text: str) -> None:
        out = shorten_snippet(text)
        assert len(out) == SNIPPET_CHARS
        assert out.endswith("…")


class TestCitationFields:
    def test_build_citations_keeps_full_text_apart_from_the_preview(self) -> None:
        citation = CitationFormatter.build_citations(_contexts())[0]
        assert len(citation.text_snippet) <= SNIPPET_CHARS
        assert TAIL not in citation.text_snippet
        assert citation.text == LONG_TEXT

    def test_full_text_is_not_serialized(self) -> None:
        serialized = CitationFormatter.to_dict(CitationFormatter.build_citations(_contexts()))[0]
        assert "text" not in serialized
        assert TAIL not in json.dumps(serialized)

    def test_endnotes_use_the_preview_without_a_second_ellipsis(self) -> None:
        citations = CitationFormatter.build_citations(_contexts())
        out = CitationFormatter.format_answer_with_citations("A", citations, format="endnote")
        assert "…*" in out
        assert "..." not in out
        short = [Citation("c", "s", "f.txt", "Complete sentence.")]
        out = CitationFormatter.format_answer_with_citations("A", short, format="endnote")
        assert "f.txt — *Complete sentence.*" in out


class TestEvaluationSeesWholePassages:
    """Faithfulness judged against a 180-character preview would fail correct answers."""

    def test_evaluation_runner_scores_against_the_full_passage(self) -> None:
        runner = EvaluationRunner()
        runner.pipeline = MagicMock()
        runner.pipeline.query.return_value = (
            "answer",
            CitationFormatter.build_citations(_contexts()),
        )
        for name, field in (
            ("scorer", "faithfulness_score"),
            ("relevance_scorer", "relevance_score"),
            ("context_precision_scorer", "context_precision_score"),
            ("context_recall_scorer", "context_recall_score"),
        ):
            setattr(runner, name, MagicMock(**{"score.return_value": {field: 1.0}}))

        runner._evaluate_single(
            EvalExample(question="q", reference_answer="r"), use_hybrid=False, use_reranker=False
        )

        contexts = runner.scorer.score.call_args[0][1]
        assert contexts[0]["document"] == LONG_TEXT
        assert runner.context_precision_scorer.score.call_args[0][1][0]["document"] == LONG_TEXT

    def test_healing_baseline_scores_against_the_full_passage(self, tmp_path: Path) -> None:
        pipeline = MagicMock()
        pipeline.query.return_value = ("answer", CitationFormatter.build_citations(_contexts()))
        pipeline.query_with_healing.side_effect = [_recovered(), _abstained()]
        faithfulness = MagicMock(**{"score.return_value": {"faithfulness_score": 1.0}})

        evaluator = HealingEvaluator(
            pipeline,
            dataset=_dataset(tmp_path),
            results_dir=tmp_path / "out",
            faithfulness_scorer=faithfulness,
            relevance_scorer=MagicMock(**{"score.return_value": {"relevance_score": 1.0}}),
        )
        evaluator.run()

        assert faithfulness.score.call_args_list[0][0][1][0]["document"] == LONG_TEXT


class TestApi:
    @pytest.fixture(autouse=True)
    def _fresh_pipeline(self) -> Iterator[None]:
        api_app.reset_pipeline()
        yield
        api_app.reset_pipeline()

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        client = TestClient(api_app.app)
        doc = tmp_path / "long.txt"
        doc.write_text(LONG_TEXT, encoding="utf-8")
        assert client.post("/ingest", json={"source": str(doc)}).status_code == 200
        return client

    def _stub_llm(self) -> Any:
        return patch.object(
            api_app.get_pipeline().generator,
            "generate_async",
            AsyncMock(return_value="Stub answer [1]."),
        )

    def test_query_returns_a_preview_and_chunks_returns_the_rest(self, client: TestClient) -> None:
        with self._stub_llm():
            response = client.post("/query", json={"question": "hybrid?", "self_heal": False})
        assert response.status_code == 200
        assert TAIL not in response.text
        citation = response.json()["citations"][0]
        assert len(citation["text_snippet"]) <= SNIPPET_CHARS
        assert citation["text_snippet"].endswith("…")

        passage = client.get(f"/chunks/{citation['chunk_id']}")
        assert passage.status_code == 200
        assert passage.json() == {"chunk_id": citation["chunk_id"], "text": LONG_TEXT}

    def test_generation_still_gets_the_whole_passage(self, client: TestClient) -> None:
        with self._stub_llm() as generate:
            client.post("/query", json={"question": "hybrid?", "self_heal": False})
        assert TAIL in generate.call_args[0][1][0]["document"]

    def test_streaming_citations_are_previews(self, client: TestClient) -> None:
        async def stream(query: str, contexts: list[dict[str, Any]]) -> Any:
            yield "token"

        with patch.object(api_app.get_pipeline().generator, "generate_stream", stream):
            response = client.post(
                "/query/stream", json={"question": "hybrid?", "self_heal": False}
            )
        assert '"citations"' in response.text
        assert TAIL not in response.text

    def test_saved_history_holds_only_the_preview(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "long enough pw"})
        chat_id = client.post("/chats", json={}).json()["id"]
        with self._stub_llm():
            turn = client.post(
                f"/chats/{chat_id}/messages", json={"question": "hybrid?", "self_heal": False}
            )
        assert turn.status_code == 200
        assert TAIL not in turn.text
        store = api_app.get_chat_store()
        with store._connect() as conn:
            saved = conn.execute("SELECT citations FROM messages WHERE role='assistant'").fetchone()
        assert TAIL not in saved[0]
        assert json.loads(saved[0])[0]["text_snippet"].endswith("…")

    def test_imported_citations_keep_the_text_they_came_with(self, client: TestClient) -> None:
        # The browser may hold the only remaining copy of a passage from a removed document.
        client.post("/auth/register", json={"username": "alice", "password": "long enough pw"})
        body = {
            "import_id": "old",
            "title": "Old",
            "created_at": 1.0,
            "messages": [
                {
                    "role": "assistant",
                    "text": "a",
                    "citations": [
                        {
                            "chunk_id": "gone",
                            "source": "s",
                            "filename": "f",
                            "text_snippet": LONG_TEXT,
                            "score": 0.5,
                        }
                    ],
                }
            ],
        }
        client.post("/chats/import", json=body)
        chat_id = client.get("/chats").json()[0]["id"]
        message = client.get(f"/chats/{chat_id}").json()["messages"][0]
        assert message["citations"][0]["text_snippet"] == LONG_TEXT

    def test_unknown_chunk_is_404(self, client: TestClient) -> None:
        response = client.get("/chunks/does-not-exist")
        assert response.status_code == 404
        assert "no longer" in response.json()["detail"]

    def test_chunk_is_found_in_any_language_collection(self, tmp_path: Path) -> None:
        doc = tmp_path / "de.txt"
        doc.write_text("Der Umsatz stieg in Deutschland.", encoding="utf-8")
        client = TestClient(api_app.app)
        with patch("src.pipeline._detect_language", return_value="de"):
            client.post("/ingest", json={"source": str(doc)})
        pipeline = api_app.get_pipeline()
        chunk_id = pipeline._get_vector_store("de").get_all_chunks()[0]["id"]
        assert pipeline._get_vector_store("en").count() == 0
        assert (
            client.get(f"/chunks/{chunk_id}").json()["text"] == "Der Umsatz stieg in Deutschland."
        )

    def test_chunk_is_gone_after_a_reset(self, client: TestClient) -> None:
        chunk_id = api_app.get_pipeline().vector_store.get_all_chunks()[0]["id"]
        api_app.get_pipeline().reset()
        assert client.get(f"/chunks/{chunk_id}").status_code == 404

    def test_chunks_require_the_api_key_when_configured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunk_id = api_app.get_pipeline().vector_store.get_all_chunks()[0]["id"]
        monkeypatch.setattr(api_app, "_RAG_API_KEY", "deployment-key")
        assert client.get(f"/chunks/{chunk_id}").status_code == 401
        keyed = client.get(
            f"/chunks/{chunk_id}", headers={"Authorization": "Bearer deployment-key"}
        )
        assert keyed.status_code == 200
