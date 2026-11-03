"""Tests for page-aware PDF ingestion and page-specific retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.generation.citations import Citation, CitationFormatter
from src.generation.generator import Generator
from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.utils.page_query import page_filter, strip_page_reference

PAGE_TEXTS = [
    "Annual report title page for the Northwind company.",
    "Revenue grew because the Northwind company opened new stores.",
    "",  # a scanned or blank page has no extractable text
    "Northwind expects revenue to keep growing next year.",
    "",  # blank back cover
]


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a real, minimal PDF with one line of text per page."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_id, text in zip(page_ids, pages, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_id + 1} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 10 Tf 40 700 Td ({text}) Tj ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def report_pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "report.pdf", PAGE_TEXTS)


@pytest.fixture
def pipeline(report_pdf: Path) -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    p.ingest(report_pdf)
    return p


def _pages(contexts: list[dict[str, Any]]) -> list[int]:
    return [c["metadata"]["page"] for c in contexts]


class TestPageFilterParsing:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is on page 5?", {"page": 5}),
            ("what's on PAGE 12", {"page": 12}),
            ("see p. 7 please", {"page": 7}),
            ("see pg 7 please", {"page": 7}),
            ("page number 9", {"page": 9}),
            ("page #9", {"page": 9}),
            ("Was steht auf Seite 4?", {"page": 4}),
            ("siehe S. 4", {"page": 4}),
            ("¿Qué dice la página 6?", {"page": 6}),
            ("¿Qué dice la pagina 6?", {"page": 6}),
            ("What is on the first page?", {"page": 1}),
            ("summarize the 3rd page", {"page": 3}),
            ("read page one of the report", {"page": 1}),
            ("Was steht auf der zweiten Seite?", {"page": 2}),
            ("Resume la primera página", {"page": 1}),
            ("What does the last page say?", {"is_last_page": True}),
            ("Was steht auf der letzten Seite?", {"is_last_page": True}),
            ("¿Qué hay en la última página?", {"is_last_page": True}),
            (
                "summarize pages 3-5",
                {"$and": [{"page": {"$gte": 3}}, {"page": {"$lte": 5}}]},
            ),
            (
                "summarize pages 3 to 5",
                {"$and": [{"page": {"$gte": 3}}, {"page": {"$lte": 5}}]},
            ),
            (
                "pp. 5-3",  # reversed range
                {"$and": [{"page": {"$gte": 3}}, {"page": {"$lte": 5}}]},
            ),
            (
                "Seiten 3 bis 5",
                {"$and": [{"page": {"$gte": 3}}, {"page": {"$lte": 5}}]},
            ),
            (
                "páginas 3 a 5",
                {"$and": [{"page": {"$gte": 3}}, {"page": {"$lte": 5}}]},
            ),
            (
                "summarize the first 3 pages",
                {"$and": [{"page": {"$gte": 1}}, {"page": {"$lte": 3}}]},
            ),
            (
                "the first two pages",
                {"$and": [{"page": {"$gte": 1}}, {"page": {"$lte": 2}}]},
            ),
            ("pages 2, 4 and 6", {"$or": [{"page": 2}, {"page": 4}, {"page": 6}]}),
            ("compare page 1 and page 3", {"$or": [{"page": 1}, {"page": 3}]}),
            ("the first page and the last page", {"$or": [{"page": 1}, {"is_last_page": True}]}),
        ],
    )
    def test_recognised(self, question: str, expected: dict[str, Any]) -> None:
        assert page_filter(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "What is RAG?",
            "How does the PageRank algorithm work?",
            "Show me the webpage 5 times",
            "how does paging work",
            "What is a page table?",
            "How do I reduce first page load time?",
            "Tell me about the U.S. 5 year plan",
            "Which pages discuss revenue?",
            "On the next page",
            "What is on the second to last page?",
            "Auf der einen Seite",
        ],
    )
    def test_not_recognised(self, question: str) -> None:
        assert page_filter(question) is None

    def test_strip_leaves_the_topic(self) -> None:
        assert strip_page_reference("What does page 3 say about revenue?") == (
            "What does say about revenue?"
        )
        assert "page" not in strip_page_reference("what is on the first page?").lower()


class TestPageAwareIngestion:
    def test_pdf_keeps_each_page_text(self, report_pdf: Path) -> None:
        doc = DocumentLoader().load(report_pdf)[0]
        assert len(doc.pages) == 5
        assert "title page" in doc.pages[0]
        assert doc.pages[2] == ""
        assert doc.metadata["pages"] == 5

    def test_chunks_record_their_page_and_skip_blank_pages(self, report_pdf: Path) -> None:
        doc = DocumentLoader().load(report_pdf)[0]
        chunks = Chunker(chunk_size=200, chunk_overlap=20).chunk(doc)
        assert [c.metadata["page"] for c in chunks] == [1, 2, 4]
        assert "Revenue grew" in chunks[1].content
        # The blank back cover can't answer anything, so page 4 counts as the last page.
        assert [c.metadata["is_last_page"] for c in chunks] == [False, False, True]

    def test_chunk_never_mixes_pages(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "short.pdf", ["alpha page one", "bravo page two"])
        doc = DocumentLoader().load(pdf)[0]
        chunks = Chunker(chunk_size=800, chunk_overlap=100).chunk(doc)
        assert [c.content for c in chunks] == ["alpha page one", "bravo page two"]

    def test_every_strategy_is_page_aware(self, report_pdf: Path) -> None:
        doc = DocumentLoader().load(report_pdf)[0]
        for strategy in ("recursive", "fixed", "sentence"):
            chunks = Chunker(chunk_size=200, chunk_overlap=20, strategy=strategy).chunk(doc)
            assert {c.metadata["page"] for c in chunks} == {1, 2, 4}, strategy

    def test_text_files_have_no_page_metadata(self, sample_document: Any, chunker: Chunker) -> None:
        for chunk in chunker.chunk(sample_document):
            assert "page" not in chunk.metadata
            assert "is_last_page" not in chunk.metadata

    def test_chunk_ids_stay_unique_across_pages(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "dup.pdf", ["same text", "same text"])
        chunks = Chunker().chunk(DocumentLoader().load(pdf)[0])
        assert len({c.chunk_id for c in chunks}) == 2


class TestFilteredSearch:
    @pytest.fixture
    def store(self, report_pdf: Path, temp_chroma_dir: Path) -> VectorStore:
        store = VectorStore(collection_name="paged", persist_path=temp_chroma_dir)
        store.add_chunks(Chunker().chunk(DocumentLoader().load(report_pdf)[0]))
        return store

    def test_page_is_stored_as_a_number(self, store: VectorStore) -> None:
        pages = sorted(c["metadata"]["page"] for c in store.get_all_chunks())
        assert pages == [1, 2, 4]

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("page 2", [2]),
            ("pages 2-4", [2, 4]),
            ("pages 1 and 4", [1, 4]),
            ("the last page", [4]),
            ("page 3", []),
            ("page 99", []),
        ],
    )
    def test_vector_search_respects_page_filter(
        self, store: VectorStore, question: str, expected: list[int]
    ) -> None:
        hits = store.similarity_search("revenue", k=10, where=page_filter(question))
        assert sorted(_pages(hits)) == expected

    def test_ids_matching_pages_through_results(self, store: VectorStore) -> None:
        everything = {c["id"] for c in store.get_all_chunks()}
        assert store.ids_matching({"page": {"$gte": 1}}, page_size=1) == everything
        assert store.ids_matching({"page": 99}) == set()

    def test_has_paged_chunks(self, store: VectorStore, temp_chroma_dir: Path) -> None:
        assert store.has_paged_chunks() is True
        other = VectorStore(collection_name="empty", persist_path=temp_chroma_dir)
        assert other.has_paged_chunks() is False

    def test_hybrid_keyword_side_respects_the_filter(self, store: VectorStore) -> None:
        hybrid = HybridRetriever(store, alpha=0.5)
        hybrid.build_index()
        # "stores" is a strong keyword match, but it only occurs on page 2.
        assert _pages(hybrid._bm25_search("stores", k=5)) == [2]
        assert hybrid._bm25_search("stores", k=5, where={"page": 4}) == []
        assert set(_pages(hybrid.search("stores", k=5, where={"page": 4}))) == {4}


class TestPageSpecificPipeline:
    def _answer(self, question: str, contexts: list[dict[str, Any]], **_: Any) -> str:
        return f"answer from pages {_pages(contexts)} [1]."

    @pytest.mark.parametrize("use_hybrid", [False, True])
    def test_first_page_question_answers_from_page_one(
        self, pipeline: RAGPipeline, use_hybrid: bool
    ) -> None:
        with patch.object(pipeline.generator, "generate", side_effect=self._answer) as generate:
            answer, citations = pipeline.query(
                "What is on the first page?", use_hybrid=use_hybrid, self_heal=False
            )
        assert answer == "answer from pages [1] [1]."
        generate.assert_called_once()
        assert [c.page for c in citations] == [1]
        assert "title page" in citations[0].text_snippet

    def test_whole_page_comes_back_in_reading_order_despite_top_k(self, tmp_path: Path) -> None:
        text = "\n\n".join(f"Paragraph {i} " + "filler words " * 60 for i in range(8))
        pdf = tmp_path / "dense.pdf"
        write_pdf(pdf, ["cover"])
        p = RAGPipeline()
        p.reset()
        # Pages come from a Document directly because the helper writes single-line pages.
        doc = DocumentLoader().load(pdf)[0]
        doc.pages = [text, "other page"]
        with patch.object(p.loader, "load", return_value=[doc]):
            p.ingest(pdf)

        with patch.object(p.generator, "generate", side_effect=self._answer) as generate:
            _, citations = p.query("What is on page 1?", top_k=2, self_heal=False)

        contexts = generate.call_args[0][1]
        assert len(contexts) > 2
        assert {c.page for c in citations} == {1}
        indexes = [c["metadata"]["chunk_index"] for c in contexts]
        assert indexes == sorted(indexes)

    def test_page_with_no_text_does_not_fall_back_to_other_pages(
        self, pipeline: RAGPipeline
    ) -> None:
        with patch.object(pipeline.generator, "generate") as generate:
            answer, citations = pipeline.query("What is on page 3?", self_heal=False)
        generate.assert_not_called()
        assert citations == []
        assert "could not find" in answer

    def test_text_only_corpus_ignores_page_references(self, sample_docs_dir: Path) -> None:
        p = RAGPipeline()
        p.reset()
        p.ingest(sample_docs_dir)
        with patch.object(p.generator, "generate", return_value="ok [1].") as generate:
            answer, citations = p.query("What is on page 1 about RAG?", self_heal=False)
        generate.assert_called_once()
        assert answer == "ok [1]."
        assert citations and all(c.page is None for c in citations)

    def test_page_spanning_language_collections_is_found(self, report_pdf: Path) -> None:
        """Language routing is per chunk, so one page can live in several collections."""
        p = RAGPipeline()
        p.reset()
        routed = iter(["en", "de", "es"])
        with patch("src.pipeline._detect_language", side_effect=lambda _text: next(routed)):
            p.ingest(report_pdf)

        for lang in ("en", "de", "es"):
            assert p._get_vector_store(lang).count() == 1
        contexts = p._retrieve("pages", k=20, lang="en", where={"page": {"$gte": 1}})
        assert _pages(contexts) == [1, 2, 4]

    def test_question_language_does_not_pick_the_page_collection(self, report_pdf: Path) -> None:
        p = RAGPipeline()
        p.reset()
        with patch("src.pipeline._detect_language", return_value="de"):
            p.ingest(report_pdf)
        assert p._get_vector_store("en").count() == 0

        with (
            patch("src.pipeline._detect_language", return_value="en"),
            patch.object(p.generator, "generate", return_value="ok [1]."),
        ):
            _, citations = p.query("What is on page 2?", self_heal=False)
        assert [c.page for c in citations] == [2]

    def test_page_question_with_reranker(self, pipeline: RAGPipeline) -> None:
        with (
            patch.object(pipeline, "_apply_reranker", side_effect=lambda q, c, top_k: c[:top_k]),
            patch.object(pipeline.generator, "generate", return_value="ok [1]."),
        ):
            _, citations = pipeline.query("page 4", use_reranker=True, self_heal=False)
        assert [c.page for c in citations] == [4]

    @pytest.mark.asyncio
    async def test_async_query_is_page_aware(self, pipeline: RAGPipeline) -> None:
        async def generate(query: str, contexts: list[dict[str, Any]]) -> str:
            return f"pages {_pages(contexts)}"

        with patch.object(pipeline.generator, "generate_async", side_effect=generate):
            answer, citations = await pipeline.query_async("What is on page 2?", self_heal=False)
        assert answer == "pages [2]"
        assert [c.page for c in citations] == [2]

    @pytest.mark.asyncio
    async def test_stream_answer_is_page_aware(self, pipeline: RAGPipeline) -> None:
        async def generate_stream(query: str, contexts: list[dict[str, Any]]):
            yield f"pages {_pages(contexts)}"

        with patch.object(pipeline.generator, "generate_stream", side_effect=generate_stream):
            events = [e async for e in pipeline.stream_answer("What is on page 4?")]

        assert ("token", "pages [4]") in events
        assert [c.page for c in dict(events)["citations"]] == [4]


class TestHealingWithPages:
    QUESTION = "What is on page 2?"

    def _configure(self, pipeline: RAGPipeline, **overrides: Any) -> RAGPipeline:
        pipeline.config = pipeline.config.model_copy(
            update={"verifier_enabled": False, "healing_llm_rewrite": False, **overrides}
        )
        pipeline._healing_engine = None
        return pipeline

    def test_page_question_passes_without_a_lexical_or_score_match(
        self, pipeline: RAGPipeline
    ) -> None:
        # A floor no similarity score can reach, and a query whose words aren't on the page.
        p = self._configure(pipeline, min_retrieval_score=1.5)
        with patch.object(
            p.generator, "generate", return_value="Revenue grew at the Northwind stores [1]."
        ):
            result = p.query_with_healing(self.QUESTION)
        assert result.info.status == "passed"
        assert result.info.attempts == 1
        assert [c.page for c in result.citations] == [2]

    def test_filter_survives_rewrites_and_retries(self, pipeline: RAGPipeline) -> None:
        p = self._configure(pipeline)
        answers = iter(["no citation here", "still no citation", "Revenue grew stores [1]."])
        seen: list[dict[str, Any]] = []

        original = p._retrieve

        def spying_retrieve(query: str, **kwargs: Any) -> list[dict[str, Any]]:
            seen.append(kwargs)
            return original(query, **kwargs)

        with (
            patch.object(p, "_retrieve", side_effect=spying_retrieve),
            patch.object(p.generator, "generate", side_effect=lambda *a, **k: next(answers)),
        ):
            result = p.query_with_healing(self.QUESTION)

        assert result.info.attempts >= 2
        assert len(seen) >= 2  # a repair really did retrieve again
        assert all(call["where"] == {"page": 2} for call in seen)
        assert {c.page for c in result.citations} == {2}

    def test_page_without_text_abstains_immediately(self, pipeline: RAGPipeline) -> None:
        p = self._configure(pipeline)
        with patch.object(p.generator, "generate") as generate:
            result = p.query_with_healing("What is on page 3?")
        generate.assert_not_called()
        assert result.info.status == "abstained"
        assert result.info.reason == "no_text_on_page"
        assert result.info.attempts == 1

    def test_widening_never_shrinks_the_page_scope(self) -> None:
        policy = AdaptiveRetrievalPolicy(max_final_k=12)
        plan = policy.initial_plan("q", 20, 20, False, False, 100, where={"page": 1})
        widened = policy.retrieval_retry_plan(plan, _retrieval_failure(), "q", ["r"], 1)
        assert widened.final_k == 20
        assert widened.where == {"page": 1}
        assert policy.context_expansion_plan(plan).where == {"page": 1}
        broad = policy.retrieval_retry_plan(plan, _retrieval_failure(), "q", ["r", "s"], 2)
        assert broad.where == {"page": 1}


def _retrieval_failure() -> Any:
    from src.healing.types import FailureType

    return FailureType.RETRIEVAL_FAILURE


def test_monitored_pipeline_forwards_the_page_filter(pipeline: RAGPipeline) -> None:
    from src.monitoring import MonitoredRAGPipeline, Tracer

    monitored = MonitoredRAGPipeline(pipeline, tracer=Tracer(enabled=False))
    with patch.object(pipeline.generator, "_client") as client:
        client.complete.return_value = "ok [1]."
        _, citations = monitored.query("What is on page 2?", self_heal=False)
    assert [c.page for c in citations] == [2]


class TestPageLocators:
    def test_generator_tells_the_llm_which_page_a_passage_is_from(self) -> None:
        text = Generator._format_context(
            [
                {"document": "a", "metadata": {"source": "r.pdf", "page": 3}},
                {"document": "b", "metadata": {"source": "n.txt"}},
            ]
        )
        assert "[1] (Source: r.pdf, page 3)" in text
        assert "[2] (Source: n.txt)" in text

    def test_citations_carry_the_page(self) -> None:
        citations = CitationFormatter.build_citations(
            [
                {"id": "a", "document": "x", "metadata": {"filename": "r.pdf", "page": 3}},
                {"id": "b", "document": "y", "metadata": {"filename": "n.txt"}},
            ]
        )
        assert [c.page for c in citations] == [3, None]
        as_dicts = CitationFormatter.to_dict(citations)
        assert as_dicts[0]["page"] == 3
        assert "page" not in as_dicts[1]

    def test_formatted_answer_names_the_page(self) -> None:
        citations = [
            Citation("a", "r.pdf", "r.pdf", "x", page=3),
            Citation("b", "n.txt", "n.txt", "y"),
        ]
        out = CitationFormatter.format_answer_with_citations("A", citations)
        assert "[1] r.pdf, p. 3" in out
        assert "[2] n.txt\n" in out
        endnote = CitationFormatter.format_answer_with_citations("A", citations, format="endnote")
        assert "1. r.pdf, p. 3 — " in endnote


class TestPageApi:
    @pytest.fixture
    def client(self, report_pdf: Path, sample_docs_dir: Path) -> Any:
        import src.api as api

        api.reset_pipeline()
        with TestClient(api.app) as client:
            yield client
        api.reset_pipeline()

    def test_query_returns_page_only_for_paged_sources(
        self, client: TestClient, report_pdf: Path, sample_docs_dir: Path
    ) -> None:
        import src.api as api

        assert client.post("/ingest", json={"source": str(report_pdf), "reset": True}).is_success
        with patch.object(
            api.get_pipeline().generator, "generate_async", AsyncMock(return_value="ok [1].")
        ):
            body = client.post(
                "/query", json={"question": "What is on page 2?", "self_heal": False}
            ).json()
        assert [c["page"] for c in body["citations"]] == [2]

        assert client.post(
            "/ingest", json={"source": str(sample_docs_dir), "reset": True}
        ).is_success
        with patch.object(
            api.get_pipeline().generator, "generate_async", AsyncMock(return_value="ok [1].")
        ):
            body = client.post("/query", json={"question": "What is RAG?", "self_heal": False})
        citations = body.json()["citations"]
        assert citations and all("page" not in c for c in citations)
