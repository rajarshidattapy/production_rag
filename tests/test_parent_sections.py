"""Parent sections recorded at ingestion, for parent-child retrieval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion import chunker as chunker_module
from src.ingestion.chunker import PARENT_MAX_CHARS, Chunker
from src.ingestion.loader import Document, DocumentLoader
from src.pipeline import RAGPipeline
from src.retrieval.vector_store import VectorStore, _ChromaEmbeddingFunction
from tests.test_page_retrieval import write_pdf

MARKDOWN = """Intro before any heading.

# Guide

Overview of the guide.

## Install

Run the installer.

```bash
# not a heading, just a comment
pip install thing
```

## Configure

Set the options.

# Appendix

Extra notes.
"""


def _squash(text: str) -> str:
    return "".join(text.split())


def _md(text: str = MARKDOWN) -> Document:
    return Document(content=text, metadata={"source": "g.md", "filetype": ".md"}, doc_id="g.md")


class TestSections:
    def test_markdown_headings_become_parents_with_paths(self) -> None:
        parents, chunks = Chunker(chunk_size=800, chunk_overlap=100).chunk_with_parents(_md())
        assert [p.metadata.get("section") for p in parents] == [
            None,
            "Guide",
            "Guide > Install",
            "Guide > Configure",
            "Appendix",
        ]
        assert [c.metadata.get("section") for c in chunks] == [
            None,
            "Guide",
            "Guide > Install",
            "Guide > Configure",
            "Appendix",
        ]

    def test_hash_lines_in_code_blocks_are_not_headings(self) -> None:
        parents, _ = Chunker().chunk_with_parents(_md())
        install = next(p for p in parents if p.metadata.get("section") == "Guide > Install")
        assert "# not a heading" in install.content
        assert "not a heading" not in str([p.metadata.get("section") for p in parents])

    def test_every_chunk_points_at_the_parent_that_contains_it(self) -> None:
        text = "\n\n".join(f"Paragraph {i}. " + "words here " * 40 for i in range(30))
        doc = Document(content=text, metadata={"source": "a.txt"}, doc_id="a.txt")
        parents, chunks = Chunker(chunk_size=500, chunk_overlap=80).chunk_with_parents(doc)
        by_id = {p.chunk_id: p for p in parents}
        assert len(parents) > 1
        for chunk in chunks:
            parent = by_id[chunk.metadata["parent_id"]]
            assert _squash(chunk.content) in _squash(parent.content)

    def test_parents_stay_within_the_size_limit_and_lose_no_text(self) -> None:
        text = "\n\n".join(f"Paragraph {i}. " + "words here " * 40 for i in range(30))
        text += "\n\n" + "x" * (PARENT_MAX_CHARS * 2 + 10)  # one paragraph with no breaks
        doc = Document(content=text, metadata={"source": "a.txt"}, doc_id="a.txt")
        parents, _ = Chunker().chunk_with_parents(doc)
        assert all(len(p.content) <= PARENT_MAX_CHARS for p in parents)
        assert _squash("".join(p.content for p in parents)) == _squash(text)

    def test_short_text_is_one_parent(self, sample_document: Document) -> None:
        parents, chunks = Chunker(chunk_size=200, chunk_overlap=20).chunk_with_parents(
            sample_document
        )
        assert len(parents) == 1
        assert {c.metadata["parent_id"] for c in chunks} == {parents[0].chunk_id}

    def test_pdf_pages_are_parents(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "r.pdf", ["Page one text.", "", "Page three text."])
        parents, chunks = Chunker().chunk_with_parents(DocumentLoader().load(pdf)[0])
        assert [p.metadata["page"] for p in parents] == [1, 3]  # no parent for a blank page
        assert [c.metadata["page"] for c in chunks] == [1, 3]
        assert [c.metadata["is_last_page"] for c in chunks] == [False, True]

    def test_ids_are_stable_across_runs(self) -> None:
        first = Chunker().chunk_with_parents(_md())
        second = Chunker().chunk_with_parents(_md())
        assert [p.chunk_id for p in first[0]] == [p.chunk_id for p in second[0]]
        assert [c.chunk_id for c in first[1]] == [c.chunk_id for c in second[1]]

    def test_parents_carry_the_document_metadata(self) -> None:
        doc = _md()
        doc.metadata["owner"] = "user-1"
        parents, _ = Chunker().chunk_with_parents(doc)
        assert all(p.metadata["owner"] == "user-1" for p in parents)
        assert [p.metadata["parent_index"] for p in parents] == list(range(len(parents)))

    def test_chunk_still_returns_only_chunks(self) -> None:
        parents, chunks = Chunker().chunk_with_parents(_md())
        assert [c.chunk_id for c in Chunker().chunk(_md())] == [c.chunk_id for c in chunks]


class TestStoredParents:
    @pytest.fixture
    def pipeline(self, tmp_path: Path) -> RAGPipeline:
        p = RAGPipeline()
        p.reset()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text(MARKDOWN, encoding="utf-8")
        write_pdf(tmp_path / "docs" / "r.pdf", ["Revenue grew in stores.", "Margins improved."])
        p.ingest(tmp_path / "docs", owner="user-1")
        return p

    def _all_chunks(self, p: RAGPipeline) -> list[dict]:
        return [
            c for lang in ("en", "de", "es") for c in p._get_vector_store(lang).get_all_chunks()
        ]

    def test_every_stored_chunks_parent_can_be_fetched(self, pipeline: RAGPipeline) -> None:
        chunks = self._all_chunks(pipeline)
        parent_ids = sorted({c["metadata"]["parent_id"] for c in chunks})
        parents = pipeline.get_parents(parent_ids)
        assert [p["id"] for p in parents] == parent_ids
        by_id = {p["id"]: p for p in parents}
        for chunk in chunks:
            parent = by_id[chunk["metadata"]["parent_id"]]
            assert _squash(chunk["document"]) in _squash(parent["document"])
            assert parent["metadata"]["owner"] == "user-1"
        assert {p["metadata"].get("page") for p in parents} >= {1, 2}

    def test_parents_are_not_searchable_chunks(self, pipeline: RAGPipeline) -> None:
        stored = pipeline.stats()["chunks_in_store"]
        assert stored == len(self._all_chunks(pipeline))
        hits = pipeline._retrieve("Run the installer", k=20)
        parent_ids = {c["metadata"]["parent_id"] for c in self._all_chunks(pipeline)}
        assert not {h["id"] for h in hits} & parent_ids

    def test_parents_are_not_embedded(self, tmp_path: Path) -> None:
        store = VectorStore(collection_name="parents_only", persist_path=tmp_path / "s")
        parents, _ = Chunker().chunk_with_parents(_md())
        with patch.object(_ChromaEmbeddingFunction, "_encode", side_effect=AssertionError):
            assert store.add_unsearchable(parents) == len(parents)
        assert len(store.get_chunks([p.chunk_id for p in parents])) == len(parents)

    def test_running_again_stores_no_parent_twice(
        self, pipeline: RAGPipeline, tmp_path: Path
    ) -> None:
        before = pipeline._get_parent_store().count()
        with patch.object(
            VectorStore, "add_unsearchable", wraps=VectorStore.add_unsearchable, autospec=True
        ) as add:
            pipeline.ingest(tmp_path / "docs", owner="user-1")
        add.assert_not_called()
        assert pipeline._get_parent_store().count() == before

    def test_parents_are_stored_before_their_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = RAGPipeline()
        p.reset()
        order: list[str] = []
        real_parents, real_chunks = VectorStore.add_unsearchable, p._store_chunks

        def parents(self: VectorStore, chunks: list) -> int:
            order.append("parents")
            return real_parents(self, chunks)

        def chunks(lang: str, items: list) -> int:
            order.append("chunks")
            return real_chunks(lang, items)

        monkeypatch.setattr(VectorStore, "add_unsearchable", parents)
        monkeypatch.setattr(p, "_store_chunks", chunks)
        (tmp_path / "a.md").write_text(MARKDOWN, encoding="utf-8")
        p.ingest(tmp_path / "a.md")
        assert order[0] == "parents"

    def test_reset_clears_parents(self, pipeline: RAGPipeline) -> None:
        assert pipeline._get_parent_store().count() > 0
        pipeline.reset()
        assert pipeline._get_parent_store().count() == 0

    def test_two_owners_get_separate_parents(self, tmp_path: Path) -> None:
        p = RAGPipeline()
        p.reset()
        doc = tmp_path / "same.md"
        doc.write_text(MARKDOWN, encoding="utf-8")
        p.ingest(doc, owner="alice")
        p.ingest(doc, owner="bob")
        owners = {parent["metadata"]["owner"] for parent in p._get_parent_store().get_all_chunks()}
        assert owners == {"alice", "bob"}

    def test_section_limit_is_configurable_in_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(chunker_module, "PARENT_MAX_CHARS", 50)
        text = "\n\n".join("Paragraph number one two three." for _ in range(6))
        doc = Document(content=text, metadata={"source": "a.txt"}, doc_id="a.txt")
        parents, _ = Chunker(chunk_size=40, chunk_overlap=5).chunk_with_parents(doc)
        assert len(parents) > 1 and all(len(p.content) <= 50 for p in parents)
