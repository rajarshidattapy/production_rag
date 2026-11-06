"""Document versions, dates, and source authority recorded at ingestion."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pypdf
import pytest
from pydantic import ValidationError

from src.config import Settings, settings
from src.ingestion.loader import DocumentLoader
from src.pipeline import RAGPipeline, _document_key, _source_authority
from tests.test_page_retrieval import write_pdf

FIELDS = (
    "document_key",
    "version_id",
    "document_date",
    "document_date_source",
    "ingested_at",
    "authority",
    "content_hash",
)


def _pdf_with_dates(path: Path, mod: str | None, created: str | None = None) -> Path:
    plain = write_pdf(path.with_name("plain_" + path.name), ["Some text on a page."])
    writer = pypdf.PdfWriter(clone_from=str(plain))
    info = {}
    if mod:
        info["/ModDate"] = mod
    if created:
        info["/CreationDate"] = created
    writer.add_metadata(info)
    with open(path, "wb") as f:
        writer.write(f)
    plain.unlink()
    return path


def _chunks(p: RAGPipeline) -> list[dict]:
    return [c for lang in ("en", "de", "es") for c in p._get_vector_store(lang).get_all_chunks()]


@pytest.fixture
def pipeline() -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    return p


class TestLoaderFacts:
    def test_text_file_records_hash_and_modified_time(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_bytes(b"Hello version one.")
        os.utime(path, (1_700_000_000, 1_700_000_000))
        meta = DocumentLoader().load(path)[0].metadata
        assert meta["content_hash"] == hashlib.sha256(b"Hello version one.").hexdigest()
        assert meta["file_modified_at"] == meta["document_date"] == 1_700_000_000
        assert meta["document_date_source"] == "file_modified"

    def test_pdf_uses_its_own_modified_date(self, tmp_path: Path) -> None:
        pdf = _pdf_with_dates(tmp_path / "r.pdf", "D:20240315120000Z", "D:20200101000000Z")
        meta = DocumentLoader().load(pdf)[0].metadata
        assert meta["document_date"] == 1710504000.0  # 2024-03-15 12:00 UTC
        assert meta["document_date_source"] == "pdf_metadata"

    def test_pdf_falls_back_to_its_creation_date(self, tmp_path: Path) -> None:
        pdf = _pdf_with_dates(tmp_path / "r.pdf", None, "D:20200101000000Z")
        assert DocumentLoader().load(pdf)[0].metadata["document_date"] == 1577836800.0

    @pytest.mark.parametrize("mod", [None, "not a date", "D:2024"])
    def test_pdf_without_a_usable_date_uses_the_file_time(
        self, tmp_path: Path, mod: str | None
    ) -> None:
        pdf = _pdf_with_dates(tmp_path / "r.pdf", mod)
        meta = DocumentLoader().load(pdf)[0].metadata
        if meta["document_date_source"] == "pdf_metadata":
            pytest.skip("this pypdf version accepts the partial date")
        assert meta["document_date"] == meta["file_modified_at"]


class TestRecordedOnChunks:
    def test_every_chunk_and_parent_carries_it(self, pipeline: RAGPipeline, tmp_path: Path) -> None:
        before = time.time()
        (tmp_path / "guide.md").write_text("# Guide\n\nRun the installer.\n", encoding="utf-8")
        pipeline.ingest(tmp_path / "guide.md", owner="u1")
        chunks = _chunks(pipeline)
        parents = pipeline._get_parent_store().get_all_chunks()
        for record in [*chunks, *parents]:
            meta = record["metadata"]
            assert all(field in meta for field in FIELDS), meta
            assert before <= meta["ingested_at"] <= time.time()
            assert meta["authority"] == 50
            assert meta["document_key"] == "u1:guide.md"
            assert meta["version_id"] == meta["content_hash"][:16]

    def test_edited_file_is_stored_as_a_new_version(
        self, pipeline: RAGPipeline, tmp_path: Path
    ) -> None:
        doc = tmp_path / "policy.txt"
        doc.write_text(
            "Vacation policy: employees get 20 days of leave each year.", encoding="utf-8"
        )
        assert pipeline.ingest(doc) == 1
        # Same first 50 characters and same position, so the old chunk IDs would have collided.
        doc.write_text(
            "Vacation policy: employees get 20 days of leave each year, plus 5.", encoding="utf-8"
        )
        assert pipeline.ingest(doc) == 1

        versions = {c["metadata"]["version_id"]: c["document"] for c in _chunks(pipeline)}
        assert len(versions) == 2
        assert any(text.endswith("plus 5.") for text in versions.values())
        assert {c["metadata"]["document_key"] for c in _chunks(pipeline)} == {"shared:policy.txt"}

    def test_unchanged_file_adds_nothing(self, pipeline: RAGPipeline, tmp_path: Path) -> None:
        doc = tmp_path / "a.txt"
        doc.write_text("Stable text.", encoding="utf-8")
        assert pipeline.ingest(doc) == 1
        assert pipeline.ingest(doc) == 0

    def test_uploads_of_the_same_name_are_versions_of_one_document(
        self, pipeline: RAGPipeline, tmp_path: Path
    ) -> None:
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        first = uploads / f"{'a' * 32}_report.txt"
        second = uploads / f"{'b' * 32}_report.txt"
        first.write_text("Report, first draft.", encoding="utf-8")
        second.write_text("Report, final version.", encoding="utf-8")
        pipeline.ingest(first, owner="u1")
        pipeline.ingest(second, owner="u1")
        keys = {c["metadata"]["document_key"] for c in _chunks(pipeline)}
        assert keys == {"u1:uploads/report.txt"}
        assert len({c["metadata"]["version_id"] for c in _chunks(pipeline)}) == 2

    def test_other_owners_have_separate_documents(self, tmp_path: Path) -> None:
        assert _document_key(tmp_path / "x.txt", tmp_path, "a") != _document_key(
            tmp_path / "x.txt", tmp_path, "b"
        )


class TestAuthority:
    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            ("policies/hr/leave.pdf", 95),
            ("policies/travel.pdf", 90),
            ("uploads/abc_report.pdf", 30),
            ("misc/notes.txt", 50),
            ("policiesX/fake.pdf", 50),  # a folder name, not a string prefix
        ],
    )
    def test_longest_matching_folder_wins(
        self, tmp_path: Path, relative: str, expected: int
    ) -> None:
        rules = {"policies": 90, "policies/hr": 95, "uploads": 30}
        assert _source_authority(tmp_path / relative, tmp_path, rules, 50) == expected

    def test_files_outside_the_data_directory_get_the_default(self, tmp_path: Path) -> None:
        assert _source_authority(Path("/elsewhere/x.txt"), tmp_path / "data", {"x": 90}, 40) == 40

    def test_configured_authority_reaches_the_chunks(
        self, pipeline: RAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policies = tmp_path / "policies"
        policies.mkdir()
        (policies / "leave.txt").write_text("Official leave policy.", encoding="utf-8")
        pipeline.config = pipeline.config.model_copy(update={"source_authority": {"policies": 90}})
        pipeline.ingest(policies)
        assert {c["metadata"]["authority"] for c in _chunks(pipeline)} == {90}

    def test_values_must_be_between_0_and_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_SOURCE_AUTHORITY", '{"policies": 150}')
        with pytest.raises(ValidationError):
            Settings()
        monkeypatch.setenv("RAG_SOURCE_AUTHORITY", '{"policies": 90, "uploads": 30}')
        assert Settings().source_authority == {"policies": 90, "uploads": 30}

    def test_default_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_DEFAULT_SOURCE_AUTHORITY", "20")
        assert Settings().default_source_authority == 20
        assert settings.default_source_authority == 50
