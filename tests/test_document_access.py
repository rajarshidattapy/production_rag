"""Collections and reader roles recorded on chunks at ingestion, for fine-grained authorization."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.pipeline import RAGPipeline, _folder_rule
from src.retrieval.vector_store import VectorStore


@pytest.fixture
def library(tmp_path: Path) -> Path:
    for folder, name, text in (
        ("policies/hr", "leave.txt", "Leave policy for staff, managed by HR."),
        ("policies", "travel.txt", "Travel policy for everyone."),
        ("misc", "notes.txt", "General notes with no restrictions."),
    ):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def pipeline(library: Path) -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    p.config = p.config.model_copy(
        update={
            "source_collections": {"policies": "policies", "policies/hr": "hr-policies"},
            "source_roles": {"policies/hr": ["managers", "hr", "hr"]},
        }
    )
    for folder in ("policies", "misc"):
        p.ingest(library / folder)
    return p


def _by_file(records: list[dict]) -> dict[str, dict]:
    return {r["metadata"]["filename"]: r["metadata"] for r in records}


def _chunks(p: RAGPipeline) -> list[dict]:
    return [c for lang in ("en", "de", "es") for c in p._get_vector_store(lang).get_all_chunks()]


def test_chunks_record_collection_and_roles(pipeline: RAGPipeline) -> None:
    meta = _by_file(_chunks(pipeline))
    assert meta["leave.txt"]["collection"] == "hr-policies"
    assert meta["leave.txt"]["restricted"] is True
    assert meta["leave.txt"]["roles"] == ["hr", "managers"]  # sorted, duplicates removed
    assert meta["travel.txt"]["collection"] == "policies"
    assert meta["travel.txt"]["restricted"] is False
    assert "roles" not in meta["travel.txt"]
    assert meta["notes.txt"]["collection"] == "default"
    assert meta["notes.txt"]["restricted"] is False


def test_parents_carry_the_same_access(pipeline: RAGPipeline) -> None:
    parents = _by_file(pipeline._get_parent_store().get_all_chunks())
    assert parents["leave.txt"]["roles"] == ["hr", "managers"]
    assert parents["notes.txt"]["restricted"] is False


def test_roles_are_stored_as_a_list_a_filter_can_match(pipeline: RAGPipeline) -> None:
    hr = {"roles": {"$contains": "hr"}}
    unrestricted = {"restricted": False}
    found = {}
    for lang in ("en", "de", "es"):
        store = pipeline._get_vector_store(lang)
        for name, where in (("hr", hr), ("open", unrestricted)):
            for hit in store.similarity_search("policy", k=10, where=where):
                found.setdefault(name, set()).add(hit["metadata"]["filename"])
    assert found["hr"] == {"leave.txt"}
    assert found["open"] == {"travel.txt", "notes.txt"}


def test_only_string_lists_stay_lists(tmp_path: Path) -> None:
    store = VectorStore(collection_name="list_types", persist_path=tmp_path / "s")
    store.add_chunks(
        [
            Chunk(
                content="x",
                chunk_id="c1",
                metadata={"roles": ["a"], "empty": [], "mixed": [1, "a"], "n": 3},  # type: ignore[dict-item]
            )
        ]
    )
    meta = store.get_chunks(["c1"])[0]["metadata"]
    assert meta["roles"] == ["a"]
    assert meta["n"] == 3
    assert meta["empty"] == "[]" and meta["mixed"] == "[1, 'a']"


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("policies/hr/leave.txt", "hr-policies"),
        ("policies/travel.txt", "policies"),
        ("policies-old/x.txt", None),
        ("misc/notes.txt", None),
    ],
)
def test_longest_matching_folder(tmp_path: Path, relative: str, expected: str | None) -> None:
    rules = {"policies": "policies", "policies/hr": "hr-policies"}
    assert _folder_rule(tmp_path / relative, tmp_path, rules) == expected


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RAG_SOURCE_ROLES", '{"policies": ["HR"]}'),
        ("RAG_SOURCE_ROLES", '{"policies": ["has space"]}'),
        ("RAG_SOURCE_ROLES", '{"policies": "hr"}'),
        ("RAG_SOURCE_COLLECTIONS", '{"policies": "Bad Name"}'),
    ],
)
def test_invalid_names_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError):
        Settings()


def test_valid_configuration_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SOURCE_ROLES", '{"policies/hr": ["hr", "managers"]}')
    monkeypatch.setenv("RAG_SOURCE_COLLECTIONS", '{"policies/hr": "hr-policies"}')
    loaded = Settings()
    assert loaded.source_roles == {"policies/hr": ["hr", "managers"]}
    assert loaded.source_collections == {"policies/hr": "hr-policies"}


def test_uploads_are_unrestricted_by_default(tmp_path: Path) -> None:
    p = RAGPipeline()
    p.reset()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / f"{'a' * 32}_mine.txt").write_text("My own document.", encoding="utf-8")
    p.ingest(uploads, owner="u1")
    meta = _chunks(p)[0]["metadata"]
    assert meta["owner"] == "u1"
    assert meta["restricted"] is False
    assert meta["collection"] == "default"
