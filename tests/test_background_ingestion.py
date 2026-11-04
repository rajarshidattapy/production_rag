"""Tests for streaming ingestion and background ingest jobs."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import api as api_app
from src import pipeline as pipeline_module
from src.chat import store as store_module
from src.ingestion.loader import DocumentLoader
from src.pipeline import IngestProgress, RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from tests.test_page_retrieval import write_pdf

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(store_module, "_SCRYPT_N", 2**4)
    monkeypatch.setattr(store_module, "_SCRYPT_P", 1)
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    (folder / "nested").mkdir(parents=True)
    for i in range(4):
        (folder / f"doc{i}.txt").write_text(
            f"Document number {i} explains retrieval and generation in plain English."
        )
    (folder / "nested" / "deep.md").write_text("A nested document about hybrid search ranking.")
    return folder


@pytest.fixture
def pipeline() -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    return p


def _stored_ids(p: RAGPipeline) -> set[str]:
    return {
        c["id"] for lang in ("en", "de", "es") for c in p._get_vector_store(lang).get_all_chunks()
    }


class Gate:
    """Holds the ingest worker at a chosen file until the test lets it continue."""

    def __init__(self, hold_at: int = 1) -> None:
        self.hold_at = hold_at
        self.reached = threading.Event()
        self.release = threading.Event()
        self.loaded = 0
        self._real = DocumentLoader._load_file

    def __call__(self, loader: DocumentLoader, path: Path) -> Any:
        self.loaded += 1
        if self.loaded == self.hold_at:
            self.reached.set()
            assert self.release.wait(30), "test never released the worker"
        return self._real(loader, path)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[Gate]:
    g = Gate()
    monkeypatch.setattr(DocumentLoader, "_load_file", lambda self, path: g(self, path))
    yield g
    g.release.set()


class TestStreamingIngest:
    def test_returns_new_chunks_and_reports_progress(
        self, pipeline: RAGPipeline, corpus: Path
    ) -> None:
        snapshots: list[tuple[int, int]] = []
        seen: list[IngestProgress] = []

        def on_progress(progress: IngestProgress) -> None:
            snapshots.append((progress.files_done, progress.chunks_ingested))
            seen[:] = [progress]

        count = pipeline.ingest(corpus, on_progress=on_progress)

        progress = seen[0]
        assert count == progress.chunks_ingested == 5
        assert progress.files_total == progress.files_done == progress.files_indexed == 5
        assert snapshots[0] == (0, 0)
        assert snapshots == sorted(snapshots)  # totals only ever move forward
        assert "nested" in " ".join(progress.indexed_names)  # names say where a file was found
        assert len(_stored_ids(pipeline)) == 5

    def test_text_is_searchable_before_the_collection_finishes(
        self,
        pipeline: RAGPipeline,
        corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 1)
        stored_when_loading: list[int] = []
        real = pipeline.loader._load_file

        def load(path: Path) -> Any:
            stored_when_loading.append(pipeline.vector_store.count())
            return real(path)

        with (
            patch.object(pipeline.loader, "_load_file", load),
            patch("src.pipeline._detect_language", return_value="en"),
        ):
            pipeline.ingest(corpus)
        assert stored_when_loading == [0, 1, 2, 3, 4]

    def test_memory_does_not_hold_the_collection(
        self, pipeline: RAGPipeline, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 2)
        sizes: list[int] = []
        real = pipeline._store_chunks

        def spy(lang: str, chunks: list[Any]) -> int:
            sizes.append(len(chunks))
            return real(lang, chunks)

        with (
            patch.object(pipeline, "_store_chunks", spy),
            patch("src.pipeline._detect_language", return_value="en"),
        ):
            pipeline.ingest(corpus)
        assert sizes == [2, 2, 1]

    def test_bad_file_is_recorded_and_the_rest_are_indexed(
        self, pipeline: RAGPipeline, corpus: Path
    ) -> None:
        (corpus / "broken.pdf").write_bytes(b"this is not a pdf")
        seen: list[IngestProgress] = []
        count = pipeline.ingest(corpus, on_progress=lambda p: seen.__setitem__(slice(None), [p]))
        progress = seen[0]
        assert count == 5
        assert progress.files_total == progress.files_done == 6
        assert progress.files_failed == 1
        assert progress.failures[0]["file"] == "broken.pdf"
        assert progress.failures[0]["error"]

    def test_only_requested_file_still_raises(self, pipeline: RAGPipeline, tmp_path: Path) -> None:
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"this is not a pdf")
        with pytest.raises(Exception, match="(?i)pdf|stream|eof"):
            pipeline.ingest(bad)

    def test_pdf_without_text_is_reported_not_silently_skipped(
        self, pipeline: RAGPipeline, tmp_path: Path
    ) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        write_pdf(folder / "scan.pdf", ["", ""])
        write_pdf(folder / "real.pdf", ["Real text on this page."])
        seen: list[IngestProgress] = []
        pipeline.ingest(folder, on_progress=lambda p: seen.__setitem__(slice(None), [p]))
        assert seen[0].files_indexed == 1
        assert seen[0].failures[0]["file"] == "scan.pdf"
        assert "scanned" in seen[0].failures[0]["error"]

    def test_failure_details_are_capped_but_the_count_is_exact(
        self, pipeline: RAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_MAX_LISTED_FILES", 3)
        folder = tmp_path / "bad"
        folder.mkdir()
        for i in range(10):
            (folder / f"{i}.pdf").write_bytes(b"nope")
        seen: list[IngestProgress] = []
        pipeline.ingest(folder, on_progress=lambda p: seen.__setitem__(slice(None), [p]))
        assert seen[0].files_failed == 10
        assert len(seen[0].failures) == 3

    def test_running_again_embeds_nothing_twice(self, pipeline: RAGPipeline, corpus: Path) -> None:
        first = pipeline.ingest(corpus)
        before = _stored_ids(pipeline)
        with patch.object(
            type(pipeline.vector_store), "add_chunks", side_effect=AssertionError("re-embedded")
        ):
            assert pipeline.ingest(corpus) == 0
        assert first == 5
        assert _stored_ids(pipeline) == before

    def test_stopping_keeps_what_was_stored_and_a_rerun_finishes(
        self, pipeline: RAGPipeline, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 1)
        seen: list[IngestProgress] = []
        checks = iter([False, False, True])
        count = pipeline.ingest(
            corpus,
            on_progress=lambda p: seen.__setitem__(slice(None), [p]),
            should_stop=lambda: next(checks, True),
        )
        assert seen[0].stopped is True
        assert 0 < count < 5
        assert len(_stored_ids(pipeline)) == count

        assert pipeline.ingest(corpus) == 5 - count
        assert len(_stored_ids(pipeline)) == 5

    def test_stopping_inside_a_long_file_does_not_count_it_as_done(
        self, pipeline: RAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 2)
        folder = tmp_path / "long"
        folder.mkdir()
        (folder / "long.txt").write_text(
            "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
        )
        seen: list[IngestProgress] = []
        checks = iter([False, True])
        with patch("src.pipeline._detect_language", return_value="en"):
            count = pipeline.ingest(
                folder,
                on_progress=lambda p: seen.__setitem__(slice(None), [p]),
                should_stop=lambda: next(checks, True),
            )
            assert count == 2
            assert seen[0].files_done == seen[0].files_indexed == 0
            total = pipeline.ingest(folder)
        assert count + total == pipeline.vector_store.count() > 2

    def test_rerun_routes_short_text_to_the_same_collection(self) -> None:
        # Unseeded, langdetect answers "de" or "en" for this at random. Different answers
        # mean different collections, so a re-run would store the chunk a second time.
        assert len({pipeline_module._detect_language("Index") for _ in range(40)}) == 1

    def test_live_hybrid_index_keeps_up(self, pipeline: RAGPipeline, corpus: Path) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        pipeline._hybrid_retrievers["en"] = retriever
        with patch("src.pipeline._detect_language", return_value="en"):
            pipeline.ingest(corpus)
            added = [c for call in retriever.add_chunks.call_args_list for c in call.args[0]]
            assert len(added) == 5
            retriever.add_chunks.reset_mock()
            pipeline.ingest(corpus)
        retriever.add_chunks.assert_not_called()

    def test_directory_scan_skips_the_job_status_file(
        self, pipeline: RAGPipeline, tmp_path: Path
    ) -> None:
        (tmp_path / "ingest_jobs.json").write_text('{"secret": "job data"}')
        (tmp_path / "real.txt").write_text("A real document about retrieval.")
        names = [p.name for p in pipeline.loader.list_files(tmp_path)]
        assert names == ["real.txt"]

    def test_missing_source_still_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(FileNotFoundError):
            pipeline.ingest("/no/such/place")

    def test_monitored_pipeline_forwards_progress(self, corpus: Path) -> None:
        from src.monitoring import MonitoredRAGPipeline, Tracer

        monitored = MonitoredRAGPipeline(RAGPipeline(), tracer=Tracer(enabled=False))
        monitored.reset()
        seen: list[int] = []
        monitored.ingest(corpus, on_progress=lambda p: seen.append(p.files_done))
        assert seen[-1] == 5


def _signed_in(username: str = "alice") -> TestClient:
    client = TestClient(api_app.app)
    response = client.post("/auth/register", json={"username": username, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return client


def _wait(
    client: TestClient, job_id: str, done: Callable[[dict[str, Any]], bool] | None = None
) -> Any:
    done = done or (lambda job: job["status"] not in ("pending", "running"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/ingest/jobs/{job_id}").json()
        if done(job):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never got there: {job}")


def _start(client: TestClient, source: Path, **extra: Any) -> str:
    response = client.post("/ingest/async", json={"source": str(source), **extra})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


class TestJobLifecycle:
    def test_job_reports_progress_and_result(self, corpus: Path) -> None:
        client = TestClient(api_app.app)
        job = _wait(client, _start(client, corpus, label="My corpus"))
        assert job["status"] == "completed"
        assert job["label"] == "My corpus"
        assert job["files_total"] == job["files_done"] == 5
        assert job["files_failed"] == 0
        assert job["chunks_ingested"] == job["total_chunks"] == 5
        assert job["created_at"] > 0
        assert "source" not in job and "user_id" not in job and "owner" not in job

    def test_progress_is_visible_while_running(self, corpus: Path, gate: Gate) -> None:
        gate.hold_at = 3
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        assert gate.reached.wait(30)
        job = client.get(f"/ingest/jobs/{job_id}").json()
        assert job["status"] == "running"
        assert job["files_total"] == 5
        assert job["files_done"] == 2
        gate.release.set()
        assert _wait(client, job_id)["status"] == "completed"

    def test_label_defaults_to_the_folder_name(self, corpus: Path) -> None:
        client = TestClient(api_app.app)
        assert _wait(client, _start(client, corpus))["label"] == "corpus"

    def test_bad_files_are_listed_on_the_job(self, corpus: Path) -> None:
        (corpus / "broken.pdf").write_bytes(b"not a pdf")
        client = TestClient(api_app.app)
        job = _wait(client, _start(client, corpus))
        assert job["status"] == "completed"
        assert job["files_failed"] == 1
        assert job["failures"][0]["file"] == "broken.pdf"

    def test_jobs_run_one_at_a_time_in_order(
        self, corpus: Path, tmp_path: Path, gate: Gate
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "x.txt").write_text("Another document about ranking.")
        client = TestClient(api_app.app)
        first = _start(client, corpus)
        assert gate.reached.wait(30)
        second = _start(client, other)
        assert client.get(f"/ingest/jobs/{second}").json()["status"] == "pending"
        gate.release.set()
        assert _wait(client, first)["status"] == "completed"
        assert _wait(client, second)["status"] == "completed"

    def test_store_outage_fails_the_job_and_keeps_its_totals(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.retrieval.vector_store import VectorStoreUnavailableError

        client = TestClient(api_app.app)
        pipeline = api_app.get_pipeline()
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 1)
        real = pipeline._store_chunks
        calls = iter([True, False])

        def flaky(lang: str, chunks: list[Any]) -> int:
            if next(calls, False):
                return real(lang, chunks)
            raise VectorStoreUnavailableError("Lost connection to Chroma")

        monkeypatch.setattr(pipeline, "_store_chunks", flaky)
        job = _wait(client, _start(client, corpus))
        assert job["status"] == "failed"
        assert "Chroma" in job["error"]
        assert job["chunks_ingested"] == 1

    def test_progress_saves_are_throttled_but_the_end_is_always_saved(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saves: list[str] = []
        real = api_app._persist_ingest_job

        def counting(job_id: str, record: dict[str, Any]) -> None:
            saves.append(record["status"])
            real(job_id, record)

        monkeypatch.setattr(api_app, "_persist_ingest_job", counting)
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        _wait(client, job_id)
        assert saves[-1] == "completed"
        assert len(saves) <= 5
        assert json.loads(api_app._ingest_jobs_path().read_text())[job_id]["status"] == "completed"


class TestCancelAndRetry:
    def test_cancel_stops_after_the_current_step_and_keeps_text(
        self, corpus: Path, gate: Gate, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 1)
        gate.hold_at = 3
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        assert gate.reached.wait(30)

        response = client.post(f"/ingest/jobs/{job_id}/cancel")
        assert response.status_code == 200
        assert response.json()["cancel_requested"] is True
        gate.release.set()

        job = _wait(client, job_id)
        assert job["status"] == "cancelled"
        assert 0 < job["chunks_ingested"] < 5
        assert job["files_done"] < 5
        assert job["total_chunks"] == job["chunks_ingested"]

    def test_cancelling_a_queued_job_means_it_never_runs(
        self, corpus: Path, tmp_path: Path, gate: Gate
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "reset-target.txt").write_text("Would be indexed.")
        client = TestClient(api_app.app)
        first = _start(client, corpus)
        assert gate.reached.wait(30)
        queued = _start(client, other, reset=True)
        assert client.post(f"/ingest/jobs/{queued}/cancel").status_code == 200
        gate.release.set()
        assert _wait(client, first)["status"] == "completed"
        job = _wait(client, queued)
        assert job["status"] == "cancelled"
        assert api_app.get_pipeline().stats()["chunks_in_store"] == 5  # its reset never ran

    def test_cancelling_a_finished_job_conflicts(self, corpus: Path) -> None:
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        _wait(client, job_id)
        assert client.post(f"/ingest/jobs/{job_id}/cancel").status_code == 409

    def test_retry_resumes_without_embedding_stored_text_again(
        self, corpus: Path, gate: Gate, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline_module, "_INGEST_STEP_CHUNKS", 1)
        gate.hold_at = 3
        client = TestClient(api_app.app)
        job_id = _start(client, corpus, label="Corpus")
        assert gate.reached.wait(30)
        client.post(f"/ingest/jobs/{job_id}/cancel")
        gate.release.set()
        stopped = _wait(client, job_id)

        response = client.post(f"/ingest/jobs/{job_id}/retry")
        assert response.status_code == 202
        new_id = response.json()["job_id"]
        assert new_id != job_id
        job = _wait(client, new_id)
        assert job["status"] == "completed"
        assert job["label"] == "Corpus"
        assert job["chunks_ingested"] == 5 - stopped["chunks_ingested"]
        assert job["total_chunks"] == 5

    def test_a_completed_job_cannot_be_rerun(self, corpus: Path) -> None:
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        _wait(client, job_id)
        assert client.post(f"/ingest/jobs/{job_id}/retry").status_code == 409

    def test_retry_after_the_source_is_gone_says_so(self, corpus: Path, tmp_path: Path) -> None:
        client = TestClient(api_app.app)
        api_app._record_ingest_job(
            "old", status="cancelled", source=str(tmp_path / "gone"), created_at=1.0
        )
        assert client.post("/ingest/jobs/old/retry").status_code == 400

    def test_unknown_job_is_404(self) -> None:
        client = TestClient(api_app.app)
        assert client.post("/ingest/jobs/nope/cancel").status_code == 404
        assert client.post("/ingest/jobs/nope/retry").status_code == 404


class TestRestartAndShutdown:
    def test_shutdown_stops_a_running_job_as_interrupted(self, corpus: Path, gate: Gate) -> None:
        with TestClient(api_app.app) as client:
            job_id = _start(client, corpus)
            assert gate.reached.wait(30)
        gate.release.set()  # leaving the block ran the app's shutdown
        job = _wait(TestClient(api_app.app), job_id)
        assert job["status"] == "interrupted"

    def test_job_cut_off_by_a_crash_can_be_run_again_after_restart(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _signed_in()
        user = api_app.get_chat_store().user_for_session(client.cookies["rag_session"])
        assert user is not None
        api_app._record_ingest_job(
            "crashed",
            status="running",
            source=str(corpus),
            label="corpus",
            user_id=user.id,
            created_at=time.time(),
            files_total=5,
            files_done=2,
        )
        api_app._ingest_jobs.clear()
        monkeypatch.setattr(api_app, "_INSTANCE_ID", "a-new-process")

        listed = client.get("/ingest/jobs").json()
        assert [(j["job_id"], j["status"]) for j in listed] == [("crashed", "interrupted")]
        assert listed[0]["files_done"] == 2

        response = client.post("/ingest/jobs/crashed/retry")
        assert response.status_code == 202
        assert _wait(client, response.json()["job_id"])["status"] == "completed"

    def test_cancel_after_restart_explains_it_is_not_running_here(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = TestClient(api_app.app)
        api_app._record_ingest_job("orphan", status="running", source=str(corpus))
        api_app._ingest_jobs.clear()
        monkeypatch.setattr(api_app, "_INSTANCE_ID", "a-new-process")
        assert client.post("/ingest/jobs/orphan/cancel").status_code == 409


class TestJobOwnership:
    def test_list_needs_signing_in_and_shows_only_your_jobs(self, corpus: Path) -> None:
        assert TestClient(api_app.app).get("/ingest/jobs").status_code == 401
        alice, bob = _signed_in("alice"), _signed_in("bob")
        job_id = _start(alice, corpus)
        _wait(alice, job_id)
        assert [j["job_id"] for j in alice.get("/ingest/jobs").json()] == [job_id]
        assert bob.get("/ingest/jobs").json() == []

    def test_newest_jobs_list_first(self, corpus: Path) -> None:
        client = _signed_in()
        first = _start(client, corpus)
        _wait(client, first)
        second = _start(client, corpus)
        _wait(client, second)
        assert [j["job_id"] for j in client.get("/ingest/jobs").json()] == [second, first]

    def test_another_users_job_looks_missing(self, corpus: Path) -> None:
        alice, bob = _signed_in("alice"), _signed_in("bob")
        job_id = _start(alice, corpus, label="alice's private folder")
        _wait(alice, job_id)
        anonymous = TestClient(api_app.app)
        for other in (bob, anonymous):
            assert other.get(f"/ingest/jobs/{job_id}").status_code == 404
            assert other.post(f"/ingest/jobs/{job_id}/cancel").status_code == 404
            assert other.post(f"/ingest/jobs/{job_id}/retry").status_code == 404

    def test_api_key_clients_keep_working_without_an_account(self, corpus: Path) -> None:
        client = TestClient(api_app.app)
        job_id = _start(client, corpus)
        assert _wait(client, job_id)["status"] == "completed"
        assert client.get("/ingest/jobs").status_code == 401

    def test_routes_follow_the_api_key(self, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _signed_in()
        monkeypatch.setattr(api_app, "_RAG_API_KEY", "deployment-key")
        assert client.get("/ingest/jobs").status_code == 401
        assert client.post("/ingest/jobs/x/cancel").status_code == 401
        assert client.post("/ingest/jobs/x/retry").status_code == 401
        assert client.post("/ingest/async", json={"source": str(corpus)}).status_code == 401


class TestChatNotes:
    def _chat(self, client: TestClient, title: str = "Docs") -> str:
        return client.post("/chats", json={"title": title}).json()["id"]

    def test_result_lands_in_the_chat_without_a_browser_watching(self, corpus: Path) -> None:
        client = _signed_in()
        chat_id = self._chat(client)
        _wait(client, _start(client, corpus, chat_id=chat_id))
        chat = client.get(f"/chats/{chat_id}").json()
        assert len(chat["files"]) == 5
        assert chat["messages"][-1]["role"] == "assistant"
        assert chat["messages"][-1]["text"].startswith("Added doc0.txt, doc1.txt")

    def test_upload_prefix_is_not_shown(self, tmp_path: Path) -> None:
        client = _signed_in()
        chat_id = self._chat(client)
        batch = "a" * 32
        uploaded = client.post(
            f"/upload?filename=report.pdf&batch={batch}",
            content=write_pdf(tmp_path / "r.pdf", ["Some text."]).read_bytes(),
        ).json()
        _wait(client, _start(client, Path(uploaded["directory"]), chat_id=chat_id))
        assert client.get(f"/chats/{chat_id}").json()["files"] == ["report.pdf"]

    def test_large_collections_record_a_count_not_every_name(self, tmp_path: Path) -> None:
        folder = tmp_path / "big"
        folder.mkdir()
        for i in range(25):
            (folder / f"{i}.txt").write_text(f"Document {i} about retrieval.")
        client = _signed_in()
        chat_id = self._chat(client)
        _wait(client, _start(client, folder, chat_id=chat_id))
        chat = client.get(f"/chats/{chat_id}").json()
        assert chat["files"] == ["25 documents"]
        assert chat["messages"][-1]["text"] == "Added 25 documents to this chat."

    def test_failures_are_mentioned(self, corpus: Path) -> None:
        (corpus / "broken.pdf").write_bytes(b"nope")
        client = _signed_in()
        chat_id = self._chat(client)
        _wait(client, _start(client, corpus, chat_id=chat_id))
        text = client.get(f"/chats/{chat_id}").json()["messages"][-1]["text"]
        assert "1 of 6 could not be indexed." in text

    def test_empty_folder_says_nothing_was_found(self, tmp_path: Path) -> None:
        folder = tmp_path / "empty"
        folder.mkdir()
        client = _signed_in()
        chat_id = self._chat(client)
        job = _wait(client, _start(client, folder, chat_id=chat_id))
        assert job["status"] == "completed"
        text = client.get(f"/chats/{chat_id}").json()["messages"][-1]["text"]
        assert text == "No documents were found in empty."

    def test_cancel_is_mentioned(self, corpus: Path, gate: Gate) -> None:
        gate.hold_at = 3
        client = _signed_in()
        chat_id = self._chat(client)
        job_id = _start(client, corpus, chat_id=chat_id)
        assert gate.reached.wait(30)
        client.post(f"/ingest/jobs/{job_id}/cancel")
        gate.release.set()
        _wait(client, job_id)
        text = client.get(f"/chats/{chat_id}").json()["messages"][-1]["text"]
        assert "Indexing was cancelled after" in text

    def test_rerun_replaces_what_the_cancelled_attempt_recorded(
        self, tmp_path: Path, gate: Gate
    ) -> None:
        folder = tmp_path / "big"
        folder.mkdir()
        for i in range(30):
            (folder / f"{i:02}.txt").write_text(f"Document {i} about retrieval and search.")
        gate.hold_at = 22  # past the point where a chat lists a count rather than names
        client = _signed_in()
        chat_id = self._chat(client)
        job_id = _start(client, folder, chat_id=chat_id)
        assert gate.reached.wait(30)
        client.post(f"/ingest/jobs/{job_id}/cancel")
        gate.release.set()
        _wait(client, job_id)
        assert client.get(f"/chats/{chat_id}").json()["files"] == ["22 documents"]

        new_id = client.post(f"/ingest/jobs/{job_id}/retry").json()["job_id"]
        _wait(client, new_id)
        assert client.get(f"/chats/{chat_id}").json()["files"] == ["30 documents"]

    def test_rerun_after_an_interrupted_rerun_still_replaces_the_first_attempt(
        self, corpus: Path
    ) -> None:
        client = _signed_in()
        chat_id = self._chat(client)
        user = api_app.get_chat_store().user_for_session(client.cookies["rag_session"])
        assert user is not None
        api_app.get_chat_store().add_files(user.id, chat_id, ["3 documents"], note="Cancelled.")
        api_app._record_ingest_job(
            "second-attempt",
            status="interrupted",
            source=str(corpus),
            user_id=user.id,
            chat_id=chat_id,
            replaces_chat_files=["3 documents"],
        )
        new_id = client.post("/ingest/jobs/second-attempt/retry").json()["job_id"]
        _wait(client, new_id)
        assert "3 documents" not in client.get(f"/chats/{chat_id}").json()["files"]

    def test_interruption_adds_no_note_because_a_rerun_is_expected(
        self, corpus: Path, gate: Gate
    ) -> None:
        with TestClient(api_app.app) as client:
            client.post("/auth/register", json={"username": "alice", "password": PASSWORD})
            chat_id = client.post("/chats", json={"title": "Docs"}).json()["id"]
            job_id = _start(client, corpus, chat_id=chat_id)
            assert gate.reached.wait(30)
        gate.release.set()
        assert _wait(client, job_id)["status"] == "interrupted"
        assert client.get(f"/chats/{chat_id}").json()["messages"] == []

    def test_deleted_chat_does_not_fail_the_job(self, corpus: Path, gate: Gate) -> None:
        client = _signed_in()
        chat_id = self._chat(client)
        job_id = _start(client, corpus, chat_id=chat_id)
        assert gate.reached.wait(30)
        client.delete(f"/chats/{chat_id}")
        gate.release.set()
        assert _wait(client, job_id)["status"] == "completed"

    def test_chat_must_be_yours(self, corpus: Path) -> None:
        alice, bob = _signed_in("alice"), _signed_in("bob")
        chat_id = self._chat(alice)
        body = {"source": str(corpus), "chat_id": chat_id}
        assert bob.post("/ingest/async", json=body).status_code == 404
        assert TestClient(api_app.app).post("/ingest/async", json=body).status_code == 401
        assert alice.get(f"/chats/{chat_id}").json()["messages"] == []


class TestUploadBatches:
    def test_batch_shares_a_directory(self, tmp_path: Path) -> None:
        client = TestClient(api_app.app)
        batch = "b" * 32
        pdf = write_pdf(tmp_path / "x.pdf", ["Text."]).read_bytes()
        one = client.post(f"/upload?filename=a.pdf&batch={batch}", content=pdf).json()
        two = client.post(f"/upload?filename=b.pdf&batch={batch}", content=pdf).json()
        assert one["directory"] == two["directory"]
        assert Path(one["directory"]).name == batch
        assert Path(one["source"]).parent == Path(one["directory"])
        job = _wait(client, _start(client, Path(one["directory"])))
        assert job["files_total"] == 2

    def test_upload_without_a_batch_is_unchanged(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "x.pdf", ["Text."]).read_bytes()
        uploaded = TestClient(api_app.app).post("/upload?filename=a.pdf", content=pdf).json()
        assert Path(uploaded["source"]).parent.name == "uploads"

    @pytest.mark.parametrize("batch", ["..", "../../etc", "a" * 31, "A" * 32, "x/y"])
    def test_batch_cannot_escape_the_upload_directory(self, batch: str) -> None:
        response = TestClient(api_app.app).post(
            "/upload", params={"filename": "a.pdf", "batch": batch}, content=b"%PDF"
        )
        assert response.status_code == 422
