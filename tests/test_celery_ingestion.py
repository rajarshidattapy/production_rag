"""Opt-in Celery delivery for background indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from src import api
from src import celery_app as celery_module
from src.config import settings


@pytest.fixture
def celery_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ingest_queue", "celery")
    api._ingest_jobs.clear()
    api._active_ingest_jobs.clear()
    yield
    api._ingest_jobs.clear()
    api._active_ingest_jobs.clear()


def test_default_queue_needs_no_broker() -> None:
    from src.config import Settings

    configured = Settings(_env_file=None)
    assert configured.ingest_queue == "local"


def test_enqueue_publishes_a_self_contained_celery_message(
    celery_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(celery_module, "submit_ingest_job", published.append)

    job_id = api._enqueue_ingest_job(source, label="Corpus", user_id="user-1")

    assert published == [
        {
            "job_id": job_id,
            "reset": False,
            "record": {
                "status": "pending",
                "source": str(source),
                "created_at": published[0]["record"]["created_at"],
                "label": "Corpus",
                "user_id": "user-1",
            },
        }
    ]
    assert job_id not in api._active_ingest_jobs
    assert api._read_persisted_ingest_jobs()[job_id]["status"] == "pending"


def test_broker_failure_is_a_503_and_marks_the_job_failed(
    celery_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()

    def unavailable(_message: dict[str, Any]) -> None:
        raise celery_module.CeleryQueueUnavailableError("broker down")

    monkeypatch.setattr(celery_module, "submit_ingest_job", unavailable)
    with pytest.raises(HTTPException) as raised:
        api._enqueue_ingest_job(source)

    assert raised.value.status_code == 503
    assert list(api._read_persisted_ingest_jobs().values())[-1]["status"] == "failed"


def test_worker_restores_the_job_record_before_running(
    celery_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    message = {
        "job_id": "job-1",
        "reset": True,
        "record": {
            "status": "pending",
            "source": str(source),
            "created_at": 1.0,
            "user_id": "user-1",
        },
    }
    seen: list[tuple[str, Path, bool, str | None]] = []

    def run(job_id: str, path: Path, reset: bool, owner: str | None) -> None:
        seen.append((job_id, path, reset, owner))
        api._record_ingest_job(job_id, status="completed")

    monkeypatch.setattr(api, "_run_ingest_job", run)
    assert api._run_queued_ingest_job(message) == "completed"
    assert seen == [("job-1", source.resolve(), True, "user-1")]


def test_cancellation_is_visible_to_a_separate_worker(celery_mode: None) -> None:
    api._record_ingest_job("job-1", status="running", source="unused")

    response = api.cancel_ingest_job("job-1", None)

    assert response.cancel_requested is True
    assert api._persisted_cancel_requested("job-1") is True


def test_progress_save_cannot_clear_a_cancellation_request(celery_mode: None) -> None:
    api._record_ingest_job("job-1", status="running", cancel_requested=True)
    stale = dict(api._read_persisted_ingest_jobs()["job-1"])
    stale["cancel_requested"] = False

    api._persist_ingest_job("job-1", stale)

    assert api._read_persisted_ingest_jobs()["job-1"]["cancel_requested"] is True


def test_remote_celery_work_is_not_reported_as_interrupted(
    celery_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "_INSTANCE_ID", "api-replica")
    view = api._ingest_job_view(
        "job-1", {"status": "running", "owner": "celery-worker", "created_at": 1.0}
    )
    assert view.status == "running"


def test_celery_is_serial_and_acknowledges_only_finished_jobs() -> None:
    assert celery_module.celery_app.conf.worker_concurrency == 1
    assert celery_module.celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_module.celery_app.conf.task_acks_late is True
    assert celery_module.celery_app.conf.task_reject_on_worker_lost is True


def test_publish_uses_the_job_id_as_the_celery_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def apply_async(*, args: list[Any], task_id: str) -> None:
        calls.append({"args": args, "task_id": task_id})

    monkeypatch.setattr(celery_module.run_ingest_job, "apply_async", apply_async)
    message = {"job_id": "job-1", "reset": False, "record": {}}
    celery_module.submit_ingest_job(message)
    assert calls == [{"args": [message], "task_id": "job-1"}]
