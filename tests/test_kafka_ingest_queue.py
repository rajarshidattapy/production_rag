"""Tests for the Kafka-backed ingest job queue.

The unit tests use fake Kafka clients. ``test_real_broker_round_trip`` runs against a real
broker when RAG_TEST_KAFKA_BOOTSTRAP is set, e.g. ``docker compose --profile kafka up kafka``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from src import api as api_app
from src.ingest_queue import KafkaIngestQueue, QueueUnavailableError


class FakeMessage:
    def __init__(self, value: bytes, offset: int) -> None:
        self._value, self._offset = value, offset

    def topic(self) -> str:
        return "jobs"

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return self._offset

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None


class FakeConsumer:
    """One partition: delivers from the current position unless paused."""

    def __init__(self, values: list[bytes]) -> None:
        self.log = [FakeMessage(v, i) for i, v in enumerate(values)]
        self.position = 0
        self.paused = False
        self.committed: list[int] = []
        self.polls_while_paused = 0
        self.closed = False

    def subscribe(self, _topics: list[str]) -> None:
        pass

    def assignment(self) -> list[str]:
        return ["jobs-0"]

    def pause(self, _partitions: Any) -> None:
        self.paused = True

    def resume(self, _partitions: Any) -> None:
        self.paused = False

    def poll(self, timeout: float) -> FakeMessage | None:
        if self.paused:
            self.polls_while_paused += 1
        if self.paused or self.position >= len(self.log):
            time.sleep(min(timeout, 0.01))
            return None
        msg = self.log[self.position]
        self.position += 1
        return msg

    def seek(self, tp: Any) -> None:
        self.position = tp.offset

    def commit(self, message: FakeMessage, asynchronous: bool) -> None:
        self.committed.append(message.offset() + 1)

    def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self, error: Any = None, remaining: int = 0) -> None:
        self.sent: list[tuple[bytes, bytes]] = []
        self._error, self._remaining = error, remaining
        self._callbacks: deque[Any] = deque()

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.sent.append((key, value))
        self._callbacks.append(on_delivery)

    def flush(self, _timeout: float) -> int:
        while self._callbacks:
            self._callbacks.popleft()(self._error, None)
        return self._remaining


def _job(i: int) -> bytes:
    return json.dumps({"job_id": f"job-{i}", "reset": False, "record": {}}).encode()


def _queue(run: Any, consumer: Any = None, producer: Any = None) -> KafkaIngestQueue:
    return KafkaIngestQueue(
        run,
        bootstrap_servers="unused:9092",
        topic="jobs",
        group_id="test",
        timeout_seconds=1.0,
        producer=producer,
        consumer=consumer,
    )


def _wait_for(condition: Any, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_jobs_run_one_at_a_time_in_order_and_are_committed_after() -> None:
    consumer = FakeConsumer([_job(0), _job(1), _job(2)])
    ran: list[str] = []
    active = 0
    overlap = False

    def run(job: dict[str, Any]) -> str:
        nonlocal active, overlap
        active += 1
        overlap = overlap or active > 1
        time.sleep(0.05)
        ran.append(job["job_id"])
        assert consumer.committed == [i + 1 for i in range(len(ran) - 1)], "committed too early"
        active -= 1
        return "completed"

    queue = _queue(run, consumer=consumer)
    queue.start()
    _wait_for(lambda: len(consumer.committed) == 3)
    queue.close(5)

    assert ran == ["job-0", "job-1", "job-2"]
    assert not overlap
    assert consumer.committed == [1, 2, 3]
    assert consumer.closed


def test_consumer_keeps_polling_while_a_job_runs() -> None:
    """Polling while paused is what keeps a long job's consumer in the group."""
    consumer = FakeConsumer([_job(0)])
    release = threading.Event()
    queue = _queue(lambda _job: (release.wait(5), "completed")[1], consumer=consumer)
    queue.start()
    _wait_for(lambda: consumer.polls_while_paused >= 3)
    release.set()
    _wait_for(lambda: consumer.committed == [1])
    queue.close(5)
    assert not consumer.paused


def test_interrupted_job_is_left_on_the_topic() -> None:
    consumer = FakeConsumer([_job(0)])
    queue = _queue(lambda _job: "interrupted", consumer=consumer)
    queue.start()
    _wait_for(lambda: consumer.paused)
    queue.close(5)
    assert consumer.committed == []


def test_failed_and_cancelled_jobs_are_committed() -> None:
    consumer = FakeConsumer([_job(0), _job(1), _job(2)])
    results = iter(["failed", "cancelled"])

    def run(_job: dict[str, Any]) -> str:
        status = next(results, None)
        if status is None:
            raise RuntimeError("boom")
        return status

    queue = _queue(run, consumer=consumer)
    queue.start()
    _wait_for(lambda: len(consumer.committed) == 3)
    queue.close(5)


def test_unreadable_message_is_skipped() -> None:
    consumer = FakeConsumer([b"not json", _job(1)])
    ran: list[str] = []
    queue = _queue(lambda job: (ran.append(job["job_id"]), "completed")[1], consumer=consumer)
    queue.start()
    _wait_for(lambda: len(consumer.committed) == 2)
    queue.close(5)
    assert ran == ["job-1"]


def test_reassigned_partition_does_not_run_the_job_twice() -> None:
    """If the pause is lost to a rebalance, the running job is redelivered: put it back."""
    consumer = FakeConsumer([_job(0), _job(1)])
    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    def run(job: dict[str, Any]) -> str:
        ran.append(job["job_id"])
        started.set()
        release.wait(5)
        return "completed"

    queue = _queue(run, consumer=consumer)
    queue.start()
    assert started.wait(5)
    consumer.position, consumer.paused = 0, False  # what a revoke-and-reassign does
    _wait_for(lambda: consumer.paused)
    release.set()
    _wait_for(lambda: len(consumer.committed) == 2)
    queue.close(5)
    assert ran == ["job-0", "job-1"]


def test_submit_returns_once_the_broker_has_the_job() -> None:
    producer = FakeProducer()
    _queue(lambda _job: "completed", producer=producer).submit({"job_id": "abc", "x": 1})
    assert producer.sent == [(b"abc", json.dumps({"job_id": "abc", "x": 1}).encode())]


@pytest.mark.parametrize("producer", [FakeProducer(error="broker down"), FakeProducer(remaining=1)])
def test_submit_raises_when_not_delivered(producer: FakeProducer) -> None:
    with pytest.raises(QueueUnavailableError):
        _queue(lambda _job: "completed", producer=producer).submit({"job_id": "abc"})


# ---------------------------------------------------------------------------
# API: what happens to a job record around the queue
# ---------------------------------------------------------------------------


class _Recording:
    def __init__(self, error: Exception | None = None) -> None:
        self.jobs: list[dict[str, Any]] = []
        self._error = error

    def submit(self, job: dict[str, Any]) -> None:
        if self._error:
            raise self._error
        self.jobs.append(json.loads(json.dumps(job)))  # what goes over the wire


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(api_app._settings, "data_dir", tmp_path)
    (tmp_path / "doc.txt").write_text("Kafka keeps waiting jobs on a topic.")
    return tmp_path


def test_unavailable_queue_fails_the_request_and_the_job(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_app, "_ingest_queue", _Recording(QueueUnavailableError("down")))
    with pytest.raises(HTTPException) as exc:
        api_app._enqueue_ingest_job(data_dir / "doc.txt", label="doc")
    assert exc.value.status_code == 503
    job_id = next(
        i for i, j in api_app._read_persisted_ingest_jobs().items() if j.get("label") == "doc"
    )
    assert api_app._find_ingest_job(job_id)["status"] == "failed"


def test_job_picked_up_elsewhere_keeps_its_owner_and_chat(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _Recording()
    monkeypatch.setattr(api_app, "_ingest_queue", queue)
    job_id = api_app._enqueue_ingest_job(
        data_dir / "doc.txt", label="doc", user_id="u1", chat_id="c1"
    )
    # Another process: nothing about the job in memory.
    with api_app._ingest_jobs_lock:
        api_app._ingest_jobs.pop(job_id)

    seen: dict[str, Any] = {}

    def fake_run(jid: str, path: Path, reset: bool, owner: str | None) -> str:
        seen.update(api_app._find_ingest_job(jid), path=path, owner=owner)
        return "completed"

    monkeypatch.setattr(api_app, "_run_ingest_job", fake_run)
    assert api_app._run_queued_ingest_job(queue.jobs[0]) == "completed"
    assert seen["user_id"] == "u1" and seen["chat_id"] == "c1" and seen["label"] == "doc"
    assert seen["owner"] == "u1"
    assert seen["path"] == (data_dir / "doc.txt").resolve()


def test_queued_path_is_checked_again_when_the_job_runs(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A message on the topic gets the same data-directory check as a request."""
    monkeypatch.setattr(api_app, "_run_ingest_job", lambda *_a: pytest.fail("job should not run"))
    job_id = str(uuid.uuid4())
    message = {"job_id": job_id, "reset": False, "record": {"source": "/etc/passwd"}}
    assert api_app._run_queued_ingest_job(message) == "failed"
    assert api_app._find_ingest_job(job_id)["status"] == "failed"


@pytest.fixture
def kafka_mode(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> _Recording:
    monkeypatch.setattr(api_app._settings, "ingest_queue", "kafka")
    queue = _Recording()
    monkeypatch.setattr(api_app, "_ingest_queue", queue)
    return queue


def _as_other_replica(job_id: str, **fields: Any) -> None:
    """Write to the shared status file as another process would, leaving memory alone."""
    record = {**api_app._find_ingest_job(job_id), **fields, "owner": "replica-b"}
    api_app._persist_ingest_job(job_id, record)


def test_status_comes_from_the_replica_running_the_job(kafka_mode: _Recording) -> None:
    job_id = api_app._enqueue_ingest_job(
        api_app._settings.data_dir / "doc.txt", user_id="u1", label="doc"
    )
    _as_other_replica(job_id, status="running", files_done=1)

    view = api_app._ingest_job_view(job_id, api_app._find_ingest_job(job_id))
    assert view.status == "running"  # not "pending" from this process's copy, nor "interrupted"
    assert view.files_done == 1


def test_cancel_through_another_replica_stops_a_waiting_job(
    kafka_mode: _Recording, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = api_app._enqueue_ingest_job(api_app._settings.data_dir / "doc.txt", label="doc")
    _as_other_replica(job_id, cancel_requested=True)
    monkeypatch.setattr(api_app, "get_pipeline", lambda: pytest.fail("job should not run"))
    assert api_app._run_queued_ingest_job(kafka_mode.jobs[0]) == "cancelled"


def test_job_stopped_by_shutdown_waits_to_run_again(
    kafka_mode: _Recording, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = api_app._enqueue_ingest_job(api_app._settings.data_dir / "doc.txt", label="doc")
    api_app._interrupt_active_ingest_jobs()
    assert api_app._run_queued_ingest_job(kafka_mode.jobs[0]) == "interrupted"
    job = api_app._find_ingest_job(job_id)
    assert job["status"] == "pending"  # so "Run again" isn't offered on top of redelivery
    assert not api_app._can_cancel_here(job_id, {**job, "status": "running"})


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_KAFKA_BOOTSTRAP"), reason="RAG_TEST_KAFKA_BOOTSTRAP not set"
)
def test_real_broker_round_trip() -> None:
    topic = f"rag-test-{uuid.uuid4().hex[:8]}"
    ran: list[str] = []
    done = threading.Event()

    def run(job: dict[str, Any]) -> str:
        ran.append(job["job_id"])
        if len(ran) == 2:
            done.set()
        return "completed"

    queue = KafkaIngestQueue(
        run,
        bootstrap_servers=os.environ["RAG_TEST_KAFKA_BOOTSTRAP"],
        topic=topic,
        group_id=f"{topic}-group",
        timeout_seconds=10.0,
    )
    queue.submit({"job_id": "first", "reset": False, "record": {}})
    queue.submit({"job_id": "second", "reset": False, "record": {}})
    queue.ping()
    queue.start()
    try:
        assert done.wait(60), f"only ran {ran}"
    finally:
        queue.close(10)
    assert ran == ["first", "second"]
