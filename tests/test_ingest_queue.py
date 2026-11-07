"""RabbitMQ transport and cross-process ingest state tests."""

from __future__ import annotations

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pika
import pytest

from src.ingestion.jobs import IngestJobStore
from src.ingestion.queue import IngestQueueUnavailable, consume_jobs, publish_job


class FakeConnection:
    def __init__(self, channel: MagicMock) -> None:
        self._channel = channel
        self.is_open = True

    def channel(self) -> MagicMock:
        return self._channel

    def close(self) -> None:
        self.is_open = False


def test_publish_declares_durable_queue_and_confirms_persistent_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = MagicMock()
    channel.basic_publish.return_value = True
    connection = FakeConnection(channel)
    monkeypatch.setattr(pika, "BlockingConnection", lambda _parameters: connection)

    publish_job("amqp://example", "rag.ingest", "job-1")

    channel.queue_declare.assert_called_once_with(queue="rag.ingest", durable=True)
    channel.confirm_delivery.assert_called_once_with()
    call = channel.basic_publish.call_args.kwargs
    assert json.loads(call["body"]) == {"job_id": "job-1"}
    assert call["mandatory"] is True
    assert call["properties"].delivery_mode == pika.DeliveryMode.Persistent.value
    assert connection.is_open is False


def test_publish_surfaces_broker_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_parameters):
        raise pika.exceptions.AMQPConnectionError("offline")

    monkeypatch.setattr(pika, "BlockingConnection", fail)
    with pytest.raises(IngestQueueUnavailable, match="unavailable"):
        publish_job("amqp://example", "rag.ingest", "job-1")


class FakeConsumerConnection(FakeConnection):
    def __init__(self, channel: MagicMock, shutdown: Event, body: bytes) -> None:
        super().__init__(channel)
        self.shutdown = shutdown
        self.body = body
        self.delivered = False

    def process_data_events(self, time_limit: int) -> None:
        assert time_limit == 1
        if not self.delivered:
            self.delivered = True
            method = SimpleNamespace(delivery_tag=7)
            self._channel.callback(self._channel, method, None, self.body)


@pytest.mark.parametrize(
    ("status", "ack_method", "requeue"),
    [("completed", "basic_ack", None), ("interrupted", "basic_nack", True)],
)
def test_consumer_acknowledges_completion_and_requeues_interruption(
    monkeypatch: pytest.MonkeyPatch, status: str, ack_method: str, requeue: bool | None
) -> None:
    shutdown = Event()
    channel = MagicMock()
    channel.is_open = True

    def basic_consume(**kwargs) -> None:
        channel.callback = kwargs["on_message_callback"]

    channel.basic_consume.side_effect = basic_consume
    connection = FakeConsumerConnection(channel, shutdown, json.dumps({"job_id": "job-1"}).encode())
    monkeypatch.setattr(pika, "BlockingConnection", lambda _parameters: connection)

    def handler(job_id: str, event: Event) -> str:
        assert job_id == "job-1"
        assert event is shutdown
        shutdown.set()
        return status

    consume_jobs("amqp://example", "rag.ingest", handler, shutdown)

    if ack_method == "basic_ack":
        channel.basic_ack.assert_called_once_with(delivery_tag=7)
        channel.basic_nack.assert_not_called()
    else:
        channel.basic_nack.assert_called_once_with(delivery_tag=7, requeue=requeue)
        channel.basic_ack.assert_not_called()
    channel.basic_qos.assert_called_once_with(prefetch_count=1)


def test_job_store_merge_preserves_cancel_flag(tmp_path) -> None:
    store = IngestJobStore(tmp_path / "app.db")
    store.put("job-1", status="running", source="docs")
    store.put("job-1", cancel_requested=True)
    result = store.put("job-1", files_done=3, chunks_ingested=10)

    assert result["cancel_requested"] is True
    assert result["files_done"] == 3
    assert store.get("job-1") == result
