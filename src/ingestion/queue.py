"""RabbitMQ transport for durable, one-at-a-time ingest jobs."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Any

import pika


class IngestQueueUnavailable(RuntimeError):
    """A job could not be durably handed to RabbitMQ."""


def _connection_parameters(url: str) -> pika.URLParameters:
    parameters = pika.URLParameters(url)
    # The worker keeps this connection alive while indexing runs in another thread.
    parameters.heartbeat = 30
    parameters.blocked_connection_timeout = 30
    parameters.connection_attempts = 3
    parameters.retry_delay = 1
    return parameters


def publish_job(url: str, queue_name: str, job_id: str) -> None:
    """Publish a persistent job and wait for the broker's confirmation."""
    connection: pika.BlockingConnection | None = None
    try:
        connection = pika.BlockingConnection(_connection_parameters(url))
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        channel.confirm_delivery()
        accepted = channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps({"job_id": job_id}).encode(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.DeliveryMode.Persistent,
            ),
            mandatory=True,
        )
        if not accepted:
            raise IngestQueueUnavailable("RabbitMQ did not confirm the ingest job.")
    except (pika.exceptions.AMQPError, OSError) as exc:
        raise IngestQueueUnavailable(f"RabbitMQ is unavailable: {exc}") from exc
    finally:
        if connection is not None and connection.is_open:
            connection.close()


def consume_jobs(
    url: str,
    queue_name: str,
    handler: Callable[[str, Event], str],
    shutdown: Event,
) -> None:
    """Consume with one in-flight job while servicing RabbitMQ heartbeats.

    The handler runs on one worker thread. The connection remains on this thread,
    where Pika requires it, and continues processing heartbeats during a long index.
    An interrupted job is negatively acknowledged so another worker can resume it.
    """
    connection = pika.BlockingConnection(_connection_parameters(url))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_qos(prefetch_count=1)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-index")
    active: dict[str, Any] = {"future": None, "tag": None}

    def received(
        _channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        _properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            payload = json.loads(body)
            job_id = payload["job_id"]
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("job_id must be a non-empty string")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return
        active["tag"] = method.delivery_tag
        active["future"] = executor.submit(handler, job_id, shutdown)

    channel.basic_consume(queue=queue_name, on_message_callback=received, auto_ack=False)
    try:
        while not shutdown.is_set() or active["future"] is not None:
            connection.process_data_events(time_limit=1)
            future: Future[str] | None = active["future"]
            if future is None or not future.done():
                continue
            tag = active["tag"]
            try:
                status = future.result()
            except Exception:
                channel.basic_nack(delivery_tag=tag, requeue=True)
            else:
                if status == "interrupted":
                    channel.basic_nack(delivery_tag=tag, requeue=True)
                else:
                    channel.basic_ack(delivery_tag=tag)
            active["future"] = None
            active["tag"] = None
            if shutdown.is_set():
                break
    finally:
        if channel.is_open:
            channel.stop_consuming()
        executor.shutdown(wait=True)
        if connection.is_open:
            connection.close()
