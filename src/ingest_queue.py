"""Where async ingest jobs wait their turn: in this process, or on a Kafka topic.

The local queue keeps the original behaviour: one worker thread, jobs in arrival order, and a
job still waiting when the process exits is lost. The Kafka queue keeps waiting jobs on a topic,
so they survive a restart and replicas share one queue. Jobs still run one at a time: the topic
has a single partition, so only one consumer in the group receives jobs at any moment.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Runs one job and returns its final status. A job that ends "interrupted" (the process is
# shutting down) is left on the Kafka topic so it runs again after a restart.
JobRunner = Callable[[dict[str, Any]], str]

_REQUEUE_STATUSES = {"interrupted"}
_POLL_SECONDS = 0.5


class QueueUnavailableError(RuntimeError):
    """The job could not be handed to the queue."""


class IngestQueue(Protocol):
    def start(self) -> None: ...

    def submit(self, job: dict[str, Any]) -> None: ...

    def ping(self) -> None: ...

    def close(self, timeout: float) -> None: ...


class LocalIngestQueue:
    """One worker thread in this process. Waiting jobs are lost when the process exits."""

    def __init__(self, run: JobRunner) -> None:
        self._run = run
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-ingest")

    def start(self) -> None:
        pass

    def submit(self, job: dict[str, Any]) -> None:
        self._executor.submit(self._run, job)

    def ping(self) -> None:
        pass

    def close(self, timeout: float) -> None:
        pass  # running jobs were told to stop; the thread exits with them


class KafkaIngestQueue:
    """Jobs wait on a single-partition Kafka topic and run on one consumer thread.

    A job's offset is committed only after it ends, so a job that was running or waiting when
    a process stopped is delivered again. Re-running a job is safe: text already indexed is
    skipped. While a job runs, the partition is paused and the consumer keeps polling, so a job
    longer than ``max.poll.interval.ms`` doesn't get the consumer evicted and the job handed to
    another replica.
    """

    def __init__(
        self,
        run: JobRunner,
        *,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        timeout_seconds: float,
        client_config: dict[str, str] | None = None,
        producer: Any = None,
        consumer: Any = None,
    ) -> None:
        self._run = run
        self._topic = topic
        self._timeout = timeout_seconds
        common: dict[str, Any] = {
            "bootstrap.servers": bootstrap_servers,
            # An auto-created topic gets the broker's default partition count; see _ensure_topic.
            "allow.auto.create.topics": False,
            **(client_config or {}),
        }
        self._producer_config = {
            **common,
            "acks": "all",
            "enable.idempotence": True,
            "message.timeout.ms": int(timeout_seconds * 1000),
        }
        self._consumer_config = {
            **common,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            # Keeps the partition where it is when another replica joins the group.
            "partition.assignment.strategy": "cooperative-sticky",
        }
        self._admin_config = {k: v for k, v in common.items() if k != "allow.auto.create.topics"}
        self._topic_checked = producer is not None or consumer is not None
        self._producer = producer
        self._consumer = consumer
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-ingest")

    def _get_producer(self) -> Any:
        with self._lock:
            if self._producer is None:
                from confluent_kafka import Producer

                self._ensure_topic_once()

                self._producer = Producer(self._producer_config)
            return self._producer

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._consumer is None:
            from confluent_kafka import Consumer

            with self._lock:
                self._ensure_topic_once()
            self._consumer = Consumer(self._consumer_config)
        self._consumer.subscribe([self._topic])
        self._thread = threading.Thread(
            target=self._consume, name="rag-ingest-consumer", daemon=True
        )
        self._thread.start()

    def _ensure_topic_once(self) -> None:
        if not self._topic_checked:
            self._ensure_topic()
            self._topic_checked = True

    def _ensure_topic(self) -> None:
        """Create the topic with one partition, and refuse one with more.

        A second partition would let two replicas run jobs at the same time, including a
        reset underneath another job. A broker that auto-creates topics uses its own default.
        """
        from confluent_kafka import KafkaError, KafkaException
        from confluent_kafka.admin import AdminClient, NewTopic

        admin = AdminClient(self._admin_config)
        future = admin.create_topics([NewTopic(self._topic, num_partitions=1)])[self._topic]
        try:
            future.result(self._timeout)
            logger.info("Created Kafka topic %s", self._topic)
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                # e.g. no permission to create topics: the operator creates it instead.
                logger.warning("Could not create Kafka topic %s: %s", self._topic, exc)
        topic = admin.list_topics(self._topic, timeout=self._timeout).topics.get(self._topic)
        if topic is not None and topic.error is None and len(topic.partitions) != 1:
            raise ValueError(
                f"Kafka topic {self._topic} has {len(topic.partitions)} partitions; ingest jobs "
                "need exactly one, so that they run one at a time."
            )

    def submit(self, job: dict[str, Any]) -> None:
        """Return once the broker has stored the job, or raise ``QueueUnavailableError``."""
        errors: list[Any] = []

        def delivered(err: Any, _msg: Any) -> None:
            if err is not None:
                errors.append(err)

        producer = self._get_producer()
        try:
            producer.produce(
                self._topic,
                key=job["job_id"].encode(),
                value=json.dumps(job).encode(),
                on_delivery=delivered,
            )
            remaining = producer.flush(self._timeout)
        except Exception as exc:
            raise QueueUnavailableError(str(exc)) from exc
        if errors or remaining:
            raise QueueUnavailableError(str(errors[0]) if errors else "delivery timed out")

    def ping(self) -> None:
        metadata = self._get_producer().list_topics(self._topic, timeout=self._timeout)
        topic = metadata.topics.get(self._topic)
        if topic is None or topic.error is not None:
            raise QueueUnavailableError(f"Kafka topic {self._topic} is unavailable")

    def close(self, timeout: float) -> None:
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Ingest consumer did not stop within %.0f s", timeout)
                return
        if self._consumer is not None:
            self._consumer.close()
        if self._producer is not None:
            self._producer.flush(self._timeout)

    def _consume(self) -> None:
        running: tuple[Any, Future[str]] | None = None
        while running is not None or not self._closing.is_set():
            try:
                running = self._step(running)
            except Exception:
                logger.exception("Ingest consumer error")
                self._closing.wait(_POLL_SECONDS)

    def _step(self, running: tuple[Any, Future[str]] | None) -> tuple[Any, Future[str]] | None:
        consumer = self._consumer
        if running is not None:
            msg, future = running
            if not future.done():
                self._hold(consumer.poll(_POLL_SECONDS))
                return running
            self._finish(msg, future)
            consumer.resume(consumer.assignment())
            return None

        msg = consumer.poll(_POLL_SECONDS)
        if msg is None:
            return None
        if msg.error() is not None:
            logger.warning("Ingest consumer: %s", msg.error())
            return None
        try:
            job = json.loads(msg.value())
        except ValueError:
            logger.error("Skipping unreadable ingest job at offset %s", msg.offset())
            consumer.commit(message=msg, asynchronous=False)
            return None
        consumer.pause(consumer.assignment())
        return msg, self._executor.submit(self._run, job)

    def _hold(self, msg: Any) -> None:
        """A message while paused means the assignment changed: put it back and pause again."""
        if msg is None or msg.error() is not None:
            return
        from confluent_kafka import TopicPartition

        self._consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
        self._consumer.pause(self._consumer.assignment())

    def _finish(self, msg: Any, future: Future[str]) -> None:
        try:
            status = future.result()
        except Exception:
            logger.exception("Ingest job at offset %s raised", msg.offset())
            status = "failed"
        if status in _REQUEUE_STATUSES:
            return  # not committed: delivered again after the restart
        self._consumer.commit(message=msg, asynchronous=False)
        from confluent_kafka import TopicPartition

        try:
            # _hold may have moved the position back to this job; don't run it twice.
            self._consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset() + 1))
        except Exception:
            logger.debug("Partition no longer assigned after job finished", exc_info=True)


def build_ingest_queue(settings: Any, run: JobRunner) -> IngestQueue:
    if settings.ingest_queue == "kafka":
        return KafkaIngestQueue(
            run,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            topic=settings.kafka_ingest_topic,
            group_id=settings.kafka_consumer_group,
            timeout_seconds=settings.kafka_timeout_seconds,
            client_config=settings.kafka_client_config,
        )
    return LocalIngestQueue(run)
