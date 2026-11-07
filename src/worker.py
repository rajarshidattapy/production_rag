"""RabbitMQ-backed indexing worker.

Run with ``python -m src.worker``. One process accepts one message at a time;
scale the service deliberately only after measuring write contention in Chroma.
"""

from __future__ import annotations

import logging
import signal
from threading import Event

import pika

from src.api import _run_ingest_job
from src.config import settings
from src.ingestion.queue import consume_jobs

logger = logging.getLogger(__name__)


def main() -> None:
    if not settings.rabbitmq_url:
        raise SystemExit("RAG_RABBITMQ_URL must be set for the indexing worker.")

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    shutdown = Event()

    def stop(_signum: int, _frame: object) -> None:
        logger.info("Indexing worker shutdown requested")
        shutdown.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while not shutdown.is_set():
        try:
            logger.info("Waiting for jobs on RabbitMQ queue %s", settings.ingest_queue_name)
            consume_jobs(
                settings.rabbitmq_url,
                settings.ingest_queue_name,
                _run_ingest_job,
                shutdown,
            )
        except (pika.exceptions.AMQPError, OSError) as exc:
            if shutdown.is_set():
                break
            logger.warning("RabbitMQ connection lost: %s; retrying in 5 seconds", exc)
            shutdown.wait(5)


if __name__ == "__main__":
    main()
