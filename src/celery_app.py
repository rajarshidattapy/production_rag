"""Optional Celery worker for async document indexing.

The API keeps its existing in-process, one-thread queue unless
``RAG_INGEST_QUEUE=celery``. Replica deployments can opt in and start this module as a
separate worker. Celery handles delivery; the API's ingest-job record remains the source of
truth for user-visible progress and cancellation.
"""

from __future__ import annotations

from typing import Any

from celery import Celery

from src.config import settings


class CeleryQueueUnavailableError(RuntimeError):
    """A job could not be handed durably to the Celery broker."""


celery_app = Celery("production_rag", broker=settings.celery_broker_url)
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    task_default_queue=settings.celery_task_queue,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds,
    },
)


@celery_app.task(name="rag.ingest", ignore_result=True)
def run_ingest_job(message: dict[str, Any]) -> str:
    """Run one serialized indexing job in the worker process."""
    # Import lazily so API-only processes do not initialize the FastAPI application merely
    # by loading Celery configuration.
    from src.api import _run_queued_ingest_job

    return _run_queued_ingest_job(message)


def submit_ingest_job(message: dict[str, Any]) -> None:
    """Publish a job and return only after the broker accepts it."""
    try:
        run_ingest_job.apply_async(args=[message], task_id=message["job_id"])
    except Exception as exc:
        raise CeleryQueueUnavailableError(str(exc)) from exc
