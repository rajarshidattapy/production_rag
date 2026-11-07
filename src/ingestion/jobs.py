"""Durable ingest-job state shared by API and worker processes.

RabbitMQ owns delivery; SQLite owns the user-visible job record. Keeping those
responsibilities separate lets API replicas poll and cancel work without sharing
Python memory with the indexing worker.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    user_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_user_created
    ON ingest_jobs(user_id, created_at DESC);
"""


class IngestJobStore:
    """Small process-safe job store backed by the existing application SQLite file."""

    def __init__(self, path: Path | str, *, max_jobs: int = 500) -> None:
        self.path = Path(path)
        self.max_jobs = max_jobs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, job_id: str, **fields: Any) -> dict[str, Any]:
        """Atomically merge fields into a job, preserving concurrent cancellation flags."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM ingest_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            record = json.loads(row["payload"]) if row else {}
            record.update(fields)
            status = str(record.get("status", "pending"))
            created_at = float(record.get("created_at", now))
            record["status"] = status
            record["created_at"] = created_at
            connection.execute(
                """
                INSERT INTO ingest_jobs (id, status, user_id, created_at, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    user_id = excluded.user_id,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    job_id,
                    status,
                    record.get("user_id"),
                    created_at,
                    now,
                    json.dumps(record, separators=(",", ":")),
                ),
            )
            self._prune(connection)
        return record

    def _prune(self, connection: sqlite3.Connection) -> None:
        """Bound terminal history without ever dropping queued or running work."""
        count = connection.execute("SELECT COUNT(*) FROM ingest_jobs").fetchone()[0]
        excess = max(count - self.max_jobs, 0)
        if excess == 0:
            return
        connection.execute(
            """
            DELETE FROM ingest_jobs WHERE id IN (
                SELECT id FROM ingest_jobs
                WHERE status NOT IN ('pending', 'running')
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (excess,),
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM ingest_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[tuple[str, dict[str, Any]]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, payload FROM ingest_jobs
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [(row["id"], json.loads(row["payload"])) for row in rows]

    def delete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM ingest_jobs WHERE id = ?", (job_id,))
