"""Durable prompt versions, atomic activation, and production rollback.

Prompt text is immutable and addressed by its SHA-256 hash.  The active pointer is the only
mutable state, and changing it plus writing its audit event happens in one SQLite transaction.
Every read opens the database afresh, so another process (including the management CLI) can
activate or roll back a prompt without restarting the API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import string
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_PROMPT_CHARS = 100_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    name TEXT NOT NULL,
    hash TEXT NOT NULL,
    prompt TEXT NOT NULL,
    required_fields TEXT NOT NULL,
    created_at REAL NOT NULL,
    created_by TEXT NOT NULL,
    PRIMARY KEY (name, hash)
);
CREATE TABLE IF NOT EXISTS active_prompts (
    name TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT NOT NULL,
    FOREIGN KEY (name, hash) REFERENCES prompt_versions(name, hash)
);
CREATE TABLE IF NOT EXISTS prompt_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    from_hash TEXT,
    to_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompt_activations_name
    ON prompt_activations(name, id DESC);
"""


class PromptError(ValueError):
    """Base error for invalid prompt-management operations."""


class PromptNotFoundError(PromptError):
    """The requested prompt name or version does not exist."""


class PromptValidationError(PromptError):
    """A prompt does not preserve its required template placeholders."""


@dataclass(frozen=True)
class PromptVersion:
    name: str
    hash: str
    prompt: str
    required_fields: tuple[str, ...]
    created_at: float
    created_by: str
    active: bool = False


@dataclass(frozen=True)
class PromptActivation:
    id: int
    name: str
    from_hash: str | None
    to_hash: str
    action: str
    actor: str
    created_at: float


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def validate_prompt(prompt: str, required_fields: tuple[str, ...]) -> None:
    """Reject empty/runaway templates and any placeholder-contract change."""
    if not prompt.strip():
        raise PromptValidationError("Prompt must not be empty.")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise PromptValidationError(f"Prompt exceeds {_MAX_PROMPT_CHARS:,} characters.")

    fields: set[str] = set()
    try:
        parsed = string.Formatter().parse(prompt)
        for _, field, _, _ in parsed:
            if field is None:
                continue
            # Attribute/index traversal is unnecessary here and makes validation ambiguous.
            if not field or "." in field or "[" in field or "]" in field:
                raise PromptValidationError(f"Unsupported placeholder: {field!r}.")
            fields.add(field)
    except ValueError as exc:
        raise PromptValidationError(f"Malformed template braces: {exc}") from exc

    expected = set(required_fields)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise PromptValidationError("Placeholder contract changed: " + "; ".join(details) + ".")


class PromptStore:
    """SQLite-backed immutable prompt versions and their active pointers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _check_name(name: str) -> None:
        if not _NAME_RE.fullmatch(name):
            raise PromptValidationError(
                "Prompt names must start with a lowercase letter and contain only "
                "lowercase letters, digits, dots, underscores, or hyphens."
            )

    def register(
        self,
        name: str,
        prompt: str,
        required_fields: tuple[str, ...],
        *,
        actor: str = "application",
    ) -> str:
        """Add an immutable version; initialize new names without replacing active versions."""
        self._check_name(name)
        validate_prompt(prompt, required_fields)
        digest = prompt_hash(prompt)
        now = time.time()
        fields_json = json.dumps(sorted(required_fields), separators=(",", ":"))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO prompt_versions "
                "(name, hash, prompt, required_fields, created_at, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, digest, prompt, fields_json, now, actor),
            )
            current = conn.execute(
                "SELECT hash FROM active_prompts WHERE name = ?", (name,)
            ).fetchone()
            if current is None:
                conn.execute(
                    "INSERT INTO active_prompts (name, hash, updated_at, updated_by) "
                    "VALUES (?, ?, ?, ?)",
                    (name, digest, now, actor),
                )
                conn.execute(
                    "INSERT INTO prompt_activations "
                    "(name, from_hash, to_hash, action, actor, created_at) "
                    "VALUES (?, NULL, ?, 'initialize', ?, ?)",
                    (name, digest, actor, now),
                )
        return digest

    def _matching_hash(self, conn: sqlite3.Connection, name: str, version: str) -> str:
        rows = conn.execute(
            "SELECT hash FROM prompt_versions WHERE name = ? AND hash LIKE ? ORDER BY hash",
            (name, f"{version}%"),
        ).fetchall()
        if not rows:
            raise PromptNotFoundError(f"No version {version!r} exists for {name}.")
        if len(rows) > 1:
            raise PromptError(f"Version prefix {version!r} is ambiguous for {name}.")
        return str(rows[0]["hash"])

    def activate(
        self,
        name: str,
        version: str,
        required_fields: tuple[str, ...],
        *,
        actor: str,
        action: str = "activate",
    ) -> str:
        """Atomically point a prompt name at a validated immutable version."""
        self._check_name(name)
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._activate_locked(
                conn, name, version, required_fields, actor=actor, action=action, now=now
            )

    def _activate_locked(
        self,
        conn: sqlite3.Connection,
        name: str,
        version: str,
        required_fields: tuple[str, ...],
        *,
        actor: str,
        action: str,
        now: float,
    ) -> str:
        """Move an active pointer while the caller holds a write transaction."""
        digest = self._matching_hash(conn, name, version)
        row = conn.execute(
            "SELECT prompt FROM prompt_versions WHERE name = ? AND hash = ?",
            (name, digest),
        ).fetchone()
        assert row is not None
        validate_prompt(str(row["prompt"]), required_fields)
        current = conn.execute("SELECT hash FROM active_prompts WHERE name = ?", (name,)).fetchone()
        previous = str(current["hash"]) if current else None
        if previous == digest:
            return digest
        conn.execute(
            "INSERT INTO active_prompts (name, hash, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "hash = excluded.hash, updated_at = excluded.updated_at, "
            "updated_by = excluded.updated_by",
            (name, digest, now, actor),
        )
        conn.execute(
            "INSERT INTO prompt_activations "
            "(name, from_hash, to_hash, action, actor, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, previous, digest, action, actor, now),
        )
        return digest

    def rollback(
        self,
        name: str,
        required_fields: tuple[str, ...],
        *,
        actor: str,
        to_version: str | None = None,
    ) -> str:
        """Activate an explicit version, or the version active immediately beforehand."""
        self._check_name(name)
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if to_version is None:
                current = conn.execute(
                    "SELECT hash FROM active_prompts WHERE name = ?", (name,)
                ).fetchone()
                if current is None:
                    raise PromptNotFoundError(f"No active prompt named {name}.")
                event = conn.execute(
                    "SELECT from_hash FROM prompt_activations "
                    "WHERE name = ? AND to_hash = ? AND from_hash IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1",
                    (name, current["hash"]),
                ).fetchone()
                if event is None:
                    raise PromptNotFoundError(f"No previous active version exists for {name}.")
                to_version = str(event["from_hash"])
            return self._activate_locked(
                conn,
                name,
                to_version,
                required_fields,
                actor=actor,
                action="rollback",
                now=now,
            )

    def resolve(self, name: str, fallback: str, required_fields: tuple[str, ...]) -> PromptVersion:
        """Register the code fallback, then return the currently active validated version."""
        self.register(name, fallback, required_fields, actor="application")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT v.*, 1 AS active FROM prompt_versions v "
                "JOIN active_prompts a ON a.name = v.name AND a.hash = v.hash "
                "WHERE v.name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise PromptNotFoundError(f"No active prompt named {name}.")
        version = self._version(row)
        validate_prompt(version.prompt, required_fields)
        return version

    def current(self, name: str) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT v.*, 1 AS active FROM prompt_versions v "
                "JOIN active_prompts a ON a.name = v.name AND a.hash = v.hash "
                "WHERE v.name = ?",
                (name,),
            ).fetchone()
        return self._version(row) if row else None

    def versions(self, name: str | None = None) -> list[PromptVersion]:
        where = "WHERE v.name = ?" if name is not None else ""
        params = (name,) if name is not None else ()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT v.*, CASE WHEN a.hash = v.hash THEN 1 ELSE 0 END AS active "
                "FROM prompt_versions v LEFT JOIN active_prompts a ON a.name = v.name "
                f"{where} ORDER BY v.name, v.created_at DESC, v.hash",
                params,
            ).fetchall()
        return [self._version(row) for row in rows]

    def history(self, name: str) -> list[PromptActivation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_activations WHERE name = ? ORDER BY id DESC", (name,)
            ).fetchall()
        return [PromptActivation(**dict(row)) for row in rows]

    @staticmethod
    def _version(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            name=str(row["name"]),
            hash=str(row["hash"]),
            prompt=str(row["prompt"]),
            required_fields=tuple(json.loads(row["required_fields"])),
            created_at=float(row["created_at"]),
            created_by=str(row["created_by"]),
            active=bool(row["active"]),
        )


_stores: dict[Path, PromptStore] = {}
_stores_lock = threading.Lock()
_fallback_warnings: set[Path] = set()


def get_prompt_store(path: str | Path | None = None) -> PromptStore:
    if path is None:
        from src.config import settings

        path = settings.prompt_db_file
    resolved = Path(path).resolve()
    with _stores_lock:
        store = _stores.get(resolved)
        if store is None:
            store = PromptStore(resolved)
            _stores[resolved] = store
        return store


def resolve_prompt(name: str, fallback: str, required_fields: tuple[str, ...]) -> str:
    """Return an active prompt, falling back to code if runtime storage is unavailable."""
    try:
        store = get_prompt_store()
        version = store.resolve(name, fallback, required_fields)
        _fallback_warnings.discard(store.path.resolve())
        return version.prompt
    except Exception as exc:
        from src.config import settings

        path = settings.prompt_db_file.resolve()
        if path not in _fallback_warnings:
            logger.error("Prompt store unavailable; using built-in prompts: %s", exc)
            _fallback_warnings.add(path)
        return fallback
