"""SQLite store for user accounts, login sessions, and chat history.

SQLite keeps the service free of another server to run, and is safe for concurrent
workers on one host. It must not sit on a network filesystem, so replicas on separate
hosts need this class reimplemented over a database server.

Each call opens its own connection, so one store can be shared across worker threads.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_TITLE = "New chat"
_TITLE_FROM_QUESTION_CHARS = 42

# OWASP's scrypt option that uses the least memory per login (16 MiB), which limits how
# much a burst of logins can allocate. Parameters are stored with each hash, so raising
# them later only affects new passwords.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 5
_SCRYPT_MAXMEM = 64 * 1024 * 1024

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    files TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    import_id TEXT,
    UNIQUE (user_id, import_id)
);
CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    healing TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""
_SCHEMA_VERSION = 1


class UsernameTakenError(ValueError):
    """The username already belongs to an account."""


class AccountLockedError(RuntimeError):
    """Too many failed logins; the account refuses attempts until the lockout ends."""


@dataclass(frozen=True)
class User:
    id: str
    username: str


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${derived.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    _, n, r, p, salt, expected = stored.split("$")
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=int(n),
        r=int(r),
        p=int(p),
        maxmem=_SCRYPT_MAXMEM,
    )
    return secrets.compare_digest(derived.hex(), expected)


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    return _hash_password(secrets.token_hex(16))


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _chat_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "files": json.loads(row["files"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class ChatStore:
    """Accounts, sessions, and per-user chats. Every chat read or write is scoped to a user."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Holds password hashes and private conversations: readable by the owner only.
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            if conn.execute("PRAGMA user_version").fetchone()[0] < _SCHEMA_VERSION:
                conn.executescript(_SCHEMA)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

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

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------

    def create_user(self, username: str, password: str) -> User:
        """Create an account. Raises ``UsernameTakenError`` if the name is in use."""
        user = User(id=uuid.uuid4().hex, username=_normalize_username(username))
        password_hash = _hash_password(password)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (user.id, user.username, password_hash, time.time()),
                )
        except sqlite3.IntegrityError:
            raise UsernameTakenError(user.username) from None
        return user

    def authenticate(self, username: str, password: str) -> User | None:
        """Return the user if the password is right, ``None`` if not.

        Takes the same time whether or not the username exists, so timing can't be used
        to find valid accounts. Raises ``AccountLockedError`` during a lockout.
        """
        name = _normalize_username(username)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
        if row is None:
            _verify_password(password, _dummy_password_hash())
            return None
        if row["locked_until"] > time.time():
            raise AccountLockedError(name)

        # Hashing is slow, so it happens outside any transaction.
        if not _verify_password(password, row["password_hash"]):
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET "
                    "locked_until = CASE WHEN failed_logins + 1 >= ? THEN ? "
                    "ELSE locked_until END, "
                    "failed_logins = CASE WHEN failed_logins + 1 >= ? THEN 0 "
                    "ELSE failed_logins + 1 END "
                    "WHERE id = ?",
                    (
                        MAX_FAILED_LOGINS,
                        time.time() + LOCKOUT_SECONDS,
                        MAX_FAILED_LOGINS,
                        row["id"],
                    ),
                )
            return None

        if row["failed_logins"]:
            with self._connect() as conn:
                conn.execute("UPDATE users SET failed_logins = 0 WHERE id = ?", (row["id"],))
        return User(id=row["id"], username=row["username"])

    def set_password(self, username: str, password: str) -> bool:
        """Change a password, clear any lockout, and sign the user out everywhere."""
        password_hash = _hash_password(password)
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE users SET password_hash = ?, failed_logins = 0, locked_until = 0 "
                "WHERE username = ? RETURNING id",
                (password_hash, _normalize_username(username)),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
        return True

    def list_usernames(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT username FROM users ORDER BY username").fetchall()
        return [row["username"] for row in rows]

    def delete_user(self, username: str) -> bool:
        """Delete an account along with its sessions and chat history."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE username = ?", (_normalize_username(username),)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, ttl_seconds: float) -> str:
        """Start a login session and return its token. Only a hash of the token is stored."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (_hash_token(token), user_id, now, now + ttl_seconds),
            )
        return token

    def user_for_session(self, token: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT users.id, users.username FROM sessions "
                "JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
                (_hash_token(token), time.time()),
            ).fetchone()
        return None if row is None else User(id=row["id"], username=row["username"])

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))

    # ------------------------------------------------------------------
    # Chats
    # ------------------------------------------------------------------

    def create_chat(self, user_id: str, title: str = DEFAULT_TITLE) -> dict[str, Any]:
        chat_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO chats (id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) RETURNING *",
                (chat_id, user_id, title, now, now),
            ).fetchone()
        return _chat_summary(row)

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        """Return the user's chats, most recently active first, without their messages."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC",
                (user_id,),
            ).fetchall()
        return [_chat_summary(row) for row in rows]

    def chat_exists(self, user_id: str, chat_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
            ).fetchone()
        return row is not None

    def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any] | None:
        """Return a chat with its messages, or ``None`` if this user has no such chat."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
            ).fetchone()
            if row is None:
                return None
            messages = conn.execute(
                "SELECT role, text, citations, healing, created_at FROM messages "
                "WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return {
            **_chat_summary(row),
            "messages": [
                {
                    "role": m["role"],
                    "text": m["text"],
                    "citations": json.loads(m["citations"]),
                    "healing": json.loads(m["healing"]) if m["healing"] else None,
                    "created_at": m["created_at"],
                }
                for m in messages
            ],
        }

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
            )
        return cursor.rowcount > 0

    def add_exchange(
        self,
        user_id: str,
        chat_id: str,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        healing: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Save a question and its answer together. An untitled chat is named after the question.

        Returns the updated chat summary, or ``None`` if this user has no such chat.
        """
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE chats SET updated_at = ?, "
                "title = CASE WHEN title = ? THEN ? ELSE title END "
                "WHERE id = ? AND user_id = ? RETURNING *",
                (now, DEFAULT_TITLE, question[:_TITLE_FROM_QUESTION_CHARS], chat_id, user_id),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "INSERT INTO messages (chat_id, role, text, created_at) VALUES (?, 'user', ?, ?)",
                (chat_id, question, now),
            )
            conn.execute(
                "INSERT INTO messages (chat_id, role, text, citations, healing, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?)",
                (
                    chat_id,
                    answer,
                    json.dumps(citations),
                    json.dumps(healing) if healing else None,
                    now,
                ),
            )
        return _chat_summary(row)

    def add_files(self, user_id: str, chat_id: str, filenames: list[str]) -> dict[str, Any] | None:
        """Record documents added to a chat, with a note in its history."""
        now = time.time()
        with self._connect() as conn:
            # Read-modify-write: take the write lock first so concurrent adds can't lose files.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT files FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
            ).fetchone()
            if row is None:
                return None
            files = list(dict.fromkeys([*json.loads(row["files"]), *filenames]))
            conn.execute(
                "UPDATE chats SET files = ?, updated_at = ? WHERE id = ?",
                (json.dumps(files), now, chat_id),
            )
            conn.execute(
                "INSERT INTO messages (chat_id, role, text, created_at) "
                "VALUES (?, 'assistant', ?, ?)",
                (chat_id, f"Added {', '.join(filenames)} to this chat.", now),
            )
        return self.get_chat(user_id, chat_id)

    def import_chat(
        self,
        user_id: str,
        import_id: str,
        title: str,
        files: list[str],
        created_at: float,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Copy in a chat saved elsewhere. Importing the same ``import_id`` again does nothing.

        Returns whether a new chat was created.
        """
        chat_id = uuid.uuid4().hex
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO chats "
                "(id, user_id, title, files, created_at, updated_at, import_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, user_id, title, json.dumps(files), created_at, created_at, import_id),
            )
            if cursor.rowcount == 0:
                return False
            conn.executemany(
                "INSERT INTO messages (chat_id, role, text, citations, healing, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chat_id,
                        m["role"],
                        m["text"],
                        json.dumps(m.get("citations") or []),
                        json.dumps(m["healing"]) if m.get("healing") else None,
                        created_at,
                    )
                    for m in messages
                ],
            )
        return True
