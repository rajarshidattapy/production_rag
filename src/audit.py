"""Audit trail of security-sensitive actions.

Each record is one JSON line saying when, which action, who did it and from where, what it
acted on, and whether it worked. Records never hold document text, questions, answers,
passwords, or tokens, so the log can be kept longer, and shown to more people, than the data.

API requests are recorded by ``AuditMiddleware``; routes add details with ``note``. Admin
commands run outside the API call ``record`` themselves.
"""

from __future__ import annotations

import contextvars
import getpass
import json
import logging
import logging.handlers
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.config import settings

# Not under "src", so records never mix into the application log or depend on its level.
_logger = logging.getLogger("rag.audit")
_logger.propagate = False
_logger.setLevel(logging.INFO)
_handler_target: str | None = None
_handler_lock = threading.Lock()

# Longer strings are cut, so a client can't grow a record without limit.
_MAX_TEXT = 200
_MAX_ITEMS = 20

# Requests recorded whatever their outcome, by method and route template.
ROUTE_ACTIONS: dict[tuple[str, str], str] = {
    ("POST", "/auth/register"): "auth.register",
    ("POST", "/auth/login"): "auth.login",
    ("POST", "/auth/logout"): "auth.logout",
    ("POST", "/upload"): "document.upload",
    ("POST", "/ingest"): "document.ingest",
    ("POST", "/ingest/async"): "document.ingest_queued",
    ("POST", "/ingest/jobs/{job_id}/cancel"): "ingest_job.cancel",
    ("POST", "/ingest/jobs/{job_id}/retry"): "ingest_job.retry",
    ("POST", "/query"): "query",
    ("POST", "/query/stream"): "query",
    ("POST", "/chats/{chat_id}/messages"): "query",
    ("GET", "/chunks/{chunk_id}"): "passage.read",
    ("POST", "/chats/import"): "chat.import",
    ("POST", "/chats/{chat_id}/files"): "chat.files_add",
    ("DELETE", "/chats/{chat_id}"): "chat.delete",
    ("POST", "/mcp"): "mcp.request",
}

# The record of the request being handled. Routes add to it; the middleware writes it.
_current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "audit_record", default=None
)


def _ensure_handler() -> bool:
    """Point the audit logger at the configured file. False when auditing is off."""
    global _handler_target
    if not settings.audit_log_enabled:
        return False
    target = str(settings.audit_log_file)
    with _handler_lock:
        if target == _handler_target:
            return True
        for old in list(_logger.handlers):
            _logger.removeHandler(old)
            old.close()
        if target == "-":
            handler: logging.Handler = logging.StreamHandler(sys.stdout)
        else:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Usernames and addresses are personal data: readable by this account only.
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
            # Reopens the file after logrotate moves it. Each record is one appended line,
            # so several processes can share the file.
            handler = logging.handlers.WatchedFileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
        _handler_target = target
    return True


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        return {str(k)[:_MAX_TEXT]: _clean(v) for k, v in list(value.items())[:_MAX_ITEMS]}
    if isinstance(value, list | tuple | set):
        return [_clean(v) for v in list(value)[:_MAX_ITEMS]]
    text = str(value)
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "..."


def record(
    action: str, *, outcome: str, actor: dict[str, Any] | None = None, **details: Any
) -> None:
    """Write one audit record. ``details`` must be identifiers and counts, never content."""
    if not _ensure_handler():
        return
    entry: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "action": action,
        "outcome": outcome,
        "actor": actor or {"type": "anonymous"},
    }
    entry.update((key, _clean(value)) for key, value in details.items() if value is not None)
    _logger.info(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))


def user_actor(user: Any) -> dict[str, Any]:
    return {"type": "user", "id": user.id, "username": user.username}


def cli_actor(name: str | None = None) -> dict[str, Any]:
    """An operator running an admin command on the server."""
    if not name:
        try:
            name = getpass.getuser()
        except (OSError, KeyError):
            name = "unknown"
    return {"type": "cli", "name": name}


def note(**details: Any) -> None:
    """Add details to the record of the request being handled. Does nothing outside one."""
    current = _current.get()
    if current is not None:
        current.update(details)


def note_user(user: Any) -> None:
    """Record who the request is from, once a route knows."""
    if user is not None:
        note(actor=user_actor(user))


def _outcome(status: int) -> str:
    if status < 400:
        return "success"
    return "denied" if status in (401, 403, 429) else "failure"


class AuditMiddleware(BaseHTTPMiddleware):
    """Record each request to a route in ROUTE_ACTIONS, and any request refused for access.

    Add it last, so it is outermost and sees the X-Request-ID set by the request ID middleware.
    """

    async def dispatch(self, request: Request, call_next):
        details: dict[str, Any] = {}
        token = _current.set(details)
        status = 500
        response = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            _current.reset(token)
            route = request.scope.get("route")
            action = ROUTE_ACTIONS.get((request.method, getattr(route, "path", None)))
            # A 401 elsewhere is usually just "not signed in yet"; a bad API key always counts.
            if action is None and (status == 403 or "denied_by" in details):
                action = "access.denied"
            if action is not None:
                headers = response.headers if response is not None else request.headers
                record(
                    action,
                    outcome=_outcome(status),
                    actor=details.pop("actor", None),
                    status=status,
                    method=request.method,
                    path=getattr(route, "path", None) or "unmatched",
                    params=dict(request.path_params) or None,
                    client=request.client.host if request.client else None,
                    request_id=headers.get("x-request-id"),
                    **details,
                )
