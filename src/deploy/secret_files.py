"""Secrets read from files mounted by a secrets manager, before environment variables.

Docker and Compose secrets, Kubernetes secrets, Vault Agent, and the AWS, GCP, and Azure
secrets-store CSI drivers all hand secrets to a container as files. For a variable such as
``OPENAI_API_KEY`` the value comes from the first of:

1. the file named by ``OPENAI_API_KEY_FILE``,
2. ``openai_api_key`` in ``RAG_SECRETS_DIR`` (``/run/secrets`` unless set),
3. the ``OPENAI_API_KEY`` variable, then ``.env``.

``RAG_SECRETS_DIR`` and the ``*_FILE`` variables are read from the process environment only,
since ``.env`` is one of the places the secrets would otherwise come from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

DEFAULT_SECRETS_DIR = Path("/run/secrets")


class SecretFileError(RuntimeError):
    """A secret file was configured but couldn't be read."""


def secrets_dir() -> Path:
    return Path(os.environ.get("RAG_SECRETS_DIR") or DEFAULT_SECRETS_DIR)


def _read(name: str, path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        # The message names the file, never its contents.
        raise SecretFileError(f"Cannot read the secret for {name} from {path}: {exc}") from None
    return value or None


def read_secret_file(name: str) -> str | None:
    """The secret a file provides for variable ``name``, or None if no file does.

    A file that exists but can't be read is an error, not a fallback to the environment:
    the operator meant that file to be used.
    """
    if explicit := os.environ.get(f"{name}_FILE"):
        return _read(name, Path(explicit))
    path = secrets_dir() / name.lower()
    return _read(name, path) if path.is_file() else None


def secret_value(name: str, fallback: str | None = None) -> str | None:
    """A secret looked up now, so a test or a rotated file is seen without a restart."""
    return read_secret_file(name) or os.environ.get(name) or fallback


class SecretFilesSource(PydanticBaseSettingsSource):
    """Settings source giving the listed fields their secret file, ahead of the environment."""

    def __init__(self, settings_cls: type[BaseSettings], fields: tuple[str, ...]) -> None:
        super().__init__(settings_cls)
        self._fields = fields

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        prefix = self.config.get("env_prefix", "")
        values: dict[str, Any] = {}
        for field_name in self._fields:
            alias = self.settings_cls.model_fields[field_name].validation_alias
            key = alias if isinstance(alias, str) else field_name
            env_name = alias if isinstance(alias, str) else f"{prefix}{field_name}".upper()
            if (value := read_secret_file(env_name)) is not None:
                values[key] = value
        return values
