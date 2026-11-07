"""Tests for the audit log and for secrets read from a secrets manager's files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts import manage_users
from src import api as api_app
from src.chat import store as store_module
from src.config import Settings, settings
from src.generation import llm_client
from src.monitoring.config import MonitoringSettings
from src.secret_files import SecretFileError

PASSWORD = "correct horse battery"
QUESTION = "What does the confidential merger memo say?"


@pytest.fixture(autouse=True)
def _fast_hashing_and_fresh_pipeline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(store_module, "_SCRYPT_N", 2**4)
    monkeypatch.setattr(store_module, "_SCRYPT_P", 1)
    store_module._dummy_password_hash.cache_clear()
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()
    store_module._dummy_password_hash.cache_clear()


def _audit_records() -> list[dict[str, Any]]:
    path = settings.audit_log_file
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _client(username: str = "alice") -> TestClient:
    client = TestClient(api_app.app)
    response = client.post("/auth/register", json={"username": username, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return client


class TestAuditLog:
    def test_sign_ins_record_who_and_whether_it_worked_but_never_the_password(self) -> None:
        _client("alice")
        anon = TestClient(api_app.app)
        anon.post("/auth/login", json={"username": "alice", "password": "wrong password!"})
        anon.post("/auth/login", json={"username": "alice", "password": PASSWORD})

        register, failed, signed_in = _audit_records()
        assert register["action"] == "auth.register" and register["outcome"] == "success"
        assert register["actor"]["username"] == "alice"
        assert failed["action"] == "auth.login" and failed["outcome"] == "denied"
        assert failed["actor"] == {"type": "anonymous"} and failed["username"] == "alice"
        assert signed_in["outcome"] == "success" and signed_in["actor"]["type"] == "user"
        assert signed_in["request_id"] and signed_in["client"]
        log = settings.audit_log_file.read_text(encoding="utf-8")
        assert PASSWORD not in log and "wrong password!" not in log

    def test_queries_record_their_size_not_their_text(self, tmp_path: Path) -> None:
        doc = tmp_path / "memo.txt"
        doc.write_text("The merger closes in March. " * 20, encoding="utf-8")
        client = _client()
        assert client.post("/ingest", json={"source": str(doc)}).status_code == 200
        with patch.object(
            api_app.get_pipeline().generator,
            "generate_async",
            AsyncMock(return_value="It closes in March [1]."),
        ):
            response = client.post("/query", json={"question": QUESTION, "self_heal": False})
        assert response.status_code == 200

        ingest, query = _audit_records()[1:]
        assert ingest["action"] == "document.ingest" and ingest["source"] == "memo.txt"
        assert ingest["chunks_ingested"] >= 1 and ingest["actor"]["username"] == "alice"
        assert query["action"] == "query" and query["question_chars"] == len(QUESTION)
        log = settings.audit_log_file.read_text(encoding="utf-8")
        assert "merger" not in log and "March" not in log

    def test_uploads_record_a_fingerprint_not_the_bytes(self) -> None:
        content = b"%PDF-1.4 secret board minutes"
        response = TestClient(api_app.app).post(
            "/upload", params={"filename": "minutes.pdf"}, content=content
        )
        assert response.status_code == 200
        (upload,) = _audit_records()
        assert upload["action"] == "document.upload" and upload["filename"] == "minutes.pdf"
        assert upload["size_bytes"] == len(content) and len(upload["sha256"]) == 64
        assert "board minutes" not in settings.audit_log_file.read_text(encoding="utf-8")

    def test_deleting_a_chat_is_recorded(self) -> None:
        client = _client()
        chat_id = client.post("/chats", json={}).json()["id"]
        assert client.delete(f"/chats/{chat_id}").status_code == 204
        assert client.delete(f"/chats/{chat_id}").status_code == 404

        deleted, missing = _audit_records()[1:]
        assert deleted["action"] == "chat.delete" and deleted["params"] == {"chat_id": chat_id}
        assert deleted["outcome"] == "success" and missing["outcome"] == "failure"

    def test_a_bad_api_key_is_recorded_on_any_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api_app, "_RAG_API_KEY", "deployment-key")
        client = TestClient(api_app.app)
        assert client.get("/stats", headers={"Authorization": "Bearer guess"}).status_code == 403
        assert client.get("/stats").status_code == 401
        assert client.get("/healthz").status_code == 200

        wrong, missing = _audit_records()
        assert wrong["action"] == "access.denied" and wrong["denied_by"] == "invalid_api_key"
        assert missing["denied_by"] == "missing_api_key" and missing["path"] == "/stats"
        assert "guess" not in settings.audit_log_file.read_text(encoding="utf-8")

    def test_reading_without_signing_in_is_not_an_access_failure(self) -> None:
        assert TestClient(api_app.app).get("/auth/me").status_code == 401
        assert _audit_records() == []

    def test_admin_commands_are_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(manage_users, "_read_password", lambda: PASSWORD)
        manage_users.main(["create", "bob"])
        with pytest.raises(SystemExit):
            manage_users.main(["delete", "nobody"])

        created, missing = _audit_records()
        assert created["action"] == "admin.user_create" and created["outcome"] == "success"
        assert created["actor"]["type"] == "cli" and created["username"] == "bob"
        assert missing["action"] == "admin.user_delete" and missing["outcome"] == "failure"

    def test_turning_it_off_writes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "audit_log_enabled", False)
        _client()
        assert not settings.audit_log_file.exists()


class TestSecretFiles:
    @pytest.fixture
    def secrets_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        directory = tmp_path / "secrets"
        directory.mkdir()
        monkeypatch.setenv("RAG_SECRETS_DIR", str(directory))
        return directory

    def test_a_mounted_secret_wins_over_the_environment(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (secrets_dir / "openai_api_key").write_text("sk-from-file\n", encoding="utf-8")
        (secrets_dir / "rag_api_key").write_text("deployment-key", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("RAG_REDIS_URL", "redis://from-env:6379/0")

        loaded = Settings(_env_file=None)
        assert loaded.openai_api_key == "sk-from-file"
        assert loaded.api_key == "deployment-key"
        assert loaded.redis_url == "redis://from-env:6379/0"  # no file, so the variable
        assert llm_client._openai_api_key() == "sk-from-file"
        assert "sk-from-file" not in repr(loaded)

    def test_a_file_variable_names_the_secret_explicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_file = tmp_path / "anthropic.key"
        key_file.write_text("sk-ant-from-file", encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key_file))
        assert Settings(_env_file=None).anthropic_api_key == "sk-ant-from-file"
        assert llm_client.anthropic_client_kwargs() == {"api_key": "sk-ant-from-file"}

    def test_a_missing_named_file_stops_start_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_API_KEY_FILE", str(tmp_path / "missing"))
        with pytest.raises(SecretFileError, match="RAG_API_KEY"):
            Settings(_env_file=None)

    def test_an_empty_file_falls_back_to_the_environment(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (secrets_dir / "openai_api_key").write_text("\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert Settings(_env_file=None).openai_api_key == "sk-from-env"

    def test_monitoring_keys_come_from_files_too(self, secrets_dir: Path) -> None:
        (secrets_dir / "monitor_langfuse_secret_key").write_text("sk-lf-file", encoding="utf-8")
        assert MonitoringSettings(_env_file=None).langfuse_secret_key == "sk-lf-file"
