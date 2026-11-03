"""Tests for user accounts and server-side chat history."""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts import manage_users
from src import api as api_app
from src.chat import store as store_module
from src.chat.store import AccountLockedError, ChatStore, UsernameTakenError
from src.config import settings

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _fast_hashing_and_fresh_pipeline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(store_module, "_SCRYPT_N", 2**4)
    monkeypatch.setattr(store_module, "_SCRYPT_P", 1)
    store_module._dummy_password_hash.cache_clear()
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()
    store_module._dummy_password_hash.cache_clear()


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chat.db")


def _client(username: str | None = "alice") -> TestClient:
    client = TestClient(api_app.app)
    if username:
        response = client.post("/auth/register", json={"username": username, "password": PASSWORD})
        assert response.status_code == 201, response.text
    return client


def _stub_llm(answer: str = "Stub answer [1].") -> Any:
    return patch.object(
        api_app.get_pipeline().generator, "generate_async", AsyncMock(return_value=answer)
    )


def _ingest(client: TestClient, source: Path) -> None:
    assert client.post("/ingest", json={"source": str(source)}).status_code == 200


def _ask(client: TestClient, chat_id: str, question: str = "What is RAG?", **extra: Any) -> Any:
    body = {"question": question, "self_heal": False, **extra}
    return client.post(f"/chats/{chat_id}/messages", json=body)


class TestPasswordStorage:
    def test_password_is_salted_and_hashed(self, store: ChatStore) -> None:
        store.create_user("alice", PASSWORD)
        store.create_user("bob", PASSWORD)
        with store._connect() as conn:
            hashes = [r[0] for r in conn.execute("SELECT password_hash FROM users")]
        assert all(PASSWORD not in h and h.startswith("scrypt$") for h in hashes)
        assert hashes[0] != hashes[1]

    def test_real_scrypt_parameters_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(store_module, "_SCRYPT_N", 2**14)
        monkeypatch.setattr(store_module, "_SCRYPT_P", 5)
        stored = store_module._hash_password(PASSWORD)
        assert stored.startswith("scrypt$16384$8$5$")
        assert store_module._verify_password(PASSWORD, stored)
        assert not store_module._verify_password("wrong", stored)

    def test_hash_parameters_can_change_without_breaking_old_passwords(
        self, store: ChatStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.create_user("alice", PASSWORD)
        monkeypatch.setattr(store_module, "_SCRYPT_N", 2**5)
        assert store.authenticate("alice", PASSWORD) is not None

    def test_session_tokens_are_not_stored_in_the_clear(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        token = store.create_session(user.id, 60)
        with store._connect() as conn:
            stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
        assert token != stored
        assert store.user_for_session(token) == user

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_database_is_private_to_the_owner(self, store: ChatStore) -> None:
        assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


class TestAccountRules:
    def test_usernames_are_case_insensitive(self, store: ChatStore) -> None:
        store.create_user("Alice", PASSWORD)
        with pytest.raises(UsernameTakenError):
            store.create_user("ALICE", PASSWORD)
        user = store.authenticate("aLiCe", PASSWORD)
        assert user is not None and user.username == "alice"

    def test_lockout_after_repeated_failures(self, store: ChatStore) -> None:
        store.create_user("alice", PASSWORD)
        for _ in range(store_module.MAX_FAILED_LOGINS):
            assert store.authenticate("alice", "wrong") is None
        # The right password doesn't help during a lockout.
        with pytest.raises(AccountLockedError):
            store.authenticate("alice", PASSWORD)

    def test_lockout_ends(self, store: ChatStore, monkeypatch: pytest.MonkeyPatch) -> None:
        store.create_user("alice", PASSWORD)
        for _ in range(store_module.MAX_FAILED_LOGINS):
            store.authenticate("alice", "wrong")
        real_time = time.time
        monkeypatch.setattr(
            store_module.time, "time", lambda: real_time() + store_module.LOCKOUT_SECONDS + 1
        )
        assert store.authenticate("alice", PASSWORD) is not None

    def test_success_resets_the_failure_count(self, store: ChatStore) -> None:
        store.create_user("alice", PASSWORD)
        for _ in range(3):
            for _ in range(store_module.MAX_FAILED_LOGINS - 1):
                store.authenticate("alice", "wrong")
            assert store.authenticate("alice", PASSWORD) is not None

    def test_unknown_user_still_does_the_hashing_work(self, store: ChatStore) -> None:
        with patch.object(
            store_module, "_verify_password", wraps=store_module._verify_password
        ) as verify:
            assert store.authenticate("nobody", PASSWORD) is None
        verify.assert_called_once()

    def test_set_password_ends_sessions_and_lockout(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        token = store.create_session(user.id, 60)
        for _ in range(store_module.MAX_FAILED_LOGINS):
            store.authenticate("alice", "wrong")

        assert store.set_password("alice", "a whole new password") is True

        assert store.user_for_session(token) is None
        assert store.authenticate("alice", "a whole new password") is not None
        assert store.set_password("nobody", "whatever password") is False

    def test_expired_session_is_rejected_and_purged(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        expired = store.create_session(user.id, -1)
        assert store.user_for_session(expired) is None
        store.create_session(user.id, 60)
        with store._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_deleting_a_user_deletes_their_history(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        token = store.create_session(user.id, 60)
        chat = store.create_chat(user.id)
        store.add_exchange(user.id, chat["id"], "q", "a", [], None)

        assert store.delete_user("alice") is True

        assert store.user_for_session(token) is None
        with store._connect() as conn:
            for table in ("chats", "messages", "sessions"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


class TestChatStore:
    def test_concurrent_writers_do_not_conflict(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        chat = store.create_chat(user.id)
        errors: list[Exception] = []

        def write(n: int) -> None:
            try:
                for i in range(10):
                    store.add_exchange(user.id, chat["id"], f"q{n}-{i}", "a", [], None)
                    store.add_files(user.id, chat["id"], [f"{n}-{i}.pdf"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        loaded = store.get_chat(user.id, chat["id"])
        assert loaded is not None
        assert len(loaded["messages"]) == 6 * 10 * 3
        assert len(loaded["files"]) == 6 * 10  # no add lost another's file

    def test_exchange_is_all_or_nothing(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        chat = store.create_chat(user.id)
        with pytest.raises(TypeError):
            store.add_exchange(user.id, chat["id"], "q", "a", [object()], None)  # type: ignore[list-item]
        loaded = store.get_chat(user.id, chat["id"])
        assert loaded is not None
        assert loaded["messages"] == []
        assert loaded["title"] == store_module.DEFAULT_TITLE

    def test_title_comes_from_first_question_only(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        chat = store.create_chat(user.id)
        store.add_exchange(user.id, chat["id"], "x" * 100, "a", [], None)
        store.add_exchange(user.id, chat["id"], "second question", "a", [], None)
        titled = store.create_chat(user.id, "report.pdf")
        store.add_exchange(user.id, titled["id"], "question", "a", [], None)
        assert store.get_chat(user.id, chat["id"])["title"] == "x" * 42  # type: ignore[index]
        assert store.get_chat(user.id, titled["id"])["title"] == "report.pdf"  # type: ignore[index]

    def test_reopening_keeps_data_and_schema(self, store: ChatStore) -> None:
        user = store.create_user("alice", PASSWORD)
        chat = store.create_chat(user.id)
        reopened = ChatStore(store.path)
        assert reopened.list_chats(user.id)[0]["id"] == chat["id"]


class TestAuthApi:
    def test_register_signs_in_with_a_protected_cookie(self) -> None:
        client = TestClient(api_app.app)
        response = client.post("/auth/register", json={"username": "Alice", "password": PASSWORD})
        assert response.status_code == 201
        assert response.json() == {"username": "alice"}
        cookie = response.headers["set-cookie"].lower()
        assert "rag_session=" in cookie
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "secure" not in cookie  # plain HTTP here
        assert client.get("/auth/me").json() == {"username": "alice"}

    def test_cookie_is_secure_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "session_cookie_secure", True)
        response = TestClient(api_app.app).post(
            "/auth/register", json={"username": "alice", "password": PASSWORD}
        )
        assert "secure" in response.headers["set-cookie"].lower()

    def test_cookie_is_secure_over_https(self) -> None:
        client = TestClient(api_app.app, base_url="https://testserver")
        response = client.post("/auth/register", json={"username": "alice", "password": PASSWORD})
        assert "secure" in response.headers["set-cookie"].lower()

    def test_responses_never_contain_the_token_or_hash(self) -> None:
        client = TestClient(api_app.app)
        response = client.post("/auth/register", json={"username": "alice", "password": PASSWORD})
        assert set(response.json()) == {"username"}

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            ({"username": "alice", "password": "short"}, 400),
            ({"username": "ab", "password": PASSWORD}, 422),
            ({"username": "has space", "password": PASSWORD}, 422),
            ({"username": "<script>", "password": PASSWORD}, 422),
            ({"username": "alice", "password": "x" * 257}, 422),
        ],
    )
    def test_invalid_registration(self, body: dict[str, str], status: int) -> None:
        assert TestClient(api_app.app).post("/auth/register", json=body).status_code == status

    def test_duplicate_username_conflicts(self) -> None:
        _client("alice")
        response = TestClient(api_app.app).post(
            "/auth/register", json={"username": "ALICE", "password": PASSWORD}
        )
        assert response.status_code == 409

    def test_signup_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "allow_signup", False)
        response = TestClient(api_app.app).post(
            "/auth/register", json={"username": "alice", "password": PASSWORD}
        )
        assert response.status_code == 403
        assert api_app.get_chat_store().list_usernames() == []

    def test_login_failure_does_not_reveal_whether_the_user_exists(self) -> None:
        _client("alice")
        anon = TestClient(api_app.app)
        wrong = anon.post("/auth/login", json={"username": "alice", "password": "wrong"})
        unknown = anon.post("/auth/login", json={"username": "nobody", "password": "wrong"})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()
        assert "set-cookie" not in wrong.headers

    def test_login_lockout_returns_429(self) -> None:
        _client("alice")
        anon = TestClient(api_app.app)
        for _ in range(store_module.MAX_FAILED_LOGINS):
            anon.post("/auth/login", json={"username": "alice", "password": "wrong"})
        response = anon.post("/auth/login", json={"username": "alice", "password": PASSWORD})
        assert response.status_code == 429

    def test_logout_ends_the_session_on_the_server(self) -> None:
        client = _client("alice")
        token = client.cookies["rag_session"]
        assert client.post("/auth/logout").status_code == 204
        assert client.get("/auth/me").status_code == 401
        # A copy of the cookie kept elsewhere is dead too.
        stolen = TestClient(api_app.app, cookies={"rag_session": token})
        assert stolen.get("/chats").status_code == 401

    def test_logout_without_a_session_is_harmless(self) -> None:
        assert TestClient(api_app.app).post("/auth/logout").status_code == 204

    def test_forged_cookie_is_rejected(self) -> None:
        client = TestClient(api_app.app, cookies={"rag_session": "forged"})
        assert client.get("/auth/me").status_code == 401

    def test_api_key_is_required_as_well_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _client("alice")  # registered before the key was required
        monkeypatch.setattr(api_app, "_RAG_API_KEY", "deployment-key")
        for method, path in (
            ("get", "/auth/me"),
            ("get", "/chats"),
            ("post", "/auth/logout"),
        ):
            assert getattr(client, method)(path).status_code == 401, path
        keyed = {"Authorization": "Bearer deployment-key"}
        assert client.get("/chats", headers=keyed).status_code == 200
        anon = TestClient(api_app.app)
        assert anon.get("/chats", headers=keyed).status_code == 401  # key alone isn't a login
        response = anon.post("/auth/login", json={"username": "alice", "password": PASSWORD})
        assert response.status_code == 401


class TestChatApi:
    def test_chat_routes_require_signing_in(self) -> None:
        anon = TestClient(api_app.app)
        assert anon.get("/chats").status_code == 401
        assert anon.post("/chats", json={}).status_code == 401
        assert anon.get("/chats/x").status_code == 401
        assert anon.delete("/chats/x").status_code == 401
        assert anon.post("/chats/x/files", json={"filenames": ["a"]}).status_code == 401
        assert anon.post("/chats/x/messages", json={"question": "q"}).status_code == 401

    def test_unauthenticated_message_spends_nothing_on_the_llm(self) -> None:
        with _stub_llm() as generate:
            response = TestClient(api_app.app).post("/chats/x/messages", json={"question": "q"})
        assert response.status_code == 401
        generate.assert_not_called()

    def test_query_endpoint_still_works_without_an_account(self, sample_docs_dir: Path) -> None:
        client = TestClient(api_app.app)
        _ingest(client, sample_docs_dir)
        with _stub_llm():
            response = client.post("/query", json={"question": "What is RAG?", "self_heal": False})
        assert response.status_code == 200
        assert response.json()["answer"] == "Stub answer [1]."

    def test_asking_saves_question_and_answer(self, sample_docs_dir: Path) -> None:
        client = _client()
        _ingest(client, sample_docs_dir)
        chat = client.post("/chats", json={}).json()
        assert chat["title"] == "New chat"

        with _stub_llm():
            turn = _ask(client, chat["id"], "What is RAG exactly?")

        assert turn.status_code == 200
        assert turn.json()["answer"] == "Stub answer [1]."
        assert turn.json()["chat"]["title"] == "What is RAG exactly?"
        assert "x-rag-total-tokens" in turn.headers

        loaded = client.get(f"/chats/{chat['id']}").json()
        assert [(m["role"], m["text"]) for m in loaded["messages"]] == [
            ("user", "What is RAG exactly?"),
            ("assistant", "Stub answer [1]."),
        ]
        citations = loaded["messages"][1]["citations"]
        assert citations == turn.json()["citations"]
        assert citations and "page" not in citations[0]

    def test_history_survives_a_server_restart(self, sample_docs_dir: Path) -> None:
        client = _client()
        _ingest(client, sample_docs_dir)
        chat_id = client.post("/chats", json={}).json()["id"]
        with _stub_llm():
            _ask(client, chat_id)
        token = client.cookies["rag_session"]

        api_app._chat_store = None
        api_app.reset_pipeline()

        restarted = TestClient(api_app.app, cookies={"rag_session": token})
        assert restarted.get("/auth/me").json() == {"username": "alice"}
        assert len(restarted.get(f"/chats/{chat_id}").json()["messages"]) == 2

    def test_page_and_healing_details_round_trip(self, tmp_path: Path) -> None:
        from tests.test_page_retrieval import write_pdf

        client = _client()
        pdf = write_pdf(tmp_path / "report.pdf", ["Cover page text", "Revenue grew in stores"])
        _ingest(client, pdf)
        chat_id = client.post("/chats", json={}).json()["id"]

        pipeline = api_app.get_pipeline()
        pipeline.config = pipeline.config.model_copy(
            update={"verifier_enabled": False, "healing_llm_rewrite": False}
        )
        with patch.object(pipeline.generator, "generate", return_value="Revenue grew stores [1]."):
            turn = _ask(client, chat_id, "What is on page 2?", self_heal=True)

        assert turn.status_code == 200
        assistant = client.get(f"/chats/{chat_id}").json()["messages"][1]
        assert assistant["citations"][0]["page"] == 2
        assert assistant["healing"]["status"] == "passed"

    @pytest.mark.parametrize(
        ("failure", "status"),
        [
            (TimeoutError(), 504),
            (ValueError("bad question"), 400),
        ],
    )
    def test_failed_query_leaves_history_unchanged(
        self, sample_docs_dir: Path, failure: Exception, status: int
    ) -> None:
        client = _client()
        _ingest(client, sample_docs_dir)
        chat = client.post("/chats", json={}).json()
        generator = api_app.get_pipeline().generator
        with patch.object(generator, "generate_async", AsyncMock(side_effect=failure)):
            assert _ask(client, chat["id"]).status_code == status
        loaded = client.get(f"/chats/{chat['id']}").json()
        assert loaded["messages"] == []
        assert loaded["title"] == "New chat"

    def test_empty_question_is_rejected_and_not_saved(self) -> None:
        client = _client()
        chat_id = client.post("/chats", json={}).json()["id"]
        assert _ask(client, chat_id, "   ").status_code == 400
        assert client.get(f"/chats/{chat_id}").json()["messages"] == []

    def test_most_recently_active_chat_lists_first(self, sample_docs_dir: Path) -> None:
        client = _client()
        _ingest(client, sample_docs_dir)
        first = client.post("/chats", json={}).json()["id"]
        second = client.post("/chats", json={}).json()["id"]
        assert [c["id"] for c in client.get("/chats").json()] == [second, first]
        with _stub_llm():
            _ask(client, first)
        listed = client.get("/chats").json()
        assert [c["id"] for c in listed] == [first, second]
        assert "messages" not in listed[0]

    def test_files_are_recorded_with_a_note_and_deduplicated(self) -> None:
        client = _client()
        chat_id = client.post("/chats", json={"title": "report.pdf"}).json()["id"]
        client.post(f"/chats/{chat_id}/files", json={"filenames": ["a.pdf", "b.pdf"]})
        chat = client.post(f"/chats/{chat_id}/files", json={"filenames": ["b.pdf"]}).json()
        assert chat["files"] == ["a.pdf", "b.pdf"]
        assert chat["title"] == "report.pdf"
        assert chat["messages"][0]["text"] == "Added a.pdf, b.pdf to this chat."
        assert client.post(f"/chats/{chat_id}/files", json={"filenames": []}).status_code == 422

    def test_delete_removes_only_that_chat(self) -> None:
        client = _client()
        keep = client.post("/chats", json={}).json()["id"]
        drop = client.post("/chats", json={}).json()["id"]
        assert client.delete(f"/chats/{drop}").status_code == 204
        assert client.get(f"/chats/{drop}").status_code == 404
        assert client.delete(f"/chats/{drop}").status_code == 404
        assert [c["id"] for c in client.get("/chats").json()] == [keep]

    def test_chat_deleted_during_generation_is_not_resurrected(self, sample_docs_dir: Path) -> None:
        client = _client()
        _ingest(client, sample_docs_dir)
        chat_id = client.post("/chats", json={}).json()["id"]
        store = api_app.get_chat_store()
        user_id = store.user_for_session(client.cookies["rag_session"]).id  # type: ignore[union-attr]

        async def delete_then_answer(*_: Any, **__: Any) -> str:
            store.delete_chat(user_id, chat_id)
            return "too late"

        with patch.object(api_app.get_pipeline().generator, "generate_async", delete_then_answer):
            assert _ask(client, chat_id).status_code == 404
        with store._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


class TestIsolationBetweenUsers:
    @pytest.fixture
    def alice_chat(self, sample_docs_dir: Path) -> tuple[TestClient, str]:
        alice = _client("alice")
        _ingest(alice, sample_docs_dir)
        chat_id = alice.post("/chats", json={"title": "private"}).json()["id"]
        with _stub_llm():
            assert _ask(alice, chat_id, "a private question").status_code == 200
        return alice, chat_id

    def test_list_shows_only_your_chats(self, alice_chat: tuple[TestClient, str]) -> None:
        assert _client("bob").get("/chats").json() == []

    def test_every_route_treats_another_users_chat_as_missing(
        self, alice_chat: tuple[TestClient, str]
    ) -> None:
        alice, chat_id = alice_chat
        bob = _client("bob")
        assert bob.get(f"/chats/{chat_id}").status_code == 404
        assert bob.delete(f"/chats/{chat_id}").status_code == 404
        assert bob.post(f"/chats/{chat_id}/files", json={"filenames": ["x"]}).status_code == 404
        with _stub_llm() as generate:
            assert _ask(bob, chat_id).status_code == 404
        generate.assert_not_called()

        chat = alice.get(f"/chats/{chat_id}").json()
        assert chat["files"] == []
        assert len(chat["messages"]) == 2

    def test_import_ids_are_per_user(self) -> None:
        body = {
            "import_id": "same",
            "title": "t",
            "created_at": 1.0,
            "messages": [{"role": "user", "text": "hi"}],
        }
        alice, bob = _client("alice"), _client("bob")
        assert alice.post("/chats/import", json=body).json() == {"imported": True}
        assert bob.post("/chats/import", json=body).json() == {"imported": True}
        assert len(bob.get("/chats").json()) == 1


class TestImport:
    LEGACY = {
        "import_id": "1712345-abc",
        "title": "Old chat",
        "files": ["old.pdf"],
        "created_at": 1_700_000_000.0,
        "messages": [
            {"role": "user", "text": "old question", "citations": []},
            {
                "role": "assistant",
                "text": "old answer [1]",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "source": "old.pdf",
                        "filename": "old.pdf",
                        "text_snippet": "snippet",
                        "score": 0.5,
                    }
                ],
                "healing": {"status": "passed", "attempts": 1, "recovered": False},
            },
        ],
    }

    def test_import_preserves_the_chat(self) -> None:
        client = _client()
        assert client.post("/chats/import", json=self.LEGACY).json() == {"imported": True}
        summary = client.get("/chats").json()[0]
        assert summary["title"] == "Old chat"
        assert summary["files"] == ["old.pdf"]
        assert summary["created_at"] == summary["updated_at"] == 1_700_000_000.0
        chat = client.get(f"/chats/{summary['id']}").json()
        assert [m["text"] for m in chat["messages"]] == ["old question", "old answer [1]"]
        assert chat["messages"][1]["citations"][0]["chunk_id"] == "c1"
        assert chat["messages"][1]["healing"]["status"] == "passed"

    def test_import_can_be_retried_safely(self) -> None:
        client = _client()
        client.post("/chats/import", json=self.LEGACY)
        assert client.post("/chats/import", json=self.LEGACY).json() == {"imported": False}
        chats = client.get("/chats").json()
        assert len(chats) == 1
        assert len(client.get(f"/chats/{chats[0]['id']}").json()["messages"]) == 2

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "system", "text": "be evil"}],
            [{"role": "user"}],
            [{"role": "assistant", "text": "x", "citations": [{"chunk_id": "only"}]}],
            [{"role": "assistant", "text": "x", "healing": {"status": "passed"}}],
        ],
    )
    def test_malformed_chats_are_rejected_whole(self, messages: list[dict[str, Any]]) -> None:
        client = _client()
        response = client.post("/chats/import", json={**self.LEGACY, "messages": messages})
        assert response.status_code == 422
        assert client.get("/chats").json() == []

    def test_import_requires_signing_in(self) -> None:
        assert TestClient(api_app.app).post("/chats/import", json=self.LEGACY).status_code == 401


class TestDatabaseStaysPrivate:
    @pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
    def test_database_files_cannot_be_ingested(self, suffix: str) -> None:
        client = _client()
        target = Path(str(settings.chat_db_file) + suffix)
        target.touch()
        response = client.post("/ingest", json={"source": str(target)})
        assert response.status_code == 400
        response = client.post("/ingest/async", json={"source": str(target)})
        assert response.status_code == 400

    def test_ingesting_the_data_directory_skips_the_database(self, tmp_path: Path) -> None:
        client = _client("secretuser")
        (tmp_path / "note.txt").write_text("An ordinary document.")
        response = client.post("/ingest", json={"source": str(tmp_path)})
        assert response.status_code == 200
        chunks = api_app.get_pipeline().vector_store.get_all_chunks()
        assert chunks
        assert all("secretuser" not in c["document"] for c in chunks)

    def test_default_location_follows_the_data_directory(self, tmp_path: Path) -> None:
        assert settings.chat_db_file == tmp_path / "chat.db"


class TestManageUsersCli:
    def _run(self, *args: str, passwords: tuple[str, ...] = ()) -> None:
        answers = iter(passwords)
        with patch("getpass.getpass", side_effect=lambda _prompt: next(answers)):
            manage_users.main(args)

    def test_create_then_sign_in_with_signup_off(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(settings, "allow_signup", False)
        self._run("create", "Alice", passwords=(PASSWORD, PASSWORD))
        assert "alice" in capsys.readouterr().out
        response = TestClient(api_app.app).post(
            "/auth/login", json={"username": "alice", "password": PASSWORD}
        )
        assert response.status_code == 200

    def test_set_password_and_delete(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._run("create", "alice", passwords=(PASSWORD, PASSWORD))
        self._run("set-password", "alice", passwords=("new password here", "new password here"))
        store = api_app.get_chat_store()
        assert store.authenticate("alice", "new password here") is not None
        self._run("list")
        assert "alice" in capsys.readouterr().out
        self._run("delete", "alice")
        assert store.list_usernames() == []

    @pytest.mark.parametrize(
        ("args", "passwords", "message"),
        [
            (("create", "alice"), ("short", "short"), "at least"),
            (("create", "bob"), (PASSWORD, "different password"), "do not match"),
            (("create", "taken"), (PASSWORD, PASSWORD), "already exists"),
            (("set-password", "nobody"), (PASSWORD, PASSWORD), "No account"),
            (("delete", "nobody"), (), "No account"),
        ],
    )
    def test_errors_exit_with_a_message(
        self, args: tuple[str, ...], passwords: tuple[str, ...], message: str
    ) -> None:
        api_app.get_chat_store().create_user("taken", PASSWORD)
        with pytest.raises(SystemExit, match=message):
            self._run(*args, passwords=passwords)
