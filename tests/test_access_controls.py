"""Store reset protection, per-address login limits, per-user documents, and follow-ups."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src import api as api_app
from src.chat import store as store_module
from src.chat.rate_limit import AttemptLimiter
from src.config import settings
from src.generation.generator import Generator
from src.pipeline import RAGPipeline
from tests.test_page_retrieval import write_pdf

PASSWORD = "correct horse battery"
ALICE_SECRET = "Alice private merger codename is bluefalcon and it closes in March."
BOB_SECRET = "Bob private salary figures mention redheron and a bonus plan."
SHARED_TEXT = "The company handbook describes the vacation policy for everyone."
REAL_CONDENSE = Generator.condense_question


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(store_module, "_SCRYPT_N", 2**4)
    monkeypatch.setattr(store_module, "_SCRYPT_P", 1)
    api_app.reset_pipeline()
    yield
    api_app.reset_pipeline()


def _account(username: str) -> TestClient:
    client = TestClient(api_app.app)
    response = client.post("/auth/register", json={"username": username, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return client


def _document(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _stub_llm(answer: str = "Stub answer [1].") -> Any:
    generator = api_app.get_pipeline().generator
    return patch.object(generator, "generate_async", AsyncMock(return_value=answer))


def _answer_from_context(_query: str, contexts: list[dict[str, Any]], **_: Any) -> str:
    """Passes the citation checks, because it repeats what the first passage says."""
    return f"{contexts[0]['document']} [1]."


def _ask(client: TestClient, **extra: Any) -> Any:
    body = {"question": "What is in the documents?", "self_heal": False, "top_k": 10, **extra}
    # Monitoring and self-healing answer through the sync path, so stub both.
    generator = api_app.get_pipeline().generator
    with _stub_llm(), patch.object(generator, "generate", side_effect=_answer_from_context):
        return client.post("/query", json=body)


def _cited(response: Any) -> str:
    return " ".join(c["text_snippet"] for c in response.json()["citations"])


class TestStoreReset:
    @pytest.mark.parametrize("route", ["/ingest", "/ingest/async"])
    def test_reset_is_refused_by_default(
        self, route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "allow_store_reset", False)
        client = TestClient(api_app.app)
        keep = _document(tmp_path, "keep.txt", SHARED_TEXT)
        assert client.post("/ingest", json={"source": str(keep)}).status_code == 200
        before = client.get("/stats").json()["chunks_in_store"]

        response = client.post(route, json={"source": str(keep), "reset": True})

        assert response.status_code == 403
        assert "RAG_ALLOW_STORE_RESET" in response.json()["detail"]
        assert client.get("/stats").json()["chunks_in_store"] == before == 1

    def test_ingest_without_reset_still_works_when_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "allow_store_reset", False)
        doc = _document(tmp_path, "a.txt", SHARED_TEXT)
        client = TestClient(api_app.app)
        assert client.post("/ingest", json={"source": str(doc)}).status_code == 200

    def test_an_operator_can_turn_it_on(self, tmp_path: Path) -> None:
        client = TestClient(api_app.app)  # the test configuration enables reset
        doc = _document(tmp_path, "a.txt", SHARED_TEXT)
        other = _document(tmp_path, "b.txt", "Something else entirely about ranking.")
        client.post("/ingest", json={"source": str(doc)})
        response = client.post("/ingest", json={"source": str(other), "reset": True})
        assert response.status_code == 200
        assert response.json()["total_chunks"] == 1


class TestLoginLimits:
    def _limit(self, monkeypatch: pytest.MonkeyPatch, attempts: int = 3) -> None:
        monkeypatch.setattr(settings, "auth_ip_max_attempts", attempts)

    def test_failed_logins_block_the_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._limit(monkeypatch)
        _account("alice")
        api_app._auth_limiter.clear()
        client = TestClient(api_app.app)
        for name in ("guess1", "guess2", "guess3"):  # different accounts, so no account lockout
            body = {"username": name, "password": "wrong password"}
            assert client.post("/auth/login", json=body).status_code == 401

        body = {"username": "alice", "password": PASSWORD}
        response = client.post("/auth/login", json=body)
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) > 0

    def test_blocked_address_does_no_password_hashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._limit(monkeypatch, attempts=1)
        client = TestClient(api_app.app)
        client.post("/auth/login", json={"username": "x", "password": "wrong password"})
        with patch.object(store_module, "_verify_password") as verify:
            assert (
                client.post("/auth/login", json={"username": "x", "password": "y"}).status_code
                == 429
            )
        verify.assert_not_called()

    def test_other_addresses_are_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._limit(monkeypatch, attempts=1)
        attacker = TestClient(api_app.app, client=("203.0.113.9", 1000))
        wrong = {"username": "x", "password": "wrong password"}
        attacker.post("/auth/login", json=wrong)
        assert attacker.post("/auth/login", json=wrong).status_code == 429
        other = TestClient(api_app.app, client=("198.51.100.7", 1000))
        assert other.post("/auth/login", json=wrong).status_code == 401

    def test_successful_logins_do_not_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._limit(monkeypatch, attempts=2)
        _account("alice")
        api_app._auth_limiter.clear()
        client = TestClient(api_app.app)
        for _ in range(5):
            body = {"username": "alice", "password": PASSWORD}
            assert client.post("/auth/login", json=body).status_code == 200

    def test_registrations_count_because_they_hash_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._limit(monkeypatch, attempts=2)
        client = TestClient(api_app.app)
        for name in ("user1", "user2"):
            body = {"username": name, "password": PASSWORD}
            assert client.post("/auth/register", json=body).status_code == 201
        body = {"username": "user3", "password": PASSWORD}
        assert client.post("/auth/register", json=body).status_code == 429
        assert api_app.get_chat_store().list_usernames() == ["user1", "user2"]

    def test_limit_clears_after_the_window(self) -> None:
        limiter = AttemptLimiter()
        limiter.record("a")
        assert limiter.blocked("a", 1, 60.0) is True
        assert limiter.blocked("a", 1, 0.0) is False
        assert limiter.blocked("never-seen", 1, 60.0) is False

    def test_tracked_addresses_are_capped(self) -> None:
        limiter = AttemptLimiter(max_tracked=3)
        for i in range(10):
            limiter.record(str(i))
        assert len(limiter._attempts) == 3
        assert limiter.blocked("9", 1, 60.0) is True
        assert limiter.blocked("0", 1, 60.0) is False


class TestPerUserDocuments:
    @pytest.fixture
    def clients(self, tmp_path: Path) -> tuple[TestClient, TestClient, TestClient]:
        alice, bob = _account("alice"), _account("bob")
        anonymous = TestClient(api_app.app)
        for client, name, text in (
            (alice, "alice.txt", ALICE_SECRET),
            (bob, "bob.txt", BOB_SECRET),
            (anonymous, "shared.txt", SHARED_TEXT),
        ):
            doc = _document(tmp_path, name, text)
            assert client.post("/ingest", json={"source": str(doc)}).status_code == 200
        return alice, bob, anonymous

    @pytest.mark.parametrize(
        "mode",
        [{}, {"use_hybrid": True}, {"use_reranker": True}, {"use_hybrid": True, "self_heal": True}],
    )
    def test_each_user_retrieves_only_their_own_and_shared(
        self, clients: Any, mode: dict[str, Any]
    ) -> None:
        alice, bob, anonymous = clients
        pipeline = api_app.get_pipeline()
        pipeline.config = pipeline.config.model_copy(
            update={
                "verifier_enabled": False,
                "healing_llm_rewrite": False,
                "min_query_coverage": 0.0,
                "min_retrieval_score": 0.0,
            }
        )
        with patch.object(pipeline, "_apply_reranker", side_effect=lambda q, c, top_k: c[:top_k]):
            as_alice = _cited(_ask(alice, **mode))
            as_bob = _cited(_ask(bob, **mode))
            as_nobody = _cited(_ask(anonymous, **mode))

        assert "bluefalcon" in as_alice and "redheron" not in as_alice
        assert "redheron" in as_bob and "bluefalcon" not in as_bob
        assert "handbook" in as_alice and "handbook" in as_bob
        assert "handbook" in as_nobody
        assert "bluefalcon" not in as_nobody and "redheron" not in as_nobody

    def test_keyword_match_cannot_reach_another_users_document(self, clients: Any) -> None:
        _alice, bob, _anon = clients
        cited = _cited(_ask(bob, question="bluefalcon merger codename", use_hybrid=True))
        assert "bluefalcon" not in cited

    def test_a_forged_session_sees_only_shared(self, clients: Any) -> None:
        forged = TestClient(api_app.app, cookies={"rag_session": "forged"})
        cited = _cited(_ask(forged))
        assert "handbook" in cited and "bluefalcon" not in cited

    def test_streaming_is_scoped(self, clients: Any) -> None:
        _alice, bob, _anon = clients

        async def stream(query: str, contexts: list[dict[str, Any]]) -> Any:
            yield "token"

        with patch.object(api_app.get_pipeline().generator, "generate_stream", stream):
            response = bob.post(
                "/query/stream",
                json={"question": "What is in the documents?", "self_heal": False, "top_k": 10},
            )
        assert "redheron" in response.text
        assert "bluefalcon" not in response.text

    def test_chat_route_is_scoped(self, clients: Any) -> None:
        _alice, bob, _anon = clients
        chat_id = bob.post("/chats", json={}).json()["id"]
        with _stub_llm():
            response = bob.post(
                f"/chats/{chat_id}/messages",
                json={"question": "What is in the documents?", "self_heal": False, "top_k": 10},
            )
        cited = _cited(response)
        assert "redheron" in cited and "bluefalcon" not in cited

    def test_monitored_pipeline_is_scoped(self, clients: Any) -> None:
        from src.monitoring import MonitoredRAGPipeline, Tracer

        _alice, bob, _anon = clients
        api_app._pipeline = MonitoredRAGPipeline(RAGPipeline(), tracer=Tracer(enabled=False))
        cited = _cited(_ask(bob))
        assert "redheron" in cited and "bluefalcon" not in cited

    def test_full_passage_route_hides_other_users_chunks(self, clients: Any) -> None:
        alice, bob, anonymous = clients
        citation = next(
            c for c in _ask(alice).json()["citations"] if "bluefalcon" in c["text_snippet"]
        )
        path = f"/chunks/{citation['chunk_id']}"
        assert "bluefalcon" in alice.get(path).json()["text"]
        assert bob.get(path).status_code == 404
        assert anonymous.get(path).status_code == 404

    def test_background_jobs_and_their_reruns_keep_the_owner(self, tmp_path: Path) -> None:
        alice, bob = _account("alice"), _account("bob")
        folder = tmp_path / "private"
        folder.mkdir()
        _document(folder, "a.txt", ALICE_SECRET)
        job_id = alice.post("/ingest/async", json={"source": str(folder)}).json()["job_id"]
        from tests.test_background_ingestion import _wait

        assert _wait(alice, job_id)["status"] == "completed"
        assert "bluefalcon" in _cited(_ask(alice))
        assert "bluefalcon" not in _cited(_ask(bob))

        api_app._ingest_jobs[job_id]["status"] = "cancelled"
        new_id = alice.post(f"/ingest/jobs/{job_id}/retry").json()["job_id"]
        assert _wait(alice, new_id)["chunks_ingested"] == 0  # already stored under the same owner

    def test_two_users_can_add_the_same_file(self, tmp_path: Path) -> None:
        alice, bob = _account("alice"), _account("bob")
        doc = _document(tmp_path, "same.txt", SHARED_TEXT)
        assert alice.post("/ingest", json={"source": str(doc)}).json()["chunks_ingested"] == 1
        assert bob.post("/ingest", json={"source": str(doc)}).json()["chunks_ingested"] == 1
        assert "handbook" in _cited(_ask(alice))
        assert "handbook" in _cited(_ask(bob))

    def test_page_questions_are_scoped(self, tmp_path: Path) -> None:
        alice, bob = _account("alice"), _account("bob")
        pdf = write_pdf(tmp_path / "secret.pdf", ["bluefalcon on the first page."])
        assert alice.post("/ingest", json={"source": str(pdf)}).status_code == 200
        bobs = _document(tmp_path, "bob.txt", BOB_SECRET)
        assert bob.post("/ingest", json={"source": str(bobs)}).status_code == 200

        assert "bluefalcon" in _cited(_ask(alice, question="What is on page 1?"))
        # Bob has no paged documents, so his question isn't treated as a page lookup at all.
        cited = _cited(_ask(bob, question="What is on page 1?"))
        assert "bluefalcon" not in cited and "redheron" in cited

    def test_library_and_cli_use_is_unrestricted(self, tmp_path: Path) -> None:
        pipeline = RAGPipeline()
        pipeline.reset()
        pipeline.ingest(_document(tmp_path, "a.txt", ALICE_SECRET), owner="some-user")
        pipeline.ingest(_document(tmp_path, "b.txt", SHARED_TEXT))
        hits = pipeline._retrieve("documents", k=10)
        assert len(hits) == 2
        assert {h["metadata"]["owner"] for h in hits} == {"some-user", "shared"}


class TestFollowUps:
    @pytest.fixture
    def chat(self, tmp_path: Path) -> tuple[TestClient, str]:
        client = _account("alice")
        doc = _document(tmp_path, "a.txt", ALICE_SECRET)
        assert client.post("/ingest", json={"source": str(doc)}).status_code == 200
        return client, client.post("/chats", json={}).json()["id"]

    def _send(self, client: TestClient, chat_id: str, question: str) -> Any:
        body = {"question": question, "self_heal": False}
        return client.post(f"/chats/{chat_id}/messages", json=body)

    def test_first_message_costs_no_extra_llm_call(self, chat: Any) -> None:
        client, chat_id = chat
        condense = AsyncMock()
        with _stub_llm(), patch.object(Generator, "condense_question", condense):
            response = self._send(client, chat_id, "What is the codename?")
        condense.assert_not_called()
        assert "searched_for" not in response.json()

    def test_follow_up_is_rewritten_for_retrieval_and_generation(self, chat: Any) -> None:
        client, chat_id = chat
        generator = api_app.get_pipeline().generator
        with _stub_llm():
            self._send(client, chat_id, "What is the codename?")
        standalone = "When does the bluefalcon merger close?"
        condense = AsyncMock(return_value=f"  {standalone}\n")
        with (
            patch.object(Generator, "condense_question", condense),
            patch.object(generator, "generate_async", AsyncMock(return_value="March [1].")) as gen,
        ):
            response = self._send(client, chat_id, "When does it close?")

        assert response.status_code == 200
        assert response.json()["searched_for"] == standalone
        assert gen.call_args[0][0] == standalone
        question, history = condense.call_args[0]
        assert question == "When does it close?"
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["text"] == "What is the codename?"

    def test_history_shows_what_the_user_typed(self, chat: Any) -> None:
        client, chat_id = chat
        with _stub_llm():
            self._send(client, chat_id, "What is the codename?")
        with (
            _stub_llm(),
            patch.object(Generator, "condense_question", AsyncMock(return_value="Rewritten?")),
        ):
            self._send(client, chat_id, "And then?")
        messages = client.get(f"/chats/{chat_id}").json()["messages"]
        assert [m["text"] for m in messages if m["role"] == "user"] == [
            "What is the codename?",
            "And then?",
        ]

    def test_rewritten_page_reference_still_filters_by_page(self, tmp_path: Path) -> None:
        client = _account("alice")
        pdf = write_pdf(tmp_path / "r.pdf", ["Cover text here.", "Revenue grew in stores."])
        client.post("/ingest", json={"source": str(pdf)})
        chat_id = client.post("/chats", json={}).json()["id"]
        with _stub_llm():
            self._send(client, chat_id, "What is on page 1?")
        with (
            _stub_llm(),
            patch.object(
                Generator,
                "condense_question",
                AsyncMock(return_value="What does the report say on page 2?"),
            ),
        ):
            response = self._send(client, chat_id, "What about page two then")
        assert [c["page"] for c in response.json()["citations"]] == [2]

    @pytest.mark.parametrize(
        "condense",
        [
            AsyncMock(side_effect=RuntimeError("llm down")),
            AsyncMock(side_effect=TimeoutError()),
            AsyncMock(return_value="   "),
            AsyncMock(return_value="x" * 2001),
        ],
    )
    def test_a_failed_rewrite_asks_the_question_as_written(
        self, chat: Any, condense: AsyncMock
    ) -> None:
        client, chat_id = chat
        generator = api_app.get_pipeline().generator
        with _stub_llm():
            self._send(client, chat_id, "What is the codename?")
        with (
            patch.object(Generator, "condense_question", condense),
            patch.object(generator, "generate_async", AsyncMock(return_value="Ok [1].")) as gen,
        ):
            response = self._send(client, chat_id, "When does it close?")
        assert response.status_code == 200
        assert gen.call_args[0][0] == "When does it close?"
        assert "searched_for" not in response.json()

    def test_a_chat_with_only_a_document_note_has_nothing_to_resolve(self, chat: Any) -> None:
        client, chat_id = chat
        client.post(f"/chats/{chat_id}/files", json={"filenames": ["a.pdf"]})
        condense = AsyncMock()
        with _stub_llm(), patch.object(Generator, "condense_question", condense):
            self._send(client, chat_id, "What is the codename?")
        condense.assert_not_called()

    @pytest.mark.asyncio
    async def test_condense_prompt_is_bounded_and_keeps_recent_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Generator, "condense_question", REAL_CONDENSE)
        generator = Generator(provider="openai", model="m")
        history = [{"role": "user", "text": f"message {i} " + "x" * 2000} for i in range(10)]
        with patch.object(
            generator._client, "complete_async", AsyncMock(return_value="Standalone?")
        ) as complete:
            assert await generator.condense_question("And then?", history) == "Standalone?"
        prompt = complete.call_args.kwargs["prompt"]
        assert "message 9" in prompt and "message 3" not in prompt
        assert len(prompt) < 3500
        assert prompt.endswith("Latest message: And then?")
