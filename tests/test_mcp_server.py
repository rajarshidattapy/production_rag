"""The MCP server: read-only tools, limited to shared documents not restricted to roles."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.mcpserver import MCPServer

from src import api as api_app
from src import mcp_server
from src.config import settings
from src.pipeline import SHARED_OWNER, RAGPipeline, retrieval_owners

TOOLS = {
    "search_knowledge_base",
    "ask_knowledge_base",
    "get_passage",
    "get_section",
    "list_documents",
    "knowledge_base_info",
}


@pytest.fixture
def library(tmp_path: Path) -> Path:
    for folder, name, text in (
        ("policies/hr", "salaries.txt", "Zebracorn salary bands for every grade are secret."),
        ("policies", "travel.txt", "Zebracorn travel policy: book economy class for flights."),
        ("misc", "notes.txt", "Zebracorn notes. Contact alice@example.com about parking."),
        ("private", "diary.txt", "Zebracorn diary entry that only its owner may read."),
    ):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / name).write_text(text, encoding="utf-8")
    (tmp_path / "guide.md").write_text(
        "# Handbook\n\n## Expenses\n\nZebracorn expenses are claimed within thirty days. "
        + "Receipts are required for every claim over ten euros. " * 30,
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def pipeline(library: Path) -> RAGPipeline:
    p = RAGPipeline()
    p.reset()
    p.config = p.config.model_copy(update={"source_roles": {"policies/hr": ["hr"]}})
    for path in ("policies", "misc", "guide.md"):
        p.ingest(library / path)
    p.ingest(library / "private", owner="u1")
    return p


@pytest.fixture
def client(pipeline: RAGPipeline) -> MCPServer:
    return mcp_server.build_server(lambda: pipeline)


async def _call(client: MCPServer, tool: str, **arguments: Any) -> dict[str, Any]:
    async with Client(client) as session:
        result = await session.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def _error(client: MCPServer, tool: str, **arguments: Any) -> str:
    async with Client(client) as session:
        result = await session.call_tool(tool, arguments)
    assert result.is_error
    return " ".join(getattr(c, "text", "") for c in result.content)


# Long enough to be detected as English, so it searches the English collection.
EVERYTHING = "What do the Zebracorn documents say about the policy for the staff?"


def _everything(**extra: Any) -> dict[str, Any]:
    return {"query": EVERYTHING, "top_k": 20, "use_hybrid": True, **extra}


async def _passage_from(client: MCPServer, document: str) -> dict[str, Any]:
    found = await _call(client, "search_knowledge_base", **_everything(document=document))
    return found["passages"][0]


class TestTools:
    @pytest.mark.asyncio
    async def test_only_read_only_tools_are_offered(self, client: MCPServer) -> None:
        async with Client(client) as session:
            tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == TOOLS
        for tool in tools:
            assert tool.annotations is not None and tool.annotations.read_only_hint
            assert tool.annotations.destructive_hint is False
            assert tool.description and tool.output_schema
        names = " ".join(t.name for t in tools)
        for admin in ("ingest", "upload", "reset", "delete", "user", "chat"):
            assert admin not in names

    @pytest.mark.asyncio
    async def test_search_returns_passages_with_their_source(self, client: MCPServer) -> None:
        found = await _call(client, "search_knowledge_base", **_everything())
        assert found["query"] == EVERYTHING
        travel = next(p for p in found["passages"] if p["document"] == "travel.txt")
        assert "economy" in travel["text"]
        assert travel["chunk_id"] and isinstance(travel["score"], float)
        assert travel["injection_suspected"] is False
        scores = [p["score"] for p in found["passages"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_can_keep_to_one_document(self, client: MCPServer) -> None:
        found = await _call(client, "search_knowledge_base", **_everything(document="notes.txt"))
        assert {p["document"] for p in found["passages"]} == {"notes.txt"}

    @pytest.mark.asyncio
    async def test_reranked_search(self, client: MCPServer, pipeline: RAGPipeline) -> None:
        def rerank(_query: str, contexts: list[dict], top_k: int = 5) -> list[dict]:
            ranked = [{**c, "rerank_score": float(i)} for i, c in enumerate(contexts)]
            return sorted(ranked, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

        with patch.object(pipeline, "_get_reranker") as reranker:
            reranker.return_value.rerank.side_effect = rerank
            found = await _call(client, "search_knowledge_base", **_everything(use_reranker=True))
        scores = [p["score"] for p in found["passages"]]
        assert scores == sorted(scores, reverse=True) and scores[0] > 0

    @pytest.mark.asyncio
    async def test_bad_arguments_are_explained(self, client: MCPServer) -> None:
        assert "top_k" in await _error(client, "search_knowledge_base", query="x", top_k=500)
        assert "empty" in await _error(client, "search_knowledge_base", query="   ")
        assert "limit" in await _error(client, "list_documents", limit=0)

    @pytest.mark.asyncio
    async def test_ask_gives_a_grounded_answer_with_citations(
        self, client: MCPServer, pipeline: RAGPipeline
    ) -> None:
        async def answer(question: str, contexts: list[dict], **_kw: Any) -> str:
            assert any("economy" in c["document"] for c in contexts)
            return "Book economy class [1]."

        with patch.object(pipeline.generator, "generate_async", side_effect=answer):
            result = await _call(
                client, "ask_knowledge_base", question="Which class do I book for flights?"
            )
        assert result["answer"] == "Book economy class [1]."
        assert result["grounded"] is True
        first = result["citations"][0]
        assert first["number"] == 1 and first["document"] and first["snippet"]
        assert result.get("healing") is None

    @pytest.mark.asyncio
    async def test_ask_says_when_nothing_supports_an_answer(
        self, client: MCPServer, pipeline: RAGPipeline
    ) -> None:
        pipeline.reset()
        with patch.object(pipeline.generator, "generate_async") as generate:
            result = await _call(client, "ask_knowledge_base", question="Anything?")
        generate.assert_not_called()
        assert result["grounded"] is False and result["citations"] == []

    @pytest.mark.asyncio
    async def test_ask_reports_self_healing(self, client: MCPServer, pipeline: RAGPipeline) -> None:
        from src.healing.types import HealingInfo, HealingResult

        healed = HealingResult(
            answer="Healed [1].",
            citations=[],
            info=HealingInfo(status="passed", attempts=1, recovered=False),
        )

        async def query_with_healing(*_args: Any, **_kwargs: Any) -> HealingResult:
            assert retrieval_owners.get() == (SHARED_OWNER,)
            return healed

        with patch.object(pipeline, "query_with_healing_async", side_effect=query_with_healing):
            result = await _call(client, "ask_knowledge_base", question="q", self_heal=True)
        assert result["answer"] == "Healed [1]."
        assert result["healing"]["status"] == "passed"

    @pytest.mark.asyncio
    async def test_get_passage_returns_the_whole_text(self, client: MCPServer) -> None:
        hit = await _passage_from(client, "travel.txt")
        passage = await _call(client, "get_passage", chunk_id=hit["chunk_id"])
        assert passage["text"] == hit["text"] and passage["document"] == "travel.txt"
        assert "No passage" in await _error(client, "get_passage", chunk_id="missing")

    @pytest.mark.asyncio
    async def test_get_section_returns_the_parent_section(self, client: MCPServer) -> None:
        found = await _call(client, "search_knowledge_base", **_everything(document="guide.md"))
        # The shortest piece of the Expenses section: its heading.
        passage = min(
            (p for p in found["passages"] if p["section"] == "Handbook > Expenses"),
            key=lambda p: len(p["text"]),
        )
        section = await _call(client, "get_section", chunk_id=passage["chunk_id"])
        assert section["document"] == "guide.md"
        assert section["section_id"] != passage["chunk_id"]
        assert len(section["text"]) > len(passage["text"])
        assert passage["text"] in section["text"]
        assert section["section"] == "Handbook > Expenses"

    @pytest.mark.asyncio
    async def test_list_documents(self, client: MCPServer) -> None:
        listed = await _call(client, "list_documents")
        paths = [d["path"] for d in listed["documents"]]
        assert paths == ["guide.md", "misc/notes.txt", "policies/travel.txt"]
        assert listed["total"] == 3 and listed["truncated"] is False
        travel = listed["documents"][2]
        assert travel["document"] == "travel.txt" and travel["versions"] == 1
        assert travel["chunks"] >= 1 and travel["last_ingested_at"]

        narrowed = await _call(client, "list_documents", limit=1, name_contains="POLICIES")
        assert [d["path"] for d in narrowed["documents"]] == ["policies/travel.txt"]
        first = await _call(client, "list_documents", limit=1)
        assert first["total"] == 3 and first["truncated"] is True

    @pytest.mark.asyncio
    async def test_list_documents_sees_new_documents(
        self, client: MCPServer, pipeline: RAGPipeline, library: Path
    ) -> None:
        assert (await _call(client, "list_documents"))["total"] == 3
        (library / "late.txt").write_text("Added later.", encoding="utf-8")
        pipeline.ingest(library / "late.txt")
        assert (await _call(client, "list_documents"))["total"] == 4

    @pytest.mark.asyncio
    async def test_info_counts_only_what_a_client_can_read(
        self, client: MCPServer, pipeline: RAGPipeline
    ) -> None:
        info = await _call(client, "knowledge_base_info")
        listed = await _call(client, "list_documents")
        assert info["documents"] == 3
        assert info["chunks"] == sum(d["chunks"] for d in listed["documents"])
        assert info["chunks"] < pipeline.stats()["chunks_in_store"]
        assert info["embedding_model"] == pipeline.config.embedding_model
        assert "key" not in " ".join(info).lower()


class TestScope:
    @pytest.mark.asyncio
    async def test_search_never_returns_private_or_restricted_documents(
        self, client: MCPServer
    ) -> None:
        for hybrid in (True, False):
            found = await _call(client, "search_knowledge_base", **_everything(use_hybrid=hybrid))
            documents = {p["document"] for p in found["passages"]}
            assert documents == {"travel.txt", "notes.txt", "guide.md"}

    @pytest.mark.asyncio
    async def test_page_and_document_filters_stay_in_scope(self, client: MCPServer) -> None:
        for document in ("diary.txt", "salaries.txt"):
            found = await _call(client, "search_knowledge_base", **_everything(document=document))
            assert found["passages"] == []

    @pytest.mark.asyncio
    async def test_passages_out_of_scope_cannot_be_fetched_by_id(
        self, client: MCPServer, pipeline: RAGPipeline
    ) -> None:
        hidden = [
            c["id"]
            for lang in ("en", "de", "es")
            for c in pipeline._get_vector_store(lang).get_all_chunks()
            if c["metadata"]["filename"] in {"diary.txt", "salaries.txt"}
        ]
        assert len(hidden) >= 2
        for chunk_id in hidden:
            assert "No passage" in await _error(client, "get_passage", chunk_id=chunk_id)
            assert "No passage" in await _error(client, "get_section", chunk_id=chunk_id)

    @pytest.mark.asyncio
    async def test_answers_use_only_documents_in_scope(
        self, client: MCPServer, pipeline: RAGPipeline
    ) -> None:
        seen: list[str] = []

        async def answer(question: str, contexts: list[dict], **_kw: Any) -> str:
            seen.extend(c["metadata"]["filename"] for c in contexts)
            return "Answer [1]."

        with patch.object(pipeline.generator, "generate_async", side_effect=answer):
            await _call(client, "ask_knowledge_base", question=EVERYTHING, top_k=20)
        assert seen and not {"diary.txt", "salaries.txt"} & set(seen)

    @pytest.mark.asyncio
    async def test_scope_does_not_leak_into_the_caller(self, client: MCPServer) -> None:
        await _call(client, "search_knowledge_base", query="economy")
        assert retrieval_owners.get() is None

    @pytest.mark.asyncio
    async def test_text_is_redacted_when_answers_are(
        self, client: MCPServer, pipeline: RAGPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "redact_answers", True)
        found = await _call(client, "search_knowledge_base", **_everything(document="notes.txt"))
        passage = found["passages"][0]
        assert "alice@example.com" not in passage["text"]
        full = await _call(client, "get_passage", chunk_id=passage["chunk_id"])
        assert "alice@example.com" not in full["text"]

        async def leaky(*_args: Any, **_kw: Any) -> str:
            return "Write to alice@example.com [1]."

        with patch.object(pipeline.generator, "generate_async", side_effect=leaky):
            result = await _call(client, "ask_knowledge_base", question="parking contact")
        assert "alice@example.com" not in result["answer"]
        assert result["grounded"] and result["citations"]
        assert all("alice@example.com" not in c["snippet"] for c in result["citations"])


class TestHttp:
    HEADERS = {"accept": "application/json, text/event-stream"}

    @pytest.fixture
    def http(self, monkeypatch: pytest.MonkeyPatch, library: Path) -> Iterator[TestClient]:
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_allowed_hosts", "testserver")
        monkeypatch.setattr(api_app, "_RAG_API_KEY", "secret-key")
        monkeypatch.setattr(settings, "source_roles", {"policies/hr": ["hr"]})
        api_app.reset_pipeline()
        api_app.get_pipeline().ingest(library / "policies")
        with TestClient(api_app.app) as client:
            yield client
        api_app.reset_pipeline()

    def _rpc(self, client: TestClient, method: str, key: str = "secret-key", **params: Any):
        return client.post(
            "/mcp",
            headers={**self.HEADERS, "authorization": f"Bearer {key}"},
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )

    def test_tools_over_http(self, http: TestClient) -> None:
        init = self._rpc(
            http,
            "initialize",
            protocolVersion="2025-06-18",
            capabilities={},
            clientInfo={"name": "test", "version": "1"},
        )
        assert init.status_code == 200, init.text
        assert init.json()["result"]["serverInfo"]["name"] == "knowledge-base"
        listed = self._rpc(http, "tools/list")
        assert {t["name"] for t in listed.json()["result"]["tools"]} == TOOLS
        called = self._rpc(
            http,
            "tools/call",
            name="search_knowledge_base",
            arguments={"query": EVERYTHING, "use_hybrid": True},
        )
        passages = called.json()["result"]["structuredContent"]["passages"]
        assert {p["document"] for p in passages} == {"travel.txt"}  # salaries.txt is restricted

    def test_the_api_key_is_required(self, http: TestClient) -> None:
        assert http.post("/mcp", headers=self.HEADERS, json={}).status_code == 401
        assert self._rpc(http, "tools/list", key="wrong").status_code == 403

    def test_unknown_hosts_are_refused(self, http: TestClient) -> None:
        response = http.post(
            "/mcp",
            headers={**self.HEADERS, "authorization": "Bearer secret-key", "host": "evil.test"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert response.status_code == 421

    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "mcp_enabled", False)
        with TestClient(api_app.app) as client:
            assert client.post("/mcp", headers=self.HEADERS, json={}).status_code == 404


def test_allowed_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mcp_allowed_hosts", "rag.example.com, ")
    security = mcp_server.transport_security()
    assert security.enable_dns_rebinding_protection
    assert {"rag.example.com", "rag.example.com:*", "localhost:*"} <= set(security.allowed_hosts)
    assert "https://rag.example.com" in security.allowed_origins
    monkeypatch.setattr(settings, "mcp_allowed_hosts", "*")
    assert not mcp_server.transport_security().enable_dns_rebinding_protection
