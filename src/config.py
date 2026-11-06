"""Application configuration via environment variables with pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

# Collection and role names: lowercase, so "HR" and "hr" can't become two different roles.
AccessName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]


class Settings(BaseSettings):
    """Central configuration for the RAG pipeline.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Paths ----------
    data_dir: Path = Path("data")
    chroma_path: Path = Path("data/chroma_db")
    chroma_host: str | None = None
    chroma_port: int | None = None
    golden_dataset_path: Path = Path("data/golden_dataset/dataset.jsonl")
    eval_results_path: Path = Path("data/eval_results")

    # ---------- Embedding model ----------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    # LRU cache size for repeated/paraphrased query embeddings (0 disables caching).
    embedding_query_cache_size: int = 256

    # ---------- Chunking ----------
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ---------- Ingestion limits ----------
    max_file_size_mb: float = 50.0
    max_pdf_pages: int = 1000

    # ---------- Generation limits ----------
    # Caps the total characters of retrieved context sent to the LLM, as a
    # cost/runaway-prompt guardrail.
    max_context_chars: int = 12000
    # Overall limit for one query, across retries and every self-healing attempt.
    # Each LLM call also has its own client timeout. 0 disables this limit.
    query_timeout_seconds: float = 180.0

    # ---------- Retrieval ----------
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    top_k_final: int = 5
    # Upper bound for each Chroma add() call during ingestion. The effective
    # value is also capped by Chroma's own advertised maximum.
    ingest_batch_size: int = Field(default=2000, ge=1)
    # Drop chunks that repeat content already indexed from another document. Exact duplicates
    # use content_fingerprint; near duplicates use the embedding model and this cosine-like
    # similarity score. Keep the default high so overlapping chunks in one document survive.
    semantic_deduplication: bool = True
    semantic_deduplication_threshold: float = Field(default=0.985, ge=0.9, le=1.0)

    # Hybrid search (Phase 2)
    hybrid_alpha: float = 0.6  # weight for vector search (1.0 = pure vector, 0.0 = pure BM25)
    rrf_k: int = 60  # constant for Reciprocal Rank Fusion
    # HNSW index (ChromaDB). ef_search applies on startup; M / ef_construction only to new collections.
    hnsw_ef_search: int = 200
    hnsw_max_neighbors: int = 16
    hnsw_ef_construction: int = 100
    bm25_max_index_mb: int = 500  # max BM25 index file; ~1 KB/chunk, so ~2000 for 1M chunks

    # ---------- Re-ranker (Phase 2) ----------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------- LLM ----------
    # Deliberately uses the conventional, un-prefixed variable so the OpenAI
    # SDK and other tooling can share the same .env file.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY", repr=False)
    openrouter_mode: str = "auto"
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---------- Evaluation (Phase 3) ----------
    # Also used by the self-healing verifier as its faithfulness pass bar.
    faithfulness_threshold: float = 0.7
    eval_llm_model: str = "gpt-4o-mini"

    # ---------- Self-healing (Phase 4) ----------
    # Off by default so existing clients keep today's latency/cost profile;
    # enable globally here or per request (`self_heal: true` on /query).
    self_healing_enabled: bool = False
    # Total generate→verify attempts per query, including the first (3 = up to 2 repairs).
    max_healing_attempts: int = Field(default=3, ge=1, le=6)
    # LLM-judge part of the verifier. Deterministic checks always run.
    verifier_enabled: bool = True
    verifier_llm_model: str | None = None  # defaults to llm_model
    relevance_threshold: float = 0.7
    # Floor for the top vector-similarity / cross-encoder score. Not applied to
    # RRF scores, which are rank-derived rather than absolute.
    min_retrieval_score: float = 0.2
    # Fraction of the query's content terms that must appear in retrieved text.
    min_query_coverage: float = 0.4
    # Fraction of cited sentences that must lexically overlap their cited passage.
    min_citation_support: float = 0.5
    healing_llm_rewrite: bool = True
    healing_fetch_k_multiplier: float = 2.0
    healing_max_fetch_k: int = 100
    healing_context_budget_multiplier: float = 1.5
    healing_escalate_hybrid: bool = True
    # Loading the cross-encoder is expensive; only turn it on during healing
    # retries when explicitly allowed.
    healing_rerank_on_retry: bool = False

    # ---------- Document versions and authority ----------
    # How far to trust each source, 0 to 100, for resolving conflicts between documents.
    # Keys are folders inside data_dir, as JSON, e.g. {"policies": 90, "uploads": 30}.
    # The longest matching folder wins; anything else gets the default.
    source_authority: dict[str, Annotated[int, Field(ge=0, le=100)]] = {}
    default_source_authority: int = Field(default=50, ge=0, le=100)
    # Which logical collection a folder's documents belong to, and which roles may read them,
    # also as JSON by folder, e.g. {"policies/hr": "hr-policies"} and {"policies/hr": ["hr"]}.
    # Documents in no folder with roles are readable as today, subject to their owner.
    source_collections: dict[str, AccessName] = {}
    source_roles: dict[str, list[AccessName]] = {}

    # ---------- Screening text before indexing ----------
    # Credentials (API keys, private keys, tokens, passwords) are replaced before indexing.
    redact_credentials: bool = True
    # Personal data (emails, phones, card numbers, IBANs, SSNs, IP addresses): "flag" records
    # what each chunk contains and keeps the text, "redact" replaces it, "off" does neither.
    pii_mode: Literal["off", "flag", "redact"] = "flag"
    # Instruction-like text ("ignore previous instructions"): "flag" records it on the chunk,
    # "exclude" keeps the whole parent section, and its chunks, out of the index.
    injection_mode: Literal["flag", "exclude"] = "flag"
    # Sources at or above this authority are recorded but never treated as suspect, so a
    # security team's own documents about prompt injection aren't flagged or excluded.
    injection_trusted_authority: int | None = Field(default=None, ge=0, le=100)
    # Redact credentials and personal data from answers, citations, and full passages before
    # they are returned, whatever was stored. Streaming then sends the answer once it's whole.
    redact_answers: bool = False

    # ---------- Accounts and chat history ----------
    # SQLite file holding accounts and chats. Defaults to <data_dir>/chat.db.
    chat_db_path: Path | None = None
    # Turn off once accounts exist; scripts/manage_users.py can still create them.
    allow_signup: bool = True
    min_password_length: int = Field(default=8, ge=8)
    session_ttl_hours: float = Field(default=720.0, gt=0)
    # Always set on requests that arrive over HTTPS. Turn on when a proxy terminates
    # TLS and forwards plain HTTP, since the app can't see the original scheme there.
    session_cookie_secure: bool = False
    # Sign-ins and registrations from one address. Behind a proxy this needs uvicorn's
    # FORWARDED_ALLOW_IPS, or every user shares the proxy's address and this one limit.
    auth_ip_max_attempts: int = Field(default=20, ge=1)
    auth_ip_window_seconds: float = Field(default=300.0, gt=0)
    # `reset` on /ingest deletes every user's documents, so the API refuses it unless enabled.
    allow_store_reset: bool = False

    # ---------- Logging ----------
    log_level: str = "INFO"

    # ---------- API / CLI ----------
    api_url: str = "http://localhost:8000"

    @property
    def chat_db_file(self) -> Path:
        return self.chat_db_path or self.data_dir / "chat.db"


settings = Settings()
