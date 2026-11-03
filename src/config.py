"""Application configuration via environment variables with pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ---------- Logging ----------
    log_level: str = "INFO"

    # ---------- API / CLI ----------
    api_url: str = "http://localhost:8000"


settings = Settings()
