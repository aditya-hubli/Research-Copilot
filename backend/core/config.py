from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Personal Research Copilot API"
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "huggingface"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # API keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    huggingface_api_key: str | None = None

    strict_llm_mode: bool = False
    agent_retry_limit: int = 2
    retrieval_query_expansions: int = 3
    retrieval_candidate_multiplier: int = 4

    # ── Auth ─────────────────────────────────────────────────────────────────
    require_api_key: bool = False
    backend_api_key: str | None = None
    require_graph_review: bool = False

    # ── Token / chunk budgets ─────────────────────────────────────────────────
    max_prompt_tokens: int = 3500
    max_chunks: int = 6
    default_chunk_size: int = 400
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    pdf_extract_max_pages: int = 25

    # ── Stage timeouts ────────────────────────────────────────────────────────
    fast_stage_timeout_seconds: int = 3
    full_stage_timeout_seconds: int = 30

    # ── Queue ─────────────────────────────────────────────────────────────────
    queue_max_size: int = 200
    queue_worker_count: int = 2

    # ── Caches ───────────────────────────────────────────────────────────────
    analysis_cache_ttl_seconds: int = 21600
    analysis_cache_max_entries: int = 4000
    embedding_cache_ttl_seconds: int = 86400
    embedding_cache_max_entries: int = 10000
    metadata_cache_ttl_seconds: int = 21600
    metadata_cache_max_entries: int = 10000

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_dimension: int = 256
    faiss_data_dir: str = "backend/data/faiss"
    graph_review_data_path: str = "backend/data/reviews/reviews.json"

    # ── Citation crawl ────────────────────────────────────────────────────────
    citation_crawl_enabled: bool = True
    citation_crawl_depth: int = 2       # hops from root paper (1 = direct refs only)
    citation_crawl_max_papers: int = 20 # max new papers to index per crawl session

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "researchcopilot"

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
