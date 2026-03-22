from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    queued = "queued"
    running = "running"
    partial = "partial"
    complete = "complete"
    failed = "failed"
    canceled = "canceled"


class AnalyzePaperRequest(BaseModel):
    paper_url: str = Field(min_length=1, max_length=2048)
    title: str = Field(min_length=1, max_length=512)
    abstract: str = Field(default="", max_length=30000)
    user_id: str = Field(default="local-user", min_length=1, max_length=128)
    priority: int = Field(default=5, ge=1, le=10)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    user_id: str = Field(default="local-user", min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=5000)
    paper_url: str | None = Field(default=None, max_length=2048)
    paper_title: str | None = Field(default=None, max_length=512)
    paper_abstract: str | None = Field(default=None, max_length=30000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=4, ge=1, le=8)
    ensure_current_paper_indexed: bool = False
    current_paper_only: bool = False


class ChatCitation(BaseModel):
    title: str
    url: str
    section: str | None = None
    score: float | None = None


class ChatEvidenceSnippet(BaseModel):
    title: str
    url: str
    section: str | None = None
    snippet: str
    score: float | None = None


class ChatAgentStep(BaseModel):
    agent: str
    status: str = "completed"
    detail: str = ""
    used_llm: bool = False
    observations: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    evidence_snippets: list[ChatEvidenceSnippet] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    used_context_chunks: int = 0
    support_score: float = 0.0
    retrieval_mode: str = "lexical"
    graph_connections: list[str] = Field(default_factory=list)
    user_interest_topics: list[str] = Field(default_factory=list)
    autonomy_notes: list[str] = Field(default_factory=list)
    agent_trace: list[ChatAgentStep] = Field(default_factory=list)


class IndexPaperResponse(BaseModel):
    paper_url: str
    indexed_chunks: int
    already_indexed: bool


class IndexPapersRequest(BaseModel):
    papers: list[AnalyzePaperRequest] = Field(default_factory=list, max_length=100)


class IndexPaperResult(BaseModel):
    paper_url: str
    indexed_chunks: int
    already_indexed: bool
    error: str | None = None


class IndexPapersResponse(BaseModel):
    indexed_count: int
    skipped_count: int
    failed_count: int
    results: list[IndexPaperResult] = Field(default_factory=list)


class IndexStatsResponse(BaseModel):
    vectors: int
    unique_papers: int


class ResolvePaperResponse(BaseModel):
    paper_url: str
    resolved_url: str
    title: str
    abstract: str
    source: str
    provider: str
    providers: list[str] = Field(default_factory=list)
    canonical_url: str | None = None
    pdf_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None


class HealthComponentResponse(BaseModel):
    ok: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    degraded: bool = False
    app_name: str
    app_env: str
    components: dict[str, HealthComponentResponse] = Field(default_factory=dict)


class QueueStatsResponse(BaseModel):
    queue_depth: int
    active_jobs: int
    worker_count: int
    max_queue_size: int
    cached_results: int
    canceled_jobs: int


class JobCancelResponse(BaseModel):
    job_id: str
    canceled: bool
    detail: str


class RelatedPaper(BaseModel):
    title: str
    url: str
    score: float | None = None


class RelatedPapersPreviewResponse(BaseModel):
    title: str
    provider: str
    related_papers: list[RelatedPaper] = Field(default_factory=list)


class FastStageResult(BaseModel):
    summary: str
    key_concepts: list[str] = Field(default_factory=list)


class FullStageResult(BaseModel):
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    related_papers: list[RelatedPaper] = Field(default_factory=list)
    research_connections: list[str] = Field(default_factory=list)
    user_interest_topics: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    ideas: list[str] = Field(default_factory=list)
    planner: dict[str, Any] = Field(default_factory=dict)
    autonomy_notes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    review_required: bool = False
    review_id: str | None = None


class AnalyzePaperResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    summary: str
    key_concepts: list[str] = Field(default_factory=list)
    poll_url: str
    from_cache: bool = False


class AnalysisStatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    summary: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    related_papers: list[RelatedPaper] = Field(default_factory=list)
    research_connections: list[str] = Field(default_factory=list)
    user_interest_topics: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    ideas: list[str] = Field(default_factory=list)
    planner: dict[str, Any] = Field(default_factory=dict)
    autonomy_notes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    review_required: bool = False
    review_id: str | None = None
    error: str | None = None


class FaissMaintenanceResponse(BaseModel):
    vectors_before: int
    vectors_after: int
    duplicates_removed: int
    unique_papers_after: int


class JobMetricsResponse(BaseModel):
    total_jobs: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    average_queue_wait_seconds: float = 0.0
    average_run_duration_seconds: float = 0.0


class CacheMetricsResponse(BaseModel):
    analysis_results: int


class IndexMetricsSnapshotResponse(BaseModel):
    vectors: int
    unique_papers: int


class ReviewMetricsResponse(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int


class MetricsSummaryResponse(BaseModel):
    app_name: str
    app_env: str
    llm_provider: str
    llm_model: str
    strict_llm_mode: bool
    require_graph_review: bool
    queue: QueueStatsResponse
    caches: CacheMetricsResponse
    index: IndexMetricsSnapshotResponse
    jobs: JobMetricsResponse
    reviews: ReviewMetricsResponse


class GraphReviewItemResponse(BaseModel):
    review_id: str
    status: str
    user_id: str
    paper_url: str
    title: str
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    note: str = ""
    created_at: float
    decided_at: float | None = None


class GraphReviewListResponse(BaseModel):
    reviews: list[GraphReviewItemResponse] = Field(default_factory=list)


class GraphReviewDecisionResponse(BaseModel):
    review_id: str
    status: str
    detail: str
