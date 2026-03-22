from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.llm_runtime import LLMInvocationError, LLMUnavailableError, is_llm_available
from backend.api.auth import verify_api_key
from backend.api.cache import result_cache
from backend.api.chat_service import run_chat_query
from backend.api.job_store import job_store
from backend.api.models import (
    AnalysisStatus,
    AnalysisStatusResponse,
    AnalyzePaperRequest,
    AnalyzePaperResponse,
    HealthComponentResponse,
    CacheMetricsResponse,
    ChatRequest,
    ChatResponse,
    FaissMaintenanceResponse,
    GraphReviewDecisionResponse,
    GraphReviewItemResponse,
    GraphReviewListResponse,
    IndexPaperResponse,
    IndexPapersRequest,
    IndexPapersResponse,
    IndexPaperResult,
    IndexMetricsSnapshotResponse,
    IndexStatsResponse,
    JobMetricsResponse,
    JobCancelResponse,
    MetricsSummaryResponse,
    ReadinessResponse,
    RelatedPapersPreviewResponse,
    ReviewMetricsResponse,
    RelatedPaper,
    ResolvePaperResponse,
    QueueStatsResponse,
)
from backend.api.queue_worker import BackgroundJobQueue
from backend.api.pipeline_service import ingest_paper_for_retrieval, run_fast_stage, run_full_stage
from backend.tools.citation_tools import fetch_citations, start_citation_crawl_async
from backend.core.config import get_settings
from backend.db.faiss_store import get_faiss_store
from backend.db.neo4j_client import get_neo4j_client
from backend.review_store import GraphReviewRecord, graph_review_store
from backend.tools.graph_tools import update_graph
from backend.tools.metadata_tools import resolve_paper_metadata, search_related_papers_by_title

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    job_queue.start()
    try:
        yield
    finally:
        job_queue.stop()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(request.headers.get("X-Request-ID", "")).strip() or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
                          request_id, request.method, request.url.path, duration_ms)
        raise
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info("request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                 request_id, request.method, request.url.path, response.status_code, duration_ms)
    return response


def _queue_stats_payload() -> QueueStatsResponse:
    stats = job_queue.stats()
    return QueueStatsResponse(
        queue_depth=stats["queue_depth"], active_jobs=stats["active_jobs"],
        worker_count=stats["worker_count"], max_queue_size=stats["max_queue_size"],
        cached_results=result_cache.size(), canceled_jobs=stats["canceled_jobs"],
    )


def _review_item_response(record: GraphReviewRecord) -> GraphReviewItemResponse:
    payload = dict(record.payload)
    return GraphReviewItemResponse(
        review_id=record.review_id, status=record.status,
        user_id=str(payload.get("user_id", "")).strip(),
        paper_url=str(payload.get("paper_url", "")).strip(),
        title=str(payload.get("title", "")).strip(),
        concepts=list(payload.get("concepts", [])),
        methods=list(payload.get("methods", [])),
        datasets=list(payload.get("datasets", [])),
        note=record.note, created_at=record.created_at, decided_at=record.decided_at,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _readiness_payload() -> ReadinessResponse:
    components: dict[str, HealthComponentResponse] = {}
    queue_running = job_queue.is_running()
    queue_stats = job_queue.stats()
    components["queue"] = HealthComponentResponse(
        ok=queue_running,
        detail=f"workers={queue_stats['worker_count']}, queue_depth={queue_stats['queue_depth']}, active_jobs={queue_stats['active_jobs']}",
    )
    try:
        store = get_faiss_store()
        components["vector_store"] = HealthComponentResponse(
            ok=True, detail=f"vectors={store.vector_count()}, unique_papers={store.unique_paper_count()}")
    except Exception as exc:
        components["vector_store"] = HealthComponentResponse(ok=False, detail=str(exc))

    review_path = graph_review_store.storage_path
    components["review_store"] = HealthComponentResponse(
        ok=review_path.parent.exists() and review_path.parent.is_dir(),
        detail=f"path={review_path.as_posix()}")

    graph_status = get_neo4j_client().healthcheck()
    components["graph_store"] = HealthComponentResponse(
        ok=bool(graph_status.get("ok", False)), detail=str(graph_status.get("detail", "")))

    ready = all(components[key].ok for key in ("queue", "vector_store", "review_store"))
    degraded = not components["graph_store"].ok
    status = "ok" if ready and not degraded else "degraded" if ready else "not_ready"
    return ReadinessResponse(status=status, ready=ready, degraded=degraded,
                              app_name=settings.app_name, app_env=settings.app_env,
                              components=components)


@app.get("/health/ready", response_model=ReadinessResponse)
def health_ready() -> ReadinessResponse:
    return _readiness_payload()


def _resolve_llm_key(x_llm_api_key: str | None, x_openai_api_key: str | None) -> str:
    return str(x_llm_api_key or x_openai_api_key or "").strip()


def process_background_job(job_id: str) -> None:
    record = job_store.get(job_id)
    if record is None or record.status == AnalysisStatus.canceled:
        return
    try:
        queue_wait = max(0.0, time.time() - record.created_at)
        if queue_wait > settings.fast_stage_timeout_seconds:
            logger.warning("Job %s spent %.2fs in queue", job_id, queue_wait)

        job_store.set_status(job_id, AnalysisStatus.running)
        payload = AnalyzePaperRequest(**record.request_payload)
        # Support both old (openai_api_key) and new (llm_api_key) runtime context keys
        llm_api_key = str(
            record.runtime_context.get("llm_api_key") or
            record.runtime_context.get("openai_api_key") or ""
        ).strip()

        fast_stage = record.fast_stage or run_fast_stage(payload, openai_api_key=llm_api_key)

        start_full = time.monotonic()
        full_stage = run_full_stage(payload, fast_stage, openai_api_key=llm_api_key)
        full_stage_duration = time.monotonic() - start_full

        job_store.set_full_stage(job_id, full_stage)
        job_store.set_status(job_id, AnalysisStatus.complete)
        result_cache.set(payload, fast_stage=fast_stage, full_stage=full_stage,
                          openai_api_key=llm_api_key)

        if full_stage_duration > settings.full_stage_timeout_seconds:
            logger.warning("Job %s full-stage latency %.2fs exceeded target %ss",
                            job_id, full_stage_duration, settings.full_stage_timeout_seconds)
    except Exception as exc:
        logger.exception("Background job failed: %s", job_id)
        job_store.set_error(job_id, str(exc))


def on_job_canceled(job_id: str, reason: str) -> None:
    job_store.set_canceled(job_id, detail=reason)


job_queue = BackgroundJobQueue(
    process_job=process_background_job,
    max_size=settings.queue_max_size,
    worker_count=settings.queue_worker_count,
    on_cancel=on_job_canceled,
)


@app.post("/analyze-paper", response_model=AnalyzePaperResponse)
def analyze_paper(
    payload: AnalyzePaperRequest,
    _auth: None = Depends(verify_api_key),
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-Api-Key"),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
) -> AnalyzePaperResponse:
    llm_api_key = _resolve_llm_key(x_llm_api_key, x_openai_api_key)
    llm_provider = str(x_llm_provider or "").strip() or None
    runtime_context = {"openai_api_key": llm_api_key, "llm_api_key": llm_api_key,
                        "llm_provider": llm_provider}

    cached = result_cache.get(payload, openai_api_key=llm_api_key)
    if cached is not None:
        cached_record = job_store.create_job(request_payload=payload.model_dump(),
                                              fast_stage=cached.fast_stage,
                                              runtime_context=runtime_context)
        job_store.set_full_stage(cached_record.job_id, cached.full_stage)
        job_store.set_status(cached_record.job_id, AnalysisStatus.complete)
        return AnalyzePaperResponse(
            job_id=cached_record.job_id, status=AnalysisStatus.complete,
            summary=cached.fast_stage.summary, key_concepts=cached.fast_stage.key_concepts,
            poll_url=f"/analysis-status/{cached_record.job_id}", from_cache=True)

    fast_start = time.monotonic()
    fast_stage = run_fast_stage(payload, openai_api_key=llm_api_key)
    fast_stage_duration = time.monotonic() - fast_start
    if fast_stage_duration > settings.fast_stage_timeout_seconds:
        logger.warning("Fast-stage latency %.2fs exceeded target %ss for %s",
                        fast_stage_duration, settings.fast_stage_timeout_seconds, payload.paper_url)

    record = job_store.create_job(payload.model_dump(), fast_stage=fast_stage,
                                   runtime_context=runtime_context)
    if not job_queue.enqueue(record.job_id, priority=payload.priority):
        job_store.set_error(record.job_id, "Request queue is full")
        raise HTTPException(status_code=429, detail="Server queue is full. Try again shortly.")

    return AnalyzePaperResponse(
        job_id=record.job_id, status=AnalysisStatus.partial,
        summary=fast_stage.summary, key_concepts=fast_stage.key_concepts,
        poll_url=f"/analysis-status/{record.job_id}", from_cache=False)


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    _auth: None = Depends(verify_api_key),
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-Api-Key"),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
) -> ChatResponse:
    llm_api_key = _resolve_llm_key(x_llm_api_key, x_openai_api_key)
    llm_provider = str(x_llm_provider or "").strip() or None

    if settings.strict_llm_mode and not llm_api_key:
        raise HTTPException(status_code=503,
                             detail="No LLM API key provided. Add one in the extension Settings tab.")
    try:
        return run_chat_query(payload, openai_api_key=llm_api_key,
                               llm_api_key=llm_api_key, llm_provider=llm_provider)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMInvocationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index-paper", response_model=IndexPaperResponse)
def index_paper(
    payload: AnalyzePaperRequest,
    _auth: None = Depends(verify_api_key),
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-Api-Key"),
) -> IndexPaperResponse:
    llm_api_key = _resolve_llm_key(x_llm_api_key, x_openai_api_key)
    result = ingest_paper_for_retrieval(payload, openai_api_key=llm_api_key)
    return IndexPaperResponse(paper_url=payload.paper_url,
                               indexed_chunks=int(result.get("indexed_chunks", 0)),
                               already_indexed=bool(result.get("already_indexed", False)))


@app.get("/resolve-paper", response_model=ResolvePaperResponse)
def resolve_paper(paper_url: str, title: str = "", abstract: str = "",
                   _auth: None = Depends(verify_api_key)) -> ResolvePaperResponse:
    normalized_url = str(paper_url or "").strip()
    if not normalized_url:
        raise HTTPException(status_code=400, detail="paper_url is required")
    resolved = resolve_paper_metadata(paper_url=normalized_url, title=title, abstract=abstract)
    providers_raw = str(resolved.get("providers", "")).strip()
    providers = [item.strip() for item in providers_raw.split(",") if item.strip()]
    resolved_url = (str(resolved.get("pdf_url", "")).strip() or
                    str(resolved.get("canonical_url", "")).strip() or normalized_url)
    return ResolvePaperResponse(
        paper_url=normalized_url, resolved_url=resolved_url,
        title=str(resolved.get("title", "")).strip(),
        abstract=str(resolved.get("abstract", "")).strip(),
        source=str(resolved.get("source", "unknown")).strip() or "unknown",
        provider=str(resolved.get("provider", "")).strip(),
        providers=providers,
        canonical_url=str(resolved.get("canonical_url", "")).strip() or None,
        pdf_url=str(resolved.get("pdf_url", "")).strip() or None,
        arxiv_id=str(resolved.get("arxiv_id", "")).strip() or None,
        doi=str(resolved.get("doi", "")).strip() or None,
    )


@app.get("/related-papers-preview", response_model=RelatedPapersPreviewResponse)
def related_papers_preview(title: str, paper_url: str = "", abstract: str = "",
                            limit: int = 5, _auth: None = Depends(verify_api_key)) -> RelatedPapersPreviewResponse:
    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise HTTPException(status_code=400, detail="title is required")
    result = search_related_papers_by_title(normalized_title, paper_url=str(paper_url or "").strip(),
                                             abstract=str(abstract or "").strip(),
                                             limit=min(max(1, int(limit)), 8))
    return RelatedPapersPreviewResponse(
        title=normalized_title, provider=str(result.get("provider", "")).strip(),
        related_papers=[
            RelatedPaper(title=str(item.get("title", "Untitled")),
                          url=str(item.get("url", "")).strip(),
                          score=float(item.get("score", 0.0) or 0.0))
            for item in list(result.get("related_papers", []))
        ])


@app.get("/citation-papers")
def citation_papers(
    paper_url: str,
    user_id: str = "local-user",
    limit: int = 15,
    _auth: None = Depends(verify_api_key),
) -> dict:
    """
    Fetch direct citations of a paper from Semantic Scholar.
    Returns them immediately for the mind map.
    Each citation is also asynchronously indexed into FAISS and written
    to Neo4j as a CITES edge so RAG can retrieve their content.
    """
    url = str(paper_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="paper_url is required")

    citations = fetch_citations(url, limit=min(int(limit), 30))

    # Persist each citation: FAISS + Neo4j CITES edges (background, non-blocking)
    if citations:
        start_citation_crawl_async(
            root_url=url,
            root_title="",
            user_id=str(user_id or "local-user"),
        )

    valid = [c for c in citations if c.get("url") and c.get("title")]
    return {
        "paper_url": url,
        "citations": [
            {"title": c["title"], "url": c["url"], "score": 1.0}
            for c in valid[:limit]
        ],
        "total": len(valid),
    }


@app.post("/index-papers", response_model=IndexPapersResponse)
def index_papers(
    payload: IndexPapersRequest,
    _auth: None = Depends(verify_api_key),
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-Api-Key"),
) -> IndexPapersResponse:
    llm_api_key = _resolve_llm_key(x_llm_api_key, x_openai_api_key)
    results: list[IndexPaperResult] = []
    indexed_count = skipped_count = failed_count = 0

    for paper in payload.papers:
        try:
            result = ingest_paper_for_retrieval(paper, openai_api_key=llm_api_key)
            indexed_chunks = int(result.get("indexed_chunks", 0))
            already_indexed = bool(result.get("already_indexed", False))
            if already_indexed:
                skipped_count += 1
            else:
                indexed_count += indexed_chunks
            results.append(IndexPaperResult(paper_url=paper.paper_url,
                                             indexed_chunks=indexed_chunks,
                                             already_indexed=already_indexed))
        except Exception as exc:
            failed_count += 1
            results.append(IndexPaperResult(paper_url=paper.paper_url, indexed_chunks=0,
                                             already_indexed=False, error=str(exc)))

    return IndexPapersResponse(indexed_count=indexed_count, skipped_count=skipped_count,
                                failed_count=failed_count, results=results)


@app.get("/index-stats", response_model=IndexStatsResponse)
def index_stats(user_id: str | None = None, _auth: None = Depends(verify_api_key)) -> IndexStatsResponse:
    store = get_faiss_store()
    return IndexStatsResponse(vectors=store.vector_count(user_id=user_id),
                               unique_papers=store.unique_paper_count(user_id=user_id))


@app.get("/queue-stats", response_model=QueueStatsResponse)
def queue_stats(_auth: None = Depends(verify_api_key)) -> QueueStatsResponse:
    return _queue_stats_payload()


@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(_auth: None = Depends(verify_api_key)) -> MetricsSummaryResponse:
    store = get_faiss_store()
    return MetricsSummaryResponse(
        app_name=settings.app_name, app_env=settings.app_env,
        llm_provider=settings.llm_provider, llm_model=settings.llm_model,
        strict_llm_mode=settings.strict_llm_mode, require_graph_review=settings.require_graph_review,
        queue=_queue_stats_payload(),
        caches=CacheMetricsResponse(analysis_results=result_cache.size()),
        index=IndexMetricsSnapshotResponse(vectors=store.vector_count(),
                                            unique_papers=store.unique_paper_count()),
        jobs=JobMetricsResponse(**job_store.metrics_snapshot()),
        reviews=ReviewMetricsResponse(**graph_review_store.counts()),
    )


@app.post("/maintenance/faiss", response_model=FaissMaintenanceResponse)
def maintenance_faiss(_auth: None = Depends(verify_api_key)) -> FaissMaintenanceResponse:
    return FaissMaintenanceResponse(**get_faiss_store().compact())


@app.get("/graph-reviews", response_model=GraphReviewListResponse)
def graph_reviews(status: str | None = None,
                   _auth: None = Depends(verify_api_key)) -> GraphReviewListResponse:
    return GraphReviewListResponse(
        reviews=[_review_item_response(item) for item in graph_review_store.list(status=status)])


@app.post("/graph-reviews/{review_id}/approve", response_model=GraphReviewDecisionResponse)
def approve_graph_review(review_id: str, _auth: None = Depends(verify_api_key)) -> GraphReviewDecisionResponse:
    record = graph_review_store.get(review_id)
    if record is None:
        raise HTTPException(status_code=404, detail="review not found")
    if record.status != "pending":
        return GraphReviewDecisionResponse(review_id=review_id, status=record.status,
                                            detail=f"Review already {record.status}.")
    payload = dict(record.payload)
    result = update_graph(user_id=str(payload.get("user_id", "")).strip(),
                           paper_url=str(payload.get("paper_url", "")).strip(),
                           title=str(payload.get("title", "")).strip(),
                           concepts=list(payload.get("concepts", [])),
                           methods=list(payload.get("methods", [])),
                           datasets=list(payload.get("datasets", [])))
    if not bool(result.get("graph_updated", False)):
        raise HTTPException(status_code=502, detail=str(result.get("error", "graph update failed")))
    graph_review_store.set_status(review_id, "approved", note="Approved and applied.")
    return GraphReviewDecisionResponse(review_id=review_id, status="approved",
                                        detail="Graph review approved and graph updated.")


@app.post("/graph-reviews/{review_id}/reject", response_model=GraphReviewDecisionResponse)
def reject_graph_review(review_id: str, _auth: None = Depends(verify_api_key)) -> GraphReviewDecisionResponse:
    record = graph_review_store.get(review_id)
    if record is None:
        raise HTTPException(status_code=404, detail="review not found")
    if record.status != "pending":
        return GraphReviewDecisionResponse(review_id=review_id, status=record.status,
                                            detail=f"Review already {record.status}.")
    graph_review_store.set_status(review_id, "rejected", note="Rejected by reviewer.")
    return GraphReviewDecisionResponse(review_id=review_id, status="rejected",
                                        detail="Graph review rejected.")


@app.post("/cancel-job/{job_id}", response_model=JobCancelResponse)
def cancel_job(job_id: str, _auth: None = Depends(verify_api_key)) -> JobCancelResponse:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    if record.status in (AnalysisStatus.complete, AnalysisStatus.failed, AnalysisStatus.canceled):
        return JobCancelResponse(job_id=job_id, canceled=False,
                                  detail=f"Job already in terminal state: {record.status}")
    if not job_queue.cancel(job_id):
        return JobCancelResponse(job_id=job_id, canceled=False,
                                  detail="Job is already running and cannot be canceled.")
    job_store.set_canceled(job_id, detail="Job canceled by user")
    return JobCancelResponse(job_id=job_id, canceled=True, detail="Job canceled successfully.")


@app.get("/analysis-status/{job_id}", response_model=AnalysisStatusResponse)
def analysis_status(job_id: str, _auth: None = Depends(verify_api_key)) -> AnalysisStatusResponse:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    fast_stage = record.fast_stage
    full_stage = record.full_stage
    return AnalysisStatusResponse(
        job_id=record.job_id, status=record.status,
        summary=fast_stage.summary if fast_stage else None,
        key_concepts=fast_stage.key_concepts if fast_stage else [],
        methods=full_stage.methods if full_stage else [],
        datasets=full_stage.datasets if full_stage else [],
        related_papers=full_stage.related_papers if full_stage else [],
        research_connections=full_stage.research_connections if full_stage else [],
        user_interest_topics=full_stage.user_interest_topics if full_stage else [],
        research_gaps=full_stage.research_gaps if full_stage else [],
        ideas=full_stage.ideas if full_stage else [],
        planner=full_stage.planner if full_stage else {},
        autonomy_notes=full_stage.autonomy_notes if full_stage else [],
        confidence=full_stage.confidence if full_stage else 0.0,
        review_required=full_stage.review_required if full_stage else False,
        review_id=full_stage.review_id if full_stage else None,
        error=record.error,
    )
