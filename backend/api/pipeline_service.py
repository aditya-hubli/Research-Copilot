from __future__ import annotations

import logging
import re
from typing import Any

from backend.agents.llm_runtime import call_structured_agent
from backend.agents.prompts import PARSER_SYSTEM_PROMPT
from backend.agents.workflow import run_agent_workflow
from backend.tools.citation_tools import start_citation_crawl_async
from backend.api.models import (
    AnalyzePaperRequest,
    FastStageResult,
    FullStageResult,
    RelatedPaper,
)
from backend.core.config import get_settings
from backend.pipeline.chunker import semantic_chunk_document
from backend.pipeline.token_budget import trim_to_token_limit
from backend.tools.metadata_tools import get_paper_metadata, resolve_paper_metadata
from backend.tools.pdf_tools import extract_pdf_text
from backend.tools.vector_tools import (
    add_chunks_to_index,
    has_indexed_paper,
    retrieve_context_chunks,
)

logger = logging.getLogger(__name__)

# NOTE: Do NOT cache settings at module level — always call get_settings() inside
# functions so the lru_cache on get_settings() is used correctly.

_METHOD_HINTS = (
    "transformer", "attention mechanism", "graph neural network", "gnn",
    "multi-agent", "reinforcement learning", "rlhf", "contrastive learning",
    "diffusion model", "variational autoencoder", "vae", "bayesian",
    "chain-of-thought", "instruction tuning", "in-context learning",
    "retrieval-augmented generation", "rag", "knowledge distillation",
    "mixture of experts", "moe", "cnn", "rnn", "lstm",
)
_DATASET_HINTS = (
    "imagenet", "cifar", "coco", "mnist", "squad", "glue", "superglue",
    "humaneval", "gsm8k", "mmlu", "big-bench", "wikitext", "pubmed",
    "openreview", "arxiv", "hellaswag", "truthfulqa",
)
_STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "into", "using",
    "based", "towards", "study", "paper", "approach", "method", "model",
    "show", "shows", "propose", "proposes", "present", "presents",
    "novel", "new", "state", "sota", "results", "work",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_like_direct_pdf(url: str) -> bool:
    n = str(url or "").strip().lower()
    return n.endswith(".pdf") or ".pdf?" in n or ".pdf#" in n


def _clamp_chunk_size(value: int) -> int:
    settings = get_settings()
    return min(max(value, 300), settings.max_chunk_size)


def _resolve_index_inputs(payload: AnalyzePaperRequest) -> dict[str, str]:
    if _looks_like_direct_pdf(payload.paper_url):
        return {
            "paper_url": payload.paper_url,
            "title": str(payload.title or "").strip(),
            "abstract": str(payload.abstract or "").strip(),
            "provider": "direct-pdf",
        }
    resolved = resolve_paper_metadata(
        paper_url=payload.paper_url,
        title=payload.title,
        abstract=payload.abstract,
    )
    effective_url = (
        str(resolved.get("pdf_url", "")).strip()
        or str(resolved.get("canonical_url", "")).strip()
        or payload.paper_url
    )
    return {
        "paper_url": effective_url,
        "title": str(payload.title or "").strip() or str(resolved.get("title", "")).strip(),
        "abstract": str(payload.abstract or "").strip() or str(resolved.get("abstract", "")).strip(),
        "provider": str(resolved.get("provider", "")).strip(),
    }


def _prepare_chunks_for_index(
    payload: AnalyzePaperRequest,
    resolved_inputs: dict[str, str],
) -> list[dict[str, str | int]]:
    settings = get_settings()
    effective_url   = resolved_inputs["paper_url"]
    effective_title    = resolved_inputs["title"] or payload.title
    effective_abstract = resolved_inputs["abstract"] or payload.abstract

    get_paper_metadata(paper_url=effective_url, title=effective_title, abstract=effective_abstract)

    body_text = ""
    try:
        body_text = extract_pdf_text(effective_url, max_pages=settings.pdf_extract_max_pages)
        if body_text:
            logger.info("PDF extraction: %d chars from %s", len(body_text), effective_url)
        else:
            logger.warning("PDF extraction returned empty text for %s", effective_url)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", effective_url, exc)

    chunk_size = _clamp_chunk_size(settings.default_chunk_size)
    chunks = semantic_chunk_document(
        title=effective_title,
        abstract=effective_abstract,
        body_text=body_text,
        chunk_size=chunk_size,
        overlap=settings.chunk_overlap,
    )

    sanitized: list[dict[str, str | int]] = []
    for chunk in chunks:
        text = _normalize_whitespace(str(chunk.get("text", "")))
        if not text:
            continue
        chunk["text"] = trim_to_token_limit(text, settings.max_chunk_size)
        sanitized.append(chunk)
    return sanitized


def ingest_paper_for_retrieval(
    payload: AnalyzePaperRequest,
    openai_api_key: str | None = None,
) -> dict[str, int | bool]:
    settings = get_settings()
    resolved_inputs = _resolve_index_inputs(payload)
    effective_url = resolved_inputs["paper_url"]

    already_indexed = has_indexed_paper(payload.paper_url, user_id=payload.user_id)
    if not already_indexed and effective_url != payload.paper_url:
        already_indexed = has_indexed_paper(effective_url, user_id=payload.user_id)

    if already_indexed:
        logger.debug("Paper already indexed: %s", effective_url)
        return {"indexed_chunks": 0, "already_indexed": True}

    chunks = _prepare_chunks_for_index(payload, resolved_inputs=resolved_inputs)
    if not chunks:
        logger.warning("No chunks produced for %s", effective_url)
        return {"indexed_chunks": 0, "already_indexed": False}

    indexed_count = add_chunks_to_index(
        paper_url=effective_url,
        chunks=chunks,
        title=resolved_inputs["title"] or payload.title,
        user_id=payload.user_id,
        openai_api_key=openai_api_key,
    )
    logger.info("Indexed %d chunks for %s", indexed_count, effective_url)
    return {"indexed_chunks": indexed_count, "already_indexed": False}


def _extract_concepts(title: str, abstract: str, max_items: int = 6) -> list[str]:
    combined = f"{title} {abstract}".strip()
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", combined)
    ranked: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.lower()
        if key in _STOPWORDS or key in seen or len(key) < 4:
            continue
        seen.add(key)
        ranked.append(word)
        if len(ranked) >= max_items:
            break
    return ranked


def _extract_methods(abstract: str, max_items: int = 5) -> list[str]:
    lower = abstract.lower()
    return [h.title() for h in _METHOD_HINTS if h in lower][:max_items]


def _extract_datasets(abstract: str, max_items: int = 5) -> list[str]:
    lower = abstract.lower()
    return [h.upper() if len(h) <= 6 else h.title() for h in _DATASET_HINTS if h in lower][:max_items]


def _sanitize_str_list(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def run_fast_stage(
    payload: AnalyzePaperRequest,
    openai_api_key: str | None = None,
) -> FastStageResult:
    settings = get_settings()
    normalized_abstract = _normalize_whitespace(payload.abstract)
    heuristic_summary = trim_to_token_limit(normalized_abstract or payload.title, max_tokens=150)

    llm_response = call_structured_agent(
        system_prompt=PARSER_SYSTEM_PROMPT,
        payload={
            "paper_url": payload.paper_url,
            "title": payload.title,
            "abstract": normalized_abstract,
            "summary_hint": heuristic_summary,
            "retrieval_context": [],
        },
        max_output_tokens=400,
        openai_api_key=openai_api_key,
    )

    if llm_response:
        summary  = trim_to_token_limit(str(llm_response.get("summary", heuristic_summary)), max_tokens=150)
        methods  = _sanitize_str_list(llm_response.get("methods", []), max_items=5)
        datasets = _sanitize_str_list(llm_response.get("datasets", []), max_items=5)
        concept_seed = " ".join([payload.title, normalized_abstract, " ".join(methods), " ".join(datasets)])
        concepts = _extract_concepts(concept_seed, normalized_abstract)
    else:
        summary  = heuristic_summary
        concepts = _extract_concepts(payload.title, normalized_abstract)

    return FastStageResult(summary=summary, key_concepts=concepts)


def _content_similarity_query(payload: AnalyzePaperRequest, fast_stage: FastStageResult) -> str:
    settings = get_settings()
    abstract = _normalize_whitespace(payload.abstract)
    summary  = _normalize_whitespace(fast_stage.summary)
    title_lower = _normalize_whitespace(payload.title).lower()
    parts: list[str] = []
    if abstract:
        parts.append(abstract)
    if summary and summary.lower() != title_lower:
        parts.append(summary)
    if fast_stage.key_concepts:
        parts.append(" ".join(str(c).strip() for c in fast_stage.key_concepts[:6] if str(c).strip()))
    merged = " ".join(p for p in parts if p).strip()
    return trim_to_token_limit(merged, max_tokens=300) if merged else (summary or payload.title)


def run_full_stage(
    payload: AnalyzePaperRequest,
    fast_stage: FastStageResult,
    openai_api_key: str | None = None,
) -> FullStageResult:
    settings = get_settings()
    ingest_paper_for_retrieval(payload, openai_api_key=openai_api_key)

    resolved = resolve_paper_metadata(
        paper_url=payload.paper_url,
        title=payload.title,
        abstract=payload.abstract,
    )
    exclude_urls = [payload.paper_url]
    for candidate in (
        str(resolved.get("pdf_url", "")).strip(),
        str(resolved.get("canonical_url", "")).strip(),
    ):
        if candidate and candidate not in exclude_urls:
            exclude_urls.append(candidate)

    semantic_query = _content_similarity_query(payload, fast_stage)
    context_chunks = retrieve_context_chunks(
        query=semantic_query,
        top_k=settings.max_chunks,
        user_id=payload.user_id,
        exclude_urls=exclude_urls,
        openai_api_key=openai_api_key,
    )
    logger.info("Full stage: retrieved %d context chunks for '%s'", len(context_chunks), payload.title[:60])

    initial_state = {
        "paper_url": payload.paper_url,
        "title": payload.title,
        "abstract": payload.abstract,
        "user_id": payload.user_id,
        "summary": fast_stage.summary,
        "concepts": list(fast_stage.key_concepts),
        "context_chunks": context_chunks,
        "openai_api_key": openai_api_key,
    }

    final_state = run_agent_workflow(initial_state)

    # Fire citation crawl in background — does not block this response
    start_citation_crawl_async(
        root_url=payload.paper_url,
        root_title=payload.title,
        user_id=payload.user_id,
        openai_api_key=openai_api_key,
    )

    related_papers = [
        RelatedPaper(
            title=str(item.get("title", "Untitled")),
            url=str(item.get("url", "")),
            score=item.get("score"),
        )
        for item in final_state.get("related_papers", [])
    ]

    methods  = final_state.get("methods")  or _extract_methods(payload.abstract)
    datasets = final_state.get("datasets") or _extract_datasets(payload.abstract)

    return FullStageResult(
        methods=methods[:6],
        datasets=datasets[:5],
        related_papers=related_papers[:6],
        research_connections=final_state.get("graph_connections", [])[:6],
        user_interest_topics=final_state.get("user_interest_topics", [])[:5],
        research_gaps=final_state.get("research_gaps", [])[:3],
        ideas=final_state.get("ideas", [])[:3],
        planner=dict(final_state.get("planner", {})),
        autonomy_notes=list(final_state.get("autonomy_notes", []))[:15],
        confidence=float(final_state.get("confidence", 0.0) or 0.0),
        review_required=bool(final_state.get("review_required", False)),
        review_id=str(final_state.get("review_id", "")).strip() or None,
    )
