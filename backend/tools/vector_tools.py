from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from backend.core.config import get_settings
from backend.db.faiss_store import get_faiss_store
from backend.tools.embedding_tools import create_embedding, has_embedding_provider

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")
_TITLE_SECTION = "title"
_SECTION_WEIGHTS = {
    "title": 0.18,
    "abstract": 0.92,
    "introduction": 1.04,
    "method": 1.08,
    "methods": 1.08,
    "results": 1.05,
    "conclusion": 0.98,
    "body": 1.0,
}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _normalize_section(section: str | None) -> str:
    return str(section or "").strip().lower() or "body"


def _expand_query_variants(query: str, max_variants: int) -> list[str]:
    normalized = " ".join((query or "").split()).strip()
    if not normalized:
        return []

    tokens = [token for token in _TOKEN_RE.findall(normalized) if len(token) >= 4]
    variants = [normalized]
    if tokens:
        variants.append(" ".join(tokens[:7]))

    if len(tokens) >= 6:
        variants.append(" ".join(tokens[-6:]))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)

    return deduped[: max(1, max_variants)]


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def _section_weight(section: str | None) -> float:
    normalized_section = _normalize_section(section)
    return float(_SECTION_WEIGHTS.get(normalized_section, 1.0))


def _combine_scores(vector_score: float, lexical_score: float, section: str | None) -> float:
    semantic_blend = (0.9 * float(vector_score)) + (0.1 * float(lexical_score))
    weighted = semantic_blend * _section_weight(section)
    return max(0.0, min(1.0, weighted))


def _rerank_content_first(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    non_title = [item for item in items if _normalize_section(item.get("section")) != _TITLE_SECTION]
    title_only = [item for item in items if _normalize_section(item.get("section")) == _TITLE_SECTION]
    return non_title + title_only


def _paper_result_score(scores: list[float], non_title_scores: list[float]) -> float:
    if not scores:
        return 0.0

    top_scores = sorted(scores, reverse=True)[:3]
    base = (sum(top_scores) / len(top_scores)) * 0.7 + (max(top_scores) * 0.3)
    support_bonus = min(0.05, max(0, len(top_scores) - 1) * 0.02)
    if non_title_scores:
        base = max(base, max(non_title_scores))
        base += min(0.04, len(non_title_scores) * 0.015)
    else:
        base *= 0.7
    return round(min(1.0, max(0.0, base + support_bonus)), 4)


def _lexical_fallback_records(
    *,
    store,
    query: str,
    user_id: str | None,
    exclude_urls: set[str],
    include_urls: set[str] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    records = store.records(user_id=user_id)
    ranked: list[dict[str, Any]] = []

    for item in records:
        metadata = item.get("metadata", {})
        paper_url = str(metadata.get("paper_url", "")).strip()
        if paper_url and paper_url in exclude_urls:
            continue
        if include_urls and paper_url not in include_urls:
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        section = str(metadata.get("section", "Body"))
        lexical = _lexical_overlap_score(query, text)
        weighted_score = _combine_scores(0.0, lexical, section)
        if weighted_score <= 0.0:
            continue
        if lexical <= 0.0:
            continue

        ranked.append(
            {
                "paper_url": paper_url,
                "title": str(metadata.get("title", "Untitled")),
                "section": section,
                "text": text,
                "score": round(weighted_score, 4),
                "vector_score": 0.0,
                "lexical_score": round(lexical, 4),
            }
        )

    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    ranked = _rerank_content_first(ranked)
    return ranked[: max(1, top_k)]


def add_chunks_to_index(
    paper_url: str,
    chunks: list[dict[str, Any]],
    title: str,
    user_id: str,
    openai_api_key: str | None = None,
) -> int:
    store = get_faiss_store()
    embeddings: list[list[float]] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        embedding = create_embedding(
            text,
            dimension=store.dimension,
            openai_api_key=openai_api_key,
        )
        embeddings.append(embedding)
        texts.append(text)
        metadatas.append(
            {
                "paper_url": paper_url,
                "title": title,
                "user_id": user_id,
                "section": chunk.get("section", "Body"),
            }
        )

    if embeddings:
        store.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

    return len(embeddings)


def has_indexed_paper(paper_url: str, user_id: str) -> bool:
    store = get_faiss_store()
    return store.contains_paper_url(paper_url, user_id=user_id)


def vector_search(
    query: str,
    top_k: int = 5,
    exclude_urls: Iterable[str] | None = None,
    include_urls: Iterable[str] | None = None,
    user_id: str | None = None,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    store = get_faiss_store()
    retrieval_cap = max(1, settings.max_chunks)
    requested_top_k = max(1, min(top_k, 8))
    query_variants = _expand_query_variants(query, settings.retrieval_query_expansions)
    if not query_variants:
        return []

    blocked_urls = set(exclude_urls or [])
    allowed_urls = {str(item).strip() for item in (include_urls or []) if str(item).strip()}
    if not has_embedding_provider(openai_api_key):
        fallback_chunks = _lexical_fallback_records(
            store=store,
            query=query,
            user_id=user_id,
            exclude_urls=blocked_urls,
            include_urls=allowed_urls,
            top_k=requested_top_k * 4,
        )
        fallback_papers: dict[str, dict[str, Any]] = {}
        for chunk in fallback_chunks:
            url = str(chunk.get("paper_url", "")).strip()
            if not url:
                continue
            score = float(chunk.get("score", 0.0) or 0.0)
            section = str(chunk.get("section", "Body"))
            existing = fallback_papers.get(url)
            if existing is None:
                fallback_papers[url] = {
                    "title": str(chunk.get("title", "Untitled")),
                    "url": url,
                    "_chunk_scores": [score],
                    "_non_title_scores": [score] if _normalize_section(section) != _TITLE_SECTION else [],
                }
                continue
            existing["_chunk_scores"] = list(existing.get("_chunk_scores", [])) + [score]
            if _normalize_section(section) != _TITLE_SECTION:
                existing["_non_title_scores"] = list(existing.get("_non_title_scores", [])) + [score]

        ranked_fallback = sorted(
            [
                {
                    "title": str(item.get("title", "Untitled")),
                    "url": str(item.get("url", "")),
                    "score": _paper_result_score(
                        list(item.get("_chunk_scores", [])),
                        list(item.get("_non_title_scores", [])),
                    ),
                }
                for item in fallback_papers.values()
            ],
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return ranked_fallback[:requested_top_k]

    # Pull extra candidates so de-dup and exclusions can still return useful results.
    candidate_k = max(
        requested_top_k * max(2, settings.retrieval_candidate_multiplier),
        retrieval_cap,
    )

    paper_candidates: dict[str, dict[str, Any]] = {}

    for variant in query_variants:
        query_embedding = create_embedding(
            variant,
            dimension=store.dimension,
            openai_api_key=openai_api_key,
        )
        hits = store.search(query_embedding=query_embedding, top_k=candidate_k)

        for hit in hits:
            metadata = hit.get("metadata", {})
            paper_url = str(metadata.get("paper_url", "")).strip()
            metadata_user = str(metadata.get("user_id", "")).strip()
            if user_id and metadata_user != user_id:
                continue
            if not paper_url or paper_url in blocked_urls:
                continue
            if allowed_urls and paper_url not in allowed_urls:
                continue

            text = str(hit.get("text", ""))
            title = str(metadata.get("title", "Untitled"))
            section = str(metadata.get("section", ""))
            vector_score = float(hit.get("score", 0.0))
            lexical_score = _lexical_overlap_score(variant, text)
            combined_score = _combine_scores(vector_score, lexical_score, section)

            existing = paper_candidates.get(paper_url)
            if existing is None:
                paper_candidates[paper_url] = {
                    "title": title,
                    "url": paper_url,
                    "score": round(combined_score, 4),
                    "_chunk_scores": [combined_score],
                    "_non_title_scores": [combined_score] if _normalize_section(section) != _TITLE_SECTION else [],
                }
                continue
            existing_scores = list(existing.get("_chunk_scores", []))
            existing_scores.append(combined_score)
            existing["_chunk_scores"] = existing_scores
            if _normalize_section(section) != _TITLE_SECTION:
                non_title_scores = list(existing.get("_non_title_scores", []))
                non_title_scores.append(combined_score)
                existing["_non_title_scores"] = non_title_scores
            existing["score"] = _paper_result_score(
                list(existing.get("_chunk_scores", [])),
                list(existing.get("_non_title_scores", [])),
            )

    ranked = sorted(paper_candidates.values(), key=lambda item: float(item["score"]), reverse=True)
    ranked = [
        {
            "title": str(item.get("title", "Untitled")),
            "url": str(item.get("url", "")),
            "score": round(float(item.get("score", 0.0) or 0.0), 4),
        }
        for item in ranked
    ]
    if ranked:
        return ranked[:requested_top_k]

    fallback_chunks = _lexical_fallback_records(
        store=store,
        query=query,
        user_id=user_id,
        exclude_urls=blocked_urls,
        include_urls=allowed_urls,
        top_k=requested_top_k * 3,
    )

    fallback_papers: dict[str, dict[str, Any]] = {}
    for chunk in fallback_chunks:
        url = str(chunk.get("paper_url", "")).strip()
        if not url:
            continue
        score = float(chunk.get("score", 0.0))
        existing = fallback_papers.get(url)
        if existing is None or score > float(existing.get("score", 0.0)):
            fallback_papers[url] = {
                "title": str(chunk.get("title", "Untitled")),
                "url": url,
                "score": round(score, 4),
            }

    fallback_ranked = sorted(
        fallback_papers.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    return fallback_ranked[:requested_top_k]


def retrieve_context_chunks(
    query: str,
    top_k: int,
    user_id: str,
    exclude_urls: Iterable[str] | None = None,
    include_urls: Iterable[str] | None = None,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    store = get_faiss_store()
    requested_k = max(1, top_k)
    query_variants = _expand_query_variants(query, settings.retrieval_query_expansions)
    if not query_variants:
        return []

    blocked_urls = set(exclude_urls or [])
    allowed_urls = {str(item).strip() for item in (include_urls or []) if str(item).strip()}
    if not has_embedding_provider(openai_api_key):
        fallback = _lexical_fallback_records(
            store=store,
            query=query,
            user_id=user_id,
            exclude_urls=blocked_urls,
            include_urls=allowed_urls,
            top_k=requested_k,
        )
        return [
            {
                "text": item["text"],
                "paper_url": item["paper_url"],
                "title": item["title"],
                "section": item["section"],
                "score": item["score"],
                "vector_score": item["vector_score"],
                "lexical_score": item["lexical_score"],
            }
            for item in fallback
        ]

    candidate_k = max(
        requested_k * max(2, settings.retrieval_candidate_multiplier),
        requested_k,
    )

    chunk_candidates: dict[str, dict[str, Any]] = {}

    for variant in query_variants:
        query_embedding = create_embedding(
            variant,
            dimension=store.dimension,
            openai_api_key=openai_api_key,
        )
        hits = store.search(query_embedding=query_embedding, top_k=candidate_k)

        for hit in hits:
            metadata = hit.get("metadata", {})
            if user_id and str(metadata.get("user_id", "")).strip() != user_id:
                continue

            paper_url = str(metadata.get("paper_url", "")).strip()
            if paper_url and paper_url in blocked_urls:
                continue
            if allowed_urls and paper_url not in allowed_urls:
                continue

            chunk_text = str(hit.get("text", "")).strip()
            if not chunk_text:
                continue

            section = str(metadata.get("section", "Body"))
            vector_score = float(hit.get("score", 0.0))
            lexical_score = _lexical_overlap_score(variant, chunk_text)
            combined_score = _combine_scores(vector_score, lexical_score, section)

            dedupe_key = f"{paper_url}|{section}|{chunk_text[:120]}"
            existing = chunk_candidates.get(dedupe_key)
            if existing is None or combined_score > float(existing.get("score", 0.0)):
                chunk_candidates[dedupe_key] = {
                    "text": chunk_text,
                    "paper_url": paper_url,
                    "title": metadata.get("title", "Untitled"),
                    "section": section,
                    "score": round(combined_score, 4),
                    "vector_score": round(vector_score, 4),
                    "lexical_score": round(lexical_score, 4),
                }

    ranked = sorted(
        chunk_candidates.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    ranked = _rerank_content_first(ranked)
    if ranked:
        return ranked[:requested_k]

    fallback = _lexical_fallback_records(
        store=store,
        query=query,
        user_id=user_id,
        exclude_urls=blocked_urls,
        include_urls=allowed_urls,
        top_k=requested_k,
    )
    return [
        {
            "text": item["text"],
            "paper_url": item["paper_url"],
            "title": item["title"],
            "section": item["section"],
            "score": item["score"],
            "vector_score": item["vector_score"],
            "lexical_score": item["lexical_score"],
        }
        for item in fallback
    ]
