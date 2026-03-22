"""
Citation graph crawler.

Starting from a root paper, fetches references via Semantic Scholar,
ingests each discovered paper into FAISS, and writes CITES edges to Neo4j.

BFS up to CITATION_CRAWL_DEPTH hops, capped at CITATION_CRAWL_MAX_PAPERS
total new papers per crawl session. Runs in a daemon background thread so
it never blocks the main analysis pipeline.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from urllib.parse import quote

from backend.core.config import get_settings
from backend.tools.metadata_tools import _build_http_session, _extract_arxiv_id, _normalize_space
from backend.tools.vector_tools import has_indexed_paper

logger = logging.getLogger(__name__)


# ── Semantic Scholar helpers ──────────────────────────────────────────────────

def _s2_id(paper_url: str) -> str:
    """Build a Semantic Scholar paper identifier from a URL."""
    arxiv_id = _extract_arxiv_id(paper_url)
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"
    return f"URL:{quote(paper_url.strip(), safe='')}"


def fetch_citations(paper_url: str, limit: int = 30) -> list[dict]:
    """
    Fetch direct references for a paper via the Semantic Scholar graph API.
    Returns list of dicts: {title, url, abstract}.
    Empty list on any failure — never raises.
    """
    s2_id = _s2_id(paper_url)
    session = _build_http_session()
    try:
        resp = session.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}/references",
            params={
                "fields": "title,url,abstract,openAccessPdf,externalIds",
                "limit": min(limit, 100),
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("S2 citation fetch failed for %s: %s", paper_url, exc)
        return []

    results: list[dict] = []
    for item in data.get("data", []):
        ref = item.get("citedPaper") or {}
        if not isinstance(ref, dict):
            continue

        title = _normalize_space(str(ref.get("title", "")))
        if not title:
            continue

        # Prefer open-access PDF URL, then ArXiv abs, then whatever S2 gives
        url = ""
        oa = ref.get("openAccessPdf")
        if isinstance(oa, dict):
            url = str(oa.get("url", "") or "").strip()
        if not url:
            ext = ref.get("externalIds") or {}
            arxiv = str(ext.get("ArXiv", "") or "").strip()
            if arxiv:
                url = f"https://arxiv.org/abs/{arxiv}"
        if not url:
            url = str(ref.get("url", "") or "").strip()
        if not url:
            continue

        abstract = _normalize_space(str(ref.get("abstract", "") or ""))
        results.append({"title": title, "url": url, "abstract": abstract})

    logger.debug("fetch_citations(%s): %d refs returned", paper_url[:60], len(results))
    return results


# ── BFS crawler ───────────────────────────────────────────────────────────────

def crawl_and_index_citations(
    root_url: str,
    root_title: str,
    user_id: str,
    openai_api_key: str | None = None,
) -> None:
    """
    BFS citation crawler.

    Visits papers breadth-first up to citation_crawl_depth hops from root_url.
    Each new paper is:
      1. Ingested into FAISS (download → chunk → embed → store)
      2. Written to Neo4j as a Paper node with a CITES edge from its parent

    Stops when citation_crawl_max_papers new papers have been indexed or the
    BFS queue is exhausted, whichever comes first.
    """
    # Late imports to avoid circular deps at module load time
    from backend.api.models import AnalyzePaperRequest
    from backend.api.pipeline_service import ingest_paper_for_retrieval
    from backend.db.neo4j_client import get_neo4j_client

    settings = get_settings()
    if not settings.citation_crawl_enabled:
        return

    max_depth   = settings.citation_crawl_depth
    max_papers  = settings.citation_crawl_max_papers
    neo4j       = get_neo4j_client()

    # Queue entries: (url, title, abstract, depth, parent_url)
    queue: deque[tuple[str, str, str, int, str]] = deque()
    queue.append((root_url, root_title, "", 0, ""))
    visited: set[str] = {root_url}
    indexed_count = 0

    logger.info(
        "Citation crawl started | root=%s | max_depth=%d | max_papers=%d | user=%s",
        root_url[:80], max_depth, max_papers, user_id,
    )

    while queue and indexed_count < max_papers:
        url, title, abstract, depth, parent_url = queue.popleft()

        # ── Ingest if this is not the root and not already in FAISS ──────────
        if url != root_url:
            if not has_indexed_paper(url, user_id=user_id):
                try:
                    result = ingest_paper_for_retrieval(
                        AnalyzePaperRequest(
                            paper_url=url,
                            title=title or "Untitled",
                            abstract=abstract,
                            user_id=user_id,
                        ),
                        openai_api_key=openai_api_key,
                    )
                    if not result.get("already_indexed"):
                        indexed_count += 1
                        logger.debug(
                            "Crawl indexed [%d/%d] depth=%d: %s",
                            indexed_count, max_papers, depth, title[:70],
                        )
                except Exception as exc:
                    logger.debug("Crawl ingest failed for %s: %s", url[:60], exc)

            # ── Write CITES edge: parent → this paper ─────────────────────
            if parent_url:
                try:
                    neo4j.upsert_citation_edge(
                        citing_url=parent_url,
                        cited_url=url,
                        cited_title=title or "Untitled",
                        user_id=user_id,
                    )
                except Exception as exc:
                    logger.debug("Neo4j CITES edge failed: %s", exc)

        # ── Fetch next hop if within depth limit ──────────────────────────────
        if depth < max_depth and indexed_count < max_papers:
            try:
                citations = fetch_citations(url, limit=25)
            except Exception:
                citations = []

            for cited in citations:
                cited_url = str(cited.get("url", "")).strip()
                if cited_url and cited_url not in visited:
                    visited.add(cited_url)
                    queue.append((
                        cited_url,
                        cited.get("title", ""),
                        cited.get("abstract", ""),
                        depth + 1,
                        url,
                    ))

    logger.info(
        "Citation crawl complete | indexed=%d | visited=%d | root=%s",
        indexed_count, len(visited), root_url[:80],
    )


# ── Public entry point ────────────────────────────────────────────────────────

def start_citation_crawl_async(
    root_url: str,
    root_title: str,
    user_id: str,
    openai_api_key: str | None = None,
) -> None:
    """
    Launch the citation crawler in a daemon background thread.
    Returns immediately — the crawl runs independently behind the scenes.
    """
    settings = get_settings()
    if not settings.citation_crawl_enabled:
        return

    thread = threading.Thread(
        target=crawl_and_index_citations,
        args=(root_url, root_title, user_id, openai_api_key),
        daemon=True,
        name=f"citation-crawl:{root_url[-40:]}",
    )
    thread.start()
    logger.info("Citation crawl thread launched for: %s", root_url[:80])
