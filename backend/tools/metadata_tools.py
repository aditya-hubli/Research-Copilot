from __future__ import annotations

import hashlib
import json
import re
from threading import Lock
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

import requests
from cachetools import TTLCache
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from backend.core.config import get_settings

# NOTE: Do NOT cache settings at module level — always call get_settings() inside functions
_metadata_lock = Lock()
_metadata_cache: TTLCache | None = None


def _get_metadata_cache() -> TTLCache:
    global _metadata_cache
    if _metadata_cache is None:
        settings = get_settings()
        _metadata_cache = TTLCache(
            maxsize=max(1, settings.metadata_cache_max_entries),
            ttl=max(1, settings.metadata_cache_ttl_seconds),
        )
    return _metadata_cache


_USER_AGENT = "PersonalResearchCopilot/0.1 (+https://localhost)"
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)([^?#]+)", re.IGNORECASE)
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]{4,}")


def _build_http_session() -> requests.Session:
    retry = Retry(
        total=2, connect=2, read=2, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "application/json, application/atom+xml;q=0.9, text/xml;q=0.8, */*;q=0.6",
    })
    return session


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalized_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_space(title).lower()).strip()


def _title_token_set(title: str) -> set[str]:
    return {token.lower() for token in _TITLE_TOKEN_RE.findall(title or "")}


def _title_similarity(query_title: str, candidate_title: str) -> float:
    query_tokens = _title_token_set(query_title)
    candidate_tokens = _title_token_set(candidate_title)
    if not query_tokens or not candidate_tokens:
        return 0.0
    return round(min(1.0, len(query_tokens & candidate_tokens) / max(1, len(query_tokens))), 4)


def _extract_arxiv_id(paper_url: str) -> str:
    match = _ARXIV_ID_RE.search(paper_url or "")
    if not match:
        return ""
    raw = match.group(1).strip("/")
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]
    return raw.strip()


def _extract_doi(paper_url: str) -> str:
    normalized = unquote(str(paper_url or "")).strip()
    match = _DOI_RE.search(normalized)
    if not match:
        return ""
    return match.group(0).strip().rstrip(".,;)")


def _decode_openalex_abstract(inverted_index: object) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    positions: dict[int, str] = {}
    for token, idxs in inverted_index.items():
        if not isinstance(idxs, list):
            continue
        for idx in idxs:
            if isinstance(idx, int):
                positions[idx] = str(token)
    if not positions:
        return ""
    return _normalize_space(" ".join(word for _, word in sorted(positions.items())))


def _clean_crossref_abstract(value: str) -> str:
    if not value:
        return ""
    return _normalize_space(_HTML_TAG_RE.sub(" ", value))


def _provider_entry(provider: str, *, title: str = "", abstract: str = "",
                    canonical_url: str = "", pdf_url: str = "") -> dict[str, str]:
    return {
        "provider": provider,
        "title": _normalize_space(title),
        "abstract": _normalize_space(abstract),
        "canonical_url": canonical_url.strip(),
        "pdf_url": pdf_url.strip(),
    }


def _search_semantic_scholar_by_title(session: requests.Session, title: str,
                                       limit: int) -> list[dict[str, str | float]]:
    try:
        response = session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": max(limit * 2, 8),
                    "fields": "title,url,abstract,openAccessPdf"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    data = payload.get("data", []) if isinstance(payload, dict) else []
    results: list[dict[str, str | float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_title = _normalize_space(str(item.get("title", "")))
        if not item_title:
            continue
        open_access_pdf = item.get("openAccessPdf")
        pdf_url = str(open_access_pdf.get("url", "")).strip() if isinstance(open_access_pdf, dict) else ""
        url = pdf_url or str(item.get("url", "")).strip()
        results.append({"title": item_title, "url": url,
                         "score": _title_similarity(title, item_title), "provider": "semantic-scholar"})
    return results


def _search_openalex_by_title(session: requests.Session, title: str,
                               limit: int) -> list[dict[str, str | float]]:
    try:
        response = session.get(
            "https://api.openalex.org/works",
            params={"search": title, "per-page": max(limit * 2, 8)},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    data = payload.get("results", []) if isinstance(payload, dict) else []
    results: list[dict[str, str | float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_title = _normalize_space(str(item.get("display_name", "")))
        if not item_title:
            continue
        best_oa = item.get("best_oa_location")
        primary = item.get("primary_location")
        url = ""
        if isinstance(best_oa, dict):
            url = str(best_oa.get("landing_page_url", "") or best_oa.get("pdf_url", "")).strip()
        if not url and isinstance(primary, dict):
            url = str(primary.get("landing_page_url", "") or primary.get("pdf_url", "")).strip()
        if not url:
            url = str(item.get("id", "")).strip()
        results.append({"title": item_title, "url": url,
                         "score": _title_similarity(title, item_title), "provider": "openalex"})
    return results


def _resolve_arxiv(session: requests.Session, paper_url: str, arxiv_id: str) -> dict[str, str]:
    if not arxiv_id:
        return {}
    try:
        response = session.get("https://export.arxiv.org/api/query",
                                params={"id_list": arxiv_id}, timeout=10)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
    except Exception:
        return {}

    namespaces = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespaces)
    if entry is None:
        return {}

    title = entry.findtext("atom:title", default="", namespaces=namespaces)
    abstract = entry.findtext("atom:summary", default="", namespaces=namespaces)
    canonical_url = entry.findtext("atom:id", default="", namespaces=namespaces)
    pdf_url = ""
    for link in entry.findall("atom:link", namespaces):
        href = str(link.attrib.get("href", "")).strip()
        link_type = str(link.attrib.get("type", "")).lower()
        link_title = str(link.attrib.get("title", "")).lower()
        if "pdf" in link_type or link_title == "pdf" or href.lower().endswith(".pdf"):
            pdf_url = href
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return _provider_entry("arxiv", title=title, abstract=abstract,
                            canonical_url=canonical_url or paper_url, pdf_url=pdf_url)


def _resolve_semantic_scholar(session: requests.Session, paper_url: str,
                               arxiv_id: str) -> dict[str, str]:
    fields = "title,abstract,url,openAccessPdf"
    identifiers = [f"URL:{paper_url}"]
    if arxiv_id:
        identifiers.append(f"ARXIV:{arxiv_id}")

    for identifier in identifiers:
        try:
            response = session.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{quote(identifier, safe='')}",
                params={"fields": fields}, timeout=10,
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        open_access_pdf = payload.get("openAccessPdf")
        pdf_url = str(open_access_pdf.get("url", "")).strip() if isinstance(open_access_pdf, dict) else ""
        entry = _provider_entry("semantic-scholar",
                                 title=str(payload.get("title", "")),
                                 abstract=str(payload.get("abstract", "")),
                                 canonical_url=str(payload.get("url", "") or paper_url),
                                 pdf_url=pdf_url)
        if entry["title"] or entry["abstract"] or entry["pdf_url"]:
            return entry
    return {}


def _resolve_openalex(session: requests.Session, paper_url: str,
                      doi: str, arxiv_id: str) -> dict[str, str]:
    payload: dict[str, object] = {}
    try:
        if doi:
            response = session.get(
                f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='/')}",
                timeout=10)
            if response.ok:
                payload = response.json()
        if not payload and arxiv_id:
            response = session.get("https://api.openalex.org/works", params={
                "filter": f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}",
                "per-page": 1}, timeout=10)
            if response.ok:
                results = response.json().get("results", [])
                if results:
                    payload = results[0]
        if not payload:
            response = session.get("https://api.openalex.org/works",
                                    params={"search": paper_url, "per-page": 1}, timeout=10)
            if response.ok:
                results = response.json().get("results", [])
                if results:
                    payload = results[0]
    except Exception:
        return {}

    if not isinstance(payload, dict) or not payload:
        return {}

    best_oa = payload.get("best_oa_location")
    primary = payload.get("primary_location")
    pdf_url = ""
    if isinstance(best_oa, dict):
        pdf_url = str(best_oa.get("pdf_url", "") or "").strip()
    if not pdf_url and isinstance(primary, dict):
        pdf_url = str(primary.get("pdf_url", "") or "").strip()

    abstract = _decode_openalex_abstract(payload.get("abstract_inverted_index"))
    entry = _provider_entry("openalex",
                             title=str(payload.get("display_name", "")),
                             abstract=abstract,
                             canonical_url=str(payload.get("id", "") or paper_url),
                             pdf_url=pdf_url)
    return entry if (entry["title"] or entry["abstract"] or entry["pdf_url"]) else {}


def _resolve_crossref(session: requests.Session, doi: str) -> dict[str, str]:
    if not doi:
        return {}
    try:
        response = session.get(f"https://api.crossref.org/works/{quote(doi, safe='')}", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    message = data.get("message", {}) if isinstance(data, dict) else {}
    if not isinstance(message, dict):
        return {}

    title_items = message.get("title")
    title = str(title_items[0]) if isinstance(title_items, list) and title_items else ""
    abstract = _clean_crossref_abstract(str(message.get("abstract", "")))
    canonical_url = str(message.get("URL", "") or "")
    entry = _provider_entry("crossref", title=title, abstract=abstract, canonical_url=canonical_url)
    return entry if (entry["title"] or entry["abstract"]) else {}


def _metadata_cache_key(paper_url: str, title: str, abstract: str) -> str:
    payload = {"paper_url": paper_url.strip(), "title": title.strip(),
               "abstract_hash": hashlib.sha256(abstract.strip().encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata_resolution_cache_key(paper_url: str) -> str:
    return "resolve:" + hashlib.sha256(paper_url.strip().encode()).hexdigest()


def get_paper_metadata(paper_url: str, title: str = "", abstract: str = "") -> dict[str, str]:
    cache = _get_metadata_cache()
    cache_key = _metadata_cache_key(paper_url, title, abstract)
    with _metadata_lock:
        cached = cache.get(cache_key)
        if cached:
            return dict(cached)

    parsed = urlparse(paper_url)
    host = parsed.netloc.lower()
    source = "unknown"
    if "arxiv.org" in host:
        source = "arxiv"
    elif "semanticscholar.org" in host:
        source = "semantic-scholar"
    elif "openreview.net" in host:
        source = "openreview"

    metadata = {"paper_url": paper_url, "title": title, "abstract": abstract,
                 "source": source, "host": host}
    with _metadata_lock:
        cache[cache_key] = metadata
    return dict(metadata)


def resolve_paper_metadata(paper_url: str, title: str = "", abstract: str = "") -> dict[str, str]:
    cache = _get_metadata_cache()
    normalized_url = paper_url.strip()
    base_metadata = get_paper_metadata(normalized_url, title=title, abstract=abstract)
    cache_key = _metadata_resolution_cache_key(normalized_url)

    with _metadata_lock:
        cached = cache.get(cache_key)
        if cached:
            merged = dict(cached)
            if title.strip():
                merged["title"] = title.strip()
            if abstract.strip():
                merged["abstract"] = abstract.strip()
            return merged

    arxiv_id = _extract_arxiv_id(normalized_url)
    doi = _extract_doi(normalized_url)
    resolved_entries: list[dict[str, str]] = []
    session = _build_http_session()

    for resolver in (
        lambda: _resolve_arxiv(session, normalized_url, arxiv_id),
        lambda: _resolve_semantic_scholar(session, normalized_url, arxiv_id),
        lambda: _resolve_openalex(session, normalized_url, doi, arxiv_id),
        lambda: _resolve_crossref(session, doi),
    ):
        try:
            entry = resolver()
            if entry:
                resolved_entries.append(entry)
        except Exception:
            continue

    resolved: dict[str, str] = {
        "paper_url": normalized_url, "source": base_metadata.get("source", "unknown"),
        "host": base_metadata.get("host", ""), "provider": "", "title": "", "abstract": "",
        "canonical_url": "", "pdf_url": "", "arxiv_id": arxiv_id, "doi": doi, "providers": "",
    }
    provider_names: list[str] = []
    for entry in resolved_entries:
        provider = entry.get("provider", "").strip()
        if provider and provider not in provider_names:
            provider_names.append(provider)
        if not resolved["provider"] and provider:
            resolved["provider"] = provider
        for field in ("title", "abstract", "canonical_url", "pdf_url"):
            if not resolved[field] and entry.get(field):
                resolved[field] = entry[field]

    resolved["providers"] = ",".join(provider_names)
    if title.strip():
        resolved["title"] = title.strip()
    if abstract.strip():
        resolved["abstract"] = abstract.strip()

    with _metadata_lock:
        cache[cache_key] = dict(resolved)
    return dict(resolved)


def search_related_papers_by_title(title: str, *, paper_url: str = "",
                                    abstract: str = "", limit: int = 5) -> dict[str, object]:
    cache = _get_metadata_cache()
    normalized_title = _normalize_space(title)
    if not normalized_title:
        return {"provider": "", "related_papers": []}

    cache_payload = {
        "title": normalized_title, "paper_url": paper_url.strip(),
        "abstract_hash": hashlib.sha256(_normalize_space(abstract).encode()).hexdigest(),
        "limit": max(1, limit),
    }
    cache_key = "related:" + hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with _metadata_lock:
        cached = cache.get(cache_key)
        if cached:
            return dict(cached)

    normalized_title_key = _normalized_title_key(normalized_title)
    current_url = paper_url.strip()
    target_limit = min(max(1, int(limit)), 8)
    session = _build_http_session()
    aggregated: dict[str, dict[str, object]] = {}
    provider_names: list[str] = []

    for provider_name, search_fn in (
        ("semantic-scholar", _search_semantic_scholar_by_title),
        ("openalex", _search_openalex_by_title),
    ):
        try:
            results = search_fn(session, normalized_title, target_limit)
        except Exception:
            results = []

        if results and provider_name not in provider_names:
            provider_names.append(provider_name)

        for item in results:
            candidate_title = _normalize_space(str(item.get("title", "")))
            candidate_url = str(item.get("url", "")).strip()
            if not candidate_title:
                continue
            if normalized_title_key and _normalized_title_key(candidate_title) == normalized_title_key:
                continue
            if current_url and candidate_url and candidate_url == current_url:
                continue
            dedupe_key = candidate_url or _normalized_title_key(candidate_title)
            if not dedupe_key:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            existing = aggregated.get(dedupe_key)
            if existing is None or score > float(existing.get("score", 0.0) or 0.0):
                aggregated[dedupe_key] = {"title": candidate_title, "url": candidate_url,
                                           "score": round(score, 4)}

    ranked = sorted(aggregated.values(),
                    key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[:target_limit]
    payload = {"provider": ",".join(provider_names), "related_papers": ranked}
    with _metadata_lock:
        cache[cache_key] = dict(payload)
    return payload
