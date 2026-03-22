from __future__ import annotations

from io import BytesIO
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

_USER_AGENT = "PersonalResearchCopilot/0.1 (+https://localhost)"
_PDF_HINT_RE = re.compile(r"(?:\.pdf(?:$|[?#])|/pdf(?:$|[/?#]))", re.IGNORECASE)
_PDF_TEXT_HINT_RE = re.compile(r"(pdf|download|full text|manuscript)", re.IGNORECASE)


def _normalize_pdf_url(paper_url: str) -> str:
    parsed = urlparse(paper_url)
    host = parsed.netloc.lower()
    path = parsed.path or ""

    if "arxiv.org" in host:
        if path.startswith("/abs/"):
            paper_id = path.split("/abs/", 1)[1]
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
        if path.startswith("/pdf/") and not path.endswith(".pdf"):
            return f"https://arxiv.org{path}.pdf"

    if "openreview.net" in host and path == "/forum":
        query = parse_qs(parsed.query)
        paper_id = (query.get("id") or [""])[0]
        if paper_id:
            pdf_query = urlencode({"id": paper_id})
            return urlunparse((parsed.scheme or "https", parsed.netloc, "/pdf", "", pdf_query, ""))

    return paper_url


def _build_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _looks_like_pdf(response: requests.Response, requested_url: str) -> bool:
    content_type = (response.headers.get("Content-Type", "") or "").lower()
    if "pdf" in content_type:
        return True
    if requested_url.lower().endswith(".pdf"):
        return True
    return response.content.startswith(b"%PDF")


def _is_probable_pdf_url(url: str) -> bool:
    normalized = (url or "").strip()
    if not normalized:
        return False
    return bool(_PDF_HINT_RE.search(normalized))


def _extract_pdf_candidates_from_html(source_url: str, html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    candidates: list[str] = []
    seen: set[str] = set()

    def add(url: str | None) -> None:
        if not url:
            return
        absolute = urljoin(source_url, url.strip())
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return
        key = absolute.strip()
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    for meta in soup.find_all("meta"):
        content = (meta.get("content") or "").strip()
        name_key = (
            str(meta.get("name") or "")
            + " "
            + str(meta.get("property") or "")
            + " "
            + str(meta.get("itemprop") or "")
        ).lower()
        if any(hint in name_key for hint in ("pdf", "citation_pdf_url", "fulltext")):
            add(content)
            continue
        if content and _is_probable_pdf_url(content):
            add(content)

    for link in soup.find_all("link"):
        href = (link.get("href") or "").strip()
        rel = " ".join(link.get("rel") or []).lower()
        type_hint = str(link.get("type") or "").lower()
        if "pdf" in rel or "pdf" in type_hint or _is_probable_pdf_url(href):
            add(href)

    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        text = anchor.get_text(" ", strip=True)
        if _is_probable_pdf_url(href):
            add(href)
            continue
        if text and _PDF_TEXT_HINT_RE.search(text) and "/" in href:
            add(href)

    return candidates


def _discover_candidate_urls(
    paper_url: str,
    session: requests.Session,
    timeout_seconds: int,
) -> list[str]:
    normalized_url = _normalize_pdf_url(paper_url)
    queue: list[str] = [normalized_url]
    if normalized_url != paper_url:
        queue.append(paper_url)

    seen: set[str] = set()
    discovered: list[str] = []

    # Shallow crawl: seed URLs + one expansion layer from HTML.
    expansion_budget = 12
    while queue and len(discovered) < 20 and expansion_budget > 0:
        current = queue.pop(0)
        if not current or current in seen:
            continue
        seen.add(current)
        discovered.append(current)

        if _is_probable_pdf_url(current):
            continue

        try:
            response = session.get(current, timeout=timeout_seconds)
            response.raise_for_status()
        except Exception:
            continue

        content_type = (response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type:
            continue

        html_candidates = _extract_pdf_candidates_from_html(current, response.text)
        for candidate in html_candidates:
            if candidate in seen or candidate in queue:
                continue
            queue.append(candidate)
            expansion_budget -= 1
            if expansion_budget <= 0:
                break

    return discovered


def extract_pdf_text(paper_url: str, max_pages: int = 8, timeout_seconds: int = 25) -> str:
    session = _build_http_session()
    candidate_urls = _discover_candidate_urls(
        paper_url=paper_url,
        session=session,
        timeout_seconds=timeout_seconds,
    )

    for url in candidate_urls:
        try:
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
        except Exception:
            continue

        if not _looks_like_pdf(response, url):
            continue

        reader = PdfReader(BytesIO(response.content))
        pages = reader.pages[:max_pages]
        extracted = [page.extract_text() or "" for page in pages]
        text = "\n".join(item.strip() for item in extracted if item.strip())
        if text:
            return text

    return ""
