from __future__ import annotations

from backend.tools.pdf_tools import _extract_pdf_candidates_from_html, _normalize_pdf_url


def test_normalize_pdf_url_for_arxiv_and_openreview() -> None:
    arxiv_abs = "https://arxiv.org/abs/1706.03762"
    openreview_forum = "https://openreview.net/forum?id=abc123"

    assert _normalize_pdf_url(arxiv_abs) == "https://arxiv.org/pdf/1706.03762.pdf"
    assert _normalize_pdf_url(openreview_forum) == "https://openreview.net/pdf?id=abc123"


def test_extract_pdf_candidates_from_html_discovers_meta_and_anchor_links() -> None:
    source_url = "https://papers.example.org/paper/xyz"
    html = """
    <html>
      <head>
        <meta name="citation_pdf_url" content="/downloads/paper-xyz.pdf" />
      </head>
      <body>
        <a href="/assets/supplementary.pdf">Supplementary PDF</a>
        <a href="/paper/xyz/fulltext">Read full text</a>
      </body>
    </html>
    """

    candidates = _extract_pdf_candidates_from_html(source_url, html)

    assert "https://papers.example.org/downloads/paper-xyz.pdf" in candidates
    assert "https://papers.example.org/assets/supplementary.pdf" in candidates
