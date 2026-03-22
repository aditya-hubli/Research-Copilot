from __future__ import annotations

from backend.pipeline.chunker import semantic_chunk_document, split_semantic_sections


def test_split_semantic_sections_detects_headers() -> None:
    body = """Introduction
    Intro content line one.
    Method
    Method content line two.
    Results
    Results content line three.
    Conclusion
    Conclusion content.
    """

    sections = split_semantic_sections(body)
    names = [name for name, _ in sections]

    assert "Introduction" in names
    assert "Method" in names
    assert "Results" in names
    assert "Conclusion" in names


def test_semantic_chunk_document_contains_title_and_abstract() -> None:
    title = "Attention Is All You Need"
    abstract = "This paper introduces transformer-based sequence modeling methods."
    body = "Body content " * 200

    chunks = semantic_chunk_document(
        title=title,
        abstract=abstract,
        body_text=body,
        chunk_size=50,
        overlap=10,
    )

    assert len(chunks) > 0
    section_names = {str(chunk["section"]) for chunk in chunks}
    assert "Title" in section_names
    assert "Abstract" in section_names


def test_semantic_chunk_document_overlap_behavior() -> None:
    body = " ".join(f"token{i}" for i in range(120))
    chunks = semantic_chunk_document(
        title="",
        abstract="",
        body_text=body,
        chunk_size=40,
        overlap=10,
    )

    assert len(chunks) >= 3
    first_tokens = str(chunks[0]["text"]).split()
    second_tokens = str(chunks[1]["text"]).split()

    assert first_tokens[-10:] == second_tokens[:10]
