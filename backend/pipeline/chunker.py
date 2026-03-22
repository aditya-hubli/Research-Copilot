from __future__ import annotations

import re

SECTION_NAMES = (
    "Introduction",
    "Method",
    "Methods",
    "Results",
    "Conclusion",
)


def _split_words_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    safe_chunk_size = max(1, chunk_size)
    safe_overlap = max(0, min(overlap, safe_chunk_size - 1))
    step = safe_chunk_size - safe_overlap

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + safe_chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def split_semantic_sections(body_text: str) -> list[tuple[str, str]]:
    if not body_text.strip():
        return []

    pattern = r"(?im)^\s*(" + "|".join(re.escape(name) for name in SECTION_NAMES) + r")\s*$"
    matches = list(re.finditer(pattern, body_text))
    if not matches:
        return [("Body", body_text)]

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        section_name = match.group(1).title()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body_text)
        content = body_text[start:end].strip()
        if content:
            sections.append((section_name, content))
    return sections if sections else [("Body", body_text)]


def semantic_chunk_document(
    title: str,
    abstract: str,
    body_text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict[str, str | int]]:
    sections: list[tuple[str, str]] = []
    if title.strip():
        sections.append(("Title", title.strip()))
    if abstract.strip():
        sections.append(("Abstract", abstract.strip()))
    sections.extend(split_semantic_sections(body_text))

    chunks: list[dict[str, str | int]] = []
    for section_name, text in sections:
        for index, chunk_text in enumerate(_split_words_with_overlap(text, chunk_size, overlap)):
            chunks.append(
                {
                    "section": section_name,
                    "chunk_index": index,
                    "text": chunk_text,
                }
            )
    return chunks
