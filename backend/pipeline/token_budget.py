from __future__ import annotations

from typing import Iterable

import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback approximation when tokenizer is unavailable.
        return len(text.split())


def trim_to_token_limit(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> str:
    if max_tokens <= 0 or not text:
        return ""

    try:
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:
        words = text.split()
        return " ".join(words[:max_tokens])


def cap_chunks(chunks: Iterable[str], max_chunks: int) -> list[str]:
    if max_chunks <= 0:
        return []
    return list(chunks)[:max_chunks]


def enforce_prompt_budget(
    system_prompt: str,
    user_prompt: str,
    max_prompt_tokens: int,
) -> tuple[str, str]:
    system_tokens = count_tokens(system_prompt)
    available_for_user = max(max_prompt_tokens - system_tokens, 0)
    trimmed_user = trim_to_token_limit(user_prompt, available_for_user)
    return system_prompt, trimmed_user
