from __future__ import annotations

from backend.pipeline.token_budget import cap_chunks, enforce_prompt_budget, trim_to_token_limit


def test_trim_to_token_limit_reduces_large_input() -> None:
    text = "word " * 500
    trimmed = trim_to_token_limit(text, max_tokens=50)
    assert len(trimmed.split()) <= 55


def test_cap_chunks_limits_output() -> None:
    chunks = ["a", "b", "c", "d"]
    capped = cap_chunks(chunks, max_chunks=2)
    assert capped == ["a", "b"]


def test_enforce_prompt_budget_keeps_system_prompt_intact() -> None:
    system_prompt = "system prompt " * 10
    user_prompt = "user text " * 400

    result_system, result_user = enforce_prompt_budget(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_prompt_tokens=120,
    )

    assert result_system == system_prompt
    assert len(result_user) > 0
    assert result_user != user_prompt
