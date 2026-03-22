from __future__ import annotations

import importlib

import pytest


def _reload_runtime_modules():
    import backend.agents.llm_runtime as llm_runtime
    import backend.core.config as config

    config.get_settings.cache_clear()

    config = importlib.reload(config)
    llm_runtime = importlib.reload(llm_runtime)
    return llm_runtime, config


def test_call_structured_agent_strict_mode_raises_without_llm(monkeypatch) -> None:
    monkeypatch.setenv("STRICT_LLM_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    llm_runtime, _ = _reload_runtime_modules()

    with pytest.raises(llm_runtime.LLMUnavailableError):
        llm_runtime.call_structured_agent(
            system_prompt="You are strict.",
            payload={"k": "v"},
            max_output_tokens=32,
            openai_api_key=None,
        )


def test_call_structured_agent_non_strict_returns_none_without_llm(monkeypatch) -> None:
    monkeypatch.setenv("STRICT_LLM_MODE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    llm_runtime, _ = _reload_runtime_modules()

    result = llm_runtime.call_structured_agent(
        system_prompt="You are non-strict.",
        payload={"k": "v"},
        max_output_tokens=32,
        openai_api_key=None,
    )
    assert result is None
