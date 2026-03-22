from __future__ import annotations

import json
import logging
from typing import Any

from backend.core.config import get_settings
from backend.pipeline.token_budget import enforce_prompt_budget

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_OPENAI      = "gpt-4o-mini"
_DEFAULT_MODEL_ANTHROPIC   = "claude-3-haiku-20240307"
_DEFAULT_MODEL_GEMINI      = "gemini-2.0-flash-lite"
_DEFAULT_MODEL_HUGGINGFACE = "Qwen/Qwen2.5-7B-Instruct"   # free, fast, great at JSON


class LLMUnavailableError(RuntimeError):
    pass


class LLMInvocationError(RuntimeError):
    pass


# ── Provider detection ────────────────────────────────────────────────────────

def detect_provider(api_key: str | None = None, provider_hint: str | None = None) -> str:
    """
    Priority: explicit hint > key prefix > LLM_PROVIDER env var.
    """
    hint = str(provider_hint or "").strip().lower()
    if hint in {"anthropic", "claude"}:
        return "anthropic"
    if hint in {"google", "gemini"}:
        return "gemini"
    if hint in {"huggingface", "hf", "hugging_face"}:
        return "huggingface"
    if hint == "openai":
        return "openai"

    key = str(api_key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AIza") or key.startswith("ya29."):
        return "gemini"
    if key.startswith("hf_"):
        return "huggingface"
    if key.startswith("sk-"):
        return "openai"

    # Fall back to LLM_PROVIDER from .env
    settings = get_settings()
    configured = str(settings.llm_provider or "huggingface").strip().lower()
    if configured in {"anthropic", "claude"}:
        return "anthropic"
    if configured in {"gemini", "google"}:
        return "gemini"
    if configured in {"huggingface", "hf"}:
        return "huggingface"
    if configured == "openai":
        return "openai"
    return "huggingface"


def _effective_key(api_key: str | None, provider: str | None = None) -> str:
    """Return per-request key if set, else fall back to env config for the provider."""
    per_request = str(api_key or "").strip()
    if per_request:
        return per_request
    settings = get_settings()
    if provider == "anthropic":
        return str(getattr(settings, "anthropic_api_key", None) or "").strip()
    if provider == "gemini":
        return str(getattr(settings, "gemini_api_key", None) or "").strip()
    if provider == "huggingface":
        return str(getattr(settings, "huggingface_api_key", None) or "").strip()
    return str(settings.openai_api_key or "").strip()


def is_llm_available(
    api_key: str | None = None,
    provider_hint: str | None = None,
    openai_api_key: str | None = None,
    llm_api_key: str | None = None,
    llm_provider: str | None = None,
    **_extra: Any,
) -> bool:
    resolved_key  = str(llm_api_key or openai_api_key or api_key or "").strip()
    resolved_hint = str(llm_provider or provider_hint or "").strip() or None
    provider      = detect_provider(resolved_key, resolved_hint)
    return bool(_effective_key(resolved_key, provider))


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> dict[str, Any] | None:
    """Try to close a JSON object that was cut off mid-generation."""
    s = raw.strip()
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    stack: list[str] = []
    in_string = False
    escape_next = False
    last_non_ws = ""

    for ch in s:
        if escape_next:
            escape_next = False
            last_non_ws = ch
            continue
        if ch == "\\" and in_string:
            escape_next = True
            last_non_ws = ch
            continue
        if ch == '"':
            in_string = not in_string
            last_non_ws = ch
            continue
        if in_string:
            last_non_ws = ch
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
        if ch.strip():
            last_non_ws = ch

    if not stack and not in_string:
        return None  # already well-formed — let the normal path handle it

    candidate = s
    if in_string:
        candidate += '"'
    # strip trailing comma before we close
    candidate = candidate.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]
    candidate += "".join(reversed(stack))

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            logger.debug("_repair_truncated_json: recovered truncated JSON")
            return parsed
    except Exception:
        pass
    return None


def _fix_json_string(s: str) -> str:
    """Escape literal newlines/tabs inside JSON string values so json.loads can parse them."""
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == "\\" and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                result.append("\\r")
                continue
            if ch == "\t":
                result.append("\\t")
                continue
        result.append(ch)
    return "".join(result)


def _safe_parse_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`").lstrip("json").strip().rstrip("`").strip()

    # Build candidates: original, cleaned, and newline-fixed versions
    candidates = [raw, cleaned, _fix_json_string(raw), _fix_json_string(cleaned)]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Try extracting the outermost {...} block (handles preamble text)
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        block = raw[start:end + 1]
        for candidate in (block, _fix_json_string(block)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # Last resort: attempt to close a truncated JSON object
    return _repair_truncated_json(raw)


# ── Provider implementations ──────────────────────────────────────────────────

def _call_openai(system: str, user: str, max_tokens: int, api_key: str) -> dict[str, Any] | None:
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed")
        return None
    settings = get_settings()
    model = str(settings.llm_model or _DEFAULT_MODEL_OPENAI).strip() or _DEFAULT_MODEL_OPENAI
    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model, temperature=0.1, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        result = _safe_parse_json(completion.choices[0].message.content or "")
        logger.info("OpenAI call succeeded model=%s parsed=%s", model, result is not None)
        return result
    except Exception as exc:
        logger.warning("OpenAI call failed (model=%s): %s", model, exc)
        return None


def _call_anthropic(system: str, user: str, max_tokens: int, api_key: str) -> dict[str, Any] | None:
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None
    json_instruction = "\n\nIMPORTANT: Respond with a valid JSON object only. No prose, no markdown fences."
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=_DEFAULT_MODEL_ANTHROPIC, max_tokens=max_tokens,
            system=system + json_instruction,
            messages=[{"role": "user", "content": user}],
        )
        content = "".join(block.text for block in message.content if hasattr(block, "text"))
        result = _safe_parse_json(content)
        logger.info("Anthropic call succeeded parsed=%s", result is not None)
        return result
    except Exception as exc:
        logger.warning("Anthropic call failed: %s", exc)
        return None


def _call_gemini(system: str, user: str, max_tokens: int, api_key: str) -> dict[str, Any] | None:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("google-genai package not installed")
        return None
    json_instruction = "\n\nIMPORTANT: Respond with a valid JSON object only. No prose, no markdown fences."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_DEFAULT_MODEL_GEMINI,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system + json_instruction,
                temperature=0.1,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        result = _safe_parse_json(response.text or "")
        logger.info("Gemini call succeeded parsed=%s", result is not None)
        return result
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None


def _call_huggingface(system: str, user: str, max_tokens: int, api_key: str) -> dict[str, Any] | None:
    """
    Uses HuggingFace Inference API via huggingface_hub.
    Free tier: ~1000 requests/day. Token from huggingface.co/settings/tokens
    Default model: Qwen/Qwen2.5-7B-Instruct (excellent JSON following)
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        logger.warning("huggingface_hub not installed. Run: pip install huggingface_hub")
        return None

    settings = get_settings()
    # Allow overriding model via LLM_MODEL env var
    configured_model = str(settings.llm_model or "").strip()
    model = configured_model if configured_model and configured_model != "gpt-4o-mini" else _DEFAULT_MODEL_HUGGINGFACE

    json_instruction = "\n\nIMPORTANT: Respond with a valid JSON object ONLY. No explanation, no markdown, no extra text. Start your response with { and end with }."

    try:
        client = InferenceClient(
            model=model,
            token=api_key,
        )
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system + json_instruction},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        result = _safe_parse_json(raw)
        if result is None:
            logger.warning("HuggingFace parse failed. raw=%r", raw[:500])
        logger.info("HuggingFace call succeeded model=%s parsed=%s", model, result is not None)
        return result
    except Exception as exc:
        logger.warning("HuggingFace call failed (model=%s): %s", model, exc)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def call_structured_agent(
    system_prompt: str,
    payload: dict[str, Any],
    max_output_tokens: int = 400,
    openai_api_key: str | None = None,
    llm_api_key: str | None = None,
    llm_provider: str | None = None,
    **_extra: Any,
) -> dict[str, Any] | None:
    settings = get_settings()
    raw_key       = str(llm_api_key or openai_api_key or "").strip()
    provider      = detect_provider(raw_key, llm_provider)
    effective_key = _effective_key(raw_key, provider)

    logger.info("call_structured_agent: provider=%s key_present=%s", provider, bool(effective_key))

    if not effective_key:
        if settings.strict_llm_mode:
            raise LLMUnavailableError("No LLM API key available.")
        logger.warning("call_structured_agent: no key — falling back to heuristic")
        return None

    user_prompt = json.dumps(payload, ensure_ascii=True)
    bounded_system, bounded_user = enforce_prompt_budget(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_prompt_tokens=settings.max_prompt_tokens,
    )

    if provider == "anthropic":
        result = _call_anthropic(bounded_system, bounded_user, max_output_tokens, effective_key)
    elif provider == "gemini":
        result = _call_gemini(bounded_system, bounded_user, max_output_tokens, effective_key)
    elif provider == "huggingface":
        result = _call_huggingface(bounded_system, bounded_user, max_output_tokens, effective_key)
    else:
        result = _call_openai(bounded_system, bounded_user, max_output_tokens, effective_key)

    if result is None:
        logger.warning("call_structured_agent: LLM returned None for provider=%s", provider)
        if settings.strict_llm_mode:
            raise LLMInvocationError(f"LLM call failed for provider={provider}")
    return result
