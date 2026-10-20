"""OpenRouter client. All LLM calls in this repo go through this module."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEXT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_VLM_MODEL = "google/gemini-3.1-pro-preview"


def openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to the repo-root .env (see .env.example)."
        )
    return api_key


def openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = (os.getenv("OPENROUTER_TITLE") or os.getenv("OPENROUTER_APP_TITLE") or "Digital Coach").strip()
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def openrouter_client() -> OpenAI:
    """OpenAI SDK pointed at OpenRouter."""
    kwargs: dict[str, Any] = {
        "api_key": openrouter_api_key(),
        "base_url": OPENROUTER_BASE_URL,
    }
    headers = openrouter_headers()
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def text_model(override: str | None = None) -> str:
    return (override or os.getenv("OPENROUTER_MODEL") or DEFAULT_TEXT_MODEL).strip()


def vlm_model(override: str | None = None) -> str:
    return (override or os.getenv("OPENROUTER_VLM_MODEL") or DEFAULT_VLM_MODEL).strip()


def chat_json_text(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """One chat completion. Returns the assistant text."""
    client = openrouter_client()
    response = client.chat.completions.create(
        model=text_model(model),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()
