"""Groq (OpenAI-compatible) streaming, used as a last-resort fallback model.

Model ids in FALLBACK_MODEL use the prefix ``groq:``, for example ``groq:openai/gpt-oss-120b``.
Uses httpx, which is already a dependency, so no new package is needed.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GROQ_PREFIX = "groq:"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(Exception):
    """A Groq call failed. ``status_code`` lets the shared retry check treat 429/5xx as retryable."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(f"{status_code} {message}")
        self.status_code = status_code


def is_groq_model(model: str) -> bool:
    return model.startswith(GROQ_PREFIX)


def contents_to_messages(system: str, contents: list[Any]) -> list[dict[str, str]]:
    """Convert Gemini-style contents to chat messages. Text only; image parts are skipped."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for item in contents:
        text = "\n".join(
            part.text for part in (getattr(item, "parts", None) or []) if getattr(part, "text", None)
        ).strip()
        if not text:
            continue
        role = "assistant" if getattr(item, "role", "user") == "model" else "user"
        messages.append({"role": role, "content": text})
    return messages


async def stream_groq(
    model: str,
    system: str,
    contents: list[Any],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str]:
    """Yield answer text from Groq. Reasoning output is not requested, so only answer text arrives."""
    api_key = get_settings().GROQ_API_KEY
    if not api_key:
        raise GroqError(None, "GROQ_API_KEY is not set")
    body: dict[str, Any] = {
        "model": model.removeprefix(GROQ_PREFIX),
        "messages": contents_to_messages(system, contents),
        "stream": True,
        "reasoning_effort": "low",
        "include_reasoning": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        async with client.stream("POST", GROQ_URL, headers=headers, json=body) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", "replace")[:200]
                raise GroqError(response.status_code, detail)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    choices = json.loads(data).get("choices") or []
                except json.JSONDecodeError:
                    continue
                delta = (choices[0].get("delta") or {}) if choices else {}
                text = delta.get("content")
                if text:
                    yield text


async def complete(
    model: str,
    system: str,
    user_text: str,
    *,
    max_tokens: int = 400,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """One non-streaming Groq call for short helper tasks (follow-up suggestions, memory)."""
    api_key = get_settings().GROQ_API_KEY
    if not api_key:
        raise GroqError(None, "GROQ_API_KEY is not set")
    body: dict[str, Any] = {
        "model": model.removeprefix(GROQ_PREFIX),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0), transport=transport) as client:
        response = await client.post(GROQ_URL, headers=headers, json=body)
    if response.status_code != 200:
        raise GroqError(response.status_code, response.text[:200])
    choices = response.json().get("choices") or []
    return ((choices[0].get("message") or {}).get("content") or "") if choices else ""
