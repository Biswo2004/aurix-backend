"""Gemini streaming, prompts, and chat persistence for AURIX AI."""

import asyncio
import base64
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import db
from app.config import get_settings
from app.models import Conversation, File, Memory, Message
from app.services.retriever import load_conversation_chunks, load_tabular_summaries, rank_chunks
from app.services.safety import check_message

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are AURIX AI, a helpful and honest assistant. "
    "When the question is about the user's uploaded files, answer from the provided file context. "
    "Cite those sources as [1], [2], matching the numbered excerpts. "
    'If the files do not contain the answer, say "I couldn\'t find that in your files" instead of guessing. '
    "For general questions, answer from your own knowledge and do not invent file citations. "
    "Use the stored facts about the user when they are relevant. "
    "Reply in well-structured markdown."
)

_NAME_PART = re.compile(r"[^\W\d_][\w'’.-]*", re.UNICODE)


def first_name(raw: str | None) -> str | None:
    """Return a safe first name for the prompt, or None. Letters, apostrophes, dots and hyphens only."""
    if not raw:
        return None
    match = _NAME_PART.search(raw.strip())
    if match is None:
        return None
    name = match.group(0).strip(".-'’")[:30]
    return name or None


def system_instruction_for(user_name: str | None) -> str:
    name = first_name(user_name)
    if name is None:
        return SYSTEM_INSTRUCTION
    return (
        SYSTEM_INSTRUCTION
        + f' The user\'s first name is "{name}". When you greet the user or start a conversation, '
        f'address them by first name, for example: "Hello {name}! I am AURIX AI, your helpful and honest '
        'assistant. How can I help you today?" Do not repeat their name in every reply.'
    )


_SUGGESTION_INSTRUCTION = (
    "Suggest exactly 3 short follow-up questions the user might ask next. "
    "Return only a JSON array of 3 strings, with no markdown."
)

_MAX_IMAGES = 6
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMError(Exception):
    """A model call failed after retries."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def get_helper_model() -> str:
    """Model for short helper calls (suggestions, memory): the fallback model if set, else the primary."""
    settings = get_settings()
    return settings.FALLBACK_MODEL or settings.GEMMA_MODEL


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=get_settings().GOOGLE_API_KEY)


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    if code in {429, 500, 502, 503, 504}:
        return True
    text = str(exc).upper()
    return any(token in text for token in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def _status_of(exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return str(status or exc.__class__.__name__)


async def with_gemini_retries(operation: Callable[[], Awaitable[Any]]) -> Any:
    """Run ``operation`` and retry 429/5xx responses up to 3 times."""
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt >= 3:
                raise
            logger.warning("gemini retry attempt=%s status=%s", attempt + 1, _status_of(exc))
            await asyncio.sleep(delay)
            delay *= 2
    if last_error is not None:
        raise last_error
    raise LLMError("AURIX AI is busy right now. Please try again in a moment.")


def extract_json_array(text: str) -> list[str]:
    cleaned = _JSON_FENCE.sub("", (text or "").strip()).strip()
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match is None:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    values: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
    return values


def title_from_message(message: str) -> str:
    words = message.strip().split()
    title = " ".join(words[:6]).strip()
    return (title or "New chat")[:255]


def extract_answer_text(chunk: Any) -> str:
    """Return visible answer text, skipping thought or reasoning parts."""
    candidates = getattr(chunk, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        if parts:
            visible: list[str] = []
            for part in parts:
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", None)
                if text:
                    visible.append(text)
            return "".join(visible)
    return ""


def _text_part(text: str) -> Any:
    if hasattr(types.Part, "from_text"):
        return types.Part.from_text(text=text)
    return types.Part(text=text)


def _content(role: str, parts: list[Any]) -> Any:
    return types.Content(role=role, parts=parts)


async def _await_maybe(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _stream_model_text(contents: list[Any], user_name: str | None = None) -> AsyncIterator[str]:
    settings = get_settings()
    client = get_genai_client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction_for(user_name),
        temperature=0.3,
    )
    delay = 0.5
    emitted = False
    for attempt in range(4):
        # First try the primary model; after a 429/5xx switch to the fallback model (if configured).
        model = settings.GEMMA_MODEL if attempt == 0 or not settings.FALLBACK_MODEL else settings.FALLBACK_MODEL
        try:
            stream = await _await_maybe(
                client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
            )
            async for chunk in stream:
                text = extract_answer_text(chunk)
                if not text:
                    continue
                emitted = True
                yield text
            return
        except Exception as exc:
            if emitted or not is_retryable(exc) or attempt >= 3:
                logger.warning("gemini stream failed model=%s status=%s", model, _status_of(exc))
                raise LLMError("AURIX AI is busy right now. Please try again in a moment.") from exc
            logger.warning("gemini stream retry attempt=%s model=%s status=%s", attempt + 1, model, _status_of(exc))
            await asyncio.sleep(delay)
            delay *= 2


async def _suggestion_text(question: str, answer: str) -> str:
    settings = get_settings()
    client = get_genai_client()
    contents = (
        "Write follow-up questions as a JSON array.\n"
        f"Question: {question[:2000]}\n"
        f"Answer: {answer[:4000]}"
    )
    config = types.GenerateContentConfig(
        system_instruction=_SUGGESTION_INSTRUCTION,
        temperature=0.3,
    )

    async def _call() -> str:
        result = await _await_maybe(
            client.aio.models.generate_content(
                model=get_helper_model(),
                contents=contents,
                config=config,
            )
        )
        text = getattr(result, "text", None)
        if text:
            return text
        return extract_answer_text(result)

    return await with_gemini_retries(_call)


async def fetch_suggestions(question: str, answer: str) -> list[str]:
    try:
        raw = await asyncio.wait_for(_suggestion_text(question, answer), timeout=12)
        return extract_json_array(raw)[:3]
    except Exception as exc:
        logger.info("suggestions skipped: %s", exc.__class__.__name__)
        return []


def _snippet(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 200:
        return compact
    return compact[:197] + "..."


async def _recent_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(10)
    )
    return list(reversed(result.scalars().all()))


async def _memory_facts(session: AsyncSession, user_id: str) -> list[str]:
    result = await session.execute(
        select(Memory.fact).where(Memory.user_id == user_id).order_by(Memory.created_at.desc()).limit(30)
    )
    return [fact for fact in reversed(result.scalars().all())]


async def _image_parts(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: str,
) -> list[Any]:
    result = await session.execute(
        select(File)
        .where(
            File.conversation_id == conversation_id,
            File.user_id == user_id,
            File.image_b64.is_not(None),
        )
        .order_by(File.created_at.desc())
        .limit(_MAX_IMAGES)
    )
    parts: list[Any] = []
    for record in reversed(result.scalars().all()):
        if not record.image_b64:
            continue
        try:
            data = base64.b64decode(record.image_b64)
            parts.append(types.Part.from_bytes(data=data, mime_type=record.mime))
        except Exception as exc:
            logger.info("image attach skipped file=%s error=%s", record.name, exc.__class__.__name__)
    return parts


def _history_contents(messages: list[Message]) -> list[Any]:
    contents: list[Any] = []
    for message in messages:
        text = message.content.strip()
        if not text:
            continue
        role = "user" if message.role == "user" else "model"
        contents.append(_content(role, [_text_part(text)]))
    return contents


def _context_entries(
    summaries: list[tuple[uuid.UUID, str, str]],
    chunks: list[Any],
) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    index = 1
    for file_id, file_name, summary in summaries:
        blocks.append(f"[{index}] {file_name}\n{summary}")
        sources.append(
            {
                "index": index,
                "file_id": str(file_id),
                "file_name": file_name,
                "page": None,
                "snippet": _snippet(summary),
            }
        )
        index += 1
    for chunk in chunks:
        page_label = f" page {chunk.page}" if chunk.page else ""
        blocks.append(f"[{index}] {chunk.file_name}{page_label}\n{chunk.text}")
        sources.append(
            {
                "index": index,
                "file_id": str(chunk.file_id),
                "file_name": chunk.file_name,
                "page": chunk.page,
                "snippet": _snippet(chunk.text),
            }
        )
        index += 1
    return "\n\n".join(blocks), sources


def _user_prompt(
    message: str,
    memories: list[str],
    context: str,
    file_mode: bool,
) -> str:
    lines: list[str] = []
    if memories:
        lines.append("Known facts about the user:")
        lines.extend(f"- {fact}" for fact in memories)
        lines.append("")
    if file_mode:
        if context:
            lines.append("File context:")
            lines.append(context)
            lines.append("")
        else:
            lines.append("No matching excerpts were retrieved from the uploaded files.")
            lines.append("")
    lines.append("Question:")
    lines.append(message.strip())
    return "\n".join(lines)


async def _save_turn(
    session: AsyncSession,
    conversation: Conversation,
    user_text: str,
    assistant_text: str,
    sources: list[dict[str, Any]] | None,
    *,
    first_turn: bool,
) -> Message:
    user_time = datetime.now(timezone.utc)
    if first_turn and conversation.title == "New chat":
        conversation.title = title_from_message(user_text)
    session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=user_text,
            sources=None,
            created_at=user_time,
        )
    )
    assistant = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_text,
        sources=sources or [],
        created_at=user_time + timedelta(microseconds=1),
    )
    session.add(assistant)
    conversation.updated_at = user_time + timedelta(microseconds=1)
    await session.flush()
    await session.commit()
    return assistant


async def iter_chat_sse(
    session: AsyncSession,
    user_id: str,
    conversation_id: uuid.UUID,
    message: str,
    use_files: bool,
    user_name: str | None = None,
) -> AsyncIterator[str]:
    """Yield SSE events for one chat turn and persist both messages."""
    from app.db import get_conversation_for_user
    from app.services.memory import extract_and_store_memories

    t_start = time.perf_counter()

    conversation = await get_conversation_for_user(session, conversation_id, user_id)
    if conversation is None:
        yield sse({"type": "error", "message": "Conversation not found"})
        return

    allowed, refusal = check_message(message)
    history = await _recent_messages(session, conversation.id)
    first_turn = not any(item.role == "user" for item in history)

    if not allowed:
        assistant = await _save_turn(
            session,
            conversation,
            message,
            refusal,
            [],
            first_turn=first_turn,
        )
        yield sse({"type": "token", "text": refusal})
        yield sse(
            {
                "type": "done",
                "message_id": str(assistant.id),
                "sources": [],
                "suggestions": [],
            }
        )
        return

    file_count = await session.execute(
        select(File.id).where(File.conversation_id == conversation.id, File.user_id == user_id).limit(1)
    )
    file_mode = use_files and file_count.first() is not None
    memories = await _memory_facts(session, user_id)
    sources: list[dict[str, Any]] = []
    context = ""
    image_parts: list[Any] = []
    if file_mode:
        stored_chunks = await load_conversation_chunks(session, conversation.id, user_id)
        ranked = rank_chunks(stored_chunks, message, limit=5)
        summaries = await load_tabular_summaries(session, conversation.id, user_id)
        context, sources = _context_entries(summaries, ranked)
        image_parts = await _image_parts(session, conversation.id, user_id)

    prompt = _user_prompt(message, memories, context, file_mode)
    contents = _history_contents(history)
    user_parts = [*image_parts, _text_part(prompt)]
    contents.append(_content("user", user_parts))
    # End the read transaction so the pool connection is not held while the model streams.
    await session.commit()

    pieces: list[str] = []
    t_prep = time.perf_counter()
    t_first: float | None = None
    try:
        async for text in _stream_model_text(contents, user_name):
            if t_first is None:
                t_first = time.perf_counter()
            pieces.append(text)
            yield sse({"type": "token", "text": text})
    except LLMError as exc:
        yield sse({"type": "error", "message": exc.message})
        return
    except Exception:
        logger.exception("chat generation failed conversation=%s", conversation_id)
        yield sse(
            {
                "type": "error",
                "message": "Something went wrong while generating a reply. Please try again.",
            }
        )
        return

    answer = "".join(pieces).strip()
    if not answer:
        answer = "I couldn't generate a response. Please try again."
        yield sse({"type": "token", "text": answer})

    try:
        assistant = await _save_turn(
            session,
            conversation,
            message,
            answer,
            sources,
            first_turn=first_turn,
        )
    except Exception:
        logger.exception("chat persist failed conversation=%s", conversation_id)
        yield sse(
            {
                "type": "error",
                "message": "The reply was generated but could not be saved. Please try again.",
            }
        )
        return

    t_answer = time.perf_counter()
    # Send "done" right away so the UI unlocks as soon as the answer is complete.
    yield sse(
        {
            "type": "done",
            "message_id": str(assistant.id),
            "sources": sources,
            "suggestions": [],
        }
    )
    # Start memory extraction before the suggestions call so it survives a client disconnect.
    db.spawn_background(extract_and_store_memories(user_id, message, answer))
    t_done = time.perf_counter()
    suggestions = await fetch_suggestions(message, answer)
    logger.info(
        "chat timing prep_ms=%d first_token_ms=%d answer_ms=%d save_ms=%d suggestions_ms=%d file_mode=%s",
        (t_prep - t_start) * 1000,
        ((t_first or t_prep) - t_prep) * 1000,
        (t_answer - (t_first or t_prep)) * 1000,
        (t_done - t_answer) * 1000,
        (time.perf_counter() - t_done) * 1000,
        file_mode,
    )
    if suggestions:
        yield sse({"type": "suggestions", "suggestions": suggestions})
