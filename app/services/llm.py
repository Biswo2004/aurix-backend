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
from app.services import groq as groq_service
from app.services.retriever import load_conversation_chunks, load_tabular_summaries, rank_chunks
from app.services.safety import check_message

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are AURIX AI, a helpful and honest assistant. "
    "When the question is about the user's uploaded files, answer from the provided file context. "
    "Cite every fact taken from the files by putting the excerpt number in square brackets right after it, "
    "like [2] or [1][3]. Each excerpt header shows its file and page, and the page is shown to the user. "
    "Cite only excerpts you actually used, and never invent a number. "
    "Write every citation as a plain number in square brackets such as [2]; never use 【 】 marks, line ranges or any other format. "
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


def system_instruction_for(user_name: str | None, part_of_day: str | None = None) -> str:
    name = first_name(user_name)
    part = part_of_day if part_of_day in {"morning", "afternoon", "evening", "night"} else None
    if name is None and part is None:
        return SYSTEM_INSTRUCTION
    text = SYSTEM_INSTRUCTION
    if name is not None:
        text += f' The user\'s first name is "{name}".'
    if part is not None:
        text += f" It is currently {part} for the user."
    greeting = "greet them"
    if name is not None:
        greeting += " by first name"
    if part is not None:
        greeting += f" with a greeting that fits the {part}"
    text += (
        " If the user's message is only a greeting or small talk (for example \"hi\" or \"hello\"), "
        f"reply warmly in 2 to 3 sentences: {greeting}, add one short uplifting quote that fits the moment "
        "(name its author only if you are certain of it, otherwise write a short original line), "
        "and invite them to ask something. Vary your wording every time, never reuse the same opening, "
        "and mention that you are AURIX AI at most once per conversation. "
        "For every other message, answer directly with no greeting and no quote, "
        "and do not repeat their name in every reply."
    )
    return text


_SUGGESTION_INSTRUCTION = (
    "Suggest exactly 3 follow-up questions the user might ask next, based on the answer. "
    "Write each from the user's point of view, specific to the topic, under 60 characters, "
    "and different from each other (for example: go deeper, a practical example, a related next step). "
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
    """Model for short helper calls (suggestions, memory): the light model."""
    return get_settings().light_model


# Models that rejected a thinking setting with a 400; we stop sending it to them for this process.
_thinking_rejected: set[str] = set()


def thinking_config_for(model: str) -> Any:
    """Thinking setting per model family, or None to send nothing.

    Gemma 4: "minimal" turns thinking off. Gemini 3.x: "low" gives the fastest first word
    ("minimal" is rejected by gemini-3.8-flash).
    """
    if model in _thinking_rejected:
        return None
    settings = get_settings()
    level = settings.GEMMA_THINKING if model.lower().startswith("gemma") else settings.GEMINI_THINKING
    if not level:
        return None
    try:
        return types.ThinkingConfig(thinking_level=level)
    except Exception:  # older google-genai without thinking_level
        return None


def thinking_was_rejected(exc: Exception, model: str) -> bool:
    """True if a 400 error was caused by the thinking setting; remembers the model so we stop sending it."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if str(status) != "400" and "400" not in str(exc):
        return False
    if "thinking" not in str(exc).lower():
        return False
    _thinking_rejected.add(model)
    logger.warning("thinking setting rejected model=%s; retrying without it", model)
    return True


def build_config(model: str, **kwargs: Any) -> Any:
    # Gemini 3.x ignores or discourages temperature; only Gemma gets it.
    if not model.lower().startswith("gemma"):
        kwargs.pop("temperature", None)
    thinking = thinking_config_for(model)
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    return types.GenerateContentConfig(**kwargs)


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


def _has_images(contents: list[Any]) -> bool:
    for item in contents:
        for part in getattr(item, "parts", None) or []:
            if getattr(part, "inline_data", None) is not None:
                return True
    return False


async def _stream_model_text(
    contents: list[Any], user_name: str | None = None, part_of_day: str | None = None
) -> AsyncIterator[str]:
    settings = get_settings()
    client = get_genai_client()
    instruction = system_instruction_for(user_name, part_of_day)
    delay = 0.5
    emitted = False
    attempt = 0
    # Primary first; after each 429/5xx move to the next fallback model; the last model gets one extra try.
    chain = [settings.primary_model, *[m for m in settings.fallback_models if m != settings.primary_model]]
    # Groq is text-only and needs its key: leave it out when images are attached or the key is missing.
    if not settings.GROQ_API_KEY or _has_images(contents):
        chain = [m for m in chain if not groq_service.is_groq_model(m)]
    if not chain:
        raise LLMError("No usable model is configured. Check PRIMARY_MODEL, FALLBACK_MODEL and GROQ_API_KEY.")
    max_attempts = len(chain) + 1
    while True:
        model = chain[min(attempt, len(chain) - 1)]
        try:
            if groq_service.is_groq_model(model):
                async for text in groq_service.stream_groq(model, instruction, contents):
                    emitted = True
                    yield text
                return
            stream = await _await_maybe(
                client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=build_config(model, system_instruction=instruction, temperature=0.3),
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
            if not emitted and thinking_was_rejected(exc, model):
                continue  # same model, no thinking setting; does not use up a retry
            if emitted or not is_retryable(exc) or attempt >= max_attempts - 1:
                logger.warning("gemini stream failed model=%s status=%s", model, _status_of(exc))
                raise LLMError("AURIX AI is busy right now. Please try again in a moment.") from exc
            attempt += 1
            logger.warning("gemini stream retry attempt=%s model=%s status=%s", attempt, model, _status_of(exc))
            if chain[min(attempt, len(chain) - 1)] == model:
                await asyncio.sleep(delay)  # same model again: back off. A different model: switch at once.
                delay *= 2


async def _suggestion_text(question: str, answer: str) -> str:
    client = get_genai_client()
    contents = (
        "Write follow-up questions as a JSON array.\n"
        f"Question: {question[:2000]}\n"
        f"Answer: {answer[:4000]}"
    )
    return await helper_generate(client, contents, _SUGGESTION_INSTRUCTION, 0.3)


async def helper_generate(client: Any, contents: str, instruction: str, temperature: float) -> str:
    """Short helper call: Groq first when it is the primary model, then the light model, then others once each."""
    settings = get_settings()
    candidates = [settings.primary_model, *settings.fallback_models]
    groq_models = [m for m in candidates if groq_service.is_groq_model(m)]
    models: list[str] = []
    if settings.GROQ_API_KEY and groq_models and groq_service.is_groq_model(settings.primary_model):
        models.append(groq_models[0])  # the primary is Groq: it is also the fastest choice for short helper calls
    light = get_helper_model()
    if light not in models:
        models.append(light)
    if settings.GROQ_API_KEY and groq_models and groq_models[0] not in models:
        models.append(groq_models[0])
    for extra in candidates:
        if extra not in models and not groq_service.is_groq_model(extra) and len(models) < 3:
            models.append(extra)
    last: Exception | None = None
    for model in models:
        for _ in range(2):  # second pass only if the thinking setting was rejected
            try:
                if groq_service.is_groq_model(model):
                    return await groq_service.complete(model, instruction, contents)
                result = await _await_maybe(
                    client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=build_config(model, system_instruction=instruction, temperature=temperature),
                    )
                )
                text = getattr(result, "text", None)
                return text or extract_answer_text(result)
            except Exception as exc:
                last = exc
                if thinking_was_rejected(exc, model):
                    continue
                if not is_retryable(exc):
                    raise
                logger.warning("helper call failed model=%s status=%s", model, _status_of(exc))
                break
    assert last is not None
    raise last


def settings_primary() -> str:
    return get_settings().primary_model


async def fetch_suggestions(question: str, answer: str) -> list[str]:
    try:
        raw = await asyncio.wait_for(_suggestion_text(question, answer), timeout=15)
        return extract_json_array(raw)[:3]
    except Exception as exc:
        logger.warning("suggestions skipped: %s %s", exc.__class__.__name__, str(exc)[:160])
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


_CITATION = re.compile(r"\[(\d{1,2})\]")
# gpt-oss sometimes writes citations as 【4†L1-L5】 instead of [4].
_ODD_CITATION = re.compile(r"【\s*(\d{1,2})(?:\s*[†:][^】]*)?】")


def normalize_citations(text: str) -> str:
    """Rewrite 【4†L1-L5】 style markers as [4] so they are counted, saved and shown the same way."""
    return _ODD_CITATION.sub(r"[\1]", text)

# Shown only for answers based on uploaded files, when the model could not write follow-ups in time.
_FILE_FOLLOW_UPS = [
    "Summarize the key points",
    "Which pages cover this in detail?",
    "Explain this in simpler words",
]


def _cited_sources(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the excerpts the answer cites. If it cites none, keep the three best matches."""
    if not sources:
        return sources
    cited = {int(n) for n in _CITATION.findall(answer)}
    kept = [src for src in sources if src["index"] in cited]
    return kept or sources[:3]


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
    part_of_day: str | None = None,
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
        async for text in _stream_model_text(contents, user_name, part_of_day):
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

    answer = normalize_citations("".join(pieces).strip())
    if not answer:
        answer = "I couldn't generate a response. Please try again."
        yield sse({"type": "token", "text": answer})

    sources = _cited_sources(answer, sources)
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
    if not suggestions and file_mode:
        suggestions = list(_FILE_FOLLOW_UPS)
    if suggestions:
        yield sse({"type": "suggestions", "suggestions": suggestions})
