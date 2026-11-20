"""Non-streaming extraction of durable user facts."""

import logging

from sqlalchemy import delete, select

from app import db
from app.models import Memory
from app.services import llm as llm_service

logger = logging.getLogger(__name__)

_MAX_FACTS_PER_TURN = 3
_MAX_STORED_FACTS = 100
_FACT_CHARS = 500

_MEMORY_INSTRUCTION = (
    "Extract 0 to 3 durable facts the user stated about themselves. "
    "Durable means identity, stable preferences, or long-term details. "
    "Do not save questions, file contents, passwords, or one-off tasks. "
    "Return only a JSON array of short strings. If there is nothing to save, return []."
)


async def _generate_fact_text(user_message: str, assistant_message: str) -> str:
    client = llm_service.get_genai_client()
    contents = (
        "Conversation excerpt:\n"
        f"User: {user_message[:4000]}\n"
        f"Assistant: {assistant_message[:4000]}"
    )
    return await llm_service.helper_generate(client, contents, _MEMORY_INSTRUCTION, 0.2)


async def extract_and_store_memories(user_id: str, user_message: str, assistant_message: str) -> None:
    """Store new durable facts for the user. Failures are logged and ignored."""
    try:
        raw = await _generate_fact_text(user_message, assistant_message)
        facts = [fact.strip()[:_FACT_CHARS] for fact in llm_service.extract_json_array(raw) if fact.strip()]
        facts = facts[:_MAX_FACTS_PER_TURN]
        if not facts:
            return
        async with db.SessionLocal() as session:
            existing = await session.execute(select(Memory).where(Memory.user_id == user_id))
            known = {row.fact.casefold() for row in existing.scalars().all()}
            for fact in facts:
                if fact.casefold() in known:
                    continue
                session.add(Memory(user_id=user_id, fact=fact))
                known.add(fact.casefold())
            await session.flush()
            stored = await session.execute(
                select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
            )
            rows = list(stored.scalars().all())
            overflow = rows[_MAX_STORED_FACTS:]
            if overflow:
                await session.execute(delete(Memory).where(Memory.id.in_([row.id for row in overflow])))
            await session.commit()
    except Exception as exc:
        logger.info("memory extraction skipped: %s", exc.__class__.__name__)
