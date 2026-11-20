"""Streaming chat and message feedback."""

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import db
from app.auth import get_current_user
from app.db import get_conversation_for_user, get_db
from app.limiter import limiter
from app.models import Conversation, Message
from app.schemas import ChatRequest, FeedbackIn, FeedbackOut
from app.services.llm import iter_chat_sse

router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/conversations/{conversation_id}/chat")
@limiter.limit("30/minute")
async def chat(
    conversation_id: uuid.UUID,
    body: ChatRequest,
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    async with db.SessionLocal() as session:
        conversation = await get_conversation_for_user(session, conversation_id, user_id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    async def generate():
        async with db.SessionLocal() as session:
            async for event in iter_chat_sse(
                session,
                user_id,
                conversation_id,
                body.message,
                body.use_files,
                body.user_name,
            ):
                yield event

    return StreamingResponse(generate(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/messages/{message_id}/feedback", response_model=FeedbackOut)
async def set_feedback(
    message_id: uuid.UUID,
    body: FeedbackIn,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> FeedbackOut:
    result = await session.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id, Conversation.user_id == user_id)
    )
    message = result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback can only be recorded for assistant messages",
        )
    reason = body.reason.strip() if body.reason else ""
    stored = body.value if not reason else f"{body.value}: {reason}"
    message.feedback = stored[:600]
    await session.flush()
    return FeedbackOut(id=message.id, feedback=message.feedback)


def parse_sse_payload(raw: str) -> dict:
    """Decode one SSE data line. Used by tests and local debugging."""
    prefix = "data: "
    if not raw.startswith(prefix):
        raise ValueError("Not an SSE data line")
    payload = json.loads(raw[len(prefix) :])
    if not isinstance(payload, dict):
        raise ValueError("SSE payload must be an object")
    return payload
