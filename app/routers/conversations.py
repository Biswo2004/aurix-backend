"""Conversation list, create, read, rename, and delete."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.db import get_db
from app.models import Conversation, Message, utcnow
from app.schemas import (
    ConversationCreateResponse,
    ConversationDetail,
    ConversationListItem,
    ConversationRename,
    FileOut,
    MessageOut,
)

router = APIRouter(tags=["conversations"])


def _preview(text: str | None) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if not compact:
        return None
    if len(compact) <= 160:
        return compact
    return compact[:157] + "..."


@router.get("/conversations", response_model=list[ConversationListItem])
async def list_conversations(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> list[ConversationListItem]:
    last_message = (
        select(Message.content)
        .where(Message.conversation_id == Conversation.id)
        .order_by(Message.created_at.desc())
        .limit(1)
        .correlate(Conversation)
        .scalar_subquery()
    )
    result = await session.execute(
        select(Conversation, last_message)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    items: list[ConversationListItem] = []
    for conversation, preview in result.all():
        items.append(
            ConversationListItem(
                id=conversation.id,
                title=conversation.title,
                last_message_preview=_preview(preview),
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
        )
    return items


@router.post("/conversations", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Conversation:
    conversation = Conversation(user_id=user_id, title="New chat")
    session.add(conversation)
    await session.flush()
    return conversation


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationDetail:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .options(selectinload(Conversation.messages), selectinload(Conversation.files))
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageOut.model_validate(message) for message in conversation.messages],
        files=[FileOut.model_validate(record) for record in conversation.files],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationCreateResponse)
async def rename_conversation(
    conversation_id: uuid.UUID,
    body: ConversationRename,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Conversation:
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    conversation.title = body.title.strip()
    conversation.updated_at = utcnow()
    await session.flush()
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await session.delete(conversation)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
