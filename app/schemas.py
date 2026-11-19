"""Pydantic request and response models."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListItem(BaseModel):
    id: uuid.UUID
    title: str
    last_message_preview: str | None
    created_at: datetime
    updated_at: datetime


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    sources: list[dict] | None = None
    feedback: str | None = None
    created_at: datetime


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    mime: str
    size: int
    created_at: datetime


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
    files: list[FileOut]


class ChatRequest(BaseModel):
    message: str
    use_files: bool = True


class FeedbackIn(BaseModel):
    value: Literal["up", "down"]
    reason: str | None = Field(default=None, max_length=500)


class FeedbackOut(BaseModel):
    id: uuid.UUID
    feedback: str


class HealthOut(BaseModel):
    status: str
