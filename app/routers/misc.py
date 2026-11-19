"""Health check and user memory deletion."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.db import get_db
from app.models import Memory
from app.schemas import HealthOut

router = APIRouter(tags=["misc"])


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok")


@router.delete("/api/memories", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memories(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    await session.execute(delete(Memory).where(Memory.user_id == user_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
