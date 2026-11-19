"""Async SQLAlchemy engine, sessions, and ownership lookups."""

import asyncio
import ssl
import uuid
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# Supabase session-pooler URLs are issued with sslmode=require, which encrypts
# the connection without verifying the pooler's certificate chain. asyncpg's
# default context verifies that chain and fails on Windows (and on some
# Supavisor certificates) with CERTIFICATE_VERIFY_FAILED.
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "statement_cache_size": 0,
        "ssl": _ssl_context,
    },
)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

background_tasks: set[asyncio.Task[None]] = set()


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def spawn_background(coro: "asyncio.Future[None] | asyncio.coroutines.Coroutine[object, object, None]") -> None:
    """Run a coroutine after the response without blocking the client."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a session and commit when the request succeeds."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_conversation_for_user(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: str,
):
    """Return a conversation owned by ``user_id`` or ``None``."""
    from app.models import Conversation

    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()
