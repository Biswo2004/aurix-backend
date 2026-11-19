"""Pytest fixtures backed by a temporary SQLite database."""

import os
import tempfile
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app import db
from app.db import Base


@pytest.fixture
async def database() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    handle, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(handle)
    engine = create_async_engine(
        "sqlite+aiosqlite:///" + path.replace("\\", "/"),
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    previous = db.SessionLocal
    db.SessionLocal = session_factory
    try:
        yield session_factory
    finally:
        db.SessionLocal = previous
        await engine.dispose()
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
async def client(database: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncClient]:
    from app.auth import get_current_user
    from app.main import app

    async def override_user() -> str:
        return "user-123"

    app.dependency_overrides[get_current_user] = override_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
    app.dependency_overrides.clear()


@pytest.fixture
async def anon_client() -> AsyncIterator[AsyncClient]:
    from app.main import app

    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
