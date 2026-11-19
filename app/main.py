"""AURIX AI FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from app import models
from app.config import get_settings
from app.db import Base, engine
from app.limiter import limiter
from app.routers import chat, conversations, files, misc

logger = logging.getLogger("aurix")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


async def prepare_database() -> None:
    """Create tables and enable RLS so the public Data API cannot read them."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        if connection.dialect.name != "postgresql":
            return
        bypass = await connection.scalar(
            text("SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )
        if not bypass:
            logger.warning(
                "row level security was not enabled because the database role does not bypass RLS"
            )
            return
        for table in ("conversations", "messages", "files", "chunks", "memories"):
            await connection.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))
    logger.info("database schema ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    await prepare_database()
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title="AURIX AI", version="1.0.0", lifespan=lifespan)
    application.state.limiter = limiter
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(misc.router)
    application.include_router(conversations.router, prefix="/api")
    application.include_router(files.router, prefix="/api")
    application.include_router(chat.router, prefix="/api")
    _register_handlers(application)

    @application.middleware("http")
    async def log_request(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request failed method=%s path=%s", request.method, request.url.path)
            raise
        logger.info(
            "request method=%s path=%s status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    return application


def _register_handlers(application: FastAPI) -> None:
    @application.exception_handler(RateLimitExceeded)
    async def rate_limited(_request: Request, _exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait and try again."},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @application.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc.__class__.__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


configure_logging()
app = create_app()

# Imported so model metadata is registered before startup creates tables.
_ = models
