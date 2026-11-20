"""Runtime configuration loaded from the environment."""

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url


def normalize_database_url(url: str) -> str:
    """Return a SQLAlchemy asyncpg URL with a correctly encoded userinfo.

    Passwords that contain reserved characters such as ``@``, ``#``, or ``%``
    are percent-encoded when the raw URL would otherwise be split in the wrong
    place. URLs that already parse to the intended host are left unchanged.
    A plain ``postgresql://`` scheme is rewritten to ``postgresql+asyncpg://``.
    """
    raw = url.strip()
    if not raw:
        raise ValueError("DATABASE_URL is empty")

    scheme, separator, rest = raw.partition("://")
    if not separator or "@" not in rest:
        raise ValueError("DATABASE_URL must include a scheme and credentials")

    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" not in userinfo:
        raise ValueError("DATABASE_URL must include a username and password")

    expected_host = hostpart.split("/", 1)[0].split(":", 1)[0]
    parsed_ok = False
    try:
        parsed = make_url(raw)
        parsed_ok = bool(parsed.host) and parsed.host == expected_host
    except Exception:
        parsed_ok = False

    if parsed_ok:
        normalized = raw
    else:
        username, password = userinfo.split(":", 1)
        encoded_user = quote(username, safe="")
        encoded_password = quote(password, safe="")
        normalized = f"{scheme}://{encoded_user}:{encoded_password}@{hostpart}"

    if normalized.startswith("postgresql://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgresql://") :]
    elif normalized.startswith("postgres://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgres://") :]

    parsed = make_url(normalized)
    query = dict(parsed.query)
    removed = False
    for key in ("sslmode", "ssl", "channel_binding"):
        if key in query:
            query.pop(key)
            removed = True
    if removed:
        # asyncpg rejects libpq's sslmode query parameter. TLS is set in connect_args.
        normalized = parsed.set(query=query).render_as_string(hide_password=False)
    return normalized


class Settings(BaseSettings):
    """Application settings. Secrets come only from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    SUPABASE_URL: str = Field(min_length=1)
    DATABASE_URL: str = Field(min_length=1)
    GOOGLE_API_KEY: str = Field(min_length=1)
    GEMMA_MODEL: str = Field(min_length=1)
    ALLOWED_ORIGINS: str = Field(min_length=1)
    # Optional second model used when the primary model returns 503/429, and for helper calls.
    FALLBACK_MODEL: str = ""

    @field_validator("SUPABASE_URL")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("DATABASE_URL", "GOOGLE_API_KEY", "GEMMA_MODEL", "ALLOWED_ORIGINS", "FALLBACK_MODEL")
    @classmethod
    def _strip_value(cls, value: str) -> str:
        return value.strip()

    @property
    def async_database_url(self) -> str:
        return normalize_database_url(self.DATABASE_URL)

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def jwks_url(self) -> str:
        return f"{self.SUPABASE_URL}/auth/v1/.well-known/jwks.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
