"""Runtime configuration loaded from the environment."""

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field, field_validator, model_validator
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
    # Model roles. PRIMARY_MODEL answers chats; FALLBACK_MODEL takes over on 429/5xx;
    # LIGHT_MODEL does short helper calls (suggestions, memory). GEMMA_MODEL / HELPER_MODEL are the old names.
    PRIMARY_MODEL: str = ""
    GEMMA_MODEL: str = ""
    ALLOWED_ORIGINS: str = Field(min_length=1)
    FALLBACK_MODEL: str = ""
    LIGHT_MODEL: str = ""
    HELPER_MODEL: str = ""
    # Thinking level per model family. Empty string = do not send a thinking setting.
    # Gemma 4: "minimal" (off) or "high". Gemini 3.x: "low", "medium" or "high" ("minimal" is rejected by 3.8 Flash); empty = model default (fastest on Flash-Lite).
    GROQ_API_KEY: str = ""  # optional; enables fallback ids like "groq:openai/gpt-oss-120b"
    GEMMA_THINKING: str = "minimal"
    GEMINI_THINKING: str = ""

    @field_validator("SUPABASE_URL")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("DATABASE_URL", "GOOGLE_API_KEY", "PRIMARY_MODEL", "GEMMA_MODEL", "ALLOWED_ORIGINS", "FALLBACK_MODEL", "LIGHT_MODEL", "HELPER_MODEL", "GEMMA_THINKING", "GEMINI_THINKING", "GROQ_API_KEY")
    @classmethod
    def _strip_value(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _need_primary(self) -> "Settings":
        if not (self.PRIMARY_MODEL or self.GEMMA_MODEL):
            raise ValueError("Set PRIMARY_MODEL (or the older GEMMA_MODEL)")
        return self

    @property
    def primary_model(self) -> str:
        return self.PRIMARY_MODEL or self.GEMMA_MODEL

    @property
    def fallback_models(self) -> list[str]:
        """FALLBACK_MODEL may hold several ids separated by commas, tried in order."""
        return [m.strip() for m in self.FALLBACK_MODEL.split(",") if m.strip()]

    @property
    def light_model(self) -> str:
        google_fallbacks = [m for m in self.fallback_models if not m.startswith("groq:")]
        return self.LIGHT_MODEL or self.HELPER_MODEL or (google_fallbacks[0] if google_fallbacks else "") or self.primary_model

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
