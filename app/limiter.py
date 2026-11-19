"""Shared rate limiter instance."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_address(request: Request) -> str:
    """Prefer the first forwarded client address when a proxy sets it."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_address, headers_enabled=True)
