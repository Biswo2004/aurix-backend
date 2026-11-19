"""Supabase JWT verification using the project's JWKS endpoint."""

import json
import logging
import time
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from app.config import get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)
_JWKS_TTL_SECONDS = 3600
_cache: dict[str, Any] = {"keys": None, "fetched_at": 0.0}

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing, invalid, or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


async def _fetch_jwks(*, force: bool = False) -> list[dict[str, Any]]:
    """Fetch JWKS from Supabase and cache the key set."""
    now = time.time()
    cached = _cache["keys"]
    if not force and isinstance(cached, list) and now - float(_cache["fetched_at"]) < _JWKS_TTL_SECONDS:
        return cached

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(settings.jwks_url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("jwks fetch failed: %s", exc.__class__.__name__)
        if isinstance(cached, list):
            return cached
        raise _UNAUTHORIZED from exc

    keys = payload.get("keys")
    if not isinstance(keys, list) or not keys:
        logger.warning("jwks payload did not contain keys")
        raise _UNAUTHORIZED

    _cache["keys"] = keys
    _cache["fetched_at"] = now
    return keys


def _public_key_for(jwk: dict[str, Any]) -> Any:
    algorithm = jwk.get("alg")
    kty = jwk.get("kty")
    encoded = json.dumps(jwk)
    if algorithm == "ES256" or kty == "EC":
        return ECAlgorithm.from_jwk(encoded)
    if algorithm == "RS256" or kty == "RSA":
        return RSAAlgorithm.from_jwk(encoded)
    raise jwt.InvalidTokenError("Unsupported signing key")


def _select_jwk(keys: list[dict[str, Any]], token: str) -> dict[str, Any]:
    header = jwt.get_unverified_header(token)
    algorithm = header.get("alg")
    if algorithm not in {"ES256", "RS256"}:
        raise jwt.InvalidTokenError("Unsupported signing algorithm")
    kid = header.get("kid")
    if kid:
        for key in keys:
            if key.get("kid") == kid:
                return key
        raise jwt.InvalidTokenError("Signing key not found")
    if len(keys) == 1:
        return keys[0]
    raise jwt.InvalidTokenError("Signing key not found")


async def _decode_token(token: str) -> dict[str, Any]:
    keys = await _fetch_jwks()
    try:
        jwk = _select_jwk(keys, token)
    except jwt.InvalidTokenError:
        keys = await _fetch_jwks(force=True)
        jwk = _select_jwk(keys, token)
    public_key = _public_key_for(jwk)
    payload = jwt.decode(
        token,
        public_key,
        algorithms=["ES256", "RS256"],
        audience="authenticated",
        leeway=30,
        options={"require": ["exp", "sub", "aud"]},
    )
    return payload


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """FastAPI dependency that returns the Supabase user id (``sub``)."""
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials.strip():
        raise _UNAUTHORIZED
    token = credentials.credentials.strip()
    try:
        payload = await _decode_token(token)
    except jwt.PyJWTError as exc:
        logger.info("jwt rejected: %s", exc.__class__.__name__)
        raise _UNAUTHORIZED from exc
    except HTTPException:
        raise
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise _UNAUTHORIZED
    return subject.strip()
