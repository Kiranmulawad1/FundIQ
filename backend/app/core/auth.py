"""Clerk JWT verification with cached JWKS.

Clerk issues RS256-signed JWTs. We fetch its JWKS once, cache for 1 hour,
verify signatures locally — no per-request round-trip to Clerk.

For local development with `CLERK_SECRET_KEY` unset, we return a synthetic
user so the API is usable without Clerk credentials. The config layer hard-
fails on missing Clerk secrets in production, so this fallback cannot leak.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, Header
from jwt import PyJWKClient
from pydantic import BaseModel, EmailStr, Field

from app.core.config import get_settings
from app.core.exceptions import AuthError
from app.core.logging import get_logger, user_id_ctx

logger = get_logger(__name__)

_JWKS_CACHE_TTL_SECONDS = 3600


class AuthenticatedUser(BaseModel):
    """Typed user surface — no raw claim dicts leak into routes."""

    id: str
    email: EmailStr | None = None
    org_id: str | None = None
    is_dev_user: bool = Field(default=False, description="True when running without Clerk.")


class _JWKSCache:
    """Single-process JWKS cache. Threadsafe via PyJWKClient internals."""

    def __init__(self) -> None:
        self._client: PyJWKClient | None = None
        self._loaded_at: float = 0.0

    def get(self, jwks_url: str) -> PyJWKClient:
        now = time.time()
        if self._client is None or now - self._loaded_at > _JWKS_CACHE_TTL_SECONDS:
            self._client = PyJWKClient(jwks_url, cache_keys=True)
            self._loaded_at = now
        return self._client

    def invalidate(self) -> None:
        self._client = None
        self._loaded_at = 0.0


_jwks_cache = _JWKSCache()


def _decode_clerk_token(token: str, jwks_url: str) -> dict[str, Any]:
    client = _jwks_cache.get(jwks_url)
    try:
        signing_key = client.get_signing_key_from_jwt(token).key
    except jwt.exceptions.PyJWKClientError:
        # Could be key rotation — refresh JWKS once and retry.
        _jwks_cache.invalidate()
        client = _jwks_cache.get(jwks_url)
        try:
            signing_key = client.get_signing_key_from_jwt(token).key
        except jwt.exceptions.PyJWKClientError as e:
            raise AuthError(f"Could not resolve signing key: {e}") from e
    except jwt.InvalidTokenError as e:
        # Malformed JWT (e.g. "not-a-real-jwt") — PyJWKClient raises a
        # JWT InvalidTokenError when the input isn't a valid JWT shape.
        # Without this catch, it bubbles up as a 500.
        raise AuthError(f"Invalid token: {e}") from e

    try:
        return jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk audience varies by instance
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("Token expired.") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}") from e


def _parse_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


async def _verify_clerk_token(token: str) -> AuthenticatedUser:
    settings = get_settings()
    if settings.clerk_jwks_url is None:
        raise AuthError("CLERK_JWKS_URL not configured.")

    claims = _decode_clerk_token(token, settings.clerk_jwks_url)
    user = AuthenticatedUser(
        id=str(claims.get("sub", "")),
        email=claims.get("email"),
        org_id=claims.get("org_id"),
    )
    if not user.id:
        raise AuthError("Token missing subject claim.")
    return user


async def current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthenticatedUser:
    """Required-auth dependency. Returns a dev user when Clerk is not configured."""
    settings = get_settings()
    token = _parse_bearer(authorization)

    if settings.clerk_secret_key is None:
        # Dev mode — config validator forbids this in production.
        # .local is a reserved TLD per RFC 6762 — Pydantic's EmailStr
        # rejects it. example.com is the canonical placeholder.
        user = AuthenticatedUser(id="dev_user", email="dev@example.com", is_dev_user=True)
        user_id_ctx.set(user.id)
        logger.debug("auth.dev_bypass", user_id=user.id)
        return user

    if token is None:
        raise AuthError("Missing Bearer token.")

    user = await _verify_clerk_token(token)
    user_id_ctx.set(user.id)
    return user


async def optional_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthenticatedUser | None:
    """Optional-auth dependency for public endpoints that adapt to auth state."""
    if authorization is None:
        return None
    try:
        return await current_user(authorization=authorization)
    except AuthError:
        return None


CurrentUser = Annotated[AuthenticatedUser, Depends(current_user)]
OptionalUser = Annotated[AuthenticatedUser | None, Depends(optional_user)]


async def fetch_jwks_eagerly() -> None:
    """Optional lifespan hook — pre-warm the JWKS cache so the first request is fast."""
    settings = get_settings()
    if settings.clerk_jwks_url is None:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(settings.clerk_jwks_url)
            r.raise_for_status()
        _jwks_cache.get(settings.clerk_jwks_url)
        logger.info("auth.jwks.prewarmed")
    except httpx.HTTPError as e:
        logger.warning("auth.jwks.prewarm_failed", error=str(e))
