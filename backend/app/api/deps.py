"""Shared FastAPI dependencies — DB session, Redis, current user.

Routes import the `Annotated` aliases below rather than reconstructing
`Depends(...)` everywhere. Keeps signatures short and refactors trivial.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthenticatedUser, current_user, optional_user
from app.core.db import get_session


def get_redis(request: Request) -> Redis:
    return request.app.state.redis  # type: ignore[no-any-return]


SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]
CurrentUserDep = Annotated[AuthenticatedUser, Depends(current_user)]
OptionalUserDep = Annotated[AuthenticatedUser | None, Depends(optional_user)]
