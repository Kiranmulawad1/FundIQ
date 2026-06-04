"""Integration: insert + read a Startup through the transactional session.

Confirms:
  - DB engine + sessionmaker wiring works
  - SQLModel serialization round-trips through Postgres
  - Per-test rollback isolates state (next test sees an empty table)
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Startup
from app.models.base import Sector, StartupStage
from tests.conftest import make_startup


@pytest.mark.integration
async def test_insert_and_read_startup(db_session: AsyncSession) -> None:
    s = Startup(**make_startup(name="Helios Energy", sector=Sector.CLEANTECH))
    db_session.add(s)
    await db_session.flush()

    row = (
        await db_session.execute(select(Startup).where(Startup.name == "Helios Energy"))
    ).scalar_one()
    assert row.id == s.id
    # SQLModel currently round-trips enum columns as their string value,
    # not as the enum instance. Value equality holds because StrEnum
    # compares equal to its string. Enum-instance coercion on load lands
    # with the explicit `sa.Enum(...)` columns in Phase 2.
    assert row.sector == Sector.CLEANTECH
    assert row.stage == StartupStage.PRE_SEED


@pytest.mark.integration
async def test_transaction_rollback_isolates_tests(db_session: AsyncSession) -> None:
    """If rollback works, this test sees zero rows even though the previous
    test inserted one."""
    count = (await db_session.execute(select(func.count()).select_from(Startup))).scalar_one()
    assert count == 0


@pytest.mark.integration
async def test_jsonb_defaults_round_trip(db_session: AsyncSession) -> None:
    s = Startup(**make_startup())
    db_session.add(s)
    await db_session.flush()
    await db_session.refresh(s)
    assert s.profile == {}
    assert s.frs_scores == {}
