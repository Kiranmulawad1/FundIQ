"""Pure model tests — no DB. Verifies SQLModel construction and enum surface."""

from __future__ import annotations

import uuid

import pytest

from app.models import EMBEDDING_DIM, Grant, Startup
from app.models.base import GrantPortal, GrantStatus, Sector, StartupStage


@pytest.mark.unit
def test_startup_minimal_construction() -> None:
    s = Startup(
        owner_user_id="user_abc",
        name="Acme",
        sector=Sector.DEEPTECH,
        stage=StartupStage.PRE_SEED,
    )
    assert s.country == "DE"
    assert s.profile == {}
    assert s.frs_scores == {}
    assert isinstance(s.id, uuid.UUID)


@pytest.mark.unit
def test_grant_minimal_construction() -> None:
    g = Grant(
        title="EXIST-Gründerstipendium",
        summary="Stipend for academic founders.",
        body="Long body...",
        portal=GrantPortal.EXIST,
        status=GrantStatus.OPEN,
        source_url="https://www.exist.de/gruenderstipendium",
    )
    assert g.title.startswith("EXIST")
    assert g.country == "DE"
    assert g.embedding is None  # populated by RAG pipeline later


@pytest.mark.unit
def test_embedding_dim_matches_e5_large() -> None:
    """multilingual-e5-large is 1024-dim. Drift here = silent retrieval bug."""
    assert EMBEDDING_DIM == 1024


@pytest.mark.unit
@pytest.mark.parametrize(
    "stage",
    [StartupStage.IDEA, StartupStage.PRE_SEED, StartupStage.SEED, StartupStage.GROWTH],
)
def test_startup_stage_string_values_stable(stage: StartupStage) -> None:
    """Enum string values are part of the public API — they appear in JSON
    responses, gold-set jsonl, and the React frontend. Don't rename casually."""
    assert isinstance(stage.value, str)
    assert stage.value == stage.value.lower()


@pytest.mark.unit
def test_all_grant_portals_present() -> None:
    """8 portals per master plan: BMBF, EXIST, KfW, EIC, Horizon, Bayern, NRW, BW."""
    expected = {"bmbf", "exist", "kfw", "eic", "horizon", "bayern", "nrw", "bw"}
    assert {p.value for p in GrantPortal} == expected
