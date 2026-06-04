"""SQLModel table registry.

Importing this package eagerly registers every table with SQLModel's
metadata — Alembic's autogenerate needs them present in the metadata
object at the moment it inspects the DB. Order doesn't matter (FKs
resolved by string).
"""

from __future__ import annotations

from app.models.alert import Alert
from app.models.application import GrantApplication
from app.models.eval import EvalResult
from app.models.feedback import UserFeedback
from app.models.grant import EMBEDDING_DIM, Grant
from app.models.prompt import PromptVersion
from app.models.roadmap import FundingRoadmap
from app.models.scrape_run import ScrapeRun, ScrapeRunStatus, ScrapeRunTrigger
from app.models.session import AgentSession
from app.models.startup import Startup

__all__ = [
    "EMBEDDING_DIM",
    "AgentSession",
    "Alert",
    "EvalResult",
    "FundingRoadmap",
    "Grant",
    "GrantApplication",
    "PromptVersion",
    "ScrapeRun",
    "ScrapeRunStatus",
    "ScrapeRunTrigger",
    "Startup",
    "UserFeedback",
]
