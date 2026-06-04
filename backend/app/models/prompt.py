"""PromptVersion — semver-tracked prompt registry.

Schema notes:
  - Prompts live as files in `backend/prompts/` for editing/PR review.
  - This table is the *runtime* registry: the file gets hashed + loaded
    once on startup, and every generated output records which version
    produced it (FK from `eval_results.prompt_version`, etc.).
  - Unique on (name, version) — prevents accidental overwrites.
"""

from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class PromptVersion(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_name_version"),)

    name: str = Field(max_length=128, index=True, description="e.g. 'researcher.system'")
    version: str = Field(max_length=32, description="Semver, e.g. '1.4.0'")
    content: str = Field(description="Full prompt text.")
    content_hash: str = Field(max_length=64, index=True, description="sha256 of content.")
    description: str | None = None
    is_active: bool = Field(default=True, index=True)
