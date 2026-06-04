# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# FundIQ backend image — multi-stage build with uv.
# Stage 1 (builder): install uv, resolve and install deps into a venv.
# Stage 2 (runtime): copy the venv + source into a slim base.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12

# ===========================================================================
# Stage 1 — builder
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# uv ships as a single static binary.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# Build deps for native wheels (asyncpg, psycopg, torch CPU wheels, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: lockfile + project metadata only — cached unless deps change.
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Layer 2: project source.
COPY backend ./backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===========================================================================
# Stage 2 — runtime
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/backend"

# Runtime deps only — no compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 fundiq \
    && useradd --system --uid 1000 --gid fundiq --no-create-home fundiq

WORKDIR /app

COPY --from=builder --chown=fundiq:fundiq /opt/venv /opt/venv
COPY --from=builder --chown=fundiq:fundiq /app/backend /app/backend

USER fundiq

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
