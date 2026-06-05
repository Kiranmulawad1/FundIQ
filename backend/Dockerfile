# syntax=docker/dockerfile:1.7
#
# FundIQ backend — production image. Build context is the REPO ROOT
# (see render.yaml `rootDir: ./`); paths below are relative to that.
#
# Two-stage build keeps the runtime image at ~250MB. The old
# sentence-transformers + torch path was ~2.5GB and didn't fit on any
# free-tier host. Now all embedding + rerank work is hosted (Gemini +
# Cohere), so the backend is just a thin Python layer.
#
# Layout note:
#   - The Python package is built as a wheel from `backend/app/**` (see
#     pyproject.toml `tool.hatch.build.targets.wheel`) and installed
#     into /opt/venv, so the runtime stage doesn't ship the source tree
#     for `app/` itself. It DOES ship `backend/alembic*` because Alembic
#     reads files from the working directory.

# ---------------------------------------------------------------------------
# Stage 1 — build the venv with uv. Pin uv to a known-good major.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# Build essentials for any wheel that needs to compile (asyncpg, psycopg).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential libpq-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv ships as a single static binary in the GHCR image — no Python
# bootstrap step needed.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Copy lockfile + manifest first so dep installs cache independently of
# app source — only re-runs when pyproject.toml or uv.lock changes.
# README.md travels with pyproject.toml because pyproject declares
# `readme = "README.md"` and hatchling reads it during the wheel build.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project as a wheel.
COPY backend ./backend
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — slim runtime. No build tools.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Render injects PORT at runtime; fall back to 8000 locally.
    PORT=8000 \
    ENVIRONMENT=production

# Run from the backend/ directory so `alembic upgrade head` finds
# alembic.ini + alembic/ without extra flags.
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Alembic config + migrations only — the `app` package itself is
# already in /opt/venv as an installed wheel.
COPY --from=builder /app/backend/alembic.ini ./alembic.ini
COPY --from=builder /app/backend/alembic ./alembic

RUN useradd --create-home --shell /usr/sbin/nologin fundiq \
 && chown -R fundiq:fundiq /app
USER fundiq

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request; \
        sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=3).status==200 else 1)"

EXPOSE 8000

# Run alembic migrations on every boot so a fresh Render deploy or a
# manual `render restart` brings schema up to date without an extra
# console step. Idempotent — head-equal runs are sub-second.
#
# Single uvicorn worker fits a 512MB free-tier host; scale `--workers $N`
# once on paid plans.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
