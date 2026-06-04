# FundIQ

AI funding intelligence platform for EU and German startups — a decision-intelligence system that tells founders what they qualify for, why they will win or lose, and what to do next.

> **Status:** Phase 1 (Foundation) — monorepo scaffold, FastAPI base, SQLModel schema, Alembic.
> Demo, architecture diagram, and eval results land in Phase 12.

## Quickstart (local dev)

```bash
# 1. Install dependencies
uv sync

# 2. Start dependencies (Postgres + pgvector, Redis, Neo4j)
docker compose -f infra/compose/docker-compose.yml --env-file .env up -d postgres redis neo4j

# 3. Run the initial migration
cd backend && uv run alembic upgrade head

# 4. Start the API
uv run uvicorn app.main:app --reload --app-dir backend
```

Then visit:
- `http://localhost:8000/docs` — OpenAPI UI
- `http://localhost:8000/health` — liveness
- `http://localhost:8000/ready` — readiness (verifies DB + Redis)

## Documentation

- [`fundiq_master_build_prompt.md`](fundiq_master_build_prompt.md) — full project spec
- [`docs/adr/`](docs/adr/) — Architecture Decision Records

## License

Proprietary — iiterate Technologies GmbH internship + M.Sc. thesis project.
