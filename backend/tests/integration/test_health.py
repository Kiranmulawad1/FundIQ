"""Integration tests for /health, /ready, and request-id middleware."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.integration
async def test_health_returns_ok(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["version"]


@pytest.mark.integration
async def test_ready_pings_postgres_and_redis(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["redis"]["status"] == "ok"


@pytest.mark.integration
async def test_request_id_echoed_when_provided(client: AsyncClient) -> None:
    inbound = "client-supplied-rid-7777"
    r = await client.get("/health", headers={"X-Request-ID": inbound})
    assert r.headers["x-request-id"] == inbound


@pytest.mark.integration
async def test_request_id_generated_when_absent(client: AsyncClient) -> None:
    r = await client.get("/health")
    rid = r.headers.get("x-request-id")
    assert rid
    assert "-" in rid  # `<timestamp>-<uuid>` shape


@pytest.mark.integration
async def test_admin_costs_stub(client: AsyncClient) -> None:
    r = await client.get("/admin/costs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "stub"
    assert body["total_usd"] == 0.0


@pytest.mark.integration
async def test_validation_error_envelope(client: AsyncClient) -> None:
    """Hit a non-existent route — exception handler shape sanity check."""
    r = await client.get("/this-route-does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert "code" in body
    assert "request_id" in body
