"""Tests for the latency metrics + /metrics endpoint (item 23)."""

from __future__ import annotations

import pytest

from metering.metrics import LatencyStats


def test_empty_snapshot_is_zeroed() -> None:
    snap = LatencyStats().snapshot()
    assert snap == {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}


def test_records_count_mean_and_max() -> None:
    stats = LatencyStats()
    for ms in (10.0, 20.0, 30.0):
        stats.record(ms)
    snap = stats.snapshot()
    assert snap["count"] == 3
    assert snap["mean_ms"] == 20.0
    assert snap["max_ms"] == 30.0


def test_negative_samples_ignored() -> None:
    stats = LatencyStats()
    stats.record(-5.0)
    assert stats.snapshot()["count"] == 0


def test_percentiles_are_monotonic() -> None:
    stats = LatencyStats()
    for ms in range(1, 101):
        stats.record(float(ms))
    snap = stats.snapshot()
    assert snap["p50_ms"] <= snap["p95_ms"] <= snap["max_ms"]
    assert snap["max_ms"] == 100.0


def test_single_sample_percentile() -> None:
    stats = LatencyStats()
    stats.record(42.0)
    snap = stats.snapshot()
    assert snap["p50_ms"] == 42.0
    assert snap["p95_ms"] == 42.0


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_usage_and_latency() -> None:
    from httpx import ASGITransport, AsyncClient

    from config.settings import settings
    from main import app
    from metering.metrics import latency_stats
    from metering.usage import usage_store

    usage_store.reset_all()
    latency_stats.reset()
    usage_store.add("acme", 100)
    usage_store.inc_request("acme")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics", headers={"X-Internal-Secret": settings.INTERNAL_SECRET})

    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"]["total_tokens"] == 100
    assert body["usage"]["total_requests"] == 1
    assert body["usage"]["per_org"]["acme"]["tokens"] == 100
    assert "latency" in body
    assert "p95_ms" in body["latency"]


@pytest.mark.asyncio
async def test_metrics_endpoint_requires_auth() -> None:
    from httpx import ASGITransport, AsyncClient

    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 401
