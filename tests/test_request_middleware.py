"""Tests for RequestLoggingMiddleware — correlation ids + access logs."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.core.middleware import RequestLoggingMiddleware
from ai_service.logging import get_request_id


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str | None]:
        # The request id must be visible inside the handler (contextvar bound).
        return {"seen_request_id": get_request_id()}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("kaboom")

    return app


class TestRequestLoggingMiddleware:
    def test_generates_and_echoes_request_id(self) -> None:
        with TestClient(_build_app()) as client:
            res = client.get("/ping")
        assert res.status_code == 200
        rid = res.headers.get("X-Request-ID")
        assert rid and len(rid) == 32
        # The same id the handler saw is echoed back in the header.
        assert res.json()["seen_request_id"] == rid

    def test_honours_inbound_request_id(self) -> None:
        with TestClient(_build_app()) as client:
            res = client.get("/ping", headers={"X-Request-ID": "caller-supplied-id"})
        assert res.headers["X-Request-ID"] == "caller-supplied-id"
        assert res.json()["seen_request_id"] == "caller-supplied-id"

    def test_logs_request_completed_with_status_and_latency(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="ai_service.request"), TestClient(_build_app()) as client:
            client.get("/ping")
        completed = [r for r in caplog.records if r.getMessage() == "request.completed"]
        assert len(completed) == 1
        rec = completed[0]
        assert rec.status_code == 200  # type: ignore[attr-defined]
        assert rec.path == "/ping"  # type: ignore[attr-defined]
        assert isinstance(rec.duration_ms, float)  # type: ignore[attr-defined]

    def test_request_id_is_reset_after_request(self) -> None:
        with TestClient(_build_app()) as client:
            client.get("/ping")
        # No id should linger in this (test) context once the request is done.
        assert get_request_id() is None

    def test_logs_request_failed_on_unhandled_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level("INFO", logger="ai_service.request"),
            TestClient(_build_app()) as client,
            pytest.raises(RuntimeError),
        ):
            client.get("/boom")
        failed = [r for r in caplog.records if r.getMessage() == "request.failed"]
        assert len(failed) == 1
        assert failed[0].path == "/boom"  # type: ignore[attr-defined]
