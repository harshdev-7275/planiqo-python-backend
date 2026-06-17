"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.config import Settings, get_settings
from ai_service.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Default settings for tests — clear the cache so env-var changes are visible."""
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """FastAPI app instance with test-friendly settings."""
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """HTTP test client wrapping the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client
