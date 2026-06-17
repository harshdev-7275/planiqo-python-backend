"""Settings/config tests — env parsing and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_service.config import Settings


class TestSettingsDefaults:
    def test_required_fields_have_defaults(self) -> None:
        settings = Settings()
        assert settings.app_name
        assert settings.app_version
        assert settings.host
        assert settings.port > 0
        assert settings.log_level
        assert isinstance(settings.cors_origins, list)


class TestCorsOriginsParsing:
    def test_parses_comma_separated_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "CORS_ORIGINS", "https://a.example, https://b.example ,https://c.example"
        )
        settings = Settings()
        assert settings.cors_origins == [
            "https://a.example",
            "https://b.example",
            "https://c.example",
        ]

    def test_empty_string_yields_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "")
        settings = Settings()
        assert settings.cors_origins == []


class TestEnvironmentValidation:
    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_accepts_known_environments(self, env: str) -> None:
        settings = Settings(app_env=env)
        assert settings.app_env == env

    def test_rejects_unknown_environment(self) -> None:
        with pytest.raises(ValidationError):
            Settings(app_env="qa-test")
