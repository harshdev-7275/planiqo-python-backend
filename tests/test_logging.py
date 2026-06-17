"""Tests for the structured logging configuration."""

from __future__ import annotations

import io
import json
import logging

from ai_service.logging import configure_logging, get_logger


class TestConfigureLogging:
    def test_configures_root_logger(self) -> None:
        configure_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1

    def test_idempotent(self) -> None:
        """Calling twice must not stack handlers."""
        configure_logging(level="INFO", json_output=False)
        first_count = len(logging.getLogger().handlers)
        configure_logging(level="INFO", json_output=False)
        second_count = len(logging.getLogger().handlers)
        assert first_count == second_count

    def test_json_output_emits_parseable_json(self) -> None:
        configure_logging(level="INFO", json_output=True)
        logger = get_logger("test.json")
        # Capture log output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        # Replace the existing formatter with a JSON formatter to capture.
        from pythonjsonlogger import jsonlogger

        handler.setFormatter(jsonlogger.JsonFormatter("%(message)s"))  # type: ignore[no-untyped-call]
        logger.addHandler(handler)
        logger.propagate = False
        try:
            logger.info("hello world")
            output = stream.getvalue().strip()
            parsed = json.loads(output)
            assert parsed["message"] == "hello world"
        finally:
            logger.removeHandler(handler)
            logger.propagate = True

    def test_quiet_loggers_are_silenced(self) -> None:
        configure_logging(level="INFO", json_output=False)
        for name in ("uvicorn.access", "httpx", "httpcore"):
            assert logging.getLogger(name).level == logging.WARNING


class TestGetLogger:
    def test_returns_named_logger(self) -> None:
        logger = get_logger("my.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my.module"
