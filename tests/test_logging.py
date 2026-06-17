"""Tests for the structured logging configuration."""

from __future__ import annotations

import io
import json
import logging

from ai_service.logging import (
    bind_request_id,
    configure_logging,
    get_logger,
    get_request_id,
    new_request_id,
    reset_request_id,
)


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


class TestRequestId:
    def test_bind_get_reset_roundtrip(self) -> None:
        assert get_request_id() is None
        token = bind_request_id("abc123")
        try:
            assert get_request_id() == "abc123"
        finally:
            reset_request_id(token)
        assert get_request_id() is None

    def test_new_request_id_is_unique_hex(self) -> None:
        a, b = new_request_id(), new_request_id()
        assert a != b
        assert len(a) == 32 and a.isalnum()


class TestContextAndStaticFields:
    def _capture(self) -> tuple[io.StringIO, logging.Logger]:
        configure_logging(
            level="INFO", json_output=True, service="ai-service", version="9.9.9", environment="test"
        )
        logger = get_logger("test.ctx")
        return io.StringIO(), logger

    def test_request_id_and_static_fields_appear_in_json(self) -> None:
        configure_logging(
            level="INFO", json_output=True, service="ai-service", version="9.9.9", environment="test"
        )
        root = logging.getLogger()
        # Mirror the configured handler onto a capturable stream (same filter).
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        from pythonjsonlogger import jsonlogger

        handler.setFormatter(jsonlogger.JsonFormatter("%(message)s"))  # type: ignore[no-untyped-call]
        # Reuse the context filter the real handler installed.
        for f in root.handlers[0].filters:
            handler.addFilter(f)
        logger = get_logger("test.ctx")
        logger.addHandler(handler)
        logger.propagate = False
        token = bind_request_id("req-xyz")
        try:
            logger.info("hello")
            parsed = json.loads(stream.getvalue().strip())
            assert parsed["request_id"] == "req-xyz"
            assert parsed["service"] == "ai-service"
            assert parsed["version"] == "9.9.9"
            assert parsed["env"] == "test"
        finally:
            reset_request_id(token)
            logger.removeHandler(handler)
            logger.propagate = True

    def test_no_request_id_field_when_unbound(self) -> None:
        configure_logging(level="INFO", json_output=True, service="ai-service")
        root = logging.getLogger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        from pythonjsonlogger import jsonlogger

        handler.setFormatter(jsonlogger.JsonFormatter("%(message)s"))  # type: ignore[no-untyped-call]
        for f in root.handlers[0].filters:
            handler.addFilter(f)
        logger = get_logger("test.ctx2")
        logger.addHandler(handler)
        logger.propagate = False
        try:
            logger.info("hello")
            parsed = json.loads(stream.getvalue().strip())
            assert "request_id" not in parsed
        finally:
            logger.removeHandler(handler)
            logger.propagate = True
