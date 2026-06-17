"""Deployment entrypoint shim.

This project uses a ``src`` layout; the FastAPI app lives at
``ai_service.main:app`` (see ``src/ai_service/main.py``). Some hosts default
their start command to ``uvicorn main:app`` (a top-level ``main`` module).

This module re-exports ``app`` so that command works without extra config.
Where you control the start command, prefer ``ai_service.main:app`` directly.
"""

from __future__ import annotations

from ai_service.main import app

__all__ = ["app"]
