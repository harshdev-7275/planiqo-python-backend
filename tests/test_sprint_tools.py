"""Tests for sprint agent tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.agents.tools.sprints import make_sprint_tools

SCOPED = "11111111-1111-1111-1111-111111111111"


class TestUnscopedSprintTools:
    def test_returns_one_tool(self) -> None:
        client = MagicMock()
        tools = make_sprint_tools(client, "acme")
        assert len(tools) == 1

    def test_tool_name_is_list_sprints(self) -> None:
        client = MagicMock()
        tools = make_sprint_tools(client, "acme")
        assert tools[0].name == "list_sprints"

    def test_has_project_id_param(self) -> None:
        client = MagicMock()
        tools = make_sprint_tools(client, "acme")
        assert "project_id" in tools[0].args

    @pytest.mark.asyncio
    async def test_calls_backend(self) -> None:
        expected = [{"id": "sp-1", "name": "Sprint 1", "status": "active"}]
        client = MagicMock()
        client.list_sprints = AsyncMock(return_value=expected)
        tools = make_sprint_tools(client, "acme")
        result = await tools[0].ainvoke({"project_id": SCOPED})
        assert result == expected
        client.list_sprints.assert_awaited_once()


class TestScopedSprintTools:
    def test_no_project_id_param(self) -> None:
        client = MagicMock()
        tools = make_sprint_tools(client, "acme", scoped_project_id=SCOPED)
        assert "project_id" not in tools[0].args

    @pytest.mark.asyncio
    async def test_always_uses_scoped_project(self) -> None:
        client = MagicMock()
        client.list_sprints = AsyncMock(return_value=[])
        tools = make_sprint_tools(client, "acme", scoped_project_id=SCOPED)
        await tools[0].ainvoke({})
        called_project = client.list_sprints.call_args.args[1]
        assert str(called_project) == SCOPED
