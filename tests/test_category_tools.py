"""Tests for category agent tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.agents.tools.categories import make_category_tools

SCOPED = "11111111-1111-1111-1111-111111111111"


def _tool_by_name(tools: list[Any], name: str) -> Any:
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


class TestUnscopedCategoryTools:
    def test_returns_one_tool(self) -> None:
        client = MagicMock()
        tools = make_category_tools(client, "acme")
        assert len(tools) == 1

    def test_tool_name_is_list_categories(self) -> None:
        client = MagicMock()
        tools = make_category_tools(client, "acme")
        assert tools[0].name == "list_categories"

    def test_has_project_id_param(self) -> None:
        client = MagicMock()
        tools = make_category_tools(client, "acme")
        assert "project_id" in tools[0].args

    @pytest.mark.asyncio
    async def test_calls_backend(self) -> None:
        expected = [{"id": "cat-1", "name": "Backend", "color": "#6366f1"}]
        client = MagicMock()
        client.list_categories = AsyncMock(return_value=expected)
        tools = make_category_tools(client, "acme")
        result = await tools[0].ainvoke({"project_id": SCOPED})
        assert result == expected
        client.list_categories.assert_awaited_once()


class TestScopedCategoryTools:
    def test_no_project_id_param(self) -> None:
        client = MagicMock()
        tools = make_category_tools(client, "acme", scoped_project_id=SCOPED)
        assert "project_id" not in tools[0].args

    @pytest.mark.asyncio
    async def test_always_uses_scoped_project(self) -> None:
        client = MagicMock()
        client.list_categories = AsyncMock(return_value=[])
        tools = make_category_tools(client, "acme", scoped_project_id=SCOPED)
        await tools[0].ainvoke({})
        called_project = client.list_categories.call_args.args[1]
        assert str(called_project) == SCOPED
