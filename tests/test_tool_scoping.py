"""Tests for project-scoped agent tools (hard restriction).

When a project is selected in the chat UI, the agent's tools are rebuilt in a
scoped variant: the issue/project tools can ONLY touch the selected project,
and the org-wide graph tools are filtered to that project. This is enforcement
at the tool layer — the LLM cannot reach another project even if it tries.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.agents.tools import build_all_tools
from ai_service.agents.tools.issues import make_issue_tools
from ai_service.agents.tools.projects import make_project_tools

SCOPED = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


def _tool_by_name(tools: list[Any], name: str) -> Any:
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


class TestScopedIssueTools:
    @pytest.mark.asyncio
    async def test_list_issues_has_no_project_id_param_when_scoped(self) -> None:
        client = MagicMock()
        client.list_issues = AsyncMock(return_value=[])
        tools = make_issue_tools(client, "acme", scoped_project_id=SCOPED)
        list_issues = _tool_by_name(tools, "list_issues")
        # project_id is bound, not a parameter the LLM can set.
        assert "project_id" not in list_issues.args

    @pytest.mark.asyncio
    async def test_list_issues_always_uses_scoped_project(self) -> None:
        client = MagicMock()
        client.list_issues = AsyncMock(return_value=[])
        tools = make_issue_tools(client, "acme", scoped_project_id=SCOPED)
        list_issues = _tool_by_name(tools, "list_issues")
        await list_issues.ainvoke({"limit": 5})
        called_project = client.list_issues.call_args.args[1]
        assert str(called_project) == SCOPED

    @pytest.mark.asyncio
    async def test_get_issue_uses_scoped_project(self) -> None:
        client = MagicMock()
        client.get_issue = AsyncMock(return_value=None)
        tools = make_issue_tools(client, "acme", scoped_project_id=SCOPED)
        get_issue = _tool_by_name(tools, "get_issue")
        await get_issue.ainvoke({"issue_id": "33333333-3333-3333-3333-333333333333"})
        called_project = client.get_issue.call_args.args[1]
        assert str(called_project) == SCOPED

    def test_unscoped_keeps_project_id_param(self) -> None:
        client = MagicMock()
        tools = make_issue_tools(client, "acme")
        list_issues = _tool_by_name(tools, "list_issues")
        assert "project_id" in list_issues.args


class TestScopedProjectTools:
    @pytest.mark.asyncio
    async def test_list_projects_returns_only_scoped_project(self) -> None:
        client = MagicMock()
        client.get_project = AsyncMock(return_value={"id": SCOPED, "name": "Web"})
        tools = make_project_tools(client, "acme", scoped_project_id=SCOPED)
        list_projects = _tool_by_name(tools, "list_projects")
        result = await list_projects.ainvoke({})
        assert result == [{"id": SCOPED, "name": "Web"}]

    @pytest.mark.asyncio
    async def test_list_projects_empty_when_scoped_project_missing(self) -> None:
        client = MagicMock()
        client.get_project = AsyncMock(return_value=None)
        tools = make_project_tools(client, "acme", scoped_project_id=SCOPED)
        list_projects = _tool_by_name(tools, "list_projects")
        assert await list_projects.ainvoke({}) == []

    @pytest.mark.asyncio
    async def test_get_project_ignores_other_id_and_returns_scoped(self) -> None:
        client = MagicMock()
        client.get_project = AsyncMock(return_value={"id": SCOPED})
        tools = make_project_tools(client, "acme", scoped_project_id=SCOPED)
        get_project = _tool_by_name(tools, "get_project")
        await get_project.ainvoke({"project_id": OTHER})
        called_project = client.get_project.call_args.args[1]
        assert str(called_project) == SCOPED


class TestBuildAllToolsScoping:
    def test_passes_scoped_project_to_issue_tools(self) -> None:
        client = MagicMock()
        tools = build_all_tools(client, "acme", scoped_project_id=SCOPED)
        list_issues = _tool_by_name(tools, "list_issues")
        assert "project_id" not in list_issues.args

    def test_unscoped_build_keeps_project_id(self) -> None:
        client = MagicMock()
        tools = build_all_tools(client, "acme")
        list_issues = _tool_by_name(tools, "list_issues")
        assert "project_id" in list_issues.args
