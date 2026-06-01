import pytest
from unittest.mock import AsyncMock

from graph.schema import CONSTRAINTS, NodeLabel, RelType, apply_constraints


def test_node_labels_defined():
    assert NodeLabel.USER    == "User"
    assert NodeLabel.ISSUE   == "Issue"
    assert NodeLabel.PROJECT == "Project"
    assert NodeLabel.SPRINT  == "Sprint"
    assert NodeLabel.STATUS  == "Status"
    assert NodeLabel.LABEL   == "Label"


def test_rel_types_defined():
    assert RelType.MEMBER_OF    == "MEMBER_OF"
    assert RelType.ASSIGNED_TO  == "ASSIGNED_TO"
    assert RelType.REPORTED     == "REPORTED"
    assert RelType.COMMENTED_ON == "COMMENTED_ON"
    assert RelType.CHANGED      == "CHANGED"
    assert RelType.IN_PROJECT   == "IN_PROJECT"
    assert RelType.IN_SPRINT    == "IN_SPRINT"
    assert RelType.HAS_STATUS   == "HAS_STATUS"
    assert RelType.BLOCKS       == "BLOCKS"


def test_constraints_list_not_empty():
    assert len(CONSTRAINTS) > 0


def test_constraints_are_valid_cypher_strings():
    for stmt in CONSTRAINTS:
        assert isinstance(stmt, str)
        assert "CREATE CONSTRAINT" in stmt
        assert "IF NOT EXISTS" in stmt
        assert "IS UNIQUE" in stmt


def test_constraints_cover_core_nodes():
    combined = " ".join(CONSTRAINTS)
    assert "User" in combined
    assert "Issue" in combined
    assert "Project" in combined
    assert "Sprint" in combined


@pytest.mark.asyncio
async def test_apply_constraints_runs_each_statement():
    mock_client = AsyncMock()
    mock_client.run = AsyncMock(return_value=[])
    await apply_constraints(mock_client)
    assert mock_client.run.call_count == len(CONSTRAINTS)


@pytest.mark.asyncio
async def test_apply_constraints_uses_parameterised_calls():
    mock_client = AsyncMock()
    mock_client.run = AsyncMock(return_value=[])
    await apply_constraints(mock_client)
    for call in mock_client.run.call_args_list:
        args = call[0]
        assert isinstance(args[0], str)
        assert "CREATE CONSTRAINT" in args[0]
