"""Unit tests for the output render contract (models/blocks.py).

Pure schema — no LLM, no I/O. These tests pin the two correctness rules
that make the contract trustworthy: the emission tier is an intrinsic
property of the block type, and ``health_report`` computes its own
``overall`` / refuses an unsourced quote rather than trusting the model.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from models.blocks import (
    Block,
    BreakdownBlock,
    BreakdownSegment,
    CodeBlock,
    HealthItem,
    HealthReportBlock,
    HealthSection,
    ProseBlock,
    RankingTableBlock,
    Ref,
    TableColumn,
    TableRow,
    tier_of,
)

_ADAPTER = TypeAdapter(Block)


# --- tier is intrinsic to the type, not LLM-supplied -------------------------


def test_prose_block_is_inline_tier() -> None:
    assert tier_of(ProseBlock(markdown="hi")) == "inline"


def test_breakdown_block_is_composed_tier() -> None:
    block = BreakdownBlock(
        dimension="status",
        segments=[BreakdownSegment(label="todo", value=3)],
    )
    assert tier_of(block) == "composed"


# --- discriminated union routing ---------------------------------------------


def test_union_selects_model_by_type() -> None:
    obj = _ADAPTER.validate_python({"type": "code", "language": "py", "code": "x = 1"})
    assert isinstance(obj, CodeBlock)


def test_union_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"type": "definitely_not_a_block"})


# --- ranking table integrity -------------------------------------------------


def test_ranking_table_rejects_row_column_mismatch() -> None:
    with pytest.raises(ValidationError):
        RankingTableBlock(
            columns=[TableColumn(key="a", label="A")],
            rows=[TableRow(cells=["x", "y"])],
        )


# --- health_report: derive, don't trust -------------------------------------


def test_health_report_overall_derived_overrides_llm_claim() -> None:
    report = HealthReportBlock(
        overall="healthy",  # the model lies; the validator must override it
        sections=[
            HealthSection(
                kind="at_risk",
                heading="Risks",
                items=[HealthItem(text="db is slow", severity="critical")],
            ),
        ],
    )
    assert report.overall == "critical"


def test_health_report_overall_healthy_when_no_at_risk_severity() -> None:
    report = HealthReportBlock(
        overall="critical",
        sections=[
            HealthSection(
                kind="going_well",
                heading="Wins",
                items=[HealthItem(text="velocity up", severity="info")],
            ),
        ],
    )
    assert report.overall == "healthy"


def test_health_item_quote_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HealthItem(text="user unhappy", quote="this tool is terrible")


def test_health_item_quote_with_evidence_is_accepted() -> None:
    item = HealthItem(
        text="user unhappy",
        quote="this tool is terrible",
        evidence=Ref(kind="issue", id="42", label="#42", href="/issues/42"),
    )
    assert item.quote == "this tool is terrible"


# --- camelCase wire format (matches the frontend Zod mirror) -----------------


def test_serializes_camelcase_by_alias() -> None:
    from models.blocks import Assignee, EntityCardBlock

    block = EntityCardBlock(
        entity="issue",
        id="1",
        title="Login bug",
        href="/issues/1",
        assignees=[Assignee(id="u1", name="Alice", avatar_url="http://x/a.png")],
    )
    dumped = block.model_dump(by_alias=True)
    assert dumped["assignees"][0]["avatarUrl"] == "http://x/a.png"
    assert "avatar_url" not in dumped["assignees"][0]


def test_accepts_snake_case_input() -> None:
    from models.blocks import Assignee

    assignee = Assignee(id="u1", name="Alice", avatar_url="http://x/a.png")
    assert assignee.avatar_url == "http://x/a.png"
