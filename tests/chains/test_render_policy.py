"""The render policy must stay in lockstep with the block contract."""

from __future__ import annotations

from chains.render_policy import BLOCK_TYPES, RENDER_POLICY_PROMPT
from models.blocks import (
    ActionProposalBlock,
    BreakdownBlock,
    CodeBlock,
    EntityCardBlock,
    HealthReportBlock,
    ProgressBlock,
    ProseBlock,
    RankingTableBlock,
)

_MODELS = [
    ProseBlock,
    CodeBlock,
    EntityCardBlock,
    RankingTableBlock,
    ActionProposalBlock,
    BreakdownBlock,
    ProgressBlock,
    HealthReportBlock,
]


def test_block_types_match_the_contract() -> None:
    actual = {m.model_fields["type"].default for m in _MODELS}
    assert set(BLOCK_TYPES) == actual


def test_policy_describes_every_block_type() -> None:
    for block_type in BLOCK_TYPES:
        assert block_type in RENDER_POLICY_PROMPT


def test_policy_states_prose_by_default_guardrail() -> None:
    lowered = RENDER_POLICY_PROMPT.lower()
    assert "default" in lowered
    assert "prose" in lowered


def test_policy_requires_evidence_for_quotes() -> None:
    assert "evidence" in RENDER_POLICY_PROMPT.lower()
