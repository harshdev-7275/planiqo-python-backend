"""Unit tests for the boundary validator + emission router.

Pure functions — no LLM, no I/O. The validator must never raise and must
always return at least one block; the router decision must be deterministic.
"""

from __future__ import annotations

from chains.render_contract import (
    attach_render_blocks,
    ensure_wire_blocks,
    needs_presentation_pass,
    serialize_blocks,
    validate_blocks,
)
from models.blocks import ActionProposalBlock, ProseBlock


# --- validate_blocks: parse / validate / repair / fallback -------------------


def test_keeps_valid_block_and_reports_no_drops() -> None:
    res = validate_blocks([{"type": "prose", "markdown": "hi"}])
    assert len(res.blocks) == 1
    assert res.dropped == 0
    assert res.used_fallback is False


def test_drops_invalid_block_but_keeps_valid_one() -> None:
    res = validate_blocks(
        [
            {"type": "prose", "markdown": "ok"},
            {"type": "code"},  # missing required language + code
        ]
    )
    assert res.dropped == 1
    assert len(res.blocks) == 1
    assert isinstance(res.blocks[0], ProseBlock)


def test_falls_back_to_prose_when_all_invalid() -> None:
    res = validate_blocks([{"type": "nope"}], fallback_markdown="plain answer")
    assert res.used_fallback is True
    assert isinstance(res.blocks[0], ProseBlock)
    assert res.blocks[0].markdown == "plain answer"


def test_fallback_uses_placeholder_when_no_markdown_given() -> None:
    res = validate_blocks("garbage")
    assert res.used_fallback is True
    assert res.blocks[0].markdown == "(no content)"


def test_accepts_envelope_shape() -> None:
    res = validate_blocks({"version": "v1", "blocks": [{"type": "prose", "markdown": "hi"}]})
    assert res.dropped == 0
    assert len(res.blocks) == 1


def test_accepts_single_block_dict() -> None:
    res = validate_blocks({"type": "prose", "markdown": "hi"})
    assert res.used_fallback is False
    assert len(res.blocks) == 1


# --- needs_presentation_pass: deterministic router ---------------------------


def test_router_ships_simple_inline_result() -> None:
    res = validate_blocks([{"type": "prose", "markdown": "hi"}])
    assert needs_presentation_pass(res) is False


def test_router_escalates_on_composed_block() -> None:
    res = validate_blocks(
        [{"type": "breakdown", "dimension": "status", "segments": [{"label": "todo", "value": 3}]}]
    )
    assert needs_presentation_pass(res) is True


def test_router_escalates_over_inline_budget() -> None:
    res = validate_blocks([{"type": "prose", "markdown": str(i)} for i in range(3)])
    assert needs_presentation_pass(res) is True


def test_router_escalates_when_a_block_was_dropped() -> None:
    res = validate_blocks([{"type": "prose", "markdown": "ok"}, {"type": "code"}])
    assert needs_presentation_pass(res) is True


# --- camelCase serialization for the wire ------------------------------------


def test_serialize_blocks_emits_camelcase() -> None:
    out = serialize_blocks(
        [ActionProposalBlock(action="create", entity="issue", summary="s", executor_ref="x")]
    )
    assert out[0]["confirmRequired"] is True
    assert out[0]["executorRef"] == "x"
    assert "confirm_required" not in out[0]


# --- ensure_wire_blocks ------------------------------------------------------


def test_ensure_wire_blocks_falls_back_to_prose() -> None:
    assert ensure_wire_blocks(message="hello", raw=None) == [
        {"type": "prose", "markdown": "hello"}
    ]


def test_ensure_wire_blocks_validates_and_drops_bad_blocks() -> None:
    out = ensure_wire_blocks(
        message="m", raw=[{"type": "prose", "markdown": "x"}, {"type": "code"}]
    )
    assert len(out) == 1
    assert out[0]["type"] == "prose"


# --- attach_render_blocks (service edge) -------------------------------------


def test_attach_render_blocks_adds_prose_for_plain_message() -> None:
    resp = {"intent": "QUERY_ISSUES", "result": {"message": "hi"}, "status": "executed"}
    out = attach_render_blocks(resp)
    assert out["result"]["blocks"] == [{"type": "prose", "markdown": "hi"}]


def test_attach_render_blocks_validates_producer_blocks() -> None:
    resp = {
        "result": {
            "message": "m",
            "blocks": [{"type": "progress", "kind": "sprint", "completed": 1, "total": 2}],
        }
    }
    out = attach_render_blocks(resp)
    blocks = out["result"]["blocks"]
    assert blocks[0]["type"] == "progress"
    assert blocks[0]["completed"] == 1


def test_attach_render_blocks_ignores_response_without_result_message() -> None:
    resp = {"message": "top-level-only"}
    out = attach_render_blocks(resp)
    assert "blocks" not in out
