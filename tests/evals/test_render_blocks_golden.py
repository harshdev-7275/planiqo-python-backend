"""Golden fixtures for the render contract — the screenshot format-probes
(ranking table, fenced code, 3-section health report, action proposal, mixed
report, entity card) pinned as prompt-shaped payloads, plus the two grounding
failures that MUST drop (row/column mismatch, unsourced quote).

Each case is a raw block payload run through the boundary validator; the test
asserts the expected blocks survive (or the bad ones fall back to prose).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chains.render_contract import validate_blocks

_GOLDEN = Path(__file__).parent / "render_blocks_golden.jsonl"


def _cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in _GOLDEN.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_golden_block_render(case: dict[str, Any]) -> None:
    result = validate_blocks(case["raw"])
    if case.get("expect_fallback"):
        assert result.used_fallback is True
        assert result.dropped == case["expect_dropped"]
    else:
        assert result.used_fallback is False
        assert [block.type for block in result.blocks] == case["expect_types"]
