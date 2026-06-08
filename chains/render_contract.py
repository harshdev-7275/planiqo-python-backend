"""Boundary validator + emission router for the output render contract.

This is the "wrapper" the read side passes its candidate blocks through
before they reach the frontend. It does four things, in order, and NEVER
raises:

  1. parse    — coerce raw LLM output into a list of block-shaped dicts
  2. validate — validate each candidate against its block schema
  3. repair   — drop candidates that fail; keep the valid ones
  4. fallback — if nothing valid survives, return a single ``prose`` block
                so the UI always has something to render and never breaks

It also exposes the deterministic emission router. ``needs_presentation_pass``
decides, by rule only (never an LLM call, never the model's mood), whether
the cheap single-pass inline result is good enough or the answer must be
re-rendered by the more expensive presentation pass.

The router is consulted once, on the inline attempt. The presentation pass
output is terminal (validated, then rendered) and is not re-routed, so there
is no escalation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import TypeAdapter, ValidationError

from models.blocks import Block, ProseBlock, tier_of

# One adapter for the discriminated union — built once, reused per call.
_BLOCK_ADAPTER: TypeAdapter[Block] = TypeAdapter(Block)

# A response with more than this many blocks, or that lost any block to
# validation, is re-rendered by the presentation pass instead of shipped.
_MAX_INLINE_BLOCKS = 2


@dataclass(frozen=True)
class RenderResult:
    """The outcome of validating one batch of candidate blocks."""

    blocks: list[Block]  # always non-empty (prose fallback guarantees it)
    dropped: int  # candidates that failed validation
    used_fallback: bool  # True when ``blocks`` is the single prose fallback


def _coerce_candidates(raw: Any) -> list[dict[str, Any]]:
    """Pull a list of block-shaped dicts out of whatever the LLM returned.

    Accepts a ``RenderResponse``-shaped dict (``{"blocks": [...]}``), a bare
    list of blocks, or a single block dict. Anything else yields nothing.
    """
    if isinstance(raw, dict):
        inner = raw.get("blocks")
        if isinstance(inner, list):
            return [b for b in inner if isinstance(b, dict)]
        if "type" in raw:
            return [raw]
        return []
    if isinstance(raw, list):
        return [b for b in raw if isinstance(b, dict)]
    return []


def validate_blocks(raw: Any, *, fallback_markdown: str = "") -> RenderResult:
    """Validate raw LLM output into typed blocks. Never raises.

    Invalid candidates are dropped (and counted). If nothing valid survives,
    the result is a single ``prose`` block carrying ``fallback_markdown`` so
    the UI always renders something.
    """
    candidates = _coerce_candidates(raw)
    valid: list[Block] = []
    dropped = 0
    for cand in candidates:
        try:
            valid.append(_BLOCK_ADAPTER.validate_python(cand))
        except ValidationError as exc:
            dropped += 1
            logger.debug("render_contract: dropped {} block: {}", cand.get("type"), exc)

    if not valid:
        text = fallback_markdown.strip() or "(no content)"
        return RenderResult(blocks=[ProseBlock(markdown=text)], dropped=dropped, used_fallback=True)
    return RenderResult(blocks=valid, dropped=dropped, used_fallback=False)


def needs_presentation_pass(result: RenderResult) -> bool:
    """Deterministic emission router. Escalate to the presentation pass when:

      1. validation was lossy (fell back, or dropped a candidate),
      2. there are more than ``_MAX_INLINE_BLOCKS`` blocks, or
      3. any surviving block is composed-tier.

    Otherwise the cheap single-pass inline result ships as-is.
    """
    if result.used_fallback or result.dropped > 0:
        return True
    if len(result.blocks) > _MAX_INLINE_BLOCKS:
        return True
    return any(tier_of(block) == "composed" for block in result.blocks)


def serialize_blocks(blocks: list[Block]) -> list[dict[str, Any]]:
    """Dump validated blocks to camelCase wire dicts the frontend expects."""
    return [block.model_dump(by_alias=True) for block in blocks]


def ensure_wire_blocks(*, message: str, raw: Any = None) -> list[dict[str, Any]]:
    """Validate any producer-supplied blocks and return camelCase wire dicts.

    Never empty: when no valid blocks survive, a single ``prose`` block built
    from ``message`` is returned, so the frontend always has something to
    render. Never raises.
    """
    result = validate_blocks(raw, fallback_markdown=message)
    return serialize_blocks(result.blocks)


def attach_render_blocks(response: dict[str, Any]) -> dict[str, Any]:
    """Service-edge wrapper: ensure the chat response carries render blocks.

    Single choke point on the ``/chat`` reply. Every result with a text
    message gets at least a prose block; any blocks a producer attached (e.g.
    the summarize agent) are validated + serialized to camelCase here. Mutates
    ``response`` in place and returns it. Defensive — a failure leaves the
    message untouched (no blocks) rather than breaking the reply.
    """
    result = response.get("result")
    if isinstance(result, dict) and isinstance(result.get("message"), str):
        try:
            result["blocks"] = ensure_wire_blocks(
                message=result["message"], raw=result.get("blocks")
            )
        except Exception as exc:  # pragma: no cover - ensure_wire_blocks shouldn't raise
            logger.warning("attach_render_blocks failed; sending message without blocks: {}", exc)
            result.pop("blocks", None)
    return response


__all__ = [
    "RenderResult",
    "attach_render_blocks",
    "ensure_wire_blocks",
    "needs_presentation_pass",
    "serialize_blocks",
    "validate_blocks",
]
