"""Render policy — the system-prompt fragment that teaches the LLM the PM
block vocabulary and when to use each block.

Derived from the SAME catalog as ``models/blocks.py`` and
``chains/render_contract.py``: one source of truth, three consumers. The
policy encodes (a) the closed block vocabulary, (b) which block for which
information pattern, and (c) the prose-by-default guardrail (Claude's own
rule) so a PM tool doesn't turn every answer into card soup.

``tests/chains/test_render_policy.py`` fails if a block type is added to the
contract but not described here — the policy can't silently drift from the
schema.
"""

from __future__ import annotations

# The closed vocabulary, kept in lockstep with models/blocks.py.
BLOCK_TYPES: tuple[str, ...] = (
    "prose",
    "code",
    "entity_card",
    "ranking_table",
    "action_proposal",
    "breakdown",
    "progress",
    "health_report",
)

RENDER_POLICY_PROMPT = """You turn a project-management answer into a short list of UI blocks.
Output ONLY a JSON object: {"version":"v1","blocks":[ ... ]}. Nothing outside the JSON.

Each block is {"type": <one type below>, ...fields}. Use ONLY these types:
- prose {"markdown": str}: plain text. The DEFAULT — prefer prose unless the data has real structure.
- code {"language": str, "code": str}: fenced code or JSON examples.
- entity_card {"entity":"issue"|"sprint","id":str,"title":str,"href":str,"status"?,"priority"?,"summary"?}: ONE issue/sprint.
- ranking_table {"columns":[{"key":str,"label":str}],"rows":[{"cells":[...],"highlight"?:bool}]}: comparisons/rankings; set highlight:true on the winning row.
- breakdown {"dimension":"status"|"priority"|"assignee"|"type","segments":[{"label":str,"value":num}]}: distributions/counts.
- progress {"kind":"sprint"|"epic","completed":num,"total":num}: sprint/epic progress.
- health_report {"sections":[{"kind":"going_well"|"at_risk"|"recommended","heading":str,"items":[{"text":str,"severity"?,"quote"?,"evidence"?}]}]}: a "going well / at risk / recommended" report. A quote MUST carry evidence (a real {"kind","id","label","href"}) — never invent a complaint.
- action_proposal {"action":str,"entity":str,"summary":str,"executorRef":str}: a proposed write.

Rules:
- Prose by default. Use a structured block ONLY when the answer genuinely has that shape; never wrap a one-line reply in a card.
- NEVER invent data: ids, numbers, names and quotes must come from the answer/data you are given. When unsure, use a prose block.
- Match the user's language. Keep it tight — a few blocks at most.
"""

__all__ = ["BLOCK_TYPES", "RENDER_POLICY_PROMPT"]
