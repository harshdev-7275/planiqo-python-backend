"""Output render contract — the PM-scoped block vocabulary.

The AI assistant's read-side output is rendered as *domain-constrained*
generative UI: the LLM never emits arbitrary UI, only typed blocks drawn
from this fixed catalog. This module is the SINGLE SOURCE OF TRUTH for
that catalog. Three consumers derive from it and therefore cannot drift:

  * the prompt policy that tells the LLM which block to use when,
  * the boundary validator (``chains/render_contract.py``),
  * the frontend block registry (block ``type`` -> React component).

Wire format is **camelCase** (``avatarUrl``, ``dueDate``, ``confirmRequired``)
so the frontend Zod mirror in ``frontend/src/types/renderBlocks.ts`` matches
field-for-field. Models still accept snake_case on the way in
(``populate_by_name``), so Python-side construction stays idiomatic; only
``model_dump(by_alias=True)`` emits camelCase for the response.

Two design rules are baked into the types on purpose:

  * ``TIER`` is an intrinsic property of the block *type* (a ``ClassVar``),
    NEVER a field the LLM emits. The model only sends ``type`` + props, so
    the wire format stays minimal (token budget) and the emission router's
    inline/composed decision can't be spoofed by the model.

  * ``health_report`` DERIVES ``overall`` from the worst section severity
    rather than trusting an LLM-asserted value, and REJECTS an unsourced
    ``quote``. This is the same "don't trust the model's summary, compute
    it" philosophy as the deterministic write executor — a model that
    lists three critical risks cannot also claim the sprint is "healthy",
    and it cannot fabricate a "brand new complaint" no user ever wrote.

Unknown extra fields are ignored (Pydantic default), not rejected: dropping
an otherwise-valid block because the model added a stray key is worse UX
than silently ignoring the key. The ``action_proposal`` payload is opaque
here on purpose — the deterministic write executor validates it, not this
layer.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

# --- shared vocabularies -----------------------------------------------------

Tier = Literal["inline", "composed"]
Status = Literal["backlog", "todo", "in_progress", "in_review", "done", "blocked"]
Priority = Literal["urgent", "high", "medium", "low"]
Severity = Literal["info", "warning", "critical"]

_SEVERITY_RANK: dict[Severity, int] = {"info": 0, "warning": 1, "critical": 2}
_OVERALL_BY_RANK: dict[int, Literal["healthy", "at_risk", "critical"]] = {
    0: "healthy",
    1: "at_risk",
    2: "critical",
}


class _ContractModel(BaseModel):
    """Base for every render-contract model.

    Serializes camelCase on the wire (so the frontend Zod mirror matches) and
    accepts either snake_case or camelCase on the way in.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# --- shared primitives -------------------------------------------------------


class Assignee(_ContractModel):
    id: str
    name: str
    avatar_url: str | None = None


class Ref(_ContractModel):
    """A link to a real PM entity. Used as evidence for grounded claims."""

    kind: Literal["issue", "sprint", "project"]
    id: str
    label: str
    href: str


# --- inline-tier blocks ------------------------------------------------------
# Simple, single-shape answers the read agent may emit directly in one pass.


class ProseBlock(_ContractModel):
    """Escape hatch and fallback. Plain markdown prose."""

    TIER: ClassVar[Tier] = "inline"
    type: Literal["prose"] = "prose"
    markdown: str


class CodeBlock(_ContractModel):
    TIER: ClassVar[Tier] = "inline"
    type: Literal["code"] = "code"
    language: str
    code: str
    caption: str | None = None


class MetaItem(_ContractModel):
    label: str
    value: str


class EntityCardBlock(_ContractModel):
    """A summary card for one issue or sprint."""

    TIER: ClassVar[Tier] = "inline"
    type: Literal["entity_card"] = "entity_card"
    entity: Literal["issue", "sprint"]
    id: str
    title: str
    href: str
    status: Status | None = None
    priority: Priority | None = None
    assignees: list[Assignee] = Field(default_factory=list)
    meta: list[MetaItem] = Field(default_factory=list)
    summary: str | None = None


class TableColumn(_ContractModel):
    key: str
    label: str
    align: Literal["left", "right"] = "left"


class TableRow(_ContractModel):
    cells: list[str | int | float]
    href: str | None = None
    highlight: bool = False  # the "winner in bold"


class RankingTableBlock(_ContractModel):
    """A ranking/comparison table, e.g. assignees by open issue count."""

    TIER: ClassVar[Tier] = "inline"
    type: Literal["ranking_table"] = "ranking_table"
    title: str | None = None
    columns: list[TableColumn] = Field(min_length=1)
    rows: list[TableRow] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rows_match_columns(self) -> RankingTableBlock:
        width = len(self.columns)
        for row in self.rows:
            if len(row.cells) != width:
                raise ValueError("row cell count does not match column count")
        return self


class ActionProposalBlock(_ContractModel):
    """A proposed write. Renders a confirm card; NEVER auto-executes.

    ``payload`` is an opaque dict on purpose — it is validated by the
    deterministic write executor, not here. This block only carries the
    *intent to write*; the existing confirm-and-execute path owns safety.
    """

    TIER: ClassVar[Tier] = "inline"
    type: Literal["action_proposal"] = "action_proposal"
    action: Literal["create", "update", "transition", "assign", "delete"]
    entity: Literal["issue", "sprint", "comment"]
    summary: str
    payload: dict[str, object] = Field(default_factory=dict)
    confirm_required: Literal[True] = True
    executor_ref: str


# --- composed-tier blocks ----------------------------------------------------
# Multi-shape / roll-up answers. Always routed to the presentation pass.


class BreakdownSegment(_ContractModel):
    label: str
    value: float
    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    tone: Literal["neutral", "good", "warn", "bad"] = "neutral"


class BreakdownBlock(_ContractModel):
    """A distribution: status/priority/assignee/type -> count."""

    TIER: ClassVar[Tier] = "composed"
    type: Literal["breakdown"] = "breakdown"
    title: str | None = None
    dimension: Literal["status", "priority", "assignee", "type"]
    display: Literal["bar", "stat_row"] = "bar"
    total: float | None = None
    segments: list[BreakdownSegment] = Field(default_factory=list)


class TrendPoint(_ContractModel):
    date: str
    remaining: float


class PointsProgress(_ContractModel):
    done: float
    total: float


class ProgressBlock(_ContractModel):
    """Sprint/epic progress, optionally with a burndown trend."""

    TIER: ClassVar[Tier] = "composed"
    type: Literal["progress"] = "progress"
    kind: Literal["sprint", "epic"]
    title: str | None = None
    completed: float
    total: float
    due_date: str | None = None
    points: PointsProgress | None = None
    trend: list[TrendPoint] = Field(default_factory=list)


class HealthItem(_ContractModel):
    text: str
    severity: Severity = "info"
    quote: str | None = None
    evidence: Ref | None = None
    proposal: ActionProposalBlock | None = None  # a recommendation can be actionable

    @model_validator(mode="after")
    def _quote_requires_evidence(self) -> HealthItem:
        # A quoted complaint must point at a real entity. Without this an
        # LLM can fabricate a "brand new complaint" no user ever wrote.
        if self.quote is not None and self.evidence is None:
            raise ValueError("quote requires evidence (anti-fabrication)")
        return self


class HealthSection(_ContractModel):
    kind: Literal["going_well", "at_risk", "recommended"]
    heading: str
    items: list[HealthItem] = Field(min_length=1)


class HealthReportBlock(_ContractModel):
    """A health panel: going well / at risk / recommended.

    ``overall`` is derived from the worst severity in the ``at_risk``
    section, ignoring any LLM-asserted value.
    """

    TIER: ClassVar[Tier] = "composed"
    type: Literal["health_report"] = "health_report"
    title: str | None = None
    overall: Literal["healthy", "at_risk", "critical"] = "healthy"
    sections: list[HealthSection] = Field(min_length=1)

    @model_validator(mode="after")
    def _derive_overall(self) -> HealthReportBlock:
        worst = 0
        for section in self.sections:
            if section.kind != "at_risk":
                continue
            for item in section.items:
                worst = max(worst, _SEVERITY_RANK[item.severity])
        self.overall = _OVERALL_BY_RANK[worst]
        return self


# --- the union + envelope ----------------------------------------------------

Block = Annotated[
    Union[
        ProseBlock,
        CodeBlock,
        EntityCardBlock,
        RankingTableBlock,
        ActionProposalBlock,
        BreakdownBlock,
        ProgressBlock,
        HealthReportBlock,
    ],
    Field(discriminator="type"),
]


class RenderResponse(_ContractModel):
    """The wire envelope: a versioned, ordered list of blocks."""

    version: Literal["v1"] = "v1"
    blocks: list[Block] = Field(default_factory=list)


def tier_of(block: Block) -> Tier:
    """The intrinsic tier of a block. Read from the type, never the data."""

    return block.TIER


__all__ = [
    "Assignee",
    "Block",
    "BreakdownBlock",
    "BreakdownSegment",
    "CodeBlock",
    "EntityCardBlock",
    "HealthItem",
    "HealthReportBlock",
    "HealthSection",
    "MetaItem",
    "PointsProgress",
    "Priority",
    "ProgressBlock",
    "ProseBlock",
    "RankingTableBlock",
    "Ref",
    "RenderResponse",
    "Severity",
    "Status",
    "TableColumn",
    "TableRow",
    "Tier",
    "TrendPoint",
    "tier_of",
]
