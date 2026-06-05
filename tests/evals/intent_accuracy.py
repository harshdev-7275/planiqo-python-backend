"""Intent-classification eval harness (item 21).

Why this exists
---------------
A model swap (or a prompt edit) can silently tank intent accuracy — the
stress run once swung from 97% to 73% with no test catching it. This harness
runs the classifier over a curated golden set and reports accuracy + a
confusion list, so a regression is a hard, visible number rather than a
vibe.

Two surfaces:
  * ``evaluate(classify_fn, dataset)`` — pure scoring logic, unit-tested with
    a stubbed classifier (no API needed). This is what makes the harness
    itself trustworthy.
  * the integration gate in ``test_intent_accuracy.py`` — runs the REAL
    classifier and asserts accuracy >= ``ACCURACY_THRESHOLD``. Marked
    ``integration`` so it runs manually / in a scheduled job, never on the
    rate-limited PR path.

Run the gate:   uv run pytest tests/evals -m integration
Print a report: uv run python -m tests.evals.intent_accuracy
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from models.intents import IntentResult

# The minimum acceptable accuracy on the golden set. Set conservatively below
# the observed steady-state (~0.95+) so normal model jitter doesn't flake the
# gate, but a real regression (the 73% kind) trips it.
ACCURACY_THRESHOLD = 0.85

_DATA_PATH = Path(__file__).resolve().parent / "golden_intents.json"

ClassifyFn = Callable[[str], Awaitable[IntentResult]]


@dataclass
class Mismatch:
    query: str
    expected: str
    got: str


@dataclass
class EvalResult:
    total: int
    correct: int
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def report(self) -> str:
        lines = [
            f"intent accuracy: {self.correct}/{self.total} = {self.accuracy:.1%}",
        ]
        if self.mismatches:
            lines.append("mismatches:")
            for m in self.mismatches:
                lines.append(f"  expected {m.expected:<14} got {m.got:<14} | {m.query}")
        return "\n".join(lines)


def load_golden(path: Path | None = None) -> list[dict[str, str]]:
    """Load the golden cases. Each is ``{"query": str, "expected": str}``."""
    data = json.loads((path or _DATA_PATH).read_text(encoding="utf-8"))
    cases: list[dict[str, str]] = data.get("cases", [])
    if not cases:
        raise ValueError(f"golden set at {path or _DATA_PATH} has no cases")
    return cases


async def evaluate(
    classify_fn: ClassifyFn,
    dataset: list[dict[str, str]] | None = None,
) -> EvalResult:
    """Run ``classify_fn`` over the dataset and tally accuracy.

    ``classify_fn`` is any async ``str -> IntentResult`` — the real
    ``chains.intent.classify`` for the gate, a stub for the harness's own
    unit tests."""
    cases = dataset if dataset is not None else load_golden()
    correct = 0
    mismatches: list[Mismatch] = []
    for case in cases:
        result = await classify_fn(case["query"])
        got = result.intent.value
        if got == case["expected"]:
            correct += 1
        else:
            mismatches.append(Mismatch(case["query"], case["expected"], got))
    return EvalResult(total=len(cases), correct=correct, mismatches=mismatches)


async def _main() -> int:
    from chains.intent import classify

    result = await evaluate(classify)
    print(result.report())
    return 0 if result.accuracy >= ACCURACY_THRESHOLD else 1


if __name__ == "__main__":
    import asyncio
    import sys

    sys.exit(asyncio.run(_main()))
