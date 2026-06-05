"""Lightweight in-process latency metrics for the /metrics endpoint (item 23).

Tracks per-request /chat latency with running aggregates (count, total, max)
plus a bounded recent-sample window for percentiles. In-process and
single-instance — same constraint as ``UsageStore`` — but enough for an ops
dashboard to spot a latency regression without pulling in a Prometheus client.

A process restart zeroes the stats; that's acceptable for an at-a-glance view.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any

# Keep only the most recent N samples for percentiles, so memory is bounded
# regardless of traffic. The running count/total/max are exact and unbounded.
_SAMPLE_WINDOW = 1000


class LatencyStats:
    def __init__(self, window: int = _SAMPLE_WINDOW) -> None:
        self._count = 0
        self._total_ms = 0.0
        self._max_ms = 0.0
        self._samples: deque[float] = deque(maxlen=window)

    def record(self, ms: float) -> None:
        if ms < 0:
            return
        self._count += 1
        self._total_ms += ms
        self._max_ms = max(self._max_ms, ms)
        self._samples.append(ms)

    def snapshot(self) -> dict[str, Any]:
        if self._count == 0:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        samples = sorted(self._samples)
        return {
            "count": self._count,
            "mean_ms": round(self._total_ms / self._count, 1),
            "p50_ms": round(_percentile(samples, 50), 1),
            "p95_ms": round(_percentile(samples, 95), 1),
            "max_ms": round(self._max_ms, 1),
        }

    def reset(self) -> None:
        self._count = 0
        self._total_ms = 0.0
        self._max_ms = 0.0
        self._samples.clear()


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile over a pre-sorted list. ``statistics.quantiles``
    needs >1 point, so we handle the single-sample case explicitly."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    quantiles = statistics.quantiles(sorted_samples, n=100, method="inclusive")
    # quantiles has 99 cut points (1..99); index pct-1, clamped.
    idx = min(max(int(pct) - 1, 0), len(quantiles) - 1)
    return quantiles[idx]


# Module-level singleton — recorded by the /chat handler, read by /metrics.
latency_stats = LatencyStats()
