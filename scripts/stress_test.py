    """Stress-test runner for the AI service.

Posts every query in ``stress_queries.json`` to ``POST /chat`` and prints
a one-line summary per query (intent, status, response, tokens, latency).
At the end it prints an aggregate report (counts by intent, success rate,
total tokens, total time) so you can spot misclassifications and broken
intents at a glance.

Run with:
    cd ai-service
    uv run python scripts/stress_test.py

Pre-reqs:
- The ai-service is running on $AI_SERVICE_URL (default http://localhost:8000)
- INTERNAL_SECRET is exported (matches the running service)
- The "stress" org + "stress-proj" project do NOT need to exist — the
  bot will surface "no team member named X" / "no issues found" type
  messages, which is exactly what we want to see.

This is a one-shot script — no pytest, no CI. The point is to eyeball
how the bot handles a diverse set of complex queries; the structured
intent eval harness is a separate concern (see the plan, item 4.5).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# --- config -----------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000"
SCRIPT_DIR = Path(__file__).resolve().parent
QUERIES_PATH = SCRIPT_DIR / "stress_queries.json"
# .env lives at ai-service/.env (one level up from scripts/).
ENV_PATH = SCRIPT_DIR.parent / ".env"

# Stable identity for the stress session — every query is sent "as" this
# user. Use a distinct user so the conversation_store doesn't bleed into
# other test sessions.
STRESS_USER = "stress-tester"
STRESS_ORG = "stress"
STRESS_PROJECT = "stress-proj"


def _load_env_file(path: Path) -> None:
    """Tiny .env loader — only sets vars that are not already in the
    process env. Avoids a python-dotenv dependency for one file. Stops
    silently if the file is missing — the operator may set secrets in
    the shell instead."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _check_env() -> tuple[str, str]:
    """Validate the required env vars. Returns (base_url, secret).

    Order of resolution:
      1. Already-set process env (e.g. ``INTERNAL_SECRET=... uv run …``)
      2. ai-service/.env (auto-loaded)
      3. Fail with a clear message."""
    _load_env_file(ENV_PATH)
    base_url = os.environ.get("AI_SERVICE_URL", DEFAULT_URL)
    secret = os.environ.get("INTERNAL_SECRET")
    if not secret:
        sys.stderr.write(
            "INTERNAL_SECRET is not set — the service will reject every "
            "request. Either export it in your shell (must match the "
            "value the ai-service is running with) or set it in "
            "ai-service/.env. Then try again.\n"
        )
        sys.exit(2)
    return base_url, secret


# --- per-query runner -------------------------------------------------------


async def _run_one(
    client: httpx.AsyncClient,
    query: str,
    secret: str,
) -> dict[str, Any]:
    """POST a single query to /chat. Returns a normalised result dict.

    The /chat endpoint is protected by the InternalAuthMiddleware, which
    checks the X-Internal-Secret header on every request (see
    middleware/auth.py). The Node backend is the real caller; for the
    stress test we set the same header so the middleware lets us through.
    """
    body = {
        "message": query,
        "user_id": STRESS_USER,
        "org_slug": STRESS_ORG,
        "project_id": STRESS_PROJECT,
    }
    headers = {"X-Internal-Secret": secret}
    start = time.perf_counter()
    try:
        resp = await client.post("/chat", json=body, headers=headers, timeout=60.0)
        elapsed = time.perf_counter() - start
    except httpx.HTTPError as e:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "intent": None,
            "status": "transport_error",
            "message": f"{type(e).__name__}: {e}",
            "tokens": 0,
            "elapsed": elapsed,
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "intent": None,
            "status": f"http_{resp.status_code}",
            "message": resp.text[:200],
            "tokens": 0,
            "elapsed": elapsed,
        }
    data = resp.json()
    return {
        "ok": True,
        "intent": data.get("intent"),
        "status": data.get("status"),
        "message": (data.get("result") or {}).get("message") or "",
        "tokens": data.get("tokens_used", 0),
        "elapsed": elapsed,
    }


def _short(text: str, limit: int = 100) -> str:
    """Truncate a string for display, appending an ellipsis if needed.
    Strips newlines so the single-line summary stays readable."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _print_query_line(idx: int, total: int, entry: dict[str, Any], result: dict[str, Any]) -> None:
    """Print a single-line summary for one query."""
    query = _short(entry["query"], 60)
    intent = result["intent"] or "—"
    status = result["status"] or "—"
    msg = _short(result["message"], 100)
    tok = result["tokens"]
    elapsed_ms = int(result["elapsed"] * 1000)
    expected = entry.get("expected_intent", "")
    match = ""
    if expected and result["intent"]:
        match = "✓" if expected == result["intent"] else "✗"
    logger.info(
        "[{idx:>{width}}/{total}] {q} | intent={intent:<14} status={status:<22} "
        "tokens={tok:<5} {ms:>5}ms {match} → {msg}",
        idx=idx, width=len(str(total)), total=total, q=query, intent=intent,
        status=status, tok=tok, ms=elapsed_ms, match=match, msg=msg,
    )


# --- main -------------------------------------------------------------------


async def main() -> int:
    base_url, _secret = _check_env()

    # Load the dataset.
    try:
        with QUERIES_PATH.open(encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(f"queries file not found: {QUERIES_PATH}\n")
        return 2
    except json.JSONDecodeError as e:
        sys.stderr.write(f"queries file is not valid JSON: {e}\n")
        return 2

    queries: list[dict[str, Any]] = dataset.get("queries", [])
    if not queries:
        sys.stderr.write(f"no queries in {QUERIES_PATH}\n")
        return 2

    logger.info("stress test: {n} queries against {url}", n=len(queries), url=base_url)
    logger.info("  user={u} org={o} project={p}", u=STRESS_USER, o=STRESS_ORG, p=STRESS_PROJECT)
    logger.info("─" * 100)

    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    intent_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    total_tokens = 0
    total_elapsed = 0.0
    total_ok = 0
    matches = 0
    expected_total = 0

    # Sleep between queries to stay under the Groq free-tier rate limit
    # (30 req/min for llama-3.1-8b). Without this, a 33-query burst at
    # ~1s each can hit ~33 req/min which is over the limit — the LLM
    # then rate-limits and the supervisor's defensive try/except kicks
    # in, returning "I hit an unexpected error". Better: pace the run.
    sleep_seconds = float(os.environ.get("STRESS_TEST_SLEEP", "0.5"))

    async with httpx.AsyncClient(base_url=base_url) as client:
        for i, entry in enumerate(queries, 1):
            r = await _run_one(client, entry["query"], _secret)
            _print_query_line(i, len(queries), entry, r)
            results.append((entry, r))
            if i < len(queries) and sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
            intent_counts[r["intent"] or "—"] += 1
            status_counts[r["status"] or "—"] += 1
            total_tokens += r["tokens"]
            total_elapsed += r["elapsed"]
            if r["ok"]:
                total_ok += 1
            if entry.get("expected_intent"):
                expected_total += 1
                if r["intent"] == entry["expected_intent"]:
                    matches += 1

    # Summary.
    logger.info("─" * 100)
    logger.info("summary")
    logger.info("  total queries:     {n}", n=len(queries))
    logger.info("  successful (200):  {n}/{total}", n=total_ok, total=len(queries))
    logger.info("  total tokens:      {n}", n=total_tokens)
    logger.info(
        "  total time:        {s:.1f}s  (mean {m:.0f}ms, p50 {p50:.0f}ms, p95 {p95:.0f}ms)",
        s=total_elapsed,
        m=statistics.mean(r["elapsed"] for _, r in results) * 1000,
        p50=statistics.median(r["elapsed"] for _, r in results) * 1000,
        p95=(
            statistics.quantiles(
                [r["elapsed"] for _, r in results], n=20, method="inclusive"
            )[18]
            * 1000
            if len(results) > 1
            else 0.0
        ),
    )
    if expected_total:
        logger.info(
            "  intent match:      {m}/{t} ({pct:.0f}%)  vs expected_intent in the dataset",
            m=matches, t=expected_total, pct=100 * matches / expected_total,
        )
    logger.info("  intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        logger.info("    {intent:<22} {n}", intent=intent, n=count)
    logger.info("  status distribution:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        logger.info("    {status:<22} {n}", status=status, n=count)
    return 0


if __name__ == "__main__":
    # Configure loguru for terminal output (no file, simple format).
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    import asyncio
    sys.exit(asyncio.run(main()))
