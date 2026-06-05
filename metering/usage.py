"""Per-tenant LLM token usage accounting.

Cumulative per org. The default backend is the in-process ``UsageStore``
(single instance only — a process restart zeroes the counters, multiple
instances have independent counters so the quota is under-enforced). For
multi-instance / real-billing setups, set ``METERING_BACKEND=postgres`` and
the supervisor will use ``PostgresUsageStore`` which is backed by the
node-api's ``/admin/metering/*`` routes (Postgres is the source of truth).

Per AIService.md, the metering layer must degrade gracefully — the
Postgres store swallows transport errors and returns safe defaults so a
node-api outage does not crash the chat. (See "COST TRACKING".)
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


class RequestTokens:
    """Per-request token accumulator.

    The UsageCallback receives one of these and increments it on every LLM
    end-event. The supervisor reads ``.total`` after the request finishes and
    includes it in the ``/chat`` response. Lives for one HTTP request.
    """

    def __init__(self) -> None:
        self.total: int = 0

    def add(self, tokens: int) -> None:
        if tokens > 0:
            self.total += tokens


class UsageStore:
    """In-process per-org token + request counter. Single-instance only."""

    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._requests: dict[str, int] = {}

    def add(self, org_slug: str, tokens: int) -> None:
        if tokens <= 0:
            return
        self._tokens[org_slug] = self._tokens.get(org_slug, 0) + tokens

    def get(self, org_slug: str) -> int:
        return self._tokens.get(org_slug, 0)

    def inc_request(self, org_slug: str) -> int:
        """Bump the request count for an org and return the new total."""
        self._requests[org_slug] = self._requests.get(org_slug, 0) + 1
        return self._requests[org_slug]

    def get_request_count(self, org_slug: str) -> int:
        return self._requests.get(org_slug, 0)

    def snapshot(self) -> dict[str, Any]:
        """Aggregate view for the /metrics endpoint: grand totals plus a
        per-org breakdown. Reads only — never mutates the counters."""
        orgs = sorted(set(self._tokens) | set(self._requests))
        return {
            "org_count": len(orgs),
            "total_tokens": sum(self._tokens.values()),
            "total_requests": sum(self._requests.values()),
            "per_org": {
                org: {
                    "tokens": self._tokens.get(org, 0),
                    "requests": self._requests.get(org, 0),
                }
                for org in orgs
            },
        }

    def reset(self, org_slug: str) -> None:
        self._tokens.pop(org_slug, None)
        self._requests.pop(org_slug, None)

    def reset_all(self) -> None:
        self._tokens.clear()
        self._requests.clear()


class PostgresUsageStore:
    """Async metering backend backed by the node-api's ``/admin/metering/*``
    routes. The Postgres table in node-backend is the source of truth —
    multiple ai-service instances see the same numbers, and a deploy does
    not reset the counters.

    The store is deliberately **async** because every method makes a network
    call. Callers that today use the sync ``UsageStore`` will need a small
    adapter before switching the backend; the supervisor's quota check is
    the only hot path that would need to ``await`` a method.

    Transport errors are LOGGED and DOWNGRADED to safe defaults:
      * ``get`` returns 0 (the supervisor's ``>=`` quota check then
        over-permits, never under-permits — a billing issue, not a chat
        outage).
      * ``add`` is a no-op (the tokens are lost for the duration of the
        outage; acceptable — the chat is unaffected).
    This honours AIService.md's "If AI is unavailable, the core PM tool
    still works."
    """

    def __init__(
        self,
        base_url: str,
        internal_secret: str,
        http_client: Any | None = None,
    ) -> None:
        # Accept an injected client for testability; the default is a fresh
        # httpx.AsyncClient that the caller is expected to close (the
        # singleton in this module owns its lifetime via ``aclose()``).
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Internal-Secret": internal_secret},
            timeout=5.0,
        )

    async def add(self, org_slug: str, tokens: int) -> None:
        if tokens <= 0:
            return
        try:
            await self._client.post(
                f"/admin/metering/{org_slug}/add",
                json={"tokens": tokens},
            )
        except (httpx.HTTPError, httpx.RequestError) as e:
            logger.warning("metering: node-api add failed for {}: {}", org_slug, e)

    async def get(self, org_slug: str) -> int:
        try:
            resp = await self._client.get(f"/admin/metering/{org_slug}/get")
            return int(resp.json().get("tokens", 0))
        except (httpx.HTTPError, httpx.RequestError, ValueError) as e:
            logger.warning("metering: node-api get failed for {}: {}", org_slug, e)
            return 0

    async def inc_request(self, org_slug: str) -> int:
        try:
            resp = await self._client.post(
                f"/admin/metering/{org_slug}/inc-request"
            )
            return int(resp.json().get("requests", 0))
        except (httpx.HTTPError, httpx.RequestError, ValueError) as e:
            logger.warning("metering: node-api inc_request failed for {}: {}", org_slug, e)
            return 0

    async def get_request_count(self, org_slug: str) -> int:
        try:
            resp = await self._client.get(
                f"/admin/metering/{org_slug}/get-requests"
            )
            return int(resp.json().get("requests", 0))
        except (httpx.HTTPError, httpx.RequestError, ValueError) as e:
            logger.warning("metering: node-api get_requests failed for {}: {}", org_slug, e)
            return 0

    async def reset(self, org_slug: str) -> None:
        # No node-api route for reset yet — keep the no-op for protocol
        # completeness. Operator can ``db:clear --yes`` for now.
        logger.debug("PostgresUsageStore.reset({}) — no-op (no route)", org_slug)

    async def reset_all(self) -> None:
        logger.debug("PostgresUsageStore.reset_all() — no-op (no route)")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def get_usage_store() -> Any:
    """Factory: return the configured metering backend.

    The default is ``UsageStore`` (in-process) so dev never sees a change.
    Setting ``METERING_BACKEND=postgres`` returns ``PostgresUsageStore``.
    An unknown value raises — fail fast at startup so a typo in the env
    does not silently degrade the multi-instance story.

    The return type is ``Any`` because the two backends have different
    sync/async shapes — the protocol approach hits a mypy conflict here.
    Callers should narrow with ``isinstance`` when they need to use
    backend-specific methods.
    """
    from config.settings import settings

    backend = settings.METERING_BACKEND
    if backend == "inprocess":
        return UsageStore()
    if backend == "postgres":
        return PostgresUsageStore(
            base_url=settings.NODE_API_URL,
            internal_secret=settings.INTERNAL_SECRET,
        )
    raise ValueError(
        f"Unknown METERING_BACKEND={backend!r}. "
        f"Expected 'inprocess' or 'postgres'."
    )


# Module-level singleton. ``supervisor.run`` reads this once per request.
# Today this is always the in-process ``UsageStore`` (the postgres backend
# exists as a standalone class but the supervisor is not yet wired to its
# async API — switching is a future change that requires async call-site
# updates in supervisor.run).
usage_store: UsageStore = UsageStore()
