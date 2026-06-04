"""Per-tenant LLM token usage accounting.

Cumulative per org, in-process for the process lifetime — suits a single
instance. Swap for a Redis/Postgres-backed store with a periodic (e.g. monthly)
reset before multi-instance or real billing. See AIService.md "COST TRACKING".
"""


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

    def reset(self, org_slug: str) -> None:
        self._tokens.pop(org_slug, None)
        self._requests.pop(org_slug, None)

    def reset_all(self) -> None:
        self._tokens.clear()
        self._requests.clear()


usage_store = UsageStore()
