"""In-process conversation memory for the supervisor.

Holds, per conversation thread, a rolling window of messages and at most one
*pending action* awaiting user confirmation.

Storage is process-local: it suits a single instance (see AIService.md
"SCALING CONSIDERATIONS"). Before running multiple instances, swap this for a
Redis/Postgres-backed implementation behind the same interface. Access is
await-free, so it is safe under a single asyncio event loop without a lock.
"""

import time
from typing import Any

from langchain_core.messages import BaseMessage

# Hard cap on stored messages per thread, independent of the read window, so a
# long-lived conversation can never grow without bound.
_MAX_STORED_MESSAGES = 100

# How long a pending action waits for confirmation before it is dropped. The
# supervisor checks this on every read so a user who walks away does not leave
# a stale proposal that hijacks the next unrelated message.
DEFAULT_PENDING_TTL_SECONDS = 600.0


class ConversationStore:
    def __init__(self) -> None:
        self._messages: dict[str, list[BaseMessage]] = {}
        self._pending: dict[str, dict[str, Any]] = {}

    def history(self, thread_id: str, max_turns: int) -> list[BaseMessage]:
        """Return up to the last ``max_turns`` turns (2 messages per turn)."""
        if max_turns <= 0:
            return []
        msgs = self._messages.get(thread_id, [])
        return list(msgs[-(max_turns * 2):])

    def append(self, thread_id: str, *messages: BaseMessage) -> None:
        bucket = self._messages.setdefault(thread_id, [])
        bucket.extend(messages)
        if len(bucket) > _MAX_STORED_MESSAGES:
            del bucket[:-_MAX_STORED_MESSAGES]

    def get_pending(
        self, thread_id: str, ttl_seconds: float | None = None, now: float | None = None
    ) -> dict[str, Any] | None:
        """Return the pending action for ``thread_id`` if it exists and (if
        ``ttl_seconds`` is set) was created within the TTL. Expired entries are
        cleared on read."""
        action = self._pending.get(thread_id)
        if action is None:
            return None
        if ttl_seconds is not None:
            ts = action.get("ts")
            if ts is None or (now if now is not None else time.time()) - ts > ttl_seconds:
                self._pending.pop(thread_id, None)
                return None
        return action

    def set_pending(
        self,
        thread_id: str,
        action: dict[str, Any],
        now: float | None = None,
    ) -> None:
        """Store a pending action. ``ts`` (epoch seconds) is recorded so a
        later ``get_pending(ttl_seconds=...)`` can expire it. ``now`` is for
        tests."""
        action.setdefault("ts", now if now is not None else time.time())
        self._pending[thread_id] = action

    def clear_pending(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)

    def reset(self, thread_id: str) -> None:
        self._messages.pop(thread_id, None)
        self._pending.pop(thread_id, None)

    def reset_all(self) -> None:
        self._messages.clear()
        self._pending.clear()


conversation_store = ConversationStore()
