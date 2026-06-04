from langchain_core.messages import AIMessage, HumanMessage

from memory.store import DEFAULT_PENDING_TTL_SECONDS, ConversationStore


def test_history_empty_for_new_thread() -> None:
    store = ConversationStore()
    assert store.history("t1", max_turns=10) == []


def test_append_and_history_roundtrip() -> None:
    store = ConversationStore()
    store.append("t1", HumanMessage(content="hi"), AIMessage(content="hello"))
    assert [m.content for m in store.history("t1", max_turns=10)] == ["hi", "hello"]


def test_history_windows_to_last_n_turns() -> None:
    store = ConversationStore()
    for i in range(15):
        store.append("t1", HumanMessage(content=f"u{i}"), AIMessage(content=f"a{i}"))
    hist = store.history("t1", max_turns=3)  # 3 turns == 6 messages
    assert len(hist) == 6
    assert hist[0].content == "u12"
    assert hist[-1].content == "a14"


def test_history_zero_turns_returns_empty() -> None:
    store = ConversationStore()
    store.append("t1", HumanMessage(content="hi"))
    assert store.history("t1", max_turns=0) == []


def test_threads_are_isolated() -> None:
    store = ConversationStore()
    store.append("t1", HumanMessage(content="one"))
    store.append("t2", HumanMessage(content="two"))
    assert [m.content for m in store.history("t1", 10)] == ["one"]
    assert [m.content for m in store.history("t2", 10)] == ["two"]


def test_pending_set_get_clear() -> None:
    store = ConversationStore()
    assert store.get_pending("t1") is None
    store.set_pending("t1", {"intent": "CREATE_ISSUE", "message": "x"})
    pending = store.get_pending("t1")
    assert pending is not None and pending["intent"] == "CREATE_ISSUE"
    store.clear_pending("t1")
    assert store.get_pending("t1") is None


def test_pending_is_per_thread() -> None:
    store = ConversationStore()
    store.set_pending("t1", {"intent": "CREATE_ISSUE"})
    assert store.get_pending("t2") is None


def test_set_pending_records_timestamp() -> None:
    store = ConversationStore()
    store.set_pending("t1", {"intent": "CREATE_ISSUE"}, now=1_000_000.0)
    pending = store.get_pending("t1")
    assert pending is not None and pending["ts"] == 1_000_000.0


def test_get_pending_drops_expired_and_returns_none() -> None:
    """A pending older than the TTL must be cleared on read and reported as gone."""
    store = ConversationStore()
    store.set_pending("t1", {"intent": "CREATE_ISSUE"}, now=1_000_000.0)
    # Exactly at TTL is still alive (>, not >=)
    assert store.get_pending("t1", ttl_seconds=10.0, now=1_000_010.0) is not None
    # Past TTL — expired
    assert store.get_pending("t1", ttl_seconds=10.0, now=1_000_011.0) is None
    # And the store is now clean: a subsequent no-TTL read still returns nothing.
    assert store.get_pending("t1") is None


def test_default_ttl_is_long_enough_for_normal_conversations() -> None:
    """Sanity: the default TTL should comfortably cover a 10-min coffee break."""
    assert DEFAULT_PENDING_TTL_SECONDS >= 60.0


def test_reset_clears_one_thread_only() -> None:
    store = ConversationStore()
    store.append("t1", HumanMessage(content="hi"))
    store.set_pending("t1", {"intent": "X"})
    store.append("t2", HumanMessage(content="keep"))
    store.reset("t1")
    assert store.history("t1", 10) == []
    assert store.get_pending("t1") is None
    assert [m.content for m in store.history("t2", 10)] == ["keep"]
