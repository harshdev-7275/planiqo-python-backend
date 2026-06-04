from metering.usage import RequestTokens, UsageStore


def test_add_accumulates_per_org() -> None:
    store = UsageStore()
    store.add("acme", 10)
    store.add("acme", 5)
    assert store.get("acme") == 15


def test_orgs_are_isolated() -> None:
    store = UsageStore()
    store.add("acme", 10)
    store.add("globex", 3)
    assert store.get("acme") == 10
    assert store.get("globex") == 3


def test_unknown_org_is_zero() -> None:
    assert UsageStore().get("nope") == 0


def test_add_ignores_nonpositive() -> None:
    store = UsageStore()
    store.add("acme", 0)
    store.add("acme", -5)
    assert store.get("acme") == 0


def test_reset_one_and_all() -> None:
    store = UsageStore()
    store.add("a", 1)
    store.add("b", 2)
    store.reset("a")
    assert store.get("a") == 0
    assert store.get("b") == 2
    store.reset_all()
    assert store.get("b") == 0


def test_inc_request_accumulates_per_org() -> None:
    store = UsageStore()
    assert store.inc_request("acme") == 1
    assert store.inc_request("acme") == 2
    assert store.inc_request("acme") == 3
    assert store.get_request_count("acme") == 3


def test_inc_request_is_per_org() -> None:
    store = UsageStore()
    store.inc_request("acme")
    store.inc_request("acme")
    store.inc_request("globex")
    assert store.get_request_count("acme") == 2
    assert store.get_request_count("globex") == 1


def test_reset_clears_request_count_too() -> None:
    store = UsageStore()
    store.add("a", 100)
    store.inc_request("a")
    store.reset("a")
    assert store.get("a") == 0
    assert store.get_request_count("a") == 0


def test_request_tokens_starts_at_zero_and_accumulates() -> None:
    rt = RequestTokens()
    assert rt.total == 0
    rt.add(7)
    rt.add(3)
    assert rt.total == 10


def test_request_tokens_ignores_nonpositive() -> None:
    rt = RequestTokens()
    rt.add(0)
    rt.add(-5)
    assert rt.total == 0
