"""Tests for the persona resolver chain.

The resolver is data, not code — phrases and tone live in the Persona
config, the LLM phrases them per the persona's voice_directive. The
resolver itself is just a 4-step chain (override → user → org → global)
with a per-(org, user) cache.
"""

from __future__ import annotations

import pytest

from chains.persona_resolver import (
    Persona,
    reset_cache,
    resolve_persona,
)
from config.settings import settings


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_cache()
    yield
    reset_cache()


# --- the four built-in personas exist and have the right shape ----------------


def test_senior_pm_is_a_persona() -> None:
    p = resolve_persona("acme", "u1")
    assert p.name == "senior_pm"
    assert 0.0 <= p.opinion_strength <= 1.0
    assert 0.0 <= p.suggestion_aggressiveness <= 1.0
    assert p.voice_directive  # non-empty
    assert isinstance(p.rules, list)


def test_three_builtins_are_loadable() -> None:
    """The three shipped personas all resolve to a real Persona (not a
    string or a placeholder)."""
    for name in ("senior_pm", "auditor", "assistant"):
        # Use the override path to force a specific one
        p = resolve_persona("acme", "u1", override=name)
        assert p.name == name


def test_unknown_persona_falls_back_to_default() -> None:
    """An unknown persona name (typo, deprecated) must not crash the chat —
    it falls back to the global default."""
    p = resolve_persona("acme", "u1", override="this-does-not-exist")
    assert p.name == settings.PERSONA_DEFAULT


# --- the resolution chain (override > cache > default) -----------------------


def test_override_wins_over_default() -> None:
    """When the user said 'be more concise' mid-thread, the override
    argument is passed in. Override must win over the default."""
    p = resolve_persona("acme", "u1", override="auditor")
    assert p.name == "auditor"


def test_cache_hits_skip_the_resolver() -> None:
    """Two calls with the same (org, user) within TTL must return the same
    Persona instance — the second is a cache hit, no re-resolve."""
    a = resolve_persona("acme", "u1")
    b = resolve_persona("acme", "u1")
    assert a is b  # same object, not just equal


def test_different_orgs_get_different_cache_entries() -> None:
    """The cache key is (org, user) — different orgs are isolated."""
    from chains.persona_resolver import _cache

    resolve_persona("acme", "u1")
    resolve_persona("globex", "u1")
    # Two distinct cache entries, one per (org, user).
    assert len(_cache) == 2
    assert ("acme", "u1") in _cache
    assert ("globex", "u1") in _cache

    # The cached persona for each is the same NAME (we have no per-org
    # override yet) but they were stored as two separate cache entries.
    assert _cache[("acme", "u1")][0].name == _cache[("globex", "u1")][0].name


def test_override_is_not_cached() -> None:
    """An override is for one turn only — it must not poison the cache."""
    with_override = resolve_persona("acme", "u1", override="auditor")
    assert with_override.name == "auditor"

    # Next call without override returns the default again.
    default = resolve_persona("acme", "u1")
    assert default.name == settings.PERSONA_DEFAULT


# --- reset_cache is the test seam --------------------------------------------


def test_reset_cache_clears_all_entries() -> None:
    """Tests + admin tools call reset_cache() to force a re-resolve.

    The cache is module-level, so we can poke at it directly. The actual
    object identity comparison is not the right test (Pydantic v2 returns
    value-equal objects from a class-level dict); we just verify that the
    cache is empty after reset and that the next resolve repopulates it.
    """
    from chains.persona_resolver import _cache

    resolve_persona("acme", "u1")
    assert _cache.get(("acme", "u1")) is not None  # cache populated
    assert len(_cache) == 1

    reset_cache()
    assert _cache == {}  # cache fully cleared

    resolve_persona("acme", "u1")
    assert _cache.get(("acme", "u1")) is not None  # repopulated on next resolve
    assert len(_cache) == 1


# --- the persona itself is a well-typed Pydantic model -----------------------


def test_persona_model_rejects_garbage() -> None:
    """Persona validates its own fields — opinion_strength must be 0..1."""
    with pytest.raises(ValueError):
        Persona(name="x", voice_directive="x", opinion_strength=2.0)


def test_verbosity_enum_is_closed() -> None:
    """Verbosity is a closed enum — no free-form string."""
    with pytest.raises(ValueError):
        Persona(name="x", voice_directive="x", verbosity="extremely verbose")  # type: ignore[arg-type]
