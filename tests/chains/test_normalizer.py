"""Unit tests for chains/normalizer.py.

The normalizer applies PM-domain conventions as a post-pass over the LLM's
extracted entities. The rules are intentionally rule-based and auditable —
a tiny list of phrases, not a learned model.
"""

from __future__ import annotations

from chains.normalizer import normalize_entities


# --- bug → type=bug + priority=high ------------------------------------------


def test_bug_in_text_sets_type_when_type_is_null() -> None:
    out = normalize_entities(
        "Create a bug for the login page",
        {"title": "login", "type": None, "priority": None},
    )
    assert out["type"] == "bug"
    assert out["priority"] == "high"


def test_bug_in_text_sets_type_when_type_is_missing() -> None:
    out = normalize_entities("file a bug please", {"title": "x"})
    assert out["type"] == "bug"
    assert out["priority"] == "high"


def test_bug_word_matches_case_insensitively() -> None:
    out = normalize_entities("I have a BUG", {})
    assert out["type"] == "bug"


def test_type_bug_implies_priority_high_even_without_bug_word() -> None:
    """If the LLM already extracted type=bug, still default priority=high
    — but the bug word need not be in the raw text for that to fire."""
    out = normalize_entities("add a thing", {"type": "bug"})
    assert out["priority"] == "high"


def test_explicit_priority_is_not_overwritten_by_bug_convention() -> None:
    """If the LLM set priority=critical, the bug-default of 'high' must
    NOT overwrite it."""
    out = normalize_entities(
        "critical bug", {"type": "bug", "priority": "critical"},
    )
    assert out["priority"] == "critical"


# --- priority keywords ------------------------------------------------------


def test_critical_keyword_sets_priority_critical() -> None:
    out = normalize_entities("this is critical", {"type": "task"})
    assert out["priority"] == "critical"


def test_urgent_keyword_sets_priority_critical() -> None:
    out = normalize_entities("urgent fix needed", {})
    assert out["priority"] == "critical"


def test_blocker_keyword_sets_priority_critical() -> None:
    out = normalize_entities("blocker on the rollout", {})
    assert out["priority"] == "critical"


def test_prod_is_down_phrase_sets_priority_critical() -> None:
    """Multi-word phrase must win over single-word 'down' if both
    matched — the longer phrase is more specific."""
    out = normalize_entities("prod is down", {})
    assert out["priority"] == "critical"


def test_p0_keyword_sets_priority_critical() -> None:
    out = normalize_entities("this is p0", {})
    assert out["priority"] == "critical"


def test_sev1_keyword_sets_priority_critical() -> None:
    out = normalize_entities("sev1 incident", {})
    assert out["priority"] == "critical"


def test_asap_keyword_sets_priority_high() -> None:
    out = normalize_entities("asap please", {})
    assert out["priority"] == "high"


def test_important_keyword_sets_priority_high() -> None:
    out = normalize_entities("this is important", {})
    assert out["priority"] == "high"


def test_soon_keyword_sets_priority_high() -> None:
    out = normalize_entities("need it soon", {})
    assert out["priority"] == "high"


def test_p1_keyword_sets_priority_high() -> None:
    out = normalize_entities("p1 fix", {})
    assert out["priority"] == "high"


# --- literal priority aliases set by the LLM --------------------------------
# Regression for stress query #32 ("Add a P0 bug …") where the LLM put the
# literal "P0" straight into entities['priority'] and the preview showed
# "with P0 priority" instead of "critical". Canonicalization must fire even
# when priority is already set, as long as the value is a known alias.


def test_literal_p0_priority_is_canonicalized_to_critical() -> None:
    out = normalize_entities("Add a P0 bug", {"type": "bug", "priority": "P0"})
    assert out["priority"] == "critical"


def test_literal_p1_priority_is_canonicalized_to_high() -> None:
    out = normalize_entities("ship it", {"priority": "P1"})
    assert out["priority"] == "high"


def test_literal_p2_priority_is_canonicalized_to_medium() -> None:
    out = normalize_entities("ship it", {"priority": "p2"})
    assert out["priority"] == "medium"


def test_literal_sev1_priority_is_canonicalized_to_critical() -> None:
    out = normalize_entities("incident", {"priority": "Sev1"})
    assert out["priority"] == "critical"


def test_literal_important_priority_is_canonicalized_to_high() -> None:
    out = normalize_entities("ship it", {"priority": "important"})
    assert out["priority"] == "high"


def test_canonical_priority_value_is_preserved_exactly() -> None:
    """A value that is already canonical must pass through untouched (and not,
    e.g., get re-derived from the message's keywords)."""
    out = normalize_entities("this is urgent", {"priority": "low"})
    assert out["priority"] == "low"


def test_unrecognized_priority_literal_is_left_untouched() -> None:
    """An unknown priority string is not a value we can safely map, so it is
    left as-is rather than guessed at."""
    out = normalize_entities("ship it", {"priority": "whenever"})
    assert out["priority"] == "whenever"


# --- no-trigger cases ------------------------------------------------------


def test_no_keywords_means_no_priority_set() -> None:
    out = normalize_entities("add a feature for the dashboard", {"type": "feature"})
    assert "priority" not in out


def test_text_without_bug_word_keeps_existing_type() -> None:
    out = normalize_entities("add a feature", {"type": "feature"})
    assert out["type"] == "feature"


def test_empty_text_does_not_set_priority() -> None:
    out = normalize_entities("", {})
    assert "priority" not in out


def test_explicit_priority_wins_over_keyword() -> None:
    out = normalize_entities("critical fix", {"priority": "low"})
    assert out["priority"] == "low"


# --- input shape edge cases -------------------------------------------------


def test_none_text_does_not_crash() -> None:
    out = normalize_entities(None, {})  # type: ignore[arg-type]
    assert out == {}


def test_none_entities_does_not_crash() -> None:
    out = normalize_entities("create a bug", None)  # type: ignore[arg-type]
    assert out["type"] == "bug"
    assert out["priority"] == "high"


def test_empty_string_value_treated_as_unset() -> None:
    """The LLM sometimes returns 'priority: ""' — the convention should
    still fire (treat empty string as 'unset')."""
    out = normalize_entities("urgent fix", {"priority": ""})
    assert out["priority"] == "critical"


def test_does_not_mutate_input() -> None:
    entities = {"title": "x", "type": None, "priority": None}
    snapshot = dict(entities)
    normalize_entities("create a bug", entities)
    assert entities == snapshot


def test_does_not_add_other_keys() -> None:
    """The normalizer only ever sets type and priority — it does not
    invent titles, assignees, or sprints."""
    out = normalize_entities("create a bug for Alice in Sprint 1", {"title": "x"})
    # Pre-existing keys are preserved; the normalizer only *adds* type/priority
    # and never invents an assignee or sprint from the text.
    assert set(out.keys()) <= {"title", "type", "priority"}


# --- integration with the supervisor's preview ------------------------------


def test_normalize_changes_what_preview_would_show() -> None:
    """Smoke test: when the LLM returns type=None, priority=None, the
    normalized entities produce the convention defaults. The supervisor's
    preview should reflect those."""
    raw_entities = {"title": "Login broken", "type": None, "priority": None}
    normalized = normalize_entities("Create a bug for the login", raw_entities)
    assert normalized["type"] == "bug"
    assert normalized["priority"] == "high"
    # And the other keys are preserved.
    assert normalized["title"] == "Login broken"
