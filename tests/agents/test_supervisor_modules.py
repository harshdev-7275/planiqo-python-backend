"""Public-API smoke tests for the modules split out of supervisor.py.

The bulk of the behaviour is covered through supervisor's re-exports in
test_supervisor.py / test_validation.py; these lock the NEW import paths so a
future change that drops a public name fails loudly rather than silently
breaking the supervisor's backwards-compat aliases.
"""

from chains.title_clean import clean_title
from chains.yes_no import is_affirmation, is_negation
from agents.preview import build_preview, effective_title, join_clauses
from models.intents import Intent, IntentResult


# --- chains.yes_no -----------------------------------------------------------


def test_yes_no_public_api() -> None:
    assert is_affirmation("yes please do it") is True
    assert is_affirmation("ok to delete production") is False  # not all words in vocab
    assert is_negation("no") is True
    assert is_negation("nah whatever, leave it") is False


# --- chains.title_clean ------------------------------------------------------


def test_clean_title_public_api() -> None:
    assert clean_title("Create a bug for the login page to Alice") == "bug for the login page"
    assert clean_title("'Onboarding wizard'") == "Onboarding wizard"
    assert clean_title(None) == ""


# --- agents.preview ----------------------------------------------------------


def test_join_clauses_oxford_grammar() -> None:
    assert join_clauses(["a"]) == "a"
    assert join_clauses(["a", "b"]) == "a and b"
    assert join_clauses(["a", "b", "c"]) == "a, b, and c"


def test_effective_title_and_preview() -> None:
    entities = {"title": "Create a bug: checkout fails", "priority": "high"}
    assert effective_title(entities) == "bug: checkout fails"

    ir = IntentResult(intent=Intent.CREATE_SPRINT, confidence=1.0, entities={"name": "Q3"})
    assert "create sprint 'Q3'" in build_preview(ir)
