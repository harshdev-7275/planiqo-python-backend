"""Tests for the extraction grounding guard (items 8 + 9)."""

from __future__ import annotations

from chains.extract_guard import (
    extract_quoted,
    is_grounded,
    novel_words,
    strip_novel,
)


# --- is_grounded / novel_words ----------------------------------------------


def test_grounded_when_all_words_present() -> None:
    assert is_grounded("login page", "create a bug for the login page") is True


def test_not_grounded_when_word_invented() -> None:
    # "authentication" + "production" never appear in the message.
    msg = "create a bug for the login page"
    assert is_grounded("login authentication failure on production", msg) is False
    assert novel_words("login authentication failure on production", msg) == {
        "authentication", "failure", "production",
    }


def test_stopwords_and_verbs_do_not_count_as_novel() -> None:
    # "Create"/"a"/"bug" are stopwords; only real content matters.
    assert is_grounded("checkout", "log a bug for checkout") is True


def test_empty_extraction_is_trivially_grounded() -> None:
    assert is_grounded("", "anything") is True
    assert is_grounded(None, "anything") is True


# --- strip_novel ------------------------------------------------------------


def test_strip_novel_prunes_only_invented_tokens() -> None:
    msg = "the login form freezes with a long email address"
    # "addresses" (plural) is a morphological drift; the rest is grounded.
    assert strip_novel("login form freezes with long email addresses", msg) == (
        "login form freezes with long email"
    )


def test_strip_novel_returns_empty_when_fully_fabricated() -> None:
    msg = "create a bug for login"
    assert strip_novel("database connection pool exhaustion", msg) == ""


def test_strip_novel_noop_when_grounded() -> None:
    assert strip_novel("login page", "bug for the login page") == "login page"


# --- extract_quoted ---------------------------------------------------------


def test_extract_quoted_straight_quotes() -> None:
    assert extract_quoted("Create a sprint called 'Q3 Hardening' next week") == "Q3 Hardening"


def test_extract_quoted_double_quotes() -> None:
    assert extract_quoted('rename it to "Deploy hotfix"') == "Deploy hotfix"


def test_extract_quoted_none_when_absent() -> None:
    assert extract_quoted("create a sprint for the mobile team") is None
