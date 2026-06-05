"""Yes/no reply detection for write-confirmation.

The supervisor proposes a mutating action and waits for the user to confirm.
This module decides whether a free-text reply means "yes, do it" or "no, cancel"
— robustly enough that a button click ("yes"/"no"), a terse "ok", and a phrase
like "yeah go ahead" all resolve, while an unrelated sentence that merely
contains affirmation words does not.

Pure and dependency-free so it can be unit-tested in isolation and reused by any
caller that needs the same affirmation/negation semantics. The supervisor
re-exports ``is_affirmation``/``is_negation`` as ``_is_affirmation``/
``_is_negation`` for backwards compatibility.
"""

from __future__ import annotations

import re

# Exact-phrase matches. Kept for short replies where word-set matching is too
# permissive (e.g. "ok" alone vs. "ok to delete production" — the latter is
# not a confirmation, but every word *is* in _AFFIRM_WORDS).
AFFIRMATIONS = frozenset({
    "yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "ok", "okay", "k",
    "sure", "do it", "go ahead", "yes please", "please do", "proceed", "sounds good",
})
NEGATIONS = frozenset({
    "no", "n", "nope", "cancel", "stop", "nevermind", "never mind", "abort",
    "dont", "don't",
})

# Vocabulary for word-set matching — every word in the reply must come from
# this set AND at least one word from the core set must be present. This lets
# "yes please do it" / "yeah go ahead" / "ok sounds good" all match without
# letting unrelated sentences through.
AFFIRM_WORDS: frozenset[str] = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "k", "sure", "confirm",
    "confirmed", "proceed", "please", "do", "it", "go", "ahead", "sounds", "good",
})
AFFIRM_CORE: frozenset[str] = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "k", "sure", "confirm",
    "confirmed", "proceed",
})
NEGATE_WORDS: frozenset[str] = frozenset({
    "no", "n", "nope", "cancel", "stop", "abort", "never", "mind", "nevermind",
    "dont", "don't",
    # Contextual negations — common replies after the bot says "No X matching
    # 'Y' found". The user means "yeah I see, cancel" without literally
    # typing "no". "found" is in vocab but NOT in core: it can be a positive
    # ("I found it"), so a single-token "found" must not match on its own.
    "not", "found", "doesn't", "doesnt", "exist", "wrong", "missing",
    "none", "those", "of", "one",
})
NEGATE_CORE: frozenset[str] = frozenset({
    "no", "n", "nope", "cancel", "stop", "abort", "dont", "don't",
    # The load-bearing words for contextual negations. "found" is
    # deliberately NOT here — it can be positive ("I found the bug").
    "not", "doesn't", "doesnt", "wrong", "none", "missing",
})

# Common sentence punctuation that would otherwise survive ``.split()`` and
# block word-set matching (e.g. "yes," / "no.").
_PUNCT_RE = re.compile(r"[.!?,;:]")


def normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace, replace punctuation with spaces,
    collapse runs of spaces. Keeps word boundaries clean for matching."""
    cleaned = _PUNCT_RE.sub(" ", text.strip().lower())
    return " ".join(cleaned.split())


def is_word_set_reply(normalized: str, vocab: frozenset[str], core: frozenset[str]) -> bool:
    """True iff the reply is non-empty, every word is in ``vocab``, and at
    least one core (load-bearing) word is present. Prevents stray confirmations
    like 'ok' from being triggered by unrelated sentences that happen to
    contain only affirmation words."""
    words = normalized.split()
    if not words:
        return False
    if not all(w in vocab for w in words):
        return False
    return any(w in core for w in words)


def is_affirmation(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    if norm in AFFIRMATIONS:  # exact phrase like "do it" / "sounds good"
        return True
    return is_word_set_reply(norm, AFFIRM_WORDS, AFFIRM_CORE)


def is_negation(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    if norm in NEGATIONS:  # exact phrase like "never mind"
        return True
    return is_word_set_reply(norm, NEGATE_WORDS, NEGATE_CORE)
