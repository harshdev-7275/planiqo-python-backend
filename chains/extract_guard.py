"""Extraction grounding guard (items 8 + 9).

The classifier prompt says "transcribe the user's words, never paraphrase" —
but models drift. This is the deterministic backstop: a title or sprint name
the LLM returns should be GROUNDED in the user's message, i.e. every meaningful
word in the extraction actually appears in what the user typed. A value with
novel words the user never said is a likely fabrication; the supervisor grounds
it back to the user's own words rather than ask them to confirm an invented one.

Pure functions only — no LLM, no I/O — so the behaviour is identical every run
and trivially testable.
"""

from __future__ import annotations

import re

# Words that carry no identifying content: they may appear on either side
# without making a value "grounded" or "fabricated". Includes the create
# verbs and the unit-of-work nouns the title-cleaner already strips, so e.g.
# the LLM turning "log a bug for checkout" into a title "checkout" is grounded
# even though "checkout" is the only shared content word.
_STOPWORDS = frozenset({
    "a", "an", "the", "for", "to", "in", "on", "of", "and", "or", "with",
    "is", "are", "be", "this", "that", "it", "please", "we", "i", "my",
    "create", "add", "open", "file", "log", "make", "raise", "submit",
    "track", "new", "called", "named", "about",
    "issue", "bug", "task", "story", "ticket", "epic", "sprint",
})

_WORD_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r"['\"‘’“”]([^'\"‘’“”]+)['\"‘’“”]")


def _content_words(text: str) -> set[str]:
    """Meaningful, identity-bearing words: lowercased alphanumerics, minus
    stopwords and single characters."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS}


def novel_words(extracted: str, source: str) -> set[str]:
    """Content words present in *extracted* but absent from *source* — the
    words the model appears to have invented."""
    src = _content_words(source)
    return {w for w in _content_words(extracted) if w not in src}


def is_grounded(extracted: str | None, source: str) -> bool:
    """True if *extracted* invents no content not found in *source*.

    Empty/whitespace extraction is trivially grounded — emptiness is a
    separate concern (the supervisor's missing-title gate)."""
    if not extracted or not extracted.strip():
        return True
    return not novel_words(extracted, source)


def strip_novel(extracted: str, source: str) -> str:
    """Return *extracted* with its fabricated tokens removed.

    A token is dropped only when ALL of its content words are novel (absent
    from *source*), so a minor morphological drift inside a longer phrase
    ("addresses" in "long email addresses") is pruned while genuinely grounded
    words survive in their original order. If the model fabricated the whole
    value, the result is empty and the caller falls back to asking the user.
    This is safer than re-deriving a title from the entire (possibly very long)
    message, which would produce a sentence-as-title."""
    novel = novel_words(extracted, source)
    if not novel:
        return extracted.strip()
    kept: list[str] = []
    for token in extracted.split():
        words = _WORD_RE.findall(token.lower())
        if words and all(w in novel for w in words):
            continue  # token is entirely invented — drop it
        kept.append(token)
    return " ".join(kept).strip()


def extract_quoted(text: str) -> str | None:
    """Return the first quoted span in *text* (handles straight and curly
    quotes), or None. Sprint/story names are usually quoted — when the model
    fabricates a generic name, the user's quoted original is the ground truth."""
    match = _QUOTED_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None
