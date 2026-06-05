"""Issue-title cleaning.

The LLM often copies the user's full verb phrase ("Create a bug for the login
page to Alice") into the title field; the confirmation preview should show the
issue's *name* ("login page"), not the sentence the user typed. This module does
that cleanup with conservative, word-level operations (no aggressive regex
rewriting) so it never mangles a legitimate title.

Pure and dependency-free. The supervisor re-exports ``clean_title`` as
``_clean_title`` for backwards compatibility.
"""

from __future__ import annotations

import re

# Title cleaning for the preview. The LLM often copies the user's full verb
# phrase into the title field; strip a leading imperative verb so the preview
# shows the issue's name, not the command.
LEADING_VERBS: frozenset[str] = frozenset({
    "create", "add", "open", "file", "log", "make",
    "raise", "submit", "track",
})
DETERMINERS: frozenset[str] = frozenset({"a", "an", "the"})
# Trailing prepositional phrases to strip — the assignee / sprint / project
# meta the LLM copies from the user's full sentence. Word-level only, no
# regex; we scan from the END so the rightmost preposition wins (avoids
# eating "for checkout to Bob" when only "to Bob" is meta).
TRAILING_PREPOSITIONS: frozenset[str] = frozenset({
    "for", "to", "in", "on", "at", "with", "by", "from", "of",
})

# If a trailing strip would leave the title shorter than this, restore the
# original — the strip ate too much meaningful text.
MIN_TITLE_AFTER_STRIP = 8

MAX_TITLE_LEN = 200

# Quote characters the LLM may wrap a title in — straight, backtick, and the
# common unicode single/double/angle variants. Stripped from both ends so the
# preview shows "Onboarding wizard", not "'Onboarding wizard'".
QUOTE_CHARS = "'\"`‘’“”„‹›«»"


def strip_surrounding_quotes(text: str) -> str:
    return text.strip().strip(QUOTE_CHARS).strip()


def strip_trailing_prepositional_meta(text: str) -> str:
    """Drop the last prepositional phrase at the end of ``text`` if it
    would leave at least MIN_TITLE_AFTER_STRIP chars. Scans from the end
    so only the *rightmost* prepositional phrase is considered — the
    leftmost-preposition regex would over-eat ("for checkout to Bob"
    would all get stripped when only "to Bob" is meta)."""
    words = text.split()
    if len(words) < 2:
        return text
    for n in range(1, min(6, len(words))):
        tail = " ".join(words[-n:])
        if tail.split()[0].lower() in TRAILING_PREPOSITIONS:
            new_text = " ".join(words[:-n])
            if len(new_text.strip()) >= MIN_TITLE_AFTER_STRIP:
                return new_text
            return text  # strip would leave too little — restore
    return text


def clean_title(raw: str | None) -> str:
    """Clean an LLM-extracted issue title.

    Two safe operations:
      1. Drop a leading imperative verb + optional determiner
         ("Create a …" → "…").
      2. Drop the *last* prepositional phrase at the end of the string
         ("to Alice", "in Sprint 1", "for the team") — only if at least
         MIN_TITLE_AFTER_STRIP chars remain after the strip.

    Plus: collapse whitespace, strip trailing punctuation, cap at
    MAX_TITLE_LEN. Returns "" for empty/None input (the caller falls back
    to a sentinel like "(untitled)").
    """
    if not raw:
        return ""
    # 0. Strip wrapping quotes (straight + unicode) and collapse internal
    #    whitespace early, so a multi-line title ("Login broken\non mobile")
    #    tokenizes the same as a single-line one for the verb/meta passes.
    text = strip_surrounding_quotes(raw)
    text = re.sub(r"\s+", " ", text).strip()
    # 1. Leading verb + optional determiner.
    parts = text.split(maxsplit=2)
    if parts and parts[0].lower() in LEADING_VERBS:
        rest = parts[1:]
        if rest and rest[0].lower() in DETERMINERS:
            rest = rest[1:]
        text = " ".join(rest) if rest else ""
    # 2. Trailing prepositional phrase (scans from end — see helper docstring).
    text = strip_trailing_prepositional_meta(text)
    # 3. Collapse whitespace, strip trailing punctuation + any quotes the
    #    verb/meta passes exposed, cap length.
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,;:!?")
    text = strip_surrounding_quotes(text)
    if len(text) > MAX_TITLE_LEN:
        text = text[:MAX_TITLE_LEN].rstrip()
    return text
