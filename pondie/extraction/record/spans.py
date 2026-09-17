"""Resolve verbatim quotes emitted by an extractor into EvidenceSpan offsets.

An LLM cannot count characters reliably, so it is asked for verbatim quotes and
this module locates them in the normalized source text. The offsets are computed
here, deterministically, which is what lets the integrity gate assert
normalized[start_char:end_char] == span.text for every span.

EvidenceSpan.text is always set to the document substring rather than to the
quote the model produced, so a whitespace-tolerant match can never introduce a
span whose text disagrees with the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Characters that publishers and models substitute for each other freely; a model
# routinely straightens curly quotes and dashes when echoing a quote. Written as
# explicit codepoints because several of these are visually indistinguishable.
#
# Every mapping must be single-character to single-character: folding has to
# preserve string length, or the offsets it produces would not address the
# original document. Unicode NFC/NFD normalization is deliberately NOT applied
# for the same reason -- composing "e" + U+0301 into U+00E9 would shorten the
# text and shift every following offset.
_EQUIVALENT = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": " ",
    "﻿": " ",
}


class SpanResolutionError(ValueError):
    """Raised when a quote cannot be located unambiguously in the source text."""


@dataclass(frozen=True)
class ResolvedSpan:
    start_char: int
    end_char: int
    text: str
    exact: bool
    #: True when only the case-insensitive pass found it. Counted separately by `build`, so
    #: the next run can say what ignoring case bought rather than assuming it bought
    #: anything -- the corpus cannot be asked, because a resolved span keeps the document's
    #: text and not the quote that located it.
    cased: bool = False

    def as_record(self) -> dict[str, object]:
        return {"text": self.text, "start_char": self.start_char, "end_char": self.end_char}


def fold(value: str) -> str:
    """Length-preserving character folding. len(fold(v)) == len(v) always."""

    return value.translate(str.maketrans(_EQUIVALENT))


def fold_label(value: str) -> str:
    """Fold a label for joining, where length does not have to be preserved.

    `Cell.level` joins to `FactorLevel.level` on the string, and extraction-readme.md §3
    invariant 3 asks that the comparison use the same normalization the mapper applies. That
    is this: `fold`, then collapse whitespace runs, then casefold. Deliberately narrow --
    only differences that cannot be semantic. `Healthy controls` and `healthy controls` are
    the same level; `AD` and `AD group` are not, and calling them equal here would hide the
    join failure rather than report it.

    Not `fold`, and it must not be used where an offset survives the call: collapsing
    whitespace changes the length, which is the one thing `fold` promises never to do.
    """

    return re.sub(r"\s+", " ", fold(value)).strip().casefold()


def _tolerant_pattern(quote: str, *, ignore_case: bool = False) -> re.Pattern[str]:
    """Build a regex matching the quote with any whitespace run between tokens.

    `ignore_case` is the third pass `resolve` tries, and it is a separate pass rather than a
    flag on the second so that no quote which already matches can match somewhere *else*:
    a case-sensitive hit at offset 900 must not be displaced by a case-different one at 50.
    Strictly additive, so only quotes that used to fail can now resolve.

    Case is the one perturbation a model makes mechanically -- it lowercases a mid-sentence
    quote it starts with a capital, or title-cases a phrase -- and `fold` cannot absorb it,
    because `fold` is a character translation and casefolding is not length-preserving for
    all of Unicode. Matching case-insensitively against the *folded original* sidesteps
    that: the haystack is untouched, so the offsets still address the document and
    `EvidenceSpan.text` is still the document substring rather than the model's quote.
    """

    tokens = [re.escape(token) for token in fold(quote).split()]
    if not tokens:
        raise SpanResolutionError("quote is empty")
    return re.compile(r"\s+".join(tokens), re.IGNORECASE if ignore_case else 0)


#: An elision a model writes when it joins two non-adjacent fragments of a sentence:
#: "our design ... allows us to identify". The result is not in the document and never can
#: be, but each fragment is. Measured over 104,957 proposed quotes on the 903-paper run,
#: 176 of the 2,952 that nothing could place are this shape, and for 169 of them EVERY
#: fragment resolves on its own.
ELLIPSIS = re.compile(r"\s*(?:\.\s*\.\s*\.|\u2026)\s*")


def resolve_elided(
    normalized: str,
    quote: str,
    *,
    near: int | None = None,
    folded_text: str | None = None,
) -> list[ResolvedSpan]:
    """Resolve an elided quote as the several spans it actually cites.

    An `EvidenceSet` already holds several spans, so an elided quote is not a malformed
    quote needing repair -- it is two citations written as one, and this is the shape the
    schema has for it.

    All or nothing: a fragment that does not place makes the whole quote unresolved, because
    half the support offered is not the support offered. Raises rather than returning a
    partial list, so the caller counts it as a drop exactly as before.
    """

    fragments = [part for part in ELLIPSIS.split(quote) if part.strip()]
    if len(fragments) < 2:
        raise SpanResolutionError(f"quote is not elided: {quote[:60]!r}")
    haystack = folded_text if folded_text is not None else fold(normalized)
    placed = [
        resolve(normalized, fragment, near=near, folded_text=haystack)
        for fragment in fragments
    ]
    return sorted(placed, key=lambda span: span.start_char)


def resolve(
    normalized: str,
    quote: str,
    *,
    near: int | None = None,
    folded_text: str | None = None,
) -> ResolvedSpan:
    """Locate one quote in the normalized text.

    near biases selection when a quote occurs more than once; pass the start of
    the enclosing section to disambiguate a phrase that repeats across the paper.
    """

    if not quote or not quote.strip():
        raise SpanResolutionError("quote is empty")

    exact = [match.start() for match in re.finditer(re.escape(quote), normalized)]
    if exact:
        start = _pick(exact, near)
        return ResolvedSpan(
            start, start + len(quote), normalized[start : start + len(quote)], True
        )

    haystack = folded_text if folded_text is not None else fold(normalized)
    for ignore_case in (False, True):
        matches = list(_tolerant_pattern(quote, ignore_case=ignore_case).finditer(haystack))
        if not matches:
            continue
        starts = [match.start() for match in matches]
        chosen = matches[starts.index(_pick(starts, near))]
        return ResolvedSpan(
            chosen.start(),
            chosen.end(),
            normalized[chosen.start() : chosen.end()],
            False,
            cased=ignore_case,
        )
    raise SpanResolutionError(f"quote not found in source text: {quote[:80]!r}")


def _pick(starts: list[int], near: int | None) -> int:
    if near is None:
        return starts[0]
    return min(starts, key=lambda start: abs(start - near))


def verify(normalized: str, span: dict[str, object]) -> None:
    """Assert the schema invariant for one serialized EvidenceSpan."""

    start, end, text = span["start_char"], span["end_char"], span["text"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise SpanResolutionError(f"offsets must be integers: {span!r}")
    if not 0 <= start < end <= len(normalized):
        raise SpanResolutionError(f"offsets outside document: {start}-{end}")
    actual = normalized[start:end]
    if actual != text:
        raise SpanResolutionError(
            f"span text disagrees with source at {start}-{end}: {text!r} != {actual!r}"
        )
