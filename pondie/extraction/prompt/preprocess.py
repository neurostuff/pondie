"""Sentence boundaries in the corpus text.

`sentence_spans` is what indexed evidence numbers a paper's sentences by, and
`evidence.cited` reads a cited number back with the same splitter, so the two cannot
disagree about where a sentence starts.

This module used to find coordinates a paper reports in prose and offer them to the
extractor. The upstream parse now extracts those itself, into `stage1/analyses.json`, and
the sweep went with the `prose` stage. It also once held text-transformation strategies,
measured in `docs/text-preprocessing-experiments.md` and never wired in. Both are in the
history.

Standard library only.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------ sentence splitting

#: Tokens whose trailing period does not end a sentence. Mostly citation and unit
#: furniture; `et al.` and `Fig.` alone account for most bad splits in this corpus. A
#: single capital is in the list because an initial is the other common case.
_NON_TERMINAL = frozenset("""
et al e.g i.e cf vs etc approx ca resp viz fig figs tab tabs eq ref refs no nos dr prof
mr mrs ms st inc ltd co univ dept min sec ms mm cm ml mg kg vol ed eds pp al s.d s.e
i.v p.o a.m p.m
""".split()) | frozenset(chr(c) for c in range(ord("a"), ord("z") + 1))

#: A period, question or exclamation mark followed by space and something that could
#: start a sentence. Python's `re` will not take a variable-width lookbehind, so the
#: "unless the word before it is an abbreviation" half is a token check on the match and
#: not part of the pattern.
_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.?$")


def ends_mid_sentence(text: str) -> bool:
    """Whether a run of text ends on a period that does not end a sentence.

    Public so a caller cutting text on the same punctuation gets the same answer: a cutter
    without this guard ended 9.6% of its units at `et al.` or `e.g.`, and a second
    abbreviation list would drift from this one.
    """

    word = _LAST_WORD.search(text.rstrip(".!?"))
    return bool(word and word.group(1).lower().rstrip(".") in _NON_TERMINAL)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Every sentence of `text` as (start, end) offsets into it, in document order.

    Offsets rather than strings, so a sentence cited by number resolves to exactly the
    span it names. A decimal point or an abbreviation's period is not a boundary; a
    heading or a table row is one unit of its own, since a row's cells are not sentences.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.split("\n"):
        start, stripped = 0, line.strip()
        if stripped and not stripped.startswith(("#", "|")):
            # The decimal guard keeps the length, so offsets in `guarded` are offsets in `line`.
            guarded = re.sub(r"(\d)\.(\d)", lambda m: f"{m[1]}\x00{m[2]}", line)
            for boundary in _BOUNDARY.finditer(guarded):
                if ends_mid_sentence(guarded[start : boundary.start() + 1]):
                    continue
                spans.append((offset + start, offset + boundary.start() + 1))
                start = boundary.end()
        spans.append((offset + start, offset + len(line)))
        offset += len(line) + 1
    trimmed = []
    for begin, end in spans:
        piece = text[begin:end]
        begin += len(piece) - len(piece.lstrip())
        end -= len(piece) - len(piece.rstrip())
        if end - begin > 2:
            trimmed.append((begin, end))
    return trimmed
