"""What `review/evidence_retrieval.py` must not get wrong.

Every case here is a failure that actually happened, taken from the seventy hand-judged
retrievals in docs/evidence-top1-judgements.md. The module exists only to prevent them,
so a test that stops failing means the fix was undone, not that the case got easier.

The section priors are load-bearing in one direction only: they must never turn a
section into a hard filter. 61% of reviewer evidence is in Methods, but the other 39%
is not, and a filter would make those unreachable rather than merely lower-ranked.
"""

from __future__ import annotations

import pytest

from pondie.extraction.evidence import retrieval as er

# --- aliases ----------------------------------------------------------------


# --- literal match ----------------------------------------------------------


# --- sections ---------------------------------------------------------------

SAMPLE = """# A study

We did a thing.

## Introduction

DTI is a non-invasive method that maps the diffusivity of water molecules.

## Materials and methods

### Participants

Twenty patients were recruited.

### MRI data acquisition

MRIs were acquired with TR = 2 s.

## Results

### Imaging results

Volume was reduced in patients.

## Discussion

Our findings suggest a mechanism.

## Acknowledgements

Grant 2013DFA11140, to BH.
"""


def test_sections_cover_the_whole_text():
    spans = er.sectionize(SAMPLE)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(SAMPLE)
    for (_, end, _), (start, _, _) in zip(spans, spans[1:]):
        assert end == start


def test_numbered_headings_are_recognised():
    assert er.classify_heading("2.3. Statistical analysis") == "methods"
    assert er.classify_heading("3 RESULTS") == "results"


# --- section priors ---------------------------------------------------------


# --- units ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The audit in docs/regex-audit.md, findings 1 and 2. Both are corpus-measured:
# 1.3M units cut mid-citation, and 1,710 papers reading their sections as unknown.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
def test_a_heading_of_any_markdown_level_keeps_its_text(level):
    """Capped at `#{1,4}`, the fifth `#` of a level-5 heading fell out of the group and
    into the heading *text*, so `##### Results` arrived as `# Results`."""

    text = f"{'#' * level} Results\n\nbody\n"
    assert [(m.group(1), m.group(2)) for m in er._HEADING.finditer(text)] == [
        ("#" * level, "Results")
    ]


@pytest.mark.parametrize("raw", ["Results", "##### Results", "# Results", "3.1 Results"])
def test_a_section_is_classified_whatever_marker_survives_the_heading(raw):
    """Every `_SECTION_PATTERNS` entry for a named section is `^`-anchored, so one stray
    `#` unclassifies the section rather than misclassifying it -- a silent failure, and the
    reason the canonicaliser strips markers it should no longer receive."""

    assert er.classify_heading(raw) == "results"
