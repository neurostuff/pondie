"""Cutting a paper into sections, for the passes that need to know where a sentence sits.

What is left of a local retriever. It scored sentences against a field with a cross-encoder
and its picks were unioned with the model's quotes; the four fixes it made before ranking --
field-name aliases, unit surface forms, exact-literal precedence, a section prior rather than
a section filter -- are written up in docs/evidence-top1-judgements.md and
docs/evidence-union-design.md, and the measurement that the union was worth 21 points still
holds. The ranking went with the local models.

`sectionize` outlived it because two passes want the same question answered: `repair` builds
its premise from the Methods and Results, and `grounding` scores a proposal against the
section it came from. Neither ranks anything.
"""

from __future__ import annotations

import re

# One abbreviation list for the repo. `preprocess` owns it because that is where it was
# measured against scispaCy; importing it is cheaper than the drift of a second copy.

# --- sections ---------------------------------------------------------------

#: Heading text -> canonical section. Matched against the lowercased heading with
#: punctuation and numbering stripped, longest pattern first.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"materials?\s+and\s+methods?|methods?\s+and\s+materials?", "methods"),
    (r"^methods?$|^methodology$|^experimental\s+", "methods"),
    (
        r"participants?|subjects?|procedures?|acquisition|data\s+analysis|"
        r"statistical\s+analys|image\s+processing|preprocessing|pre-processing|"
        r"pharmacotherapy|stimulation|paradigm|task|apparatus|measures?|instruments?",
        "methods",
    ),
    (r"^results?$|^findings?$|^imaging\s+results", "results"),
    (r"^discussion|^conclusions?|^limitations?|^general\s+discussion", "discussion"),
    (r"^abstract|^summary|^objectives?|^background\s+and\s+aims", "abstract"),
    (r"^introduction|^background$", "intro"),
    (r"^tables?\b|^figures?\b", "tables"),
    (
        r"acknowledg|funding|conflict|competing\s+interest|references|"
        r"supplementary|author\s+contribution|ethics|data\s+availability",
        "back",
    ),
]

#: `{1,6}` and not `{1,4}`: markdown has six levels and `text.py` writes all of them. Capped
#: at four, the fifth `#` of a level-5 heading fell out of the group and into the heading
#: *text*, so `##### Results` arrived here as `# Results` and every `^`-anchored entry in
#: `_SECTION_PATTERNS` missed it. That is 1,710 papers of the corpus reading their Results
#: and Discussion as `unknown`, and section is what disambiguates a phrase the paper repeats.
_HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*$", re.MULTILINE)


def _canon_heading(raw: str) -> str:
    """Strip numbering and punctuation so '2.3. Statistical analysis' matches."""
    # A leading `#` is stripped as well as the numbering. It cannot arrive from `_HEADING`
    # any more, but every `_SECTION_PATTERNS` entry is `^`-anchored and one stray marker
    # silently unclassifies the section rather than misclassifying it -- a failure with
    # nothing on the face of it to say so, which is why the guard outlives its cause.
    text = re.sub(r"^[#\d.\s]+", "", raw).strip(" .:#").lower()
    return re.sub(r"\s+", " ", text)


def classify_heading(raw: str) -> str | None:
    text = _canon_heading(raw)
    if not text:
        return None
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def sectionize(text: str) -> list[tuple[int, int, str]]:
    """(start, end, label) spans covering the whole text.

    A subsection heading that does not itself name a section (`### Participants` does,
    `### Pretreatment rsFC` does not) inherits the enclosing section rather than
    resetting it, which is what keeps a Results subsection from being read as Methods.
    """

    marks: list[tuple[int, str, int]] = []
    for match in _HEADING.finditer(text):
        level = len(match.group(1))
        label = classify_heading(match.group(2))
        if label:
            marks.append((match.start(), label, level))

    if not marks:
        return [(0, len(text), "unknown")]

    spans: list[tuple[int, int, str]] = []
    if marks[0][0] > 0:
        # Everything before the first heading is title + abstract in this corpus.
        spans.append((0, marks[0][0], "abstract"))
    for index, (start, label, _level) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        spans.append((start, end, label))
    return spans


# --- field priors -----------------------------------------------------------


# --- aliases ----------------------------------------------------------------


# --- value surface forms ----------------------------------------------------


# --- literal match ----------------------------------------------------------


# --- units ------------------------------------------------------------------


#: Below this a unit is a fragment, above it a page. Both are useless as evidence.
MIN_UNIT, MAX_UNIT = 15, 900


# --- scoring ----------------------------------------------------------------
